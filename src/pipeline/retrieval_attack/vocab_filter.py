"""
安全候选词表 V_safe 构建模块。

离线从检索器的 tokenizer 词表中过滤出可用于 AGGD 对抗替换的安全 token 子集，
并缓存对应的嵌入子矩阵，供 AGGD 优化主循环直接使用。

过滤规则：
  1. 移除所有 special tokens（[CLS], [SEP], [PAD], [UNK], [MASK] 等）
  2. 移除不安全词元
"""

import logging
from typing import Set, Tuple

import torch

from src.utils.token import is_safe_candidate

logger = logging.getLogger(__name__)


def build_safe_vocab(
    tokenizer,
    frozen_token_dict: dict,
    use_safe_vocab: bool = True,
) -> torch.LongTensor:
    """
    从 tokenizer 的完整词表中筛选出安全候选 token ID（调用 ``is_safe_candidate``）。

    Args:
        tokenizer: HuggingFace tokenizer 实例（如 Jina Code Embeddings 的 tokenizer）
        frozen_token_dict: 结构化冻结集合（含 "structure" / "keywords" /
            "punctuation_and_operators" 三个键）；``is_safe_candidate`` 仅使用其中的
            "structure" 与 "punctuation_and_operators"。
        use_safe_vocab: 是否启用 frozen_token 过滤。为 False 时跳过关键字/标点冻结过滤，
            仅移除 special tokens，返回完整的非 special token 子集。

    Returns:
        safe_ids: 一维 LongTensor，包含所有通过过滤的 token ID（即，安全token的token_id(在这里就是索引)的列表）
    """
    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)

    # 获取额外的 special token（部分模型会在 added_tokens 里添加额外 special）
    if hasattr(tokenizer, "added_tokens_encoder"):
        special_ids.update(tokenizer.added_tokens_encoder.values())

    safe_ids = []
    rejected_counts = {"special": 0, "frozen": 0}

    for token_id in range(vocab_size):
        if token_id in special_ids:
            rejected_counts["special"] += 1
            continue

        if use_safe_vocab:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            if not is_safe_candidate(token_text, frozen_token_dict):
                rejected_counts["frozen"] += 1
                continue

        safe_ids.append(token_id)

    logger.info(
        f"词表过滤完成: 总词表 {vocab_size}, 保留 {len(safe_ids)}, "
        f"过滤 special={rejected_counts['special']}, "
        f"frozen={rejected_counts['frozen']}"
        + ("" if use_safe_vocab else "（全词表模式：跳过 frozen_token 过滤）")
    )

    return torch.tensor(safe_ids, dtype=torch.long)


def build_safe_embedding_matrix(
    word_embeddings: torch.nn.Embedding, # 嵌入矩阵，shape: (vocab_size, embed_dim)，以Jina为例，是(151644, 896)
    safe_ids: torch.LongTensor, # 安全 token ID 列表，shape: (|V_safe|,)
    device: str = "cuda",
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """
    根据安全 token ID 列表，从词嵌入矩阵中提取对应的嵌入子矩阵。

    Args:
        word_embeddings: 模型词嵌入层（nn.Embedding）
        safe_ids: 安全 token ID（由 ``build_safe_vocab`` 返回）
        device: 目标设备

    Returns:
        (safe_ids, safe_emb_matrix) 元组:
          - safe_ids: 形状 ``(|V_safe|,)`` 的 LongTensor
          - safe_emb_matrix: 形状 ``(|V_safe|, embed_dim)`` 的 FloatTensor
    """
    safe_ids = safe_ids.to(device)
    with torch.no_grad():
        safe_emb_matrix = word_embeddings.weight[safe_ids].clone()

    logger.info(
        f"安全嵌入子矩阵: shape={safe_emb_matrix.shape}, "
        f"device={safe_emb_matrix.device}, dtype={safe_emb_matrix.dtype}"
    )

    return safe_ids, safe_emb_matrix


def get_word_embeddings(model):
    """
    从 SentenceTransformer 模型中获取词嵌入层（nn.Embedding）。

    不同底层架构的词嵌入层路径不同，按优先级依次尝试：
      1. Qwen2 / LLaMA / Mistral 等 decoder-only 架构：
             model[0].auto_model.embed_tokens
      2. BERT / RoBERTa 等 encoder 架构：
             model[0].auto_model.embeddings.word_embeddings
      3. ModernBERT（如 GTE ModernBERT）：
             model[0].auto_model.embeddings.tok_embeddings

    Args:
        model: SentenceTransformer 模型实例

    Returns:
        nn.Embedding 词嵌入层
    """
    auto_model = model[0].auto_model

    # 路径1：Qwen2 / Qwen3 / LLaMA 等 decoder-only 架构
    #   - jinaai/jina-code-embeddings-0.5b（Qwen2）
    #   - microsoft/harrier-oss-v1-*（Qwen3）
    if hasattr(auto_model, "embed_tokens"):
        return auto_model.embed_tokens

    # 路径2：BERT / RoBERTa 等 encoder 架构
    if hasattr(auto_model, "embeddings") and hasattr(auto_model.embeddings, "word_embeddings"):
        return auto_model.embeddings.word_embeddings

    # 路径3：ModernBERT（如 Alibaba-NLP/gte-modernbert-base）使用 tok_embeddings
    if hasattr(auto_model, "embeddings") and hasattr(auto_model.embeddings, "tok_embeddings"):
        return auto_model.embeddings.tok_embeddings

    raise RuntimeError(
        f"无法从 {type(auto_model).__name__} 中获取词嵌入层。"
        "请检查模型架构，手动确认词嵌入层的属性路径。"
    )
