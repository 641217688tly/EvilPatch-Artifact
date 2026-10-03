"""
模型工具模块，提供与模型相关的工具函数，比如输出格式化、输入截断等
"""

import logging
import re
from typing import List, Dict, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


def probe_max_batch_size(
    embedder,
    poison_target_ids: torch.LongTensor,
    safety_margin: float = 0.85,
    fallback_batch_size: int = 96,
    max_probe_upper: int = 8192,
) -> int:
    """
    通过二分探测法确定 embedder 前向传播能承受的最大 batch_size。

    使用投毒目标的真实 token ID 序列（而非零填充张量）进行探测，以准确反映
    实际推理时的显存占用（包括注意力激活值、KV-cache 等与 token 内容相关的开销）。

    探测完成后乘以 safety_margin（默认 0.9）作为安全余量，预留 10% 显存缓冲
    给优化循环中梯度计算（反向传播）所需的额外显存。

    Args:
        embedder: 已加载的 BaseEmbedder 实例（需具备 embed_token_ids 方法）
        poison_target_ids: 投毒目标的初始 token ID 张量，形状 ``(1, seq_len)``，
                           包含 prompt prefix + buggy code 的完整编码序列
        safety_margin: 安全余量系数，取探测到的最大值的此比例作为实际 batch_size
        fallback_batch_size: 探测失败时的回退 batch_size
        max_probe_upper: 二分搜索上界的硬上限，防止搜索范围过大

    Returns:
        int: 推荐的最大安全 batch_size（>= 1）
    """
    device = embedder.device
    if not device.startswith("cuda") or not torch.cuda.is_available():
        logger.info(
            f"非 CUDA 设备（{device}），跳过 batch_size 探测，使用回退值 {fallback_batch_size}"
        )
        return fallback_batch_size

    seq_len = poison_target_ids.shape[1]
    attn_mask_1 = torch.ones_like(poison_target_ids)  # (1, seq_len)，自动继承 device 和 dtype

    torch.cuda.empty_cache()

    try:
        free_mem, _ = torch.cuda.mem_get_info(device)
    except Exception as e:
        logger.warning(f"无法获取显存信息（{e}），使用回退 batch_size={fallback_batch_size}")
        return fallback_batch_size

    # ── 单样本热身探测：用真实 token ID 测量 per-sample 显存开销 ──
    try:
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device)

        with torch.no_grad():
            output = embedder.embed_token_ids(poison_target_ids, attn_mask_1)

        torch.cuda.synchronize(device)  # 确保前向传播完全结束再读取峰值统计
        peak_mem = torch.cuda.max_memory_allocated(device)
        per_sample_cost = peak_mem - mem_before

        del output  # 释放热身输出张量，避免其占用显存影响后续探测
        torch.cuda.empty_cache()

        if per_sample_cost <= 0:
            # 极少发生：模型已在显存中分配了足量缓存，peak 未超过 before
            per_sample_cost = free_mem // 64

        estimated_upper = int(free_mem * 0.85 / max(per_sample_cost, 1))
        high = min(max(estimated_upper, 2), max_probe_upper)
    except Exception as e:
        logger.warning(f"单样本探测失败（{e}），使用回退 batch_size={fallback_batch_size}")
        torch.cuda.empty_cache()
        return fallback_batch_size

    low = 1
    last_success = 1

    logger.info(
        f"开始二分探测 batch_size: seq_len={seq_len}, "
        f"free_mem={free_mem / 1024**2:.0f} MiB, "
        f"per_sample≈{per_sample_cost / 1024**2:.1f} MiB, "
        f"搜索范围=[{low}, {high}]"
    )

    # ── 二分搜索 ──
    while low <= high:
        mid = (low + high) // 2
        batch_ids = None
        batch_mask = None
        output = None
        try:
            batch_ids = poison_target_ids.expand(mid, -1)   # (mid, seq_len)，不分配新显存
            batch_mask = attn_mask_1.expand(mid, -1)         # (mid, seq_len)，与实际评估代码一致
            with torch.no_grad():
                output = embedder.embed_token_ids(batch_ids, batch_mask)

            del batch_ids, batch_mask, output  # 三者全部释放，避免残留影响下一轮探测
            torch.cuda.synchronize(device)     # 确保异步 CUDA 操作全部完成
            torch.cuda.empty_cache()

            last_success = mid
            low = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = str(e)
            if (
                not isinstance(e, torch.cuda.OutOfMemoryError)
                and "out of memory" not in err_msg.lower()
            ):
                logger.warning(f"探测 batch_size={mid} 时遇到非 OOM 错误: {e}")
                if batch_ids is not None:
                    del batch_ids
                if batch_mask is not None:
                    del batch_mask
                if output is not None:
                    del output
                break
            if batch_ids is not None:
                del batch_ids
            if batch_mask is not None:
                del batch_mask
            if output is not None:
                del output
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            high = mid - 1

    optimal_bs = max(1, int(last_success * safety_margin))

    torch.cuda.empty_cache()
    logger.info(
        f"探测完成: 最大可行 batch_size={last_success}, "
        f"应用 {safety_margin:.0%} 安全余量后 → batch_size={optimal_bs}"
    )
    return optimal_bs


def normalize_scores(scores: List[float]) -> List[float]:
    """
    对分数列表进行归一化处理。

    参数:
        scores: 原始分数列表

    返回:
        归一化后的分数列表（范围 0-1）
    """
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def truncate_text(text: str, max_length: int = 65535) -> str:
    """
    截断文本到指定最大长度。

    参数:
        text: 原始文本
        max_length: 最大长度，默认 65535

    返回:
        截断后的文本
    """
    if len(text) > max_length:
        logger.warning(f"Truncating text from {len(text)} to {max_length} chars")
        return text[:max_length]
    return text


def com_similarity(
    retriever_name: str,
    target_text: str,
    compare_texts: List[str],
    target_side: str = "document",
    compare_side: str = "query",
) -> List[float]:
    """
    使用指定检索器计算目标文本与一批比较文本之间的余弦相似度，并逐条打印结果。

    本函数会根据 retriever_name 自动查找对应的模型配置文件
    （configs/models/{retriever_name}.yml），加载检索器模型，并按指定编码侧
    （query / document）分别编码目标文本和比较文本，计算余弦相似度后打印并返回。

    Args:
        retriever_name: 检索器配置文件名（不含路径和 .yml 后缀），
                        如 ``"harrier-oss-v1-0.6b"``、``"jina-code-embeddings-0.5b"``、
                        ``"gte-modernbert-base"``。
        target_text: 被比较的目标文本（单条）。
        compare_texts: 参与比较的文本列表（多条）。
        target_side: 目标文本的编码侧，``"query"`` 或 ``"document"``，默认 ``"document"``。
        compare_side: 比较文本的编码侧，``"query"`` 或 ``"document"``，默认 ``"query"``。

    Returns:
        余弦相似度列表，长度与 ``compare_texts`` 相同，顺序一一对应。

    Raises:
        FileNotFoundError: 找不到对应的模型配置文件时抛出。
        ValueError: retriever_name 未命中任何已注册模型前缀时抛出。

    Example:
        >>> sims = com_similarity(
        ...     retriever_name="harrier-oss-v1-0.6b",
        ...     target_text="void foo() { return 0; }",
        ...     compare_texts=["int bar() { return 1; }", "float baz() {}"],
        ... )
        [0] sim=0.8732  int bar() { return 1; }
        [1] sim=0.7541  float baz() {}
    """
    import os
    from src.utils.io import load_yaml
    from src.models.retriever.Base import create_embedder

    # 自动推断配置文件路径
    config_path = os.path.join("configs", "models", f"{retriever_name}.yml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"找不到检索器配置文件: {config_path}。"
            f"请确认 retriever_name='{retriever_name}' 与 configs/models/ 下的文件名一致。"
        )

    config = load_yaml(config_path)
    embedder = create_embedder(config)

    embed_fn = {
        "query": embedder.embed_queries,
        "document": embedder.embed_documents,
    }

    # 编码目标文本
    target_emb = np.asarray(embed_fn[target_side]([target_text]))[0]  # (dim,)

    # 编码比较文本
    compare_embs = np.asarray(embed_fn[compare_side](compare_texts))  # (N, dim)

    # 余弦相似度（向量已 L2 归一化时等价于点积）
    target_vec = torch.from_numpy(target_emb).float()
    compare_vec = torch.from_numpy(compare_embs).float()

    target_norm = target_vec / (target_vec.norm() + 1e-12)
    compare_norm = compare_vec / (compare_vec.norm(dim=1, keepdim=True) + 1e-12)
    sims: List[float] = (compare_norm @ target_norm).tolist()

    # 逐条打印
    model_tag = embedder.get_name(with_provider=False)
    print(
        f"\n[com_similarity] model={model_tag}  "
        f"target_side={target_side}  compare_side={compare_side}"
    )
    print(f"target: {target_text[:80]}{'...' if len(target_text) > 80 else ''}")
    print("-" * 60)
    for i, (sim, text) in enumerate(zip(sims, compare_texts)):
        preview = text[:80].replace("\n", " ")
        if len(text) > 80:
            preview += "..."
        print(f"[{i}] sim={sim:.4f}  {preview}")
    print("-" * 60)

    return sims


def extract_code_from_response(response_text: str) -> str:
    """
    从 LLM 响应中提取代码块。

    LLM 可能会返回带有 ```c 或 ``` 的 markdown 代码块，
    需要提取其中的实际代码内容。

    示例:
        输入: "```c\nint main() {...}\n```"
        输出: "int main() {...}"

    参数:
        response_text: LLM 的原始响应文本

    返回:
        提取出的代码内容，如果没有代码块则返回原始文本
    """
    code_pattern = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
    match = code_pattern.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()

