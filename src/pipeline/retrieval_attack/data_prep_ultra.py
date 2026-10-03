"""
方案四（Ultra）数据准备模块。

相比方案三（data_prep_v3.py）的核心变化：
  - 正样本直接从 poison_targets_set.json 的 entry["retrieval_attack"]["positive_samples"] 读取
  - 投毒对象判定改为检查 entry["retrieval_attack"] 中是否已有 "poisoned_buggy_code"

期望相似度最大范式仅使用正样本，因此不再准备对比学习所需的负样本。

输入/输出文件格式为嵌套字典 {CWE: [entries...]}，与 vics_vinj.py 一致。
"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch

from src.utils.token import count_mutable_tokens

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
#  待处理条目构建
# ────────────────────────────────────────────────────────────────

def build_pending_entries(
    target_data: Dict[str, List[Dict[str, Any]]],
    process_cwe: List[str],
    tokenizer,
    config: Dict[str, Any],
) -> List[tuple]:
    """
    从嵌套投毒目标数据中构建待处理条目列表（按 CWE 完成比例均衡交错排序）。

    遍历 process_cwe 指定的 CWE 类别，收集尚未完成检索攻击的条目，并按可扰动
    token 数量过滤。为使固定时间截断后各 CWE 的（已完成/可完成总数）比例保持接近，
    不再对全部待处理条目做全局 token 长度升序，而是采用**分数排名交错**：

      - 每个 CWE 内先按总 token 数升序排列（cheap-first，最大化固定时间内的完成数）；
      - 为该 CWE 内第 j 个待处理条目赋予分数排名
        ``r = (completed_c + j + 0.5) / T_c``，其中 completed_c 为该 CWE 已完成条目数，
        T_c = completed_c + 该 CWE 可完成的待处理条目数（被丢弃的过大/过小/无正样本条目
        不计入，使分数只反映可完成样本）；
      - 全局按 r 升序排列（tie-break：token 长度、CWE 名），逐条处理时任意时刻截断，
        各 CWE 都完成了各自约相同的比例；completed_c 偏移量令断点续跑时自动纠偏。

    跳过策略：
      - 已完成：entry["retrieval_attack"] 中已存在 "poisoned_buggy_code"；
      - 无正样本：entry["retrieval_attack"]["positive_samples"] 为空（无法参与优化）。

    Args:
        target_data: 投毒目标数据 ({CWE: [{entity, retrieval_attack, generation_attack}, ...]})
        process_cwe: 要处理的 CWE 类别列表
        tokenizer: 当前攻击配置所指定检索器的 tokenizer
        config: 完整攻击配置；过滤边界和安全词表规则均从中解析

    Returns:
        [(cwe_id, index_in_cwe_list, entry_ref), ...] 格式的待处理列表，
        其中 entry_ref 为原 dict 引用，可就地修改后写回。
    """
    attack_cfg = config.get("attack", {})
    sampling_cfg = config.get("sampling", {})
    min_mutable_tokens = int(sampling_cfg.get("min_condidate_num", 50))
    configured_max = int(sampling_cfg.get("max_condidate_num", 6000))
    candidate_budget = int(attack_cfg.get("n", 6000))
    if min_mutable_tokens < 0:
        raise ValueError("sampling.min_condidate_num 不能小于0")
    if configured_max < min_mutable_tokens:
        raise ValueError(
            "sampling.max_condidate_num 不能小于 min_condidate_num"
        )
    if candidate_budget <= 0:
        raise ValueError("attack.n 必须大于0")
    effective_max = min(configured_max, candidate_budget)
    if effective_max < min_mutable_tokens:
        raise ValueError(
            "min_condidate_num 超过 min(max_condidate_num, attack.n)，"
            "将不可能保留任何投毒样本"
        )

    use_safe_vocab = bool(attack_cfg.get("use_safe_vocab", True))
    # frozen_token_dict = (
    #     attack_cfg.get("frozen_token_dict", {}) if use_safe_vocab else {}
    # )
    frozen_token_dict = (
        attack_cfg.get("frozen_token_dict", {}) # 统一使用冻结词表计算可扰动词元数量, 不再区分是否启用安全词表
    )
    logger.info(
        "可扰动词元过滤：闭区间 [%d, %d] "
        "(configured_max=%d, attack.n=%d, safe_vocab=%s)",
        min_mutable_tokens,
        effective_max,
        configured_max,
        candidate_budget,
        use_safe_vocab,
    )
    # 每个 CWE 的待处理条目（含总 token 数）与已完成计数，用于分数排名交错
    per_cwe_pending: Dict[str, List[tuple]] = {}  # cwe -> [(n_total, idx, entry), ...]
    per_cwe_completed: Dict[str, int] = {}

    for cwe in process_cwe:
        entries = target_data.get(cwe, [])
        if not entries:
            logger.warning(f"[{cwe}] 目标数据中无该 CWE 类别，跳过")
            continue

        cwe_skipped = 0
        cwe_no_pos = 0
        cwe_too_small = 0
        cwe_too_large = 0
        cwe_items: List[tuple] = []

        for idx, entry in enumerate(entries): # entry包含了retrieval_attack和entity
            ra = entry.get("retrieval_attack", {})
            if ra.get("poisoned_buggy_code"):
                cwe_skipped += 1
                continue

            # 保险措施：positive_samples 为空的目标无法参与对抗优化，直接剔除
            if not ra.get("positive_samples"):
                cwe_no_pos += 1
                continue

            buggy_code = entry.get("entity", {}).get("buggy_code", "")
            n_mutable = count_mutable_tokens(buggy_code, tokenizer, frozen_token_dict)
            if n_mutable < min_mutable_tokens:
                cwe_too_small += 1
                continue
            if n_mutable > effective_max:
                cwe_too_large += 1
                continue

            n_total = len(tokenizer.encode(buggy_code, add_special_tokens=True)) # 计算投毒目标的总token数
            cwe_items.append((n_total, idx, entry))

        # CWE 内按总 token 数升序（cheap-first）
        cwe_items.sort(key=lambda x: x[0])
        per_cwe_pending[cwe] = cwe_items
        per_cwe_completed[cwe] = cwe_skipped

        logger.info(
            f"[{cwe}] 待处理: {len(cwe_items)}, 已完成跳过: {cwe_skipped}, "
            f"无正样本剔除: {cwe_no_pos}, "
            f"过小丢弃: {cwe_too_small}, 过大丢弃: {cwe_too_large}, "
            f"允许可扰动区间: [{min_mutable_tokens}, {effective_max}], "
            f"总计: {len(entries)}"
        )

    # ── 分数排名交错：各 CWE 均衡推进（completed 偏移量保证断点续跑自纠偏）──
    ranked: List[tuple] = []  # (fractional_rank, n_total, cwe, idx, entry)
    for cwe, cwe_items in per_cwe_pending.items():
        completed_c = per_cwe_completed.get(cwe, 0)
        # T_c 只统计可完成样本（已完成 + 可处理待处理），排除被丢弃的条目
        t_c = completed_c + len(cwe_items)
        if t_c == 0:
            continue
        for j, (n_total, idx, entry) in enumerate(cwe_items):
            frac_rank = (completed_c + j + 0.5) / t_c
            ranked.append((frac_rank, n_total, cwe, idx, entry))

    # 全局按分数排名升序；tie-break：token 长度升序、CWE 名，保证确定性
    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    pending: List[tuple] = [
        (cwe, idx, entry) for _, _, cwe, idx, entry in ranked
    ]

    if ranked:
        token_lens = [n_total for _, n_total, _, _, _ in ranked]
        logger.info(
            f"待处理条目总计: {len(pending)}"
            f"，总 token 区间: [{min(token_lens)}, {max(token_lens)}]"
        )
    else:
        logger.info(f"待处理条目总计: {len(pending)}")
    return pending


# ────────────────────────────────────────────────────────────────
#  统一数据准备入口
# ────────────────────────────────────────────────────────────────

def prepare_attack_data(
    entry: Dict[str, Any],
    embedder,
) -> Dict[str, Any]:
    """
    方案四统一数据准备入口。

    投毒对象使用 document-side 编码；正样本直接从
    entry["retrieval_attack"]["positive_samples"] 读取并使用 query-side 编码。
    期望相似度最大范式仅使用正样本，因此不涉及负样本。

    Args:
        entry: 单个投毒目标条目 ({entity, retrieval_attack, generation_attack})
        embedder: BaseEmbedder 实例

    Returns:
        包含以下键的字典:
          - poison_embedding: document-side 投毒对象嵌入 (embed_dim,)
          - pos_embeddings: query-side 正样本嵌入 (num_pos, embed_dim)
          - pos_items: 正样本元数据列表
    """
    entity = entry["entity"]
    buggy_code = entity["buggy_code"]

    positive_items = entry.get("retrieval_attack", {}).get("positive_samples", [])
    if not positive_items:
        raise ValueError(
            f"投毒目标 id={entity.get('id')} 的 retrieval_attack.positive_samples 为空"
        )

    st_model = embedder.model
    st_model.eval()

    with torch.no_grad():
        poison_emb = embedder.embed_documents([buggy_code])
        poison_embedding = torch.from_numpy(np.asarray(poison_emb[0])).float()

        positive_codes = [item["buggy_code"] for item in positive_items]
        positive_emds = embedder.embed_queries(positive_codes)
        positive_embeddings = torch.from_numpy(np.asarray(positive_emds)).float()

        for idx, item in enumerate(positive_items):
            item["embedding"] = positive_embeddings[idx].tolist()

        logger.info(f"正样本 query-side 嵌入完成: |Q+|={positive_embeddings.shape[0]}")

    return {
        "poison_embedding": poison_embedding,
        "pos_embeddings": positive_embeddings,
        "pos_items": positive_items,
    }
