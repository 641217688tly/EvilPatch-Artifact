"""
Approximate Greedy Gradient Descent (AGGD) 原版优化器（基线 / 消融）。

本模块忠实复现 AGGD 论文（Corpus Poisoning，Algorithm 1）的原版算法，作为
本课题改进版 ``aggs_optimizer``（整篇文档就地替换 + 安全词表）的消融对照。

与改进版 AGGS 的核心差异
------------------------
- AGGS: 直接对**整篇代码 token 序列**做就地替换，并使用安全词表 V_safe 保持语义。
- AGGD（原版）: 在代码序列**首部插入一段定长对抗序列**（Adversarial Perturbation
  Sequence），并**仅优化该插入序列**；优化在**全词表**上进行（``use_safe_vocab=false``），
  不冻结关键字、不做语义保持。

与基线 PABS 的核心差异
----------------------
- PABS: 联合搜索插入位置（位置感知）+ 束搜索（beam）优化插入序列。
- AGGD（原版）: 插入位置**固定**在代码首部（无位置搜索），采用**贪婪 + depth**
  单束搜索（等价于「移除位置感知 / 束搜索的 PABS」+ AGGD 的 depth 机制）。

算法概述（对应论文 Algorithm 1，仅优化插入序列）
-----------------------------------------------
输入: 干净代码 C, 正样本集 Q+, 迭代轮数 N, 候选集大小 n,
      对抗序列长度 L
输出: 优化后的对抗代码 A

1. a_code ← prefix + Tokenize(C): 构建完整代码 token 序列
2. a_seq  ← [init]*L: 初始化长度为 L 的对抗序列
3. p ← prefix_len + 1: 固定插入位置（代码区首部，紧跟前导 special/BOS token）
4. d ← 0: 初始化搜索深度
5. for j = 1 to N:
   a) full = Insert(a_code, a_seq, p): 拼接完整对抗样本，对抗位置为 [p, p+L)
   b) L(a) ← 计算损失并反向传播获取梯度
   c) 对抗序列每个位置 i: 计算一阶泰勒近似分数 s(v) = e_v^T · (-∇_{e_{t_i}} L(a))
   d) 依据搜索深度 d 选取排名在 [d·k, (d+1)·k) 的候选（k = max(1, n/L)）
   e) 评估全局候选集，选取损失最小者 a'
   f) if L(a') < L(a): 接受更新 a_seq, d ← 0; else: d ← d + 1
   g) 检查成功阈值、Loss patience 与候选空间耗尽早停
6. return decode(Insert(a_code, a_seq*, p))
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.pipeline.retrieval_attack.optimizer.pabs_optimizer import insert_adv_sequence
from src.pipeline.retrieval_attack.loss_func import (
    GradientStorage,
    compute_avg_similarity,
    compute_expected_sim_loss_and_grad,
    compute_per_sample_similarity,
    compute_retrieval_attack_loss_batch,
    compute_weighted_expected_sim_loss_and_grad,
)
from src.utils.dataclass import RetrievalAdvIterLog

logger = logging.getLogger(__name__)


def _compute_candidate_scores(
    grad: torch.Tensor,
    safe_emb_matrix: torch.Tensor,
) -> torch.Tensor:
    """AGGD 自有的位置内候选排序分数，不依赖 ABGS。"""
    with torch.no_grad():
        return torch.matmul(safe_emb_matrix, (-grad).T)


def _evaluate_candidates_pool(
    embedder,
    candidates: List[Tuple[torch.LongTensor, int, int]],
    attention_mask: torch.LongTensor,
    pos_embeddings: torch.Tensor,
    loss_func: str = "expected_sim",
    dynamic_weight: bool = False,
    tau_stop: float = 0.70,
    alpha: float = 1.0,
    softmin_temperature: float = 0.1,
    hybrid_softmin_weight: float = 0.5,
    batch_size: int = 64,
) -> List[Tuple[torch.LongTensor, float, torch.Tensor]]:
    """AGGD 专用的候选批量评估，保持原返回结构。"""
    results: List[Tuple[torch.LongTensor, float, torch.Tensor]] = []
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            batch_ids_list: List[torch.LongTensor] = []
            for base_ids, position, new_token_id in batch:
                candidate_ids = base_ids.clone()
                candidate_ids[0, position] = new_token_id
                batch_ids_list.append(candidate_ids)
            batch_ids = torch.cat(batch_ids_list, dim=0)
            batch_mask = attention_mask.expand(len(batch), -1)
            batch_embeddings = embedder.embed_token_ids(batch_ids, batch_mask)
            losses = compute_retrieval_attack_loss_batch(
                batch_embeddings,
                pos_embeddings,
                loss_func=loss_func,
                dynamic_weight=dynamic_weight,
                tau_stop=tau_stop,
                alpha=alpha,
                softmin_temperature=softmin_temperature,
                hybrid_softmin_weight=hybrid_softmin_weight,
            )
            for index in range(len(batch)):
                results.append((
                    batch_ids[index:index + 1].clone(),
                    float(losses[index].item()),
                    batch_embeddings[index:index + 1].detach(),
                ))
    return results


# ─────────────────────────────────────────────────────────────────
#  AGGD 候选生成（带 depth 偏移，仅作用于插入的对抗序列）
# ─────────────────────────────────────────────────────────────────

def select_adv_candidates_with_depth(
    scores: torch.Tensor,
    adv_positions_in_full: List[int],
    safe_ids: torch.LongTensor,
    current_full_token_ids: List[int],
    depth: int,
    n: int,
) -> List[Tuple[int, int]]:
    """
    从分数矩阵中为对抗序列的每个位置选取候选 token（含 AGGD 的 depth 偏移）。

    与改进版 AGGS 的 ``select_candidates`` 同源（保留 depth 机制），
    但操作位置限定在**插入的对抗序列**在完整序列中的绝对位置。

    对每个对抗位置 i:
      1. 取该位置的分数列，降序排列
      2. 排除该位置当前 token（避免无意义替换）
      3. 根据搜索深度 d 和每位置候选数 k = max(1, n/L)，
         选取排名在 [d·k, (d+1)·k) 的候选（depth 机制，探索更低排名候选以跳出局部最优）

    Args:
        scores: 分数矩阵 ``(|V_safe|, full_seq_len)``
        adv_positions_in_full: 对抗序列在完整序列中的绝对位置列表（长度 L）
        safe_ids: 候选 token ID 映射表 ``(|V_safe|,)``（原版 AGGD 下即全词表非 special token）
        current_full_token_ids: 当前完整序列的 token ID 列表
        depth: 当前搜索深度 d（从 0 开始）
        n: 全局候选集目标大小

    Returns:
        候选列表 [(absolute_position_in_full, new_token_id), ...]；
        若当前深度已超出词表范围（候选耗尽）则返回空列表
    """
    m = len(adv_positions_in_full)  # 对抗序列长度 L
    if m == 0:
        return []

    k = max(1, n // m)  # 每个对抗位置的候选数

    with torch.no_grad():
        pos_tensor = torch.tensor(
            adv_positions_in_full, dtype=torch.long, device=scores.device
        )
        sub_scores = scores[:, pos_tensor].clone()  # (|V_safe|, m)

        # 排除各位置当前 token（将自身分数置为 -inf）
        for local_i, pos in enumerate(adv_positions_in_full):
            current_tid = current_full_token_ids[pos]
            match = (safe_ids == current_tid).nonzero(as_tuple=True)[0]
            if match.numel() > 0:
                sub_scores[match[0], local_i] = float("-inf")

        vocab_size = sub_scores.shape[0]
        start = depth * k
        if start >= vocab_size:
            # depth 偏移已超出词表：候选耗尽
            return []

        top_end = min((depth + 1) * k, vocab_size)
        _, top_indices = torch.topk(
            sub_scores, top_end, dim=0, largest=True, sorted=True
        )
        # top_indices: (top_end, m)，取 depth 偏移窗口 [start:top_end]
        selected_indices = top_indices[start:top_end, :]  # (k', m)
        selected_tids = safe_ids[selected_indices].tolist()  # (k', m) Python list

    candidates: List[Tuple[int, int]] = [
        (adv_positions_in_full[local_i], selected_tids[ki][local_i])
        for local_i in range(m)
        for ki in range(len(selected_tids))
    ]
    return candidates


# ─────────────────────────────────────────────────────────────────
#  AGGD 主优化循环
# ─────────────────────────────────────────────────────────────────

def aggd_optimize(
    embedder,
    tokenizer,
    poison_buggy_code: str,
    pos_embeddings: torch.Tensor,
    safe_ids: torch.LongTensor,
    safe_emb_matrix: torch.Tensor,
    grad_storage: GradientStorage,
    frozen_token_dict: dict,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    AGGD 原版近似贪婪梯度下降对抗攻击主循环。

    函数签名与 ``aggs_optimize`` / ``abgs_optimize`` / ``pabs_optimize`` 保持一致，
    便于 ``run_attack_ultra.py`` 动态切换。

    Args:
        embedder: BaseEmbedder 实例（底层模型需设为 eval 模式并保留梯度）
        tokenizer: 模型对应的 tokenizer
        poison_buggy_code: 投毒对象的原始 buggy code 文本
        pos_embeddings: 正样本嵌入 ``(|Q+|, dim)``
        safe_ids: 候选 token ID ``(|V_safe|,)``；原版 AGGD 通常配置
                  ``use_safe_vocab=false``，此时该表为全词表非 special token
        safe_emb_matrix: 候选嵌入子矩阵 ``(|V_safe|, dim)``
        grad_storage: 已注册到词嵌入层的 GradientStorage 实例
        frozen_token_dict: 冻结 token 集合（原版 AGGD 中通常为空字典）
        config: 完整配置字典

    Returns:
        结果字典（与 aggs_optimize / abgs_optimize / pabs_optimize 返回格式一致）
    """
    attack_cfg = config.get("attack", config)
    N = attack_cfg.get("N", 100)
    n = attack_cfg.get("n", 750)
    tau_stop = attack_cfg.get("tau_stop", 0.80)  # 早停阈值
    loss_func_type = attack_cfg.get("loss_func", "expected_sim")  # 损失函数类型
    dynamic_weight = attack_cfg.get("dynamic_weight", False)  # 是否使用动态权重
    alpha = attack_cfg.get("scaling_factor", 1.0)  # 动态权重缩放系数
    adv_seq_length = attack_cfg.get("adv_seq_length", 20)  # 对抗序列长度 L
    patience = int(attack_cfg.get("patience", 10))  # 0 表示禁用 Loss patience
    batch_size = config.get("model", {}).get("batch_size", 96)  # 候选评估批处理大小
    device = pos_embeddings.device

    if patience < 0:
        raise ValueError(f"patience 必须大于等于0，当前值为 {patience}")

    # ── (1) 初始化代码 token 序列: a_code ← prefix + Tokenize(C) ──
    logger.info("========== AGGD 原版贪婪梯度下降优化初始化 ==========")

    prefix_ids = embedder.get_instruct_tids(side="document")
    prefix_len = len(prefix_ids)
    code_ids = tokenizer.encode(poison_buggy_code, add_special_tokens=True)
    full_code_ids_list = prefix_ids + code_ids
    code_ids_tensor = torch.tensor(
        [full_code_ids_list], dtype=torch.long, device=device
    )  # (1, prefix_len + code_len)

    logger.info(
        f"(1) 初始化代码序列: a_code ← prefix + Tokenize(C)，"
        f"代码长度 {len(code_ids)} tokens"
        + (f"，prompt prefix {prefix_len} tokens" if prefix_len else "")
    )

    # ── (2) 初始化对抗序列: a_seq ← [init]*L ──
    # 初始占位 token 优先级：mask_token_id > unk_token_id > 0
    #   - mask/unk 为特殊占位符，语义中性，便于后续被梯度替换
    #   - 若二者均不存在则退化为 0（PAD/首个 token），仅作占位，会在优化中被替换
    init_tid = tokenizer.mask_token_id
    if init_tid is None:
        init_tid = tokenizer.unk_token_id
    if init_tid is None:
        init_tid = 0
    adv_seq = torch.full(
        (1, adv_seq_length), init_tid, dtype=torch.long, device=device
    )  # (1, L)
    logger.info(
        f"(2) 初始化对抗序列: a_seq ← [init]*{adv_seq_length}（init_tid={init_tid}）"
    )

    # ── (3) 固定插入位置: p ← prefix_len + 1（代码区首部，紧跟前导 special/BOS token） ──
    # 沿用 PABS 的不变量 p >= prefix_len + 1，保证对抗序列落在代码区而非 prefix 区；
    # 插入到代码首个 token（通常为 BOS/CLS）之后，可避免破坏 last-token pooling 的末位 EOS。
    insert_pos = prefix_len + 1
    logger.info(
        f"(3) 固定插入位置: p ← prefix_len + 1 = {insert_pos}（无位置搜索）"
    )

    # 构建完整序列的注意力掩码（长度 = 代码 + 对抗序列，对所有位置恒定）
    total_len = len(full_code_ids_list) + adv_seq_length
    attn_mask = torch.ones(1, total_len, dtype=torch.long, device=device)

    # 对抗序列在完整序列中的绝对位置: [p, p+1, ..., p+L-1]
    adv_positions_in_full = list(range(insert_pos, insert_pos + adv_seq_length))

    # ── (4) 计算干净代码基线（不含对抗序列） ──
    code_attn_mask = torch.ones_like(code_ids_tensor, device=device)
    with torch.no_grad():
        clean_emb = embedder.embed_token_ids(code_ids_tensor, code_attn_mask)
    clean_sims = compute_per_sample_similarity(clean_emb.squeeze(0), pos_embeddings)
    clean_avg_sim = clean_sims.mean().item()
    logger.info(
        f"(4) 计算干净代码基线: Sim(q+, C)，平均相似度到 Q+ = {clean_avg_sim:.4f}"
    )

    # ── (5) 初始化搜索深度 ──
    depth = 0
    logger.info(f"(5) 初始化搜索深度: d ← 0")

    weight_label = f"dynamic_weight={dynamic_weight}, alpha={alpha}"
    logger.info(
        f"算法参数：N={N}（迭代轮数）, n={n}（候选集大小）, "
        f"L={adv_seq_length}（对抗序列长度）, "
        f"tau_stop={tau_stop}（早停阈值）, "
        f"patience={patience}（Loss Patience，0 表示禁用）, "
        f"loss_func={loss_func_type}, {weight_label}, "
        f"|V_safe|={safe_ids.shape[0]}"
    )

    # 记录最佳状态
    best_adv_seq = adv_seq.clone()
    best_loss = float("inf")
    best_iteration = 0

    iteration_logs: List[str] = []
    early_stopped = False
    stop_reason = "max_iterations"
    patience_counter = 0
    completed_iterations = 0
    init_loss: Optional[float] = None
    init_sim: Optional[List[float]] = None
    init_avg_sim: float = clean_avg_sim

    start_time = datetime.now().isoformat()

    # ── 主迭代循环: for j = 1 to N do ──
    for j in range(1, N + 1):
        completed_iterations = j
        logger.info(f"========== 第 {j}/{N} 轮迭代开始（depth={depth}）==========")

        # (6) 拼接完整对抗样本: full = Insert(a_code, a_seq, p)
        full_ids = insert_adv_sequence(code_ids_tensor, adv_seq, insert_pos)  # (1, total_len)
        full_token_list = full_ids[0].tolist()

        # ── (6) 计算损失 L(a) 与梯度 ──
        if loss_func_type == "expected_sim" and dynamic_weight:
            loss_before, grad, cur_emb = compute_weighted_expected_sim_loss_and_grad(
                embedder, full_ids, attn_mask,
                pos_embeddings, grad_storage,
                tau_stop, alpha,
            )
        elif loss_func_type == "expected_sim":
            loss_before, grad, cur_emb = compute_expected_sim_loss_and_grad(
                embedder, full_ids, attn_mask,
                pos_embeddings, grad_storage,
            )

        logger.info(
            f"(6) 计算损失: L(a)（{loss_func_type}, weighted={dynamic_weight}）, "
            f"loss={loss_before:.6f}"
        )

        avg_sim_before = compute_avg_similarity(cur_emb.squeeze(0), pos_embeddings)
        logger.info(f"    当前平均相似度到 Q+：avg_sim={avg_sim_before:.4f}")

        if j == 1:
            best_loss = loss_before
            init_loss = loss_before
            init_sim = compute_per_sample_similarity(
                cur_emb.squeeze(0), pos_embeddings
            ).tolist()

        # ── (7) 批量计算一阶泰勒近似分数 ──
        scores = _compute_candidate_scores(grad, safe_emb_matrix)  # (|V_safe|, total_len)
        logger.info(
            f"(7) 计算一阶泰勒近似分数: score(v) = e_v^T · (-∇_(e_t) L(a))，"
            f"scores shape={tuple(scores.shape)}"
        )

        # ── (8) 构造全局候选集 D^(j)（仅作用于对抗序列位置，含 depth 偏移） ──
        adv_candidates = select_adv_candidates_with_depth(
            scores, adv_positions_in_full, safe_ids,
            full_token_list, depth, n,
        )
        k = max(1, n // adv_seq_length)
        logger.info(
            f"(8) 构造全局候选集: D^(j) ← {{a[t_i←v] | v∈V_ranked[d·k:(d+1)·k]}}，"
            f"{len(adv_candidates)} 个候选 (depth={depth}, k={k}, L={adv_seq_length})"
        )

        if not adv_candidates:
            logger.warning(
                f"  候选集为空（depth={depth} 已超出词表范围），"
                "以 candidate_space_exhausted 正常结束优化"
            )
            early_stopped = True
            stop_reason = "candidate_space_exhausted"
            iteration_logs.append(
                f"AGGD Iteration {j}/{N}\n"
                f"  depth: {depth}\n"
                "  [Stop] candidate_space_exhausted: 候选池为空，"
                "返回历史最优 Loss 序列\n"
            )
            break

        # ── (9) 评估候选集并选择最优 a' ──
        # candidate_pool 元素格式: (base_full_ids, absolute_position, new_token_id)
        candidate_pool: List[Tuple[torch.LongTensor, int, int]] = [
            (full_ids, abs_pos, new_tid) for abs_pos, new_tid in adv_candidates
        ]
        if not candidate_pool:
            logger.warning(
                f"  第 {j} 轮实际候选池为空，以 candidate_space_exhausted 正常结束优化"
            )
            early_stopped = True
            stop_reason = "candidate_space_exhausted"
            iteration_logs.append(
                f"AGGD Iteration {j}/{N}\n"
                f"  depth: {depth}\n"
                "  [Stop] candidate_space_exhausted: 实际候选池为空，"
                "返回历史最优 Loss 序列\n"
            )
            break
        logger.info(
            f"(9) 选择最优候选: a' ← argmin_{{x∈D^(j)}} L(x)，"
            f"评估 {len(candidate_pool)} 个候选（batch_size={batch_size}）..."
        )
        evaluated = _evaluate_candidates_pool(
            embedder, candidate_pool, attn_mask,
            pos_embeddings,
            loss_func=loss_func_type,
            dynamic_weight=dynamic_weight,
            tau_stop=tau_stop,
            alpha=alpha,
            batch_size=batch_size,
        )

        # 在候选池评估结果中选取损失最小者
        best_idx = min(range(len(evaluated)), key=lambda i: evaluated[i][1])
        cand_full_ids, cand_loss, cand_emb = evaluated[best_idx]
        cand_pos, cand_new_tid = adv_candidates[best_idx]
        logger.info(f"    最优候选损失：cand_loss={cand_loss:.6f}")

        # ── (10) 判断是否更新并调整搜索深度 ──
        log_entry = RetrievalAdvIterLog(
            iteration=j,
            total_iterations=N,
            loss_before=loss_before,
            depth=depth,
        )

        if cand_loss < loss_before:  # 接受更新: L(a') < L(a)
            old_tid = full_ids[0, cand_pos].item()
            old_text = tokenizer.decode([old_tid], skip_special_tokens=False, clean_up_tokenization_spaces=True)
            new_text = tokenizer.decode([cand_new_tid], skip_special_tokens=False, clean_up_tokenization_spaces=True)

            # 从最优候选完整序列中提取更新后的对抗序列
            adv_seq = cand_full_ids[:, insert_pos:insert_pos + adv_seq_length].clone()
            depth = 0

            cur_sims = compute_per_sample_similarity(cand_emb.squeeze(0), pos_embeddings)
            per_cur_sims = cur_sims.tolist()
            avg_sim_after = compute_avg_similarity(cand_emb.squeeze(0), pos_embeddings)

            loss_improved = cand_loss < best_loss
            if loss_improved:
                best_loss = cand_loss
                best_adv_seq = adv_seq.clone()
                best_iteration = j

            if loss_improved:
                patience_counter = 0
                logger.info(
                    f"    历史最优 Loss 更新：best_loss={best_loss:.6f}，"
                    "patience_counter 重置为 0"
                )
            else:
                patience_counter += 1
                logger.info(
                    f"    历史最优 Loss 未改善，"
                    f"patience_counter={patience_counter}/{patience}"
                )

            log_entry.updated = True
            log_entry.token_changes = [(cand_pos, old_tid, cand_new_tid, old_text, new_text)]
            log_entry.loss_after = cand_loss
            log_entry.avg_sim_before = avg_sim_before
            log_entry.avg_sim_after = avg_sim_after
            log_entry.per_sample_sims = per_cur_sims
            log_entry.depth = depth
            log_entry.adv_text_preview = tokenizer.decode(
                cand_full_ids[0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True
            )[:200]

            logger.info(f"(10) 更新决策：接受更新 (L(a') < L(a))，重置搜索深度 d←0")
            logger.info(f"    位置 {cand_pos}: \"{old_text}\" → \"{new_text}\"")
            logger.info(f"    损失变化：{loss_before:.6f} → {cand_loss:.6f} (Δ={cand_loss - loss_before:+.6f})")
            logger.info(f"    平均相似度变化：{avg_sim_before:.4f} → {avg_sim_after:.4f} (Δ={avg_sim_after - avg_sim_before:+.4f})")
            logger.info(f"    各正样本相似度：[{', '.join(f'{s:.4f}' for s in per_cur_sims)}]")

            # ── (11) 早停检查（仅在接受更新后执行）
            #   条件: ∀ q+ ∈ Q+, Sim(q+, a) > max(Sim(q+, C), τ_stop)
            thresholds = torch.max(clean_sims, torch.tensor(tau_stop, device=device))
            thresholds_list = thresholds.tolist()

            logger.info(f"(11) 早停检查: if ∀q+∈Q+, Sim(q+,a) > max(Sim(q+,C), τ_stop)")
            logger.info(f"    当前各正样本相似度：[{', '.join(f'{s:.4f}' for s in per_cur_sims)}]")
            logger.info(
                f"    阈值要求（max(干净相似度, {tau_stop})）："
                f"[{', '.join(f'{t:.4f}' for t in thresholds_list)}]"
            )

            if (cur_sims > thresholds).all():
                logger.info(f"    >>> 早停条件满足 @ 第 {j} 轮：所有正样本相似度均超过阈值")
                logger.info(
                    f"    最小当前相似度 {cur_sims.min().item():.4f} > "
                    f"阈值 {thresholds.min().item():.4f}"
                )
                early_stopped = True
                stop_reason = "success_threshold"
                iteration_logs.append(log_entry.format())
                break
            else:
                failed = (~(cur_sims > thresholds)).nonzero(as_tuple=True)[0].tolist()
                failed_str = ', '.join(
                    f'#{i+1}({per_cur_sims[i]:.4f} <= {thresholds_list[i]:.4f})'
                    for i in failed[:3]
                )
                if len(failed) > 3:
                    failed_str += f' 等 {len(failed)} 个'
                logger.info(f"    早停条件未满足：{failed_str} 未超过阈值")
        else:  # 拒绝更新: 增加搜索深度
            depth += 1
            patience_counter += 1

            log_entry.updated = False
            log_entry.token_changes = []
            log_entry.loss_after = loss_before
            log_entry.avg_sim_before = avg_sim_before
            log_entry.avg_sim_after = avg_sim_before
            log_entry.depth = depth
            log_entry.adv_text_preview = tokenizer.decode(
                full_token_list, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )[:200]

            logger.info(f"(10) 更新决策：拒绝更新 (L(a') >= L(a))，增加搜索深度 d←{depth}")
            logger.info(f"    候选损失 {cand_loss:.6f} >= 原损失 {loss_before:.6f}")
            logger.info(f"    当前损失：{loss_before:.6f}, 平均相似度：{avg_sim_before:.4f}")
            logger.info(
                f"    历史最优 Loss 未改善，"
                f"patience_counter={patience_counter}/{patience}"
            )

        iteration_logs.append(log_entry.format())

        if patience > 0 and patience_counter >= patience:
            logger.info(
                f"    >>> Patience 早停 @ 第 {j} 轮："
                f"连续 {patience} 轮历史最优 Loss 未改善"
            )
            early_stopped = True
            stop_reason = "patience"
            break

    # ── (12) 返回结果: A ← decode(Insert(a_code, a_seq*, p)) ──
    total_iters = completed_iterations

    best_full_ids = insert_adv_sequence(code_ids_tensor, best_adv_seq, insert_pos)
    final_ids_list = best_full_ids[0].tolist()[prefix_len:]
    final_text = tokenizer.decode(final_ids_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    with torch.no_grad():
        final_emb = embedder.embed_token_ids(best_full_ids, attn_mask)
    final_avg_sim = compute_avg_similarity(final_emb.squeeze(0), pos_embeddings)
    final_sim = compute_per_sample_similarity(final_emb.squeeze(0), pos_embeddings).tolist()

    end_time = datetime.now().isoformat()
    logger.info("========== AGGD 原版贪婪梯度下降优化结束 ==========")
    logger.info(f"(12) 返回最终对抗代码: A ← decode(Insert(a_code, a_seq*, p))")
    logger.info(f"    总迭代轮数：{total_iters} / {N}")
    logger.info(f"    早停状态：{'是' if early_stopped else '否'}")
    logger.info(f"    停止原因：{stop_reason}")
    logger.info(f"    最优结果轮次：{best_iteration}")
    logger.info(f"    Patience 计数：{patience_counter}/{patience}")
    logger.info(f"    最终损失：{best_loss:.6f}")
    logger.info(f"    最终平均相似度到 Q+：{final_avg_sim:.4f}")
    logger.info(f"    干净代码基线相似度：{clean_avg_sim:.4f}")
    logger.info(f"    相似度提升：{final_avg_sim - clean_avg_sim:+.4f}")
    logger.info(f"    固定插入位置：{insert_pos}")
    logger.info(f"    对抗序列 token IDs：{best_adv_seq[0].tolist()}")

    return {
        "adv_text": final_text,
        "adv_token_ids": final_ids_list,
        "init_loss": init_loss,
        "init_sim": init_sim,
        "init_avg_sim": init_avg_sim,
        "final_loss": best_loss,
        "final_sim": final_sim,
        "final_avg_sim": final_avg_sim,
        "iteration_logs": iteration_logs,
        "total_iterations": total_iters,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "patience_counter": patience_counter,
        "best_iteration": best_iteration,
        "start_time": start_time,
        "end_time": end_time,
        "algorithm": "aggd",
        "insert_position": insert_pos,
        "adv_seq_token_ids": best_adv_seq[0].tolist(),
    }
