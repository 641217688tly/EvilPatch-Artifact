"""
Approximate Greedy Gradient Search (AGGS) 核心优化器。

AGGS 是本课题在原版 AGGD（论文 Algorithm 1）基础上改进得到的贪婪梯度搜索优化器，
与束搜索版 ``abgs_optimizer`` 构成 Greedy/Beam 对偶。相较于原版 AGGD（在文档首部
插入一段定长对抗序列并仅优化该序列），AGGS 做了以下定制：
  - 直接对**整篇文档 token 序列**进行就地替换（in-place substitution），而非插入定长序列
  - 使用安全候选词表 V_safe 约束替换范围，冻结 C/C++ 关键字与标点以保持代码语义
  - 实现基于相似度阈值的早停机制

算法概述
--------
输入: 干净代码 C, 正样本集 Q+, 迭代轮数 N, 候选集大小 n
输出: 优化后的对抗代码 A

1. a ← Tokenize(C): 用干净代码初始化对抗样本
2. d ← 0: 初始化搜索深度
3. for j = 1 to N:
   a) M ← 识别可扰动的 token 位置（排除关键字和 special tokens）
   b) L(a) ← 一次前向计算损失、一次反向传播获取所有可扰动位置的梯度
   c) 构建全局分数矩阵 S ∈ R^{|V_safe|×|M|}，S[v,i] = e_v^T · (-∇_{e_{t_i}} L(a))
      （一次 GEMM 批量完成）
   d) 将 S 展平为一维向量并按分数降序全局排名，截取深度窗口 [d·n, (d+1)·n) 的
      n 个 (v, i) 组合构成全局候选集 D（不再按位置均摊预算，候选池与 |M| 解耦）
   e) 对每个候选评估损失，选择最优: a' = argmin L(x)
   f) if L(a') < L(a): 接受更新, d ← 0; else: d ← d + 1
   g) 早停检查
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.pipeline.retrieval_attack.loss_func import (
    GradientStorage,
    compute_avg_similarity,
    compute_expected_sim_loss_and_grad,
    compute_expected_sim_loss_batch,
    compute_per_sample_similarity,
    compute_weighted_expected_sim_loss_and_grad,
    compute_weighted_expected_sim_loss_batch,
)

from src.utils.dataclass import RetrievalAdvIterLog
from src.utils.token import get_mutable_positions

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  候选生成
# ─────────────────────────────────────────────────────────────────

def compute_candidate_scores(
    grad: torch.Tensor,
    safe_emb_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    用一次矩阵乘法批量计算所有安全 token 在所有位置的替换分数。

    数学公式:
        scores[v, i] = e_v^T · (-∇_{e_{t_i}} L(a))

    其中:
      - e_v: 安全候选 token v 的词嵌入向量 (embed_dim,)
      - ∇_{e_{t_i}} L(a): 位置 i 处的梯度
      - 取负梯度方向，使得 score 越大代表替换后损失下降越多

    Args:
        grad: 梯度张量，形状 ``(seq_len, embed_dim)``
        safe_emb_matrix: 安全词表嵌入矩阵，形状 ``(|V_safe|, embed_dim)``

    Returns:
        分数矩阵，形状 ``(|V_safe|, seq_len)``
    """
    # safe_emb_matrix: (|V_safe|, embed_dim)，仅包含了各安全token的嵌入向量
    # (-grad).T:       (embed_dim, seq_len)
    # 结果:            (|V_safe|, seq_len)
    with torch.no_grad():
        scores = torch.matmul(safe_emb_matrix, (-grad).T)
    return scores


def select_candidates_global(
    scores: torch.Tensor,
    mutable_positions: List[int],
    safe_ids: torch.LongTensor,
    current_token_ids: List[int],
    depth: int,
    n: int,
) -> List[Tuple[int, int]]:
    """
    从全局分数矩阵中按搜索深度窗口 [d·n, (d+1)·n) 选取候选，构造全局候选集 D^(j)。

    与旧的「按位置均摊」策略（k = max(1, ⌊n/|M|⌋)，每个可扰动位置逐列取
    [d·k, (d+1)·k) 的候选）不同：本函数将安全词表 V_safe 与可扰动位置集合 M
    张成的全局分数矩阵 ``S ∈ R^{|V_safe|×|M|}`` 视为整体，展平后按全局排名截取
    深度窗口 ``[d·n, (d+1)·n)``，即每轮恰好产生 ``n`` 个候选，**不再**为每个
    位置强制分配预算。候选池规模与代码序列长度 ``|M|`` 解耦，避免长序列
    （如 >8192 token）下候选池随 ``|M|`` 线性膨胀导致前向评估开销暴增。

    实现要点（与 ``abgs_optimizer.select_candidates_global_top_k`` 同源）：
      - 高效选择：对展平后的分数矩阵使用 ``torch.topk`` 取前 ``(d+1)·n`` 个后再
        切片 ``[d·n : (d+1)·n)``（部分选择，约 O(N) + O(k log k)），而非
        O(N log N) 的全量排序。
      - 自身 token 排除：向量化 scatter 将每个位置当前 token 的分数置为 -inf
        （满足 ``v ≠ t_i`` 约束），内存友好（不构造 ``(|V_safe|, |M|)`` 布尔矩阵）。
      - 深度耗尽保护：当 ``d·n`` 已超出全局候选总数时返回空列表。

    Args:
        scores: 分数矩阵 ``(|V_safe|, seq_len)``
        mutable_positions: 可扰动位置索引列表
        safe_ids: 安全 token ID 映射表 ``(|V_safe|,)``
        current_token_ids: 当前对抗样本的 token ID 列表
        depth: 当前搜索深度 d（从 0 开始）
        n: 全局候选集目标大小（深度窗口宽度）

    Returns:
        候选列表 [(position, new_token_id), ...]，长度 ≤ n
    """
    m = len(mutable_positions) # 文本序列中可扰动token位置的数量
    if m == 0:
        return []

    with torch.no_grad():
        device = scores.device
        # 取可扰动位置子矩阵：(|V_safe|, m)（index_select 产生副本，可安全 in-place 掩码）
        pos_tensor = torch.tensor(mutable_positions, dtype=torch.long, device=device)
        sub_scores = scores.index_select(1, pos_tensor)  # (|V_safe|, m)

        # 向量化批量排除各位置的当前 token（将自身分数置为 -inf，满足 v ≠ t_i）
        cur_tids = torch.tensor(
            [current_token_ids[p] for p in mutable_positions],
            dtype=torch.long, device=device,
        )  # (m,)
        vocab_upper = max(int(safe_ids.max().item()), int(cur_tids.max().item())) + 1
        id_to_row = torch.full((vocab_upper,), -1, dtype=torch.long, device=device)
        id_to_row[safe_ids] = torch.arange(safe_ids.numel(), device=device)
        exclude_rows = id_to_row[cur_tids]  # (m,)，-1 表示当前 token 不在安全词表
        valid = (exclude_rows >= 0).nonzero(as_tuple=True)[0]
        if valid.numel() > 0:
            sub_scores[exclude_rows[valid], valid] = float("-inf")

        # 全局展平 + torch.topk 部分选择 + 深度窗口 [d·n : (d+1)·n)
        # 行主序展平：flat_idx = row * m + col
        flat = sub_scores.reshape(-1)  # (|V_safe| * m,)
        total = flat.numel()
        start = depth * n
        if start >= total:
            return []  # 搜索深度已耗尽全局候选，返回空列表
        top_end = min((depth + 1) * n, total)
        top_vals, top_idx = torch.topk(flat, top_end, largest=True, sorted=True)

        # 切片深度窗口 [d·n : top_end]
        win_vals = top_vals[start:top_end]
        win_idx = top_idx[start:top_end]

        # 过滤被 -inf 屏蔽者（仅当窗口触及有效候选边界时才可能命中）
        win_idx = win_idx[torch.isfinite(win_vals)]

        rows = torch.div(win_idx, m, rounding_mode="floor")  # v 在 safe_ids 中的行
        cols = win_idx % m                                   # 位置在 mutable_positions 中的列

        # 一次性 GPU→CPU
        new_tids = safe_ids[rows].tolist()
        col_list = cols.tolist()

    # 组装为 [(position, new_token_id), ...] 格式
    candidates: List[Tuple[int, int]] = [
        (mutable_positions[col_list[i]], new_tids[i]) for i in range(len(new_tids))
    ]

    return candidates # [(token_id_seq_被替换的位置, 被替换的token_id), ...]


# ─────────────────────────────────────────────────────────────────
#  候选评估
# ─────────────────────────────────────────────────────────────────

def evaluate_candidates(
    embedder,
    candidates: List[Tuple[int, int]],
    current_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    pos_embeddings: torch.Tensor,
    loss_func: str = "expected_sim",
    dynamic_weight: bool = False,
    tau_stop: float = 0.80,
    alpha: float = 1.0,
    batch_size: int = 96,
) -> Tuple[Optional[Tuple[int, int]], float]:
    """
    遍历候选集，对每个候选计算替换后的损失，返回最优候选。

    采用批处理加速：将 ``batch_size`` 个候选的 token 序列打包成
    ``(B, seq_len)`` 张量，一次前向传播得到 ``(B, dim)`` 批量嵌入，
    再向量化计算损失 ``(B,)``，大幅减少前向传播次数。

    根据 ``loss_func`` 和 ``dynamic_weight`` 参数选择损失计算方式（4 路分支）：
      - ``expected_sim`` + ``dynamic_weight=False``: 期望相似度损失
      - ``expected_sim`` + ``dynamic_weight=True``:  带动态权重的期望相似度损失

    Args:
        embedder: BaseEmbedder 实例
        candidates: 候选列表 [(position, new_token_id), ...]
        current_ids: 当前对抗样本 token ID ``(1, seq_len)``
        attention_mask: 注意力掩码 ``(1, seq_len)``
        pos_embeddings: 正样本嵌入 ``(|Q+|, dim)``
        loss_func: 损失函数类型
        dynamic_weight: 是否使用动态权重
        tau_stop: 早停阈值（动态权重裕度参考线）
        alpha: 缩放系数
        batch_size: 批处理大小，控制每次前向传播的候选数量

    Returns:
        (best_candidate, best_loss) 元组
    """
    if not candidates:
        return None, float("inf")

    best_candidate = None
    best_loss = float("inf")
    total_candidates = len(candidates)
    processed = 0
    last_reported_percent = 0

    with torch.no_grad():
        for batch_start in range(0, total_candidates, batch_size):
            batch = candidates[batch_start: batch_start + batch_size]
            B = len(batch) # Batch_Size

            # 构造批量 token 序列：(B, seq_len)
            batch_ids = current_ids.expand(B, -1).clone()
            for i, (pos, new_tid) in enumerate(batch):
                batch_ids[i, pos] = new_tid

            # 构造批量 attention mask：(B, seq_len)
            batch_mask = attention_mask.expand(B, -1)

            # 一次前向传播得到 (B, dim) 批量嵌入
            batch_emb = embedder.embed_token_ids(batch_ids, batch_mask)

            # 批量计算损失 (B,)
            if loss_func == "expected_sim" and dynamic_weight:
                losses = compute_weighted_expected_sim_loss_batch(batch_emb, pos_embeddings, tau_stop, alpha)
            elif loss_func == "expected_sim":
                losses = compute_expected_sim_loss_batch(batch_emb, pos_embeddings)

            # 找出当前 batch 内最优候选
            min_idx = losses.argmin().item()
            min_loss = losses[min_idx].item()
            if min_loss < best_loss:
                best_loss = min_loss
                best_candidate = batch[min_idx]

            processed += B

            # 每完成 10% 打印一次进度
            current_percent = int(processed * 100 / total_candidates)
            if total_candidates >= 100 and current_percent >= last_reported_percent + 10:
                last_reported_percent = (current_percent // 10) * 10
                logger.info(
                    f"  候选评估进度: {processed}/{total_candidates} "
                    f"({last_reported_percent}%), "
                    f"当前最优损失: {best_loss:.6f}"
                )

    return best_candidate, best_loss

# ─────────────────────────────────────────────────────────────────
#  主优化循环
# ─────────────────────────────────────────────────────────────────

def aggs_optimize(
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
    AGGS 对抗攻击主循环，对投毒对象的 buggy_code 进行 token 级扰动优化。

    Args:
        embedder: BaseEmbedder 实例（底层模型需设为 eval 模式并保留梯度）
        tokenizer: 模型对应的 tokenizer
        poison_buggy_code: 投毒对象的原始 buggy code 文本
        pos_embeddings: 正样本嵌入 ``(|Q+|, dim)``，已移至目标 device
        safe_ids: 安全 token ID ``(|V_safe|,)``，已移至目标 device
        safe_emb_matrix: 安全嵌入子矩阵 ``(|V_safe|, dim)``，已移至目标 device
        grad_storage: 已注册到词嵌入层的 GradientStorage 实例
        frozen_token_dict: 冻结 token 集合
        config: 攻击配置字典（attack 子节点）

    Returns:
        结果字典，包含:
          - adv_text: 优化后的对抗文本
          - adv_token_ids: 优化后的 token ID 列表
          - final_loss: 最终损失值
          - final_avg_sim: 最终平均相似度
          - iteration_logs: 各轮迭代的日志字符串列表
          - total_iterations: 实际执行的迭代轮数
          - early_stopped: 是否早停
    """
    attack_cfg = config.get("attack", config)
    N = attack_cfg.get("N", 100)
    n = attack_cfg.get("n", 750)
    tau_stop = attack_cfg.get("tau_stop", 0.80) # 早停阈值
    loss_func_type = attack_cfg.get("loss_func", "expected_sim") # 损失函数类型
    dynamic_weight = attack_cfg.get("dynamic_weight", True) # 损失函数是否使用动态权重
    alpha = attack_cfg.get("scaling_factor", 1.0) # 动态权重缩放系数
    batch_size = config.get("model", {}).get("batch_size", 96) # 候选评估批处理大小
    device = pos_embeddings.device

    # ── (1) 初始化对抗样本: a ← Tokenize(C) ──
    logger.info("========== AGGS 优化初始化 ==========")

    prefix_ids = embedder.get_instruct_tids(side="document")
    prefix_len = len(prefix_ids)
    code_ids = tokenizer.encode(poison_buggy_code, add_special_tokens=True)
    full_ids = prefix_ids + code_ids
    adv_ids = torch.tensor([full_ids], dtype=torch.long, device=device)  # (1, prefix_len + code_len)
    attn_mask = torch.ones_like(adv_ids, device=device)

    logger.info(f"(1) 初始化对抗样本: a ← Tokenize(C)，原始代码长度 {len(code_ids)} tokens"
                + (f"，prompt prefix {prefix_len} tokens" if prefix_len else ""))

    # 计算干净代码的基线嵌入和相似度（用于早停判断）
    with torch.no_grad():
        clean_emb = embedder.embed_token_ids(adv_ids, attn_mask)
    clean_sims = compute_per_sample_similarity(clean_emb.squeeze(0), pos_embeddings) # 计算基线嵌入与各正样本的余弦相似度
    clean_avg_sim = clean_sims.mean().item()
    logger.info(f"(2) 计算干净代码基线: Sim(q+, C)，平均相似度到 Q+ = {clean_avg_sim:.4f}")

    # ── (3) 初始化搜索深度 ──
    depth = 0
    logger.info(f"(3) 初始化搜索深度: d ← 0")

    # 算法参数
    weight_label = f"dynamic_weight={dynamic_weight}, alpha={alpha}"
    logger.info(f"算法参数：N={N}（迭代轮数）, n={n}（候选集大小）, "
                f"tau_stop={tau_stop}（早停阈值）, "
                f"loss_func={loss_func_type}, {weight_label}, "
                f"|Q+|={pos_embeddings.shape[0]}")

    # 记录最佳状态
    best_adv_ids = adv_ids.clone()
    best_loss = float("inf")

    iteration_logs: List[str] = []
    early_stopped = False
    init_loss: Optional[float] = None
    init_sim: Optional[List[float]] = None
    init_avg_sim: float = clean_avg_sim  # 干净代码基线相似度即为初始平均相似度

    start_time = datetime.now().isoformat()

    # ── 主迭代循环: for j = 1 to N do ──
    for j in range(1, N + 1):
        logger.info(f"========== 第 {j}/{N} 轮迭代开始 ==========")

        # ── (4) 识别可扰动位置 M ──
        current_token_list = adv_ids[0].tolist()
        mutable_pos = get_mutable_positions(current_token_list, tokenizer, frozen_token_dict, offset=prefix_len)
        if not mutable_pos:
            logger.warning(f"  警告：没有可扰动位置，跳过本轮")
            continue
        logger.info(f"(4) 识别可扰动位置: M ← {{i | t_i ∉ K, i ∈ 1..|a|}}，找到 {len(mutable_pos)} 个可扰动位置")

        # ── (5) 全局候选预算 ──
        m = len(mutable_pos)
        logger.info(f"(5) 全局候选预算: 深度窗口宽度 n={n}, 搜索深度 d={depth}, |M|={m}")

        # ── (6) 计算损失 L(a) ──
        if loss_func_type == "expected_sim" and dynamic_weight:
            loss_before, grad, cur_adv_emb = compute_weighted_expected_sim_loss_and_grad(
                embedder, adv_ids, attn_mask,
                pos_embeddings, grad_storage,
                tau_stop, alpha,
            )
        elif loss_func_type == "expected_sim":
            loss_before, grad, cur_adv_emb = compute_expected_sim_loss_and_grad(
                embedder, adv_ids, attn_mask,
                pos_embeddings, grad_storage,
            )

        logger.info(f"(6) 计算损失: L(a)（{loss_func_type}, weighted={dynamic_weight}）, loss={loss_before:.6f}")

        avg_sim_before = compute_avg_similarity( # 复用 compute_*_loss_and_grad 已算好的嵌入，避免冗余前向传播
            cur_adv_emb.squeeze(0),
            pos_embeddings,
        )
        logger.info(f"    当前平均相似度到 Q+：avg_sim={avg_sim_before:.4f}")

        if j == 1: # 如果是第一轮迭代，则将当前损失作为最佳损失
            best_loss = loss_before
            init_loss = loss_before
            init_sim = compute_per_sample_similarity(cur_adv_emb.squeeze(0), pos_embeddings).tolist()

        # ── (7) 批量计算一阶泰勒近似分数 ──
        scores = compute_candidate_scores(grad, safe_emb_matrix) # shape: (|V_safe|, seq_len)
        logger.info(f"(7) 计算一阶泰勒近似分数: score(v) = e_v^T · (-∇_(e_t) L(a))，scores shape={scores.shape}")

        # ── (8) 构造全局候选集 D^(j) ──
        candidates = select_candidates_global(
            scores, mutable_pos, safe_ids,
            current_token_list, depth, n,
        )
        logger.info(f"(8) 构造全局候选集: D^(j) ← {{a[t_i←v] | (v,i)∈Flatten(S)_ranked[d·n:(d+1)·n)}}，{len(candidates)} 个候选 (depth={depth}, n={n})")

        if not candidates:
            logger.warning(f"  警告：候选集为空（depth={depth}），增加搜索深度至 {depth + 1}")
            depth += 1
            continue

        # ── (9) 评估候选集并选择最优 ──
        logger.info(f"(9) 选择最优候选: a' ← argmin_{{x∈D^(j)}} L(x)，评估 {len(candidates)} 个候选（batch_size={batch_size}）...")
        best_cand, cand_loss = evaluate_candidates(
            embedder, candidates, adv_ids, attn_mask,
            pos_embeddings,
            loss_func=loss_func_type,
            dynamic_weight=dynamic_weight,
            tau_stop=tau_stop,
            alpha=alpha,
            batch_size=batch_size,
        )
        logger.info(f"    最优候选损失：cand_loss={cand_loss:.6f}")

        # ── (10) 判断是否更新并调整搜索深度 ──
        log_entry = RetrievalAdvIterLog(
            iteration=j,
            total_iterations=N,
            loss_before=loss_before,
            depth=depth,
        )

        if best_cand is not None and cand_loss < loss_before: # 如果存在最优候选，并且其损失小于更新前的序列损失，则进行更新
            pos, new_tid = best_cand # best_cand: (position, new_token_id)
            old_tid = adv_ids[0, pos].item()
            old_text = tokenizer.decode([old_tid], skip_special_tokens=False, clean_up_tokenization_spaces=True)
            new_text = tokenizer.decode([new_tid], skip_special_tokens=False, clean_up_tokenization_spaces=True)

            adv_ids[0, pos] = new_tid # 将adv_token_seq中的第pos个位置的old_token_id替换为new_token_id
            depth = 0

            # 计算更新后的指标
            with torch.no_grad():
                new_emb = embedder.embed_token_ids(adv_ids, attn_mask)

            cur_sims = compute_per_sample_similarity( # 更新后对抗序列与各正样本的余弦相似度
                new_emb.squeeze(0), pos_embeddings
            )
            per_cur_sims = cur_sims.tolist() # cur_sims是torch.Tensor，需要转换为list
            avg_sim_after = compute_avg_similarity(new_emb.squeeze(0), pos_embeddings) # 更新后对抗序列与各正样本的平均相似度

            if cand_loss < best_loss: # 如果最优候选的损失小于最佳损失，则更新最佳损失和最佳对抗样本
                best_loss = cand_loss
                best_adv_ids = adv_ids.clone()

            log_entry.updated = True
            # AGGS 单条修改：构造单个元组加入 token_changes 列表
            log_entry.token_changes = [(pos, old_tid, new_tid, old_text, new_text)]
            log_entry.loss_after = cand_loss
            log_entry.avg_sim_before = avg_sim_before
            log_entry.avg_sim_after = avg_sim_after
            log_entry.per_sample_sims = per_cur_sims
            log_entry.depth = depth

            adv_text = tokenizer.decode(adv_ids[0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            log_entry.adv_text_preview = adv_text

            logger.info(f"(10) 更新决策：接受更新 (L(a') < L(a))，重置搜索深度 d←0")
            logger.info(f"    位置 {pos}: \"{old_text}\" → \"{new_text}\"")
            logger.info(f"    损失变化：{loss_before:.6f} → {cand_loss:.6f} (Δ={cand_loss - loss_before:+.6f})")
            logger.info(f"    平均相似度变化：{avg_sim_before:.4f} → {avg_sim_after:.4f} (Δ={avg_sim_after - avg_sim_before:+.4f})")
            logger.info(f"    各正样本相似度：[{', '.join(f'{s:.4f}' for s in per_cur_sims)}]")

            # ── (11) 早停检查（仅在接受更新后执行）
            #   条件: ∀ q+ ∈ Q+, Sim(q+, a) > max(Sim(q+, C), τ_stop)
            thresholds = torch.max(clean_sims, torch.tensor(tau_stop, device=device))
            thresholds_list = thresholds.tolist()

            logger.info(f"(11) 早停检查: if ∀q+∈Q+, Sim(q+,a) > max(Sim(q+,C), τ_stop)")
            logger.info(f"    当前各正样本相似度：[{', '.join(f'{s:.4f}' for s in per_cur_sims)}]")
            logger.info(f"    阈值要求（max(干净相似度, {tau_stop})）："
                        f"[{', '.join(f'{t:.4f}' for t in thresholds_list)}]")

            if (cur_sims > thresholds).all(): # 满足早停条件
                logger.info(f"    >>> 早停条件满足 @ 第 {j} 轮：所有正样本相似度均超过阈值")
                logger.info(f"    最小当前相似度 {cur_sims.min().item():.4f} >= 阈值 {thresholds.min().item():.4f}")
                early_stopped = True
                iteration_logs.append(log_entry.format())
                break
            else: # 不满足早停条件
                failed = (~(cur_sims > thresholds)).nonzero(as_tuple=True)[0].tolist()
                failed_str = ', '.join(
                    f'#{i+1}({per_cur_sims[i]:.4f} <= {thresholds_list[i]:.4f})'
                    for i in failed[:3]
                )
                if len(failed) > 3:
                    failed_str += f' 等 {len(failed)} 个'
                logger.info(f"    早停条件未满足：{failed_str} 未超过阈值")
        else:
            depth += 1

            log_entry.updated = False
            log_entry.token_changes = []  # 无修改
            log_entry.loss_after = loss_before
            log_entry.avg_sim_before = avg_sim_before
            log_entry.avg_sim_after = avg_sim_before
            log_entry.depth = depth

            adv_text = tokenizer.decode(adv_ids[0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
            log_entry.adv_text_preview = adv_text

            if best_cand is not None:
                logger.info(f"(10) 更新决策：拒绝更新 (L(a') >= L(a))，增加搜索深度 d←{depth}")
                logger.info(f"    候选损失 {cand_loss:.6f} >= 原损失 {loss_before:.6f}")
            else:
                logger.info(f"(10) 更新决策：无有效候选，增加搜索深度 d←{depth}")
            logger.info(f"    当前损失：{loss_before:.6f}, 平均相似度：{avg_sim_before:.4f}")

        iteration_logs.append(log_entry.format())

    # ── (12) 返回结果: A ← Tokenizer.decoder(argmin_{x∈{a}} L(x)) ──
    final_ids = best_adv_ids[0].tolist()[prefix_len:]
    final_text = tokenizer.decode(final_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    with torch.no_grad():
        final_emb = embedder.embed_token_ids(best_adv_ids, attn_mask)
    final_avg_sim = compute_avg_similarity(final_emb.squeeze(0), pos_embeddings)
    final_sim = compute_per_sample_similarity(final_emb.squeeze(0), pos_embeddings).tolist()

    end_time = datetime.now().isoformat()
    total_iters = j if early_stopped else N
    logger.info("========== AGGS 优化结束 ==========")
    logger.info(f"(12) 返回最终对抗代码: A ← Tokenizer.decoder(a_best)")
    logger.info(f"    总迭代轮数：{total_iters} / {N}")
    logger.info(f"    早停状态：{'是' if early_stopped else '否'}")
    logger.info(f"    最终损失：{best_loss:.6f}")
    logger.info(f"    最终平均相似度到 Q+：{final_avg_sim:.4f}")
    logger.info(f"    干净代码基线相似度：{clean_avg_sim:.4f}")
    logger.info(f"    相似度提升：{final_avg_sim - clean_avg_sim:+.4f}")

    return {
        "adv_text": final_text, # 优化后的对抗文本
        "adv_token_ids": final_ids, # 优化后的对抗token id列表
        "init_loss": init_loss,
        "init_sim": init_sim,
        "init_avg_sim": init_avg_sim,
        "final_loss": best_loss,
        "final_sim": final_sim,
        "final_avg_sim": final_avg_sim, # 最终平均相似度
        "iteration_logs": iteration_logs, # 各轮迭代的日志字符串列表
        "total_iterations": total_iters, # 实际执行的迭代轮数
        "early_stopped": early_stopped,
        "start_time": start_time,
        "end_time": end_time,
        "algorithm": "aggs",
    }
