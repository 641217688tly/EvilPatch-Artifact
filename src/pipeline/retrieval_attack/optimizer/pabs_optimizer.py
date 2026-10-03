"""
Position-Aware Beam Search (PABS) 位置感知束搜索优化器。

PABS 是 AGGD/ABGS 的基线方法。与 ABGS 在原始代码中**替换** token 不同，
PABS 在投毒目标代码中**插入**一段长度为 L 的独立对抗扰动序列（Adversarial
Perturbation Sequence），并联合搜索最优的插入位置和扰动序列内容。

与 ABGS 的核心差异
------------------
- ABGS: 直接替换原始代码中的 token
- PABS: 在代码的行尾候选位置插入一段额外的对抗序列

算法概述
--------
输入: 干净代码 C, 正样本集 Q+, 迭代轮数 N, 候选集大小 n,
      束宽 B, 扰动序列长度 L
输出: 优化后的对抗代码 A

1. a_code ← prefix + Tokenize(C): 构建完整代码 token 序列
2. a_seq  ← [0]*L: 初始化长度为 L 的扰动序列（全 PAD）
3. P ← 代码区域中每行行尾的 token 位置集合
4. for each p ∈ P:
   - full = Insert(a_code, a_seq, p)
   - 计算损失
5. B ← Top-B(P): 选取初始损失最小的 B 个位置作为初始束
6. for j = 1 to N:
   a) C_pool ← ∅
   b) for each (a_seq, p) ∈ B:
      - full = Insert(a_code, a_seq, p)
      - 计算损失和梯度，提取扰动序列位置 [p, p+L) 处的梯度
      - 对扰动序列中每个位置: 一阶泰勒近似选取 top-k 候选
      - C_pool ← C_pool ∪ 新候选
   c) 评估 C_pool 中所有候选的损失
   d) B ← Top-B(C_pool)
   e) 早停检查
7. return Insert(a_code, a_seq*, p*) 的解码结果
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.pipeline.retrieval_attack.loss_func import (
    GradientStorage,
    compute_avg_similarity,
    compute_expected_sim_loss,
    compute_expected_sim_loss_and_grad,
    compute_per_sample_similarity,
    compute_retrieval_attack_loss_batch,
    compute_weighted_expected_sim_loss,
    compute_weighted_expected_sim_loss_and_grad,
)
from src.utils.dataclass import RetrievalAdvIterLog

logger = logging.getLogger(__name__)


def _compute_candidate_scores(
    grad: torch.Tensor,
    safe_emb_matrix: torch.Tensor,
) -> torch.Tensor:
    """PABS 自有的位置内候选排序分数，不依赖 ABGS 实现。"""
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
    """PABS 专用的候选批量评估，保持原返回结构。"""
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
#  PABS 辅助函数
# ─────────────────────────────────────────────────────────────────

def find_line_end_positions(
    code_ids: List[int],
    tokenizer,
    prefix_len: int,
) -> List[int]:
    """
    扫描代码 token 序列，定位行尾对应的 token 位置。

    遍历 ``code_ids``，对包含换行符 ``\\n`` 的 token 记录其在完整序列
    （prefix + code）中的**后一个**位置，作为候选插入点。

    .. note::
        本项目的数据集已经过代码格式化工具预处理，所有 C/C++ 代码均保证
        包含换行符，因此仅需检测 ``\\n`` 即可覆盖全部行尾位置。

    若代码中未找到任何分隔符，则以代码倒数第二个 token 之后的位置（即 EOS/SEP token
    所在位置）作为唯一候选，确保 EOS/SEP 始终保持在插入后序列的末尾。

    .. important::
        **本函数只接受不含 prefix 的 code_ids，绝对不能传入 full_code_ids。**

        部分嵌入器的 instruction prompt 本身含有 ``\\n``（如 Jina 的
        ``"Candidate code snippet: \\n"``）。若传入包含 prefix 的完整序列，
        这些 ``\\n`` 会被错误地识别为插入候选位置，导致对抗序列被插入到
        instruction prefix 结尾处，破坏模型输入语义。

        通过只扫描 ``code_ids``（不含 prefix），并以 ``prefix_len + i + 1``
        作为绝对偏移，生成的候选位置满足不变量：
        ``all(p >= prefix_len + 1 for p in positions)``
        即所有候选位置均落在代码区域内，而非 prefix 区域 ``[0, prefix_len)``。

    Args:
        code_ids: **不含 prefix** 的代码 token ID 列表（纯代码部分，
                  可含 BOS/EOS 等 special tokens，但不含 instruction prompt tokens）
        tokenizer: HuggingFace tokenizer 实例
        prefix_len: prompt prefix 的 token 长度（用于计算完整序列中的绝对偏移）

    Returns:
        候选插入位置列表（在完整序列中的绝对索引，升序且去重），
        所有位置均满足 ``>= prefix_len + 1``
    """
    positions: List[int] = []
    for i, tid in enumerate(code_ids):
        text = tokenizer.decode([tid], skip_special_tokens=False)
        if "\n" in text:
            # 偏移量 prefix_len + i + 1 保证所有候选位置严格大于 prefix_len，
            # 永远不会落入 instruction prefix 区域 [0, prefix_len)
            positions.append(prefix_len + i + 1)

    # Fallback: 若未找到任何分隔符，插入到代码最后一个 token 之前。
    #
    # 不使用 prefix_len + len(code_ids)（即所有 code_ids 之后），原因：
    # 对于 add_special_tokens=True 的序列，code_ids[-1] 通常是 EOS token。
    # 若在 EOS 之后插入对抗序列，则完整序列变为 [..., EOS, ADV_SEQ]，
    # EOS 不再是最后一个 token，会破坏 Harrier 等 last-token pooling 模型的嵌入计算。
    # 使用 prefix_len + len(code_ids) - 1（即 EOS 在完整序列中的位置）作为插入点，
    # 结果为 [..., code_content, ADV_SEQ, EOS]，EOS 保持在末位。
    if not positions:
        # 至少保证位置满足 >= prefix_len + 1 的不变量
        fallback_pos = prefix_len + max(1, len(code_ids) - 1)
        positions.append(fallback_pos)

    # 去重并保持升序
    seen = set()
    unique: List[int] = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    # 安全性断言：确保调用方没有把 prefix 混入 code_ids
    assert all(p >= prefix_len + 1 for p in unique), (
        f"find_line_end_positions 内部错误：生成了 < prefix_len+1 的候选位置 "
        f"(prefix_len={prefix_len}, positions={unique})。"
        "请确认传入的是不含 prefix 的纯 code_ids。"
    )

    return unique


def insert_adv_sequence(
    code_ids: torch.LongTensor,
    adv_seq: torch.LongTensor,
    position: int,
) -> torch.LongTensor:
    """
    将对抗扰动序列插入到代码 token 序列的指定位置。

    拼接结构: ``code_ids[:, :position]`` + ``adv_seq`` + ``code_ids[:, position:]``

    Args:
        code_ids: 完整代码序列（含 prefix），形状 ``(1, code_len)``
        adv_seq: 扰动序列，形状 ``(1, L)``
        position: 在 ``code_ids`` 中的插入位置索引

    Returns:
        拼接后的完整序列，形状 ``(1, code_len + L)``
    """
    return torch.cat([
        code_ids[:, :position],
        adv_seq,
        code_ids[:, position:],
    ], dim=1)


def _compute_loss_for_full_seq(
    embedder,
    full_ids: torch.LongTensor,
    attn_mask: torch.LongTensor,
    pos_embeddings: torch.Tensor,
    loss_func_type: str,
    dynamic_weight: bool,
    tau_stop: float,
    alpha: float,
) -> Tuple[float, torch.Tensor]:
    """
    对一条完整序列执行无梯度前向传播并计算损失值。

    用于初始化阶段对各候选位置打分，不需要梯度。

    Returns:
        (loss_value, embedding): 损失标量值和句子嵌入 ``(1, dim)``
    """
    with torch.no_grad():
        emb = embedder.embed_token_ids(full_ids, attn_mask)
        a = emb.squeeze(0)
        if loss_func_type == "expected_sim" and dynamic_weight:
            loss = compute_weighted_expected_sim_loss(a, pos_embeddings, tau_stop, alpha)
        elif loss_func_type == "expected_sim":
            loss = compute_expected_sim_loss(a, pos_embeddings)
    return loss.item(), emb.detach()


def _select_adv_candidates_top_k(
    scores: torch.Tensor,
    adv_positions_in_full: List[int],
    safe_ids: torch.LongTensor,
    current_full_token_ids: List[int],
    n: int,
    beam_width: int,
    adv_seq_length: int,
) -> List[Tuple[int, int]]:
    """
    从分数矩阵中为扰动序列的每个位置选取 top-k 候选 token（按位置均摊预算）。

    PABS 沿用「按位置均摊」策略（每个扰动位置分配 k 个候选），与 ABGS 现采用的
    全局 Top-⌊n/|B|⌋ 策略（``select_candidates_global_top_k``）不同；此处操作的位置
    限定在扰动序列的绝对位置（在完整序列中）。

    Args:
        scores: 分数矩阵 ``(|V_safe|, full_seq_len)``
        adv_positions_in_full: 扰动序列在完整序列中的绝对位置列表
        safe_ids: 安全 token ID 映射表 ``(|V_safe|,)``
        current_full_token_ids: 当前完整序列的 token ID 列表
        n: 全局候选集目标大小
        beam_width: 当前束宽
        adv_seq_length: 扰动序列长度 L

    Returns:
        候选列表 [(absolute_position_in_full, new_token_id), ...]
    """
    m = len(adv_positions_in_full)
    if m == 0:
        return []

    k = max(1, n // (beam_width * m))

    with torch.no_grad():
        pos_tensor = torch.tensor(
            adv_positions_in_full, dtype=torch.long, device=scores.device
        )
        sub_scores = scores[:, pos_tensor].clone()  # (|V_safe|, m)

        for local_i, pos in enumerate(adv_positions_in_full):
            current_tid = current_full_token_ids[pos]
            match = (safe_ids == current_tid).nonzero(as_tuple=True)[0]
            if match.numel() > 0:
                sub_scores[match[0], local_i] = float("-inf")

        actual_k = min(k, sub_scores.shape[0])
        _, top_indices = torch.topk(sub_scores, actual_k, dim=0, largest=True, sorted=True)
        selected_tids = safe_ids[top_indices].tolist()

    candidates: List[Tuple[int, int]] = [
        (adv_positions_in_full[local_i], selected_tids[ki][local_i])
        for local_i in range(m)
        for ki in range(len(selected_tids))
    ]
    return candidates


# ─────────────────────────────────────────────────────────────────
#  PABS 主优化循环
# ─────────────────────────────────────────────────────────────────

def pabs_optimize(
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
    PABS 位置感知束搜索对抗攻击主循环。

    函数签名与 ``aggd_optimize`` / ``abgs_optimize`` 保持一致，
    便于 ``run_attack.py`` 动态切换。

    Args:
        embedder: BaseEmbedder 实例（底层模型需设为 eval 模式并保留梯度）
        tokenizer: 模型对应的 tokenizer
        poison_buggy_code: 投毒对象的原始 buggy code 文本
        pos_embeddings: 正样本嵌入 ``(|Q+|, dim)``
        safe_ids: 安全 token ID ``(|V_safe|,)``
        safe_emb_matrix: 安全嵌入子矩阵 ``(|V_safe|, dim)``
        grad_storage: 已注册到词嵌入层的 GradientStorage 实例
        frozen_token_dict: 冻结 token 集合（PABS 中通常为空字典，
                           因为 ``use_safe_vocab=false``）
        config: 完整配置字典

    Returns:
        结果字典（与 aggd_optimize / abgs_optimize 返回格式一致）
    """
    attack_cfg = config.get("attack", config)
    N = attack_cfg.get("N", 100)
    n = attack_cfg.get("n", 750)
    tau_stop = attack_cfg.get("tau_stop", 0.80)
    loss_func_type = attack_cfg.get("loss_func", "expected_sim")
    beam_width = attack_cfg.get("beam_width", 10)
    dynamic_weight = attack_cfg.get("dynamic_weight", False)
    alpha = attack_cfg.get("scaling_factor", 1.0)
    adv_seq_length = attack_cfg.get("adv_seq_length", 20)
    patience = int(attack_cfg.get("patience", 5))
    batch_size = config.get("model", {}).get("batch_size", 96)
    device = pos_embeddings.device

    if patience < 0:
        raise ValueError(f"patience 必须大于等于0，当前值为 {patience}")

    # ── (1) 初始化代码 token 序列: a_code ← prefix + Tokenize(C) ──
    logger.info("========== PABS 位置感知束搜索优化初始化 ==========")

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

    # ── (2) 初始化扰动序列: a_seq ← [0]*L ──
    adv_seq = torch.zeros(
        1, adv_seq_length, dtype=torch.long, device=device
    )  # (1, L)
    logger.info(
        f"(2) 初始化扰动序列: a_seq ← [0]*{adv_seq_length}（全 PAD）"
    )

    # ── (3) 计算候选插入位置集合: P ← 代码行尾位置 ──
    position_candidates = find_line_end_positions(
        code_ids, tokenizer, prefix_len
    )
    logger.info(
        f"(3) 候选插入位置集合: |P|={len(position_candidates)} 个行尾位置"
    )

    # 构建完整序列的注意力掩码（长度 = 代码 + 扰动序列，对所有位置恒定）
    total_len = len(full_code_ids_list) + adv_seq_length
    attn_mask = torch.ones(1, total_len, dtype=torch.long, device=device)

    # ── (4) 计算干净代码基线 ──
    code_attn_mask = torch.ones_like(code_ids_tensor, device=device)
    with torch.no_grad():
        clean_emb = embedder.embed_token_ids(code_ids_tensor, code_attn_mask)
    clean_sims = compute_per_sample_similarity(
        clean_emb.squeeze(0), pos_embeddings
    )
    clean_avg_sim = clean_sims.mean().item()
    logger.info(
        f"(4) 计算干净代码基线: Sim(q+, C)，平均相似度到 Q+ = {clean_avg_sim:.4f}"
    )

    # ── (5) 初始化束: 对每个候选位置计算初始损失，选取 Top-B ──
    logger.info(
        f"(5) 评估初始候选位置: 对 {len(position_candidates)} 个位置计算损失"
    )
    init_candidates: List[Tuple[torch.LongTensor, int, float, torch.Tensor]] = []
    for p in position_candidates:
        full_ids = insert_adv_sequence(code_ids_tensor, adv_seq, p)
        loss_val, emb = _compute_loss_for_full_seq(
            embedder, full_ids, attn_mask,
            pos_embeddings, loss_func_type, dynamic_weight, tau_stop, alpha,
        )
        init_candidates.append((adv_seq.clone(), p, loss_val, emb))

    init_candidates.sort(key=lambda x: x[2]) # 按损失值升序

    # beam 结构: List of (adv_seq_tensor, insert_position, loss, cached_emb)
    # 首轮迭代保留全部位置候选，让所有位置都参与梯度计算和候选生成；
    # 首轮结束后 Top-B 筛选（步骤 10）自然将 beam 缩减至 beam_width。
    beam: List[Tuple[torch.LongTensor, int, float, torch.Tensor]] = init_candidates
    logger.info(
        f"(6) 初始化束集合: B ← 全部候选位置，|B|={len(beam)}"
        f"（首轮迭代后缩减至 Top-{beam_width}），"
        f"最优损失={beam[0][2]:.6f}，最优位置={beam[0][1]}"
    )

    weight_label = f"dynamic_weight={dynamic_weight}, alpha={alpha}"
    logger.info(
        f"算法参数：N={N}（迭代轮数）, n={n}（候选集大小）, "
        f"B={beam_width}（束宽）, L={adv_seq_length}（扰动序列长度）, "
        f"tau_stop={tau_stop}（早停阈值）, "
        f"patience={patience}（Patience 早停轮数）, "
        f"loss_func={loss_func_type}, {weight_label}, "
        f"|Q+|={pos_embeddings.shape[0]}"
        f"|P|={len(position_candidates)}（候选位置数）"
    )

    best_adv_seq = beam[0][0].clone()
    best_insert_pos = beam[0][1]
    best_loss = beam[0][2]
    best_emb = beam[0][3]

    prev_best_full_ids: Optional[torch.LongTensor] = None

    iteration_logs: List[str] = []
    early_stopped = False
    patience_counter = 0
    init_loss: Optional[float] = None
    init_sim: Optional[List[float]] = None
    init_avg_sim: float = clean_avg_sim

    start_time = datetime.now().isoformat()

    # ── 主迭代循环: for j = 1 to N do ──
    for j in range(1, N + 1):
        logger.info(
            f"========== 第 {j}/{N} 轮迭代开始（|B|={len(beam)}）=========="
        )

        # ── (7) 初始化全局候选池: C_pool ← ∅ ──
        # candidate_pool: evaluate_candidates_pool 需要的格式
        # insert_pos_tracker: 与 candidate_pool 一一对应的插入位置追踪列表
        candidate_pool: List[Tuple[torch.LongTensor, int, int]] = []
        insert_pos_tracker: List[int] = []
        logger.info("(7) 初始化本轮迭代的全局候选池: C_pool ← ∅")

        # ── (8) 遍历束中每个方案: for each (a_seq, p) ∈ B do ──
        logger.info(
            f"(8) 遍历束中每个方案: for each (a_seq, p) ∈ B "
            f"(|B|={len(beam)})"
        )
        for beam_idx, (b_adv_seq, b_pos, b_loss, _b_emb) in enumerate(beam):
            logger.info(
                f"  束方案 {beam_idx + 1}/{len(beam)} 处理中... "
                f"(pos={b_pos}, loss={b_loss:.6f})"
            )

            # (8.1) 构建完整序列: full = Insert(a_code, a_seq, p)
            full_ids = insert_adv_sequence(code_ids_tensor, b_adv_seq, b_pos)

            # (8.2) 扰动序列在完整序列中的绝对位置: [p, p+1, ..., p+L-1]
            adv_positions_in_full = list(range(b_pos, b_pos + adv_seq_length))

            # (8.3) 计算损失和梯度
            if loss_func_type == "expected_sim" and dynamic_weight:
                loss_before, grad, _ = compute_weighted_expected_sim_loss_and_grad(
                    embedder, full_ids, attn_mask,
                    pos_embeddings, grad_storage,
                    tau_stop, alpha,
                )
            elif loss_func_type == "expected_sim":
                loss_before, grad, _ = compute_expected_sim_loss_and_grad(
                    embedder, full_ids, attn_mask,
                    pos_embeddings, grad_storage,
                )
            
            logger.info(
                f"    (8.3) 计算损失: L(a_p)（{loss_func_type}, "
                f"weighted={dynamic_weight}）, loss={loss_before:.6f}"
            )

            # (8.4) 计算一阶泰勒近似分数（在完整序列上）
            scores = _compute_candidate_scores(grad, safe_emb_matrix)
            logger.info(
                "    (8.4) 计算泰勒分数: score(v) = e_v^T · (-∇ L(a_p))"
            )

            # (8.5) 为扰动序列的每个位置选取 top-k 候选
            full_token_list = full_ids[0].tolist()
            cands = _select_adv_candidates_top_k(
                scores, adv_positions_in_full, safe_ids,
                full_token_list, n,
                beam_width=len(beam),
                adv_seq_length=adv_seq_length,
            )
            for abs_pos, new_tid in cands:
                candidate_pool.append((full_ids, abs_pos, new_tid)) # (1, seq_len), position, new_tid
                insert_pos_tracker.append(b_pos)
            logger.info(
                f"    (8.5) 生成候选: C_pool ← C_pool ∪ {{...}}，"
                f"新增 {len(cands)} 个，累计 {len(candidate_pool)} 个"
            )

        if not candidate_pool:
            logger.warning(
                f"  第 {j} 轮：候选池为空，提前结束优化"
            )
            break

        # ── (9) 评估候选池 ──
        logger.info(
            f"(9) 评估候选池: 共 {len(candidate_pool)} 个候选"
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

        # ── (10) Top-B 筛选更新束 ──
        # 将 evaluated 与 insert_pos_tracker 配对后按损失排序
        indexed_evaluated = list(zip(evaluated, insert_pos_tracker))
        indexed_evaluated.sort(key=lambda x: x[0][1])  # 按损失值升序

        new_beam: List[Tuple[torch.LongTensor, int, float, torch.Tensor]] = []
        seen_keys: set = set()
        for (cand_full_ids, cand_loss, cand_emb), cand_insert_pos in indexed_evaluated:
            if len(new_beam) >= beam_width:
                break

            key = hash(tuple(cand_full_ids[0].tolist()))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            extracted_adv = cand_full_ids[
                :, cand_insert_pos:cand_insert_pos + adv_seq_length
            ]
            new_beam.append((
                extracted_adv.clone(), cand_insert_pos, cand_loss, cand_emb,
            ))

        if not new_beam:
            best_entry = indexed_evaluated[0]
            bp = best_entry[1]
            new_beam.append((
                best_entry[0][0][:, bp:bp + adv_seq_length].clone(),
                bp,
                best_entry[0][1],
                best_entry[0][2],
            ))

        beam = new_beam

        top_loss = beam[0][2] # 排序后 evaluated[0] 为候选池中损失最小者，取其(a_{seq}, position, loss, emb)中的loss
        logger.info(
            f"(10) 更新束集合: B ← Top-B，|B|={len(beam)}，"
            f"最优损失={top_loss:.6f}，最优位置={beam[0][1]}"
        )

        if j == 1:
            init_loss = top_loss
            init_sim = compute_per_sample_similarity(
                beam[0][3].squeeze(0), pos_embeddings
            ).tolist()

        loss_before_update = best_loss

        if top_loss < best_loss:
            best_loss = top_loss
            best_adv_seq = beam[0][0].clone()
            best_insert_pos = beam[0][1]
            best_emb = beam[0][3]
            patience_counter = 0
            logger.info(f"    全局最优更新：best_loss={best_loss:.6f}，patience 重置")
        else:
            patience_counter += 1
            logger.info(
                f"    全局最优未更新，patience_counter={patience_counter}/{patience}"
            )

        # 记录迭代日志
        best_beam_emb = beam[0][3]
        avg_sim_after = compute_avg_similarity(
            best_beam_emb.squeeze(0), pos_embeddings
        )
        per_sims = compute_per_sample_similarity(
            best_beam_emb.squeeze(0), pos_embeddings
        ).tolist()

        best_beam_full_ids = insert_adv_sequence(
            code_ids_tensor, beam[0][0], beam[0][1]
        )
        baseline_full_ids = (
            prev_best_full_ids
            if prev_best_full_ids is not None
            else insert_adv_sequence(
                code_ids_tensor,
                torch.zeros(1, adv_seq_length, dtype=torch.long, device=device),
                beam[0][1],
            )
        )

        baseline_list = baseline_full_ids[0].tolist()
        curr_list = best_beam_full_ids[0].tolist()
        min_len = min(len(baseline_list), len(curr_list))
        changed_positions = [
            i for i in range(min_len) if baseline_list[i] != curr_list[i]
        ]

        if changed_positions:
            for chg_pos in changed_positions[:5]:
                old_tid = baseline_list[chg_pos] if chg_pos < len(baseline_list) else -1
                new_tid = curr_list[chg_pos] if chg_pos < len(curr_list) else -1
                old_text = tokenizer.decode([old_tid], skip_special_tokens=False, clean_up_tokenization_spaces=True) if old_tid >= 0 else "<N/A>"
                new_text = tokenizer.decode([new_tid], skip_special_tokens=False, clean_up_tokenization_spaces=True) if new_tid >= 0 else "<N/A>"
                logger.info(f"    位置 {chg_pos}: \"{old_text}\" → \"{new_text}\"")
            if len(changed_positions) > 5:
                logger.info(
                    f"    ...（共 {len(changed_positions)} 个位置发生变化）"
                )
        else:
            logger.info("    本轮束最优方案与基线相同，无 token 变化")

        logger.info(
            f"    损失变化：{loss_before_update:.6f} → {top_loss:.6f} "
            f"(Δ={top_loss - loss_before_update:+.6f})"
        )
        logger.info(
            f"    平均相似度变化：{clean_avg_sim:.4f} → {avg_sim_after:.4f} "
            f"(Δ={avg_sim_after - clean_avg_sim:+.4f})"
        )
        logger.info(
            f"    各正样本相似度：[{', '.join(f'{s:.4f}' for s in per_sims)}]"
        )

        prev_best_full_ids = best_beam_full_ids.clone()

        log_token_changes = []
        if changed_positions:
            for chg_pos in changed_positions:
                old_tid = baseline_list[chg_pos] if chg_pos < len(baseline_list) else -1
                new_tid = curr_list[chg_pos] if chg_pos < len(curr_list) else -1
                old_text = tokenizer.decode([old_tid], skip_special_tokens=False, clean_up_tokenization_spaces=True) if old_tid >= 0 else "<N/A>"
                new_text = tokenizer.decode([new_tid], skip_special_tokens=False, clean_up_tokenization_spaces=True) if new_tid >= 0 else "<N/A>"
                log_token_changes.append(
                    (chg_pos, old_tid, new_tid, old_text, new_text)
                )
            log_updated = True
        else:
            log_updated = False

        log_entry = RetrievalAdvIterLog(
            iteration=j,
            total_iterations=N,
            loss_before=loss_before_update,
            loss_after=top_loss,
            avg_sim_before=clean_avg_sim,
            avg_sim_after=avg_sim_after,
            per_sample_sims=per_sims,
            depth=len(beam),
            updated=log_updated,
            token_changes=log_token_changes,
            adv_text_preview=tokenizer.decode(
                best_beam_full_ids[0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True
            )[:200],
        )
        iteration_logs.append(log_entry.format())

        # ── (11) 早停检查 ──
        logger.info(
            f"(11) 早停检查: 检查 {len(beam)} 个方案"
        )
        for beam_idx, (b_adv_seq, b_pos, b_loss, b_emb) in enumerate(beam):
            b_sims = compute_per_sample_similarity(
                b_emb.squeeze(0), pos_embeddings
            )
            thresholds = torch.max(
                clean_sims, torch.tensor(tau_stop, device=device)
            )
            if (b_sims > thresholds).all():
                logger.info(
                    f"    >>> 早停条件满足 @ 第 {j} 轮，束方案 {beam_idx + 1}："
                    f"所有正样本相似度均超过阈值 "
                    f"(min_sim={b_sims.min().item():.4f})"
                )
                best_adv_seq = b_adv_seq.clone()
                best_insert_pos = b_pos
                best_loss = b_loss
                best_emb = b_emb
                early_stopped = True
                break
        if early_stopped:
            break

        if patience > 0 and patience_counter >= patience:
            logger.info(
                f"    >>> Patience 早停 @ 第 {j} 轮："
                f"连续 {patience} 轮全局最优未改善"
            )
            early_stopped = True
            break

    total_iters = j if early_stopped else N

    # ── (12) 返回结果 ──
    best_full_ids = insert_adv_sequence(
        code_ids_tensor, best_adv_seq, best_insert_pos
    )
    final_ids_list = best_full_ids[0].tolist()[prefix_len:]
    final_text = tokenizer.decode(final_ids_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    with torch.no_grad():
        final_emb = embedder.embed_token_ids(best_full_ids, attn_mask)
    final_avg_sim = compute_avg_similarity(
        final_emb.squeeze(0), pos_embeddings
    )
    final_sim = compute_per_sample_similarity(
        final_emb.squeeze(0), pos_embeddings
    ).tolist()

    end_time = datetime.now().isoformat()
    logger.info("========== PABS 位置感知束搜索优化结束 ==========")
    logger.info(
        "(12) 返回最终对抗代码: "
        "A ← decode(Insert(a_code, a_seq*, p*))"
    )
    logger.info(f"    总迭代轮数：{total_iters} / {N}")
    logger.info(f"    早停状态：{'是' if early_stopped else '否'}")
    logger.info(f"    最终损失：{best_loss:.6f}")
    logger.info(f"    最终平均相似度到 Q+：{final_avg_sim:.4f}")
    logger.info(f"    干净代码基线相似度：{clean_avg_sim:.4f}")
    logger.info(f"    相似度提升：{final_avg_sim - clean_avg_sim:+.4f}")
    logger.info(f"    最优插入位置：{best_insert_pos}")
    logger.info(
        f"    扰动序列 token IDs：{best_adv_seq[0].tolist()}"
    )

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
        "start_time": start_time,
        "end_time": end_time,
        "algorithm": "pabs",
        "insert_position": best_insert_pos,
        "adv_seq_token_ids": best_adv_seq[0].tolist(),
    }
