"""Approximate Beam Gradient Search (ABGS).

本实现将旧版 ABGS 的逐位置候选覆盖与全局候选探索结合，并加入：

* 跨位置可比较的替换增量分数；
* 基于初始可扰动位置数的动态 Beam；
* 部分父节点精英保留；
* 每个 Beam 独立的局部 depth 与全局扫描指针；
* 父节点级候选缓存，以及“缓存重放 + 新候选补位”；
* Loss 与平均相似度联合 patience。

``n`` 表示每轮最多真实评估的全新候选数。缓存重放可以增加候选池
竞争者数量，但不会增加真实前向传播预算。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.pipeline.retrieval_attack.loss_func import (
    GradientStorage,
    compute_retrieval_attack_loss,
    compute_retrieval_attack_loss_and_grad,
    compute_retrieval_attack_loss_batch,
    normalize_loss_func_name,
)
from src.pipeline.retrieval_attack.vocab_filter import get_word_embeddings
from src.utils.token import get_mutable_positions

logger = logging.getLogger(__name__)

CandidateKey = Tuple[int, int]
SequenceKey = Tuple[int, ...]


@dataclass
class CachedCandidate:
    """相对于一个未变化父节点已经完成真实评估的单步替换。"""

    position: int
    new_token_id: int
    loss: float
    avg_similarity: float
    similarities: Tuple[float, ...]
    embedding_cpu: torch.Tensor
    first_source: str
    evaluated_iteration: int
    admitted_once: bool = False


@dataclass
class BeamState:
    """一个可继续扩展的 ABGS Beam。"""

    token_ids: torch.LongTensor
    loss: float
    embedding: torch.Tensor
    similarities: torch.Tensor
    avg_similarity: float
    local_depth: int = 0
    global_offset: int = 0
    cached_gradient: Optional[torch.Tensor] = None
    mutable_positions: Optional[List[int]] = None
    candidate_cache: Dict[CandidateKey, CachedCandidate] = field(default_factory=dict)
    local_rank_limit: Optional[int] = None
    global_rank_limit: Optional[int] = None


@dataclass
class CandidateProposal:
    """尚未完成前向评估的候选及其全部父节点来源。"""

    token_ids: torch.LongTensor
    aliases: List[Tuple[BeamState, CandidateKey, Dict[str, Any]]]


@dataclass
class PoolEntry:
    """参与本轮 Top-B 竞争的父节点或子节点。"""

    token_ids: torch.LongTensor
    loss: float
    avg_similarity: float
    similarities: Tuple[float, ...]
    embedding_cpu: torch.Tensor
    is_parent: bool
    source: str
    metadata: Dict[str, Any]
    parent_state: Optional[BeamState] = None
    cache_records: List[CachedCandidate] = field(default_factory=list)


def _sequence_key(token_ids: torch.LongTensor) -> SequenceKey:
    return tuple(int(value) for value in token_ids[0].tolist())


def _compute_dynamic_beam_budget(
    n: int,
    mutable_count: int,
    global_basic_budget: int,
) -> Tuple[int, int, int]:
    """返回 ``(B, K, initial_global_budget)``。"""
    if n <= 0:
        raise ValueError(f"n 必须大于0，当前值为 {n}")
    if mutable_count <= 0:
        raise ValueError(f"mutable_count 必须大于0，当前值为 {mutable_count}")
    if mutable_count > n:
        raise ValueError(
            "可扰动位置数不能超过新候选预算："
            f"mutable_count={mutable_count}, n={n}"
        )
    if global_basic_budget < 0:
        raise ValueError(
            "global_basic_budget 必须大于等于0，"
            f"当前值为 {global_basic_budget}"
        )

    beam_width = max(1, n // (mutable_count + global_basic_budget))
    per_beam_budget = n // beam_width
    initial_global_budget = max(0, per_beam_budget - mutable_count)
    return beam_width, per_beam_budget, initial_global_budget


def _compute_replacement_scores(
    grad: torch.Tensor,
    safe_emb_matrix: torch.Tensor,
    current_token_embeddings: torch.Tensor,
) -> torch.Tensor:
    """计算 ``(e_v-e_ti)^T(-grad_i)``。"""
    if grad.shape != current_token_embeddings.shape:
        raise ValueError(
            "grad 与 current_token_embeddings 的形状必须一致，"
            f"当前分别为 {tuple(grad.shape)} 和 "
            f"{tuple(current_token_embeddings.shape)}"
        )
    with torch.no_grad():
        negative_grad = -grad
        candidate_projection = torch.matmul(safe_emb_matrix, negative_grad.T)
        current_projection = (
            current_token_embeddings.to(
                device=negative_grad.device,
                dtype=negative_grad.dtype,
            )
            * negative_grad
        ).sum(dim=-1)
        return candidate_projection - current_projection.unsqueeze(0)


def _mask_identity_replacements(
    scores: torch.Tensor,
    safe_ids: torch.LongTensor,
    mutable_token_ids: torch.LongTensor,
) -> None:
    """原地将 ``v == t_i`` 的分数设为负无穷。"""
    if safe_ids.numel() == 0 or mutable_token_ids.numel() == 0:
        return
    upper = max(int(safe_ids.max().item()), int(mutable_token_ids.max().item())) + 1
    id_to_row = torch.full(
        (upper,), -1, dtype=torch.long, device=scores.device
    )
    id_to_row[safe_ids] = torch.arange(safe_ids.numel(), device=scores.device)
    rows = id_to_row[mutable_token_ids]
    valid_columns = (rows >= 0).nonzero(as_tuple=True)[0]
    if valid_columns.numel() > 0:
        scores[rows[valid_columns], valid_columns] = float("-inf")


def _select_local_depth_candidates(
    scores: torch.Tensor,
    mutable_positions: Sequence[int],
    safe_ids: torch.LongTensor,
    depth: int,
) -> List[Tuple[int, int, float]]:
    """为每个位置选择其局部排名 ``depth`` 处的候选。"""
    if depth < 0:
        raise ValueError(f"local depth 必须大于等于0，当前值为 {depth}")
    if not mutable_positions or safe_ids.numel() == 0 or depth >= safe_ids.numel():
        return []
    with torch.no_grad():
        top_values, top_rows = torch.topk(
            scores,
            k=depth + 1,
            dim=0,
            largest=True,
            sorted=True,
        )
        selected_values = top_values[depth]
        selected_rows = top_rows[depth]
        finite = torch.isfinite(selected_values)
        columns = finite.nonzero(as_tuple=True)[0]
        if columns.numel() == 0:
            return []
        positions = [int(mutable_positions[index]) for index in columns.tolist()]
        token_ids = safe_ids[selected_rows[columns]].tolist()
        values = selected_values[columns].tolist()
    return [
        (positions[index], int(token_ids[index]), float(values[index]))
        for index in range(len(positions))
    ]


def _materialize_replacement(
    parent_ids: torch.LongTensor,
    position: int,
    new_token_id: int,
) -> torch.LongTensor:
    child_ids = parent_ids.clone()
    child_ids[0, position] = int(new_token_id)
    return child_ids


def _evaluate_candidate_sequences(
    embedder,
    candidate_ids: Sequence[torch.LongTensor],
    attention_mask: torch.LongTensor,
    pos_embeddings: torch.Tensor,
    *,
    loss_func: str,
    dynamic_weight: bool,
    tau_stop: float,
    alpha: float,
    softmin_temperature: float,
    hybrid_softmin_weight: float,
    batch_size: int,
) -> List[Tuple[float, torch.Tensor, torch.Tensor, float]]:
    """批量评估 ABGS 全新候选。"""
    results: List[Tuple[float, torch.Tensor, torch.Tensor, float]] = []
    if not candidate_ids:
        return results

    pos_norm = F.normalize(pos_embeddings, dim=-1)
    with torch.no_grad():
        for start in range(0, len(candidate_ids), batch_size):
            batch_list = candidate_ids[start : start + batch_size]
            batch_ids = torch.cat(batch_list, dim=0)
            batch_mask = attention_mask.expand(batch_ids.shape[0], -1)
            batch_emb = embedder.embed_token_ids(batch_ids, batch_mask)
            losses = compute_retrieval_attack_loss_batch(
                batch_emb,
                pos_embeddings,
                loss_func=loss_func,
                dynamic_weight=dynamic_weight,
                tau_stop=tau_stop,
                alpha=alpha,
                softmin_temperature=softmin_temperature,
                hybrid_softmin_weight=hybrid_softmin_weight,
            )
            similarities = F.normalize(batch_emb, dim=-1) @ pos_norm.T
            averages = similarities.mean(dim=-1)
            for index in range(batch_ids.shape[0]):
                results.append((
                    float(losses[index].item()),
                    batch_emb[index : index + 1].detach(),
                    similarities[index].detach(),
                    float(averages[index].item()),
                ))
    return results


def _has_future_search_space(state: BeamState) -> bool:
    if state.cached_gradient is None:
        return True
    local_future = (
        state.local_rank_limit is not None
        and state.local_depth < state.local_rank_limit
    )
    global_future = (
        state.global_rank_limit is not None
        and state.global_offset < state.global_rank_limit
    )
    return bool(local_future or global_future)


def _update_patience_counter(
    counter: int,
    iteration_best_loss: float,
    iteration_best_avg_similarity: float,
    historical_best_loss: float,
    historical_best_avg_similarity: float,
    loss_delta: float,
    similarity_delta: float,
) -> Tuple[int, bool, bool]:
    """Loss 或 AvgSim 任一显著改善就重置 patience。"""
    loss_improved = iteration_best_loss < historical_best_loss - loss_delta
    similarity_improved = (
        iteration_best_avg_similarity
        > historical_best_avg_similarity + similarity_delta
    )
    next_counter = 0 if loss_improved or similarity_improved else counter + 1
    return next_counter, loss_improved, similarity_improved


def abgs_optimize(
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
    """执行局部—全局混合的 Approximate Beam Gradient Search。"""
    attack_cfg = config.get("attack", config)
    N = int(attack_cfg.get("N", 50))
    n = int(attack_cfg.get("n", 6000))
    tau_stop = float(attack_cfg.get("tau_stop", 0.95))
    elite_num = int(attack_cfg.get("elite_num", 1))
    global_basic_budget = int(attack_cfg.get("global_basic_budget", 50))
    patience = int(attack_cfg.get("patience", 10))
    patience_loss_delta = float(attack_cfg.get("patience_loss_delta", 1e-4))
    patience_similarity_delta = float(
        attack_cfg.get("patience_similarity_delta", 1e-4)
    )
    loss_func_type = normalize_loss_func_name(
        attack_cfg.get("loss_func", "expected_sim")
    )
    requested_dynamic_weight = bool(attack_cfg.get("dynamic_weight", True))
    dynamic_weight = requested_dynamic_weight and loss_func_type == "expected_sim"
    alpha = float(attack_cfg.get("scaling_factor", 1.0))
    softmin_temperature = float(attack_cfg.get("softmin_temperature", 0.1))
    hybrid_softmin_weight = float(
        attack_cfg.get("hybrid_softmin_weight", 0.5)
    )
    batch_size = int(config.get("model", {}).get("batch_size", 64))
    log_every_n_iter = max(
        1, int(config.get("logging", {}).get("log_every_n_iter", 1))
    )

    if N <= 0:
        raise ValueError(f"N 必须大于0，当前值为 {N}")
    if elite_num < 0:
        raise ValueError(f"elite_num 必须大于等于0，当前值为 {elite_num}")
    if patience < 0:
        raise ValueError(f"patience 必须大于等于0，当前值为 {patience}")
    if patience_loss_delta < 0 or patience_similarity_delta < 0:
        raise ValueError("patience 的改善阈值必须大于等于0")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于0，当前值为 {batch_size}")
    if "beam_width" in attack_cfg:
        logger.warning(
            "ABGS 现使用完全动态 Beam；配置 beam_width=%s 将被忽略",
            attack_cfg.get("beam_width"),
        )
    if requested_dynamic_weight and not dynamic_weight:
        logger.warning(
            "loss_func=%s 不使用 dynamic_weight；本次运行按 false 处理",
            loss_func_type,
        )

    device = pos_embeddings.device
    prefix_ids = embedder.get_instruct_tids(side="document")
    prefix_len = len(prefix_ids)
    code_ids = tokenizer.encode(poison_buggy_code, add_special_tokens=True)
    full_ids = prefix_ids + code_ids
    init_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(init_ids, device=device)

    initial_mutable_positions = get_mutable_positions(
        init_ids[0].tolist(),
        tokenizer,
        frozen_token_dict,
        offset=prefix_len,
    )
    m0 = len(initial_mutable_positions)
    beam_width, per_beam_budget, initial_global_budget = (
        _compute_dynamic_beam_budget(n, m0, global_basic_budget)
    )
    effective_elite_num = min(elite_num, beam_width)

    with torch.no_grad():
        initial_embedding = embedder.embed_token_ids(init_ids, attention_mask)
        initial_loss = float(compute_retrieval_attack_loss(
            initial_embedding.squeeze(0),
            pos_embeddings,
            loss_func=loss_func_type,
            dynamic_weight=dynamic_weight,
            tau_stop=tau_stop,
            alpha=alpha,
            softmin_temperature=softmin_temperature,
            hybrid_softmin_weight=hybrid_softmin_weight,
        ).item())
        pos_norm = F.normalize(pos_embeddings, dim=-1)
        initial_similarities = (
            F.normalize(initial_embedding, dim=-1) @ pos_norm.T
        ).squeeze(0)
        initial_avg_similarity = float(initial_similarities.mean().item())

    clean_similarities = initial_similarities.detach().clone()
    success_thresholds = torch.maximum(
        clean_similarities,
        torch.full_like(clean_similarities, tau_stop),
    )
    word_embeddings = get_word_embeddings(embedder.model)

    beam: List[BeamState] = [BeamState(
        token_ids=init_ids.clone(),
        loss=initial_loss,
        embedding=initial_embedding.detach(),
        similarities=initial_similarities.detach(),
        avg_similarity=initial_avg_similarity,
        mutable_positions=list(initial_mutable_positions),
    )]

    best_ids = init_ids.clone()
    best_embedding = initial_embedding.detach().clone()
    best_similarities = initial_similarities.detach().clone()
    best_loss = initial_loss
    best_iteration = 0
    best_avg_similarity = initial_avg_similarity
    best_avg_similarity_iteration = 0
    patience_counter = 0
    max_local_depth = 0
    iteration_logs: List[str] = []
    completed_iterations = 0
    early_stopped = False
    stop_reason = "max_iterations"
    start_time = datetime.now().isoformat()

    total_fresh_evaluated = 0
    total_cache_replayed = 0
    total_cache_hits = 0
    total_global_scanned = 0
    total_sequence_duplicates = 0

    resolved_attack_config = {
        key: value for key, value in attack_cfg.items()
        if key not in {"frozen_token_dict", "beam_width"}
    }
    resolved_attack_config.update({
        "optimizer": "abgs",
        "N": N,
        "n": n,
        "dynamic_beam_width": beam_width,
        "per_beam_budget": per_beam_budget,
        "initial_mutable_tokens": m0,
        "initial_local_budget": m0,
        "initial_global_budget": initial_global_budget,
        "elite_num": elite_num,
        "effective_elite_num": effective_elite_num,
        "global_basic_budget": global_basic_budget,
        "patience": patience,
        "patience_loss_delta": patience_loss_delta,
        "patience_similarity_delta": patience_similarity_delta,
        "loss_func": loss_func_type,
        "dynamic_weight": dynamic_weight,
        "dynamic_weight_detached": dynamic_weight,
        "scaling_factor": alpha,
        "softmin_temperature": softmin_temperature,
        "hybrid_softmin_weight": hybrid_softmin_weight,
        "replacement_score": "embedding_delta",
        "depth_strategy": "per_beam_local_depth_and_global_offset",
        "cache_strategy": "replay_and_global_fresh_backfill",
        "candidate_budget_semantics": "fresh_forward_evaluations",
    })

    logger.info("========== ABGS 局部—全局混合束搜索初始化 ==========")
    logger.info(
        "初始 loss=%.6f, avg_sim=%.6f, mutable=%d, dynamic_B=%d, "
        "per_beam=%d, local/global=%d/%d, elite=%d",
        initial_loss,
        initial_avg_similarity,
        m0,
        beam_width,
        per_beam_budget,
        m0,
        initial_global_budget,
        effective_elite_num,
    )

    for iteration in range(1, N + 1):
        completed_iterations = iteration
        previous_beam = sorted(beam, key=lambda state: state.loss)
        elite_count = min(elite_num, len(previous_beam), beam_width)
        elite_indices = set(range(elite_count))
        old_best_loss = best_loss
        old_best_avg = best_avg_similarity

        pool_entries: List[PoolEntry] = []
        fresh_proposals: Dict[SequenceKey, CandidateProposal] = {}
        round_stats = {
            "local_fresh": 0,
            "local_replay": 0,
            "local_duplicate": 0,
            "local_admitted_skip": 0,
            "global_fresh": 0,
            "global_replay": 0,
            "global_cache_hits": 0,
            "global_duplicate": 0,
            "global_admitted_skip": 0,
            "global_scanned": 0,
            "gradient_cache_hits": 0,
        }
        per_parent_lines: List[str] = []

        def register_pair(
            state: BeamState,
            parent_index: int,
            pair: CandidateKey,
            source: str,
            rank: int,
            round_pairs: set[CandidateKey],
        ) -> str:
            """注册候选，返回 fresh/replay/duplicate/admitted。"""
            nonlocal total_sequence_duplicates
            if pair in round_pairs:
                round_stats[f"{source}_duplicate"] += 1
                return "duplicate"
            round_pairs.add(pair)

            cached = state.candidate_cache.get(pair)
            if cached is not None:
                if source == "global":
                    round_stats["global_cache_hits"] += 1
                if cached.admitted_once:
                    round_stats[f"{source}_admitted_skip"] += 1
                    return "admitted"
                replay_ids = _materialize_replacement(
                    state.token_ids, pair[0], pair[1]
                )
                pool_entries.append(PoolEntry(
                    token_ids=replay_ids,
                    loss=cached.loss,
                    avg_similarity=cached.avg_similarity,
                    similarities=cached.similarities,
                    embedding_cpu=cached.embedding_cpu,
                    is_parent=False,
                    source=f"{source}_cache_replay",
                    metadata={
                        "parent_index": parent_index,
                        "position": pair[0],
                        "new_token_id": pair[1],
                        "rank": rank,
                    },
                    cache_records=[cached],
                ))
                round_stats[f"{source}_replay"] += 1
                return "replay"

            child_ids = _materialize_replacement(
                state.token_ids, pair[0], pair[1]
            )
            sequence_key = _sequence_key(child_ids)
            alias = (
                state,
                pair,
                {
                    "parent_index": parent_index,
                    "position": pair[0],
                    "new_token_id": pair[1],
                    "source": source,
                    "rank": rank,
                },
            )
            if sequence_key in fresh_proposals:
                fresh_proposals[sequence_key].aliases.append(alias)
                total_sequence_duplicates += 1
            else:
                fresh_proposals[sequence_key] = CandidateProposal(
                    token_ids=child_ids,
                    aliases=[alias],
                )
            round_stats[f"{source}_fresh"] += 1
            return "fresh"

        for parent_index, state in enumerate(previous_beam):
            depth_before = state.local_depth
            offset_before = state.global_offset
            round_pairs: set[CandidateKey] = set()

            if state.mutable_positions is None:
                state.mutable_positions = get_mutable_positions(
                    state.token_ids[0].tolist(),
                    tokenizer,
                    frozen_token_dict,
                    offset=prefix_len,
                )
            mutable_positions = state.mutable_positions
            mutable_count = len(mutable_positions)
            if mutable_count > per_beam_budget:
                raise RuntimeError(
                    "Beam 的可扰动位置数超过单 Beam 预算；这违反可扰动位置"
                    "不会在单 token 替换后增加的算法不变量："
                    f"mutable={mutable_count}, K={per_beam_budget}"
                )
            global_budget = max(0, per_beam_budget - mutable_count)

            if state.cached_gradient is None and mutable_positions:
                recomputed_loss, gradient, recomputed_embedding = (
                    compute_retrieval_attack_loss_and_grad(
                        embedder,
                        state.token_ids,
                        attention_mask,
                        pos_embeddings,
                        grad_storage,
                        loss_func=loss_func_type,
                        dynamic_weight=dynamic_weight,
                        tau_stop=tau_stop,
                        alpha=alpha,
                        softmin_temperature=softmin_temperature,
                        hybrid_softmin_weight=hybrid_softmin_weight,
                    )
                )
                state.loss = float(recomputed_loss)
                state.embedding = recomputed_embedding.detach()
                state.cached_gradient = gradient.detach()
                with torch.no_grad():
                    state.similarities = (
                        F.normalize(state.embedding, dim=-1) @ pos_norm.T
                    ).squeeze(0).detach()
                    state.avg_similarity = float(state.similarities.mean().item())
            elif state.cached_gradient is not None:
                round_stats["gradient_cache_hits"] += 1

            if mutable_positions:
                mutable_tensor = torch.tensor(
                    mutable_positions, dtype=torch.long, device=device
                )
                mutable_gradient = state.cached_gradient.index_select(
                    0, mutable_tensor
                )
                mutable_token_ids = state.token_ids[0].index_select(
                    0, mutable_tensor
                )
                with torch.no_grad():
                    current_token_embeddings = (
                        word_embeddings.weight.index_select(
                            0, mutable_token_ids
                        ).detach()
                    )
                scores = _compute_replacement_scores(
                    mutable_gradient,
                    safe_emb_matrix,
                    current_token_embeddings,
                )
                _mask_identity_replacements(scores, safe_ids, mutable_token_ids)
                finite_count = int(torch.isfinite(scores).sum().item())
                state.global_rank_limit = finite_count
                state.local_rank_limit = int(safe_ids.numel())

                local_candidates = _select_local_depth_candidates(
                    scores,
                    mutable_positions,
                    safe_ids,
                    state.local_depth,
                )
                for position, new_token_id, _ in local_candidates:
                    register_pair(
                        state,
                        parent_index,
                        (position, new_token_id),
                        "local",
                        state.local_depth,
                        round_pairs,
                    )

                global_fresh_for_parent = 0
                flat_scores = scores.reshape(-1)
                top_indices = torch.empty(
                    0, dtype=torch.long, device=scores.device
                )
                requested_end = state.global_offset
                while (
                    global_fresh_for_parent < global_budget
                    and state.global_offset < finite_count
                ):
                    if state.global_offset >= top_indices.numel():
                        remaining = global_budget - global_fresh_for_parent
                        requested_end = min(
                            finite_count,
                            max(
                                requested_end * 2,
                                state.global_offset + remaining + 64,
                            ),
                        )
                        _, top_indices = torch.topk(
                            flat_scores,
                            k=requested_end,
                            largest=True,
                            sorted=True,
                        )
                    flat_index = int(top_indices[state.global_offset].item())
                    rank = state.global_offset
                    state.global_offset += 1
                    round_stats["global_scanned"] += 1
                    row = flat_index // mutable_count
                    column = flat_index % mutable_count
                    pair = (
                        int(mutable_positions[column]),
                        int(safe_ids[row].item()),
                    )
                    action = register_pair(
                        state,
                        parent_index,
                        pair,
                        "global",
                        rank,
                        round_pairs,
                    )
                    if action == "fresh":
                        global_fresh_for_parent += 1

                del scores, flat_scores, top_indices

            if parent_index in elite_indices:
                state.local_depth += 1
                pool_entries.append(PoolEntry(
                    token_ids=state.token_ids,
                    loss=state.loss,
                    avg_similarity=state.avg_similarity,
                    similarities=tuple(float(x) for x in state.similarities.tolist()),
                    embedding_cpu=state.embedding.detach().cpu(),
                    is_parent=True,
                    source="elite_parent",
                    metadata={
                        "parent_index": parent_index,
                        "previous_local_depth": depth_before,
                        "previous_global_offset": offset_before,
                    },
                    parent_state=state,
                ))

            per_parent_lines.append(
                f"[Parent {parent_index + 1}] loss={state.loss:.6f}, "
                f"avg_sim={state.avg_similarity:.6f}, mutable={mutable_count}, "
                f"local_depth={depth_before}, global_offset="
                f"{offset_before}->{state.global_offset}, "
                f"budget(local/global)={mutable_count}/{global_budget}, "
                f"cache={len(state.candidate_cache)}, elite={parent_index in elite_indices}"
            )

        unique_proposals = list(fresh_proposals.values())
        evaluated = _evaluate_candidate_sequences(
            embedder,
            [proposal.token_ids for proposal in unique_proposals],
            attention_mask,
            pos_embeddings,
            loss_func=loss_func_type,
            dynamic_weight=dynamic_weight,
            tau_stop=tau_stop,
            alpha=alpha,
            softmin_temperature=softmin_temperature,
            hybrid_softmin_weight=hybrid_softmin_weight,
            batch_size=batch_size,
        )
        if len(evaluated) != len(unique_proposals):
            raise RuntimeError("ABGS 候选评估结果数量与候选数量不一致")

        for proposal, (loss, embedding, similarities, avg_similarity) in zip(
            unique_proposals, evaluated
        ):
            cache_records: List[CachedCandidate] = []
            sources: List[str] = []
            for parent_state, pair, metadata in proposal.aliases:
                record = CachedCandidate(
                    position=pair[0],
                    new_token_id=pair[1],
                    loss=loss,
                    avg_similarity=avg_similarity,
                    similarities=tuple(float(x) for x in similarities.tolist()),
                    embedding_cpu=embedding.detach().cpu(),
                    first_source=str(metadata["source"]),
                    evaluated_iteration=iteration,
                )
                parent_state.candidate_cache[pair] = record
                cache_records.append(record)
                sources.append(str(metadata["source"]))
            primary_metadata = proposal.aliases[0][2]
            pool_entries.append(PoolEntry(
                token_ids=proposal.token_ids,
                loss=loss,
                avg_similarity=avg_similarity,
                similarities=tuple(float(x) for x in similarities.tolist()),
                embedding_cpu=embedding.detach().cpu(),
                is_parent=False,
                source="fresh_" + "+".join(sorted(set(sources))),
                metadata=dict(primary_metadata),
                cache_records=cache_records,
            ))

        fresh_this_round = len(unique_proposals)
        if fresh_this_round > n:
            raise RuntimeError(
                f"本轮真实评估候选数 {fresh_this_round} 超过预算 n={n}"
            )
        total_fresh_evaluated += fresh_this_round
        replay_this_round = round_stats["local_replay"] + round_stats["global_replay"]
        cache_hits_this_round = (
            replay_this_round
            + round_stats["local_admitted_skip"]
            + round_stats["global_admitted_skip"]
        )
        total_cache_replayed += replay_this_round
        total_cache_hits += cache_hits_this_round
        total_global_scanned += round_stats["global_scanned"]

        raw_pool_size = len(pool_entries)
        child_competitor_count = sum(
            1 for entry in pool_entries if not entry.is_parent
        )
        pool_entries.sort(key=lambda entry: (entry.loss, 0 if entry.is_parent else 1))
        unique_pool: List[PoolEntry] = []
        seen_sequences: set[SequenceKey] = set()
        for entry in pool_entries:
            key = _sequence_key(entry.token_ids)
            if key in seen_sequences:
                total_sequence_duplicates += 1
                continue
            seen_sequences.add(key)
            unique_pool.append(entry)

        if not unique_pool:
            # elite_num=0 时父节点不会进入候选池；若本轮也没有任何可用
            # 子节点，就没有状态能够组成下一轮 Beam。此时保留当前 Beam
            # 及历史 best_*，按候选空间耗尽正常结束，最终统一返回 best_ids。
            early_stopped = True
            stop_reason = "candidate_space_exhausted"
            empty_pool_lines = [
                f"{'=' * 12} ABGS Iteration {iteration} / {N} {'=' * 12}",
                (
                    f"[Budget] target_B={beam_width}, "
                    f"actual_before={len(previous_beam)}, "
                    f"per_beam={per_beam_budget}, "
                    f"fresh={fresh_this_round}/{n}, "
                    f"cache_replay={replay_this_round}, "
                    f"pool={raw_pool_size}, unique_pool=0"
                ),
                *per_parent_lines,
                (
                    f"[Search] local fresh/replay/dup/admitted_skip="
                    f"{round_stats['local_fresh']}/{round_stats['local_replay']}/"
                    f"{round_stats['local_duplicate']}/"
                    f"{round_stats['local_admitted_skip']}; "
                    f"global fresh/replay/scanned/dup/admitted_skip="
                    f"{round_stats['global_fresh']}/{round_stats['global_replay']}/"
                    f"{round_stats['global_scanned']}/"
                    f"{round_stats['global_duplicate']}/"
                    f"{round_stats['global_admitted_skip']}"
                ),
                (
                    f"[Beam Update] retained_parents=0, admitted_children=0, "
                    f"size={len(beam)}, unchanged=True"
                ),
                (
                    f"[Best Loss] {old_best_loss:.6f}->{best_loss:.6f}, "
                    f"significant=False, iteration={best_iteration}"
                ),
                (
                    f"[Best AvgSim] {old_best_avg:.6f}->"
                    f"{best_avg_similarity:.6f}, significant=False, "
                    f"iteration={best_avg_similarity_iteration}"
                ),
                f"[Patience] {patience_counter}/{patience} (unchanged)",
                "[Stop] candidate_space_exhausted (empty candidate pool; "
                "returning historical best_ids)",
            ]
            iteration_logs.append("\n".join(empty_pool_lines) + "\n")
            logger.info(
                "ABGS iter=%d/%d candidate pool exhausted; "
                "returning historical best loss=%.6f from iteration=%d",
                iteration,
                N,
                best_loss,
                best_iteration,
            )
            break

        selected_entries = unique_pool[:beam_width]

        new_beam: List[BeamState] = []
        admitted_children = 0
        retained_parents = 0
        admitted_child_details: List[str] = []
        for entry in selected_entries:
            if entry.is_parent:
                retained_parents += 1
                if entry.parent_state is None:
                    raise RuntimeError("精英父节点缺少 BeamState")
                new_beam.append(entry.parent_state)
                continue
            admitted_children += 1
            admitted_child_details.append(
                f"source={entry.source}, parent={entry.metadata.get('parent_index', '?')}, "
                f"position={entry.metadata.get('position', '?')}, "
                f"new_token={entry.metadata.get('new_token_id', '?')}, "
                f"loss={entry.loss:.6f}, avg_sim={entry.avg_similarity:.6f}"
            )
            for record in entry.cache_records:
                record.admitted_once = True
            embedding = entry.embedding_cpu.to(device=device)
            similarities = torch.tensor(
                entry.similarities,
                dtype=pos_embeddings.dtype,
                device=device,
            )
            new_beam.append(BeamState(
                token_ids=entry.token_ids.clone(),
                loss=entry.loss,
                embedding=embedding,
                similarities=similarities,
                avg_similarity=entry.avg_similarity,
            ))
        beam = new_beam

        iteration_best_loss_state = min(beam, key=lambda state: state.loss)
        iteration_best_avg_state = max(
            beam, key=lambda state: state.avg_similarity
        )
        (
            patience_counter,
            loss_improved_for_patience,
            avg_improved_for_patience,
        ) = _update_patience_counter(
            patience_counter,
            iteration_best_loss_state.loss,
            iteration_best_avg_state.avg_similarity,
            old_best_loss,
            old_best_avg,
            patience_loss_delta,
            patience_similarity_delta,
        )

        if iteration_best_loss_state.loss < best_loss:
            best_loss = iteration_best_loss_state.loss
            best_ids = iteration_best_loss_state.token_ids.clone()
            best_embedding = iteration_best_loss_state.embedding.detach().clone()
            best_similarities = iteration_best_loss_state.similarities.detach().clone()
            best_iteration = iteration
        if iteration_best_avg_state.avg_similarity > best_avg_similarity:
            best_avg_similarity = iteration_best_avg_state.avg_similarity
            best_avg_similarity_iteration = iteration

        max_local_depth = max(
            max_local_depth,
            max(state.local_depth for state in beam),
        )

        success_state: Optional[BeamState] = None
        for state in beam:
            if (state.similarities > success_thresholds).all():
                success_state = state
                break

        # 仅当本轮根本没有可参与竞争的子节点，且入选 Beam 的
        # 所有状态都已经耗尽局部排名和全局排名时，才判定候选空间耗尽。
        no_child_competitors = child_competitor_count == 0
        search_exhausted = (
            no_child_competitors
            and all(not _has_future_search_space(state) for state in beam)
        )
        stop_now = False
        if success_state is not None:
            best_ids = success_state.token_ids.clone()
            best_embedding = success_state.embedding.detach().clone()
            best_similarities = success_state.similarities.detach().clone()
            best_loss = success_state.loss
            best_iteration = iteration
            early_stopped = True
            stop_reason = "success_threshold"
            stop_now = True
        elif search_exhausted:
            early_stopped = True
            stop_reason = "candidate_space_exhausted"
            stop_now = True
        elif patience > 0 and patience_counter >= patience:
            early_stopped = True
            stop_reason = "patience"
            stop_now = True

        iter_lines = [
            f"{'=' * 12} ABGS Iteration {iteration} / {N} {'=' * 12}",
            (
                f"[Budget] target_B={beam_width}, actual_before={len(previous_beam)}, "
                f"per_beam={per_beam_budget}, fresh={fresh_this_round}/{n}, "
                f"cache_replay={replay_this_round}, pool={raw_pool_size}, "
                f"unique_pool={len(unique_pool)}"
            ),
            *per_parent_lines,
            (
                f"[Search] local fresh/replay/dup/admitted_skip="
                f"{round_stats['local_fresh']}/{round_stats['local_replay']}/"
                f"{round_stats['local_duplicate']}/{round_stats['local_admitted_skip']}; "
                f"global fresh/replay/scanned/dup/admitted_skip="
                f"{round_stats['global_fresh']}/{round_stats['global_replay']}/"
                f"{round_stats['global_scanned']}/{round_stats['global_duplicate']}/"
                f"{round_stats['global_admitted_skip']}; "
                f"gradient_cache_hits={round_stats['gradient_cache_hits']}"
            ),
            (
                f"[Beam Update] retained_parents={retained_parents}, "
                f"admitted_children={admitted_children}, size={len(beam)}, "
                f"local_depths={[state.local_depth for state in beam]}, "
                f"global_offsets={[state.global_offset for state in beam]}"
            ),
            f"[Admitted Children] {admitted_child_details}",
            (
                f"[Best Loss] {old_best_loss:.6f}->{best_loss:.6f}, "
                f"significant={loss_improved_for_patience}, iteration={best_iteration}"
            ),
            (
                f"[Best AvgSim] {old_best_avg:.6f}->{best_avg_similarity:.6f}, "
                f"significant={avg_improved_for_patience}, "
                f"iteration={best_avg_similarity_iteration}"
            ),
            f"[Patience] {patience_counter}/{patience}",
        ]
        for rank, state in enumerate(beam, 1):
            iter_lines.append(
                f"[Beam #{rank}] loss={state.loss:.6f}, "
                f"avg/min/max_sim={state.avg_similarity:.6f}/"
                f"{state.similarities.min().item():.6f}/"
                f"{state.similarities.max().item():.6f}, "
                f"local_depth={state.local_depth}, "
                f"global_offset={state.global_offset}, "
                f"cache={len(state.candidate_cache)}, "
                f"per_query_sim={[round(float(x), 6) for x in state.similarities.tolist()]}"
            )
        if stop_now:
            iter_lines.append(f"[Stop] {stop_reason}")
        if (
            iteration == 1
            or iteration % log_every_n_iter == 0
            or stop_now
            or iteration == N
        ):
            iteration_logs.append("\n".join(iter_lines) + "\n")

        logger.info(
            "ABGS iter=%d/%d beam=%d fresh=%d replay=%d "
            "best_loss=%.6f best_avg=%.6f patience=%d/%d",
            iteration,
            N,
            len(beam),
            fresh_this_round,
            replay_this_round,
            best_loss,
            best_avg_similarity,
            patience_counter,
            patience,
        )
        if stop_now:
            break

    final_ids = best_ids[0].tolist()[prefix_len:]
    final_text = tokenizer.decode(
        final_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    final_avg_similarity = float(best_similarities.mean().item())
    final_global_offsets = [state.global_offset for state in beam]
    final_local_depths = [state.local_depth for state in beam]
    end_time = datetime.now().isoformat()

    logger.info("========== ABGS 局部—全局混合束搜索结束 ==========")
    logger.info(
        "iterations=%d/%d, reason=%s, final_loss=%.6f, final_avg=%.6f, "
        "fresh=%d, replay=%d, cache_hits=%d, global_scanned=%d",
        completed_iterations,
        N,
        stop_reason,
        best_loss,
        final_avg_similarity,
        total_fresh_evaluated,
        total_cache_replayed,
        total_cache_hits,
        total_global_scanned,
    )

    return {
        "adv_text": final_text,
        "adv_token_ids": final_ids,
        "init_loss": initial_loss,
        "init_sim": clean_similarities.tolist(),
        "init_avg_sim": initial_avg_similarity,
        "final_loss": best_loss,
        "final_sim": best_similarities.tolist(),
        "final_avg_sim": final_avg_similarity,
        "iteration_logs": iteration_logs,
        "total_iterations": completed_iterations,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "best_iteration": best_iteration,
        "patience_counter": patience_counter,
        "dynamic_beam_width": beam_width,
        "per_beam_budget": per_beam_budget,
        "effective_elite_num": effective_elite_num,
        "max_local_depth": max_local_depth,
        "max_depth_reached": max_local_depth,
        "final_local_depths": final_local_depths,
        "final_beam_depths": final_local_depths,
        "final_global_offsets": final_global_offsets,
        "best_avg_similarity": best_avg_similarity,
        "best_avg_similarity_iteration": best_avg_similarity_iteration,
        "patience_loss_delta": patience_loss_delta,
        "patience_similarity_delta": patience_similarity_delta,
        "total_fresh_evaluated": total_fresh_evaluated,
        "total_cache_replayed": total_cache_replayed,
        "total_cache_hits": total_cache_hits,
        "total_global_scanned": total_global_scanned,
        "total_sequence_duplicates": total_sequence_duplicates,
        "total_candidates_generated": total_fresh_evaluated + total_cache_replayed,
        "total_candidates_evaluated": total_fresh_evaluated,
        "resolved_attack_config": resolved_attack_config,
        "start_time": start_time,
        "end_time": end_time,
        "algorithm": "abgs",
    }
