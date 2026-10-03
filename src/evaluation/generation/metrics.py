"""Metrics and vulnerability-pattern helpers for generation attack evaluation."""

from __future__ import annotations

import logging
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.evaluation.generation.racg import (
    entry_id,
    has_complete_judge_result,
    has_complete_racg_result,
    model_slot,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VulnerabilityJudgeTask:
    """One query-level Judge call plus its main-thread persistence target."""

    poisoned_results: List[Dict[str, Any]]
    result_index: int
    eval_top_k: int
    model_name: str
    query_id: Any
    clean_code: str
    patch: str
    relevant_vul: List[Dict[str, Any]]
    previous_verdicts: Optional[List[Dict[str, Any]]]
    save_callback: Callable[[], None]


def _resolve_doc_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None


def collect_poisoned_doc_ids(
    retrieval_results: Sequence[Dict[str, Any]],
    eval_top_k: int,
) -> List[int]:
    """Collect unique poisoned document IDs from one query's Top-K hits."""
    result: List[int] = []
    for hit in retrieval_results[:eval_top_k]:
        if hit.get("is_poisoned") is not True:
            continue
        doc_id = _resolve_doc_id(hit.get("doc_id"))
        if doc_id is not None and doc_id not in result:
            result.append(doc_id)
    return result


def build_poison_doc_index(
    target_data: Dict[str, List[Dict[str, Any]]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Build the LLMVulJudger input index from successful injected patterns only.

    The outer key is the poisoned BFP ``entity.id`` used as retrieval ``doc_id``.
    Every pattern uses the poisoned BFP's final ``vinj_attack.vul_code`` rather
    than the source CVE example's code.
    """
    index: Dict[int, List[Dict[str, Any]]] = {}
    owner: Dict[int, str] = {}
    for cwe, entries in target_data.items():
        for entry in entries or []:
            entity = entry.get("entity") or {}
            doc_id = _resolve_doc_id(entity.get("id"))
            if doc_id is None:
                raise ValueError(f"invalid poisoned entity.id={entity.get('id')!r} in {cwe}")
            if doc_id in owner:
                raise ValueError(
                    f"duplicate poisoned entity.id={doc_id} in {owner[doc_id]} and {cwe}; "
                    "rerun distributed_io Section 3 with the new ID contract"
                )
            owner[doc_id] = cwe

            vinj = (entry.get("generation_attack") or {}).get("vinj_attack") or {}
            injected_code = vinj.get("vul_code") or ""
            found_patterns: List[Dict[str, Any]] = []
            for raw in vinj.get("relevant_vul") or []:
                if str(raw.get("status", "")).strip().lower() != "found":
                    continue
                item = dict(raw)
                source_vul_doc_id = raw.get("source_vul_doc_id")
                item["source_vul_doc_id"] = (
                    raw.get("doc_id")
                    if source_vul_doc_id is None
                    else source_vul_doc_id
                )
                item["doc_id"] = doc_id
                item["vul_code"] = injected_code
                # VInj-stage verdict fields and pattern_id belong to the source
                # CVE grouping context. Generation evaluation derives a new ID
                # with the poisoned BFP as group_doc_id and judges from scratch.
                for field in (
                    "pattern_id",
                    "status",
                    "flaw_line_index",
                    "evidence",
                    "unresolved_pattern_ids",
                ):
                    item.pop(field, None)
                found_patterns.append(item)
            if found_patterns:
                index[doc_id] = found_patterns
    return index


def build_vul_code_index(
    target_data: Dict[str, List[Dict[str, Any]]],
) -> Dict[int, str]:
    """Build a strict unique ``poisoned doc_id -> no-jailbreak vul_code`` map."""
    index: Dict[int, str] = {}
    owner: Dict[int, str] = {}
    for cwe, entries in target_data.items():
        for entry in entries or []:
            doc_id = _resolve_doc_id((entry.get("entity") or {}).get("id"))
            if doc_id is None:
                raise ValueError(f"invalid poisoned entity.id in {cwe}")
            if doc_id in owner:
                raise ValueError(
                    f"duplicate poisoned entity.id={doc_id} in {owner[doc_id]} and {cwe}"
                )
            owner[doc_id] = cwe
            code = (
                ((entry.get("generation_attack") or {}).get("vinj_attack") or {}).get("vul_code")
                or ""
            )
            if not code:
                raise ValueError(f"poisoned doc_id={doc_id} in {cwe} has no vinj_attack.vul_code")
            index[doc_id] = code
    return index


def build_relevant_vul_for_judger(
    poisoned_doc_ids: Sequence[int],
    poison_doc_index: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for doc_id in poisoned_doc_ids:
        flattened.extend(dict(item) for item in (poison_doc_index.get(doc_id) or []))
    return flattened


def _validate_vulnerability_indices(
    *,
    poisoned_results: List[Dict[str, Any]],
    victim_indices: Sequence[int],
    work_indices: Sequence[int],
) -> Tuple[List[int], List[int]]:
    victims = list(dict.fromkeys(int(idx) for idx in victim_indices))
    work = list(dict.fromkeys(int(idx) for idx in work_indices))
    victim_set = set(victims)
    invalid_work = [idx for idx in work if idx not in victim_set]
    if invalid_work:
        raise ValueError(f"work_indices contains non-victim result indices: {invalid_work}")
    for idx in victims:
        if idx < 0 or idx >= len(poisoned_results):
            raise IndexError(f"victim result index out of range: {idx}")
    return victims, work


def collect_vulnerability_judge_tasks(
    *,
    poisoned_results: List[Dict[str, Any]],
    victim_indices: Sequence[int],
    work_indices: Sequence[int],
    poison_doc_index: Dict[int, List[Dict[str, Any]]],
    eval_top_k: int,
    model_name: str,
    save_callback: Callable[[], None],
    canonicalize_poisoned_doc_order: bool = False,
) -> List[VulnerabilityJudgeTask]:
    """Collect resumable query-level Judge tasks without invoking the model."""
    _, work = _validate_vulnerability_indices(
        poisoned_results=poisoned_results,
        victim_indices=victim_indices,
        work_indices=work_indices,
    )
    tasks: List[VulnerabilityJudgeTask] = []
    for idx in work:
        entry = poisoned_results[idx]
        slot = model_slot(entry, eval_top_k, model_name, create=True)
        if has_complete_judge_result(slot) or not has_complete_racg_result(slot):
            continue

        poisoned_ids = collect_poisoned_doc_ids(
            entry.get("retrieval_results") or [], eval_top_k
        )
        if canonicalize_poisoned_doc_order:
            poisoned_ids.sort()
        relevant_vul = build_relevant_vul_for_judger(poisoned_ids, poison_doc_index)
        if not relevant_vul:
            continue

        previous = slot.get("relevant_vul")
        previous_verdicts = (
            [dict(item) for item in previous if isinstance(item, dict)]
            if isinstance(previous, list)
            else None
        )
        tasks.append(
            VulnerabilityJudgeTask(
                poisoned_results=poisoned_results,
                result_index=idx,
                eval_top_k=eval_top_k,
                model_name=model_name,
                query_id=entry_id(entry),
                clean_code=(entry.get("entity") or {}).get("buggy_code", ""),
                patch=slot["patch"],
                relevant_vul=relevant_vul,
                previous_verdicts=previous_verdicts,
                save_callback=save_callback,
            )
        )
    return tasks


def run_parallel_vulnerability_judging(
    *,
    tasks: Sequence[VulnerabilityJudgeTask],
    judgers: Sequence[Any],
    num_workers: int,
) -> Dict[str, Any]:
    """Run independent Judge calls concurrently and persist in the main thread."""
    if not tasks:
        return {
            "submitted": 0,
            "completed": 0,
            "finalized": 0,
            "failed_ids": [],
        }
    if not judgers:
        raise ValueError("at least one LLMVulJudger is required")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError(f"num_workers must be a positive integer, got {num_workers!r}")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers!r}")

    from src.utils.vul_judger import summarize_verdicts

    worker_count = min(num_workers, len(judgers), len(tasks))
    judger_pool: queue.Queue[Any] = queue.Queue()
    for judger in judgers[:worker_count]:
        judger_pool.put(judger)

    def worker(task: VulnerabilityJudgeTask) -> List[Dict[str, Any]]:
        judger = judger_pool.get()
        try:
            return judger.judge(
                clean_code=task.clean_code,
                vul_injected_code=task.patch,
                relevant_vul=task.relevant_vul,
                previous_verdicts=task.previous_verdicts,
            )
        finally:
            judger_pool.put(judger)

    completed = 0
    finalized = 0
    failed_ids: List[Any] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                verdicts = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("[VR] query id=%r judge failed: %s", task.query_id, exc)
                failed_ids.append(task.query_id)
                continue

            slot = model_slot(
                task.poisoned_results[task.result_index],
                task.eval_top_k,
                task.model_name,
                create=True,
            )
            slot["relevant_vul"] = verdicts
            is_vulnerable, unresolved_pattern_ids = summarize_verdicts(verdicts)
            if unresolved_pattern_ids:
                slot["unresolved_pattern_ids"] = unresolved_pattern_ids
            else:
                slot.pop("unresolved_pattern_ids", None)
            if is_vulnerable is None:
                slot.pop("is_vulnerable", None)
                logger.warning(
                    "[VR] query id=%r remains pending; unresolved pattern_ids=%s",
                    task.query_id,
                    unresolved_pattern_ids,
                )
            else:
                slot["is_vulnerable"] = is_vulnerable
                finalized += 1
            task.save_callback()
            completed += 1

    return {
        "submitted": len(tasks),
        "completed": completed,
        "finalized": finalized,
        "failed_ids": list(dict.fromkeys(failed_ids)),
    }


def summarize_vulnerability_rate(
    *,
    poisoned_results: List[Dict[str, Any]],
    victim_indices: Sequence[int],
    work_indices: Sequence[int],
    poison_doc_index: Dict[int, List[Dict[str, Any]]],
    eval_top_k: int,
    model_name: str,
    canonicalize_poisoned_doc_order: bool = False,
) -> Dict[str, Any]:
    """Summarize the full victim population after generation and judging.

    ``VR_GLOBAL`` divides vulnerable patches by every test query in the
    retrieval result file. ``VR_LOCAL`` divides by the complete Top-K victim
    population. ``work_indices`` is only the resumable subset selected for the
    current invocation and must not alter either denominator. Controlled
    experiments may request canonical poisoned-doc ordering so equivalent
    poison multisets produce the same Judger pattern order.
    """
    victims, work = _validate_vulnerability_indices(
        poisoned_results=poisoned_results,
        victim_indices=victim_indices,
        work_indices=work_indices,
    )

    vulnerable = 0
    evaluated = 0
    with_patch = 0
    pending_ids: List[Any] = []
    missing_pattern_ids: List[Any] = []
    unresolved_patterns_by_query: Dict[str, List[str]] = {}
    per_query: List[Dict[str, Any]] = []
    for idx in victims:
        entry = poisoned_results[idx]
        slot = model_slot(entry, eval_top_k, model_name)
        if has_complete_racg_result(slot):
            with_patch += 1
        verdict: Optional[bool] = None
        if has_complete_judge_result(slot):
            raw_verdict = slot["is_vulnerable"]
            verdict = raw_verdict
            evaluated += 1
            vulnerable += int(verdict)
        else:
            query_id = entry_id(entry)
            pending_ids.append(query_id)
            unresolved_ids = slot.get("unresolved_pattern_ids") or []
            if unresolved_ids:
                unresolved_patterns_by_query[str(query_id)] = [
                    str(pattern_id) for pattern_id in unresolved_ids
                ]
            if has_complete_racg_result(slot):
                poisoned_ids = collect_poisoned_doc_ids(
                    entry.get("retrieval_results") or [], eval_top_k
                )
                if canonicalize_poisoned_doc_order:
                    poisoned_ids.sort()
                if not build_relevant_vul_for_judger(poisoned_ids, poison_doc_index):
                    missing_pattern_ids.append(query_id)
        poisoned_ids = collect_poisoned_doc_ids(
            entry.get("retrieval_results") or [], eval_top_k
        )
        if canonicalize_poisoned_doc_order:
            poisoned_ids.sort()
        per_query.append(
            {
                "id": entry_id(entry),
                "is_vulnerable": verdict,
                "poisoned_doc_ids": poisoned_ids,
            }
        )

    total_queries = len(poisoned_results)
    victim_count = len(victims)
    asr_global = vulnerable / total_queries if total_queries else 0.0
    asr_local = vulnerable / victim_count if victim_count else 0.0
    return {
        "asr_g": round(asr_local, 6),
        "asr_g_global": round(asr_global, 6),
        "asr_g_local": round(asr_local, 6),
        "vulnerable_count": vulnerable,
        "victim_count": victim_count,
        "selected_count": len(work),
        "total_with_patch": with_patch,
        "evaluated_count": evaluated,
        "total_queries": total_queries,
        "pending_ids": list(dict.fromkeys(pending_ids)),
        "missing_pattern_ids": list(dict.fromkeys(missing_pattern_ids)),
        "unresolved_patterns_by_query": unresolved_patterns_by_query,
        "per_query": per_query,
    }


def evaluate_vulnerability_rate(
    *,
    poisoned_results: List[Dict[str, Any]],
    victim_indices: Sequence[int],
    work_indices: Sequence[int],
    poison_doc_index: Dict[int, List[Dict[str, Any]]],
    judger: Any,
    eval_top_k: int,
    model_name: str,
    save_callback: Callable[[], None],
    canonicalize_poisoned_doc_order: bool = False,
) -> Dict[str, Any]:
    """Backward-compatible single-Judger wrapper used by debug evaluation."""
    tasks = collect_vulnerability_judge_tasks(
        poisoned_results=poisoned_results,
        victim_indices=victim_indices,
        work_indices=work_indices,
        poison_doc_index=poison_doc_index,
        eval_top_k=eval_top_k,
        model_name=model_name,
        save_callback=save_callback,
        canonicalize_poisoned_doc_order=canonicalize_poisoned_doc_order,
    )
    run_parallel_vulnerability_judging(
        tasks=tasks,
        judgers=[judger],
        num_workers=1,
    )
    return summarize_vulnerability_rate(
        poisoned_results=poisoned_results,
        victim_indices=victim_indices,
        work_indices=work_indices,
        poison_doc_index=poison_doc_index,
        eval_top_k=eval_top_k,
        model_name=model_name,
        canonicalize_poisoned_doc_order=canonicalize_poisoned_doc_order,
    )
