#!/usr/bin/env python3
"""Controlled multi-optimizer RACG ablation evaluation.

The evaluator keeps only queries whose Top-K poisoned-document multiset is
identical across every configured retrieval optimizer.  Prompt references are
then selected according to a controlled recall strategy while the original
retrieval results remain intact for auditing and vulnerability evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.evaluation.generation import metrics as gen_metrics
from src.evaluation.generation.generation_attack_eval import (
    _retrieval_metadata,
    find_result_file,
    parse_counts,
    resolve_eval_scope,
    strategy_to_alpha_label,
)
from src.evaluation.generation.racg import (
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_MAX_REF_CHARS,
    atomic_save_json,
    build_apr_prompt,
    has_complete_judge_result,
    has_complete_racg_result,
    make_atomic_saver,
    model_slot,
    normalize_top_k,
    run_racg,
    validate_result_schema,
)
from src.utils.io import load_json, load_yaml, normalize_bfp_record_id

logger = logging.getLogger(__name__)

VALID_RECALL_STRATEGIES = {"poison_only", "overlap_recall", "all"}
DEBUG_CONTEXT_VERSION = 1
ReferenceIdentity = Tuple[Any, Any]


@dataclass
class MethodSpec:
    optimizer: str
    embedder: str
    poisoned_file: str
    retrieval_dir: str
    poison_doc_index: Dict[int, List[Dict[str, Any]]]
    vul_code_index: Dict[int, str]


@dataclass
class ResultState:
    source_path: str
    result_path: str
    target_count: int
    test_count: int
    results: List[Dict[str, Any]]
    query_index: Dict[Any, int]
    save: Callable[[], None]


def _project_root() -> Path:
    root = Path(os.path.abspath("")).resolve()
    while root != root.parent and not (root / "configs").exists():
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _resolve(root: Path, value: str) -> str:
    return value if os.path.isabs(value) else str((root / value).resolve())


def _required_id(value: Any, label: str) -> Any:
    normalized = normalize_bfp_record_id(value)
    if normalized is None or normalized == "" or isinstance(normalized, bool):
        raise ValueError(f"{label} must be a non-empty int/string ID, got {value!r}")
    return normalized


def _id_sort_key(value: Any) -> Tuple[int, Any]:
    normalized = _required_id(value, "document ID")
    if isinstance(normalized, int):
        return 0, normalized
    return 1, str(normalized)


def _reference_identity(hit: Mapping[str, Any], label: str) -> ReferenceIdentity:
    doc_id = _required_id(hit.get("doc_id"), f"{label}.doc_id")
    chunk_id = normalize_bfp_record_id(hit.get("chunk_id", 0))
    if chunk_id is None or isinstance(chunk_id, bool):
        raise ValueError(f"{label}.chunk_id is invalid: {hit.get('chunk_id')!r}")
    return doc_id, chunk_id


def _reference_sort_key(
    ranked_hit: Tuple[int, Dict[str, Any]],
) -> Tuple[Tuple[int, Any], Tuple[int, Any], int]:
    rank, hit = ranked_hit
    doc_id, chunk_id = _reference_identity(hit, f"retrieval rank {rank + 1}")
    return _id_sort_key(doc_id), _id_sort_key(chunk_id), rank


def _code_digest(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}")
    return hashlib.sha256(value.encode("utf-8")).digest()


def poison_signature(entry: Mapping[str, Any], top_k: int) -> Counter[Any]:
    """Return the strict poisoned doc-id multiset in Top-K."""
    signature: Counter[Any] = Counter()
    for rank, hit in enumerate((entry.get("retrieval_results") or [])[:top_k]):
        if hit.get("is_poisoned") is not True:
            continue
        signature[
            _required_id(hit.get("doc_id"), f"poisoned retrieval rank {rank + 1}.doc_id")
        ] += 1
    return signature


def build_query_index(
    results: Sequence[Dict[str, Any]],
    label: str,
) -> Dict[Any, int]:
    """Index a retrieval result file by normalized, globally unique query ID."""
    index: Dict[Any, int] = {}
    for position, entry in enumerate(results):
        query_id = _required_id(
            (entry.get("entity") or {}).get("id"),
            f"{label}[{position}].entity.id",
        )
        if query_id in index:
            raise ValueError(
                f"{label} contains duplicate query id={query_id!r} at indices "
                f"{index[query_id]} and {position}"
            )
        index[query_id] = position
    return index


def validate_retrieval_results(
    results: Sequence[Dict[str, Any]],
    *,
    label: str,
    declared_test_count: int,
    max_top_k: int,
) -> Dict[Any, int]:
    if len(results) != declared_test_count:
        raise ValueError(
            f"{label} declares test_{declared_test_count} but contains {len(results)} queries"
        )
    query_index = build_query_index(results, label)
    for query_id, position in query_index.items():
        hits = results[position].get("retrieval_results")
        if not isinstance(hits, list):
            raise TypeError(f"{label} query id={query_id!r} has no retrieval_results list")
        if len(hits) < max_top_k:
            raise ValueError(
                f"{label} query id={query_id!r} has retrieval depth {len(hits)}, "
                f"smaller than requested Top-{max_top_k}"
            )
    return query_index


def validate_cross_method_queries(
    results_by_method: Mapping[str, Sequence[Dict[str, Any]]],
    indexes_by_method: Mapping[str, Mapping[Any, int]],
    *,
    unit: str,
) -> None:
    methods = list(results_by_method)
    first = methods[0]
    expected_ids = set(indexes_by_method[first])
    for method in methods[1:]:
        actual_ids = set(indexes_by_method[method])
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids, key=_id_sort_key)
            extra = sorted(actual_ids - expected_ids, key=_id_sort_key)
            raise ValueError(
                f"{unit}: query population differs for {method}; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

    for query_id in expected_ids:
        first_entry = results_by_method[first][indexes_by_method[first][query_id]]
        expected_buggy = (first_entry.get("entity") or {}).get("buggy_code")
        for method in methods[1:]:
            entry = results_by_method[method][indexes_by_method[method][query_id]]
            actual_buggy = (entry.get("entity") or {}).get("buggy_code")
            if actual_buggy != expected_buggy:
                raise ValueError(
                    f"{unit}: query id={query_id!r} has different entity.buggy_code "
                    f"between {first} and {method}"
                )


def build_target_snapshot(
    target_data: Mapping[str, Sequence[Dict[str, Any]]],
    label: str,
) -> Dict[Any, Dict[str, Any]]:
    """Capture target fields that must remain invariant across optimizers."""
    snapshot: Dict[Any, Dict[str, Any]] = {}
    for cwe, entries in target_data.items():
        for position, entry in enumerate(entries or []):
            entity = entry.get("entity") or {}
            doc_id = _required_id(entity.get("id"), f"{label}.{cwe}[{position}].entity.id")
            if doc_id in snapshot:
                raise ValueError(f"{label} contains duplicate poisoned doc_id={doc_id!r}")
            generation = entry.get("generation_attack") or {}
            vinj = generation.get("vinj_attack") or {}
            jailbreak = generation.get("jailbreak_attack") or {}
            found_patterns = [
                pattern
                for pattern in (vinj.get("relevant_vul") or [])
                if str(pattern.get("status", "")).strip().lower() == "found"
            ]
            snapshot[doc_id] = {
                "cwe": str(cwe),
                "buggy_code": entity.get("buggy_code"),
                "fixed_code": entity.get("fixed_code"),
                "vul_code": vinj.get("vul_code"),
                "jailbreak_code": jailbreak.get("jailbreak_code"),
                "found_patterns_sha256": hashlib.sha256(
                    json.dumps(
                        found_patterns,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
    return snapshot


def validate_target_snapshots(
    snapshots: Mapping[str, Mapping[Any, Dict[str, Any]]],
) -> None:
    methods = list(snapshots)
    first = methods[0]
    expected_ids = set(snapshots[first])
    invariant_fields = (
        "cwe",
        "buggy_code",
        "fixed_code",
        "vul_code",
        "jailbreak_code",
        "found_patterns_sha256",
    )
    for method in methods[1:]:
        actual_ids = set(snapshots[method])
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids, key=_id_sort_key)
            extra = sorted(actual_ids - expected_ids, key=_id_sort_key)
            raise ValueError(
                f"poison target IDs differ for {method}; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        for doc_id in expected_ids:
            expected = snapshots[first][doc_id]
            actual = snapshots[method][doc_id]
            for field in invariant_fields:
                if actual.get(field) != expected.get(field):
                    raise ValueError(
                        f"poisoned doc_id={doc_id!r} field {field!r} differs "
                        f"between {first} and {method}"
                    )


def build_common_cohort(
    results_by_method: Mapping[str, Sequence[Dict[str, Any]]],
    indexes_by_method: Mapping[str, Mapping[Any, int]],
    top_k: int,
) -> Dict[str, Any]:
    """Build the ordered controlled victim population for one Top-K."""
    methods = list(results_by_method)
    first = methods[0]
    ordered_query_ids = [
        _required_id((entry.get("entity") or {}).get("id"), "query entity.id")
        for entry in results_by_method[first]
    ]
    raw_victim_counts = {method: 0 for method in methods}
    common_query_ids: List[Any] = []
    no_poison_count = 0
    mismatch_count = 0

    for query_id in ordered_query_ids:
        signatures: List[Counter[Any]] = []
        for method in methods:
            entry = results_by_method[method][indexes_by_method[method][query_id]]
            signature = poison_signature(entry, top_k)
            signatures.append(signature)
            if signature:
                raw_victim_counts[method] += 1
        if not any(signatures):
            no_poison_count += 1
        elif signatures[0] and all(signature == signatures[0] for signature in signatures[1:]):
            common_query_ids.append(query_id)
        else:
            mismatch_count += 1

    return {
        "query_ids": common_query_ids,
        "raw_victim_counts": raw_victim_counts,
        "no_poison_count": no_poison_count,
        "mismatch_count": mismatch_count,
    }


def select_reference_views(
    entries_by_method: Mapping[str, Dict[str, Any]],
    top_k: int,
    strategy: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Select and canonically order prompt references for every optimizer."""
    if strategy not in VALID_RECALL_STRATEGIES:
        raise ValueError(
            f"eval.eval_recall_strategy must be one of {sorted(VALID_RECALL_STRATEGIES)}, "
            f"got {strategy!r}"
        )
    ranked_by_method: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {
        method: list(enumerate((entry.get("retrieval_results") or [])[:top_k]))
        for method, entry in entries_by_method.items()
    }

    common_clean: Counter[ReferenceIdentity] = Counter()
    if strategy == "overlap_recall":
        clean_counters: List[Counter[ReferenceIdentity]] = []
        for method, ranked_hits in ranked_by_method.items():
            counter: Counter[ReferenceIdentity] = Counter()
            for rank, hit in ranked_hits:
                if hit.get("is_poisoned") is True:
                    continue
                counter[_reference_identity(hit, f"{method} retrieval rank {rank + 1}")] += 1
            clean_counters.append(counter)
        if clean_counters:
            common_clean = clean_counters[0].copy()
            for counter in clean_counters[1:]:
                common_clean &= counter

    selected_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method, ranked_hits in ranked_by_method.items():
        selected: List[Tuple[int, Dict[str, Any]]] = []
        used_clean: Counter[ReferenceIdentity] = Counter()
        for rank, hit in ranked_hits:
            is_poisoned = hit.get("is_poisoned") is True
            keep = strategy == "all" or is_poisoned
            if strategy == "overlap_recall" and not is_poisoned:
                identity = _reference_identity(hit, f"{method} retrieval rank {rank + 1}")
                if used_clean[identity] < common_clean[identity]:
                    used_clean[identity] += 1
                    keep = True
            if keep:
                selected.append((rank, hit))
        selected.sort(key=_reference_sort_key)
        selected_by_method[method] = [hit for _, hit in selected]

    methods = list(selected_by_method)
    if methods:
        first = methods[0]
        expected_poison = [
            (
                _reference_identity(hit, f"{first} poisoned reference"),
                hit.get("fixed_code"),
            )
            for hit in selected_by_method[first]
            if hit.get("is_poisoned") is True
        ]
        for method in methods[1:]:
            actual_poison = [
                (
                    _reference_identity(hit, f"{method} poisoned reference"),
                    hit.get("fixed_code"),
                )
                for hit in selected_by_method[method]
                if hit.get("is_poisoned") is True
            ]
            if actual_poison != expected_poison:
                raise ValueError(
                    "controlled poisoned references differ by chunk/fixed_code "
                    f"between {first} and {method}"
                )

        if strategy == "overlap_recall":
            expected_clean = [
                (
                    _reference_identity(hit, f"{first} clean reference"),
                    hit.get("buggy_code"),
                    hit.get("fixed_code"),
                )
                for hit in selected_by_method[first]
                if hit.get("is_poisoned") is not True
            ]
            for method in methods[1:]:
                actual_clean = [
                    (
                        _reference_identity(hit, f"{method} clean reference"),
                        hit.get("buggy_code"),
                        hit.get("fixed_code"),
                    )
                    for hit in selected_by_method[method]
                    if hit.get("is_poisoned") is not True
                ]
                if actual_clean != expected_clean:
                    raise ValueError(
                        "controlled clean references differ in code content "
                        f"between {first} and {method}"
                    )
    return selected_by_method


def _build_debug_results(
    source_results: Sequence[Dict[str, Any]],
    *,
    ablation: bool,
    vul_code_index: Mapping[int, str],
) -> Tuple[List[Dict[str, Any]], int]:
    results: List[Dict[str, Any]] = []
    replaced = 0
    for source_entry in source_results:
        entry, entry_replaced = _build_debug_entry(
            source_entry,
            ablation=ablation,
            vul_code_index=vul_code_index,
        )
        results.append(entry)
        replaced += entry_replaced
    return results, replaced


def _build_debug_entry(
    source_entry: Mapping[str, Any],
    *,
    ablation: bool,
    vul_code_index: Mapping[int, str],
) -> Tuple[Dict[str, Any], int]:
    entry = {
        key: copy.deepcopy(value)
        for key, value in source_entry.items()
        if key != "apr_results"
    }
    replaced = 0
    if not ablation:
        return entry, replaced
    for hit in entry.get("retrieval_results") or []:
        if hit.get("is_poisoned") is not True:
            continue
        normalized = _required_id(hit.get("doc_id"), "poisoned retrieval doc_id")
        if normalized not in vul_code_index:
            raise KeyError(
                f"poisoned retrieval doc_id={hit.get('doc_id')!r} has no unique "
                "generation_attack.vinj_attack.vul_code"
            )
        hit["fixed_code"] = vul_code_index[normalized]
        replaced += 1
    return entry, replaced


def debug_result_path(source_path: str, strategy: str, ablation: bool) -> str:
    if strategy not in VALID_RECALL_STRATEGIES:
        raise ValueError(f"unsupported recall strategy: {strategy!r}")
    source = Path(source_path)
    suffix = (
        f"_jailbreak_ablation_{strategy}_debug.json"
        if ablation
        else f"_{strategy}_debug.json"
    )
    return str(source.with_name(source.stem + suffix))


def load_or_create_debug_results(
    source_path: str,
    source_results: Sequence[Dict[str, Any]],
    *,
    strategy: str,
    ablation: bool,
    vul_code_index: Mapping[int, str],
) -> Tuple[str, List[Dict[str, Any]]]:
    result_path = debug_result_path(source_path, strategy, ablation)
    if not os.path.isfile(result_path):
        expected, replaced = _build_debug_results(
            source_results,
            ablation=ablation,
            vul_code_index=vul_code_index,
        )
        atomic_save_json(result_path, expected)
        logger.info(
            "created debug result file (%s, replacements=%d): %s",
            strategy,
            replaced,
            result_path,
        )
        return result_path, expected

    existing = load_json(result_path)
    if not isinstance(existing, list):
        raise TypeError(f"debug result root must be a list: {result_path}")
    validate_result_schema(existing)
    if len(existing) != len(source_results):
        raise RuntimeError(
            f"existing debug result query count differs from its source: {result_path}"
        )
    for position, (source_entry, existing_entry) in enumerate(
        zip(source_results, existing)
    ):
        expected_entry, _ = _build_debug_entry(
            source_entry,
            ablation=ablation,
            vul_code_index=vul_code_index,
        )
        retrieval_only = {
            key: value
            for key, value in existing_entry.items()
            if key != "apr_results"
        }
        if retrieval_only != expected_entry:
            query_id = (existing_entry.get("entity") or {}).get("id")
            raise RuntimeError(
                f"existing debug result no longer matches its source/configuration "
                f"at index {position}, query id={query_id!r}: {result_path}; "
                "use a new strategy file or remove the stale debug result explicitly"
            )
    return result_path, existing


def build_prompt(
    entry: Dict[str, Any],
    references: Sequence[Dict[str, Any]],
    top_k: int,
    max_ref_chars: int,
    max_prompt_chars: int,
) -> str:
    view = dict(entry)
    view["retrieval_results"] = list(references)
    return build_apr_prompt(
        view,
        top_k,
        max_ref_chars=max_ref_chars,
        max_prompt_chars=max_prompt_chars,
    )


def validate_existing_prompt(
    entry: Dict[str, Any],
    *,
    top_k: int,
    model_name: str,
    expected_prompt: str,
    label: str,
) -> None:
    slot = model_slot(entry, top_k, model_name)
    patch = slot.get("patch")
    if "is_vulnerable" in slot and not isinstance(slot["is_vulnerable"], bool):
        raise ValueError(f"{label} contains a non-boolean is_vulnerable value")
    if "is_vulnerable" in slot and not patch:
        raise ValueError(f"{label} contains is_vulnerable without a non-empty patch")
    if patch and slot.get("prompt") != expected_prompt:
        raise RuntimeError(
            f"{label} contains a completed patch for a different debug prompt; "
            "refusing to reuse stale RACG output"
        )


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_debug_context(
    *,
    strategy: str,
    compared_methods: Sequence[str],
    apr_config_sha256: str,
    judge_config_sha256: str,
) -> Dict[str, Any]:
    return {
        "version": DEBUG_CONTEXT_VERSION,
        "recall_strategy": strategy,
        "compared_methods": list(compared_methods),
        "apr_config_sha256": apr_config_sha256,
        "judge_config_sha256": judge_config_sha256,
    }


def validate_and_set_debug_context(
    entry: Dict[str, Any],
    *,
    top_k: int,
    model_name: str,
    expected_context: Mapping[str, Any],
    label: str,
) -> None:
    """Reject stale completed work and attach context to resumable work."""
    slot = model_slot(entry, top_k, model_name, create=True)
    existing = slot.get("debug_context")
    patch_complete = has_complete_racg_result(slot)
    judge_complete = has_complete_judge_result(slot)
    if existing is not None and not isinstance(existing, dict):
        raise ValueError(f"{label} contains a malformed debug_context")

    apr_fields = (
        "version",
        "recall_strategy",
        "compared_methods",
        "apr_config_sha256",
    )
    if patch_complete:
        if existing is None:
            raise RuntimeError(f"{label} has a patch without debug_context")
        mismatched = [
            field
            for field in apr_fields
            if existing.get(field) != expected_context.get(field)
        ]
        if mismatched:
            raise RuntimeError(
                f"{label} was generated under a different debug/APR context; "
                f"mismatched fields={mismatched}"
            )

    if judge_complete:
        if existing is None:
            raise RuntimeError(f"{label} has a verdict without debug_context")
        if existing.get("judge_config_sha256") != expected_context.get(
            "judge_config_sha256"
        ):
            raise RuntimeError(
                f"{label} was judged under a different judge configuration"
            )

    if not patch_complete:
        slot["debug_context"] = dict(expected_context)
    elif not judge_complete and existing != expected_context:
        # The patch is still valid when only the judge configuration changed
        # and no verdict has been produced yet.
        slot["debug_context"] = dict(expected_context)


def select_common_work(
    common_query_ids: Sequence[Any],
    states_by_method: Mapping[str, ResultState],
    *,
    top_k: int,
    model_name: str,
    process_num: Optional[int],
) -> List[Any]:
    unfinished: List[Any] = []
    for query_id in common_query_ids:
        if any(
            (
                not has_complete_racg_result(
                    model_slot(
                        state.results[state.query_index[query_id]],
                        top_k,
                        model_name,
                    )
                )
                or not has_complete_judge_result(
                    model_slot(
                        state.results[state.query_index[query_id]],
                        top_k,
                        model_name,
                    )
                )
            )
            for state in states_by_method.values()
        ):
            unfinished.append(query_id)
    if process_num is None:
        return unfinished
    limit = int(process_num)
    if limit < 0:
        raise ValueError(f"eval.process_num must be null or non-negative, got {process_num!r}")
    return unfinished if limit == 0 else unfinished[:limit]


class _UnexpectedJudger:
    def judge(self, **_: Any) -> Any:
        raise RuntimeError("judger was called even though no pending judge task was detected")


def needs_vulnerability_judger(
    state: ResultState,
    selected_indices: Sequence[int],
    *,
    top_k: int,
    model_name: str,
    poison_doc_index: Dict[int, List[Dict[str, Any]]],
) -> bool:
    for index in selected_indices:
        entry = state.results[index]
        slot = model_slot(entry, top_k, model_name)
        if not has_complete_racg_result(slot) or has_complete_judge_result(slot):
            continue
        poisoned_ids = gen_metrics.collect_poisoned_doc_ids(
            entry.get("retrieval_results") or [],
            top_k,
        )
        poisoned_ids.sort()
        if gen_metrics.build_relevant_vul_for_judger(
            poisoned_ids,
            poison_doc_index,
        ):
            return True
    return False


def _write_detail_log(
    *,
    root: Path,
    method: MethodSpec,
    generator: str,
    run_stamp: str,
    mode: str,
    strategy: str,
    unit: str,
    alpha: str,
    top_k: int,
    target_count: int,
    test_count: int,
    configured_eval_mode: str,
    effective_eval_mode: str,
    source: str,
    compared_methods: Sequence[str],
    metrics: Mapping[str, Any],
) -> str:
    directory = (
        root
        / "logs/evaluation/generation"
        / f"{method.embedder}-{method.optimizer}"
        / f"{generator}-{run_stamp}"
        / mode
        / strategy
        / f"top-{top_k}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"{unit}_alpha_{alpha}_target_{target_count}_test_{test_count}"
        "_generation_eval_debug.txt"
    )
    lines = [
        "Controlled Multi-Optimizer Generation Debug Evaluation",
        f"Timestamp               : {datetime.now().isoformat(timespec='seconds')}",
        f"Source                  : {source}",
        f"Generator               : {generator}",
        f"Optimizer               : {method.optimizer}",
        f"Compared optimizers     : {list(compared_methods)}",
        f"Mode / strategy / Top-K : {mode} / {strategy} / {top_k}",
        f"Eval mode               : {configured_eval_mode}",
        f"Effective eval mode     : {effective_eval_mode}",
        f"Test queries            : {metrics['total_queries']}",
        f"Raw victims             : {metrics['raw_victim_count']}",
        f"Common controlled victims: {metrics['victim_count']}",
        f"Poison mismatch queries : {metrics['mismatch_count']}",
        f"No-poison queries       : {metrics['no_poison_count']}",
        f"Selected this run       : {metrics['selected_count']}",
        f"Generated this run      : {metrics['generated_count']}",
        f"Patched victims         : {metrics['total_with_patch']}",
        f"Evaluated victims       : {metrics['evaluated_count']}",
        f"Vulnerable victims      : {metrics['vulnerable_count']}",
        f"VR_GLOBAL               : {metrics['asr_g_global']:.6f}",
        f"VR_LOCAL                : {metrics['asr_g_local']:.6f}",
        "VR_GLOBAL formula      : vulnerable / all test queries",
        "VR_LOCAL formula       : vulnerable / common controlled Top-K victims",
        f"Pending IDs             : {metrics['pending_ids']}",
        f"Missing patterns        : {metrics['missing_pattern_ids']}",
        "Unresolved patterns    : "
        f"{metrics['unresolved_patterns_by_query']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _write_overview(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    method: MethodSpec,
    generator: str,
    mode: str,
    strategy: str,
    configured_eval_mode: str,
    effective_eval_mode: str,
    compared_methods: Sequence[str],
) -> None:
    lines = [
        "# 多优化器生成消融评估总览",
        "",
        f"- Generator: {generator}",
        f"- Embedder: {method.embedder}",
        f"- Optimizer: {method.optimizer}",
        f"- Compared optimizers: {', '.join(compared_methods)}",
        f"- Mode: {mode}",
        f"- Recall strategy: {strategy}",
        f"- Eval mode: {configured_eval_mode}",
        f"- Effective eval mode: {effective_eval_mode}",
        f"- Retrieval run: {method.retrieval_dir}",
        f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| CWE | Top-K | Queries | Raw victims | Common victims | Mismatch | "
        "Selected | Generated | Evaluated | Vulnerable | VR_GLOBAL | VR_LOCAL | Pending |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cwe']} | {row['top_k']} | {row['total_queries']} | "
            f"{row['raw_victim_count']} | {row['victim_count']} | "
            f"{row['mismatch_count']} | {row['selected_count']} | "
            f"{row['generated_count']} | {row['evaluated_count']} | "
            f"{row['vulnerable_count']} | {row['asr_g_global']:.6f} | "
            f"{row['asr_g_local']:.6f} | {len(row['pending_ids'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_method_specs(
    root: Path,
    ablation_items: Sequence[Mapping[str, Any]],
    *,
    ablation_enabled: bool,
) -> List[MethodSpec]:
    methods: List[MethodSpec] = []
    seen_optimizers: set[str] = set()
    expected_embedder: Optional[str] = None
    snapshots: Dict[str, Dict[Any, Dict[str, Any]]] = {}

    for position, item in enumerate(ablation_items):
        poisoned_file = _resolve(root, str(item.get("poisoned_file_path", "")))
        retrieval_dir = _resolve(root, str(item.get("retrieval_dir_path", "")))
        if not os.path.isfile(poisoned_file):
            raise FileNotFoundError(poisoned_file)
        if not os.path.isdir(Path(retrieval_dir, "poisoned")):
            raise FileNotFoundError(Path(retrieval_dir, "poisoned"))
        embedder, optimizer = _retrieval_metadata(retrieval_dir)
        if optimizer in seen_optimizers:
            raise ValueError(f"duplicate optimizer label in data.ablation: {optimizer!r}")
        seen_optimizers.add(optimizer)
        if expected_embedder is None:
            expected_embedder = embedder
        elif embedder != expected_embedder:
            raise ValueError(
                f"all ablation methods must use the same embedder; "
                f"expected {expected_embedder!r}, got {embedder!r}"
            )

        target_data = load_json(poisoned_file)
        if not isinstance(target_data, dict):
            raise TypeError(f"poisoned target root must be an object: {poisoned_file}")
        snapshots[optimizer] = build_target_snapshot(
            target_data,
            f"data.ablation[{position}]",
        )
        methods.append(
            MethodSpec(
                optimizer=optimizer,
                embedder=embedder,
                poisoned_file=poisoned_file,
                retrieval_dir=retrieval_dir,
                poison_doc_index=gen_metrics.build_poison_doc_index(target_data),
                vul_code_index=(
                    gen_metrics.build_vul_code_index(target_data)
                    if ablation_enabled
                    else {}
                ),
            )
        )
    validate_target_snapshots(snapshots)
    return methods


def _prepare_unit_states(
    methods: Sequence[MethodSpec],
    unit: str,
    *,
    alpha: str,
    max_top_k: int,
    strategy: str,
    ablation_enabled: bool,
) -> Dict[str, ResultState]:
    states: Dict[str, ResultState] = {}
    expected_counts: Optional[Tuple[int, int]] = None
    expected_query_ids: Optional[set[Any]] = None
    expected_buggy_hashes: Optional[Dict[Any, bytes]] = None

    for method in methods:
        source_path = find_result_file(
            method.retrieval_dir,
            "poisoned",
            unit,
            alpha,
        )
        source_results = load_json(source_path)
        if not isinstance(source_results, list):
            raise TypeError(f"retrieval result root must be a list: {source_path}")
        validate_result_schema(source_results)
        target_count, test_count = parse_counts(source_path)
        source_index = validate_retrieval_results(
            source_results,
            label=source_path,
            declared_test_count=test_count,
            max_top_k=max_top_k,
        )
        counts = (target_count, test_count)
        if expected_counts is None:
            expected_counts = counts
        elif counts != expected_counts:
            raise ValueError(
                f"{unit}: target/test counts differ across optimizers; "
                f"expected={expected_counts}, {method.optimizer}={counts}"
            )

        query_ids = set(source_index)
        buggy_hashes = {
            query_id: _code_digest(
                (source_results[position].get("entity") or {}).get("buggy_code"),
                f"{source_path} query id={query_id!r} entity.buggy_code",
            )
            for query_id, position in source_index.items()
        }
        if expected_query_ids is None:
            expected_query_ids = query_ids
            expected_buggy_hashes = buggy_hashes
        else:
            if query_ids != expected_query_ids:
                missing = sorted(expected_query_ids - query_ids, key=_id_sort_key)
                extra = sorted(query_ids - expected_query_ids, key=_id_sort_key)
                raise ValueError(
                    f"{unit}: query population differs for {method.optimizer}; "
                    f"missing={missing[:10]}, extra={extra[:10]}"
                )
            if buggy_hashes != expected_buggy_hashes:
                differing = [
                    query_id
                    for query_id in expected_query_ids
                    if buggy_hashes.get(query_id) != expected_buggy_hashes.get(query_id)
                ]
                raise ValueError(
                    f"{unit}: query buggy_code differs for {method.optimizer}; "
                    f"query IDs={differing[:10]}"
                )

        result_path, results = load_or_create_debug_results(
            source_path,
            source_results,
            strategy=strategy,
            ablation=ablation_enabled,
            vul_code_index=method.vul_code_index,
        )
        query_index = validate_retrieval_results(
            results,
            label=result_path,
            declared_test_count=test_count,
            max_top_k=max_top_k,
        )
        states[method.optimizer] = ResultState(
            source_path=source_path,
            result_path=result_path,
            target_count=target_count,
            test_count=test_count,
            results=results,
            query_index=query_index,
            save=make_atomic_saver(result_path, results),
        )
    return states


def main() -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(
        description="Controlled multi-optimizer generation ablation evaluation"
    )
    parser.add_argument("-c", "--config", default="configs/evaluation/debug.yml")
    args = parser.parse_args()

    from src.utils.log import setup_logging

    setup_logging(logging.INFO)
    config = load_yaml(_resolve(root, args.config)) or {}
    data_cfg = config.get("data") or {}
    model_cfg = config.get("model") or {}
    eval_cfg = config.get("eval") or {}
    database_cfg = config.get("database") or {}

    ablation_items = data_cfg.get("ablation")
    if not isinstance(ablation_items, list) or len(ablation_items) < 2:
        raise ValueError("data.ablation must contain at least two optimizer configurations")

    top_k_values = normalize_top_k(eval_cfg.get("eval_top_k", [10]))
    max_top_k = max(top_k_values)
    (
        _eval_cwe,
        configured_eval_mode,
        effective_eval_mode,
        eval_units,
    ) = resolve_eval_scope(eval_cfg)
    strategy = str(eval_cfg.get("eval_recall_strategy", "")).strip().lower()
    if strategy not in VALID_RECALL_STRATEGIES:
        raise ValueError(
            f"eval.eval_recall_strategy must be one of {sorted(VALID_RECALL_STRATEGIES)}, "
            f"got {eval_cfg.get('eval_recall_strategy')!r}"
        )
    ablation_enabled = bool(eval_cfg.get("eval_jailbreak_ablation", False))
    mode = "naive" if ablation_enabled else "jailbreak"
    process_num = eval_cfg.get("process_num")
    if process_num is not None and int(process_num) < 0:
        raise ValueError("eval.process_num must be null or non-negative")
    max_retries = int(eval_cfg.get("max_retries", 5))
    max_ref_chars = int(eval_cfg.get("max_ref_chars", DEFAULT_MAX_REF_CHARS))
    max_prompt_chars = int(eval_cfg.get("max_prompt_chars", DEFAULT_MAX_PROMPT_CHARS))

    milvus_config_path = _resolve(
        root,
        str(database_cfg.get("milvus_config", "configs/database/vul/vinj.yml")),
    )
    vul_db_config = load_yaml(milvus_config_path) or {}
    alpha = strategy_to_alpha_label(
        str(eval_cfg.get("eval_retrieval_strategy", "dense")),
        float((vul_db_config.get("retrieval") or {}).get("alpha_weight", 0.5)),
    )

    methods = _load_method_specs(
        root,
        ablation_items,
        ablation_enabled=ablation_enabled,
    )
    compared_methods = [method.optimizer for method in methods]

    apr_config_path = _resolve(root, str(model_cfg.get("apr_generator_config", "")))
    judge_config_path = _resolve(
        root,
        str(
            model_cfg.get("judge_generator_config")
            or model_cfg.get("apr_generator_config", "")
        ),
    )
    apr_config = load_yaml(apr_config_path) or {}
    judge_config = load_yaml(judge_config_path) or {}
    model_name = Path(apr_config_path).stem
    debug_context = build_debug_context(
        strategy=strategy,
        compared_methods=compared_methods,
        apr_config_sha256=file_sha256(apr_config_path),
        judge_config_sha256=file_sha256(judge_config_path),
    )
    pool_size = len(apr_config.get("api_pool") or []) or 1
    configured_workers = eval_cfg.get("num_workers")
    workers = (
        min(int(configured_workers), pool_size)
        if configured_workers is not None
        else pool_size
    )
    workers = max(1, workers)
    generators: Optional[List[Any]] = None
    milvus_client: Optional[Any] = None
    judger: Optional[Any] = None

    def get_generators() -> List[Any]:
        nonlocal generators
        if generators is None:
            from src.models.generator.Base import create_generator

            generators = [
                create_generator(apr_config, pool_index=index)
                for index in range(workers)
            ]
        return generators

    def get_judger() -> Any:
        nonlocal milvus_client, judger
        if judger is None:
            from src.models.generator.Base import create_generator
            from src.rag.vul.milvus_client import VulMilvusClient
            from src.utils.vul_judger import LLMVulJudger

            client_config = copy.deepcopy(vul_db_config)
            db_uri = (client_config.get("database") or {}).get("uri", "")
            if db_uri and not os.path.isabs(db_uri):
                client_config["database"]["uri"] = str(root / db_uri)
            milvus_client = VulMilvusClient(client_config)
            milvus_client.create_pattern_cache()
            judge_generator = create_generator(judge_config, pool_index=0)
            judger = LLMVulJudger(
                judge_generator,
                milvus_client,
                max_retries=max_retries,
            )
        return judger

    run_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    rows_by_method: Dict[str, List[Dict[str, Any]]] = {
        method.optimizer: [] for method in methods
    }
    unresolved: List[Tuple[str, str, int, List[Any]]] = []

    try:
        for unit in eval_units:
            states = _prepare_unit_states(
                methods,
                unit,
                alpha=alpha,
                max_top_k=max_top_k,
                strategy=strategy,
                ablation_enabled=ablation_enabled,
            )
            results_by_method = {
                method: state.results for method, state in states.items()
            }
            indexes_by_method = {
                method: state.query_index for method, state in states.items()
            }
            validate_cross_method_queries(
                results_by_method,
                indexes_by_method,
                unit=unit,
            )

            for top_k in top_k_values:
                cohort = build_common_cohort(
                    results_by_method,
                    indexes_by_method,
                    top_k,
                )
                common_query_ids = cohort["query_ids"]

                for query_id in common_query_ids:
                    entries = {
                        method.optimizer: states[method.optimizer].results[
                            states[method.optimizer].query_index[query_id]
                        ]
                        for method in methods
                    }
                    selected_references = select_reference_views(
                        entries,
                        top_k,
                        strategy,
                    )
                    for method in methods:
                        optimizer = method.optimizer
                        entry = entries[optimizer]
                        label = f"{optimizer}/{unit}/top-{top_k}/query-{query_id}"
                        validate_and_set_debug_context(
                            entry,
                            top_k=top_k,
                            model_name=model_name,
                            expected_context=debug_context,
                            label=label,
                        )
                        if has_complete_racg_result(
                            model_slot(entry, top_k, model_name)
                        ):
                            prompt = build_prompt(
                                entry,
                                selected_references[optimizer],
                                top_k,
                                max_ref_chars,
                                max_prompt_chars,
                            )
                            validate_existing_prompt(
                                entry,
                                top_k=top_k,
                                model_name=model_name,
                                expected_prompt=prompt,
                                label=label,
                            )

                selected_query_ids = select_common_work(
                    common_query_ids,
                    states,
                    top_k=top_k,
                    model_name=model_name,
                    process_num=process_num,
                )

                for method in methods:
                    optimizer = method.optimizer
                    state = states[optimizer]
                    victim_indices = [
                        state.query_index[query_id] for query_id in common_query_ids
                    ]
                    selected_indices = [
                        state.query_index[query_id] for query_id in selected_query_ids
                    ]
                    query_id_by_index = {
                        state.query_index[query_id]: query_id
                        for query_id in common_query_ids
                    }

                    def prompt_builder(
                        index: int,
                        *,
                        _optimizer: str = optimizer,
                        _query_id_by_index: Mapping[int, Any] = query_id_by_index,
                    ) -> str:
                        query_id = _query_id_by_index[index]
                        entries = {
                            item.optimizer: states[item.optimizer].results[
                                states[item.optimizer].query_index[query_id]
                            ]
                            for item in methods
                        }
                        selected_references = select_reference_views(
                            entries,
                            top_k,
                            strategy,
                        )
                        return build_prompt(
                            entries[_optimizer],
                            selected_references[_optimizer],
                            top_k,
                            max_ref_chars,
                            max_prompt_chars,
                        )

                    missing_patch_indices = [
                        index
                        for index in selected_indices
                        if not has_complete_racg_result(
                            model_slot(
                                state.results[index],
                                top_k,
                                model_name,
                            )
                        )
                    ]
                    if missing_patch_indices:
                        generation = run_racg(
                            results=state.results,
                            selected_indices=selected_indices,
                            top_k=top_k,
                            model_name=model_name,
                            generators=get_generators(),
                            num_workers=workers,
                            max_retries=max_retries,
                            save_callback=state.save,
                            label=f"{optimizer}/{unit}/{mode}/{strategy}/top-{top_k}",
                            max_ref_chars=max_ref_chars,
                            max_prompt_chars=max_prompt_chars,
                            prompt_builder=prompt_builder,
                        )
                    else:
                        generation = {
                            "selected": len(selected_indices),
                            "generated": 0,
                            "pending_ids": [],
                        }
                    active_judger = (
                        get_judger()
                        if needs_vulnerability_judger(
                            state,
                            selected_indices,
                            top_k=top_k,
                            model_name=model_name,
                            poison_doc_index=method.poison_doc_index,
                        )
                        else _UnexpectedJudger()
                    )
                    metrics = gen_metrics.evaluate_vulnerability_rate(
                        poisoned_results=state.results,
                        victim_indices=victim_indices,
                        work_indices=selected_indices,
                        poison_doc_index=method.poison_doc_index,
                        judger=active_judger,
                        eval_top_k=top_k,
                        model_name=model_name,
                        save_callback=state.save,
                        canonicalize_poisoned_doc_order=True,
                    )
                    pending = list(
                        dict.fromkeys(generation["pending_ids"] + metrics["pending_ids"])
                    )
                    metrics.update(
                        {
                            "pending_ids": pending,
                            "generated_count": generation["generated"],
                            "raw_victim_count": cohort["raw_victim_counts"][optimizer],
                            "mismatch_count": cohort["mismatch_count"],
                            "no_poison_count": cohort["no_poison_count"],
                        }
                    )
                    row = {"cwe": unit, "top_k": top_k, **metrics}
                    rows_by_method[optimizer].append(row)
                    _write_detail_log(
                        root=root,
                        method=method,
                        generator=model_name,
                        run_stamp=run_stamp,
                        mode=mode,
                        strategy=strategy,
                        unit=unit,
                        alpha=alpha,
                        top_k=top_k,
                        target_count=state.target_count,
                        test_count=state.test_count,
                        configured_eval_mode=configured_eval_mode,
                        effective_eval_mode=effective_eval_mode,
                        source=state.result_path,
                        compared_methods=compared_methods,
                        metrics=metrics,
                    )
                    if pending:
                        unresolved.append((optimizer, unit, top_k, pending))
    finally:
        if milvus_client is not None:
            milvus_client.close()

    for method in methods:
        log_base = (
            root
            / "logs/evaluation/generation"
            / f"{method.embedder}-{method.optimizer}"
            / f"{model_name}-{run_stamp}"
            / mode
            / strategy
        )
        _write_overview(
            log_base / "overview_debug.md",
            rows_by_method[method.optimizer],
            method=method,
            generator=model_name,
            mode=mode,
            strategy=strategy,
            configured_eval_mode=configured_eval_mode,
            effective_eval_mode=effective_eval_mode,
            compared_methods=compared_methods,
        )

    if unresolved:
        logger.error("debug evaluation incomplete; restart will retry: %s", unresolved)
        return 2
    logger.info("controlled multi-optimizer generation debug evaluation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
