#!/usr/bin/env python3
"""Victim-only generation attack evaluation (RACG + VR/ASR-G)."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.evaluation.generation import metrics as gen_metrics
from src.evaluation.generation.persistence import (
    DEFAULT_RESULT_LOCK_TIMEOUT,
    ModelSlotSaver,
    ResultPersistenceError,
    load_or_create_result,
    model_run_locks,
    read_result_json,
    resolve_lock_timeout,
)
from src.evaluation.generation.racg import (
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_MAX_REF_CHARS,
    clear_generation_results,
    has_complete_judge_result,
    has_complete_racg_result,
    model_slot,
    normalize_top_k,
    run_racg,
    validate_result_schema,
    victim_indices,
)

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    root = Path(os.path.abspath("")).resolve()
    while root != root.parent and not (root / "configs").exists():
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _resolve(root: Path, value: str) -> str:
    return value if os.path.isabs(value) else str((root / value).resolve())


def strategy_to_alpha_label(strategy: str, alpha_weight: float) -> str:
    strategy = str(strategy).lower()
    if strategy == "dense":
        return "1"
    if strategy == "sparse":
        return "0"
    if strategy == "hybrid":
        return str(alpha_weight)
    raise ValueError(f"unsupported eval_retrieval_strategy={strategy!r}")


VALID_EVAL_MODES = {"per_cwe", "mixed", "both"}


def resolve_clear_options(eval_cfg: Dict[str, Any]) -> Tuple[bool, bool]:
    """Validate result-clearing options applied at every process start."""
    if "clear_judge_resultes" in eval_cfg:
        raise ValueError(
            "eval.clear_judge_resultes is misspelled; "
            "use eval.clear_judge_results"
        )

    values: Dict[str, bool] = {}
    for key in ("clear_racg_results", "clear_judge_results"):
        value = eval_cfg.get(key, False)
        if not isinstance(value, bool):
            raise TypeError(f"eval.{key} must be a boolean, got {value!r}")
        values[key] = value
    return values["clear_racg_results"], values["clear_judge_results"]


def resolve_worker_count(
    configured: Any,
    pool_size: int,
    config_key: str,
) -> int:
    """Resolve a positive worker count capped by the matching API pool."""
    if pool_size <= 0:
        raise ValueError(f"{config_key} API pool size must be positive")
    if configured is None:
        return pool_size
    if isinstance(configured, bool) or not isinstance(configured, int):
        raise TypeError(f"eval.{config_key} must be null or a positive integer")
    if configured <= 0:
        raise ValueError(f"eval.{config_key} must be positive, got {configured!r}")
    return min(configured, pool_size)


def resolve_eval_scope(
    eval_cfg: Dict[str, Any],
) -> Tuple[List[str], str, str, List[str]]:
    """Validate the three-state CWE evaluation contract.

    Returns ``(eval_cwe, configured_mode, effective_mode, eval_units)``.
    A source dataset whose only top-level group is ``CWE-MIXED`` cannot be
    split back into per-CWE units, so per_cwe/both intentionally collapse to
    one mixed unit while preserving the configured mode for logging.
    """
    if "eval_isolated" in eval_cfg:
        raise ValueError(
            "eval.eval_isolated is no longer supported; "
            "use eval.eval_mode: per_cwe | mixed | both"
        )

    raw_cwe = eval_cfg.get("eval_cwe")
    if not isinstance(raw_cwe, (list, tuple)) or isinstance(raw_cwe, (str, bytes)):
        raise TypeError("eval.eval_cwe must be a non-empty list of CWE strings")
    if not raw_cwe:
        raise ValueError("eval.eval_cwe cannot be empty")
    if any(not isinstance(cwe, str) or not cwe.strip() for cwe in raw_cwe):
        raise TypeError("every eval.eval_cwe item must be a non-empty string")
    eval_cwe = list(dict.fromkeys(cwe.strip() for cwe in raw_cwe))

    configured_mode = str(eval_cfg.get("eval_mode", "")).strip().lower()
    if configured_mode not in VALID_EVAL_MODES:
        raise ValueError(
            "eval.eval_mode must be one of per_cwe, mixed, both; "
            f"got {eval_cfg.get('eval_mode')!r}"
        )

    if "CWE-MIXED" in eval_cwe and eval_cwe != ["CWE-MIXED"]:
        raise ValueError(
            "eval.eval_cwe cannot mix CWE-MIXED with ordinary CWE groups"
        )

    effective_mode = (
        "mixed" if eval_cwe == ["CWE-MIXED"] else configured_mode
    )
    return (
        eval_cwe,
        configured_mode,
        effective_mode,
        build_eval_units(eval_cwe, effective_mode),
    )


def build_eval_units(eval_cwe: Sequence[str], eval_mode: str) -> List[str]:
    """Build ordered result-file units for a validated three-state mode."""
    cwe_units = list(dict.fromkeys(eval_cwe))
    if eval_mode == "per_cwe":
        return cwe_units
    if eval_mode == "mixed":
        return ["CWE-MIXED"]
    if eval_mode == "both":
        return cwe_units + ([] if "CWE-MIXED" in cwe_units else ["CWE-MIXED"])
    raise ValueError(
        f"eval_mode must be one of {sorted(VALID_EVAL_MODES)}, got {eval_mode!r}"
    )


def select_unfinished_victims(
    results: Sequence[Dict[str, Any]],
    all_victims: Sequence[int],
    top_k: int,
    model_name: str,
    process_num: Optional[int],
) -> List[int]:
    """Select only unfinished victims, then apply the per-run work limit."""
    unfinished = [
        idx
        for idx in all_victims
        if (
            not has_complete_racg_result(
                model_slot(results[idx], top_k, model_name)
            )
            or not has_complete_judge_result(
                model_slot(results[idx], top_k, model_name)
            )
        )
    ]
    if process_num is None:
        return unfinished
    limit = int(process_num)
    if limit < 0:
        raise ValueError(f"eval.process_num must be null or non-negative, got {process_num!r}")
    return unfinished[:limit] if limit > 0 else unfinished


def validate_query_population(
    results: Sequence[Dict[str, Any]],
    label: str,
) -> None:
    """Reject missing or duplicate normalized query IDs before VR counting."""
    from src.utils.io import normalize_bfp_record_id

    seen: Dict[Any, int] = {}
    for idx, entry in enumerate(results):
        query_id = normalize_bfp_record_id((entry.get("entity") or {}).get("id"))
        if query_id is None:
            raise ValueError(f"{label} contains a query with no entity.id at index {idx}")
        if query_id in seen:
            raise ValueError(
                f"{label} contains duplicate query id={query_id!r} at indices "
                f"{seen[query_id]} and {idx}"
            )
        seen[query_id] = idx


def parse_counts(path: str) -> Tuple[int, int]:
    match = re.search(r"_target_(\d+)_test_(\d+)_", Path(path).name)
    if not match:
        raise ValueError(f"new-contract result filename expected, got {Path(path).name}")
    return int(match.group(1)), int(match.group(2))


def find_result_file(root: str, label: str, unit: str, alpha_label: str) -> str:
    directory = Path(root, label)
    pattern = f"{unit}_alpha_{alpha_label}_target_*_test_*_retrieval_results.json"
    matches = sorted(directory.glob(pattern))
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        raise RuntimeError(f"multiple new-contract result files match {directory / pattern}")
    legacy = list(directory.glob(f"{unit}_alpha_{alpha_label}_tartget_*_retrieval_results.json"))
    legacy += list(Path(root, "baseline").glob(f"{unit}_alpha_{alpha_label}_*_retrieval_results.json"))
    if legacy:
        raise RuntimeError(
            f"legacy retrieval results detected for {unit}; rerun distributed_io Section 3 "
            "and retrieval evaluation under the new contract"
        )
    raise FileNotFoundError(directory / pattern)


def build_ablation_results(
    poisoned_results: List[Dict[str, Any]],
    vul_code_index: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], int]:
    """Create a retrieval-only no-jailbreak copy and replace every poisoned hit."""
    ablation = copy.deepcopy(poisoned_results)
    replaced = 0
    for entry in ablation:
        entry.pop("apr_results", None)
        for hit in entry.get("retrieval_results") or []:
            if hit.get("is_poisoned") is not True:
                continue
            doc_id = gen_metrics._resolve_doc_id(hit.get("doc_id"))
            if doc_id is None or doc_id not in vul_code_index:
                raise KeyError(
                    f"poisoned retrieval doc_id={hit.get('doc_id')!r} has no unique vinj vul_code"
                )
            hit["fixed_code"] = vul_code_index[doc_id]
            replaced += 1
    return ablation, replaced


def _ablation_path(poisoned_path: str) -> str:
    path = Path(poisoned_path)
    return str(path.with_name(path.stem + "_jailbreak_ablation.json"))


def _load_or_create_ablation(
    poisoned_path: str,
    poisoned_results: List[Dict[str, Any]],
    vul_code_index: Dict[int, str],
    lock_timeout: float = DEFAULT_RESULT_LOCK_TIMEOUT,
) -> Tuple[str, List[Dict[str, Any]]]:
    path = _ablation_path(poisoned_path)
    results, created = load_or_create_result(
        path, lambda: build_ablation_results(poisoned_results, vul_code_index)[0], lock_timeout
    )
    if created:
        replaced = sum(
            hit.get("is_poisoned") is True
            for entry in results for hit in entry["retrieval_results"]
        )
        logger.info("created jailbreak ablation file with %d replacements: %s", replaced, path)
    return path, results


def _retrieval_metadata(retrieval_dir: str) -> Tuple[str, str]:
    path = Path(retrieval_dir)
    embedder = path.parent.name or "unknown"
    match = re.match(r"^(.*)-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$", path.name)
    if not match:
        raise ValueError(
            "retrieval_dir_path must point to a timestamped run directory "
            "named {optimizer}-%Y-%m-%d-%H-%M"
        )
    return embedder, match.group(1)


def _write_detail_log(
    *,
    root: Path,
    embedder: str,
    optimizer: str,
    generator: str,
    run_stamp: str,
    mode: str,
    unit: str,
    alpha: str,
    top_k: int,
    target_count: int,
    test_count: int,
    configured_eval_mode: str,
    effective_eval_mode: str,
    source: str,
    metrics: Dict[str, Any],
) -> str:
    directory = (
        root
        / "logs/evaluation/generation"
        / f"{embedder}-{optimizer}"
        / f"{generator}-{run_stamp}"
        / mode
        / f"top-{top_k}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"{unit}_alpha_{alpha}_target_{target_count}_test_{test_count}_generation_eval.txt"
    )
    lines = [
        "Generation Attack Evaluation (VR / ASR-G)",
        f"Timestamp       : {datetime.now().isoformat(timespec='seconds')}",
        f"Source          : {source}",
        f"Model           : {generator}",
        f"Mode / Top-K    : {mode} / {top_k}",
        f"Eval mode       : {configured_eval_mode}",
        f"Effective mode  : {effective_eval_mode}",
        f"Test queries    : {metrics['total_queries']}",
        f"Victims         : {metrics['victim_count']}",
        f"Selected this run: {metrics['selected_count']}",
        f"Patched victims : {metrics['total_with_patch']}",
        f"Evaluated       : {metrics['evaluated_count']}",
        f"Vulnerable      : {metrics['vulnerable_count']}",
        f"VR_GLOBAL       : {metrics['asr_g_global']:.6f}",
        f"VR_LOCAL        : {metrics['asr_g_local']:.6f}",
        "VR_GLOBAL formula: vulnerable / all test queries",
        "VR_LOCAL formula : vulnerable / Top-K victim queries",
        f"Pending IDs     : {metrics['pending_ids']}",
        f"Missing patterns: {metrics['missing_pattern_ids']}",
        "Unresolved pattern IDs by pending query: "
        f"{metrics['unresolved_patterns_by_query']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _write_overview(path: Path, rows: Sequence[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    lines = [
        "# 生成攻击评估总览",
        "",
        f"- Generator: {metadata['generator']}",
        f"- Mode: {metadata['mode']}",
        f"- Eval mode: {metadata['eval_mode']}",
        f"- Effective eval mode: {metadata['effective_eval_mode']}",
        f"- Retrieval run: {metadata['retrieval_dir']}",
        f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| CWE | Top-K | Queries | Victims | Selected | Evaluated | Vulnerable | VR_GLOBAL | VR_LOCAL | Pending |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cwe']} | {row['top_k']} | {row['total_queries']} | "
            f"{row['victim_count']} | {row['selected_count']} | "
            f"{row['evaluated_count']} | {row['vulnerable_count']} | "
            f"{row['asr_g_global']:.6f} | {row['asr_g_local']:.6f} | "
            f"{len(row['pending_ids'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_evaluation(result_locks: ExitStack) -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Victim-only generation attack evaluation")
    parser.add_argument("-c", "--config", default="configs/evaluation/generation.yml")
    args = parser.parse_args()

    from src.models.generator.Base import create_generator
    from src.rag.vul.milvus_client import VulMilvusClient
    from src.utils.io import load_json, load_yaml
    from src.utils.log import setup_logging
    from src.utils.vul_judger import LLMVulJudger

    setup_logging(logging.INFO)
    config = load_yaml(_resolve(root, args.config)) or {}
    data_cfg = config.get("data") or {}
    model_cfg = config.get("model") or {}
    eval_cfg = config.get("eval") or {}
    database_cfg = config.get("database") or {}

    poisoned_file = _resolve(root, str(data_cfg.get("poisoned_file_path", "")))
    retrieval_dir = _resolve(root, str(data_cfg.get("retrieval_dir_path", "")))
    if not os.path.isfile(poisoned_file):
        raise FileNotFoundError(poisoned_file)
    embedder, optimizer = _retrieval_metadata(retrieval_dir)

    eval_top_k = normalize_top_k(eval_cfg.get("eval_top_k", [10]))
    (
        eval_cwe,
        configured_eval_mode,
        effective_eval_mode,
        eval_units,
    ) = resolve_eval_scope(eval_cfg)
    if configured_eval_mode != effective_eval_mode:
        logger.warning(
            "eval_cwe=%s is already mixed; eval_mode=%s collapses to effective mode=mixed",
            eval_cwe,
            configured_eval_mode,
        )
    logger.info(
        "evaluation scope: configured=%s effective=%s units=%s",
        configured_eval_mode,
        effective_eval_mode,
        eval_units,
    )
    process_num = eval_cfg.get("process_num")
    max_retries = int(eval_cfg.get("max_retries", 5))
    max_ref_chars = int(eval_cfg.get("max_ref_chars", DEFAULT_MAX_REF_CHARS))
    max_prompt_chars = int(eval_cfg.get("max_prompt_chars", DEFAULT_MAX_PROMPT_CHARS))
    clear_racg_results, clear_judge_results = resolve_clear_options(eval_cfg)
    result_lock_timeout = resolve_lock_timeout(
        eval_cfg.get("result_lock_timeout", DEFAULT_RESULT_LOCK_TIMEOUT)
    )
    if clear_racg_results and clear_judge_results:
        logger.info(
            "both result-clearing flags are enabled; "
            "clear_racg_results subsumes clear_judge_results"
        )

    milvus_config_path = _resolve(
        root, str(database_cfg.get("milvus_config", "configs/database/vul/vinj.yml"))
    )
    vul_db_config = load_yaml(milvus_config_path) or {}
    alpha = strategy_to_alpha_label(
        str(eval_cfg.get("eval_retrieval_strategy", "dense")),
        float((vul_db_config.get("retrieval") or {}).get("alpha_weight", 0.5)),
    )

    apr_config_path = _resolve(root, str(model_cfg.get("apr_generator_config", "")))
    judge_config_path = _resolve(
        root, str(model_cfg.get("judge_generator_config") or model_cfg.get("apr_generator_config", ""))
    )
    apr_config = load_yaml(apr_config_path) or {}
    judge_config = load_yaml(judge_config_path) or {}
    model_name = Path(apr_config_path).stem
    apr_pool_size = len(apr_config.get("api_pool") or []) or 1
    judge_pool_size = len(judge_config.get("api_pool") or []) or 1
    apr_workers = resolve_worker_count(
        eval_cfg.get("num_workers"),
        apr_pool_size,
        "num_workers",
    )
    judge_workers = resolve_worker_count(
        eval_cfg.get("judge_num_workers"),
        judge_pool_size,
        "judge_num_workers",
    )
    logger.info(
        "APR worker pool: configured=%r pool_size=%d maximum=%d",
        eval_cfg.get("num_workers"),
        apr_pool_size,
        apr_workers,
    )
    logger.info(
        "Judge worker pool: configured=%r pool_size=%d maximum=%d",
        eval_cfg.get("judge_num_workers"),
        judge_pool_size,
        judge_workers,
    )

    target_data = load_json(poisoned_file)
    poison_doc_index = gen_metrics.build_poison_doc_index(target_data)
    ablation_enabled = bool(eval_cfg.get("eval_jailbreak_ablation", False))
    vul_code_index = gen_metrics.build_vul_code_index(target_data) if ablation_enabled else {}
    mode = "naive" if ablation_enabled else "jailbreak"
    run_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_base = (
        root
        / "logs/evaluation/generation"
        / f"{embedder}-{optimizer}"
        / f"{model_name}-{run_stamp}"
        / mode
    )

    # Acquire all leases before any result initialization, clearing, or API use.
    source_paths = {
        unit: find_result_file(retrieval_dir, "poisoned", unit, alpha)
        for unit in eval_units
    }
    result_paths = [
        _ablation_path(path) if ablation_enabled else path
        for path in source_paths.values()
    ]
    result_locks.enter_context(model_run_locks(result_paths, model_name))
    logger.info(
        "result persistence: model=%s files=%d lock_timeout=%.1fs; locked read/merge/write enabled",
        model_name, len(result_paths), result_lock_timeout,
    )

    contexts: List[Dict[str, Any]] = []
    for unit, source_path in source_paths.items():
        source_results = read_result_json(source_path, result_lock_timeout)
        validate_result_schema(source_results)
        target_count, test_count = parse_counts(source_path)
        if ablation_enabled:
            result_path, results = _load_or_create_ablation(
                source_path, source_results, vul_code_index, result_lock_timeout
            )
        else:
            result_path, results = source_path, source_results
        validate_query_population(results, Path(result_path).name)
        if len(results) != test_count:
            raise ValueError(
                f"{Path(result_path).name} declares test_{test_count} but contains "
                f"{len(results)} query results"
            )
        saver = ModelSlotSaver(result_path, results, model_name, eval_top_k, result_lock_timeout)

        if clear_racg_results or clear_judge_results:
            file_changed = False
            for top_k in eval_top_k:
                clear_summary = clear_generation_results(
                    results=results,
                    selected_indices=victim_indices(results, top_k),
                    top_k=top_k,
                    model_name=model_name,
                    clear_racg_results=clear_racg_results,
                    clear_judge_results=clear_judge_results,
                )
                file_changed = (
                    file_changed or clear_summary["changed_queries"] > 0
                )
                logger.info(
                    "[clear][%s][%s][top-%d] considered=%d changed=%d "
                    "fields=%s",
                    unit,
                    model_name,
                    top_k,
                    clear_summary["considered_queries"],
                    clear_summary["changed_queries"],
                    clear_summary["field_counts"],
                )
            if file_changed:
                saver()

        for top_k in eval_top_k:
            all_victims = victim_indices(results, top_k)
            # process_num limits only this invocation's unfinished work.
            # The complete victim population remains the VR_LOCAL
            # denominator, and completed prefixes are skipped on restart.
            selected = select_unfinished_victims(
                results, all_victims, top_k, model_name, process_num
            )
            missing_patch_indices = [
                idx
                for idx in selected
                if not has_complete_racg_result(
                    model_slot(results[idx], top_k, model_name)
                )
            ]
            contexts.append(
                {
                    "unit": unit,
                    "top_k": top_k,
                    "target_count": target_count,
                    "test_count": test_count,
                    "result_path": result_path,
                    "results": results,
                    "saver": saver,
                    "all_victims": all_victims,
                    "selected": selected,
                    "missing_patch_indices": missing_patch_indices,
                }
            )

    # Phase 1: finish every selected RACG task before any Judge call starts.
    racg_task_count = sum(
        len(context["missing_patch_indices"]) for context in contexts
    )
    max_context_racg_tasks = max(
        (len(context["missing_patch_indices"]) for context in contexts),
        default=0,
    )
    active_apr_workers = min(apr_workers, max_context_racg_tasks)
    logger.info(
        "[phase 1/2][RACG] contexts=%d tasks=%d workers=%d",
        len(contexts),
        racg_task_count,
        active_apr_workers,
    )
    generators = (
        [
            create_generator(apr_config, pool_index=i)
            for i in range(active_apr_workers)
        ]
        if active_apr_workers
        else []
    )
    generated_count = 0
    racg_pending_count = 0
    for context in contexts:
        if context["missing_patch_indices"]:
            generation = run_racg(
                results=context["results"],
                selected_indices=context["selected"],
                top_k=context["top_k"],
                model_name=model_name,
                generators=generators,
                num_workers=active_apr_workers,
                max_retries=max_retries,
                save_callback=context["saver"],
                label=f"{context['unit']}/{mode}/top-{context['top_k']}",
                max_ref_chars=max_ref_chars,
                max_prompt_chars=max_prompt_chars,
            )
        else:
            generation = {
                "selected": len(context["selected"]),
                "generated": 0,
                "pending_ids": [],
            }
        context["generation"] = generation
        generated_count += generation["generated"]
        racg_pending_count += len(generation["pending_ids"])
    logger.info(
        "[phase 1/2][RACG] completed generated=%d pending=%d",
        generated_count,
        racg_pending_count,
    )

    # Phase 2: rescan all selected slots, aggregate complete patches, and judge
    # them concurrently with a dedicated Judge generator pool.
    judge_tasks: List[gen_metrics.VulnerabilityJudgeTask] = []
    for context in contexts:
        tasks = gen_metrics.collect_vulnerability_judge_tasks(
            poisoned_results=context["results"],
            victim_indices=context["all_victims"],
            work_indices=context["selected"],
            poison_doc_index=poison_doc_index,
            eval_top_k=context["top_k"],
            model_name=model_name,
            save_callback=context["saver"],
        )
        context["judge_task_count"] = len(tasks)
        judge_tasks.extend(tasks)

    active_judge_workers = min(judge_workers, len(judge_tasks)) if judge_tasks else 0
    logger.info(
        "[phase 2/2][Judge] tasks=%d pool_size=%d workers=%d",
        len(judge_tasks),
        judge_pool_size,
        active_judge_workers,
    )
    judge_run = {
        "submitted": 0,
        "completed": 0,
        "finalized": 0,
        "failed_ids": [],
    }
    if judge_tasks:
        db_uri = (vul_db_config.get("database") or {}).get("uri", "")
        if db_uri and not os.path.isabs(db_uri):
            vul_db_config["database"]["uri"] = str(root / db_uri)
        milvus_client = VulMilvusClient(vul_db_config)
        milvus_client.create_pattern_cache()
        try:
            judge_generators = [
                create_generator(judge_config, pool_index=i)
                for i in range(active_judge_workers)
            ]
            judgers = [
                LLMVulJudger(
                    judge_generator,
                    milvus_client,
                    max_retries=max_retries,
                )
                for judge_generator in judge_generators
            ]
            judge_run = gen_metrics.run_parallel_vulnerability_judging(
                tasks=judge_tasks,
                judgers=judgers,
                num_workers=active_judge_workers,
            )
        finally:
            milvus_client.close()
    logger.info(
        "[phase 2/2][Judge] completed=%d finalized=%d failed=%d",
        judge_run["completed"],
        judge_run["finalized"],
        len(judge_run["failed_ids"]),
    )

    rows: List[Dict[str, Any]] = []
    unresolved: List[Tuple[str, int, List[Any]]] = []
    for context in contexts:
        metrics = gen_metrics.summarize_vulnerability_rate(
            poisoned_results=context["results"],
            victim_indices=context["all_victims"],
            work_indices=context["selected"],
            poison_doc_index=poison_doc_index,
            eval_top_k=context["top_k"],
            model_name=model_name,
        )
        pending = list(
            dict.fromkeys(
                context["generation"]["pending_ids"] + metrics["pending_ids"]
            )
        )
        metrics["pending_ids"] = pending
        row = {"cwe": context["unit"], "top_k": context["top_k"], **metrics}
        rows.append(row)
        _write_detail_log(
            root=root,
            embedder=embedder,
            optimizer=optimizer,
            generator=model_name,
            run_stamp=run_stamp,
            mode=mode,
            unit=context["unit"],
            alpha=alpha,
            top_k=context["top_k"],
            target_count=context["target_count"],
            test_count=context["test_count"],
            configured_eval_mode=configured_eval_mode,
            effective_eval_mode=effective_eval_mode,
            source=context["result_path"],
            metrics=metrics,
        )
        if pending:
            unresolved.append((context["unit"], context["top_k"], pending))

    _write_overview(
        log_base / "overview.md",
        rows,
        {
            "generator": model_name,
            "mode": mode,
            "eval_mode": configured_eval_mode,
            "effective_eval_mode": effective_eval_mode,
            "retrieval_dir": retrieval_dir,
        },
    )
    if unresolved:
        logger.error("evaluation incomplete; restart will retry: %s", unresolved)
        return 2
    logger.info("generation attack evaluation complete")
    return 0


def main() -> int:
    # ExitStack releases every model lease on return, exceptions, and Ctrl-C.
    try:
        with ExitStack() as result_locks:
            return _run_evaluation(result_locks)
    except ResultPersistenceError as exc:
        logger.error("generation result persistence failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
