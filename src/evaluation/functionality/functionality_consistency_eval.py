#!/usr/bin/env python3
"""Victim-only CrystalBLEU functionality consistency evaluation."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import threading
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.evaluation.functionality import metrics as func_metrics
from src.evaluation.functionality.persistence import (
    FunctionalityCheckpoint,
    clear_checkpoint_top_k,
)
from src.evaluation.generation.generation_attack_eval import (
    _project_root,
    _resolve,
    _retrieval_metadata,
    find_result_file,
    parse_counts,
    resolve_eval_scope,
    strategy_to_alpha_label,
)
from src.evaluation.generation.racg import (
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_MAX_REF_CHARS,
    has_complete_racg_result,
    model_slot,
    normalize_top_k,
    run_racg,
    validate_result_schema,
    victim_indices,
)
from src.evaluation.generation.persistence import (
    DEFAULT_RESULT_LOCK_TIMEOUT,
    ModelSlotSaver,
    ResultPersistenceError,
    load_or_create_result,
    model_run_locks,
    read_result_json,
    resolve_lock_timeout,
    result_file_lock,
)
from src.utils.io import normalize_bfp_record_id

logger = logging.getLogger(__name__)

VALID_CLEAN_RETRIEVAL_STRATEGIES = {"dense", "sparse", "hybrid"}


def _index_results(results: Sequence[Dict[str, Any]], label: str) -> Dict[Any, int]:
    index: Dict[Any, int] = {}
    for i, entry in enumerate(results):
        query_id = normalize_bfp_record_id((entry.get("entity") or {}).get("id"))
        if query_id is None:
            raise ValueError(f"{label} result contains a query with no entity.id")
        if query_id in index:
            raise ValueError(f"{label} result contains duplicate query id={query_id!r}")
        index[query_id] = i
    return index


def _load_corpus(path: str, field: str) -> List[str]:
    from src.utils.io import load_json, load_jsonl

    if path.lower().endswith(".jsonl"):
        records = load_jsonl(path)
    else:
        raw = load_json(path)
        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            records = [item for group in raw.values() for item in (group or [])]
        else:
            raise ValueError(f"unsupported CrystalBLEU corpus root: {type(raw).__name__}")
    values: List[str] = []
    for record in records:
        value = record.get(field, "")
        if not value and isinstance(record.get("entity"), dict):
            value = record["entity"].get(field, "")
        if value:
            values.append(str(value))
    return values


def _selected_poison_path(original: str, ablation: bool) -> str:
    if not ablation:
        return original
    path = str(Path(original).with_name(Path(original).stem + "_jailbreak_ablation.json"))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"jailbreak ablation result is missing: {path}; run generation evaluation "
            "with eval_jailbreak_ablation=true first"
        )
    return path


def _clean_result_path(
    retrieval_dir: str,
    unit: str,
    alpha: str,
    test_count: int,
) -> Path:
    """Return the unique clean-baseline result path for one evaluation unit."""
    directory = Path(retrieval_dir, "clean")
    expected = directory / (
        f"{unit}_alpha_{alpha}_target_0_test_{test_count}_retrieval_results.json"
    )
    pattern = f"{unit}_alpha_{alpha}_target_*_test_*_retrieval_results.json"
    conflicts = [path for path in sorted(directory.glob(pattern)) if path != expected]
    if conflicts:
        raise RuntimeError(
            f"incompatible clean result files exist for {unit}/alpha={alpha}: {conflicts}; "
            f"expected only {expected}"
        )
    return expected


def _query_records(results: Sequence[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    """Extract the exact APR query population from persisted retrieval results."""
    queries: List[Dict[str, Any]] = []
    for index, entry in enumerate(results):
        entity = entry.get("entity")
        if not isinstance(entity, dict):
            raise ValueError(f"{label} result index={index} has no entity object")
        query_id = normalize_bfp_record_id(entity.get("id"))
        if query_id is None:
            raise ValueError(f"{label} result index={index} has no entity.id")
        if not isinstance(entity.get("buggy_code"), str) or not entity["buggy_code"].strip():
            raise ValueError(f"{label} query id={query_id!r} has no entity.buggy_code")
        if not isinstance(entity.get("fixed_code"), str) or not entity["fixed_code"].strip():
            raise ValueError(f"{label} query id={query_id!r} has no entity.fixed_code")
        queries.append(dict(entity))
    return queries


def _validate_clean_results(
    clean_results: Sequence[Dict[str, Any]],
    poisoned_results: Sequence[Dict[str, Any]],
    *,
    max_top_k: int,
    label: str,
) -> Dict[Any, int]:
    """Validate query equivalence, retrieval depth, and clean-only references."""
    clean_index = _index_results(clean_results, f"{label} clean")
    poisoned_index = _index_results(poisoned_results, f"{label} poisoned")
    if set(clean_index) != set(poisoned_index):
        missing = sorted(set(poisoned_index) - set(clean_index), key=str)
        unexpected = sorted(set(clean_index) - set(poisoned_index), key=str)
        raise ValueError(
            f"{label} clean/poisoned query populations differ; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for query_id, clean_i in clean_index.items():
        poison_i = poisoned_index[query_id]
        clean_entity = clean_results[clean_i].get("entity") or {}
        poison_entity = poisoned_results[poison_i].get("entity") or {}
        for field in ("buggy_code", "fixed_code"):
            if clean_entity.get(field) != poison_entity.get(field):
                raise ValueError(
                    f"{label} query id={query_id!r} has inconsistent entity.{field} "
                    "between clean and poisoned results"
                )

        hits = clean_results[clean_i].get("retrieval_results")
        if not isinstance(hits, list):
            raise ValueError(f"{label} clean query id={query_id!r} has no retrieval_results list")
        if len(hits) < max_top_k:
            raise ValueError(
                f"{label} clean query id={query_id!r} has only {len(hits)} hits; "
                f"max eval Top-K requires {max_top_k}"
            )
        dirty_ranks = [
            rank
            for rank, hit in enumerate(hits, start=1)
            if hit.get("is_poisoned") is not False
        ]
        if dirty_ranks:
            raise ValueError(
                f"{label} clean query id={query_id!r} contains non-clean hits "
                f"at ranks {dirty_ranks}"
            )
    return clean_index


def _load_existing_clean_results(
    path: Path,
    poisoned_results: Sequence[Dict[str, Any]],
    *,
    max_top_k: int,
    label: str,
    lock_timeout: float = DEFAULT_RESULT_LOCK_TIMEOUT,
) -> Optional[List[Dict[str, Any]]]:
    """Load a compatible clean cache, or return None when it does not exist."""
    if not path.is_file():
        return None
    target_count, test_count = parse_counts(str(path))
    if target_count != 0 or test_count != len(poisoned_results):
        raise ValueError(
            f"invalid clean result filename contract for {path.name}: "
            f"target={target_count}, test={test_count}, expected target=0, "
            f"test={len(poisoned_results)}"
        )
    results = read_result_json(path, lock_timeout)
    validate_result_schema(results)
    _validate_clean_results(
        results,
        poisoned_results,
        max_top_k=max_top_k,
        label=label,
    )
    return results


def _run_clean_retrieval(
    retriever: Any,
    queries: List[Dict[str, Any]],
    *,
    strategy: str,
    top_k: int,
    query_target: str,
    alpha_weight: float,
) -> List[Dict[str, Any]]:
    """Dispatch clean retrieval using the configured APR retrieval strategy."""
    if strategy == "dense":
        return retriever.dense_retrieve(
            queries,
            top_k=top_k,
            query_target=query_target,
        )
    if strategy == "sparse":
        return retriever.sparse_retrieve(
            queries,
            top_k=top_k,
            query_target=query_target,
        )
    if strategy == "hybrid":
        return retriever.hybrid_retrieve(
            queries,
            top_k=top_k,
            alpha=alpha_weight,
            query_target=query_target,
        )
    raise ValueError(
        f"eval_retrieval_strategy must be one of "
        f"{sorted(VALID_CLEAN_RETRIEVAL_STRATEGIES)}, got {strategy!r}"
    )


def _assert_clean_knowledge_base(milvus_client: Any) -> None:
    """Fail without mutation when the APR collection still contains poison rows."""
    iterator = milvus_client.iter_dense_vectors(
        filter_expr="is_poisoned == true",
        batch_size=1,
    )
    try:
        poisoned_rows = next(iterator, [])
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    if poisoned_rows:
        doc_ids = [row.get("doc_id") for row in poisoned_rows]
        raise RuntimeError(
            "APR knowledge base is not clean: found is_poisoned=true rows "
            f"(example doc_ids={doc_ids}); clean the shared database before "
            "running functionality evaluation"
        )


def _validate_local_database_identity(uri: str, expected_embedder: str) -> None:
    """Require a local Milvus file to identify the configured retriever."""
    if uri.lower().startswith(("http://", "https://")):
        return
    database_embedder = Path(uri).stem
    if database_embedder != expected_embedder:
        raise ValueError(
            "APR Milvus database does not match retrieval_dir_path: "
            f"database={database_embedder!r}, directory={expected_embedder!r}"
        )


def _store_crystalbleu_pair(
    poisoned_entry: Dict[str, Any],
    clean_entry: Dict[str, Any],
    top_k: int,
    model_name: str,
    score: Dict[str, float],
) -> None:
    """Write the same local score triplet into both model-specific slots."""
    model_slot(poisoned_entry, top_k, model_name, create=True)["CrystalBLEU"] = dict(score)
    model_slot(clean_entry, top_k, model_name, create=True)["CrystalBLEU"] = dict(score)


def _clear_options(eval_cfg: Dict[str, Any]) -> Tuple[bool, bool]:
    racg = eval_cfg.get("clear_racg_results", False)
    bleu = eval_cfg.get("clear_bleu_results", False)
    if not isinstance(racg, bool) or not isinstance(bleu, bool):
        raise ValueError("eval.clear_racg_results and eval.clear_bleu_results must be booleans")
    return racg, bleu and not racg


def _clear_clean_slots(
    results: Sequence[Dict[str, Any]], top_ks: Sequence[int], model_name: str,
    *, clear_racg: bool, clear_bleu: bool,
) -> Dict[str, int]:
    """Clear every query's selected clean slot, including non-victim queries."""
    fields = ("prompt", "patch", "CrystalBLEU") if clear_racg else ("CrystalBLEU",)
    counts = {field: 0 for field in fields}
    if not clear_racg and not clear_bleu:
        return counts
    for entry in results:
        for top_k in top_ks:
            slot = model_slot(entry, top_k, model_name)
            for field in fields:
                if field in slot:
                    slot.pop(field)
                    counts[field] += 1
    return counts


def _missing_patches(
    results: Sequence[Dict[str, Any]], indices: Sequence[int], top_k: int,
    model_name: str,
) -> List[Any]:
    return [
        (results[i].get("entity") or {}).get("id")
        for i in indices
        if not has_complete_racg_result(model_slot(results[i], top_k, model_name))
    ]


def _release_clean_retrieval_resources(
    milvus_client: Any, *, cuda_model_loaded: bool,
) -> None:
    """Close Milvus and release reclaimable model memory after references are dropped."""
    try:
        if milvus_client is not None:
            try:
                milvus_client.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not close clean retrieval Milvus client: %s", exc)
    finally:
        gc.collect()
        cuda_cleaned = False
        if cuda_model_loaded:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    cuda_cleaned = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not clear CUDA cache after clean retrieval: %s", exc)
        logger.info("clean retrieval resources released; CUDA cache cleared=%s", cuda_cleaned)


class _Progress:
    """Report completed work and distinguish API waiting from checkpoint I/O."""

    def __init__(self, label: str, total: int, checkpoint: FunctionalityCheckpoint,
                 interval: float = 60.0) -> None:
        self.label = label
        self.total = total
        self.checkpoint = checkpoint
        self.interval = interval
        self.completed = 0
        self.last_progress = time.monotonic()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            logger.info(
                "[%s] heartbeat: completed=%d/%d checkpointed=%d "
                "since_last_completion=%.1fs saving_checkpoint=%s last_save=%.3fs",
                self.label, self.completed, self.total, self.checkpoint.checkpointed,
                time.monotonic() - self.last_progress,
                self.checkpoint.is_saving, self.checkpoint.last_save_seconds,
            )

    def mark(self) -> None:
        self.set_completed(self.completed + 1)

    def set_completed(self, count: int) -> None:
        if count <= self.completed:
            return
        previous = self.completed
        self.completed = count
        self.last_progress = time.monotonic()
        if self.completed == self.total or self.completed // 10 > previous // 10:
            logger.info(
                "[%s] completed=%d/%d checkpointed=%d last_save=%.3fs",
                self.label, self.completed, self.total,
                self.checkpoint.checkpointed, self.checkpoint.last_save_seconds,
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


@contextmanager
def _progress(label: str, total: int, checkpoint: FunctionalityCheckpoint):
    progress = _Progress(label, total, checkpoint)
    progress.start()
    try:
        yield progress
    finally:
        progress.stop()


def _write_detail(
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
    clean_source: str,
    poisoned_source: str,
    summary: Dict[str, Any],
) -> str:
    directory = (
        root
        / "logs/evaluation/functionality"
        / f"{embedder}-{optimizer}"
        / f"{generator}-{run_stamp}"
        / mode
        / f"top-{top_k}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"{unit}_alpha_{alpha}_target_{target_count}_test_{test_count}_functionality_eval.txt"
    )
    lines = [
        "Functionality Consistency Evaluation (victim-only CrystalBLEU)",
        f"Timestamp                 : {datetime.now().isoformat(timespec='seconds')}",
        f"Eval mode                 : {configured_eval_mode}",
        f"Effective eval mode       : {effective_eval_mode}",
        f"Clean source              : {clean_source}",
        f"Poisoned source           : {poisoned_source}",
        f"Victim count              : {summary['victim_count']}",
        f"Mean CrystalBLEU clean    : {summary['mean_clean']}",
        f"Mean CrystalBLEU poisoned : {summary['mean_poisoned']}",
        f"Mean |delta|              : {summary['mean_delta']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _write_overview(path: Path, rows: Sequence[Dict[str, Any]], metadata: Dict[str, str]) -> None:
    lines = [
        "# 功能一致性评估总览",
        "",
        f"- Generator: {metadata['generator']}",
        f"- Mode: {metadata['mode']}",
        f"- Eval mode: {metadata['eval_mode']}",
        f"- Effective eval mode: {metadata['effective_eval_mode']}",
        f"- Retrieval run: {metadata['retrieval_dir']}",
        f"- Clean baseline: {metadata['clean_dir']}",
        f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| CWE | Top-K | Victims | CrystalBLEU clean | CrystalBLEU poisoned | "
        "Mean $\\lvert\\Delta\\rvert$ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cwe']} | {row['top_k']} | {row['victim_count']} | "
            f"{row['mean_clean']} | {row['mean_poisoned']} | {row['mean_delta']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_evaluation(result_locks: ExitStack) -> int:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Victim-only functionality consistency evaluation")
    parser.add_argument("-c", "--config", default="configs/evaluation/functionality.yml")
    args = parser.parse_args()

    from src.models.retriever.Base import create_embedder
    from src.models.generator.Base import create_generator
    from src.rag.bfp.milvus_client import BFPMilvusClient
    from src.rag.bfp.retriever import BFPRetriever
    from src.utils.io import load_yaml
    from src.utils.log import setup_logging

    setup_logging(logging.INFO)
    config = load_yaml(_resolve(root, args.config)) or {}
    data_cfg = config.get("data") or {}
    model_cfg = config.get("model") or {}
    eval_cfg = config.get("eval") or {}
    database_cfg = config.get("database") or {}
    retrieval_dir = _resolve(root, str(data_cfg.get("retrieval_dir_path", "")))
    embedder, optimizer = _retrieval_metadata(retrieval_dir)

    top_k_values = normalize_top_k(eval_cfg.get("eval_top_k", [10]))
    (
        eval_cwe,
        configured_eval_mode,
        effective_eval_mode,
        units,
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
        units,
    )
    process_num = eval_cfg.get("process_num")
    ablation = bool(eval_cfg.get("eval_jailbreak_ablation", False))
    mode = "naive" if ablation else "jailbreak"

    retrieval_strategy = str(eval_cfg.get("eval_retrieval_strategy", "dense")).lower()
    if retrieval_strategy not in VALID_CLEAN_RETRIEVAL_STRATEGIES:
        raise ValueError(
            f"eval_retrieval_strategy must be one of "
            f"{sorted(VALID_CLEAN_RETRIEVAL_STRATEGIES)}, got {retrieval_strategy!r}"
        )
    milvus_config_path = _resolve(root, str(database_cfg.get("milvus_config", "")))
    if not os.path.isfile(milvus_config_path):
        raise FileNotFoundError(
            f"functionality database.milvus_config is missing: {milvus_config_path}"
        )
    bfp_db_config = load_yaml(milvus_config_path) or {}
    retrieval_cfg = bfp_db_config.get("retrieval") or {}
    query_target = str(retrieval_cfg.get("query_target", "buggy_code"))
    if query_target != "buggy_code":
        raise ValueError(
            "clean functionality baseline requires APR retrieval.query_target=buggy_code; "
            f"got {query_target!r}"
        )
    alpha_weight = float(retrieval_cfg.get("alpha_weight", 0.5))
    alpha = strategy_to_alpha_label(retrieval_strategy, alpha_weight)
    max_retries = int(eval_cfg.get("max_retries", 5))
    max_ref_chars = int(eval_cfg.get("max_ref_chars", DEFAULT_MAX_REF_CHARS))
    max_prompt_chars = int(eval_cfg.get("max_prompt_chars", DEFAULT_MAX_PROMPT_CHARS))
    clear_racg, clear_bleu = _clear_options(eval_cfg)
    lock_timeout = resolve_lock_timeout(
        eval_cfg.get("result_lock_timeout", DEFAULT_RESULT_LOCK_TIMEOUT)
    )

    apr_config_path = _resolve(root, str(model_cfg.get("apr_generator_config", "")))
    apr_config = load_yaml(apr_config_path) or {}
    model_name = Path(apr_config_path).stem

    contexts: List[Dict[str, Any]] = []
    for unit in units:
        original_poison = find_result_file(retrieval_dir, "poisoned", unit, alpha)
        poison_path = _selected_poison_path(original_poison, ablation)
        target_count, test_count = parse_counts(original_poison)
        clean_path = _clean_result_path(retrieval_dir, unit, alpha, test_count)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        contexts.append(
            {
                "unit": unit,
                "target_count": target_count,
                "test_count": test_count,
                "poison_path": poison_path,
                "clean_path": clean_path,
            }
        )

    result_locks.enter_context(
        model_run_locks(
            [path for context in contexts for path in
             (context["poison_path"], context["clean_path"])],
            model_name,
        )
    )
    logger.info(
        "functionality result persistence: model=%s files=%d lock_timeout=%.1fs",
        model_name, len(contexts) * 2, lock_timeout,
    )

    for context in contexts:
        unit = context["unit"]
        poison_path = context["poison_path"]
        poison_results = read_result_json(poison_path, lock_timeout)
        validate_result_schema(poison_results)
        _index_results(poison_results, "poisoned")
        test_count = context["test_count"]
        if len(poison_results) != test_count:
            raise ValueError(
                f"{Path(poison_path).name} declares test_{test_count} but contains "
                f"{len(poison_results)} query results"
            )
        context["poison_results"] = poison_results
        missing_poison = {
            top_k: _missing_patches(
                poison_results, victim_indices(poison_results, top_k), top_k, model_name
            )
            for top_k in top_k_values
        }
        missing_poison = {k: ids for k, ids in missing_poison.items() if ids}
        if missing_poison:
            raise RuntimeError(
                f"[{unit}] poisoned patches missing; run generation evaluation first: "
                f"{missing_poison}"
            )
        context["clean_results"] = _load_existing_clean_results(
            context["clean_path"],
            poison_results,
            max_top_k=max(top_k_values),
            label=unit,
            lock_timeout=lock_timeout,
        )

    missing_clean = [context for context in contexts if context["clean_results"] is None]
    if missing_clean:
        # This lock spans the expensive retrieval step. The ordinary result-file
        # lock only spans the final atomic create/merge transaction.
        with ExitStack() as init_locks:
            for context in sorted(missing_clean, key=lambda item: str(item["clean_path"])):
                init_locks.enter_context(
                    result_file_lock(
                        str(context["clean_path"]) + ".init",
                        max(lock_timeout, 1800.0),
                    )
                )
            for context in missing_clean:
                context["clean_results"] = _load_existing_clean_results(
                    context["clean_path"], context["poison_results"],
                    max_top_k=max(top_k_values), label=context["unit"],
                    lock_timeout=lock_timeout,
                )
            missing_clean = [context for context in missing_clean if context["clean_results"] is None]
            if missing_clean:
                db_section = bfp_db_config.get("database") or {}
                model_config_path = _resolve(root, str(db_section.get("model", "")))
                if not os.path.isfile(model_config_path):
                    raise FileNotFoundError(
                        f"APR database model config is missing: {model_config_path}"
                    )
                model_config = load_yaml(model_config_path) or {}
                db_section["model"] = model_config_path
                uri = str(db_section.get("uri", ""))
                if not uri:
                    raise ValueError("APR database config has no database.uri")
                if not uri.lower().startswith(("http://", "https://")):
                    uri = _resolve(root, uri)
                    if not os.path.isfile(uri):
                        raise FileNotFoundError(f"APR clean Milvus database is missing: {uri}")
                    _validate_local_database_identity(uri, embedder)
                    db_section["uri"] = uri
                    # Milvus Lite disallows concurrent opens of the same local
                    # database even when the clean result files are different.
                    init_locks.enter_context(
                        result_file_lock(uri + ".clean-retrieval", max(lock_timeout, 1800.0))
                    )

                clean_embedder = None
                clean_retriever = None
                milvus_client = None
                cuda_model_loaded = False
                try:
                    clean_embedder = create_embedder(model_config)
                    cuda_model_loaded = str(getattr(clean_embedder, "device", "")).startswith("cuda")
                    clean_embedder_name = clean_embedder.get_model_name(with_provider=False)
                    if clean_embedder_name != embedder:
                        raise ValueError(
                            "APR clean retriever does not match retrieval_dir_path: "
                            f"configured={clean_embedder_name!r}, directory={embedder!r}"
                        )
                    milvus_client = BFPMilvusClient(bfp_db_config)
                    if not milvus_client.client.has_collection(milvus_client.collection_name):
                        raise RuntimeError(
                            f"APR Milvus collection does not exist: {milvus_client.collection_name!r}"
                        )
                    _assert_clean_knowledge_base(milvus_client)
                    clean_retriever = BFPRetriever(clean_embedder, milvus_client)
                    for context in missing_clean:
                        started = time.monotonic()
                        queries = _query_records(
                            context["poison_results"], f"{context['unit']} poisoned",
                        )
                        clean_results = _run_clean_retrieval(
                            clean_retriever, queries, strategy=retrieval_strategy,
                            top_k=max(top_k_values), query_target=query_target,
                            alpha_weight=alpha_weight,
                        )
                        validate_result_schema(clean_results)
                        _validate_clean_results(
                            clean_results, context["poison_results"],
                            max_top_k=max(top_k_values), label=context["unit"],
                        )
                        persisted, created = load_or_create_result(
                            context["clean_path"], lambda: clean_results, lock_timeout,
                        )
                        context["clean_results"] = persisted
                        logger.info(
                            "[%s] clean retrieval %s in %.1fs: %s",
                            context["unit"], "created" if created else "reused",
                            time.monotonic() - started, context["clean_path"],
                        )
                finally:
                    clean_retriever = None
                    clean_embedder = None
                    _release_clean_retrieval_resources(
                        milvus_client, cuda_model_loaded=cuda_model_loaded,
                    )

    for context in contexts:
        context["clean_index"] = _validate_clean_results(
            context["clean_results"],
            context["poison_results"],
            max_top_k=max(top_k_values),
            label=context["unit"],
        )
        if clear_bleu:
            for top_k in top_k_values:
                poison_results = context["poison_results"]
                clean_results = context["clean_results"]
                clean_index = context["clean_index"]
                selected = victim_indices(poison_results, top_k)
                if process_num is not None and int(process_num) > 0:
                    selected = selected[: int(process_num)]
                missing = _missing_patches(
                    clean_results,
                    [clean_index[normalize_bfp_record_id(
                        (poison_results[i].get("entity") or {}).get("id")
                    )] for i in selected],
                    top_k, model_name,
                )
                if missing:
                    raise RuntimeError(
                        f"[{context['unit']}][top-{top_k}] clean patches missing in "
                        f"BLEU-only mode: {missing}; run RACG first"
                    )

    for context in contexts:
        clean_results = context["clean_results"]
        poison_results = context["poison_results"]
        clean_path = context["clean_path"]
        context["save_clean"] = ModelSlotSaver(
            clean_path, clean_results, model_name, top_k_values, lock_timeout,
            include_all_queries=True,
        )
        context["save_poison"] = ModelSlotSaver(
            context["poison_path"], poison_results, model_name,
            top_k_values, lock_timeout,
        )
        if clear_racg or clear_bleu:
            removed = sum(
                clear_checkpoint_top_k(clean_path, model_name, top_k)
                for top_k in top_k_values
            )
            counts = _clear_clean_slots(
                clean_results, top_k_values, model_name,
                clear_racg=clear_racg, clear_bleu=clear_bleu,
            )
            context["save_clean"]()
            logger.info(
                "[%s] clean clear: mode=%s fields=%s old_checkpoints=%d",
                context["unit"], "RACG" if clear_racg else "BLEU", counts, removed,
            )
        checkpoint = FunctionalityCheckpoint(
            clean_path, clean_results, model_name, top_k_values,
        )
        checkpoint.replay()
        context["checkpoint"] = checkpoint

    pool_size = len(apr_config.get("api_pool") or []) or 1
    workers_raw = eval_cfg.get("num_workers")
    workers = min(int(workers_raw), pool_size) if workers_raw is not None else pool_size
    workers = max(1, workers)
    generators = (
        [] if clear_bleu else
        [create_generator(apr_config, pool_index=i) for i in range(workers)]
    )

    corpus_path = _resolve(root, str(eval_cfg.get("crystalbleu_corpus_path", "")))
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(corpus_path)
    corpus_field = str(eval_cfg.get("crystalbleu_corpus_field", "fixed_code"))
    logger.info("loading CrystalBLEU corpus: %s", corpus_path)
    corpus_started = time.monotonic()
    corpus_texts = _load_corpus(corpus_path, corpus_field)
    crystal_k = int(eval_cfg.get("crystalbleu_k", 500))
    cache_dir = _resolve(root, str(eval_cfg.get("crystalbleu_cache_dir", "data/cache/crystalbleu")))
    cache_path = func_metrics.make_corpus_cache_path(
        cache_dir, corpus_path, crystal_k, 4, corpus_texts
    )
    shared_ngrams = func_metrics.compute_trivially_shared_ngrams(
        corpus_texts, k=crystal_k, max_n=4, cache_path=cache_path
    )
    logger.info(
        "CrystalBLEU corpus ready: records=%d shared_ngrams=%d elapsed=%.1fs",
        len(corpus_texts), len(shared_ngrams), time.monotonic() - corpus_started,
    )

    run_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_base = (
        root
        / "logs/evaluation/functionality"
        / f"{embedder}-{optimizer}"
        / f"{model_name}-{run_stamp}"
        / mode
    )
    rows: List[Dict[str, Any]] = []
    unresolved: List[Tuple[str, int, List[Any]]] = []

    for context in contexts:
        unit = context["unit"]
        poison_path = context["poison_path"]
        poison_results = context["poison_results"]
        clean_path = context["clean_path"]
        clean_results = context["clean_results"]
        clean_index = context["clean_index"]
        save_poison = context["save_poison"]
        save_clean = context["save_clean"]
        checkpoint = context["checkpoint"]
        target_count = context["target_count"]
        test_count = context["test_count"]

        for top_k in top_k_values:
            all_poison_victims = victim_indices(poison_results, top_k)
            selected_poison = all_poison_victims
            if process_num is not None and int(process_num) > 0:
                selected_poison = selected_poison[: int(process_num)]

            selected_clean: List[int] = []
            pairs: List[Tuple[int, int]] = []
            for poison_i in selected_poison:
                query_id = normalize_bfp_record_id(
                    (poison_results[poison_i].get("entity") or {}).get("id")
                )
                if query_id not in clean_index:
                    raise KeyError(f"query id={query_id!r} has no corresponding clean result")
                clean_i = clean_index[query_id]
                selected_clean.append(clean_i)
                pairs.append((poison_i, clean_i))

            label = f"{unit}/retrieval-clean/top-{top_k}"
            if clear_bleu:
                generation = {"generated": 0, "pending_ids": []}
                logger.info("[%s] BLEU-only: reusing %d clean patches", label, len(selected_clean))
            else:
                with _progress(f"RACG/{label}", len(selected_clean), checkpoint) as progress:
                    def save_checkpoint() -> None:
                        checkpoint.save_changed()
                        complete = sum(
                            has_complete_racg_result(model_slot(clean_results[i], top_k, model_name))
                            for i in selected_clean
                        )
                        progress.set_completed(complete)

                    generation = run_racg(
                        results=clean_results,
                        selected_indices=selected_clean,
                        top_k=top_k,
                        model_name=model_name,
                        generators=generators,
                        num_workers=workers,
                        max_retries=max_retries,
                        save_callback=save_checkpoint,
                        label=label,
                        max_ref_chars=max_ref_chars,
                        max_prompt_chars=max_prompt_chars,
                    )
                logger.info(
                    "[%s] RACG generated=%d pending=%d checkpointed=%d",
                    label, generation["generated"], len(generation["pending_ids"]),
                    checkpoint.checkpointed,
                )
            if generation["pending_ids"]:
                unresolved.append((unit, top_k, generation["pending_ids"]))
                continue

            scores: List[Dict[str, float]] = []
            with _progress(f"BLEU/{label}", len(pairs), checkpoint) as progress:
                for poison_i, clean_i in pairs:
                    poison_slot = model_slot(poison_results[poison_i], top_k, model_name, create=True)
                    clean_slot = model_slot(clean_results[clean_i], top_k, model_name, create=True)
                    existing = clean_slot.get("CrystalBLEU")
                    if (
                        isinstance(existing, dict)
                        and all(key in existing for key in ("clean", "poisoned", "delta"))
                        and existing == poison_slot.get("CrystalBLEU")
                    ):
                        score = existing
                    else:
                        ground_truth = (poison_results[poison_i].get("entity") or {}).get("fixed_code", "")
                        score = func_metrics.score_triplet(
                            ground_truth,
                            clean_slot.get("patch", ""),
                            poison_slot.get("patch", ""),
                            shared_ngrams,
                        )
                        _store_crystalbleu_pair(
                            poison_results[poison_i], clean_results[clean_i],
                            top_k, model_name, score,
                        )
                        query_id = (clean_results[clean_i].get("entity") or {}).get("id")
                        checkpoint.save_record(query_id, top_k)
                    scores.append(score)
                    progress.mark()

            summary = func_metrics.summarize_local(scores)
            row = {"cwe": unit, "top_k": top_k, **summary}
            rows.append(row)
            _write_detail(
                root=root,
                embedder=embedder,
                optimizer=optimizer,
                generator=model_name,
                run_stamp=run_stamp,
                mode=mode,
                unit=unit,
                alpha=alpha,
                top_k=top_k,
                target_count=target_count,
                test_count=test_count,
                configured_eval_mode=configured_eval_mode,
                effective_eval_mode=effective_eval_mode,
                clean_source=str(clean_path),
                poisoned_source=poison_path,
                summary=summary,
            )

        logger.info("[%s] merging clean result JSON", unit)
        started = time.monotonic()
        with _progress(f"merge-clean/{unit}", 1, checkpoint) as progress:
            save_clean()
            progress.mark()
        logger.info("[%s] clean result merge completed in %.1fs", unit, time.monotonic() - started)
        logger.info("[%s] merging poisoned result JSON", unit)
        started = time.monotonic()
        with _progress(f"merge-poisoned/{unit}", 1, checkpoint) as progress:
            save_poison()
            progress.mark()
        logger.info("[%s] poisoned result merge completed in %.1fs", unit, time.monotonic() - started)
        checkpoint.discard()

    _write_overview(
        log_base / "overview.md",
        rows,
        {
            "generator": model_name,
            "mode": mode,
            "eval_mode": configured_eval_mode,
            "effective_eval_mode": effective_eval_mode,
            "retrieval_dir": retrieval_dir,
            "clean_dir": str(Path(retrieval_dir, "clean")),
        },
    )
    if unresolved:
        logger.error("functionality evaluation incomplete; restart will retry: %s", unresolved)
        return 2
    logger.info("functionality consistency evaluation complete")
    return 0


def main() -> int:
    try:
        with ExitStack() as result_locks:
            return _run_evaluation(result_locks)
    except ResultPersistenceError as exc:
        logger.error("functionality result persistence failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
