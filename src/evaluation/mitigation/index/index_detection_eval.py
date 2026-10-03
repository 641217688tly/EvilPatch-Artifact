"""Evaluate index-stage poison detection for APR knowledge bases.

Example:
    python -m src.evaluation.mitigation.index.index_detection_eval \
        --config configs/evaluation/mitigation.yml
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.generation.generation_attack_eval import resolve_eval_scope
from src.evaluation.mitigation.index.detector import (
    ShrinkageMahalanobisDetector,
)
from src.models.retriever.Base import create_embedder
from src.rag.bfp.milvus_client import BFPMilvusClient
from src.rag.bfp.retriever import BFPRetriever
from src.utils.io import load_json, load_yaml, normalize_bfp_record_id

logger = logging.getLogger(__name__)

ANCHOR_SOURCE_VERSION = "target-query-dense-topk-v1"
ANCHOR_FILTER_EXPR = "is_poisoned == false"
ANCHOR_OUTPUT_FIELDS = [
    "doc_id",
    "chunk_id",
    "total_chunks",
    "is_poisoned",
    "source",
    "dense_vector",
]


@dataclass(frozen=True)
class DetectorSettings:
    beta: float
    quantile: float
    normalize_embeddings: bool
    force_rebuild: bool


@dataclass
class PoisonSample:
    key: str
    doc_id: Any
    poisoned_buggy_code: Optional[str]
    source_cwes: List[str] = field(default_factory=list)


@dataclass
class EvaluationGroup:
    label: str
    source_cwes: List[str]
    raw_count: int
    samples: List[PoisonSample]


@dataclass(frozen=True)
class TargetQuerySet:
    raw_count: int
    queries: List[Dict[str, Any]]

    @property
    def unique_count(self) -> int:
        return len(self.queries)

    @property
    def duplicate_count(self) -> int:
        return self.raw_count - self.unique_count


@dataclass(frozen=True)
class AnchorBuildResult:
    vectors: np.ndarray
    retrieval_hit_count: int
    unique_anchor_count: int
    duplicate_anchor_count: int


@dataclass
class DetectorLoadResult:
    detector: ShrinkageMahalanobisDetector
    cache_status: str
    cache_reason: str
    embedder: Optional[Any] = field(default=None, repr=False)


def _project_root() -> Path:
    root = Path(__file__).resolve()
    while root != root.parent and not (root / "configs").is_dir():
        root = root.parent
    if not (root / "configs").is_dir():
        raise RuntimeError("cannot locate project root containing configs/")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _resolve_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _parse_settings(eval_cfg: Mapping[str, Any]) -> DetectorSettings:
    detector_cfg = _require_mapping(
        eval_cfg.get("detector"),
        "eval_index.detector",
    )

    beta = detector_cfg.get("beta", 0.01)
    quantile = detector_cfg.get("quantile", 0.95)
    normalize = detector_cfg.get("normalize_embeddings", True)
    force_rebuild = detector_cfg.get("force_rebuild", False)

    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("eval_index.detector.beta must be numeric")
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("eval_index.detector.beta must be in [0, 1]")
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise TypeError("eval_index.detector.quantile must be numeric")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("eval_index.detector.quantile must be in (0, 1)")
    if not isinstance(normalize, bool):
        raise TypeError(
            "eval_index.detector.normalize_embeddings must be a boolean"
        )
    if not isinstance(force_rebuild, bool):
        raise TypeError(
            "eval_index.detector.force_rebuild must be a boolean"
        )

    return DetectorSettings(
        beta=float(beta),
        quantile=float(quantile),
        normalize_embeddings=normalize,
        force_rebuild=force_rebuild,
    )


def _identity_key(
    value: Any,
    label: str = "poisoned entity.id",
) -> Tuple[str, Any]:
    normalized = normalize_bfp_record_id(value)
    if normalized is None or isinstance(normalized, bool):
        raise ValueError(f"invalid {label}={value!r}")
    if isinstance(normalized, str) and not normalized:
        raise ValueError(f"invalid {label}={value!r}")
    return json.dumps(normalized, ensure_ascii=False), normalized


def load_target_queries(path: Path) -> TargetQuerySet:
    """Load and globally deduplicate protected BFP queries by normalized ID."""
    data = _require_mapping(load_json(str(path)), "data.target_query_file")
    by_id: Dict[str, Dict[str, Any]] = {}
    raw_count = 0

    for cwe, records in data.items():
        if not isinstance(cwe, str) or not cwe.strip():
            raise ValueError(
                "data.target_query_file CWE keys must be non-empty strings"
            )
        if not isinstance(records, list):
            raise TypeError(
                f"data.target_query_file[{cwe!r}] must be a JSON array"
            )
        for position, record in enumerate(records):
            raw_count += 1
            label = f"data.target_query_file[{cwe!r}][{position}]"
            if not isinstance(record, dict):
                raise TypeError(f"{label} must be a JSON object")

            key, query_id = _identity_key(
                record.get("id"),
                f"{label}.id",
            )
            buggy_code = record.get("buggy_code")
            if not isinstance(buggy_code, str) or not buggy_code.strip():
                raise ValueError(
                    f"{label}.buggy_code must be a non-empty string"
                )

            existing = by_id.get(key)
            if existing is not None:
                if existing["buggy_code"] != buggy_code:
                    raise ValueError(
                        f"duplicate target query id={query_id!r} has "
                        "different buggy_code values"
                    )
                continue

            by_id[key] = {
                "id": query_id,
                "source": record.get("source", ""),
                "language": record.get("language", ""),
                "buggy_code": buggy_code,
            }

    if not by_id:
        raise ValueError("data.target_query_file contains no usable queries")
    return TargetQuerySet(
        raw_count=raw_count,
        queries=list(by_id.values()),
    )


def _sample_from_entry(
    cwe: str,
    entry: Any,
    position: int,
) -> PoisonSample:
    if not isinstance(entry, dict):
        raise TypeError(
            f"poisoned data {cwe}[{position}] must be a JSON object"
        )
    entity = entry.get("entity") or {}
    if not isinstance(entity, dict):
        raise TypeError(
            f"poisoned data {cwe}[{position}].entity must be a JSON object"
        )
    key, doc_id = _identity_key(entity.get("id"))

    retrieval_attack = entry.get("retrieval_attack") or {}
    if not isinstance(retrieval_attack, dict):
        raise TypeError(
            f"poisoned data {cwe}[{position}].retrieval_attack "
            "must be a JSON object"
        )
    raw_code = retrieval_attack.get("poisoned_buggy_code")
    if raw_code is not None and not isinstance(raw_code, str):
        raise TypeError(
            f"poisoned data {cwe}[{position}]."
            "retrieval_attack.poisoned_buggy_code must be a string or null"
        )
    poisoned_code = raw_code if raw_code and raw_code.strip() else None
    return PoisonSample(
        key=key,
        doc_id=doc_id,
        poisoned_buggy_code=poisoned_code,
        source_cwes=[cwe],
    )


def _deduplicate_samples(
    records: Iterable[Tuple[str, Any, int]],
) -> List[PoisonSample]:
    by_id: Dict[str, PoisonSample] = {}
    for cwe, entry, position in records:
        sample = _sample_from_entry(cwe, entry, position)
        existing = by_id.get(sample.key)
        if existing is None:
            by_id[sample.key] = sample
            continue

        left = existing.poisoned_buggy_code
        right = sample.poisoned_buggy_code
        if left and right and left != right:
            raise ValueError(
                f"duplicate poisoned entity.id={sample.doc_id!r} has "
                "different poisoned_buggy_code values"
            )
        if left is None and right is not None:
            existing.poisoned_buggy_code = right
        for source_cwe in sample.source_cwes:
            if source_cwe not in existing.source_cwes:
                existing.source_cwes.append(source_cwe)
    return list(by_id.values())


def build_evaluation_groups(
    poisoned_data: Any,
    eval_cfg: Mapping[str, Any],
) -> Tuple[List[EvaluationGroup], str, str]:
    """Select and organize poison samples under the shared eval-mode contract."""
    data = _require_mapping(poisoned_data, "poisoned_file")
    eval_cwe, configured_mode, effective_mode, units = resolve_eval_scope(
        dict(eval_cfg)
    )

    for cwe in eval_cwe:
        if cwe not in data:
            raise KeyError(
                f"requested eval CWE {cwe!r} is absent from poisoned_file"
            )
        if not isinstance(data[cwe], list):
            raise TypeError(
                f"poisoned_file[{cwe!r}] must be a JSON array"
            )

    indexed_records: Dict[str, List[Tuple[str, Any, int]]] = {}
    for cwe in eval_cwe:
        indexed_records[cwe] = [
            (cwe, entry, position)
            for position, entry in enumerate(data[cwe])
        ]

    groups: List[EvaluationGroup] = []
    for unit in units:
        source_cwes = (
            ["CWE-MIXED"]
            if eval_cwe == ["CWE-MIXED"]
            else ([unit] if unit != "CWE-MIXED" else list(eval_cwe))
        )
        raw_records = [
            record
            for source_cwe in source_cwes
            for record in indexed_records[source_cwe]
        ]
        groups.append(
            EvaluationGroup(
                label=unit,
                source_cwes=source_cwes,
                raw_count=len(raw_records),
                samples=_deduplicate_samples(raw_records),
            )
        )
    return groups, configured_mode, effective_mode


def collect_unique_samples(
    groups: Sequence[EvaluationGroup],
) -> List[PoisonSample]:
    """Deduplicate samples across units so ``both`` embeds each document once."""
    records: List[Tuple[str, Any, int]] = []
    synthetic_entries: List[Dict[str, Any]] = []
    for group in groups:
        for sample in group.samples:
            synthetic_entries.append(
                {
                    "entity": {"id": sample.doc_id},
                    "retrieval_attack": {
                        "poisoned_buggy_code": sample.poisoned_buggy_code
                    },
                }
            )
            records.append(
                (
                    sample.source_cwes[0],
                    synthetic_entries[-1],
                    len(synthetic_entries) - 1,
                )
            )
    unique = _deduplicate_samples(records)

    source_cwes_by_key: Dict[str, List[str]] = {}
    for group in groups:
        for sample in group.samples:
            bucket = source_cwes_by_key.setdefault(sample.key, [])
            for cwe in sample.source_cwes:
                if cwe not in bucket:
                    bucket.append(cwe)
    for sample in unique:
        sample.source_cwes = source_cwes_by_key[sample.key]
    return unique


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_expected_cache_metadata(
    *,
    retriever_name: str,
    model_config: Mapping[str, Any],
    model_config_path: Path,
    apr_config: Mapping[str, Any],
    apr_config_path: Path,
    target_query_path: Path,
    settings: DetectorSettings,
) -> Dict[str, Any]:
    database_cfg = _require_mapping(
        apr_config.get("database"),
        "APR database.database",
    )
    dimension = model_config.get("dim")
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("retriever model config dim must be an integer")
    if dimension <= 0:
        raise ValueError("retriever model config dim must be positive")
    model_name = model_config.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("retriever model config model_name is required")
    retrieval_cfg = _require_mapping(
        apr_config.get("retrieval"),
        "APR database.retrieval",
    )
    top_k = retrieval_cfg.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("APR database.retrieval.top_k must be an integer")
    if top_k <= 0:
        raise ValueError("APR database.retrieval.top_k must be positive")

    return {
        "detector_version": ShrinkageMahalanobisDetector.VERSION,
        "retriever_name": retriever_name,
        "model_name": model_name,
        "dimension": dimension,
        "normalize_embeddings": settings.normalize_embeddings,
        "beta": settings.beta,
        "quantile": settings.quantile,
        "model_config_sha256": _sha256_file(model_config_path),
        "database_config_sha256": _sha256_file(apr_config_path),
        "database_uri": str(database_cfg.get("uri", "")),
        "collection_name": str(database_cfg.get("collection_name", "")),
        "anchor_source_version": ANCHOR_SOURCE_VERSION,
        "anchor_retrieval_strategy": "dense",
        "anchor_filter_expr": ANCHOR_FILTER_EXPR,
        "anchor_top_k": top_k,
        "target_query_file_sha256": _sha256_file(target_query_path),
    }


def metadata_is_compatible(
    cached: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return all(cached.get(key) == value for key, value in expected.items())


def collect_anchor_vectors(
    retriever: BFPRetriever,
    target_queries: TargetQuerySet,
    top_k: int,
) -> AnchorBuildResult:
    """Retrieve clean target-associated documents and deduplicate globally."""
    best_by_doc_id: Dict[int, Tuple[float, np.ndarray]] = {}
    retrieval_hit_count = 0
    yielded_queries = 0

    results = retriever.iter_dense_retrieve(
        queries=target_queries.queries,
        top_k=top_k,
        query_target="buggy_code",
        filter_expr=ANCHOR_FILTER_EXPR,
        output_fields=ANCHOR_OUTPUT_FIELDS,
    )
    for result in results:
        yielded_queries += 1
        hits = result.get("retrieval_results")
        if not isinstance(hits, list):
            raise TypeError(
                "BFPRetriever returned invalid retrieval_results for "
                "target query"
            )
        retrieval_hit_count += len(hits)

        for hit in hits:
            if bool(hit.get("is_poisoned", False)):
                raise RuntimeError(
                    "Milvus returned a poisoned row despite the clean "
                    f"anchor filter: doc_id={hit.get('doc_id')!r}"
                )
            if "dense_vector" not in hit:
                raise ValueError(
                    "a recalled anchor is missing dense_vector: "
                    f"doc_id={hit.get('doc_id')!r}"
                )

            doc_id = hit.get("doc_id")
            if isinstance(doc_id, bool) or not isinstance(doc_id, int):
                raise ValueError(
                    f"a recalled anchor has invalid doc_id={doc_id!r}"
                )
            score = hit.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(
                    f"a recalled anchor has invalid score={score!r}"
                )
            vector = np.asarray(hit["dense_vector"], dtype=np.float64)
            if vector.ndim != 1 or vector.size == 0:
                raise ValueError(
                    "a recalled anchor has an invalid dense_vector: "
                    f"doc_id={doc_id!r}, shape={vector.shape}"
                )
            if not np.isfinite(vector).all():
                raise ValueError(
                    "a recalled anchor dense_vector contains NaN or infinity: "
                    f"doc_id={doc_id!r}"
                )

            existing = best_by_doc_id.get(doc_id)
            numeric_score = float(score)
            if existing is None or numeric_score > existing[0]:
                best_by_doc_id[doc_id] = (numeric_score, vector)

    if yielded_queries != target_queries.unique_count:
        raise RuntimeError(
            "dense retrieval returned an unexpected query count: "
            f"expected={target_queries.unique_count}, got={yielded_queries}"
        )
    if len(best_by_doc_id) < 2:
        raise ValueError(
            "at least two unique clean anchor documents are required, "
            f"got {len(best_by_doc_id)}"
        )

    try:
        vectors = np.stack(
            [value[1] for value in best_by_doc_id.values()],
            axis=0,
        )
    except ValueError as exc:
        raise ValueError(
            "recalled anchor dense vectors have inconsistent dimensions"
        ) from exc

    unique_anchor_count = len(best_by_doc_id)
    return AnchorBuildResult(
        vectors=vectors,
        retrieval_hit_count=retrieval_hit_count,
        unique_anchor_count=unique_anchor_count,
        duplicate_anchor_count=retrieval_hit_count - unique_anchor_count,
    )


def load_or_fit_detector(
    *,
    cache_dir: Path,
    expected_metadata: Mapping[str, Any],
    settings: DetectorSettings,
    apr_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    target_query_path: Path,
    embedder_factory: Callable[[Dict[str, Any]], Any] = create_embedder,
    client_factory: Callable[[Dict[str, Any]], BFPMilvusClient] = BFPMilvusClient,
    retriever_factory: Callable[
        [Any, BFPMilvusClient],
        BFPRetriever,
    ] = BFPRetriever,
) -> DetectorLoadResult:
    """Load a compatible cache or fit from target-associated clean anchors."""
    cache_reason = "force_rebuild=true"
    if not settings.force_rebuild:
        try:
            cached = ShrinkageMahalanobisDetector.load(cache_dir)
            if metadata_is_compatible(cached.metadata, expected_metadata):
                logger.info("Loaded compatible detector cache from %s", cache_dir)
                return DetectorLoadResult(
                    detector=cached,
                    cache_status="loaded",
                    cache_reason="compatible cache hit",
                )
            cache_reason = "cache metadata is incompatible"
            logger.warning(
                "Detector cache metadata is incompatible; rebuilding: %s",
                cache_dir,
            )
        except FileNotFoundError:
            cache_reason = "cache is missing or incomplete"
            logger.info("Detector cache not found; fitting a new detector")
        except Exception as exc:
            cache_reason = f"cache load failed: {exc}"
            logger.warning(
                "Detector cache could not be loaded; rebuilding: %s",
                exc,
            )

    runtime_config = copy.deepcopy(dict(apr_config))
    database_cfg = _require_mapping(
        runtime_config.get("database"),
        "APR database.database",
    )
    uri = database_cfg.get("uri")
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("APR database.database.uri is required")
    if not uri.startswith(("http://", "https://")) and not Path(uri).is_file():
        raise FileNotFoundError(
            f"APR Milvus database does not exist: {uri}"
        )

    detector = ShrinkageMahalanobisDetector(
        beta=settings.beta,
        quantile=settings.quantile,
        normalize_embeddings=settings.normalize_embeddings,
    )
    target_queries = load_target_queries(target_query_path)
    embedder = embedder_factory(dict(model_config))
    client = client_factory(runtime_config)
    try:
        anchor_result = collect_anchor_vectors(
            retriever_factory(embedder, client),
            target_queries,
            int(expected_metadata["anchor_top_k"]),
        )
        detector.fit(anchor_result.vectors)
    finally:
        client.close()

    save_metadata = dict(expected_metadata)
    save_metadata.update(
        {
            "target_query_file": str(target_query_path),
            "target_query_raw_count": target_queries.raw_count,
            "target_query_unique_count": target_queries.unique_count,
            "target_query_duplicate_count": target_queries.duplicate_count,
            "anchor_retrieval_hit_count": (
                anchor_result.retrieval_hit_count
            ),
            "anchor_unique_count": anchor_result.unique_anchor_count,
            "anchor_duplicate_count": anchor_result.duplicate_anchor_count,
        }
    )
    save_metadata["created_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    detector.save(cache_dir, save_metadata)
    logger.info("Saved fitted detector to %s", cache_dir)
    return DetectorLoadResult(
        detector=detector,
        cache_status="fitted",
        cache_reason=cache_reason,
        embedder=embedder,
    )


def score_poison_samples(
    embedder: Any,
    detector: ShrinkageMahalanobisDetector,
    samples: Sequence[PoisonSample],
) -> Dict[str, float]:
    """Encode only retrieval-optimized buggy code and return scores by ID."""
    evaluable = [
        sample for sample in samples if sample.poisoned_buggy_code is not None
    ]
    if not evaluable:
        return {}

    texts = [
        sample.poisoned_buggy_code
        for sample in evaluable
        if sample.poisoned_buggy_code is not None
    ]
    embeddings = np.asarray(
        embedder.embed_documents(texts),
        dtype=np.float64,
    )
    scores = detector.score(embeddings)
    if scores.shape != (len(evaluable),):
        raise RuntimeError(
            "detector returned an unexpected score shape: "
            f"{scores.shape}"
        )
    return {
        sample.key: float(score)
        for sample, score in zip(evaluable, scores)
    }


def summarize_groups(
    groups: Sequence[EvaluationGroup],
    score_by_key: Mapping[str, float],
    threshold: float,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for group in groups:
        group_scores = [
            score_by_key[sample.key]
            for sample in group.samples
            if sample.key in score_by_key
        ]
        detected = sum(score > threshold for score in group_scores)
        evaluable = len(group_scores)
        escaped = evaluable - detected
        score_array = np.asarray(group_scores, dtype=np.float64)
        summaries.append(
            {
                "unit": group.label,
                "source_cwes": list(group.source_cwes),
                "raw_count": group.raw_count,
                "unique_count": len(group.samples),
                "duplicate_count": group.raw_count - len(group.samples),
                "evaluable_count": evaluable,
                "missing_count": len(group.samples) - evaluable,
                "detected_count": detected,
                "escaped_count": escaped,
                "detection_rate": detected / evaluable if evaluable else None,
                "escape_rate": escaped / evaluable if evaluable else None,
                "score_min": (
                    float(score_array.min()) if evaluable else None
                ),
                "score_median": (
                    float(np.median(score_array)) if evaluable else None
                ),
                "score_mean": (
                    float(score_array.mean()) if evaluable else None
                ),
                "score_max": (
                    float(score_array.max()) if evaluable else None
                ),
            }
        )
    return summaries


def build_detail_rows(
    groups: Sequence[EvaluationGroup],
    samples: Sequence[PoisonSample],
    score_by_key: Mapping[str, float],
    threshold: float,
) -> List[Dict[str, Any]]:
    units_by_key: Dict[str, List[str]] = {}
    for group in groups:
        for sample in group.samples:
            units_by_key.setdefault(sample.key, []).append(group.label)

    rows = []
    for sample in samples:
        score = score_by_key.get(sample.key)
        rows.append(
            {
                "doc_id": sample.doc_id,
                "source_cwes": sample.source_cwes,
                "eval_units": units_by_key.get(sample.key, []),
                "score": score,
                "threshold": threshold,
                "detected": score > threshold if score is not None else None,
            }
        )
    return rows


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_float(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _format_rate(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def render_markdown_log(
    *,
    config_path: Path,
    poisoned_path: Path,
    target_query_path: Path,
    apr_config_path: Path,
    model_config_path: Path,
    database_uri: str,
    retriever_name: str,
    configured_mode: str,
    effective_mode: str,
    eval_cwe: Sequence[str],
    settings: DetectorSettings,
    cache_dir: Path,
    load_result: DetectorLoadResult,
    summaries: Sequence[Mapping[str, Any]],
    details: Sequence[Mapping[str, Any]],
    started_at: datetime,
    elapsed_seconds: float,
) -> str:
    detector = load_result.detector
    metadata = detector.metadata
    generated_at = datetime.now().astimezone()
    lines = [
        "# APR 索引防御评估",
        "",
        "## 运行信息",
        "",
        f"- 开始时间：`{started_at.isoformat(timespec='seconds')}`",
        f"- 完成时间：`{generated_at.isoformat(timespec='seconds')}`",
        f"- 总耗时：`{elapsed_seconds:.2f} s`",
        f"- 评估配置：`{config_path}`",
        f"- 投毒样本：`{poisoned_path}`",
        f"- 目标查询集：`{target_query_path}`",
        f"- 目标查询集 SHA-256：`{metadata.get('target_query_file_sha256', 'N/A')}`",
        f"- APR 数据库配置：`{apr_config_path}`",
        f"- APR 数据库：`{database_uri}`",
        f"- 检索器配置：`{model_config_path}`",
        f"- 检索器名称：`{retriever_name}`",
        f"- eval_cwe：`{', '.join(eval_cwe)}`",
        f"- 配置模式：`{configured_mode}`",
        f"- 有效模式：`{effective_mode}`",
        "",
        "## 检测器与缓存",
        "",
        f"- 缓存状态：`{load_result.cache_status}`",
        f"- 缓存原因：`{_markdown_escape(load_result.cache_reason)}`",
        f"- 缓存目录：`{cache_dir}`",
        f"- 锚点来源：`{metadata.get('anchor_source_version', 'N/A')}`",
        f"- 锚点检索策略：`{metadata.get('anchor_retrieval_strategy', 'N/A')}`",
        f"- 锚点 Top-k：`{metadata.get('anchor_top_k', 'N/A')}`",
        f"- 锚点过滤条件：`{metadata.get('anchor_filter_expr', 'N/A')}`",
        f"- 目标查询原始数：`{metadata.get('target_query_raw_count', 'N/A')}`",
        f"- 目标查询去重数：`{metadata.get('target_query_unique_count', 'N/A')}`",
        f"- 目标查询重复数：`{metadata.get('target_query_duplicate_count', 'N/A')}`",
        f"- 召回命中数：`{metadata.get('anchor_retrieval_hit_count', 'N/A')}`",
        f"- 全局唯一干净锚点数：`{detector.anchor_count_}`",
        f"- 锚点重复移除数：`{metadata.get('anchor_duplicate_count', 'N/A')}`",
        f"- 向量维度：`{detector.dimension}`",
        f"- beta：`{settings.beta}`",
        f"- clean quantile：`{settings.quantile}`",
        f"- 阈值：`{detector.threshold_:.12g}`",
        f"- 向量归一化：`{settings.normalize_embeddings}`",
        "",
        "## 分组结果",
        "",
        "| 评估单元 | 来源 CWE | 原始数 | 去重数 | 重复数 | 可检测数 | 缺失数 | 检出数 | 逃逸数 | 检测率 | 逃逸率 | 分数最小值 | 中位数 | 均值 | 最大值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {unit} | {cwes} | {raw} | {unique} | {duplicate} | "
            "{evaluable} | {missing} | {detected} | {escaped} | "
            "{detection_rate} | {escape_rate} | {score_min} | "
            "{score_median} | {score_mean} | {score_max} |".format(
                unit=_markdown_escape(summary["unit"]),
                cwes=_markdown_escape(", ".join(summary["source_cwes"])),
                raw=summary["raw_count"],
                unique=summary["unique_count"],
                duplicate=summary["duplicate_count"],
                evaluable=summary["evaluable_count"],
                missing=summary["missing_count"],
                detected=summary["detected_count"],
                escaped=summary["escaped_count"],
                detection_rate=_format_rate(summary["detection_rate"]),
                escape_rate=_format_rate(summary["escape_rate"]),
                score_min=_format_float(summary["score_min"]),
                score_median=_format_float(summary["score_median"]),
                score_mean=_format_float(summary["score_mean"]),
                score_max=_format_float(summary["score_max"]),
            )
        )

    lines.extend(
        [
            "",
            "## 逐文档结果（跨评估单元去重）",
            "",
            "| 文档 ID | 来源 CWE | 所属评估单元 | 分数 | 阈值 | 判定 |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for detail in details:
        score = detail["score"]
        detected = detail["detected"]
        verdict = (
            "DETECTED"
            if detected is True
            else ("ESCAPED" if detected is False else "SKIPPED_MISSING_CODE")
        )
        lines.append(
            "| {doc_id} | {cwes} | {units} | {score} | {threshold} | {verdict} |".format(
                doc_id=_markdown_escape(detail["doc_id"]),
                cwes=_markdown_escape(", ".join(detail["source_cwes"])),
                units=_markdown_escape(", ".join(detail["eval_units"])),
                score=_format_float(score),
                threshold=_format_float(detail["threshold"]),
                verdict=verdict,
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_evaluation(config_path: str) -> Path:
    started_at = datetime.now().astimezone()
    started_clock = time.perf_counter()
    root = _project_root()

    resolved_config_path = _resolve_path(root, config_path, "--config")
    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"mitigation config does not exist: {resolved_config_path}"
        )
    config = _require_mapping(
        load_yaml(str(resolved_config_path)) or {},
        "mitigation config",
    )
    data_cfg = _require_mapping(config.get("data"), "data")
    database_ref_cfg = _require_mapping(config.get("database"), "database")
    eval_cfg = _require_mapping(config.get("eval_index"), "eval_index")
    settings = _parse_settings(eval_cfg)

    poisoned_path = _resolve_path(
        root,
        data_cfg.get("poisoned_file_path"),
        "data.poisoned_file_path",
    )
    if not poisoned_path.is_file():
        raise FileNotFoundError(
            f"poisoned sample file does not exist: {poisoned_path}"
        )
    target_query_path = _resolve_path(
        root,
        data_cfg.get("target_query_file"),
        "data.target_query_file",
    )
    if not target_query_path.is_file():
        raise FileNotFoundError(
            f"target query file does not exist: {target_query_path}"
        )
    poisoned_data = load_json(str(poisoned_path))
    groups, configured_mode, effective_mode = build_evaluation_groups(
        poisoned_data,
        eval_cfg,
    )
    unique_samples = collect_unique_samples(groups)

    apr_config_path = _resolve_path(
        root,
        database_ref_cfg.get("milvus_config"),
        "database.milvus_config",
    )
    if not apr_config_path.is_file():
        raise FileNotFoundError(
            f"APR Milvus config does not exist: {apr_config_path}"
        )
    apr_config = _require_mapping(
        load_yaml(str(apr_config_path)) or {},
        "APR database config",
    )
    apr_database_cfg = _require_mapping(
        apr_config.get("database"),
        "APR database.database",
    )
    model_config_path = _resolve_path(
        root,
        apr_database_cfg.get("model"),
        "APR database.database.model",
    )
    if not model_config_path.is_file():
        raise FileNotFoundError(
            f"retriever model config does not exist: {model_config_path}"
        )
    model_config = _require_mapping(
        load_yaml(str(model_config_path)) or {},
        "retriever model config",
    )
    retriever_name = model_config_path.stem

    runtime_apr_config = copy.deepcopy(apr_config)
    runtime_database_cfg = _require_mapping(
        runtime_apr_config.get("database"),
        "APR database.database",
    )
    runtime_database_cfg["model"] = str(model_config_path)
    runtime_database_cfg["uri"] = str(
        _resolve_path(
            root,
            apr_database_cfg.get("uri"),
            "APR database.database.uri",
        )
    )

    cache_root = _resolve_path(
        root,
        data_cfg.get("detector_cache_dir"),
        "data.detector_cache_dir",
    )
    cache_dir = cache_root / retriever_name
    expected_metadata = build_expected_cache_metadata(
        retriever_name=retriever_name,
        model_config=model_config,
        model_config_path=model_config_path,
        apr_config=apr_config,
        apr_config_path=apr_config_path,
        target_query_path=target_query_path,
        settings=settings,
    )
    load_result = load_or_fit_detector(
        cache_dir=cache_dir,
        expected_metadata=expected_metadata,
        settings=settings,
        apr_config=runtime_apr_config,
        model_config=model_config,
        target_query_path=target_query_path,
    )

    evaluable_samples = [
        sample
        for sample in unique_samples
        if sample.poisoned_buggy_code is not None
    ]
    score_by_key: Dict[str, float] = {}
    if evaluable_samples:
        embedder = load_result.embedder or create_embedder(model_config)
        score_by_key = score_poison_samples(
            embedder,
            load_result.detector,
            unique_samples,
        )
    else:
        logger.warning(
            "No sample has retrieval_attack.poisoned_buggy_code; "
            "all selected samples will be logged as skipped"
        )

    threshold = load_result.detector.threshold_
    assert threshold is not None
    summaries = summarize_groups(groups, score_by_key, threshold)
    details = build_detail_rows(
        groups,
        unique_samples,
        score_by_key,
        threshold,
    )

    elapsed_seconds = time.perf_counter() - started_clock
    eval_cwe, _, _, _ = resolve_eval_scope(eval_cfg)
    database_uri = str(apr_database_cfg.get("uri", ""))
    markdown = render_markdown_log(
        config_path=resolved_config_path,
        poisoned_path=poisoned_path,
        target_query_path=target_query_path,
        apr_config_path=apr_config_path,
        model_config_path=model_config_path,
        database_uri=database_uri,
        retriever_name=retriever_name,
        configured_mode=configured_mode,
        effective_mode=effective_mode,
        eval_cwe=eval_cwe,
        settings=settings,
        cache_dir=cache_dir,
        load_result=load_result,
        summaries=summaries,
        details=details,
        started_at=started_at,
        elapsed_seconds=elapsed_seconds,
    )

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = root / "logs" / "evaluation" / "mitigation" / "index" / retriever_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{timestamp}.md"
    log_path.write_text(markdown, encoding="utf-8")
    logger.info("Index mitigation log saved to %s", log_path)
    return log_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate APR index-stage poison detection",
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation/mitigation.yml",
        help="path to the mitigation YAML config",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log_path = run_evaluation(args.config)
    print(f"Index mitigation evaluation completed: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
