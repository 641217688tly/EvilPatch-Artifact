#!/usr/bin/env python3
"""Report saved vulnerability-injection verdicts without modifying target data.

Usage: python -m src.evaluation.vinj.vinj_attack_eval --config configs/evaluation/vinj.yml
CWE-MIXED is a count-weighted aggregate of the selected ordinary CWE groups,
not an independently generated mixed-attack group from the source file.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from src.utils.log import setup_logging

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "configs/evaluation/vinj.yml"
_CWE_PATTERN = re.compile(r"CWE-[0-9]+")
_VERDICT_FIELD = "generation_attack.vinj_attack.is_vulnerable"


def _mapping(value: Any, location: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object/mapping")
    return value


def _resolve_path(value: Any, root: Path, location: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty path string")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def select_cwes(data: Any, requested: Any = None) -> List[str]:
    """Keep source/config order; never count an independent CWE-MIXED group."""
    data = _mapping(data, "Target JSON root")
    if requested is None:
        selected = [key for key in data if _CWE_PATTERN.fullmatch(key)]
    else:
        if not isinstance(requested, list) or not requested:
            raise ValueError("eval.eval_cwe must be null or a non-empty CWE list")
        selected = []
        for cwe in requested:
            if not isinstance(cwe, str) or not _CWE_PATTERN.fullmatch(cwe):
                raise ValueError(
                    f"Invalid eval.eval_cwe entry {cwe!r}; use ordinary CWE-<digits> "
                    "groups only (CWE-MIXED is added automatically)"
                )
            if cwe in selected:
                raise ValueError(f"Duplicate eval.eval_cwe entry: {cwe}")
            if cwe not in data:
                raise ValueError(f"Requested CWE group not found: {cwe}")
            selected.append(cwe)
    if not selected:
        raise ValueError("No ordinary CWE-<digits> groups available for evaluation")
    return selected


def _saved_verdict(record: Any, location: str) -> Optional[bool]:
    """Missing/null attack containers or verdicts are pending, not failures."""
    current = _mapping(record, location)
    for key in ("generation_attack", "vinj_attack"):
        location = f"{location}.{key}"
        value = current.get(key)
        if value is None:
            return None
        current = _mapping(value, location)
    verdict = current.get("is_vulnerable")
    if verdict is not None and not isinstance(verdict, bool):
        raise ValueError(f"{location}.is_vulnerable must be true, false or null")
    return verdict


def _result_row(cwe: str, success: int, failure: int, pending: int) -> Dict[str, Any]:
    judged = success + failure
    return {
        "cwe": cwe,
        "targets": judged + pending,
        "judged": judged,
        "success": success,
        "failure": failure,
        "pending": pending,
        "success_rate": round(success / judged, 6) if judged else None,
        "failure_rate": round(failure / judged, 6) if judged else None,
    }


def evaluate_targets(data: Any, requested: Any = None) -> List[Dict[str, Any]]:
    """Count each saved target record; no BFP-ID deduplication or re-judging."""
    selected = select_cwes(data, requested)
    if "CWE-MIXED" in data:
        logger.warning(
            "Ignoring source CWE-MIXED group; report CWE-MIXED aggregates only "
            "the selected ordinary CWE groups."
        )
    rows = []
    for cwe in selected:
        records = data[cwe]
        if not isinstance(records, list):
            raise ValueError(f"{cwe} must contain a list of target records")
        success = failure = pending = 0
        for index, record in enumerate(records):
            verdict = _saved_verdict(record, f"{cwe}[{index}]")
            if verdict is True:
                success += 1
            elif verdict is False:
                failure += 1
            else:
                pending += 1
        rows.append(_result_row(cwe, success, failure, pending))
    rows.append(_result_row(
        "CWE-MIXED",
        sum(row["success"] for row in rows),
        sum(row["failure"] for row in rows),
        sum(row["pending"] for row in rows),
    ))
    return rows


def render_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Use the same eight-column table in the console and Markdown report."""
    lines = [
        "| CWE | Targets | Judged | Success | Failure | Pending | Success Rate | Failure Rate |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = [str(row[key]) for key in (
            "cwe", "targets", "judged", "success", "failure", "pending"
        )]
        cells.extend(
            "N/A" if row[key] is None else f"{row[key]:.2%}"
            for key in ("success_rate", "failure_rate")
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_evaluation(config_path: str = DEFAULT_CONFIG) -> Path:
    """Read input once, validate it, and write reports to a new run directory."""
    config_file = _resolve_path(config_path, PROJECT_ROOT, "Config path")
    with config_file.open(encoding="utf-8-sig") as stream:
        config = _mapping(yaml.safe_load(stream), "Config root")
    data_cfg = _mapping(config.get("data"), "data")
    eval_cfg = _mapping(config.get("eval", {}), "eval")
    log_cfg = _mapping(config.get("logging", {}), "logging")
    verbose = log_cfg.get("verbose", False)
    if not isinstance(verbose, bool):
        raise ValueError("logging.verbose must be a boolean")
    setup_logging(logging.DEBUG if verbose else logging.INFO)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    target_file = _resolve_path(
        data_cfg.get("target_file_path"), PROJECT_ROOT, "data.target_file_path"
    )
    output_root = _resolve_path(
        log_cfg.get("output_dir", "logs/evaluation/vinj"), PROJECT_ROOT,
        "logging.output_dir",
    )
    logger.debug("Config: %s; target: %s", config_file, target_file)
    with target_file.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    rows = evaluate_targets(data, eval_cfg.get("eval_cwe"))
    summary = {
        "target_file_path": str(target_file),
        "evaluated_cwes": [row["cwe"] for row in rows[:-1]],
        "verdict_field": _VERDICT_FIELD,
        "counting_unit": "target record within each selected CWE; no ID deduplication",
        "rate_denominator": "judged = success + failure; pending excluded",
        "mixed_semantics": "sum selected ordinary CWE counts, then compute rates",
        "ignored_source_mixed": "CWE-MIXED" in data,
        "results": rows,
    }
    table = render_table(rows)
    overview = (
        "# Vulnerability Injection Evaluation\n\n"
        "Rates use judged targets (success + failure); pending targets are excluded.\n\n"
        "CWE-MIXED aggregates selected ordinary CWE records without ID deduplication. "
        "It is not the independent mixed-attack group in the source file.\n\n"
        + table + "\n"
    )
    run_dir = output_root / f"{target_file.stem}-{datetime.now():%Y-%m-%d-%H-%M-%S-%f}"
    run_dir.mkdir(parents=True, exist_ok=False)
    # A new directory and exclusive creation prevent overwriting previous reports.
    with (run_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (run_dir / "overview.md").open("x", encoding="utf-8") as stream:
        stream.write(overview)
    print(table)
    logger.info("Reports saved to %s", run_dir)
    return run_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Evaluation YAML path")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1
    try:
        run_evaluation(args.config)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        logger.error("Vulnerability injection evaluation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
