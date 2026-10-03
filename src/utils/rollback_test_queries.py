"""Snapshot-bound rollback of the mis-commented test queries.

Restores ``buggy_code`` and ``fixed_code`` of the reviewed records in
``data/query/black/test/test_query_set_v4.json`` to the text of the pre-format
CoCoNut release (``src/preprocessing/bfp/CoCoNut/dataset/c/processed/CoCoNut.jsonl``).
Only the fields named in the review manifest are touched; ``id``, ``language``,
``source`` and every other record stay byte-identical.

The pre-format release itself still contains situation-1 hiding (a ``//`` comment
that swallows the rest of the single physical line), which the caller explicitly
accepted for this step; this script performs no reformatting and no semantic
repair.

Preview: python -X utf8 src/utils/rollback_test_queries.py --check
Apply:   python -X utf8 src/utils/rollback_test_queries.py --apply
Restore: python -X utf8 src/utils/rollback_test_queries.py --restore <timestamp>

Run only while no writer is touching the query file. Backups and the execution
report are kept even on failure; a second application fails closed because the
recorded snapshot no longer matches.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "analysis/test_query_review_2026-09-22/affected_queries.json"
TARGET = "data/query/black/test/test_query_set_v4.json"
SOURCE = "src/preprocessing/bfp/CoCoNut/dataset/c/processed/CoCoNut.jsonl"
BACKUP_ROOT = "analysis/test_query_rollback_backups"
FIELDS = ("buggy_code", "fixed_code")


class RollbackError(ValueError):
    """Unsafe input, snapshot drift, or a failed transaction."""


def require(condition, message):
    if not condition:
        raise RollbackError(message)


def text_hash(value: str) -> str:
    require(isinstance(value, str), "code must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_stream(text: str) -> str:
    """Whitespace-, quote- and comment-marker-insensitive character stream."""
    body = "".join(text.split())
    for token in ("/*", "*/", "//", '"', "'", "*", "/"):
        body = body.replace(token, "")
    return body


def read_source_rows(needed):
    rows = {}
    path = ROOT / SOURCE
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if number in needed:
                obj = json.loads(line)
                rows[number] = obj
                if len(rows) == len(needed):
                    break
    missing = sorted(set(needed) - set(rows))
    require(not missing, f"pre-format rows not found: {missing[:5]}")
    return rows


def load_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(manifest.get("records"), list), "manifest has no records list")
    return manifest


def prepare(manifest_path: Path, target_path: Path):
    """Read-only preflight; every check must pass before anything is written."""
    manifest = load_manifest(manifest_path)
    target_bytes = target_path.read_bytes()
    data = json.loads(target_bytes.decode("utf-8"))
    before_hash = hashlib.sha256(target_bytes).hexdigest()

    records = manifest["records"]
    require(records, "manifest is empty")
    rows = read_source_rows({rec["source_row"] for rec in records})

    plan = []
    seen = set()
    for rec in records:
        cwe, index, ident = rec["cwe"], rec["index"], rec["id"]
        key = (cwe, index)
        require(key not in seen, f"duplicate manifest entry: {cwe}#{index}")
        seen.add(key)
        require(cwe in data, f"missing CWE group: {cwe}")
        require(0 <= index < len(data[cwe]), f"index out of range: {cwe}#{index}")
        live = data[cwe][index]
        require(live.get("id") == ident,
                f"record moved: {cwe}#{index} holds id={live.get('id')}, manifest {ident}")
        row = rows[rec["source_row"]]
        for field in FIELDS:
            if field not in rec["fields"]:
                continue
            spec = rec["fields"][field]
            current = live.get(field)
            require(isinstance(current, str), f"{cwe}#{index}.{field} is not a string")
            require(text_hash(current) == spec["current_sha256"],
                    f"{cwe}#{index}.{field} changed since the review")
            target_text = row[field]
            require(text_hash(target_text) == spec["target_sha256"],
                    f"{cwe}#{index}.{field}: pre-format row {rec['source_row']} changed")
            require(content_stream(current) == content_stream(target_text),
                    f"{cwe}#{index}.{field} no longer matches source row {rec['source_row']}")
            plan.append({"cwe": cwe, "index": index, "id": ident, "field": field,
                         "source_row": rec["source_row"],
                         "before_sha256": spec["current_sha256"],
                         "after_sha256": spec["target_sha256"],
                         "before_chars": len(current), "after_chars": len(target_text),
                         "reason": spec.get("reason", "")})

    after = copy.deepcopy(data)
    for item in plan:
        row = rows[item["source_row"]]
        after[item["cwe"]][item["index"]][item["field"]] = row[item["field"]]
    after_bytes = json.dumps(after, ensure_ascii=False, indent=2).encode("utf-8")
    return {"manifest": manifest, "plan": plan, "data": data, "after": after,
            "after_bytes": after_bytes, "before_hash": before_hash,
            "target_path": target_path,
            "manifest_hash": file_hash(manifest_path),
            "source_hash": file_hash(ROOT / SOURCE)}


def summarise(plan):
    by_cwe = Counter(item["cwe"] for item in plan)
    by_field = Counter(item["field"] for item in plan)
    return {"fields": len(plan),
            "queries": len({(i["cwe"], i["index"]) for i in plan}),
            "by_cwe": dict(sorted(by_cwe.items())), "by_field": dict(by_field)}


def apply(prepared):
    target_path = prepared["target_path"]
    require(file_hash(target_path) == prepared["before_hash"],
            "target file changed between check and apply")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup_dir = ROOT / BACKUP_ROOT / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup = backup_dir / "test_query_set_v4.before.json"
    shutil.copy2(target_path, backup)
    require(file_hash(backup) == prepared["before_hash"], "backup hash mismatch")

    temp = None
    try:
        handle = tempfile.NamedTemporaryFile(dir=str(target_path.parent), delete=False,
                                             suffix=".tmp")
        temp = Path(handle.name)
        handle.write(prepared["after_bytes"])
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        require(file_hash(target_path) == prepared["before_hash"],
                "target file changed while writing")
        os.replace(temp, target_path)
        temp = None
    finally:
        if temp is not None and temp.exists():
            temp.unlink()

    written = json.loads(target_path.read_text(encoding="utf-8"))
    require(len(written) == len(prepared["data"]), "group count changed")
    for cwe, items in written.items():
        require(len(items) == len(prepared["data"][cwe]), f"record count changed in {cwe}")
    changed = {(i["cwe"], i["index"], i["field"]) for i in prepared["plan"]}
    for cwe, items in written.items():
        for index, record in enumerate(items):
            original = prepared["data"][cwe][index]
            for field in FIELDS:
                if (cwe, index, field) in changed:
                    require(record[field] == prepared["after"][cwe][index][field],
                            f"rollback not applied: {cwe}#{index}.{field}")
                else:
                    require(record[field] == original[field],
                            f"unexpected change: {cwe}#{index}.{field}")
            require(record["id"] == original["id"], f"id changed: {cwe}#{index}")
            require(record["source"] == original["source"],
                    f"source changed: {cwe}#{index}")

    after_hash = file_hash(target_path)
    report = {"applied_at": stamp, "target": TARGET, "manifest": MANIFEST,
              "manifest_sha256": prepared["manifest_hash"],
              "source": SOURCE, "source_sha256": prepared["source_hash"],
              "backup": str(backup.relative_to(ROOT)).replace("\\", "/"),
              "sha256_before": prepared["before_hash"], "sha256_after": after_hash,
              "summary": summarise(prepared["plan"]), "plan": prepared["plan"]}
    (backup_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return backup_dir, report


def restore(stamp):
    backup_dir = ROOT / BACKUP_ROOT / stamp
    require(backup_dir.is_dir(), f"backup directory not found: {backup_dir}")
    backup = backup_dir / "test_query_set_v4.before.json"
    report = json.loads((backup_dir / "report.json").read_text(encoding="utf-8"))
    target_path = ROOT / TARGET
    current = file_hash(target_path)
    require(current == report["sha256_after"],
            "target file changed after the rollback; refusing to overwrite")
    shutil.copy2(backup, target_path)
    require(file_hash(target_path) == report["sha256_before"], "restore hash mismatch")
    return backup_dir, report["sha256_before"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="preflight only (default)")
    group.add_argument("--apply", action="store_true", help="backup and write")
    group.add_argument("--restore", metavar="TIMESTAMP", help="restore a backup")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--target", default=TARGET)
    args = parser.parse_args(argv)

    if args.restore:
        backup_dir, digest = restore(args.restore)
        print(f"restored {TARGET} from {backup_dir} (sha256 {digest})")
        return 0

    manifest_path = (ROOT / args.manifest).resolve()
    target_path = (ROOT / args.target).resolve()
    require(manifest_path.is_file(), f"manifest not found: {manifest_path}")
    require(target_path.is_file(), f"target not found: {target_path}")
    prepared = prepare(manifest_path, target_path)
    summary = summarise(prepared["plan"])
    print(f"queries to roll back : {summary['queries']}")
    print(f"fields to roll back  : {summary['fields']} {summary['by_field']}")
    print(f"per CWE              : {summary['by_cwe']}")
    print(f"target file          : {target_path}")
    print(f"sha256 before        : {prepared['before_hash']}")
    if not args.apply:
        print("preflight only; re-run with --apply to write")
        return 0
    backup_dir, report = apply(prepared)
    print(f"applied; backup      : {backup_dir}")
    print(f"sha256 after         : {report['sha256_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
