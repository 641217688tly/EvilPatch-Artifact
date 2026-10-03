"""Small, model-scoped checkpoints for functionality evaluation.

The checkpoint is durable after each completed result. The much larger shared
retrieval JSON is merged only at the end of a unit, under the generation
evaluation's cross-process file lock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.evaluation.generation.persistence import ResultPersistenceError
from src.evaluation.generation.racg import atomic_save_json, model_slot
from src.utils.io import normalize_bfp_record_id

logger = logging.getLogger(__name__)
_FIELDS = ("prompt", "patch", "CrystalBLEU")


def _query_key(value: Any) -> str:
    normalized = normalize_bfp_record_id(value)
    if normalized is None or isinstance(normalized, bool):
        raise ResultPersistenceError(f"invalid checkpoint query id={value!r}")
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _fingerprint(entry: Dict[str, Any]) -> str:
    entity = dict(entry["entity"])
    entity["id"] = normalize_bfp_record_id(entity["id"])
    payload = json.dumps(
        [entity, entry["retrieval_results"]], ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _directory(clean_path: str | Path, model_name: str) -> Path:
    path = Path(clean_path)
    model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:16]
    return path.parent / ".functionality_progress" / f"{path.name}.{model_hash}"


def clear_checkpoint_top_k(clean_path: str | Path, model_name: str, top_k: int) -> int:
    """Remove only this file/model/depth's unmerged progress records."""
    directory = _directory(clean_path, model_name)
    removed = 0
    for path in directory.glob(f"top-{top_k}-*.json"):
        path.unlink()
        removed += 1
    return removed


class FunctionalityCheckpoint:
    def __init__(
        self, clean_path: str | Path, results: List[Dict[str, Any]],
        model_name: str, top_ks: Sequence[int],
    ) -> None:
        self.directory = _directory(clean_path, model_name)
        self.results = results
        self.model_name = model_name
        self.top_ks = tuple(dict.fromkeys(top_ks))
        self.index: Dict[str, Dict[str, Any]] = {}
        self.fingerprints: Dict[str, str] = {}
        for entry in results:
            key = _query_key(entry["entity"]["id"])
            if key in self.index:
                raise ResultPersistenceError(f"duplicate checkpoint query id={key}")
            self.index[key] = entry
            self.fingerprints[key] = _fingerprint(entry)
        self.saved: Dict[Tuple[str, int], Dict[str, Any]] = {
            (key, top_k): self._slot_fields(entry, top_k)
            for key, entry in self.index.items() for top_k in self.top_ks
        }
        self.last_save_seconds = 0.0
        self.is_saving = False
        self.checkpointed = 0

    def _slot_fields(self, entry: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        slot = model_slot(entry, top_k, self.model_name)
        return {field: slot[field] for field in _FIELDS if field in slot}

    def _record_path(self, key: str, top_k: int) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"top-{top_k}-{digest}.json"

    def replay(self) -> int:
        """Apply durable, context-checked records to the in-memory clean JSON."""
        count = 0
        for top_k in self.top_ks:
            for path in sorted(self.directory.glob(f"top-{top_k}-*.json")):
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        record = json.load(handle)
                    if not isinstance(record, dict) or record.get("version") != 1:
                        raise ValueError("unsupported checkpoint format")
                    key = _query_key(record["query_id"])
                    record_top_k = record["top_k"]
                    fields = record["fields"]
                    if (
                        record.get("model_name") != self.model_name
                        or key not in self.index
                        or record_top_k != top_k
                        or record.get("fingerprint") != self.fingerprints[key]
                        or not isinstance(fields, dict)
                        or any(field not in _FIELDS for field in fields)
                        or path != self._record_path(key, top_k)
                    ):
                        raise ValueError("checkpoint does not match the current clean result")
                    slot = model_slot(self.index[key], top_k, self.model_name, create=True)
                    for field in _FIELDS:
                        if field in fields:
                            slot[field] = fields[field]
                        else:
                            slot.pop(field, None)
                    self.saved[key, top_k] = self._slot_fields(self.index[key], top_k)
                    count += 1
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    raise ResultPersistenceError(f"invalid functionality checkpoint {path}: {exc}") from exc
        if count:
            logger.info("replayed %d functionality checkpoints from %s", count, self.directory)
        return count

    def save_record(self, query_id: Any, top_k: int) -> bool:
        key = _query_key(query_id)
        if key not in self.index or top_k not in self.top_ks:
            raise ResultPersistenceError(f"checkpoint key is outside the selected clean result: {key}, top-{top_k}")
        fields = self._slot_fields(self.index[key], top_k)
        if fields == self.saved[key, top_k]:
            return False
        started = time.monotonic()
        self.is_saving = True
        try:
            atomic_save_json(
                self._record_path(key, top_k),
                {
                    "version": 1,
                    "model_name": self.model_name,
                    "query_id": normalize_bfp_record_id(query_id),
                    "top_k": top_k,
                    "fingerprint": self.fingerprints[key],
                    "fields": fields,
                },
            )
            self.saved[key, top_k] = fields
            self.checkpointed += 1
            return True
        finally:
            self.last_save_seconds = time.monotonic() - started
            self.is_saving = False

    def save_changed(self) -> int:
        changed = 0
        for key in self.index:
            for top_k in self.top_ks:
                changed += int(self.save_record(json.loads(key), top_k))
        return changed

    def discard(self) -> None:
        """Call only after both shared result files have been durably merged."""
        for top_k in self.top_ks:
            for path in self.directory.glob(f"top-{top_k}-*.json"):
                path.unlink()
        if self.directory.exists() and not any(self.directory.iterdir()):
            self.directory.rmdir()
