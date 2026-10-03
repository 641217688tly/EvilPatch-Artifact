"""Cooperative cross-process persistence for generation evaluation results.

Each process owns one model for the lifetime of a run. Short file transactions
reload the latest JSON and apply only locally changed fields in that model's
selected victim slots. Readers/writers outside this protocol are not protected.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import logging
import math
import os
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Sequence

from src.evaluation.generation.racg import (
    RACG_RESULT_FIELDS,
    atomic_save_json,
    has_poisoned_hit,
    model_slot,
    validate_result_schema,
)
from src.utils.io import normalize_bfp_record_id

logger = logging.getLogger(__name__)
DEFAULT_RESULT_LOCK_TIMEOUT = 120.0
_MISSING = object()


class ResultPersistenceError(RuntimeError):
    """A result transaction cannot safely proceed."""


class ResultLockTimeout(ResultPersistenceError):
    """The result file remained locked past the configured deadline."""


class ModelRunConflict(ResultPersistenceError):
    """Another process is already evaluating this file and model."""


def resolve_lock_timeout(value: Any = DEFAULT_RESULT_LOCK_TIMEOUT) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("eval.result_lock_timeout must be a finite non-negative number")
    return float(value)


def canonical_result_path(path: str | Path) -> Path:
    return Path(os.path.normcase(str(Path(path).resolve())))


def _try_lock(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _sidecar_lock(path: Path, timeout: float) -> Iterator[None]:
    # Keep the sidecar inode stable across JSON replacements and process exits.
    # Its existence is not evidence that the lock is currently held.
    handle = path.open("a+b")
    acquired = False
    try:
        # Windows can lock a byte beyond EOF. Writing a sentinel before locking
        # would race with another process already holding that first byte.
        deadline = time.monotonic() + timeout
        announced = False
        while True:
            try:
                _try_lock(handle)
                acquired = True
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ResultLockTimeout(f"timed out waiting for result lock: {path}") from exc
                if not announced:
                    logger.info("waiting for result lock (timeout %.1fs): %s", timeout, path)
                    announced = True
                time.sleep(min(0.05, remaining))
        yield
    finally:
        try:
            if acquired:
                _unlock(handle)
        finally:
            handle.close()


@contextmanager
def result_file_lock(
    path: str | Path, timeout: float = DEFAULT_RESULT_LOCK_TIMEOUT
) -> Iterator[None]:
    target = canonical_result_path(path)
    with _sidecar_lock(target.with_name(target.name + ".write.lock"), resolve_lock_timeout(timeout)):
        yield


@contextmanager
def model_run_locks(paths: Sequence[str | Path], model_name: str) -> Iterator[None]:
    """Acquire every run lease before callers create/clear results or call APIs."""
    model_key = hashlib.sha256(model_name.encode("utf-8")).hexdigest()
    targets = sorted({canonical_result_path(path) for path in paths}, key=str)
    with ExitStack() as stack:
        for target in targets:
            lock_path = target.with_name(target.name + f".model-{model_key}.lock")
            try:
                stack.enter_context(_sidecar_lock(lock_path, 0.0))
            except ResultLockTimeout as exc:
                raise ModelRunConflict(
                    f"another evaluation is already running for model={model_name!r}, "
                    f"file={target}; no results were cleared and no API calls were started"
                ) from exc
        yield


def _index_results(results: Any) -> Dict[Any, Dict[str, Any]]:
    if not isinstance(results, list):
        raise ResultPersistenceError("result JSON must contain a list of queries")
    index = {}
    for entry in results:
        if not isinstance(entry, dict) or not isinstance(entry.get("entity"), dict):
            raise ResultPersistenceError("each result must contain an entity object")
        query_id = normalize_bfp_record_id(entry["entity"].get("id"))
        if query_id is None or isinstance(query_id, bool):
            raise ResultPersistenceError("each result must have a valid entity.id")
        if query_id in index:
            raise ResultPersistenceError(f"duplicate normalized query id={query_id!r}")
        if not isinstance(entry.get("retrieval_results"), list):
            raise ResultPersistenceError(f"query id={query_id!r}: retrieval_results must be a list")
        apr = entry.get("apr_results", {})
        if not isinstance(apr, dict):
            raise ResultPersistenceError(f"query id={query_id!r}: apr_results must be an object")
        for top_key, top_slot in apr.items():
            if str(top_key).startswith("top_") and (
                not isinstance(top_slot, dict)
                or any(not isinstance(slot, dict) for slot in top_slot.values())
            ):
                raise ResultPersistenceError(f"query id={query_id!r}: invalid model slots at {top_key}")
        index[query_id] = entry
    validate_result_schema(results)
    return index


def _read_unlocked(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)
        _index_results(results)
        return results
    except (OSError, ValueError, ResultPersistenceError) as exc:
        raise ResultPersistenceError(f"cannot read valid generation results from {path}: {exc}") from exc


def read_result_json(
    path: str | Path, timeout: float = DEFAULT_RESULT_LOCK_TIMEOUT
) -> List[Dict[str, Any]]:
    target = canonical_result_path(path)
    with result_file_lock(target, timeout):
        return _read_unlocked(target)


def load_or_create_result(
    path: str | Path,
    factory: Callable[[], List[Dict[str, Any]]],
    timeout: float = DEFAULT_RESULT_LOCK_TIMEOUT,
) -> tuple[List[Dict[str, Any]], bool]:
    """Check/create under the same lock used by all subsequent result saves."""
    target = canonical_result_path(path)
    with result_file_lock(target, timeout):
        if target.exists():
            return _read_unlocked(target), False
        results = factory()
        _index_results(results)
        atomic_save_json(target, results)
        return results, True


def _context_fingerprint(entry: Dict[str, Any]) -> str:
    entity = dict(entry["entity"])
    entity["id"] = normalize_bfp_record_id(entity["id"])
    payload = json.dumps(
        [entity, entry["retrieval_results"]], ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelSlotSaver:
    """No-argument save callback; own model run leases before constructing it.

    Result objects stay in place because RACG/Judge tasks hold references to
    them. Only the last successfully persisted local snapshot is advanced;
    unrelated disk updates never become local edits to replay on later saves.
    """

    def __init__(
        self, path: str | Path, results: List[Dict[str, Any]], model_name: str,
        top_ks: Sequence[int], timeout: float = DEFAULT_RESULT_LOCK_TIMEOUT,
        *, include_all_queries: bool = False,
    ) -> None:
        self.path = canonical_result_path(path)
        self.results = results
        self.model_name = model_name
        self.timeout = resolve_lock_timeout(timeout)
        self._thread_lock = threading.Lock()
        index = _index_results(results)
        self._contexts = {qid: _context_fingerprint(entry) for qid, entry in index.items()}
        self._keys = [
            (qid, k) for qid, entry in index.items()
            for k in dict.fromkeys(top_ks)
            if include_all_queries or has_poisoned_hit(entry, k)
        ]
        self._baseline = self._snapshot(index)

    def _snapshot(self, index: Dict[Any, Dict[str, Any]]) -> Dict[tuple, dict]:
        if index.keys() != self._contexts.keys():
            raise ResultPersistenceError(f"query population changed: {self.path}")
        return {
            (qid, k): copy.deepcopy(model_slot(index[qid], k, self.model_name))
            for qid, k in self._keys
        }

    def __call__(self) -> None:
        with self._thread_lock:
            local_index = _index_results(self.results)
            current = self._snapshot(local_index)
            changed = [key for key in self._keys if current[key] != self._baseline[key]]
            if not changed:
                return
            with result_file_lock(self.path, self.timeout):
                latest = _read_unlocked(self.path)
                disk_index = _index_results(latest)
                if disk_index.keys() != self._contexts.keys():
                    raise ResultPersistenceError(f"query population changed: {self.path}")
                for qid, k in changed:
                    expected = self._contexts[qid]
                    if (
                        _context_fingerprint(local_index[qid]) != expected
                        or _context_fingerprint(disk_index[qid]) != expected
                    ):
                        raise ResultPersistenceError(
                            f"query input or retrieval context changed: file={self.path}, query={qid!r}"
                        )
                    before, after = self._baseline[qid, k], current[qid, k]
                    slot = model_slot(disk_index[qid], k, self.model_name, create=True)
                    fields = {
                        field for field in before.keys() | after.keys()
                        if before.get(field, _MISSING) != after.get(field, _MISSING)
                    }
                    # Also reject stale Judge writes if an uncooperative writer
                    # replaced the patch while this process was judging it.
                    for field in fields | set(RACG_RESULT_FIELDS):
                        old = before.get(field, _MISSING)
                        desired = after.get(field, _MISSING)
                        actual = slot.get(field, _MISSING)
                        if actual != old and actual != desired:
                            raise ResultPersistenceError(
                                f"concurrent same-model update: file={self.path}, query={qid!r}, "
                                f"top_{k}.{self.model_name}.{field}"
                            )
                    for field in fields:
                        if field in after:
                            slot[field] = copy.deepcopy(after[field])
                        else:
                            slot.pop(field, None)
                atomic_save_json(self.path, latest)
                self._baseline = current
