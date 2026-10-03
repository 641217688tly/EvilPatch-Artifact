"""CrystalBLEU utilities for local functionality-consistency evaluation."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import tempfile
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_C_CPP_NUMERIC_LITERAL = r"""
(?:
    0[xX](?:[0-9A-Fa-f]+(?:\.[0-9A-Fa-f]*)?|\.[0-9A-Fa-f]+)(?:[pP][+-]?\d+)?
    | 0[bB][01]+
    | (?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?
)
(?:[uUlLfF]+)?
"""
_TOKEN_RE = re.compile(
    rf"[A-Za-z_][A-Za-z0-9_]*|{_C_CPP_NUMERIC_LITERAL}|[^\sA-Za-z0-9_]",
    re.VERBOSE,
)


def default_code_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text or "")


def _corpus_signature(corpus_texts: Sequence[str], k: int, max_n: int) -> str:
    digest = hashlib.sha1()
    digest.update(
        (
            f"k={k}|max_n={max_n}|n={len(corpus_texts)}"
            f"|tokenizer_pattern={_TOKEN_RE.pattern}|tokenizer_flags={_TOKEN_RE.flags}"
        ).encode("utf-8")
    )
    for text in corpus_texts:
        digest.update(b"\x1e")
        digest.update((text or "").encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:16]


def make_corpus_cache_path(
    cache_dir: str,
    corpus_path: str,
    k: int,
    max_n: int,
    corpus_texts: Sequence[str],
) -> str:
    stem = os.path.splitext(os.path.basename(corpus_path))[0] or "corpus"
    signature = _corpus_signature(corpus_texts, k, max_n)
    return os.path.join(cache_dir, f"{stem}_k{k}_n{max_n}_{signature}.pkl")


def compute_trivially_shared_ngrams(
    corpus_texts: Sequence[str],
    k: int = 500,
    max_n: int = 4,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    cache_path: Optional[str] = None,
) -> Dict[Tuple[str, ...], int]:
    tokenizer = tokenizer or default_code_tokenize
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as handle:
                cached = pickle.load(handle)
            if isinstance(cached, dict):
                return cached
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrystalBLEU cache load failed (%s): %s", cache_path, exc)

    try:
        from nltk.util import ngrams
    except ImportError as exc:  # pragma: no cover
        raise ImportError("nltk is required for CrystalBLEU evaluation") from exc

    frequencies: Counter[Tuple[str, ...]] = Counter()
    for text in corpus_texts:
        tokens = tokenizer(text)
        for n in range(1, max_n + 1):
            if len(tokens) >= n:
                frequencies.update(ngrams(tokens, n))
    result = dict(frequencies.most_common(k))
    if cache_path:
        directory = os.path.dirname(os.path.abspath(cache_path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".crystalbleu-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                pickle.dump(result, handle)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temporary, cache_path)
            except OSError as exc:
                # Concurrent Windows readers may briefly prevent replacement;
                # the freshly computed result is still valid for this run.
                logger.warning("CrystalBLEU cache save failed (%s): %s", cache_path, exc)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return result


def crystal_bleu(
    reference: str,
    hypothesis: str,
    trivially_shared_ngrams: Dict[Tuple[str, ...], int],
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    smoothing: bool = True,
) -> float:
    if not reference or not hypothesis:
        return 0.0
    tokenizer = tokenizer or default_code_tokenize
    ref_tokens = tokenizer(reference)
    hyp_tokens = tokenizer(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    try:
        from crystalbleu import corpus_bleu
    except ImportError as exc:  # pragma: no cover
        raise ImportError("crystalbleu is required for functionality evaluation") from exc

    smoothing_function = None
    if smoothing:
        from nltk.translate.bleu_score import SmoothingFunction

        smoothing_function = SmoothingFunction().method1
    kwargs: Dict[str, Any] = {"ignoring": trivially_shared_ngrams}
    if smoothing_function is not None:
        kwargs["smoothing_function"] = smoothing_function
    return float(corpus_bleu([[ref_tokens]], [hyp_tokens], **kwargs))


def score_triplet(
    ground_truth: str,
    clean_patch: str,
    poisoned_patch: str,
    trivially_shared_ngrams: Dict[Tuple[str, ...], int],
) -> Dict[str, float]:
    clean = crystal_bleu(ground_truth, clean_patch, trivially_shared_ngrams)
    poisoned = crystal_bleu(ground_truth, poisoned_patch, trivially_shared_ngrams)
    return {
        "clean": round(clean, 6),
        "poisoned": round(poisoned, 6),
        "delta": round(abs(clean - poisoned), 6),
    }


def summarize_local(scores: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    def average(key: str) -> Optional[float]:
        values = [float(item[key]) for item in scores]
        return round(sum(values) / len(values), 6) if values else None

    return {
        "mean_clean": average("clean"),
        "mean_poisoned": average("poisoned"),
        "mean_delta": average("delta"),
        "victim_count": len(scores),
    }
