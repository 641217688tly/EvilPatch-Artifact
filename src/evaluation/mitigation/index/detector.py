"""Index-stage poison detector based on shrinkage Mahalanobis distance."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union

import numpy as np


VectorInput = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]
BatchFactory = Callable[[], Iterable[VectorInput]]


class ShrinkageMahalanobisDetector:
    """Detect embedding outliers against a clean APR-anchor distribution.

    The caller supplies the clean anchor vectors (for index mitigation these are
    target-query-associated APR documents). The detector estimates their mean
    and sample covariance, shrinks the covariance toward an isotropic matrix,
    and calibrates the threshold from an anchor-score quantile. All numerical
    operations use ``float64``.
    """

    VERSION = "1.0"
    ARRAY_FILENAME = "detector.npz"
    METADATA_FILENAME = "metadata.json"

    def __init__(
        self,
        beta: float = 0.01,
        quantile: float = 0.95,
        normalize_embeddings: bool = True,
    ) -> None:
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError(f"beta must be numeric, got {beta!r}")
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
            raise TypeError(f"quantile must be numeric, got {quantile!r}")
        if not 0.0 < float(quantile) < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        if not isinstance(normalize_embeddings, bool):
            raise TypeError(
                "normalize_embeddings must be a boolean, "
                f"got {normalize_embeddings!r}"
            )

        self.beta = float(beta)
        self.quantile = float(quantile)
        self.normalize_embeddings = normalize_embeddings

        self.mean_: Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.denominator_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None
        self.anchor_count_: Optional[int] = None
        self.metadata: Dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return all(
            value is not None
            for value in (
                self.mean_,
                self.eigenvectors_,
                self.eigenvalues_,
                self.denominator_,
                self.threshold_,
                self.anchor_count_,
            )
        )

    @property
    def dimension(self) -> int:
        if self.mean_ is None:
            raise RuntimeError("detector is not fitted")
        return int(self.mean_.shape[0])

    def _prepare_vectors(
        self,
        vectors: VectorInput,
        *,
        expected_dim: Optional[int] = None,
    ) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(
                f"vectors must be a 2-D matrix, got shape={array.shape}"
            )
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"vectors cannot be empty, got shape={array.shape}")
        if expected_dim is not None and array.shape[1] != expected_dim:
            raise ValueError(
                f"embedding dimension mismatch: expected {expected_dim}, "
                f"got {array.shape[1]}"
            )
        if not np.isfinite(array).all():
            raise ValueError("vectors contain NaN or infinite values")

        if self.normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            if np.any(norms <= 0.0):
                raise ValueError("cannot normalize a zero embedding vector")
            array = array / norms
        return array

    def fit(self, clean_vectors: VectorInput) -> "ShrinkageMahalanobisDetector":
        """Fit and calibrate the detector from an in-memory clean matrix."""
        prepared = self._prepare_vectors(clean_vectors)
        return self.fit_from_batches(lambda: (prepared,))

    def fit_from_batches(
        self,
        batch_factory: BatchFactory,
    ) -> "ShrinkageMahalanobisDetector":
        """Fit from a repeatable clean-anchor-vector batch factory.

        The factory is consumed twice: once for streaming covariance
        estimation and once for clean-score threshold calibration. This keeps
        peak memory independent of the number of clean Milvus rows.
        """
        if not callable(batch_factory):
            raise TypeError("batch_factory must be callable")

        count = 0
        mean: Optional[np.ndarray] = None
        scatter: Optional[np.ndarray] = None
        dimension: Optional[int] = None

        for raw_batch in batch_factory():
            batch = self._prepare_vectors(raw_batch, expected_dim=dimension)
            if dimension is None:
                dimension = int(batch.shape[1])
                mean = np.zeros(dimension, dtype=np.float64)
                scatter = np.zeros((dimension, dimension), dtype=np.float64)

            assert mean is not None
            assert scatter is not None
            batch_count = int(batch.shape[0])
            batch_mean = batch.mean(axis=0)
            centered = batch - batch_mean
            batch_scatter = centered.T @ centered

            if count == 0:
                mean = batch_mean
                scatter = batch_scatter
                count = batch_count
                continue

            new_count = count + batch_count
            delta = batch_mean - mean
            scatter += batch_scatter
            scatter += np.outer(delta, delta) * (
                count * batch_count / new_count
            )
            mean += delta * (batch_count / new_count)
            count = new_count

        if dimension is None or mean is None or scatter is None:
            raise ValueError("clean vector source produced no batches")
        if count < 2:
            raise ValueError(
                f"at least two clean vectors are required, got {count}"
            )

        covariance = scatter / (count - 1)
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        average_variance = float(eigenvalues.sum() / dimension)
        denominator = (
            (1.0 - self.beta) * eigenvalues
            + self.beta * average_variance
        )
        numerical_floor = np.finfo(np.float64).eps * max(
            average_variance,
            1.0,
        )
        denominator = np.maximum(denominator, numerical_floor)

        self.mean_ = mean
        self.eigenvectors_ = eigenvectors
        self.eigenvalues_ = eigenvalues
        self.denominator_ = denominator
        self.anchor_count_ = count
        self.threshold_ = None

        clean_score_batches = []
        calibration_count = 0
        for raw_batch in batch_factory():
            batch = self._prepare_vectors(
                raw_batch,
                expected_dim=dimension,
            )
            scores = self.score(batch)
            clean_score_batches.append(scores)
            calibration_count += int(scores.shape[0])

        if calibration_count != count:
            raise RuntimeError(
                "clean vector source changed between fitting and calibration: "
                f"fit_count={count}, calibration_count={calibration_count}"
            )
        clean_scores = np.concatenate(clean_score_batches)
        self.threshold_ = float(np.quantile(clean_scores, self.quantile))
        return self

    def score(self, vectors: VectorInput) -> np.ndarray:
        """Return one shrinkage Mahalanobis score per vector."""
        if any(
            value is None
            for value in (
                self.mean_,
                self.eigenvectors_,
                self.denominator_,
            )
        ):
            raise RuntimeError("detector is not fitted")

        assert self.mean_ is not None
        assert self.eigenvectors_ is not None
        assert self.denominator_ is not None
        prepared = self._prepare_vectors(
            vectors,
            expected_dim=self.dimension,
        )
        projected = (prepared - self.mean_) @ self.eigenvectors_
        return np.sum(
            (projected * projected) / self.denominator_,
            axis=1,
            dtype=np.float64,
        )

    def predict(self, vectors: VectorInput) -> np.ndarray:
        """Return ``True`` for vectors whose score exceeds the threshold."""
        if self.threshold_ is None:
            raise RuntimeError("detector threshold is not calibrated")
        return self.score(vectors) > self.threshold_

    def save(
        self,
        cache_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Atomically persist detector arrays and JSON metadata."""
        if not self.is_fitted:
            raise RuntimeError("cannot save an unfitted detector")

        assert self.mean_ is not None
        assert self.eigenvectors_ is not None
        assert self.eigenvalues_ is not None
        assert self.denominator_ is not None
        assert self.threshold_ is not None
        assert self.anchor_count_ is not None

        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        arrays_path = directory / self.ARRAY_FILENAME
        metadata_path = directory / self.METADATA_FILENAME
        token = f"{os.getpid()}-{uuid.uuid4().hex}"
        arrays_tmp = directory / f".{self.ARRAY_FILENAME}.{token}.tmp"
        metadata_tmp = directory / f".{self.METADATA_FILENAME}.{token}.tmp"

        payload = dict(metadata or {})
        payload.update(
            {
                "detector_version": self.VERSION,
                "beta": self.beta,
                "quantile": self.quantile,
                "normalize_embeddings": self.normalize_embeddings,
                "dimension": self.dimension,
                "anchor_count": int(self.anchor_count_),
                "threshold": float(self.threshold_),
            }
        )

        try:
            with arrays_tmp.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    mean=self.mean_,
                    eigenvectors=self.eigenvectors_,
                    eigenvalues=self.eigenvalues_,
                    denominator=self.denominator_,
                    threshold=np.asarray(self.threshold_, dtype=np.float64),
                    anchor_count=np.asarray(self.anchor_count_, dtype=np.int64),
                )
            metadata_tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(arrays_tmp, arrays_path)
            os.replace(metadata_tmp, metadata_path)
        finally:
            arrays_tmp.unlink(missing_ok=True)
            metadata_tmp.unlink(missing_ok=True)

        self.metadata = payload

    @classmethod
    def load(
        cls,
        cache_dir: Union[str, Path],
    ) -> "ShrinkageMahalanobisDetector":
        """Load a persisted detector without accessing the APR database."""
        directory = Path(cache_dir)
        arrays_path = directory / cls.ARRAY_FILENAME
        metadata_path = directory / cls.METADATA_FILENAME
        if not arrays_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"detector cache is incomplete under {directory}"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise TypeError("detector metadata must be a JSON object")
        if metadata.get("detector_version") != cls.VERSION:
            raise ValueError(
                "unsupported detector cache version: "
                f"{metadata.get('detector_version')!r}"
            )

        normalize_embeddings = metadata.get("normalize_embeddings")
        if not isinstance(normalize_embeddings, bool):
            raise TypeError(
                "cached normalize_embeddings must be a boolean"
            )
        detector = cls(
            beta=float(metadata["beta"]),
            quantile=float(metadata["quantile"]),
            normalize_embeddings=normalize_embeddings,
        )
        with np.load(arrays_path, allow_pickle=False) as arrays:
            required = {
                "mean",
                "eigenvectors",
                "eigenvalues",
                "denominator",
                "threshold",
                "anchor_count",
            }
            missing = required.difference(arrays.files)
            if missing:
                raise ValueError(
                    f"detector cache is missing arrays: {sorted(missing)}"
                )
            detector.mean_ = np.asarray(arrays["mean"], dtype=np.float64)
            detector.eigenvectors_ = np.asarray(
                arrays["eigenvectors"],
                dtype=np.float64,
            )
            detector.eigenvalues_ = np.asarray(
                arrays["eigenvalues"],
                dtype=np.float64,
            )
            detector.denominator_ = np.asarray(
                arrays["denominator"],
                dtype=np.float64,
            )
            detector.threshold_ = float(
                np.asarray(arrays["threshold"]).reshape(())
            )
            detector.anchor_count_ = int(
                np.asarray(arrays["anchor_count"]).reshape(())
            )

        dimension = int(metadata["dimension"])
        if detector.mean_.shape != (dimension,):
            raise ValueError("cached mean has an invalid shape")
        if detector.eigenvectors_.shape != (dimension, dimension):
            raise ValueError("cached eigenvectors have an invalid shape")
        if detector.eigenvalues_.shape != (dimension,):
            raise ValueError("cached eigenvalues have an invalid shape")
        if detector.denominator_.shape != (dimension,):
            raise ValueError("cached denominator has an invalid shape")
        if not all(
            np.isfinite(array).all()
            for array in (
                detector.mean_,
                detector.eigenvectors_,
                detector.eigenvalues_,
                detector.denominator_,
            )
        ):
            raise ValueError("detector cache contains NaN or infinite values")
        if not np.isfinite(detector.threshold_):
            raise ValueError("detector cache contains an invalid threshold")
        if detector.anchor_count_ != int(metadata["anchor_count"]):
            raise ValueError("detector cache anchor_count is inconsistent")
        if detector.threshold_ != float(metadata["threshold"]):
            raise ValueError("detector cache threshold is inconsistent")

        detector.metadata = metadata
        return detector
