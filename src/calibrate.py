"""
calibrate.py
------------
Confidence calibration module for LearnMate.

Implements:
- Platt Scaling: Logistic Regression mapping features to P(correct).
- Isotonic Regression: Non-parametric monotonic calibration.
- Calibrator persistence and serialization.
- Before/After calibration evaluation (ECE, MCE, Brier Score).

RULES:
- Calibration models must be fit ONLY on the `calib` split (never on `tune` or `dev`).
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features import FeatureExtractor, extract_features_and_labels
from src.metrics import compute_calibration_metrics
from src.utils import get_logger, load_config, seed_everything

log = get_logger(__name__)


class Calibrator:
    """
    Confidence calibrator supporting Platt scaling and Isotonic regression.
    """

    def __init__(
        self,
        method: str = "platt",
        random_state: int = 42,
    ):
        self.method = method.lower()
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = None
        self.platt_model: Optional[LogisticRegression] = None
        self.isotonic_model: Optional[IsotonicRegression] = None
        self.is_fitted: bool = False

    @classmethod
    def from_config(cls, cfg) -> "Calibrator":
        method = str(cfg.calibration.get("method", "platt"))
        seed = int(cfg.project.get("seed", 42))
        return cls(method=method, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Calibrator":
        """
        Fit calibrator on calibration split data (X, y).

        Parameters
        ----------
        X : (N, d) feature matrix
        y : (N,) binary correctness labels in {0, 1}
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32).ravel()

        assert len(X) == len(y), f"Mismatch in samples: X={len(X)}, y={len(y)}"
        assert len(np.unique(y)) <= 2, f"Target y must be binary, got classes: {np.unique(y)}"

        # Edge case: if single class in calibration data
        if len(np.unique(y)) < 2:
            log.warning("Calibration data contains only 1 class (%s). Calibrator will predict constant.", np.unique(y))
            self.scaler = None
            self.platt_model = None
            self.is_fitted = True
            self._constant_prob = float(y[0])
            return self

        self._constant_prob = None

        if self.method == "platt":
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.platt_model = LogisticRegression(
                random_state=self.random_state,
                solver="lbfgs",
                max_iter=1000,
            )
            self.platt_model.fit(X_scaled, y)

        elif self.method == "isotonic":
            # For 1D feature: direct isotonic regression
            # For multi-D features: map X via Platt logistic regression first, then fit isotonic on logits
            if X.shape[1] == 1:
                self.scaler = None
                self.platt_model = None
                self.isotonic_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                self.isotonic_model.fit(X.ravel(), y)
            else:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                self.platt_model = LogisticRegression(
                    random_state=self.random_state,
                    solver="lbfgs",
                    max_iter=1000,
                )
                self.platt_model.fit(X_scaled, y)
                # Fit isotonic on the Platt probability predictions
                platt_probs = self.platt_model.predict_proba(X_scaled)[:, 1]
                self.isotonic_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                self.isotonic_model.fit(platt_probs, y)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        log.info("Fitted %s calibrator on %d samples.", self.method.upper(), len(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities P(correct) in [0, 1].

        Returns
        -------
        1D float32 array of shape (N,)
        """
        assert self.is_fitted, "Calibrator is not fitted yet. Call fit() first."
        X = np.asarray(X, dtype=np.float32)

        if self._constant_prob is not None:
            return np.full(len(X), self._constant_prob, dtype=np.float32)

        if self.method == "platt":
            assert self.scaler is not None and self.platt_model is not None
            X_scaled = self.scaler.transform(X)
            probs = self.platt_model.predict_proba(X_scaled)[:, 1]
            return np.clip(probs, 0.0, 1.0).astype(np.float32)

        elif self.method == "isotonic":
            if self.platt_model is None:
                # 1D direct isotonic
                probs = self.isotonic_model.predict(X.ravel())
            else:
                # Multi-D: platt then isotonic
                X_scaled = self.scaler.transform(X)
                platt_probs = self.platt_model.predict_proba(X_scaled)[:, 1]
                probs = self.isotonic_model.predict(platt_probs)
            return np.clip(probs, 0.0, 1.0).astype(np.float32)

        raise ValueError(f"Unknown calibration method: {self.method}")

    def save(self, path: Path | str) -> Path:
        """Serialize calibrator to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)
        log.info("Saved %s calibrator to %s", self.method, p)
        return p

    @classmethod
    def load(cls, path: Path | str) -> "Calibrator":
        """Load calibrator from disk."""
        p = Path(path)
        with open(p, "rb") as f:
            obj = pickle.load(f)
        log.info("Loaded %s calibrator from %s", obj.method, p)
        return obj


def fit_calibrator_on_split(
    cfg,
    split: str = "calib",
    method: Optional[str] = None,
) -> tuple[Calibrator, dict]:
    """
    Load calibration predictions, extract features, fit calibrator,
    and compute training calibration metrics.
    """
    assert split == "calib", (
        f"STRICT RULE VIOLATION: Calibrator must be fitted on 'calib' split only, got '{split}'!"
    )

    preds_path = Path(cfg.paths.results) / "logs" / f"b1_standard_rag_{split}_predictions.jsonl"
    if not preds_path.exists():
        raise FileNotFoundError(f"Calibration predictions not found at {preds_path}. Run experiments first.")

    from src.pipeline import load_predictions_jsonl
    records = load_predictions_jsonl(preds_path)

    extractor = FeatureExtractor.from_config(cfg)
    X, y, feature_names = extract_features_and_labels(records, extractor)

    calib_method = method or str(cfg.calibration.get("method", "platt"))
    calibrator = Calibrator(method=calib_method, random_state=int(cfg.project.seed))
    calibrator.fit(X, y)

    # Evaluate on calibration fit data
    probs = calibrator.predict_proba(X)
    eval_metrics = compute_calibration_metrics(y_true=y, y_prob=probs, num_bins=10)

    # Save fitted model
    out_dir = Path(cfg.paths.results) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"calibrator_{calib_method}.pkl"
    calibrator.save(model_path)

    summary = {
        "method": calib_method,
        "n_samples": len(y),
        "features": feature_names,
        "ece": eval_metrics["ece"],
        "mce": eval_metrics["mce"],
        "brier_score": eval_metrics["brier_score"],
        "model_path": str(model_path),
    }
    return calibrator, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LearnMate Calibrator")
    parser.add_argument("cmd", choices=["train"], help="Command")
    parser.add_argument("--config", default="configs/b2_calibrated_confidence.yaml")
    parser.add_argument("--method", default=None, choices=["platt", "isotonic"])
    parser.add_argument("--split", default="calib")

    args = parser.parse_args()
    seed_everything(42)

    cfg = load_config(args.config)
    calibrator, summary = fit_calibrator_on_split(
        cfg=cfg,
        split=args.split,
        method=args.method,
    )

    print("\n--- Calibration Training Summary ---")
    print(f"Method:      {summary['method'].upper()}")
    print(f"Samples:     {summary['n_samples']}")
    print(f"ECE:         {summary['ece']:.4f}")
    print(f"Brier Score: {summary['brier_score']:.4f}")
    print(f"Saved to:    {summary['model_path']}\n")


if __name__ == "__main__":
    main()
