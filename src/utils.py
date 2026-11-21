"""
src/utils.py
────────────
Global utilities: seeding, logging setup, reproducibility metadata capture,
device detection, and config loading.

All seeding is centralised here.  Call `seed_everything(cfg.project.seed)`
as the very first line of every script / experiment entry-point.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure HuggingFace Hub runs offline without network retry lag in sandboxed / offline envs
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a richly-formatted logger for the given module name."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds deterministically.
    Must be called before importing transformers / sentence-transformers
    or any library that caches RNG state on import.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic CUDA ops (may slow training; essential for reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device(cfg_device: str = "auto") -> torch.device:
    """
    Resolve the compute device.

    Args:
        cfg_device: "auto" | "cuda" | "cpu"

    Returns:
        torch.device
    """
    if cfg_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg_device)
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """
    Load a YAML config, merge with default.yaml if a _base_ key is present,
    then apply any CLI overrides (OmegaConf dot-path format).

    Args:
        config_path: Path to the experiment YAML file.
        overrides:   List of OmegaConf override strings, e.g. ["retriever.top_k=10"].

    Returns:
        Merged OmegaConf DictConfig.
    """
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    base_cfg: DictConfig = OmegaConf.create({})
    if "_base_" in raw:
        base_path = config_path.parent / raw.pop("_base_")
        base_cfg = OmegaConf.create(
            yaml.safe_load(base_path.read_text(encoding="utf-8"))
        )

    exp_cfg = OmegaConf.create(raw)
    merged = OmegaConf.merge(base_cfg, exp_cfg)

    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility metadata
# ─────────────────────────────────────────────────────────────────────────────

def _git_commit_hash() -> str:
    """Return the current git HEAD commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _gpu_info() -> dict[str, Any]:
    """Return basic GPU info dict."""
    info: dict[str, Any] = {"available": torch.cuda.is_available()}
    if info["available"]:
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(props.total_memory / 1024**3, 2)
    return info


def _package_versions(packages: list[str]) -> dict[str, str]:
    """Return {package: version} for the given package names."""
    from importlib.metadata import PackageNotFoundError, version

    result: dict[str, str] = {}
    for pkg in packages:
        try:
            result[pkg] = version(pkg)
        except PackageNotFoundError:
            result[pkg] = "not installed"
    return result


TRACKED_PACKAGES = [
    "torch", "transformers", "sentence-transformers", "faiss-gpu",
    "faiss-cpu", "datasets", "scikit-learn", "numpy", "pandas",
    "ollama", "streamlit", "omegaconf",
]


def build_run_metadata(
    cfg: DictConfig,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a reproducibility metadata dict to be saved alongside every run's
    raw predictions.  Fields:
      - timestamp, git_commit, python_version, platform
      - package_versions
      - gpu_info
      - config (full merged config as dict)
      - any caller-supplied extras
    """
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(TRACKED_PACKAGES),
        "gpu_info": _gpu_info(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if extra:
        meta.update(extra)
    return meta


def save_run_metadata(
    meta: dict[str, Any],
    out_dir: str | Path,
    filename: str = "run_metadata.json",
) -> Path:
    """Write reproducibility metadata to JSON in out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Simple context-manager timer.

    Usage::

        with Timer() as t:
            do_work()
        print(f"Elapsed: {t.elapsed:.3f}s")
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed: float = time.perf_counter() - self._start

    def __repr__(self) -> str:
        return f"Timer(elapsed={self.elapsed:.3f}s)"


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Return the project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent
