"""
src/data_prep.py
────────────────
Data preparation module for LearnMate.
Handles downloading official SQuAD 2.0 files, article-level splitting,
leakage verification, and saving processed JSONL datasets and passage pools.

Critical Rules:
- TRAIN is split strictly by ARTICLE TITLE into train (80%), calib (10%), tune (10%).
- DEV is preserved intact for final reporting only and NEVER mixed with training/tuning.
- Seed is locked to 42 for deterministic replication.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig

from src.utils import ensure_dir, get_logger, project_root, seed_everything

logger = get_logger(__name__)

SQUAD_V2_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
SQUAD_V2_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


@dataclass(frozen=True)
class SquadExample:
    """Represents a single SQuAD 2.0 question-answer item."""
    qid: str
    question: str
    is_impossible: bool
    answers: list[str]
    plausible_answers: list[str]
    passage_id: str
    title: str
    context: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SquadPassage:
    """Represents a unique context passage/paragraph for retrieval pools."""
    passage_id: str
    title: str
    context: str
    char_length: int
    word_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_passage_id(title: str, context: str) -> str:
    """Generate a deterministic 16-character hex ID based on article title and context content."""
    hasher = hashlib.sha256()
    hasher.update(title.strip().encode("utf-8"))
    hasher.update(b":::")
    hasher.update(context.strip().encode("utf-8"))
    return hasher.hexdigest()[:16]


def download_file(url: str, dest_path: Path, force: bool = False) -> Path:
    """Download a file from url to dest_path with progress logging if not already present."""
    if dest_path.exists() and not force:
        logger.info(f"File already exists: {dest_path} ({dest_path.stat().st_size:,} bytes)")
        return dest_path

    ensure_dir(dest_path.parent)
    logger.info(f"Downloading {url} -> {dest_path}...")
    
    # Simple chunked download with progress
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LearnMate-Research-Prototype/1.0"}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", -1))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0 and downloaded % (5 * 1024 * 1024) < chunk_size:
                logger.debug(f"Downloaded {downloaded / 1024 / 1024:.1f} / {total_size / 1024 / 1024:.1f} MB")

    logger.info(f"Successfully downloaded {dest_path.name} ({dest_path.stat().st_size:,} bytes)")
    return dest_path


def download_squad_v2(raw_dir: str | Path, force: bool = False) -> tuple[Path, Path]:
    """Download official SQuAD 2.0 train and dev files into raw_dir."""
    raw_path = Path(raw_dir)
    ensure_dir(raw_path)
    train_dest = raw_path / "train-v2.0.json"
    dev_dest = raw_path / "dev-v2.0.json"

    download_file(SQUAD_V2_TRAIN_URL, train_dest, force=force)
    download_file(SQUAD_V2_DEV_URL, dev_dest, force=force)

    return train_dest, dev_dest


def parse_squad_json(json_path: str | Path) -> tuple[list[SquadExample], list[SquadPassage], list[str]]:
    """
    Parse official SQuAD 2.0 JSON file.

    Returns:
        examples: List of SquadExample dataclasses
        passages: List of unique SquadPassage dataclasses (deduplicated by passage_id)
        articles: Ordered list of unique article titles encountered
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"SQuAD json file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: list[SquadExample] = []
    passages_dict: dict[str, SquadPassage] = {}
    articles: list[str] = []

    for article in data.get("data", []):
        title = article.get("title", "").strip()
        if title not in articles:
            articles.append(title)

        for para in article.get("paragraphs", []):
            context = para.get("context", "").strip()
            pid = make_passage_id(title, context)

            if pid not in passages_dict:
                words = re.findall(r"\w+", context)
                passages_dict[pid] = SquadPassage(
                    passage_id=pid,
                    title=title,
                    context=context,
                    char_length=len(context),
                    word_length=len(words),
                )

            for qa in para.get("qas", []):
                qid = qa.get("id", "")
                question = qa.get("question", "").strip()
                is_impossible = bool(qa.get("is_impossible", False))
                answers = [ans["text"].strip() for ans in qa.get("answers", []) if "text" in ans]
                plausible_answers = [
                    ans["text"].strip() for ans in qa.get("plausible_answers", []) if "text" in ans
                ]

                example = SquadExample(
                    qid=qid,
                    question=question,
                    is_impossible=is_impossible,
                    answers=answers,
                    plausible_answers=plausible_answers,
                    passage_id=pid,
                    title=title,
                    context=context,
                )
                examples.append(example)

    return examples, list(passages_dict.values()), articles


def split_articles(
    articles: list[str],
    train_ratio: float = 0.80,
    calib_ratio: float = 0.10,
    tune_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Split article titles into train, calib, and tune splits.
    
    Guarantees:
    - Splits are strictly disjoint by article title.
    - Deterministic across runs for the same seed.
    - Preserves all unique articles without omission or duplication.
    """
    if abs((train_ratio + calib_ratio + tune_ratio) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + calib_ratio + tune_ratio}")

    # Sort deterministically first, then shuffle with seeded RNG
    sorted_articles = sorted(set(articles))
    rng = random.Random(seed)
    shuffled = list(sorted_articles)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_calib = int(round(n * calib_ratio))
    # Remainder goes to tune to ensure exact conservation
    n_tune = n - n_train - n_calib

    train_arts = sorted(shuffled[:n_train])
    calib_arts = sorted(shuffled[n_train:n_train + n_calib])
    tune_arts = sorted(shuffled[n_train + n_calib:])

    # Strict assertion
    train_set, calib_set, tune_set = set(train_arts), set(calib_arts), set(tune_arts)
    assert len(train_set & calib_set) == 0, "Train and Calib article overlap detected!"
    assert len(train_set & tune_set) == 0, "Train and Tune article overlap detected!"
    assert len(calib_set & tune_set) == 0, "Calib and Tune article overlap detected!"
    assert len(train_set | calib_set | tune_set) == n, "Article count mismatch after split!"

    return {
        "train": train_arts,
        "calib": calib_arts,
        "tune": tune_arts,
    }


def save_article_lists(split_dict: dict[str, list[str]], splits_dir: str | Path) -> dict[str, Path]:
    """Write article titles to text files (one title per line)."""
    p = Path(splits_dir)
    ensure_dir(p)
    paths: dict[str, Path] = {}
    for split_name, articles in split_dict.items():
        file_path = p / f"squad_{split_name}_articles.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for art in sorted(articles):
                f.write(f"{art}\n")
        paths[split_name] = file_path
        logger.info(f"Saved {len(articles)} article titles to {file_path}")
    return paths


def load_article_list(file_path: str | Path) -> list[str]:
    """Load article titles from a text file."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Article list file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_jsonl(records: Iterable[dict[str, Any]], out_path: str | Path) -> Path:
    """Save an iterable of dictionaries as JSONL."""
    p = Path(out_path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def load_jsonl(in_path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    p = Path(in_path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    records: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def prepare_squad_splits(
    cfg: DictConfig,
    force_download: bool = False,
) -> dict[str, Any]:
    """
    Main orchestration routine for Phase 1 SQuAD preparation:
    1. Downloads train-v2.0.json and dev-v2.0.json.
    2. Parses and splits TRAIN articles into train (80%), calib (10%), tune (10%).
    3. Saves article lists to data/splits/.
    4. Filters examples and passages by article partition and saves them to data/processed/.
    5. Saves full DEV examples and passage pool to data/processed/squad_dev.jsonl.
    """
    root = project_root()
    seed_everything(cfg.project.seed)

    raw_dir = root / cfg.paths.data_raw
    splits_dir = root / cfg.paths.data_splits
    processed_dir = root / cfg.paths.data_processed

    # 1. Download
    train_path, dev_path = download_squad_v2(raw_dir, force=force_download)

    # 2. Parse raw files
    logger.info("Parsing official SQuAD train-v2.0.json...")
    train_examples, train_passages, train_articles = parse_squad_json(train_path)
    logger.info(f"Parsed {len(train_examples):,} train examples across {len(train_articles)} articles.")

    logger.info("Parsing official SQuAD dev-v2.0.json...")
    dev_examples, dev_passages, dev_articles = parse_squad_json(dev_path)
    logger.info(f"Parsed {len(dev_examples):,} dev examples across {len(dev_articles)} articles.")

    # 3. Create or load article splits
    split_dict = split_articles(
        train_articles,
        train_ratio=cfg.squad.train_ratio,
        calib_ratio=cfg.squad.calib_ratio,
        tune_ratio=cfg.squad.tune_ratio,
        seed=cfg.squad.split_seed,
    )
    article_paths = save_article_lists(split_dict, splits_dir)

    # 4. Partition examples and passages
    article_to_split: dict[str, str] = {}
    for split_name, arts in split_dict.items():
        for art in arts:
            article_to_split[art] = split_name

    split_examples: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "calib": [],
        "tune": [],
    }
    split_passages: dict[str, dict[str, dict[str, Any]]] = {
        "train": {},
        "calib": {},
        "tune": {},
    }

    for ex in train_examples:
        s_name = article_to_split.get(ex.title)
        if s_name:
            split_examples[s_name].append(ex.to_dict())

    for ps in train_passages:
        s_name = article_to_split.get(ps.title)
        if s_name:
            split_passages[s_name][ps.passage_id] = ps.to_dict()

    # Save processed files
    output_files: dict[str, Path] = {}
    for s_name in ["train", "calib", "tune"]:
        ex_file = processed_dir / f"squad_{s_name}.jsonl"
        ps_file = processed_dir / f"squad_{s_name}_passages.jsonl"
        save_jsonl(split_examples[s_name], ex_file)
        save_jsonl(list(split_passages[s_name].values()), ps_file)
        output_files[f"{s_name}_examples"] = ex_file
        output_files[f"{s_name}_passages"] = ps_file
        logger.info(
            f"Split '{s_name}': {len(split_examples[s_name]):,} examples, "
            f"{len(split_passages[s_name]):,} passages -> {ex_file.name}"
        )

    # 5. Process DEV set (held out strictly for final reporting)
    dev_ex_file = processed_dir / "squad_dev.jsonl"
    dev_ps_file = processed_dir / "squad_dev_passages.jsonl"
    save_jsonl([e.to_dict() for e in dev_examples], dev_ex_file)
    save_jsonl([p.to_dict() for p in dev_passages], dev_ps_file)
    output_files["dev_examples"] = dev_ex_file
    output_files["dev_passages"] = dev_ps_file
    logger.info(
        f"Dev set: {len(dev_examples):,} examples, {len(dev_passages):,} passages -> {dev_ex_file.name}"
    )

    summary = {
        "articles": {k: len(v) for k, v in split_dict.items()},
        "dev_articles": len(dev_articles),
        "examples": {k: len(v) for k, v in split_examples.items()},
        "dev_examples": len(dev_examples),
        "passages": {k: len(v) for k, v in split_passages.items()},
        "dev_passages": len(dev_passages),
        "output_files": {k: str(v) for k, v in output_files.items()},
        "article_files": {k: str(v) for k, v in article_paths.items()},
    }
    return summary
