"""
retrieve.py
-----------
Dense retriever using sentence-transformers + FAISS.

Index type: IndexFlatIP over L2-normalised vectors => cosine similarity.
Embeddings are cached to disk (.npy + .pkl) keyed by (model_name, split).

CLI:
  python -m src.retrieve build-index --config configs/default.yaml --split train
  python -m src.retrieve search  --config configs/default.yaml --split train --query "What is RAG?"
  python -m src.retrieve recall  --config configs/default.yaml --split tune  --k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import load_config, seed_everything, get_logger, Timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RetrievedPassage:
    passage_id: str
    text: str
    score: float          # cosine similarity in [-1, 1]
    rank: int             # 1-based
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    passages: List[RetrievedPassage]
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Return float32 array (N, dim), optionally L2-normalised."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    ).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)   # avoid div-by-zero
        embeddings = embeddings / norms
    return embeddings


def _embedding_cache_paths(cache_dir: Path, model_name: str, split: str):
    """Return (emb_path, ids_path) for the cache."""
    slug = model_name.replace("/", "_").replace("-", "_")
    emb_path = cache_dir / f"embeddings_{slug}_{split}.npy"
    ids_path = cache_dir / f"embeddings_{slug}_{split}_ids.pkl"
    return emb_path, ids_path


def load_or_compute_embeddings(
    passages: List[dict],
    model: SentenceTransformer,
    cache_dir: Path,
    split: str,
    model_name: str,
    batch_size: int = 64,
    force_recompute: bool = False,
) -> tuple[np.ndarray, List[str]]:
    """
    Load embeddings from disk cache or compute and save them.

    Returns
    -------
    embeddings : np.ndarray  shape (N, dim), L2-normalised float32
    passage_ids : List[str]
    """
    emb_path, ids_path = _embedding_cache_paths(cache_dir, model_name, split)
    if not force_recompute and emb_path.exists() and ids_path.exists():
        log.info("Loading cached embeddings from %s", emb_path)
        embeddings = np.load(str(emb_path))
        with open(ids_path, "rb") as f:
            passage_ids = pickle.load(f)
        log.info("Loaded %d embeddings (dim=%d)", len(passage_ids), embeddings.shape[1])
        return embeddings, passage_ids

    log.info("Computing embeddings for %d passages (split=%s)...", len(passages), split)
    texts = [p.get("text", p.get("context", "")) for p in passages]
    passage_ids = [p.get("passage_id", p.get("id", "")) for p in passages]

    embeddings = embed_texts(texts, model, batch_size=batch_size, normalize=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_path), embeddings)
    with open(ids_path, "wb") as f:
        pickle.dump(passage_ids, f)
    log.info("Saved embeddings to %s", emb_path)
    return embeddings, passage_ids


# ---------------------------------------------------------------------------
# FAISS index wrapper
# ---------------------------------------------------------------------------

class FaissIndex:
    """
    Thin wrapper around faiss.IndexFlatIP.

    Vectors must be L2-normalised before adding/querying so that
    inner-product == cosine similarity.
    """

    def __init__(self, dim: int):
        import faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._passage_ids: List[str] = []

    # ------------------------------------------------------------------
    def add(self, embeddings: np.ndarray, passage_ids: List[str]) -> None:
        """Add L2-normalised float32 embeddings and their IDs."""
        assert embeddings.ndim == 2 and embeddings.shape[1] == self.dim, (
            f"Expected shape (N, {self.dim}), got {embeddings.shape}"
        )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(embeddings)
        self._passage_ids.extend(passage_ids)

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        query_embeddings : (Q, dim) float32, L2-normalised
        k               : number of neighbours

        Returns
        -------
        scores  : (Q, k) float32   cosine similarities
        indices : (Q, k) int64     row indices into self._passage_ids
        """
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        actual_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_embeddings, actual_k)
        return scores, indices

    def get_passage_id(self, idx: int) -> str:
        return self._passage_ids[idx]

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    # ------------------------------------------------------------------
    def save(self, index_path: Path, ids_path: Path) -> None:
        import faiss
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        with open(ids_path, "wb") as f:
            pickle.dump(self._passage_ids, f)
        log.info("Saved FAISS index (%d vectors) to %s", self.ntotal, index_path)

    @classmethod
    def load(cls, index_path: Path, ids_path: Path) -> "FaissIndex":
        import faiss
        raw = faiss.read_index(str(index_path))
        dim = raw.d
        obj = cls.__new__(cls)
        obj.dim = dim
        obj._index = raw
        with open(ids_path, "rb") as f:
            obj._passage_ids = pickle.load(f)
        log.info("Loaded FAISS index (%d vectors, dim=%d) from %s", obj.ntotal, dim, index_path)
        return obj


# ---------------------------------------------------------------------------
# Dense Retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    End-to-end retriever: embedding model + FAISS index + passage store.

    Usage
    -----
    # Build once
    r = DenseRetriever.from_config(cfg)
    passages = [{"passage_id": ..., "text": ...}, ...]
    r.build_index(passages, split="train")
    r.save_index(split="train")

    # Load later
    r = DenseRetriever.from_config(cfg)
    r.load_index(split="train")
    results = r.retrieve("What is retrieval?", k=5)
    """

    def __init__(
        self,
        model_name: str,
        index_dir: Path,
        cache_dir: Path,
        batch_size: int = 64,
        default_k: int = 5,
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.default_k = default_k

        log.info("Loading embedding model: %s", model_name)
        try:
            self._model = SentenceTransformer(model_name)
        except Exception:
            log.info("Network unreachable; loading %s from local cache...", model_name)
            self._model = SentenceTransformer(model_name, local_files_only=True)
        self._faiss: Optional[FaissIndex] = None
        self._passage_map: dict[str, dict] = {}   # passage_id -> passage dict

    @classmethod
    def from_config(cls, cfg) -> "DenseRetriever":
        """Build from an OmegaConf config (configs/default.yaml schema)."""
        # cfg.retriever.model  → embedding model name
        # cfg.paths.cache      → base cache dir; we put embeddings in a subdir
        cache_dir = Path(cfg.paths.cache) / "embeddings"
        index_dir = Path(cfg.paths.cache) / "faiss"
        return cls(
            model_name=str(cfg.retriever.model),
            index_dir=index_dir,
            cache_dir=cache_dir,
            batch_size=int(cfg.retriever.get("batch_size", 64)),
            default_k=int(cfg.retriever.get("top_k", 5)),
        )

    # ------------------------------------------------------------------
    def _index_paths(self, split: str):
        slug = self.model_name.replace("/", "_").replace("-", "_")
        base = self.index_dir / f"faiss_{slug}_{split}"
        return base.with_suffix(".index"), base.with_suffix(".ids.pkl")

    def build_index(
        self,
        passages: List[dict],
        split: str,
        force_recompute: bool = False,
    ) -> None:
        """Embed passages and populate the FAISS index."""
        embeddings, passage_ids = load_or_compute_embeddings(
            passages=passages,
            model=self._model,
            cache_dir=self.cache_dir,
            split=split,
            model_name=self.model_name,
            batch_size=self.batch_size,
            force_recompute=force_recompute,
        )
        dim = embeddings.shape[1]
        self._faiss = FaissIndex(dim)
        self._faiss.add(embeddings, passage_ids)
        # Build passage map for text lookup
        self._passage_map = {
            p.get("passage_id", p.get("id", "")): {
                "text": p.get("text", p.get("context", "")),
                **p,
            }
            for p in passages
        }
        log.info("Built FAISS index with %d passages (dim=%d)", self._faiss.ntotal, dim)

    def save_index(self, split: str) -> None:
        assert self._faiss is not None, "No index built yet."
        idx_path, ids_path = self._index_paths(split)
        self._faiss.save(idx_path, ids_path)

    def load_index(self, split: str, passages: Optional[List[dict]] = None) -> None:
        idx_path, ids_path = self._index_paths(split)
        self._faiss = FaissIndex.load(idx_path, ids_path)
        if passages:
            self._passage_map = {
                p.get("passage_id", p.get("id", "")): {
                    "text": p.get("text", p.get("context", "")),
                    **p,
                }
                for p in passages
            }

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: Optional[int] = None) -> RetrievalResult:
        k = k or self.default_k
        assert self._faiss is not None, "Index not loaded. Call build_index() or load_index() first."
        t0 = time.perf_counter()
        q_emb = embed_texts([query], self._model, batch_size=1, show_progress=False, normalize=True)
        scores, indices = self._faiss.search(q_emb, k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        passages: List[RetrievedPassage] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            pid = self._faiss.get_passage_id(int(idx))
            pdata = self._passage_map.get(pid, {})
            passages.append(RetrievedPassage(
                passage_id=pid,
                text=pdata.get("text", ""),
                score=float(score),
                rank=rank,
                metadata={k: v for k, v in pdata.items() if k not in ("text", "passage_id")},
            ))
        return RetrievalResult(query=query, passages=passages, latency_ms=latency_ms)

    def retrieve_batch(
        self,
        queries: List[str],
        k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        k = k or self.default_k
        assert self._faiss is not None, "Index not loaded."
        q_embs = embed_texts(queries, self._model, batch_size=self.batch_size,
                              show_progress=False, normalize=True)
        scores_all, indices_all = self._faiss.search(q_embs, k)

        results = []
        for qi, (query, scores, indices) in enumerate(zip(queries, scores_all, indices_all)):
            passages: List[RetrievedPassage] = []
            for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
                if idx < 0:
                    continue
                pid = self._faiss.get_passage_id(int(idx))
                pdata = self._passage_map.get(pid, {})
                passages.append(RetrievedPassage(
                    passage_id=pid,
                    text=pdata.get("text", ""),
                    score=float(score),
                    rank=rank,
                    metadata={k2: v for k2, v in pdata.items() if k2 not in ("text", "passage_id")},
                ))
            results.append(RetrievalResult(query=query, passages=passages))
        return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval_recall(
    retriever: DenseRetriever,
    examples: List[dict],
    k: int = 5,
) -> dict:
    """
    Compute Recall@k on answerable examples.

    Parameters
    ----------
    examples : list of dicts with keys: question, passage_id, is_answerable
               Only answerable examples (is_answerable=True) are evaluated.

    Returns
    -------
    dict with keys: recall_at_k, k, n_evaluated, n_retrieved_correct
    """
    answerable = [ex for ex in examples if ex.get("is_answerable", True)]
    if not answerable:
        return {"recall_at_k": 0.0, "k": k, "n_evaluated": 0, "n_retrieved_correct": 0}

    questions = [ex["question"] for ex in answerable]
    gold_pids = [ex["passage_id"] for ex in answerable]

    results = retriever.retrieve_batch(questions, k=k)

    n_correct = 0
    for result, gold_pid in zip(results, gold_pids):
        retrieved_pids = {p.passage_id for p in result.passages}
        if gold_pid in retrieved_pids:
            n_correct += 1

    return {
        "recall_at_k": n_correct / len(answerable),
        "k": k,
        "n_evaluated": len(answerable),
        "n_retrieved_correct": n_correct,
    }


def _load_passages_from_jsonl(path: Path) -> List[dict]:
    passages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line.strip()))
    return passages


def _load_examples_from_jsonl(path: Path) -> List[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_build_index(args):
    seed_everything(42)
    cfg = load_config(args.config)
    split = args.split
    passages_path = Path(cfg.paths.data_processed) / f"squad_{split}_passages.jsonl"
    passages = _load_passages_from_jsonl(passages_path)
    log.info("Loaded %d passages for split=%s", len(passages), split)

    retriever = DenseRetriever.from_config(cfg)
    retriever.build_index(passages, split=split, force_recompute=args.force)
    retriever.save_index(split=split)
    log.info("Index built and saved.")


def _cmd_search(args):
    seed_everything(42)
    cfg = load_config(args.config)
    split = args.split
    passages_path = Path(cfg.paths.data_processed) / f"squad_{split}_passages.jsonl"
    passages = _load_passages_from_jsonl(passages_path)

    retriever = DenseRetriever.from_config(cfg)
    retriever.load_index(split=split, passages=passages)

    result = retriever.retrieve(args.query, k=args.k)
    print(f"\nQuery: {result.query!r}  (latency={result.latency_ms:.1f} ms)\n")
    for p in result.passages:
        print(f"  Rank {p.rank}  score={p.score:.4f}  id={p.passage_id}")
        print(f"    {p.text[:200]}...\n")


def _cmd_recall(args):
    seed_everything(42)
    cfg = load_config(args.config)
    split = args.split
    passages_path = Path(cfg.paths.data_processed) / f"squad_{split}_passages.jsonl"
    examples_path = Path(cfg.paths.data_processed) / f"squad_{split}.jsonl"
    passages = _load_passages_from_jsonl(passages_path)
    examples = _load_examples_from_jsonl(examples_path)

    retriever = DenseRetriever.from_config(cfg)
    retriever.load_index(split=split, passages=passages)

    recall = evaluate_retrieval_recall(retriever, examples, k=args.k)
    print(f"\nRetrieval Recall@{args.k}: {recall['recall_at_k']:.4f}")
    print(f"  ({recall['n_retrieved_correct']}/{recall['n_evaluated']} answerable questions)")


def main():
    parser = argparse.ArgumentParser(description="LearnMate Dense Retriever")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build-index")
    p_build.add_argument("--config", default="configs/default.yaml")
    p_build.add_argument("--split", default="train")
    p_build.add_argument("--force", action="store_true")

    p_search = sub.add_parser("search")
    p_search.add_argument("--config", default="configs/default.yaml")
    p_search.add_argument("--split", default="train")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--k", type=int, default=5)

    p_recall = sub.add_parser("recall")
    p_recall.add_argument("--config", default="configs/default.yaml")
    p_recall.add_argument("--split", default="tune")
    p_recall.add_argument("--k", type=int, default=5)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return
    {"build-index": _cmd_build_index, "search": _cmd_search, "recall": _cmd_recall}[args.cmd](args)


if __name__ == "__main__":
    main()
