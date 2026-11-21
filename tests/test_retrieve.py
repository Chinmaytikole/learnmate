"""
tests/test_retrieve.py
----------------------
Unit tests for src/retrieve.py.

Tests use small synthetic data only — no model downloads, no FAISS GPU,
no disk I/O beyond a tmp dir.  The SentenceTransformer model is mocked.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieve import (
    FaissIndex,
    RetrievedPassage,
    RetrievalResult,
    DenseRetriever,
    embed_texts,
    load_or_compute_embeddings,
    evaluate_retrieval_recall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Return n random L2-normalised float32 vectors of dimension dim."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_passages(n: int) -> list[dict]:
    return [{"passage_id": f"p{i}", "text": f"Passage text number {i}."} for i in range(n)]


# ---------------------------------------------------------------------------
# FaissIndex tests
# ---------------------------------------------------------------------------

class TestFaissIndex:

    def test_add_and_ntotal(self):
        idx = FaissIndex(dim=8)
        vecs = _unit_vectors(10, 8)
        ids = [f"p{i}" for i in range(10)]
        idx.add(vecs, ids)
        assert idx.ntotal == 10

    def test_search_returns_shape(self):
        dim = 16
        idx = FaissIndex(dim=dim)
        vecs = _unit_vectors(20, dim)
        ids = [f"p{i}" for i in range(20)]
        idx.add(vecs, ids)

        query = _unit_vectors(1, dim)
        scores, indices = idx.search(query, k=5)
        assert scores.shape == (1, 5)
        assert indices.shape == (1, 5)

    def test_search_top1_is_most_similar(self):
        """When we query with a copy of passage 3, it should be rank 1."""
        dim = 32
        idx = FaissIndex(dim=dim)
        vecs = _unit_vectors(10, dim, seed=7)
        ids = [f"p{i}" for i in range(10)]
        idx.add(vecs, ids)

        query = vecs[3:4].copy()   # exact copy => cosine sim = 1.0
        scores, indices = idx.search(query, k=3)
        assert indices[0, 0] == 3, "Top result should be passage 3 (identical vector)"
        assert scores[0, 0] > 0.999, "Cosine similarity to self should be ~1"

    def test_search_scores_descending(self):
        dim = 16
        idx = FaissIndex(dim=dim)
        vecs = _unit_vectors(50, dim)
        ids = [f"p{i}" for i in range(50)]
        idx.add(vecs, ids)

        query = _unit_vectors(1, dim, seed=99)
        scores, _ = idx.search(query, k=10)
        s = scores[0]
        assert all(s[i] >= s[i + 1] for i in range(len(s) - 1)), \
            "Scores must be non-increasing (highest similarity first)"

    def test_get_passage_id(self):
        idx = FaissIndex(dim=4)
        vecs = _unit_vectors(3, 4)
        ids = ["alpha", "beta", "gamma"]
        idx.add(vecs, ids)
        assert idx.get_passage_id(0) == "alpha"
        assert idx.get_passage_id(2) == "gamma"

    def test_save_load_roundtrip(self, tmp_path):
        dim = 8
        idx = FaissIndex(dim=dim)
        vecs = _unit_vectors(5, dim)
        ids = [f"p{i}" for i in range(5)]
        idx.add(vecs, ids)

        idx_file = tmp_path / "test.index"
        ids_file = tmp_path / "test.ids.pkl"
        idx.save(idx_file, ids_file)

        idx2 = FaissIndex.load(idx_file, ids_file)
        assert idx2.ntotal == 5
        assert idx2._passage_ids == ids

        # Search results must match
        query = _unit_vectors(1, dim, seed=3)
        s1, i1 = idx.search(query, k=3)
        s2, i2 = idx2.search(query, k=3)
        np.testing.assert_array_equal(i1, i2)
        np.testing.assert_allclose(s1, s2, atol=1e-5)

    def test_add_wrong_dim_raises(self):
        idx = FaissIndex(dim=8)
        bad_vecs = _unit_vectors(3, 16)   # wrong dim
        with pytest.raises(AssertionError):
            idx.add(bad_vecs, ["a", "b", "c"])

    def test_k_clamped_to_ntotal(self):
        """Requesting k > ntotal should not crash; returns ntotal results."""
        idx = FaissIndex(dim=8)
        vecs = _unit_vectors(3, 8)
        idx.add(vecs, ["a", "b", "c"])
        query = _unit_vectors(1, 8)
        scores, indices = idx.search(query, k=100)   # only 3 available
        assert scores.shape[1] == 3


# ---------------------------------------------------------------------------
# embed_texts tests (mocked SentenceTransformer)
# ---------------------------------------------------------------------------

class TestEmbedTexts:

    def test_output_shape_and_type(self):
        dim = 16
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((3, dim), dtype=np.float32)

        result = embed_texts(["a", "b", "c"], mock_model, normalize=False)
        assert result.shape == (3, dim)
        assert result.dtype == np.float32

    def test_normalization(self):
        dim = 8
        mock_model = MagicMock()
        # Return unnormalized vectors (norm = sqrt(dim))
        mock_model.encode.return_value = np.ones((4, dim), dtype=np.float32)

        result = embed_texts(["a", "b", "c", "d"], mock_model, normalize=True)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)

    def test_zero_vector_no_div_by_zero(self):
        dim = 4
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((2, dim), dtype=np.float32)
        # Should not raise
        result = embed_texts(["a", "b"], mock_model, normalize=True)
        assert result.shape == (2, dim)


# ---------------------------------------------------------------------------
# load_or_compute_embeddings tests
# ---------------------------------------------------------------------------

class TestLoadOrComputeEmbeddings:

    def _make_model(self, dim: int = 8) -> MagicMock:
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda texts, **kw: np.ones(
            (len(texts), dim), dtype=np.float32
        )
        return mock_model

    def test_computes_and_caches(self, tmp_path):
        passages = _make_passages(5)
        model = self._make_model()
        embs, ids = load_or_compute_embeddings(
            passages, model, cache_dir=tmp_path,
            split="test", model_name="mock_model"
        )
        assert embs.shape == (5, 8)
        assert len(ids) == 5
        # Cache files must exist
        assert any(tmp_path.glob("*.npy"))
        assert any(tmp_path.glob("*.pkl"))
        assert model.encode.call_count == 1

    def test_loads_from_cache_on_second_call(self, tmp_path):
        passages = _make_passages(5)
        model = self._make_model()
        # First call computes
        load_or_compute_embeddings(passages, model, tmp_path, "test", "mock_model")
        call_count_after_first = model.encode.call_count

        # Second call must load from cache (no new encode call)
        embs2, ids2 = load_or_compute_embeddings(passages, model, tmp_path, "test", "mock_model")
        assert model.encode.call_count == call_count_after_first  # no extra encode
        assert embs2.shape == (5, 8)

    def test_force_recompute_ignores_cache(self, tmp_path):
        passages = _make_passages(3)
        model = self._make_model()
        load_or_compute_embeddings(passages, model, tmp_path, "test", "mock_model")
        load_or_compute_embeddings(passages, model, tmp_path, "test", "mock_model",
                                   force_recompute=True)
        assert model.encode.call_count == 2


# ---------------------------------------------------------------------------
# evaluate_retrieval_recall tests
# ---------------------------------------------------------------------------

class TestEvaluateRetrievalRecall:
    """
    Toy example:
      - 3 passages: p0, p1, p2
      - 2 answerable questions:
          Q1 gold=p0  -> retriever returns [p0, p1]  => HIT
          Q2 gold=p2  -> retriever returns [p0, p1]  => MISS
      - Expected recall@2 = 1/2 = 0.5
    """

    def _make_retriever_with_mock(self, dim: int = 8) -> DenseRetriever:
        """Build a DenseRetriever with a real FAISS index on toy data."""
        passages = _make_passages(3)   # p0, p1, p2

        # Create distinguishable vectors
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((3, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms

        with patch("src.retrieve.SentenceTransformer") as MockST:
            instance = MockST.return_value
            # Encode: for queries "q0" -> vecs[0]+noise (hits p0)
            #                     "q2" -> opposite of vecs[2] (misses p2)
            def mock_encode(texts, **kw):
                out = []
                for t in texts:
                    if t == "q0":
                        # Very similar to p0
                        v = vecs[0] * 0.99 + rng.standard_normal(dim).astype(np.float32) * 0.01
                    else:
                        # Opposite of p2 (should NOT hit p2)
                        v = -vecs[2] + rng.standard_normal(dim).astype(np.float32) * 0.01
                    nrm = np.linalg.norm(v)
                    out.append(v / (nrm if nrm > 0 else 1.0))
                return np.array(out, dtype=np.float32)

            instance.encode.side_effect = mock_encode

            ret = DenseRetriever(
                model_name="mock",
                index_dir=Path("/tmp/idx"),
                cache_dir=Path("/tmp/cache"),
                default_k=2,
            )
            ret._model = instance

        # Manually inject the index
        fidx = FaissIndex(dim=dim)
        fidx.add(vecs, ["p0", "p1", "p2"])
        ret._faiss = fidx
        ret._passage_map = {p["passage_id"]: p for p in passages}
        return ret

    def test_recall_on_toy_example(self):
        """
        Q1 (gold=p0, query="q0"): retriever should find p0 -> HIT
        Q2 (gold=p2, query="q_neg"): retriever should NOT find p2 -> MISS
        Recall@2 = 1/2 = 0.5
        """
        ret = self._make_retriever_with_mock(dim=32)

        # Re-make with larger dim for reliability
        dim = 32
        passages = _make_passages(3)
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((3, dim)).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        fidx = FaissIndex(dim=dim)
        fidx.add(vecs, ["p0", "p1", "p2"])

        def mock_encode(texts, **kw):
            out = []
            for t in texts:
                if "q0" in t:
                    v = vecs[0].copy()
                else:
                    v = -vecs[2].copy()
                nrm = np.linalg.norm(v)
                out.append(v / (nrm if nrm > 0 else 1.0))
            return np.array(out, dtype=np.float32)

        ret._faiss = fidx
        ret._passage_map = {p["passage_id"]: p for p in passages}
        ret._model = MagicMock()
        ret._model.encode.side_effect = mock_encode

        examples = [
            {"question": "q0 question", "passage_id": "p0", "is_answerable": True},
            {"question": "q_neg question", "passage_id": "p2", "is_answerable": True},
        ]
        result = evaluate_retrieval_recall(ret, examples, k=2)
        # p0 should be hit, p2 should be missed -> recall = 0.5
        assert result["n_evaluated"] == 2
        assert result["k"] == 2
        assert result["n_retrieved_correct"] == 1
        assert abs(result["recall_at_k"] - 0.5) < 1e-6

    def test_only_answerable_examples_counted(self):
        """Unanswerable examples must be excluded."""
        dim = 16
        passages = _make_passages(3)
        rng = np.random.default_rng(1)
        vecs = rng.standard_normal((3, dim)).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        fidx = FaissIndex(dim=dim)
        fidx.add(vecs, ["p0", "p1", "p2"])

        ret = DenseRetriever.__new__(DenseRetriever)
        ret.default_k = 2
        ret.batch_size = 64
        ret._faiss = fidx
        ret._passage_map = {p["passage_id"]: p for p in passages}

        # Mock encode to always return vecs[0]
        mock_model = MagicMock()
        mock_model.encode.return_value = vecs[0:1]
        ret._model = mock_model

        examples = [
            {"question": "q0", "passage_id": "p0", "is_answerable": True},
            {"question": "q1", "passage_id": "p1", "is_answerable": False},  # excluded
        ]
        result = evaluate_retrieval_recall(ret, examples, k=2)
        assert result["n_evaluated"] == 1   # only the answerable one

    def test_empty_answerable_returns_zero(self):
        """If no answerable examples, recall@k = 0.0, no crash."""
        dim = 8
        fidx = FaissIndex(dim=dim)
        fidx.add(_unit_vectors(3, dim), ["p0", "p1", "p2"])

        ret = DenseRetriever.__new__(DenseRetriever)
        ret.default_k = 5
        ret.batch_size = 64
        ret._faiss = fidx
        ret._passage_map = {}
        ret._model = MagicMock()

        examples = [{"question": "q", "passage_id": "p0", "is_answerable": False}]
        result = evaluate_retrieval_recall(ret, examples, k=5)
        assert result["recall_at_k"] == 0.0
        assert result["n_evaluated"] == 0

