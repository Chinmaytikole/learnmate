"""
scripts/setup_course_demo.py
----------------------------
Populate and process demo course materials for Phase 8.

Creates a structured 3-chapter machine learning course text:
  - Chapter 1: Optimization and Loss Functions (In-Distribution Tune)
  - Chapter 2: Convolutional Neural Networks (In-Distribution Tune)
  - Chapter 3: Transformers and Attention Mechanisms (Held-Out Test Chapter)

Runs ingestion (src/ingest.py), candidate question generation,
and consensus evaluation set preparation (data/course/eval.csv).

Usage:
  python scripts/setup_course_demo.py [--mock]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.compute_kappa import compute
from scripts.export_annotation_sheet import export_sheets
from scripts.gen_course_questions import QuestionGenerator, generate_candidates
from src.ingest import ingest_directory, save_chunks
from src.utils import ensure_dir, get_logger, load_config, seed_everything

log = get_logger(__name__)

CHAPTER_1_TEXT = """# Chapter 1: Optimization and Loss Functions in Machine Learning

Gradient descent is a foundational first-order iterative optimization algorithm used to minimize objective functions in machine learning and deep learning models. In gradient descent, parameter weights are updated proportionally to the negative gradient of the loss function with respect to the parameters. The learning rate hyperparameter determines the size of the update step taken in the opposite direction of the gradient at each training iteration. If the learning rate is chosen too large, gradient descent can overshoot the local or global minimum and diverge catastrophically. Conversely, if the learning rate is set too small, training requires an impractically large number of iterations to converge to a stationary point.

Stochastic gradient descent (SGD) addresses the computational cost of full-batch gradient descent by approximating the true gradient using only a single training example or a small mini-batch of examples at each step. Mini-batch stochastic gradient descent reduces the variance of parameter updates relative to pure SGD, leading to more stable convergence trajectories while maintaining computational efficiency on modern GPU accelerators. Momentum techniques accelerate SGD in directions of consistent gradient descent while dampening high-frequency oscillations across steep ravines. The Adam optimizer combines the advantages of AdaGrad and RMSProp by maintaining exponentially decaying moving averages of past gradients and past squared gradients, calculating individual adaptive learning rates for each parameter.
"""

CHAPTER_2_TEXT = """# Chapter 2: Convolutional Neural Networks and Computer Vision

Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed primarily for processing structured grid data such as images and spectrograms. The core computational layer of a CNN is the convolutional layer, which performs discrete spatial cross-correlation operations between learnable kernel filters and input feature maps. Parameter sharing across spatial dimensions allows convolutional kernels to detect local translation-invariant visual features such as edges, textures, corners, and motifs regardless of their absolute position in the receptive field. By stacking multiple convolutional layers hierarchically, lower layers capture low-level primitive edges while deeper layers compose high-level semantic object representations.

Pooling layers, such as max pooling and average pooling, are periodically inserted between successive convolutional layers to progressively reduce spatial dimensions of representation volumes. Max pooling extracts the maximum activation within a rectangular sliding window, providing translational invariance and reducing the total number of trainable network parameters and computational floating-point operations. Fully connected dense layers typically conclude the CNN architecture, mapping final flattened or global average-pooled feature vectors into class probability distributions via the softmax activation function. Batch normalization stabilizes intermediate feature distributions by normalizing layer inputs to zero mean and unit variance across mini-batches.
"""

CHAPTER_3_TEXT = """# Chapter 3: Transformers and Scaled Dot-Product Attention (Held-Out Chapter)

The Transformer architecture completely replaces recurrence and convolutions with multi-head self-attention mechanisms to model long-range dependencies in sequence data. The fundamental building block of Transformer attention is the scaled dot-product attention function. Given query matrix Q, key matrix K, and value matrix V, the attention weights are computed by taking the matrix product of queries and keys, dividing by the square root of the key dimension d_k to prevent vanishing gradients in softmax regions, and multiplying by the value matrix V. Because the Transformer contains no recurrent connections, positional encodings are injected into input embeddings to retain sequential token order information.

Multi-head attention extends scaled dot-product attention by linearly projecting queries, keys, and values h times with different learnable projection matrices. This allows the model to jointly attend to information from different representation subspaces at different positions simultaneously. Each layer of the Transformer encoder consists of a multi-head self-attention sublayer followed by a position-wise fully connected feed-forward network, augmented by residual skip connections and layer normalization operations. In the decoder, masked multi-head self-attention prevents positions from attending to subsequent future tokens, preserving autoregressive causality during generative sequence decoding.
"""


def setup_course_demo(
    course_dir: Path,
    mock: bool = True,
    seed: int = 42,
) -> Path:
    """Set up course demo raw files, chunking, question generation, and consensus eval CSV."""
    seed_everything(seed)
    raw_dir = ensure_dir(course_dir / "raw")
    processed_dir = ensure_dir(course_dir / "processed")
    ann_dir = ensure_dir(course_dir / "annotation")

    # 1. Write course materials
    (raw_dir / "chapter_01_optimization.md").write_text(CHAPTER_1_TEXT.strip(), encoding="utf-8")
    (raw_dir / "chapter_02_cnn.md").write_text(CHAPTER_2_TEXT.strip(), encoding="utf-8")
    (raw_dir / "chapter_03_transformers.md").write_text(CHAPTER_3_TEXT.strip(), encoding="utf-8")
    log.info("Wrote 3 course chapters to %s", raw_dir)

    # 2. Ingest & Chunk
    chunks = ingest_directory(raw_dir, min_words=100, max_words=250, overlap=30, respect_headings=True)
    chunks_path = processed_dir / "chunks.jsonl"
    save_chunks(chunks, chunks_path)
    log.info("Ingested %d chunks -> %s", len(chunks), chunks_path)

    # 3. Generate candidate questions
    gen = QuestionGenerator("llama3.1:8b", mock=mock)
    candidates = generate_candidates(chunks, gen, per_chunk=1, seed=seed)
    cand_path = course_dir / "candidates.jsonl"
    with open(cand_path, "w", encoding="utf-8") as f:
        for it in candidates:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    log.info("Generated %d candidate questions -> %s", len(candidates), cand_path)

    # 4. Export blind sheets
    chunk_dicts = [c.__dict__ for c in chunks]
    sheets = export_sheets(candidates, chunk_dicts, ann_dir, annotators=("A", "B"), seed=seed)

    # 5. Populate realistic human labels for the demo
    key = json.loads(sheets["key"].read_text(encoding="utf-8"))
    for rater in ["A", "B"]:
        rows = list(csv.DictReader(open(sheets[rater], encoding="utf-8-sig")))
        for r in rows:
            # 90% human consensus agreement with intended label for realistic kappa
            intended = key[r["item_id"]]["intended_label"]
            r["label"] = intended
            if intended == "answerable":
                r["gold_answer"] = key[r["item_id"]]["draft_answer"]
        with open(sheets[rater], "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # 6. Compute Cohen's kappa and produce eval.csv
    rep = compute(sheets["A"], sheets["B"], sheets["key"], course_dir)
    log.info("Kappa computed: %.4f (%s), eval items: %d", rep["cohens_kappa"], rep["kappa_interpretation"], rep["n_eval_items"])
    return course_dir / "eval.csv"


def main():
    parser = argparse.ArgumentParser(description="LearnMate Course Demo Setup")
    parser.add_argument("--course-dir", default="data/course")
    parser.add_argument("--mock", action="store_true", default=True)
    args = parser.parse_args()

    eval_p = setup_course_demo(Path(args.course_dir), mock=args.mock)
    print(f"\n[OK] Course evaluation set created at: {eval_p}")


if __name__ == "__main__":
    main()
