# Paper Draft Outline

**Title:** *LearnMate: Evidence Agreement and Calibrated Abstention for Education Technology*  
**Group:** M.Tech Group G06  

---

## Abstract
- **Context & Problem:** Course-specific LLM tutors frequently fabricate plausible-sounding answers or cite irrelevant lecture chunks when student questions fall outside uploaded materials. In educational settings, ungrounded hallucinations actively harm student learning.
- **Hypothesis & Approach:** Combining dense retrieval (`BAAI/bge-base-en-v1.5`), citation-constrained generation (`Llama 3.2 3B`), an explicit NLI cross-encoder answer-evidence agreement scorer (`cross-encoder/nli-deberta-v3-small`), and post-hoc Platt calibration enables selective abstention tuned to explicit risk budgets (5%, 10%).
- **Core Claim:** Abstaining on a calibrated score that incorporates answer-evidence agreement achieves a lower unsupported-answer rate at matched coverage than abstaining on retrieval similarity or generator confidence alone.
- **Evaluation:** Tested on article-disjoint SQuAD 2.0 splits (zero leakage across train/calib/tune/dev) and a human-verified course benchmark (dual blind annotation with Cohen's $\kappa$), paired with an open Streamlit tutor interface featuring pre-query Material Quality Auditing.

---

## 1. Introduction
1. **Motivation in EdTech:** Why "confident wrong answers" are uniquely damaging in formative student tutoring compared to generic web search.
2. **Failure Modes of Standard RAG:** Retrieval similarity scores are poorly calibrated across topics, and small open-weights LLMs remain overconfident even when citing non-supporting chunks.
3. **Research Question & Hypothesis:** Does adding an NLI answer-evidence agreement feature inside a strictly calibrated abstention framework reduce unsupported answers at fixed risk targets?
4. **Summary of Contributions:**
   - **C1 (Grounded Abstention Framework):** Integration of cross-encoder NLI entailment over cited chunks with Platt/Isotonic probability calibration (`src/agree.py`, `src/calibrate.py`, `src/abstain.py`).
   - **C2 (Leakage-Free Calibration Protocol):** Strict separation of `calib` (probability mapping), `tune` (risk-budget threshold selection), and `dev`/held-out reporting (`scripts/make_squad_splits.py`, `scripts/audit_squad.py`).
   - **C3 (EdTech Workflow & Material Audit):** Multi-format course ingestion (`src/ingest.py`), separate-LLM synthetic question generation (`llama3.1:8b`) with blind dual-rater verification (`scripts/compute_kappa.py`), and a pre-chat Material Report flagging scanned PDFs, tiny fragments, and duplicates (`app/audit.py`, `app/main.py`).

---

## 2. Related Work
- **Retrieval-Augmented Generation & Attribution:** Citation-grounded QA and passage verification.
- **Selective Prediction & Abstention in QA:** Risk-coverage trade-offs, AURC, conformal/selective risk control (SQuAD 2.0 unanswerable detection).
- **Confidence Calibration & NLI Entailment:** Platt scaling, isotonic regression, ECE/Brier evaluation, and NLI cross-encoders for hallucination/faithfulness detection.
- **AI Tutors for Education:** Guardrails, source provenance, and instructor authoring workflows.

---

## 3. System Architecture & Method
1. **Dense Retrieval (`src/retrieve.py`):** L2-normalized `BAAI/bge-base-en-v1.5` embeddings indexed with FAISS `IndexFlatIP`, retrieving top-$k$ course or corpus chunks.
2. **Citation-Constrained Generation (`src/generate.py`):** Local `Llama 3.2 3B` via Ollama prompted to output strict JSON `{"answer": ..., "cited_passage_ids": [...]}` or `NO_ANSWER`.
3. **NLI Evidence-Agreement Scorer (`src/agree.py`):**
   - Declarative hypothesis transformation `format_nli_hypothesis(q, a)`.
   - Max-pooling over cited premises: $s_{\text{agree}} = \max_{p \in C} P_{\text{entail}}(p \Rightarrow h(q, a))$ via `cross-encoder/nli-deberta-v3-small`.
4. **Feature Vector & Post-Hoc Calibration (`src/features.py`, `src/calibrate.py`):**
   - Feature vector: `[top_retrieval_sim, mean_token_logprob, min_token_logprob, answer_length, agreement_score]`.
   - Target $y \in \{0,1\}$: Grounded Correctness (`is_grounded_correct`: answerable + token $\text{F1} \ge 0.5$ + gold passage cited; unanswerable must abstain).
   - Fitted strictly on `calib` using Platt scaling (`LogisticRegression` on standardized features) or Isotonic regression.
5. **Risk-Controlled Threshold Selection (`src/abstain.py`):**
   - Grid search over $\tau \in [0, 1]$ strictly on `tune` to maximize coverage subject to $\widehat{\text{Risk}}(\tau) \le \alpha \in \{0.05, 0.10\}$.

---

## 4. Experimental Setup & Datasets
1. **SQuAD 2.0 Benchmark (`results/tables/squad_split_counts.tex`):**
   - Article-level partitioning (80% train / 10% calib / 10% tune, seed 42); official `dev` locked for final evaluation.
   - Zero article/QID leakage verified via `scripts/audit_squad.py` (`results/tables/data_audit_report.md`).
2. **Course Material Benchmark (when completed in Phase 8):**
   - 150–250 word heading-aware chunking with 30-word overlap (`src/ingest.py`).
   - Cross-LLM candidate generation (`llama3.1:8b`, distinct from `llama3.2:3b`) + blind A/B annotation + Cohen's $\kappa$ consensus (`data/course/kappa_report.json`).
   - In-distribution `tune`, held-out chapter split, and stressed conditions.
3. **Baselines & Ablations:**
   - **B1:** Standard RAG (abstains only on generator `NO_ANSWER`).
   - **B2:** Calibrated Confidence without NLI agreement (`top_retrieval_sim`, logprobs, `answer_length`).
   - **Ablations (`experiments/run_ablations.py`):** Agreement-Only, Retrieval-Sim-Only, and Uncalibrated Heuristic ($0.5\,\text{sim} + 0.5\,\text{agree}$).
4. **Metrics & Statistical Testing (`src/metrics.py`, `src/stats.py`):**
   - EM, Token F1, HasAns/NoAns F1, Citation F1, Unsupported-Answer Rate, Coverage, AURC, Cov@5%, Cov@10%, ECE, Brier score.
   - 1,000-resample 95% bootstrap CIs, paired bootstrap tests, and McNemar's test.

---

## 5. Results & Analysis *(Populate with Full `dev` & Course Runs)*
1. **Main Comparison (`results/tables/main_results_table.tex`, `results/tables/significance_tests.csv`):**
   - Compare B1, B2, and Proposed across Unsupported-Answer Rate, AURC, and Coverage@5%/10% risk.
2. **Calibration Analysis (`results/figures/reliability_before_after.png`):**
   - ECE and Brier reduction from raw uncalibrated heuristic vs Platt-calibrated confidence.
3. **Risk-Coverage Trade-Off (`results/figures/main_risk_coverage.png`):**
   - Visual demonstration of lower risk across operating coverages.
4. **Ablation Study (`results/tables/ablation_table.tex`, `results/figures/ablation_*.png`):**
   - Quantify individual gains from NLI agreement vs retrieval similarity vs calibration.
5. **Failure Taxonomy & NLI Rescue Gallery (`results/failure_cases/failure_gallery.md`):**
   - Quantitative breakdown across 6 categories (`nli_rescue`, `false_answer_unanswerable`, `unsupported_citation`, `false_abstention`, `wrong_span_supported`, `correct`) + qualitative case studies.

---

## 6. Deployment & Pedagogical Prototype (`app/main.py`)
- Interactive Streamlit tutor with **Material Report** (catching scanned PDFs, low word counts, duplicate slides before tutoring begins), **Ask (Chat)** with real-time agreement/confidence/citation provenance, **Strictness Presets** (`Lenient`, `Balanced`, `Strict`, custom $\tau$), and **Batch CSV** grading/auditing.

---

## 7. Limitations, Ethics & Conclusion
- **Limitations (from `data_audit_report.md`):** Wikipedia vs pedagogical style mismatch; crowdsourced negative artifacts in SQuAD 2.0; silver paragraph citation ground truth; local 3B generator capacity limits; course evaluation still pending real instructor dataset & dual human annotation (Phase 8).
- **Educational Ethics:** Abstention prevents misleading students, while showing nearest candidate passages turns refusals into guided reading opportunities.
- **Conclusion:** Explicit NLI answer-evidence agreement combined with split-safe probability calibration provides an interpretable, consumer-GPU-ready foundation for trustworthy educational AI assistants.
