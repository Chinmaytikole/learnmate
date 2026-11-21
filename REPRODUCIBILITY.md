# LearnMate Reproducibility Package

**Project:** LearnMate — Evidence Agreement and Calibrated Abstention for Education Technology (Group G06)  
**Global Seed:** `42` (`configs/default.yaml -> project.seed`)  

---

## 1. Locked Experimental Rules

1. **Zero Dev Leakage**:
   - `squad_train` (80% of official train articles): model training only.
   - `squad_calib` (10% of official train articles): fitting Platt scaling / Isotonic calibrators ONLY.
   - `squad_tune` (10% of official train articles): selecting abstention thresholds $\tau$ for target risks ($5\%$, $10\%$) ONLY.
   - `squad_dev`: strictly held out for final reporting; never used for calibration or threshold tuning.
   - Course evaluation: separate question-generation LLM (`llama3.1:8b`) vs answering LLM (`llama3.2:3b`), blind human dual-annotation verified via Cohen's $\kappa$.
2. **Artifact-First Reconstruction**:
   - Every run writes raw JSONL predictions alongside `*_metadata.json` (`src/utils.py::build_run_metadata`).
   - All final tables (`main_results_table.csv/.tex`, `ablation_table.csv/.tex`), significance tests, failure galleries, and figures regenerate directly from saved JSONL files via `scripts/generate_phase10.py`.

---

## 2. Environment & Models

| Role | Model / Tool | Notes |
|---|---|---|
| Dense Retriever | `BAAI/bge-base-en-v1.5` + `faiss-cpu` (`IndexFlatIP`) | Cosine similarity on L2-normalized vectors |
| Generator LLM | `llama3.2:3b` via Ollama | Temperature `0.0`, strict JSON citations |
| Extractive QA Ref | `deepset/deberta-v3-base-squad2` | Sanity-check reference baseline |
| Agreement Cross-Encoder | `cross-encoder/nli-deberta-v3-small` | Entailment index `2` |
| Question Gen LLM | `llama3.1:8b` via Ollama | Separate model family/size for course eval |

Setup on Windows (PowerShell):
```powershell
cd learnmate
python -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt
ollama pull llama3.2:3b
ollama pull llama3.1:8b
```

Verify the full test suite (offline/mock-safe):
```powershell
.\venv\Scripts\pytest.exe tests/ -v
```

---

## 3. End-to-End Reproduction Commands

```powershell
# Phase 1: Download SQuAD 2.0, create article-level splits, run leakage audit
.\venv\Scripts\python.exe scripts/make_squad_splits.py --config configs/default.yaml
.\venv\Scripts\python.exe scripts/audit_squad.py --config configs/default.yaml

# Phase 3: Build dense FAISS indexes and check extractive QA reference
.\venv\Scripts\python.exe -m src.retrieve build-index --config configs/default.yaml --split calib
.\venv\Scripts\python.exe -m src.retrieve build-index --config configs/default.yaml --split tune
.\venv\Scripts\python.exe -m src.retrieve build-index --config configs/default.yaml --split dev
.\venv\Scripts\python.exe -m src.extractive_qa sanity-check --config configs/default.yaml

# Phase 4 & 5: Run B1 Standard RAG across calib, tune, dev (requires Ollama running)
.\venv\Scripts\python.exe experiments/run_baselines.py --config configs/b1_standard_rag.yaml --split calib
.\venv\Scripts\python.exe experiments/run_baselines.py --config configs/b1_standard_rag.yaml --split tune
.\venv\Scripts\python.exe experiments/run_baselines.py --config configs/b1_standard_rag.yaml --split dev

# Phase 6: Run Proposed Method (+ NLI Agreement) and Ablation Suite on dev
.\venv\Scripts\python.exe experiments/run_proposed.py --config configs/proposed.yaml --eval-split dev --target-risk 0.05
.\venv\Scripts\python.exe experiments/run_ablations.py --config configs/proposed.yaml --split dev --target-risk 0.05

# Phase 7: Course ingestion, question generation, blind annotation export & kappa (once course files are placed in data/course/raw/)
.\venv\Scripts\python.exe -m src.ingest --input data/course/raw --output data/course/processed/chunks.jsonl
.\venv\Scripts\python.exe scripts/gen_course_questions.py --chunks data/course/processed/chunks.jsonl --out data/course/candidates.jsonl
.\venv\Scripts\python.exe scripts/export_annotation_sheet.py --candidates data/course/candidates.jsonl --chunks data/course/processed/chunks.jsonl --out-dir data/course/annotation
# (After human annotators A & B label data/course/annotation/annotation_*.csv)
.\venv\Scripts\python.exe scripts/compute_kappa.py

# Phase 9: Launch Streamlit Interactive Demo
.\venv\Scripts\streamlit.exe run app/main.py

# Phase 10: Reconstruct Main Tables, Significance Tests, Figures, Failure Gallery & Manifest (zero LLM calls)
.\venv\Scripts\python.exe scripts/generate_phase10.py --config configs/proposed.yaml --split dev
```

---

## 4. Key Output Artifacts

| Artifact | Path |
|---|---|
| Split leakage report | `results/tables/data_audit_report.md`, `results/tables/squad_split_counts.tex` |
| Main results (CSV / LaTeX) | `results/tables/main_results_table.csv`, `results/tables/main_results_table.tex` |
| Statistical significance tests | `results/tables/significance_tests.json`, `results/tables/significance_tests.csv` |
| Ablation comparison (CSV / LaTeX) | `results/tables/ablation_table.csv`, `results/tables/ablation_table.tex` |
| Risk-Coverage & Reliability plots | `results/figures/main_risk_coverage.png`, `results/figures/reliability_before_after.png` |
| Ablation figures | `results/figures/ablation_risk_coverage.png`, `results/figures/ablation_unsupported_rate.png`, `results/figures/ablation_aurc.png` |
| Failure-case gallery | `results/failure_cases/failure_gallery.md`, `results/failure_cases/failure_cases.json` |
| Phase 10 Reproducibility Manifest | `results/reproducibility_manifest.json` |
