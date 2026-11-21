# ─────────────────────────────────────────────────────────────────────────────
# LearnMate — Makefile
# Windows users: install GNU Make via `choco install make` or use `nmake`.
# All paths use forward slashes; PowerShell is used for setup targets.
# ─────────────────────────────────────────────────────────────────────────────

PYTHON      = python
PYTEST      = pytest
PIP         = pip
CONFIG_DIR  = configs
SRC_DIR     = src
TEST_DIR    = tests
RESULTS_DIR = results

.PHONY: help setup install test lint \
        phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase8 phase10 \
        b1 b2 proposed ablations stress course course-demo \
        app clean

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  LearnMate — available targets"
	@echo "  ───────────────────────────────────────────────────────────────"
	@echo "  setup          Install deps + pull Ollama models"
	@echo "  install        pip install -r requirements.txt"
	@echo "  test           Run all unit tests with coverage"
	@echo "  phase1         Download SQuAD, audit, build splits"
	@echo "  phase2         (no separate run — metrics/stats are library code)"
	@echo "  phase3         Build FAISS index, run extractive QA sanity check"
	@echo "  b1             Run B1 (standard RAG) experiment"
	@echo "  b2             Run B2 (calibrated confidence) experiment"
	@echo "  proposed       Run proposed method experiment"
	@echo "  ablations      Run all ablations"
	@echo "  stress         Run stress / OOD conditions"
	@echo "  course-demo    Set up 3-chapter ML course demo"
	@echo "  course         Run course material experiments"
	@echo "  phase10        Generate publication tables, figures & failure gallery"
	@echo "  app            Launch Streamlit demo"
	@echo "  clean          Remove generated artifacts (keeps data)"
	@echo ""

# ── Environment setup ─────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

setup: install
	@echo "Pulling Ollama models (requires Ollama daemon running)..."
	ollama pull llama3.2:3b
	ollama pull llama3.1:8b
	@echo "Done. Run 'make test' to verify the environment."

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

test-fast:
	$(PYTEST) $(TEST_DIR) -v -x --ignore=$(TEST_DIR)/test_pipeline.py

# ── Phase 1: Data ─────────────────────────────────────────────────────────────
phase1:
	$(PYTHON) scripts/make_squad_splits.py --config $(CONFIG_DIR)/default.yaml
	$(PYTHON) scripts/audit_squad.py       --config $(CONFIG_DIR)/default.yaml

# ── Phase 3: Retriever + Extractive QA ───────────────────────────────────────
phase3:
	$(PYTHON) -m src.retrieve build-index \
	    --config $(CONFIG_DIR)/default.yaml --split train
	$(PYTHON) -m src.extractive_qa sanity-check \
	    --config $(CONFIG_DIR)/default.yaml

# ── Experiments ───────────────────────────────────────────────────────────────
b1:
	$(PYTHON) experiments/run_baselines.py \
	    --config $(CONFIG_DIR)/b1_standard_rag.yaml

b2:
	$(PYTHON) experiments/run_baselines.py \
	    --config $(CONFIG_DIR)/b2_calibrated_confidence.yaml

proposed:
	$(PYTHON) experiments/run_proposed.py \
	    --config $(CONFIG_DIR)/proposed.yaml

ablations:
	$(PYTHON) experiments/run_ablations.py \
	    --config $(CONFIG_DIR)/proposed.yaml

stress:
	$(PYTHON) experiments/run_stress.py \
	    --config $(CONFIG_DIR)/proposed.yaml

course-demo:
	$(PYTHON) scripts/setup_course_demo.py --course-dir data/course

course:
	$(PYTHON) experiments/run_course.py \
	    --config $(CONFIG_DIR)/proposed.yaml

phase10:
	$(PYTHON) scripts/generate_phase10.py \
	    --config $(CONFIG_DIR)/proposed.yaml

# ── Demo app ──────────────────────────────────────────────────────────────────
app:
	streamlit run app/main.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .cache/embeddings 2>/dev/null || true
	@echo "Cleaned. Data, models, and results/ are preserved."
