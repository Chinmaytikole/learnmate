"""
gen_course_questions.py
-----------------------
Generate candidate course questions from chunks using a DIFFERENT LLM than the
RAG generator (default llama3.1:8b vs. generator llama3.2:3b) to limit
self-preference bias.

For each sampled chunk:
  - 1 ANSWERABLE question (answer is stated in the chunk) + draft answer
  - 1 UNANSWERABLE question (on-topic, but NOT answered by the chunk)

These labels are only *intended* labels. They are hidden from annotators and
verified blind (export_annotation_sheet.py -> compute_kappa.py).

Usage:
  python scripts/gen_course_questions.py --chunks data/course/processed/chunks.jsonl \
      --out data/course/candidates.jsonl --per-chunk 1 [--mock]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingest import load_chunks
from src.utils import get_logger, load_config, seed_everything

log = get_logger(__name__)

ANSWERABLE_PROMPT = """You are writing exam questions for a university course.
Read the passage and write ONE self-contained question that is clearly answered by a specific fact in the passage.
Do not refer to "the passage" in the question. Give a short answer copied or closely paraphrased from the passage.
Respond ONLY with JSON: {{"question": "...", "answer": "..."}}

Passage:
{text}"""

UNANSWERABLE_PROMPT = """You are writing exam questions for a university course.
Read the passage and write ONE question on the SAME TOPIC that a student might plausibly ask,
but whose answer is NOT stated or derivable from the passage.
Do not refer to "the passage" in the question.
Respond ONLY with JSON: {{"question": "..."}}

Passage:
{text}"""


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class QuestionGenerator:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434",
                 temperature: float = 0.7, mock: bool = False):
        self.model, self.temperature, self.mock = model, temperature, mock
        self._client = None
        if not mock:
            import ollama
            self._client = ollama.Client(host=host)

    def _chat(self, prompt: str) -> dict:
        resp = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature, "seed": 42},
            format="json",
        )
        return _parse_json(resp["message"]["content"])

    def answerable(self, text: str) -> dict:
        if self.mock:
            first = re.split(r"(?<=[.!?])\s+", text.strip())[0]
            return {"question": f"What does the text state about: {' '.join(first.split()[:6])}?",
                    "answer": first}
        return self._chat(ANSWERABLE_PROMPT.format(text=text))

    def unanswerable(self, text: str) -> dict:
        if self.mock:
            topic = " ".join(text.split()[:4])
            return {"question": f"What are the historical origins of {topic}?"}
        return self._chat(UNANSWERABLE_PROMPT.format(text=text))


def generate_candidates(chunks, gen: QuestionGenerator, per_chunk: int = 1,
                        max_chunks: int | None = None, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    chunks = list(chunks)
    rng.shuffle(chunks)
    if max_chunks:
        chunks = chunks[:max_chunks]
    out, n = [], 0
    for c in chunks:
        c_text = c.text if hasattr(c, "text") else c["text"]
        c_pid = c.passage_id if hasattr(c, "passage_id") else c["passage_id"]
        c_doc = c.doc if hasattr(c, "doc") else c.get("doc", "")
        c_chap = c.chapter if hasattr(c, "chapter") else c.get("chapter", "")

        for _ in range(per_chunk):
            for label, fn in (("answerable", gen.answerable), ("unanswerable", gen.unanswerable)):
                try:
                    r = fn(c_text)
                except Exception as e:  # network / model failure: skip, do not fabricate
                    log.warning("generation failed for %s: %s", c_pid, e)
                    continue
                q = str(r.get("question", "")).strip()
                if len(q.split()) < 4 or not q.endswith("?"):
                    continue
                n += 1
                out.append({
                    "item_id": f"q{n:05d}",
                    "source_passage_id": c_pid,
                    "doc": c_doc,
                    "chapter": c_chap,
                    "question": q,
                    "intended_label": label,
                    "draft_answer": str(r.get("answer", "")).strip() if label == "answerable" else "",
                })
    # drop duplicate questions
    seen, dedup = set(), []
    for it in out:
        k = it["question"].lower()
        if k not in seen:
            seen.add(k)
            dedup.append(it)
    return dedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--chunks", default="data/course/processed/chunks.jsonl")
    ap.add_argument("--out", default="data/course/candidates.jsonl")
    ap.add_argument("--per-chunk", type=int, default=1)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    seed_everything(int(cfg.project.seed))
    model = str(cfg.generator.question_gen_model)
    assert model != str(cfg.generator.model), "Question generator must differ from RAG generator"
    gen = QuestionGenerator(model, host=str(cfg.generator.get("ollama_host", "http://127.0.0.1:11434")),
                            mock=a.mock)
    items = generate_candidates(load_chunks(Path(a.chunks)), gen, a.per_chunk, a.max_chunks,
                                int(cfg.project.seed))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_ans = sum(i["intended_label"] == "answerable" for i in items)
    print(f"[OK] {len(items)} candidates ({n_ans} answerable, {len(items)-n_ans} unanswerable) -> {out}")


if __name__ == "__main__":
    main()
