"""
export_annotation_sheet.py
--------------------------
Export BLIND verification sheets for two annotators.

Each annotator gets the same items in an independently shuffled order, showing:
  item_id, question, source passage text
and EMPTY columns:
  label (answerable / unanswerable), gold_answer, notes
Intended labels and draft answers are NOT shown (blind). They are written to a
separate key file used only by compute_kappa.py.

Usage:
  python scripts/export_annotation_sheet.py --candidates data/course/candidates.jsonl \
      --out-dir data/course/annotation --annotators A B
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingest import load_chunks

SHEET_FIELDS = ["item_id", "question", "passage", "label", "gold_answer", "notes"]


def export_sheets(candidates: list[dict], chunks: list[dict], out_dir: Path,
                  annotators=("A", "B"), seed: int = 42) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_by_id = {c["passage_id"]: c["text"] for c in chunks}

    rows = [{
        "item_id": c["item_id"],
        "question": c["question"],
        "passage": text_by_id.get(c["source_passage_id"], ""),
        "label": "", "gold_answer": "", "notes": "",
    } for c in candidates]

    paths = {}
    for k, name in enumerate(annotators):
        r = list(rows)
        random.Random(seed + k).shuffle(r)
        p = out_dir / f"annotation_{name}.csv"
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
            w.writeheader()
            w.writerows(r)
        paths[name] = p

    key = out_dir / "key_intended_labels.json"   # DO NOT SHARE with annotators
    key.write_text(json.dumps({
        c["item_id"]: {"intended_label": c["intended_label"], "draft_answer": c["draft_answer"],
                       "source_passage_id": c["source_passage_id"], "chapter": c["chapter"],
                       "doc": c["doc"], "question": c["question"]}
        for c in candidates}, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["key"] = key
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/course/candidates.jsonl")
    ap.add_argument("--chunks", default="data/course/processed/chunks.jsonl")
    ap.add_argument("--out-dir", default="data/course/annotation")
    ap.add_argument("--annotators", nargs="+", default=["A", "B"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    with open(a.candidates, encoding="utf-8") as f:
        cands = [json.loads(l) for l in f if l.strip()]
    paths = export_sheets(cands, load_chunks(Path(a.chunks)), Path(a.out_dir), a.annotators, a.seed)
    for k, p in paths.items():
        print(f"[OK] {k}: {p}")
    print("Instruction: label each item 'answerable' if the passage states the answer, else 'unanswerable'.")


if __name__ == "__main__":
    main()
