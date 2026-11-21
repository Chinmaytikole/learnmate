"""
compute_kappa.py
----------------
Inter-annotator agreement and final eval-set construction.

- Cohen's kappa between annotators A and B on the label column.
- Agreement of each annotator (and of the A/B consensus) with the generator's intended label.
- Final eval set (data/course/eval.csv): items where both annotators agree.
  Human consensus label is ground truth, even if it differs from the intended label.
  Disagreements between annotators are written to disagreements.csv for adjudication.

Usage:
  python scripts/compute_kappa.py --a data/course/annotation/annotation_A.csv \
      --b data/course/annotation/annotation_B.csv --key data/course/annotation/key_intended_labels.json
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

VALID = ("answerable", "unanswerable")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters over nominal labels."""
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def interpret(k: float) -> str:
    if k < 0.0: return "poor"
    if k <= 0.20: return "slight"
    if k <= 0.40: return "fair"
    if k <= 0.60: return "moderate"
    if k <= 0.80: return "substantial"
    return "almost perfect"


def _read(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        out = {}
        for r in csv.DictReader(f):
            r["label"] = r.get("label", "").strip().lower()
            out[r["item_id"]] = r
        return out


def compute(a_path: Path, b_path: Path, key_path: Path, out_dir: Path) -> dict:
    A, B = _read(a_path), _read(b_path)
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    ids = sorted(set(A) & set(B))
    missing = [i for i in ids if A[i]["label"] not in VALID or B[i]["label"] not in VALID]
    if missing:
        raise ValueError(f"{len(missing)} items have missing/invalid labels, e.g. {missing[:3]}. "
                         f"Allowed: {VALID}")
    la = [A[i]["label"] for i in ids]
    lb = [B[i]["label"] for i in ids]
    li = [key[i]["intended_label"] for i in ids]

    kappa = cohen_kappa(la, lb)
    agree = [i for i in ids if A[i]["label"] == B[i]["label"]]
    dis = [i for i in ids if A[i]["label"] != B[i]["label"]]

    eval_rows = []
    for i in agree:
        lab = A[i]["label"]
        ga, gb = A[i].get("gold_answer", "").strip(), B[i].get("gold_answer", "").strip()
        gold = [g for g in dict.fromkeys([ga, gb, key[i]["draft_answer"].strip()]) if g] \
            if lab == "answerable" else []
        eval_rows.append({
            "item_id": i, "question": key[i]["question"], "is_answerable": lab == "answerable",
            "gold_answers": "|".join(gold),
            "gold_passage_id": key[i]["source_passage_id"] if lab == "answerable" else "",
            "source_passage_id": key[i]["source_passage_id"],
            "chapter": key[i]["chapter"], "doc": key[i]["doc"],
            "intended_label": key[i]["intended_label"],
        })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if eval_rows:
        with open(out_dir / "eval.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_rows[0]))
            w.writeheader()
            w.writerows(eval_rows)
    with open(out_dir / "disagreements.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "question", "label_A", "label_B", "intended"])
        for i in dis:
            w.writerow([i, key[i]["question"], A[i]["label"], B[i]["label"], key[i]["intended_label"]])

    n = len(ids)
    consensus_match = sum(1 for i in agree if A[i]["label"] == key[i]["intended_label"])
    report = {
        "n_items": n,
        "cohens_kappa": kappa,
        "kappa_interpretation": interpret(kappa),
        "raw_agreement": len(agree) / n,
        "n_disagreements": len(dis),
        "annotatorA_vs_intended": sum(x == y for x, y in zip(la, li)) / n,
        "annotatorB_vs_intended": sum(x == y for x, y in zip(lb, li)) / n,
        "consensus_vs_intended": consensus_match / max(1, len(agree)),
        "kappa_A_vs_intended": cohen_kappa(la, li),
        "kappa_B_vs_intended": cohen_kappa(lb, li),
        "n_eval_items": len(eval_rows),
        "n_eval_answerable": sum(r["is_answerable"] for r in eval_rows),
        "n_eval_unanswerable": sum(not r["is_answerable"] for r in eval_rows),
    }
    (out_dir / "kappa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/course/annotation/annotation_A.csv")
    ap.add_argument("--b", default="data/course/annotation/annotation_B.csv")
    ap.add_argument("--key", default="data/course/annotation/key_intended_labels.json")
    ap.add_argument("--out-dir", default="data/course")
    a = ap.parse_args()
    rep = compute(Path(a.a), Path(a.b), Path(a.key), Path(a.out_dir))
    for k, v in rep.items():
        print(f"  {k:<26}: {v:.4f}" if isinstance(v, float) else f"  {k:<26}: {v}")


if __name__ == "__main__":
    main()
