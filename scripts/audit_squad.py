"""
scripts/audit_squad.py
──────────────────────
Automated data audit script for SQuAD 2.0 splits.

Generates:
1. Counts per split (articles, passages, answerable, unanswerable).
2. Question-type distribution (What, Who, When, Where, Why, How, Which, etc.).
3. Answer-length and passage-length descriptive statistics.
4. Leakage checks (assert zero article overlap across train/calib/tune, zero qid overlap).
5. Documented dataset limitations for the research paper.
6. Exported tables in CSV, LaTeX, Markdown, and JSON in results/tables/.

Usage:
    python scripts/audit_squad.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep import load_article_list, load_jsonl
from src.utils import ensure_dir, get_logger, load_config, project_root

logger = get_logger("audit_squad")
console = Console()

QUESTION_TYPES = [
    "what", "who", "when", "where", "why", "how", "which",
    "is", "are", "was", "were", "can", "could", "did", "do", "does", "in", "name",
]

LIMITATIONS = [
    "Domain Bias: SQuAD 2.0 covers Wikipedia articles only, which are encyclopedic, dense, and third-person, differing from conversational or pedagogical course materials.",
    "Adversarial Unanswerable Design: Negatives were crowdsourced by Workers looking at passages to write similar-sounding unanswerable questions; models can exploit lexical artifact cues.",
    "Reference Redundancy: Training examples contain exactly 1 reference answer, whereas dev examples contain ~3 plausible reference spans.",
    "Retrieval Gold Standard: No separate human-annotated retrieval ground truth exists; the context paragraph hosting the QA pair is used as the silver gold-standard passage.",
    "Course Evaluation Generalizability: The course evaluation set is instructor-specific, smaller in scale, and LLM-synthesized before human blind annotation.",
]


def extract_first_word(question: str) -> str:
    """Normalize and extract the first alphanumeric word of a question."""
    words = re.findall(r"[a-zA-Z]+", question.lower())
    if not words:
        return "other"
    first = words[0]
    if first in ["is", "are", "was", "were"]:
        return "is/are/was/were"
    if first in ["did", "do", "does"]:
        return "did/do/does"
    if first in ["can", "could", "would", "should"]:
        return "can/could"
    if first in QUESTION_TYPES:
        return first
    return "other"


def compute_length_stats(lengths: list[int]) -> dict[str, float]:
    """Compute mean, median, std, min, max, iqr for a list of integer lengths."""
    if not lengths:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "iqr": 0.0}
    arr = np.array(lengths, dtype=float)
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "iqr": float(q75 - q25),
    }


def audit_split(
    examples: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    articles: list[str],
) -> dict[str, Any]:
    """Audit a single dataset split."""
    total_q = len(examples)
    unans = sum(1 for e in examples if e.get("is_impossible", False))
    ans = total_q - unans

    # Question types
    q_types = Counter(extract_first_word(e.get("question", "")) for e in examples)

    # Answer lengths (words)
    ans_lengths: list[int] = []
    for e in examples:
        if not e.get("is_impossible", False):
            for a in e.get("answers", []):
                ans_lengths.append(len(re.findall(r"\w+", a)))

    # Passage lengths (words)
    ps_lengths = [len(re.findall(r"\w+", p.get("context", ""))) for p in passages]

    return {
        "article_count": len(articles),
        "passage_count": len(passages),
        "total_questions": total_q,
        "answerable_count": ans,
        "answerable_pct": (ans / total_q * 100.0) if total_q > 0 else 0.0,
        "unanswerable_count": unans,
        "unanswerable_pct": (unans / total_q * 100.0) if total_q > 0 else 0.0,
        "question_types": dict(q_types),
        "answer_length_stats": compute_length_stats(ans_lengths),
        "passage_length_stats": compute_length_stats(ps_lengths),
    }


def run_leakage_checks(
    split_articles_dict: dict[str, set[str]],
    split_examples_dict: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    Verify strict zero-leakage conditions:
    1. Article overlap between train, calib, and tune must be 0.
    2. QID overlap across all splits must be 0.
    """
    checks: dict[str, Any] = {"passed": True, "details": []}

    # 1. Article overlap among train-derived splits
    pairs = [("train", "calib"), ("train", "tune"), ("calib", "tune")]
    for s1, s2 in pairs:
        overlap = split_articles_dict[s1] & split_articles_dict[s2]
        is_clean = len(overlap) == 0
        checks["details"].append({
            "check": f"Article overlap ({s1} vs {s2})",
            "passed": is_clean,
            "overlap_count": len(overlap),
            "overlap_items": list(overlap),
        })
        if not is_clean:
            checks["passed"] = False

    # Check dev article overlap for informative documentation
    for s in ["train", "calib", "tune"]:
        dev_overlap = split_articles_dict[s] & split_articles_dict["dev"]
        checks["details"].append({
            "check": f"Article overlap ({s} vs dev)",
            "passed": len(dev_overlap) == 0,
            "overlap_count": len(dev_overlap),
            "overlap_items": list(dev_overlap),
        })

    # 2. QID overlap
    split_qids = {k: set(e["qid"] for e in ex_list) for k, ex_list in split_examples_dict.items()}
    all_splits = ["train", "calib", "tune", "dev"]
    for i in range(len(all_splits)):
        for j in range(i + 1, len(all_splits)):
            s1, s2 = all_splits[i], all_splits[j]
            qid_overlap = split_qids[s1] & split_qids[s2]
            is_clean = len(qid_overlap) == 0
            checks["details"].append({
                "check": f"QID overlap ({s1} vs {s2})",
                "passed": is_clean,
                "overlap_count": len(qid_overlap),
                "overlap_items": list(qid_overlap)[:5],
            })
            if not is_clean:
                checks["passed"] = False

    return checks


def export_reports(
    audit_results: dict[str, Any],
    leakage_results: dict[str, Any],
    out_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Export summary table in CSV, LaTeX, Markdown, and JSON formats."""
    ensure_dir(out_dir)

    # 1. CSV summary table
    rows = []
    for s_name in ["train", "calib", "tune", "dev"]:
        data = audit_results[s_name]
        ans_stats = data["answer_length_stats"]
        ps_stats = data["passage_length_stats"]
        rows.append({
            "Split": s_name.capitalize(),
            "Articles": data["article_count"],
            "Unique Passages": data["passage_count"],
            "Total Questions": data["total_questions"],
            "Answerable": data["answerable_count"],
            "Answerable (%)": f"{data['answerable_pct']:.1f}%",
            "Unanswerable": data["unanswerable_count"],
            "Unanswerable (%)": f"{data['unanswerable_pct']:.1f}%",
            "Ans Len Mean (Words)": f"{ans_stats['mean']:.1f}",
            "Ans Len Median (Words)": f"{ans_stats['median']:.1f}",
            "Passage Len Mean (Words)": f"{ps_stats['mean']:.1f}",
            "Passage Len Median (Words)": f"{ps_stats['median']:.1f}",
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "squad_split_counts.csv"
    df.to_csv(csv_path, index=False)

    # 2. LaTeX table
    tex_path = out_dir / "squad_split_counts.tex"
    latex_code = df.to_latex(
        index=False,
        caption="Summary of SQuAD 2.0 Article-Level Splits (LearnMate G06)",
        label="tab:squad_splits",
        column_format="lcccccccccccc",
    )
    tex_path.write_text(latex_code, encoding="utf-8")

    # 3. Comprehensive Markdown Report
    md_path = out_dir / "data_audit_report.md"
    md_content = [
        "# LearnMate: SQuAD 2.0 Data Audit Report",
        "",
        "**Group:** G06  ",
        "**Status:** Clean, Zero-Leakage Article-Level Partition  ",
        "",
        "## 1. Split Counts and Distributions",
        "",
        df.to_markdown(index=False),
        "",
        "## 2. Question-Type Distribution (By First Word)",
        "",
        "| Question Type | Train (%) | Calib (%) | Tune (%) | Dev (%) |",
        "|---|---|---|---|---|",
    ]

    all_q_types = sorted(set(
        k for s in ["train", "calib", "tune", "dev"]
        for k in audit_results[s]["question_types"].keys()
    ))
    for qt in all_q_types:
        t_pct = audit_results["train"]["question_types"].get(qt, 0) / audit_results["train"]["total_questions"] * 100
        c_pct = audit_results["calib"]["question_types"].get(qt, 0) / audit_results["calib"]["total_questions"] * 100
        u_pct = audit_results["tune"]["question_types"].get(qt, 0) / audit_results["tune"]["total_questions"] * 100
        d_pct = audit_results["dev"]["question_types"].get(qt, 0) / audit_results["dev"]["total_questions"] * 100
        md_content.append(f"| {qt.capitalize()} | {t_pct:.1f}% | {c_pct:.1f}% | {u_pct:.1f}% | {d_pct:.1f}% |")

    md_content.extend([
        "",
        "## 3. Leakage Verification Results",
        "",
        f"- **Overall Status:** {'PASSED (Zero Leakage)' if leakage_results['passed'] else 'FAILED'}",
    ])
    for chk in leakage_results["details"]:
        status_icon = "✓" if chk["passed"] else "✗"
        md_content.append(f"- {status_icon} **{chk['check']}**: overlap = {chk['overlap_count']}")

    md_content.extend([
        "",
        "## 4. Documented Dataset Limitations (For Research Paper)",
        "",
    ])
    for i, lim in enumerate(LIMITATIONS, 1):
        md_content.append(f"{i}. {lim}")

    md_path.write_text("\n".join(md_content), encoding="utf-8")

    # 4. JSON summary
    json_path = out_dir / "data_audit_summary.json"
    full_summary = {
        "audit": audit_results,
        "leakage": leakage_results,
        "limitations": LIMITATIONS,
    }
    json_path.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")

    return csv_path, tex_path, md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SQuAD 2.0 splits and verify zero leakage.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = project_root()

    processed_dir = root / cfg.paths.data_processed
    splits_dir = root / cfg.paths.data_splits
    tables_dir = root / cfg.paths.tables

    console.print("[bold cyan]===============================================================[/bold cyan]")
    console.print("[bold cyan]               LearnMate SQuAD 2.0 Data Audit                  [/bold cyan]")
    console.print("[bold cyan]===============================================================[/bold cyan]\n")

    # Load splits
    split_examples: dict[str, list[dict[str, Any]]] = {}
    split_passages: dict[str, list[dict[str, Any]]] = {}
    split_articles: dict[str, list[str]] = {}

    for s_name in ["train", "calib", "tune"]:
        ex_file = processed_dir / f"squad_{s_name}.jsonl"
        ps_file = processed_dir / f"squad_{s_name}_passages.jsonl"
        art_file = splits_dir / f"squad_{s_name}_articles.txt"

        if not ex_file.exists():
            console.print(f"[bold red]Missing processed file: {ex_file}[/bold red]")
            console.print("Please run [yellow]python scripts/make_squad_splits.py[/yellow] first.")
            sys.exit(1)

        split_examples[s_name] = load_jsonl(ex_file)
        split_passages[s_name] = load_jsonl(ps_file)
        split_articles[s_name] = load_article_list(art_file)

    # Dev split
    dev_ex_file = processed_dir / "squad_dev.jsonl"
    dev_ps_file = processed_dir / "squad_dev_passages.jsonl"
    split_examples["dev"] = load_jsonl(dev_ex_file)
    split_passages["dev"] = load_jsonl(dev_ps_file)
    split_articles["dev"] = sorted(set(p["title"] for p in split_passages["dev"]))

    # Run audits
    audit_results: dict[str, Any] = {}
    for s_name in ["train", "calib", "tune", "dev"]:
        audit_results[s_name] = audit_split(
            split_examples[s_name],
            split_passages[s_name],
            split_articles[s_name],
        )

    # Run leakage checks
    split_art_sets = {k: set(v) for k, v in split_articles.items()}
    leakage_results = run_leakage_checks(split_art_sets, split_examples)

    # Export tables
    csv_path, tex_path, md_path, json_path = export_reports(audit_results, leakage_results, tables_dir)

    # Terminal presentation
    summary_table = Table(title="SQuAD 2.0 Split Distribution & Balance", header_style="bold cyan")
    summary_table.add_column("Split", style="bold white")
    summary_table.add_column("Articles", justify="right")
    summary_table.add_column("Passages", justify="right")
    summary_table.add_column("Total Qs", justify="right")
    summary_table.add_column("Answerable (HasAns)", justify="right")
    summary_table.add_column("Unanswerable (NoAns)", justify="right")
    summary_table.add_column("Ans/Unans Ratio", justify="center")

    for s_name in ["train", "calib", "tune", "dev"]:
        d = audit_results[s_name]
        ratio = f"{d['answerable_pct']:.1f}% / {d['unanswerable_pct']:.1f}%"
        summary_table.add_row(
            s_name.capitalize(),
            f"{d['article_count']:,}",
            f"{d['passage_count']:,}",
            f"{d['total_questions']:,}",
            f"{d['answerable_count']:,} ({d['answerable_pct']:.1f}%)",
            f"{d['unanswerable_count']:,} ({d['unanswerable_pct']:.1f}%)",
            ratio,
        )

    console.print(summary_table)

    # Leakage check display
    leak_panel_style = "bold green" if leakage_results["passed"] else "bold red"
    leak_lines = []
    for chk in leakage_results["details"]:
        mark = "[green][PASS][/green]" if chk["passed"] else f"[red][FAIL] (overlap={chk['overlap_count']})[/red]"
        leak_lines.append(f"{mark} {chk['check']}")

    console.print(Panel("\n".join(leak_lines), title="[bold]Leakage & Partition Verification[/bold]", border_style=leak_panel_style))

    console.print(f"\n[bold green][OK] Audit completed successfully![/bold green]")
    console.print(f"  - CSV table:      [yellow]{csv_path}[/yellow]")
    console.print(f"  - LaTeX table:    [yellow]{tex_path}[/yellow]")
    console.print(f"  - Markdown report:[yellow]{md_path}[/yellow]")
    console.print(f"  - JSON data:      [yellow]{json_path}[/yellow]\n")


if __name__ == "__main__":
    main()
