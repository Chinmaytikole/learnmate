"""
scripts/make_squad_splits.py
────────────────────────────
CLI entrypoint to download official SQuAD 2.0 data, partition the TRAIN set
by ARTICLE TITLE into train (80%), calib (10%), tune (10%), and export processed
JSONL splits and article lists.

Usage:
    python scripts/make_squad_splits.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.data_prep import prepare_squad_splits
from src.utils import get_logger, load_config, seed_everything

logger = get_logger("make_squad_splits")
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create official SQuAD 2.0 article-level splits.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download official SQuAD JSON files even if already present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    console.print(f"[bold cyan]LearnMate Data Pipeline: SQuAD 2.0 Article-Level Splits[/bold cyan]")
    console.print(f"Config: [yellow]{args.config}[/yellow] | Seed: [yellow]{cfg.project.seed}[/yellow]\n")

    summary = prepare_squad_splits(cfg, force_download=args.force_download)

    table = Table(title="SQuAD 2.0 Dataset & Split Summary", show_header=True, header_style="bold magenta")
    table.add_column("Split Name", style="cyan")
    table.add_column("Articles", justify="right")
    table.add_column("Passages (Pool)", justify="right")
    table.add_column("Total Examples", justify="right")
    table.add_column("Status / Role", style="italic green")

    for s_name in ["train", "calib", "tune"]:
        table.add_row(
            s_name,
            f"{summary['articles'][s_name]:,}",
            f"{summary['passages'][s_name]:,}",
            f"{summary['examples'][s_name]:,}",
            f"Official TRAIN partition ({cfg.squad[s_name + '_ratio']*100:.0f}%)",
        )

    table.add_row(
        "dev (official)",
        f"{summary['dev_articles']:,}",
        f"{summary['dev_passages']:,}",
        f"{summary['dev_examples']:,}",
        "Final Evaluation ONLY (Locked)",
    )

    console.print(table)
    console.print("\n[bold green][OK] SQuAD splits successfully created and saved to data/splits/ and data/processed/[/bold green]\n")


if __name__ == "__main__":
    main()
