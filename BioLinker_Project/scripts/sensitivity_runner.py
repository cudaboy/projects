"""Simple sensitivity runner for BioLinker retrieval experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from biolinker import config
from scripts.evaluate import run_ragas_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run BioLinker sensitivity sweeps")
    parser.add_argument("--modes", nargs="+", default=["vector", "graph", "both"])
    parser.add_argument("--topks", nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summaries = []
    for mode in args.modes:
        for top_k in args.topks:
            summaries.append(run_ragas_evaluation(mode, top_k, args.limit))

    out_path = config.PROCESSED_DATA_DIR / "sensitivity_summary.json"
    out_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(out_path), "runs": len(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
