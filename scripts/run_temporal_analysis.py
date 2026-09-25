"""
run_temporal_analysis.py
========================

Standalone CLI tool to train and run the Temporal & Session ML Model.
Analyzes trade logs and ML events to identify:
- Which day of the week is most profitable vs most losing
- Which time (hour/minute) has the highest expected return vs loss rate
- Which market session / ICT Kill Zone produces institutional edge vs retail stop-outs
- Day x Session cross-matrix and strategy performance breakdown

Usage:
    python run_temporal_analysis.py
    python run_temporal_analysis.py --retrain
    python run_temporal_analysis.py --export-json temporal_report.json
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.temporal_analyzer import TemporalEdgeService, TemporalModelConfig


def main():
    parser = argparse.ArgumentParser(
        description="ML Temporal & Session Profitability Analyzer for ALGO Trading Bot"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="trading_state.db",
        help="Path to trading_state.db (default: trading_state.db)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retrain of the ML model on latest DB records",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Export full breakdown JSON to specified path",
    )
    args = parser.parse_args()

    print("================================================================================")
    print("      INITIALIZING INSTITUTIONAL TEMPORAL & SESSION ML ANALYZER                 ")
    print("================================================================================")

    cfg = TemporalModelConfig(db_path=args.db)
    svc = TemporalEdgeService(cfg)

    if args.retrain:
        print("[+] Retraining model on latest records...")
        svc.train_and_update()

    report = svc.generate_ascii_report()
    print(report)

    if args.export_json:
        import json
        full_data = svc.get_full_breakdown()
        out_path = Path(args.export_json)
        out_path.write_text(json.dumps(full_data, indent=2))
        print(f"\n[+] Exported full JSON report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
