import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import joblib
import numpy as np
import pandas as pd
from ml.trap_detector import FEATURE_ORDER, ALL_FEATURES

def deep_inspect():
    print("=================== ML MODEL LEARNED WEIGHTS & METRICS ===================")
    for model_name, file_name in [
        ("SMC Scalp (5m)", "trap_detector_SMC_SCALP_5M.joblib"),
        ("Order Flow", "trap_detector_ORDER_FLOW.joblib"),
        ("SMC Swing", "trap_detector.joblib"),
    ]:
        p = f"ml/artifacts/{file_name}"
        print(f"\n>>> Strategy: {model_name} ({file_name})")
        data = joblib.load(p)
        sgd = data.get('sgd')
        scaler = data.get('scaler')
        n_samples = data.get('n_samples', 0)
        version = data.get('version', 'unknown')
        print(f"  Version / Samples trained on: {version} ({n_samples} samples)")
        
        if sgd is not None and hasattr(sgd, 'coef_'):
            coef = sgd.coef_[0]
            intercept = sgd.intercept_[0]
            print(f"  Model Intercept: {intercept:.4f}")
            # If feature names or length matches ALL_FEATURES
            print(f"  Feature count: {len(coef)}")
            if len(coef) == len(ALL_FEATURES):
                feature_weights = list(zip(ALL_FEATURES, coef))
                feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)
                print("  Top Learned Feature Weights (Positive = Predicts Genuine, Negative = Predicts Trap):")
                for feat, w in feature_weights[:8]:
                    desc = "genuine indicator" if w > 0 else "trap indicator"
                    print(f"    - {feat:18s}: {w:+.4f} ({desc})")
        else:
            print("  SGD classifier not yet fitted or has no coef_")

    print("\n=================== RECENT MODEL PREDICTIONS IN DB ===================")
    con = sqlite3.connect('trading_state.db')
    cur = con.cursor()
    recent = cur.execute("""
        SELECT symbol, strategy, kind, direction, p_genuine, model_version, allowed, label, outcome
        FROM ml_events
        WHERE p_genuine IS NOT NULL
        ORDER BY ts DESC
        LIMIT 10
    """).fetchall()
    print(f"Recent {len(recent)} evaluated events:")
    for r in recent:
        sym, strat, kind, direct, p_gen, m_ver, allow, lbl, out = r
        lbl_str = f"Label: {lbl} ({out})" if lbl is not None else "Pending"
        print(f"  [{sym}] {strat:12s} {kind:10s} {direct:5s} | P(Genuine): {p_gen:.1%} | Allowed: {bool(allow)} | {lbl_str} ({m_ver})")
        
    print("\n=================== OVERALL METRICS ON LABELED SAMPLES ===================")
    # Check predictions vs labels
    stats = cur.execute("""
        SELECT 
            strategy,
            COUNT(*) as total,
            SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) as labeled,
            SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) as genuine_count,
            SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) as trap_count,
            AVG(CASE WHEN label = 1 THEN p_genuine ELSE NULL END) as avg_p_genuine_for_genuine,
            AVG(CASE WHEN label = 0 THEN p_genuine ELSE NULL END) as avg_p_genuine_for_trap
        FROM ml_events
        GROUP BY strategy
    """).fetchall()
    for s in stats:
        strat, tot, lab, gen, trp, avg_p_gen, avg_p_trp = s
        print(f"Strategy: {strat}")
        print(f"  Total events: {tot} | Labeled: {lab} (Genuine: {gen}, Traps: {trp})")
        if avg_p_gen is not None and avg_p_trp is not None:
            print(f"  Avg P(Genuine) assigned to Genuine setups: {avg_p_gen:.1%}")
            print(f"  Avg P(Genuine) assigned to Trap setups:    {avg_p_trp:.1%}")

    con.close()

if __name__ == '__main__':
    deep_inspect()
