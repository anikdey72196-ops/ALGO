import sqlite3
from typing import Any

def get_shadow_prediction_stats(db_path: str = "trading_state.db", threshold: float = 0.50) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 
            COUNT(*),
            SUM(label IS NOT NULL),
            SUM(p_genuine IS NOT NULL),
            SUM(CASE WHEN p_genuine IS NOT NULL AND label IS NOT NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN (p_genuine >= ? AND label = 1) OR (p_genuine < ? AND label = 0) THEN 1 ELSE 0 END),
            SUM(CASE WHEN p_genuine < ? AND label = 0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN p_genuine >= ? AND label = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN p_genuine < ? AND label = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN p_genuine >= ? AND label = 0 THEN 1 ELSE 0 END),
            AVG(CASE WHEN p_genuine IS NOT NULL THEN p_genuine END)
        FROM ml_events
        """,
        (threshold, threshold, threshold, threshold, threshold, threshold),
    )
    (
        total_events,
        labeled_events,
        predicted_events,
        evaluated_events,
        correct_predictions,
        traps_caught,
        total_traps,
        genuine_caught,
        total_genuine,
        false_traps,
        missed_traps,
        avg_confidence,
    ) = cur.fetchone()

    evaluated = int(evaluated_events or 0)
    correct = int(correct_predictions or 0)
    accuracy = round((correct / evaluated * 100), 1) if evaluated > 0 else 0.0
    
    t_caught = int(traps_caught or 0)
    t_total = int(total_traps or 0)
    trap_recall = round((t_caught / t_total * 100), 1) if t_total > 0 else 0.0

    g_caught = int(genuine_caught or 0)
    g_total = int(total_genuine or 0)
    win_recall = round((g_caught / g_total * 100), 1) if g_total > 0 else 0.0

    cur_recent = cur.execute(
        """
        SELECT event_id, ts, symbol, direction, p_genuine, label, outcome, model_version, strategy
        FROM ml_events
        WHERE p_genuine IS NOT NULL
        ORDER BY ts DESC
        LIMIT 5
        """
    )
    recent_preds = []
    for r in cur_recent.fetchall():
        p_gen = float(r[4]) if r[4] is not None else 0.5
        would_veto = p_gen < threshold
        recent_preds.append({
            "event_id": r[0],
            "ts": r[1],
            "symbol": r[2],
            "direction": r[3],
            "p_genuine": round(p_gen, 3),
            "p_sl": round(1.0 - p_gen, 3),
            "predicted_action": "VETO" if would_veto else "ALLOW",
            "label": r[5],
            "outcome": r[6],
            "was_correct": (would_veto and r[5] == 0) or (not would_veto and r[5] == 1) if r[5] is not None else None,
            "model_version": r[7],
            "strategy": r[8],
        })

    return {
        "total_events": int(total_events or 0),
        "predicted_events": int(predicted_events or 0),
        "evaluated_events": evaluated,
        "correct_predictions": correct,
        "prediction_accuracy_pct": accuracy,
        "traps_caught": t_caught,
        "total_traps": t_total,
        "trap_detection_rate_pct": trap_recall,
        "genuine_predicted": g_caught,
        "total_genuine": g_total,
        "win_prediction_rate_pct": win_recall,
        "false_alarms": int(false_traps or 0),
        "missed_traps": int(missed_traps or 0),
        "avg_confidence": round(float(avg_confidence or 0.5), 3),
        "threshold": threshold,
        "recent_predictions": recent_preds,
    }

import json
res = get_shadow_prediction_stats()
print(json.dumps(res, indent=2))
