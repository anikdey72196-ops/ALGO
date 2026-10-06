import sqlite3

conn = sqlite3.connect("trading_state.db")
cur = conn.cursor()

cur.execute("""
    SELECT 
        count(*) as total_with_pred_and_label,
        sum(case when (p_genuine >= 0.50 and label = 1) or (p_genuine < 0.50 and label = 0) then 1 else 0 end) as correct_predictions,
        sum(case when p_genuine < 0.50 and label = 0 then 1 else 0 end) as traps_caught,
        sum(case when label = 0 then 1 else 0 end) as total_traps,
        sum(case when p_genuine >= 0.50 and label = 1 then 1 else 0 end) as wins_predicted,
        sum(case when label = 1 then 1 else 0 end) as total_wins,
        sum(case when p_genuine < 0.50 and label = 1 then 1 else 0 end) as false_alarms,
        sum(case when allowed = 1 then 1 else 0 end) as allowed_count
    FROM ml_events
    WHERE p_genuine IS NOT NULL AND label IS NOT NULL
""")
row = cur.fetchone()
print("Stats (threshold 0.50):")
print(f"Total evaluated: {row[0]}")
print(f"Correct predictions: {row[1]} ({row[1]/row[0]*100:.1f}%)")
print(f"Traps caught: {row[2]} / {row[3]} ({row[2]/row[3]*100:.1f}%)")
print(f"Wins predicted: {row[4]} / {row[5]} ({row[4]/row[5]*100:.1f}%)")
print(f"False alarms: {row[6]}")
