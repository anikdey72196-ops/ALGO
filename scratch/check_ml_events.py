import sqlite3

conn = sqlite3.connect("trading_state.db")
cur = conn.cursor()
cur.execute("PRAGMA table_info(ml_events)")
cols = [r[1] for r in cur.fetchall()]
print("Columns:", cols)

cur.execute("SELECT count(*), count(label), count(p_genuine), count(allowed) FROM ml_events")
print("Counts:", cur.fetchone())

cur.execute("""
    SELECT count(*), 
           sum(case when label = 1 then 1 else 0 end) as win_labeled,
           sum(case when label = 0 then 1 else 0 end) as loss_labeled
    FROM ml_events
""")
print("Labels:", cur.fetchone())

cur.execute("""
    SELECT p_genuine, label, allowed, model_version, outcome 
    FROM ml_events 
    WHERE p_genuine IS NOT NULL AND label IS NOT NULL 
    LIMIT 5
""")
print("Sample:", cur.fetchall())
