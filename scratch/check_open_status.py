import sqlite3

conn = sqlite3.connect("trading_state.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()

c.execute("SELECT id, symbol, direction, status, timestamp, strategy_name FROM trade_log WHERE status = 'OPEN'")
rows = [dict(r) for r in c.fetchall()]
print(f"TRULY OPEN positions (status='OPEN'): {len(rows)}")
for r in rows:
    print(r)
