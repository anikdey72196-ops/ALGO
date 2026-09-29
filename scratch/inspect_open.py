import sqlite3

conn = sqlite3.connect("trading_state.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()

c.execute("SELECT id, symbol, direction, status, timestamp, closed_at FROM trade_log WHERE status NOT IN ('CLOSED', 'CLOSED_TP', 'CLOSED_SL')")
rows = [dict(r) for r in c.fetchall()]
print(f"Open positions in trade_log ({len(rows)}):")
for r in rows:
    print(r)
