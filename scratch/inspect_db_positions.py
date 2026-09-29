import sqlite3

conn = sqlite3.connect("trading_state.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()

c.execute("PRAGMA table_info(trade_log)")
print("Columns:", [r["name"] for r in c.fetchall()])

c.execute("SELECT * FROM trade_log WHERE status != 'CLOSED'")
unclosed = [dict(r) for r in c.fetchall()]
print(f"\nUnclosed in trade_log ({len(unclosed)}):")
for u in unclosed:
    print(" ", u)

c.execute("SELECT * FROM trade_log WHERE symbol='GBPUSD' ORDER BY id DESC LIMIT 5")
gbp_trades = [dict(r) for r in c.fetchall()]
print(f"\nRecent GBPUSD trades ({len(gbp_trades)}):")
for g in gbp_trades:
    print(" ", g)
