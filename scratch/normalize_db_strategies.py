import sqlite3

conn = sqlite3.connect('trading_state.db')
c = conn.cursor()

c.execute("UPDATE trade_log SET strategy_name = 'SMC Swing' WHERE strategy_name IN ('SMC', 'SMC Swing (15m)', 'SMC_SWING')")
c.execute("UPDATE trade_log SET strategy_name = '5M Scalp' WHERE strategy_name IN ('SMC_SCALP_5M', 'SMC Scalp (5m)')")
c.execute("UPDATE trade_log SET strategy_name = 'ICT Institutional' WHERE strategy_name IN ('ICT', 'ICT KillZone / Silver Bullet')")
c.execute("UPDATE trade_log SET strategy_name = 'Order Flow Engine' WHERE strategy_name IN ('ORDER_FLOW', 'Order Flow (Delta & Absorption)')")

conn.commit()

rows = c.execute("SELECT strategy_name, count(*), sum(realized_pnl) FROM trade_log GROUP BY strategy_name").fetchall()
print("Consolidated strategy counts in trading_state.db:")
for r in rows:
    print(f"  {r[0]}: {r[1]} trades, PnL: ${r[2]:.2f}")

conn.close()
