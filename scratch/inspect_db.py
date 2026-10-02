"""Inspect trading_state.db trades, sessions, and daily state."""
import sqlite3
import json

conn = sqlite3.connect('trading_state.db')
cursor = conn.cursor()

print("=== RECENT TRADES ===")
try:
    cols = [d[0] for d in cursor.execute("SELECT * FROM trade_log LIMIT 1").description]
    for row in cursor.execute("SELECT * FROM trade_log ORDER BY id DESC LIMIT 10;").fetchall():
        d = dict(zip(cols, row))
        print(f"[{d.get('timestamp')}] #{d.get('id')} {d.get('symbol')} {d.get('direction')} "
              f"lots={d.get('lot_size')} entry={d.get('entry_price')} "
              f"status={d.get('status')} strat={d.get('strategy_name')}")
except Exception as e:
    print("No trade_log data or error:", e)

print("\n=== RECENT BOT SESSIONS ===")
try:
    cols = [d[0] for d in cursor.execute("SELECT * FROM bot_sessions LIMIT 1").description]
    for row in cursor.execute("SELECT * FROM bot_sessions ORDER BY id DESC LIMIT 5;").fetchall():
        d = dict(zip(cols, row))
        print(f"Session #{d.get('id')}: status={d.get('status')} duration={d.get('duration_seconds')}s "
              f"symbols={d.get('symbols')} reason={d.get('deactivation_reason')}")
except Exception as e:
    print("No bot_sessions data or error:", e)

conn.close()
