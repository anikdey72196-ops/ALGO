import sqlite3

conn = sqlite3.connect("trading_state.db")
cur = conn.cursor()
tickets = (58703462470, 58703463018, 58703463630, 58703464133)
cur.execute(
    "UPDATE trade_log SET status = 'OPEN', closed_at = NULL, duration_seconds = NULL WHERE id IN (?, ?, ?, ?)",
    tickets
)
conn.commit()
print("Updated rows:", cur.rowcount)

# Verify
rows = cur.execute("SELECT id, symbol, status FROM trade_log WHERE id IN (?, ?, ?, ?)", tickets).fetchall()
print("Current status in DB:", rows)
conn.close()
