import sqlite3, glob

for db in glob.glob('*.db'):
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        print(f"{db}: {tables}")
        if 'ml_events' in tables:
            cur.execute("SELECT count(*) FROM ml_events")
            print(f"  ml_events count: {cur.fetchone()[0]}")
    except Exception as e:
        print(f"Error reading {db}: {e}")
