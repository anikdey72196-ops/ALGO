import sqlite3
import glob

for db in glob.glob("*.db"):
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cur.fetchall()]
        for t in tables:
            try:
                cur.execute(f"PRAGMA table_info({t});")
                cols = [c[1] for c in cur.fetchall()]
                if 'symbol' in cols:
                    cur.execute(f"SELECT * FROM {t} WHERE symbol LIKE '%GBP%' ORDER BY rowid DESC LIMIT 10;")
                    rows = cur.fetchall()
                    if rows:
                        print(f"=== {db} : {t} ===")
                        print("Columns:", cols)
                        for r in rows:
                            print(r)
            except Exception as e:
                pass
        conn.close()
    except Exception as e:
        pass
