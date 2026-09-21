import sqlite3
import joblib
import os
import json

def inspect():
    print("=================== 1. SQLITE DATABASE STATE ===================")
    if os.path.exists('trading_state.db'):
        con = sqlite3.connect('trading_state.db')
        cur = con.cursor()
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        print(f"Total tables: {len(tables)}")
        for t in tables:
            try:
                count = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"  Table '{t}': {count} rows")
            except Exception as e:
                print(f"  Table '{t}': error {e}")
        
        # Check ml_events or similar tables
        ml_tables = [t for t in tables if 'ml' in t.lower() or 'event' in t.lower() or 'trap' in t.lower()]
        print(f"\nML-related tables: {ml_tables}")
        for t in ml_tables:
            print(f"\n--- Schema of {t} ---")
            cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
            for c in cols:
                print(f"  {c[1]} ({c[2]})")
            
            # Check labels distribution
            print(f"\n--- Data sample / summary of {t} ---")
            try:
                col_names = [c[1] for c in cols]
                if 'label' in col_names:
                    dist = cur.execute(f"SELECT label, COUNT(*) FROM {t} GROUP BY label").fetchall()
                    print(f"Label distribution: {dist}")
                if 'outcome' in col_names:
                    dist_out = cur.execute(f"SELECT outcome, COUNT(*) FROM {t} GROUP BY outcome").fetchall()
                    print(f"Outcome distribution: {dist_out}")
                if 'strategy' in col_names:
                    dist_strat = cur.execute(f"SELECT strategy, COUNT(*) FROM {t} GROUP BY strategy").fetchall()
                    print(f"Strategy distribution: {dist_strat}")
                if 'symbol' in col_names:
                    dist_sym = cur.execute(f"SELECT symbol, COUNT(*) FROM {t} GROUP BY symbol").fetchall()
                    print(f"Symbol distribution: {dist_sym}")
            except Exception as e:
                print(f"Error querying distributions: {e}")
        con.close()
    else:
        print("trading_state.db does not exist.")

    print("\n=================== 2. ML ARTIFACTS ===================")
    artifacts_dir = 'ml/artifacts'
    if os.path.exists(artifacts_dir):
        files = os.listdir(artifacts_dir)
        print(f"Artifact files: {files}")
        for f in files:
            p = os.path.join(artifacts_dir, f)
            print(f"\nArtifact: {f} (Size: {os.path.getsize(p)} bytes)")
            try:
                obj = joblib.load(p)
                print(f"  Loaded object type: {type(obj)}")
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, (int, float, str, bool)):
                            print(f"    {k}: {v}")
                        elif isinstance(v, (list, dict)):
                            print(f"    {k} (len {len(v)}): {str(v)[:150]}")
                        else:
                            print(f"    {k}: {type(v)}")
                elif hasattr(obj, '__dict__'):
                    print(f"  Attributes: {list(obj.__dict__.keys())}")
                    for k, v in obj.__dict__.items():
                        if not k.startswith('_') and isinstance(v, (int, float, str, bool)):
                            print(f"    {k}: {v}")
                        else:
                            print(f"    {k}: {type(v)}")
                else:
                    print(f"  Representation: {obj}")
            except Exception as e:
                print(f"  Error loading artifact: {e}")
    else:
        print(f"{artifacts_dir} does not exist.")

    print("\n=================== 3. TRADES HISTORY ===================")
    if os.path.exists('trades_history.csv'):
        with open('trades_history.csv', 'r') as f:
            lines = f.readlines()
        print(f"trades_history.csv rows: {len(lines)}")
        if len(lines) > 0:
            print(f"Header: {lines[0].strip()}")
        if len(lines) > 1:
            print(f"First row: {lines[1].strip()}")
            print(f"Last row: {lines[-1].strip()}")

if __name__ == '__main__':
    inspect()
