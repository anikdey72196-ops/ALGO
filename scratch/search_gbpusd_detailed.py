import gzip
import glob
import json
import sys

def check_log(fpath):
    opener = gzip.open if fpath.endswith(".gz") else open
    matches = []
    try:
        with opener(fpath, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "GBPUSD" in line:
                    matches.append(line.strip())
    except Exception as e:
        pass
    return matches

log_files = sorted(glob.glob("logs/trading_bot*.log*"))
for lf in log_files[-3:]:
    res = check_log(lf)
    print(f"\n=== {lf}: {len(res)} matches ===")
    for r in res:
        try:
            d = json.loads(r)
            txt = d.get("text", "")
            # filter out startup noise
            if any(k in txt for k in ["Analyzing Pair", "SIGNAL", "No signals", "rejected", "AUTHORIZED", "ORDER", "REVERSAL", "spread", "Spread", "NewsFilter", "News"]):
                print(" ", txt.strip().encode("ascii", errors="replace").decode("ascii"))
        except Exception as e:
            pass
