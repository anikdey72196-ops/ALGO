import gzip
import glob
import json

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

print("Searching logs for GBPUSD...")
log_files = sorted(glob.glob("logs/trading_bot*.log*"))
for lf in log_files[-5:]:
    res = check_log(lf)
    print(f"{lf}: {len(res)} occurrences")
    for r in res[:5]:
        try:
            d = json.loads(r)
            print("  ", d.get("text", r[:120]))
        except:
            print("  ", r[:120])
