import json

with open('logs/trading_bot.log', 'rb') as f:
    lines = f.readlines()

for line in lines:
    try:
        s = line.decode('utf-8', errors='ignore')
        if any(w in s for w in ['ACTIVATED', 'Pair 3', 'GBPUSD', 'fixed_sl', 'Session #', 'ORDER FILLED', 'AUTHORIZE']):
            print(s.strip())
    except:
        pass
