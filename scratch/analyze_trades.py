import pandas as pd

df = pd.read_csv("trades_history.csv")
print("Total rows in trades_history.csv:", len(df))
print("\nTrades per symbol:")
print(df["symbol"].value_counts())

print("\nRecent 25 trades:")
for idx, row in df.tail(25).iterrows():
    print(f"{row['id']} | {row['timestamp'][:19]} | {row['symbol']} | {row['direction']} | PnL={row['realized_pnl']} | Status={row['status']} | Strat={row['strategy_name']}")

print("\nAll GBPUSD trades:")
gbp = df[df["symbol"] == "GBPUSD"]
print(f"Total GBPUSD trades: {len(gbp)}")
for idx, row in gbp.iterrows():
    print(f"{row['id']} | {row['timestamp'][:19]} | {row['direction']} | Entry={row['entry_price']} | SL={row['stop_loss']} | TP={row['take_profit']} | Status={row['status']}")
