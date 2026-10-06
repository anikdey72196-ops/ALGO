import sqlite3
import pandas as pd
import numpy as np

con = sqlite3.connect('trading_state.db')
trades = pd.read_sql_query("SELECT * FROM trade_log WHERE status != 'OPEN'", con)
print('Closed trades count:', len(trades))
if len(trades) > 0:
    trades['dt'] = pd.to_datetime(trades['timestamp'])
    trades['day_name'] = trades['dt'].dt.day_name()
    trades['hour'] = trades['dt'].dt.hour
    trades['is_win'] = (trades['status'] == 'CLOSED_TP') | (trades['realized_pnl'] > 0)
    print('\n--- TRADES BY DAY ---')
    print(trades.groupby('day_name').agg(
        trades=('id', 'count'),
        wins=('is_win', 'sum'),
        win_rate=('is_win', lambda x: f"{x.mean()*100:.1f}%"),
        total_pnl=('realized_pnl', 'sum'),
        avg_pnl=('realized_pnl', 'mean')
    ))
    print('\n--- TRADES BY HOUR ---')
    print(trades.groupby('hour').agg(
        trades=('id', 'count'),
        wins=('is_win', 'sum'),
        win_rate=('is_win', lambda x: f"{x.mean()*100:.1f}%"),
        total_pnl=('realized_pnl', 'sum')
    ))

ml = pd.read_sql_query("SELECT * FROM ml_events WHERE outcome IS NOT NULL", con)
print('\n--- ML EVENTS COUNT:', len(ml))
if len(ml) > 0:
    ml['dt'] = pd.to_datetime(ml['ts'])
    ml['day_name'] = ml['dt'].dt.day_name()
    ml['hour'] = ml['dt'].dt.hour
    ml['is_tp'] = (ml['outcome'] == 'tp') | (ml['label'] == 1)
    print('\n--- ML EVENTS BY DAY ---')
    print(ml.groupby('day_name').agg(
        events=('event_id', 'count'),
        tps=('is_tp', 'sum'),
        tp_rate=('is_tp', lambda x: f"{x.mean()*100:.1f}%")
    ))
    print('\n--- ML EVENTS BY HOUR (top 15) ---')
    print(ml.groupby('hour').agg(
        events=('event_id', 'count'),
        tps=('is_tp', 'sum'),
        tp_rate=('is_tp', lambda x: f"{x.mean()*100:.1f}%")
    ).head(15))
con.close()
