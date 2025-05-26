#!/usr/bin/env python3
import backtrader as bt
import pandas as pd
import glob, datetime

from mastertrend_ml import MasterTrendML

# Glob all 15-minute CSVs
PAIRS = glob.glob('*_15M.csv')
# Define months to test (year, month)
MONTHS = [(2025, m) for m in range(1, 7)]  # Jan–Jun 2025

results = []
for f in PAIRS:
    pair = f.replace('_15M.csv', '')
    for year, mon in MONTHS:
        fromd = datetime.datetime(year, mon, 1)
        tod = (fromd + datetime.timedelta(days=32)).replace(day=1)

        cerebro = bt.Cerebro()
        # Load CSV via pandas to handle header and set timeframe/compression
        df = pd.read_csv(f, parse_dates=['datetime'], index_col='datetime')
        data = bt.feeds.PandasData(
            dataname=df,
            timeframe=bt.TimeFrame.Minutes,
            compression=15,
            fromdate=fromd,
            todate=tod
        )
        cerebro.adddata(data, name=pair)

        cerebro.addstrategy(MasterTrendML,
                            ml_enabled=True,
                            filter_signals=False,
                            ml_confidence_threshold=0.7)
        cerebro.broker.setcash(10000)
        cerebro.broker.setcommission(commission=0.0001)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')

        try:
            strat = cerebro.run()[0]
            ta = strat.analyzers.ta.get_analysis()
            trades = ta.total.closed if hasattr(ta, 'total') else 0
            won    = ta.won.total   if hasattr(ta, 'won')   else 0
            lost   = ta.lost.total  if hasattr(ta, 'lost')  else 0
            end_val = cerebro.broker.getvalue()
            results.append({
                'pair': pair,
                'year': year,
                'month': mon,
                'trades': trades,
                'won': won,
                'lost': lost,
                'end_val': end_val
            })
        except Exception as e:
            print(f"Error backtesting {pair} {year}-{mon}: {e}")
            continue

# Save summary
df = pd.DataFrame(results)
df.to_csv('batch_results.csv', index=False)
print(df.groupby('pair')[['end_val', 'trades', 'won', 'lost']].sum()) 