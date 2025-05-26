#!/usr/bin/env python3
import backtrader as bt
import pandas as pd
import argparse
import os
import itertools
from mastertrend_strategy import MasterTrendStrategy


# Custom analyzer to measure maximum intraday drawdown per day
class DailyMaxDrawDown(bt.Analyzer):
    def start(self):
        self.daily_high = None
        self.max_daily_dd = 0.0
        self.current_date = None
    def next(self):
        # Current bar's date
        dt = self.strategy.data.datetime.date(0)
        value = self.strategy.broker.getvalue()
        if self.current_date != dt:
            # New day: reset daily high
            self.current_date = dt
            self.daily_high = value
        # Compute drawdown from day's high
        dd = (self.daily_high - value) / self.daily_high if self.daily_high else 0.0
        if dd > self.max_daily_dd:
            self.max_daily_dd = dd
    def stop(self):
        return {'max_daily_dd': self.max_daily_dd}


def parse_list(s, typ=float):
    """Parse a comma-separated list into a list of type `typ`"""
    return [typ(item) for item in s.split(',') if item]


def main():
    parser = argparse.ArgumentParser(description='Optimize MasterTrendStrategy parameters by grid search')
    parser.add_argument('--data', default='EURUSD_data_1M.csv', help='CSV file with OHLCV data')
    parser.add_argument('--cash', type=float, default=10000.0, help='Starting capital')
    parser.add_argument('--commission', type=float, default=0.0001, help='Commission (fraction)')
    parser.add_argument('--top', type=int, default=5, help='Number of top results to display')
    parser.add_argument('--macd1', default='10,13,16', help='Comma-separated list of MACD1 periods')
    parser.add_argument('--macd2', default='26,30,34', help='Comma-separated list of MACD2 periods')
    parser.add_argument('--st-period', default='7,10,13', help='Comma-separated list of SuperTrend periods')
    parser.add_argument('--st-mult', default='2.5,3.0,3.5', help='Comma-separated list of SuperTrend multipliers')
    parser.add_argument('--profit-target', type=float, default=0.10, help='Minimum return fraction required (e.g. 0.10 for 10%)')
    parser.add_argument('--max-daily-dd', type=float, default=0.03, help='Maximum allowed daily drawdown fraction (e.g. 0.03 for 3%)')
    args = parser.parse_args()

    # Check data file
    if not os.path.exists(args.data):
        print(f"Data file {args.data} not found.")
        return

    # Prepare data feed
    data = bt.feeds.GenericCSVData(
        dataname=args.data,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes, compression=1, openinterest=-1
    )

    # Parse parameter ranges
    macd1_range = parse_list(args.macd1, int)
    macd2_range = parse_list(args.macd2, int)
    st_period_range = parse_list(args.st_period, int)
    st_mult_range = parse_list(args.st_mult, float)

    # Set up Cerebro for optimization
    cerebro = bt.Cerebro(optreturn=True)
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=args.commission)
    cerebro.adddata(data)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(DailyMaxDrawDown, _name='dailydd')

    # Add optimized strategy with correct parameter names
    cerebro.optstrategy(
        MasterTrendStrategy,
        macd_fast=macd1_range,
        macd_slow=macd2_range,
        supertrend_period=st_period_range,
        supertrend_multiplier=st_mult_range
    )

    # Run optimization
    print("Running parameter grid search...")
    results = cerebro.run()

    # Collect results
    records = []
    for run in results:
        strat = run[0]
        ret = strat.analyzers.returns.get_analysis().get('rtot', 0)
        dd = strat.analyzers.dailydd.get_analysis().get('max_daily_dd', 0)
        params = {
            'macd_fast': strat.params.macd_fast,
            'macd_slow': strat.params.macd_slow,
            'supertrend_period': strat.params.supertrend_period,
            'supertrend_multiplier': strat.params.supertrend_multiplier,
        }
        records.append((params, ret, dd))

    # Filter by prop-firm constraints
    filtered = [r for r in records if r[1] >= args.profit_target and r[2] <= args.max_daily_dd]
    if not filtered:
        print("No parameter sets met the prop-firm constraints.")
        return
    # Sort filtered by return descending
    filtered.sort(key=lambda x: x[1], reverse=True)

    # Display top N among those meeting constraints
    print(f"Top {args.top} parameter sets meeting constraints (profit >= {args.profit_target*100:.1f}%, daily DD <= {args.max_daily_dd*100:.1f}%):")
    for i, (params, ret, dd) in enumerate(filtered[:args.top], 1):
        print(f"{i}. Return: {ret*100:.2f}% | MaxDailyDD: {dd*100:.2f}% | Params: {params}")


if __name__ == '__main__':
    main() 