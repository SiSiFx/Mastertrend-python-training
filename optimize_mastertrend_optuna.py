#!/usr/bin/env python3
"""
Smart optimization of MasterTrendStrategy parameters using Optuna,
targeting prop-firm constraints (profit target & max daily drawdown).
"""
import argparse
import os

import backtrader as bt
import pandas as pd
import optuna

from mastertrend_strategy import MasterTrendStrategy

# Analyzer to track maximum intraday drawdown per day
class DailyMaxDrawDown(bt.Analyzer):
    def start(self):
        self.daily_high = None
        self.max_daily_dd = 0.0
        self.current_date = None
    def next(self):
        dt = self.strategy.data.datetime.date(0)
        value = self.strategy.broker.getvalue()
        if self.current_date != dt:
            self.current_date = dt
            self.daily_high = value
        if self.daily_high:
            dd = (self.daily_high - value) / self.daily_high
            self.max_daily_dd = max(self.max_daily_dd, dd)
    def stop(self):
        return {'max_daily_dd': self.max_daily_dd}

# Run a single backtest with given params and return (return, max_daily_dd)
def run_backtest(params, data_file, cash, commission):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    # Data feed
    data = bt.feeds.GenericCSVData(
        dataname=data_file,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes, compression=1, openinterest=-1
    )
    cerebro.adddata(data)
    # Strategy
    cerebro.addstrategy(
        MasterTrendStrategy,
        macd_fast=params['macd_fast'],
        macd_slow=params['macd_slow'],
        supertrend_period=params['st_period'],
        supertrend_multiplier=params['st_mult']
    )
    # Analyzers
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(DailyMaxDrawDown, _name='dailydd')
    # Run
    results = cerebro.run()
    strat = results[0]
    ret = strat.analyzers.returns.get_analysis().get('rtot', 0.0)
    dd = strat.analyzers.dailydd.get_analysis().get('max_daily_dd', 0.0)
    return ret, dd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optuna optimization for MasterTrendStrategy')
    parser.add_argument('--data', default='EURUSD_data_1M.csv', help='CSV file with OHLCV data')
    parser.add_argument('--cash', type=float, default=10000.0, help='Starting capital')
    parser.add_argument('--commission', type=float, default=0.0001, help='Commission fraction')
    parser.add_argument('--profit-target', type=float, default=0.10, help='Minimum return fraction (e.g. 0.10 for 10%)')
    parser.add_argument('--max-daily-dd', type=float, default=0.03, help='Max allowed daily drawdown (e.g. 0.03 for 3%)')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Data file {args.data} not found.")
        exit(1)

    # Define objective using closure over args
    def objective(trial):
        # Suggest hyperparameters
        macd_fast = trial.suggest_int('macd_fast', 8, 20)
        macd_slow = trial.suggest_int('macd_slow', 21, 40)
        st_period = trial.suggest_int('st_period', 5, 20)
        st_mult = trial.suggest_float('st_mult', 2.0, 5.0)
        # Ensure valid MACD ordering
        if macd_slow <= macd_fast:
            return -1e3
        params = {'macd_fast': macd_fast, 'macd_slow': macd_slow,
                  'st_period': st_period, 'st_mult': st_mult}
        ret, dd = run_backtest(params, args.data, args.cash, args.commission)
        # Calculate penalty for constraint violations
        penalty = 0.0
        if ret < args.profit_target:
            penalty += (args.profit_target - ret)
        if dd > args.max_daily_dd:
            penalty += (dd - args.max_daily_dd)
        return ret - penalty

    # Create and run study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.trials)

    # Report best trial
    best = study.best_trial
    print("Best trial results:")
    print(f"  Return          : {best.value*100:.2f}%")
    print(f"  Parameters      : {best.params}")
    # Re-run to get exact metrics
    ret, dd = run_backtest(best.params, args.data, args.cash, args.commission)
    print(f"  Max Daily DD    : {dd*100:.2f}%")
    print(f"  Profit Target   : {args.profit_target*100:.1f}%")
    print(f"  Trials Completed: {len(study.trials)}") 