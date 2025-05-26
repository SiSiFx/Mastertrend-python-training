#!/usr/bin/env python3
"""
Test final simplifié inspiré de ISI_MAXI_WEEP
"""
import backtrader as bt
import pandas as pd
import datetime

class SimpleStrategy(bt.Strategy):
    params = (
        ('fast_period', 20),
        ('slow_period', 50),
        ('stop_loss_pct', 0.01),  # 1%
        ('take_profit_pct', 0.02),  # 2%
    )
    
    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # Variables pour le trading
        self.order = None
        self.buy_price = None
        self.sl_price = None
        self.tp_price = None
    
    def log(self, txt, dt=None):
        if len(self) % 5000 == 0:  # Log seulement toutes les 5000 barres
            dt = dt or self.data.datetime.datetime(0)
            print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.sl_price = self.buy_price * (1 - self.params.stop_loss_pct)
                self.tp_price = self.buy_price * (1 + self.params.take_profit_pct)
                self.log(f'BUY @{order.executed.price:.5f}, SL:{self.sl_price:.5f}, TP:{self.tp_price:.5f}')
            else:
                self.log(f'SELL @{order.executed.price:.5f}')
                
        self.order = None
    
    def next(self):
        # Log périodique
        self.log(f'Close: {self.data.close[0]:.5f}, Fast MA: {self.fast_ma[0]:.5f}, Slow MA: {self.slow_ma[0]:.5f}')
        
        if self.order:
            return
            
        if not self.position:  # Pas de position
            # Crossover pour entrer en position
            if self.crossover > 0:  # Fast MA croise au-dessus de Slow MA
                self.log(f'SIGNAL ACHAT @{self.data.close[0]:.5f}')
                self.order = self.buy()
        else:  # En position
            # Stop loss
            if self.data.low[0] <= self.sl_price:
                self.log(f'STOP LOSS HIT - Low: {self.data.low[0]:.5f} <= SL: {self.sl_price:.5f}')
                self.order = self.close()
            # Take profit
            elif self.data.high[0] >= self.tp_price:
                self.log(f'TAKE PROFIT HIT - High: {self.data.high[0]:.5f} >= TP: {self.tp_price:.5f}')
                self.order = self.close()
            # Exit signal
            elif self.crossover < 0:  # Fast MA croise en-dessous de Slow MA
                self.log(f'SIGNAL VENTE @{self.data.close[0]:.5f}')
                self.order = self.close()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    # Charger juste une partie des données pour le test
    data = bt.feeds.GenericCSVData(
        dataname="EURUSD_data_1M.csv",
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1,
        fromdate=datetime.datetime(2023, 4, 1),  # Limiter les données pour le test
        todate=datetime.datetime(2023, 4, 30)
    )
    
    cerebro.adddata(data)
    
    cerebro.addstrategy(SimpleStrategy)
    
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)  # 0.01%
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print('Capital initial: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    print('Capital final: %.2f' % cerebro.broker.getvalue())
    print('Retour: %.2f%%' % ((cerebro.broker.getvalue() / 10000.0 - 1.0) * 100))
    
    # Afficher les résultats d'analyse
    trade_analysis = strat.analyzers.trades.get_analysis()
    print('\nAnalyse des transactions:')
    print('Total des transactions:', trade_analysis.total.closed if hasattr(trade_analysis, 'total') else 0)
    if hasattr(trade_analysis, 'won'):
        print('Transactions gagnantes:', trade_analysis.won.total)
    if hasattr(trade_analysis, 'lost'):
        print('Transactions perdantes:', trade_analysis.lost.total) 