#!/usr/bin/env python3
"""
Test avec des données très limitées
"""
import backtrader as bt
import datetime

class SimpleStrategy(bt.Strategy):
    params = (
        ('ma_period', 20),
    )
    
    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.ma_period)
        self.order = None
    
    def next(self):
        if self.order:
            return
            
        if not self.position:
            if self.data.close[0] > self.sma[0]:
                self.buy()
        else:
            if self.data.close[0] < self.sma[0]:
                self.sell()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    # Extraire seulement les 5000 premières lignes du fichier CSV
    import pandas as pd
    df = pd.read_csv("EURUSD_data_1M.csv", nrows=5000)
    df.to_csv("EURUSD_small.csv", index=False)
    
    # Charger les données réduites
    data = bt.feeds.GenericCSVData(
        dataname="EURUSD_small.csv",
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1
    )
    
    cerebro.adddata(data)
    cerebro.addstrategy(SimpleStrategy)
    
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)
    
    print('Capital initial: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Capital final: %.2f' % cerebro.broker.getvalue()) 