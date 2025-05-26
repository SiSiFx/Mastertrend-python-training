#!/usr/bin/env python3
"""
Test minimaliste avec les données EURUSD en 1-minute
"""
import backtrader as bt
import datetime

class MinimalStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=20)
        
    def next(self):
        if len(self) % 1000 == 0:  # Log seulement tous les 1000 barres
            print(f'Bar {len(self)}: Close={self.data.close[0]:.5f}, SMA={self.sma[0]:.5f}')
            
        if not self.position:
            if self.data.close[0] > self.sma[0]:
                self.buy()
        else:
            if self.data.close[0] < self.sma[0]:
                self.sell()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    cerebro.addstrategy(MinimalStrategy)
    
    # Charger les données
    data = bt.feeds.GenericCSVData(
        dataname="EURUSD_data_1M.csv",
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1
    )
    
    cerebro.adddata(data)
    
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)  # 0.01%
    
    print('Capital initial: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Capital final: %.2f' % cerebro.broker.getvalue()) 