#!/usr/bin/env python3
"""
Version simplifiée de ISI_MAXI_WEEP pour test
"""
import backtrader as bt
import pandas as pd
import datetime

class SimpleISIStrategy(bt.Strategy):
    params = (
        ('fast_period', 20),
        ('slow_period', 50),
    )
    
    lines = ('bull_signal', 'bear_signal')
    
    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # Pour stocker les caractéristiques des transactions
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return  # Attendre la confirmation
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXÉCUTÉ, Prix: {order.executed.price:.5f}, Coût: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'SELL EXÉCUTÉ, Prix: {order.executed.price:.5f}, Coût: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}')
                
            self.bar_executed = len(self)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Ordre Annulé/Margin/Rejeté')
            
        self.order = None
        
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
            
        self.log(f'RÉSULTAT OPÉRATION, BRUT: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
        
    def next(self):
        # Journaliser les valeurs actuelles
        self.log(f'Close: {self.data.close[0]:.5f}, Fast MA: {self.fast_ma[0]:.5f}, Slow MA: {self.slow_ma[0]:.5f}')
        
        # Initialiser les lignes de signal
        self.lines.bull_signal[0] = 0
        self.lines.bear_signal[0] = 0
        
        # Vérifier si un ordre est en attente
        if self.order:
            return
            
        # Vérifier si nous sommes en position
        if not self.position:
            # Pas en position, vérifier si on doit acheter
            if self.crossover > 0:  # Fast MA croise au-dessus de Slow MA
                self.log(f'SIGNAL ACHAT, {self.data.close[0]:.5f}')
                self.lines.bull_signal[0] = 1
                self.order = self.buy()
                
        else:
            # Déjà en position, vérifier si on doit vendre
            if self.crossover < 0:  # Fast MA croise en-dessous de Slow MA
                self.log(f'SIGNAL VENTE, {self.data.close[0]:.5f}')
                self.lines.bear_signal[0] = 1
                self.order = self.sell()

if __name__ == '__main__':
    # Créer une instance de cerebro
    cerebro = bt.Cerebro()
    
    # Ajouter la stratégie
    cerebro.addstrategy(SimpleISIStrategy)
    
    # Charger les données depuis le fichier CSV
    datapath = "EURUSD_data_1M.csv"
    
    # Paramètres pour GenericCSVData
    data = bt.feeds.GenericCSVData(
        dataname=datapath,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1
    )
    
    # Ajouter les données à cerebro
    cerebro.adddata(data)
    
    # Définir le capital initial
    cerebro.broker.setcash(10000.0)
    
    # Définir la commission - 0.1% ... commissions habituelles dans le Forex
    cerebro.broker.setcommission(commission=0.001)
    
    # Ajouter des analyseurs
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Exécuter le backtest
    print('Capital initial: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    
    # Afficher les résultats
    print('Capital final: %.2f' % cerebro.broker.getvalue())
    print('Retour: %.2f%%' % ((cerebro.broker.getvalue() / 10000.0 - 1.0) * 100))
    
    try:
        print('Sharpe Ratio:', strat.analyzers.sharpe.get_analysis()['sharperatio'])
    except:
        print('Sharpe Ratio: N/A')
        
    try:
        print('Drawdown Max:', strat.analyzers.drawdown.get_analysis()['max']['drawdown'])
    except:
        print('Drawdown Max: N/A')
    
    trade_analysis = strat.analyzers.trades.get_analysis()
    
    # Afficher les statistiques des transactions
    print('\nAnalyse des transactions:')
    print('Total des transactions:', trade_analysis.total.closed if hasattr(trade_analysis, 'total') else 0)
    print('Transactions gagnantes:', trade_analysis.won.total if hasattr(trade_analysis, 'won') else 0)
    print('Transactions perdantes:', trade_analysis.lost.total if hasattr(trade_analysis, 'lost') else 0)
    
    if hasattr(trade_analysis, 'won') and trade_analysis.won.total > 0:
        print('Gain moyen des transactions gagnantes:', trade_analysis.won.pnl.average)
    if hasattr(trade_analysis, 'lost') and trade_analysis.lost.total > 0:
        print('Perte moyenne des transactions perdantes:', trade_analysis.lost.pnl.average) 