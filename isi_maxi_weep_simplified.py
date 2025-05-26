#!/usr/bin/env python3
"""
Version simplifiée et fonctionnelle de ISI_MAXI_WEEP pour EUR/USD 1-minute
"""
import backtrader as bt
import pandas as pd
import datetime

# Indicateur de Pivot High
class PivotHigh(bt.Indicator):
    lines = ('pivot',)
    params = (('length', 10),)
    
    def __init__(self):
        self.addminperiod(self.p.length * 2 + 1)
    
    def next(self):
        if len(self) <= 2 * self.p.length:
            self.lines.pivot[0] = float('nan')
            return
            
        mid_idx = self.p.length
        high_val = self.data.high[-mid_idx]
        is_pivot = True
        
        # Vérifier s'il s'agit d'un pivot haut
        for i in range(1, self.p.length + 1):
            if high_val <= self.data.high[-mid_idx - i] or high_val <= self.data.high[-mid_idx + i]:
                is_pivot = False
                break
                
        self.lines.pivot[0] = high_val if is_pivot else float('nan')

# Indicateur de Pivot Low
class PivotLow(bt.Indicator):
    lines = ('pivot',)
    params = (('length', 10),)
    
    def __init__(self):
        self.addminperiod(self.p.length * 2 + 1)
    
    def next(self):
        if len(self) <= 2 * self.p.length:
            self.lines.pivot[0] = float('nan')
            return
            
        mid_idx = self.p.length
        low_val = self.data.low[-mid_idx]
        is_pivot = True
        
        # Vérifier s'il s'agit d'un pivot bas
        for i in range(1, self.p.length + 1):
            if low_val >= self.data.low[-mid_idx - i] or low_val >= self.data.low[-mid_idx + i]:
                is_pivot = False
                break
                
        self.lines.pivot[0] = low_val if is_pivot else float('nan')

# Stratégie ISI MAXI WEEP simplifiée
class ISI_MAXI_WEEP_Simplified(bt.Strategy):
    params = (
        ('pivot_length', 10),
        ('cooldown_bars', 5),
        ('rr_ratio', 2.0),
        ('show_logs', False),  # Mettre à True pour plus de logs
    )
    
    def __init__(self):
        # Indicateurs
        self.pivot_high = PivotHigh(self.data, length=self.p.pivot_length)
        self.pivot_low = PivotLow(self.data, length=self.p.pivot_length)
        
        # Plus efficace de stocker les valeurs récentes des pivots
        self.recent_high = None
        self.recent_low = None
        
        # État du système
        self.last_sweep_bar = 0
        self.sweep_type = None  # 'high' ou 'low'
        
        # Niveaux de référence pour les trades
        self.resistance = None
        self.support = None
        self.sl_level = None
        self.tp_level = None
        
        # Pour les ordres
        self.order = None
    
    def log(self, txt, dt=None):
        if self.p.show_logs or len(self) % 5000 == 0:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY COMPLETED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}')
            else:
                self.log(f'SELL COMPLETED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
            
        self.log(f'TRADE COMPLETED, Profit: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
    
    def next(self):
        # Afficher périodiquement l'état
        if len(self) % 5000 == 0:
            self.log(f'Bar: {len(self)}, Close: {self.data.close[0]:.5f}, Position: {self.position.size}')
        
        # Ne rien faire si un ordre est en attente
        if self.order:
            return
        
        # Mettre à jour les pivots
        if not pd.isna(self.pivot_high[0]):
            self.recent_high = self.pivot_high[0]
            self.log(f'New Pivot High: {self.recent_high:.5f}')
            
        if not pd.isna(self.pivot_low[0]):
            self.recent_low = self.pivot_low[0]
            self.log(f'New Pivot Low: {self.recent_low:.5f}')
        
        # Si des niveaux sont définis et que le prix les franchit, c'est un sweep
        current_bar = len(self)
        if self.recent_high is not None and self.data.close[0] > self.recent_high and current_bar > self.last_sweep_bar + self.p.cooldown_bars:
            self.log(f'HIGH SWEEP DETECTED: {self.data.close[0]:.5f} > {self.recent_high:.5f}')
            self.sweep_type = 'high'
            self.last_sweep_bar = current_bar
            self.resistance = self.recent_high
            
        if self.recent_low is not None and self.data.close[0] < self.recent_low and current_bar > self.last_sweep_bar + self.p.cooldown_bars:
            self.log(f'LOW SWEEP DETECTED: {self.data.close[0]:.5f} < {self.recent_low:.5f}')
            self.sweep_type = 'low'
            self.last_sweep_bar = current_bar
            self.support = self.recent_low
        
        # Logique de trading
        if not self.position:  # Pas de position ouverte
            # Achat après sweep du low (support)
            if self.sweep_type == 'low' and self.support is not None and self.data.close[0] > self.support:
                self.sl_level = self.support * 0.999  # Juste un peu en-dessous du support
                risk = self.data.close[0] - self.sl_level
                self.tp_level = self.data.close[0] + (risk * self.p.rr_ratio)
                
                self.log(f'BUY SIGNAL: Price: {self.data.close[0]:.5f}, Support: {self.support:.5f}, SL: {self.sl_level:.5f}, TP: {self.tp_level:.5f}')
                self.order = self.buy()
                
            # Vente après sweep du high (résistance)
            elif self.sweep_type == 'high' and self.resistance is not None and self.data.close[0] < self.resistance:
                self.sl_level = self.resistance * 1.001  # Juste un peu au-dessus de la résistance
                risk = self.sl_level - self.data.close[0]
                self.tp_level = self.data.close[0] - (risk * self.p.rr_ratio)
                
                self.log(f'SELL SIGNAL: Price: {self.data.close[0]:.5f}, Resistance: {self.resistance:.5f}, SL: {self.sl_level:.5f}, TP: {self.tp_level:.5f}')
                self.order = self.sell()
                
        else:  # Position ouverte
            if self.position.size > 0:  # Position longue
                # Stop loss
                if self.sl_level is not None and self.data.low[0] <= self.sl_level:
                    self.log(f'LONG STOP LOSS: Low: {self.data.low[0]:.5f} <= SL: {self.sl_level:.5f}')
                    self.order = self.close()
                    
                # Take profit
                elif self.tp_level is not None and self.data.high[0] >= self.tp_level:
                    self.log(f'LONG TAKE PROFIT: High: {self.data.high[0]:.5f} >= TP: {self.tp_level:.5f}')
                    self.order = self.close()
                    
            else:  # Position courte
                # Stop loss
                if self.sl_level is not None and self.data.high[0] >= self.sl_level:
                    self.log(f'SHORT STOP LOSS: High: {self.data.high[0]:.5f} >= SL: {self.sl_level:.5f}')
                    self.order = self.close()
                    
                # Take profit
                elif self.tp_level is not None and self.data.low[0] <= self.tp_level:
                    self.log(f'SHORT TAKE PROFIT: Low: {self.data.low[0]:.5f} <= TP: {self.tp_level:.5f}')
                    self.order = self.close()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    # Charger les données - limiter à 10000 lignes pour éviter les problèmes de performance
    import pandas as pd
    full_data = pd.read_csv("EURUSD_data_1M.csv")
    limited_data = full_data.iloc[:10000]  # Prendre seulement les 10000 premières lignes
    limited_data.to_csv("EURUSD_limited.csv", index=False)
    
    data = bt.feeds.GenericCSVData(
        dataname="EURUSD_limited.csv",
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1
    )
    
    cerebro.adddata(data)
    
    # Paramètres de la stratégie
    params = {
        'pivot_length': 10,
        'cooldown_bars': 5,
        'rr_ratio': 2.0,
        'show_logs': False,
    }
    
    cerebro.addstrategy(ISI_MAXI_WEEP_Simplified, **params)
    
    # Paramètres du broker
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)  # 0.01%
    
    # Analyseurs
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Exécuter le backtest
    print('Capital initial: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    print('Capital final: %.2f' % cerebro.broker.getvalue())
    print('Retour: %.2f%%' % ((cerebro.broker.getvalue() / 10000.0 - 1.0) * 100))
    
    # Résultats des analyseurs
    try:
        print('Sharpe Ratio:', strat.analyzers.sharpe.get_analysis()['sharperatio'])
    except:
        print('Sharpe Ratio: N/A')
        
    try:
        print('Drawdown Max:', strat.analyzers.drawdown.get_analysis()['max']['drawdown'])
    except:
        print('Drawdown Max: N/A')
    
    trade_analysis = strat.analyzers.trades.get_analysis()
    
    # Statistiques des transactions
    print('\nAnalyse des transactions:')
    print('Total des transactions:', trade_analysis.total.closed if hasattr(trade_analysis, 'total') else 0)
    print('Transactions gagnantes:', trade_analysis.won.total if hasattr(trade_analysis, 'won') else 0)
    print('Transactions perdantes:', trade_analysis.lost.total if hasattr(trade_analysis, 'lost') else 0)
    
    if hasattr(trade_analysis, 'won') and trade_analysis.won.total > 0:
        print('Gain moyen des transactions gagnantes:', trade_analysis.won.pnl.average)
    if hasattr(trade_analysis, 'lost') and trade_analysis.lost.total > 0:
        print('Perte moyenne des transactions perdantes:', trade_analysis.lost.pnl.average) 