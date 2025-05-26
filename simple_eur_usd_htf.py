#!/usr/bin/env python3
"""
Test de backtest avec pivots et timeframes multiples, inspiré de ISI_MAXI_WEEP
"""
import backtrader as bt
import pandas as pd
import datetime

# Fonction pour convertir une chaîne de timeframe en valeurs backtrader
def parse_tf_string(tf_string):
    """Convertit une chaîne de timeframe (ex: '60', 'D', 'W') vers backtrader."""
    if tf_string.isdigit():  # Minutes
        return bt.TimeFrame.Minutes, int(tf_string)
    elif tf_string == 'D':
        return bt.TimeFrame.Days, 1
    elif tf_string == 'W':
        return bt.TimeFrame.Weeks, 1
    elif tf_string == 'M':
        return bt.TimeFrame.Months, 1
    raise ValueError(f"Timeframe non supporté: {tf_string}")

# Indicateur Pivot High simplifié
class PivotHigh(bt.Indicator):
    lines = ('ph',)
    params = (('length', 10),)
    
    def __init__(self):
        self.addminperiod(2 * self.p.length + 1)
    
    def next(self):
        # Simplification: vérifier si le prix milieu (il y a "length" barres) est plus haut que tout le reste
        mid_idx = self.p.length
        if len(self.data.high) <= 2 * self.p.length:
            self.lines.ph[0] = float('nan')
            return
            
        pivot_candidate = self.data.high[-mid_idx]
        is_pivot = True
        
        # Vérifier à gauche
        for i in range(1, self.p.length + 1):
            if pivot_candidate <= self.data.high[-mid_idx - i]:
                is_pivot = False
                break
                
        # Vérifier à droite
        if is_pivot:
            for i in range(1, self.p.length + 1):
                if pivot_candidate <= self.data.high[-mid_idx + i]:
                    is_pivot = False
                    break
                    
        self.lines.ph[0] = pivot_candidate if is_pivot else float('nan')

# Indicateur Pivot Low simplifié
class PivotLow(bt.Indicator):
    lines = ('pl',)
    params = (('length', 10),)
    
    def __init__(self):
        self.addminperiod(2 * self.p.length + 1)
    
    def next(self):
        mid_idx = self.p.length
        if len(self.data.low) <= 2 * self.p.length:
            self.lines.pl[0] = float('nan')
            return
            
        pivot_candidate = self.data.low[-mid_idx]
        is_pivot = True
        
        # Vérifier à gauche
        for i in range(1, self.p.length + 1):
            if pivot_candidate >= self.data.low[-mid_idx - i]:
                is_pivot = False
                break
                
        # Vérifier à droite
        if is_pivot:
            for i in range(1, self.p.length + 1):
                if pivot_candidate >= self.data.low[-mid_idx + i]:
                    is_pivot = False
                    break
                    
        self.lines.pl[0] = pivot_candidate if is_pivot else float('nan')

# Stratégie simplifiée inspirée par ISI_MAXI_WEEP
class PivotBreakoutStrategy(bt.Strategy):
    params = (
        ('pivot_length', 10),
        ('use_htf_pivots', True),
        ('htf_timeframe_str', '60'),  # 60 minutes
        ('rr_ratio', 2.0),
    )
    
    def __init__(self):
        # Pivots sur les données principales (1M)
        self.pivot_high = PivotHigh(self.data, length=self.p.pivot_length)
        self.pivot_low = PivotLow(self.data, length=self.p.pivot_length)
        
        # Pivots sur les données HTF si disponibles
        self.htf_pivot_high = None
        self.htf_pivot_low = None
        
        if self.p.use_htf_pivots and len(self.datas) > 1:
            self.htf_data = self.datas[1]  # Timeframe plus élevé (ex: 60 minutes)
            self.htf_pivot_high = PivotHigh(self.htf_data, length=self.p.pivot_length)
            self.htf_pivot_low = PivotLow(self.htf_data, length=self.p.pivot_length)
        
        # Variables d'état
        self.resistance_level = None
        self.support_level = None
        self.order = None
        self.sl_level = None
        self.tp_level = None
        
    def log(self, txt, dt=None):
        """Journaliser les messages - seulement à des intervalles réguliers"""
        if len(self) % 1000 == 0:  # Afficher seulement tous les 1000 barres
            dt = dt or self.datetime.datetime(0)
            print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'ACHAT EXÉCUTÉ, Prix: {order.executed.price:.5f}')
            else:
                self.log(f'VENTE EXÉCUTÉE, Prix: {order.executed.price:.5f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Ordre annulé ou rejeté')
            
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'PROFIT: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    
    def next(self):
        # Log périodique
        if len(self) % 1000 == 0:
            self.log(f'Close: {self.data.close[0]:.5f}, Position: {self.position.size}')
        
        # Mise à jour des niveaux de support/résistance basés sur les pivots HTF
        if self.htf_pivot_high is not None and not pd.isna(self.htf_pivot_high[0]):
            self.resistance_level = self.htf_pivot_high[0]
            
        if self.htf_pivot_low is not None and not pd.isna(self.htf_pivot_low[0]):
            self.support_level = self.htf_pivot_low[0]
        
        # Si pas de données HTF ou option désactivée, utiliser les pivots sur la timeframe principale
        if self.resistance_level is None and not pd.isna(self.pivot_high[0]):
            self.resistance_level = self.pivot_high[0]
            
        if self.support_level is None and not pd.isna(self.pivot_low[0]):
            self.support_level = self.pivot_low[0]
        
        # Ne rien faire si un ordre est en attente ou si on n'a pas les deux niveaux
        if self.order or self.resistance_level is None or self.support_level is None:
            return
        
        # Logique de trading
        if not self.position:  # Pas de position
            # Signal d'achat : prix casse au-dessus de la résistance
            if self.data.close[0] > self.resistance_level:
                self.sl_level = self.support_level
                self.tp_level = self.data.close[0] + (self.data.close[0] - self.sl_level) * self.p.rr_ratio
                self.log(f'SIGNAL ACHAT - Résistance cassée: {self.resistance_level:.5f}, SL: {self.sl_level:.5f}, TP: {self.tp_level:.5f}')
                self.order = self.buy()
                
            # Signal de vente : prix casse en-dessous du support
            elif self.data.close[0] < self.support_level:
                self.sl_level = self.resistance_level
                self.tp_level = self.data.close[0] - (self.sl_level - self.data.close[0]) * self.p.rr_ratio
                self.log(f'SIGNAL VENTE - Support cassé: {self.support_level:.5f}, SL: {self.sl_level:.5f}, TP: {self.tp_level:.5f}')
                self.order = self.sell()
                
        else:  # Position ouverte
            if self.position.size > 0:  # Position longue
                # Stop loss
                if self.data.low[0] <= self.sl_level:
                    self.log(f'STOP LOSS LONG - Bas: {self.data.low[0]:.5f} <= SL: {self.sl_level:.5f}')
                    self.order = self.close()
                # Take profit
                elif self.data.high[0] >= self.tp_level:
                    self.log(f'TAKE PROFIT LONG - Haut: {self.data.high[0]:.5f} >= TP: {self.tp_level:.5f}')
                    self.order = self.close()
                    
            else:  # Position courte
                # Stop loss
                if self.data.high[0] >= self.sl_level:
                    self.log(f'STOP LOSS SHORT - Haut: {self.data.high[0]:.5f} >= SL: {self.sl_level:.5f}')
                    self.order = self.close()
                # Take profit
                elif self.data.low[0] <= self.tp_level:
                    self.log(f'TAKE PROFIT SHORT - Bas: {self.data.low[0]:.5f} <= TP: {self.tp_level:.5f}')
                    self.order = self.close()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    # Charger les données
    base_data = bt.feeds.GenericCSVData(
        dataname="EURUSD_data_1M.csv",
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1
    )
    
    # Ajouter les données de base (1M)
    cerebro.adddata(base_data)
    
    # Paramètres de la stratégie
    params = {
        'pivot_length': 10,
        'use_htf_pivots': True,
        'htf_timeframe_str': '60',  # 60 minutes
        'rr_ratio': 2.0,
    }
    
    # Resample pour HTF si nécessaire
    if params['use_htf_pivots']:
        htf_tf, htf_comp = parse_tf_string(params['htf_timeframe_str'])
        cerebro.resampledata(base_data, timeframe=htf_tf, compression=htf_comp)
    
    # Ajouter la stratégie
    cerebro.addstrategy(PivotBreakoutStrategy, **params)
    
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
    
    # Afficher les résultats
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