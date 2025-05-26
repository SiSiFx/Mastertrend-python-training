#!/usr/bin/env python3
"""
Version optimisée et complète de ISI_MAXI_WEEP pour EUR/USD 1-minute
Avec gestion efficace des grandes quantités de données et visualisation
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib
matplotlib.use('Agg')  # Mode sans interface graphique, utile pour les grandes simulations

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

# Observer pour tracer les niveaux de prix significatifs
class PriceLevel(bt.Observer):
    lines = ('level',)
    params = (('price', None),)
    
    def next(self):
        self.lines.level[0] = self.p.price

# Stratégie optimisée ISI MAXI WEEP
class ISI_MAXI_WEEP_Optimized(bt.Strategy):
    params = (
        ('pivot_length', 10),        # Longueur pour les pivots
        ('cooldown_bars', 5),        # Période de blocage après un trade
        ('rr_ratio', 2.0),           # Ratio risque/récompense
        ('show_logs', False),        # Activer logs détaillés
        ('log_interval', 10000),     # Intervalle pour afficher les logs de progression
        ('sl_buffer', 0.0002),       # Marge supplémentaire pour stop loss
        ('market_open_hour', 8),     # Heure d'ouverture du marché (UTC)
        ('market_close_hour', 20),   # Heure de fermeture du marché (UTC)
        ('use_time_filter', False),  # Filtrer par horaires de marché
        ('commission', 0.0001),      # Commission par défaut
    )
    
    def __init__(self):
        # Indicateurs
        self.pivot_high = PivotHigh(self.data, length=self.p.pivot_length)
        self.pivot_low = PivotLow(self.data, length=self.p.pivot_length)
        
        # Variables de la stratégie
        self.recent_high = None
        self.recent_low = None
        self.last_sweep_bar = 0
        self.sweep_type = None  # 'high' ou 'low'
        
        # Niveaux de référence pour les trades
        self.resistance = None
        self.support = None
        self.sl_level = None
        self.tp_level = None
        
        # Stats pour l'optimisation de performance
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        
        # Pour les ordres
        self.order = None
        
        # Observateurs pour afficher les niveaux
        self.resistance_level = None
        self.support_level = None
        self.sl_obs = None
        self.tp_obs = None
    
    def log(self, txt, dt=None, force=False):
        # Ne logger que si show_logs est True ou force est True
        if self.p.show_logs or force or len(self) % self.p.log_interval == 0:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt.isoformat()} {txt}')
    
    def is_market_open(self):
        # Si le filtre horaire n'est pas utilisé, toujours considérer le marché comme ouvert
        if not self.p.use_time_filter:
            return True
            
        dt = self.datas[0].datetime.datetime(0)
        hour = dt.hour
        # Forex EUR/USD est généralement plus actif pendant les heures européennes et américaines
        return self.p.market_open_hour <= hour < self.p.market_close_hour
    
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
            
        self.trade_count += 1
        if trade.pnl > 0:
            self.win_count += 1
            self.log(f'WINNING TRADE, Profit: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
        else:
            self.loss_count += 1
            self.log(f'LOSING TRADE, Loss: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
        
        # Supprimer les observateurs des niveaux précédents
        if self.resistance_level is not None:
            self.resistance_level = None
        if self.support_level is not None:
            self.support_level = None
        if self.sl_obs is not None:
            self.sl_obs = None
        if self.tp_obs is not None:
            self.tp_obs = None
    
    def next(self):
        # Afficher périodiquement l'état
        if len(self) % self.p.log_interval == 0:
            self.log(f'Bar: {len(self)}, Close: {self.data.close[0]:.5f}, Position: {self.position.size}', force=True)
            if self.trade_count > 0:
                winrate = (self.win_count / self.trade_count) * 100
                self.log(f'Performance: {self.trade_count} trades, Winrate: {winrate:.2f}%', force=True)
        
        # Ne rien faire si le marché est fermé selon le filtre horaire
        if not self.is_market_open():
            return
        
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
            
            # Ajouter un observateur pour le niveau de résistance
            if self.resistance_level is None:
                self.resistance_level = PriceLevel(price=self.resistance)
            
        if self.recent_low is not None and self.data.close[0] < self.recent_low and current_bar > self.last_sweep_bar + self.p.cooldown_bars:
            self.log(f'LOW SWEEP DETECTED: {self.data.close[0]:.5f} < {self.recent_low:.5f}')
            self.sweep_type = 'low'
            self.last_sweep_bar = current_bar
            self.support = self.recent_low
            
            # Ajouter un observateur pour le niveau de support
            if self.support_level is None:
                self.support_level = PriceLevel(price=self.support)
        
        # Logique de trading
        if not self.position:  # Pas de position ouverte
            # Achat après sweep du low (support)
            if self.sweep_type == 'low' and self.support is not None and self.data.close[0] > self.support:
                self.sl_level = self.support - self.p.sl_buffer  # Un peu en-dessous du support
                risk = self.data.close[0] - self.sl_level
                self.tp_level = self.data.close[0] + (risk * self.p.rr_ratio)
                
                self.log(f'BUY SIGNAL: Price: {self.data.close[0]:.5f}, Support: {self.support:.5f}, SL: {self.sl_level:.5f}, TP: {self.tp_level:.5f}')
                self.order = self.buy()
                
                # Ajouter des observateurs pour SL et TP
                self.sl_obs = PriceLevel(price=self.sl_level)
                self.tp_obs = PriceLevel(price=self.tp_level)
                
            # Vente après sweep du high (résistance)
            elif self.sweep_type == 'high' and self.resistance is not None and self.data.close[0] < self.resistance:
                self.sl_level = self.resistance + self.p.sl_buffer  # Un peu au-dessus de la résistance
                risk = self.sl_level - self.data.close[0]
                self.tp_level = self.data.close[0] - (risk * self.p.rr_ratio)
                
                self.log(f'SELL SIGNAL: Price: {self.data.close[0]:.5f}, Resistance: {self.resistance:.5f}, SL: {self.sl_level:.5f}, TP: {self.tp_level:.5f}')
                self.order = self.sell()
                
                # Ajouter des observateurs pour SL et TP
                self.sl_obs = PriceLevel(price=self.sl_level)
                self.tp_obs = PriceLevel(price=self.tp_level)
                
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
    
    def stop(self):
        # Logs finaux lorsque le backtest est terminé
        self.log('Strategy complete', force=True)
        if self.trade_count > 0:
            winrate = (self.win_count / self.trade_count) * 100
            self.log(f'Final Performance: {self.trade_count} trades, Wins: {self.win_count}, Losses: {self.loss_count}, Winrate: {winrate:.2f}%', force=True)

def run_backtest(csv_file, params=None, plot=False, data_limit=None):
    """
    Fonction pour exécuter un backtest de façon modulaire
    
    Args:
        csv_file (str): Chemin vers le fichier CSV contenant les données
        params (dict, optional): Paramètres de la stratégie
        plot (bool, optional): Si True, génère un graphique du backtest
        data_limit (int, optional): Limite le nombre de lignes à charger
        
    Returns:
        dict: Résultats du backtest
    """
    cerebro = bt.Cerebro()
    
    # Charger les données
    if data_limit:
        print(f"Chargement limité à {data_limit} barres depuis {csv_file}")
        full_data = pd.read_csv(csv_file)
        limited_data = full_data.iloc[:data_limit]
        temp_file = f"temp_{os.path.basename(csv_file)}"
        limited_data.to_csv(temp_file, index=False)
        csv_to_use = temp_file
    else:
        print(f"Chargement complet de {csv_file}")
        csv_to_use = csv_file
    
    data = bt.feeds.GenericCSVData(
        dataname=csv_to_use,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1
    )
    
    cerebro.adddata(data)
    
    # Paramètres de la stratégie par défaut
    default_params = {
        'pivot_length': 10,
        'cooldown_bars': 5,
        'rr_ratio': 2.0,
        'show_logs': False,
        'log_interval': 20000,
        'sl_buffer': 0.0002,
    }
    
    # Fusionner avec les paramètres fournis
    if params:
        default_params.update(params)
    
    cerebro.addstrategy(ISI_MAXI_WEEP_Optimized, **default_params)
    
    # Paramètres du broker
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=default_params.get('commission', 0.0001))  # 0.01%
    
    # Analyseurs
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    
    # Si plot est True, configurer la visualisation
    if plot:
        cerebro.addobserver(bt.observers.Broker)
        cerebro.addobserver(bt.observers.Trades)
        cerebro.addobserver(bt.observers.BuySell)
        cerebro.addobserver(bt.observers.DrawDown)
    
    # Exécuter le backtest
    start_time = datetime.datetime.now()
    print(f'Démarrage du backtest à {start_time}')
    print('Capital initial: %.2f' % cerebro.broker.getvalue())
    
    results = cerebro.run()
    strat = results[0]
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    final_value = cerebro.broker.getvalue()
    return_pct = ((final_value / 10000.0) - 1.0) * 100
    
    print(f'Fin du backtest à {end_time} (durée: {duration})')
    print(f'Capital final: {final_value:.2f}')
    print(f'Retour: {return_pct:.2f}%')
    
    # Résultats des analyseurs
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    if sharpe:
        print(f'Sharpe Ratio: {sharpe:.4f}')
    else:
        print('Sharpe Ratio: N/A')
        
    drawdown = strat.analyzers.drawdown.get_analysis()
    if drawdown:
        print(f'Drawdown Max: {drawdown["max"]["drawdown"]:.4f}')
        print(f'Drawdown Max Money: {drawdown["max"]["moneydown"]:.2f}')
    else:
        print('Drawdown Max: N/A')
    
    trade_analysis = strat.analyzers.trades.get_analysis()
    
    # Statistiques des transactions
    print('\nAnalyse des transactions:')
    total_trades = trade_analysis.total.closed if hasattr(trade_analysis, 'total') else 0
    win_trades = trade_analysis.won.total if hasattr(trade_analysis, 'won') else 0
    loss_trades = trade_analysis.lost.total if hasattr(trade_analysis, 'lost') else 0
    
    print(f'Total des transactions: {total_trades}')
    print(f'Transactions gagnantes: {win_trades}')
    print(f'Transactions perdantes: {loss_trades}')
    
    if win_trades > 0:
        win_pct = (win_trades / total_trades * 100) if total_trades > 0 else 0
        print(f'Pourcentage de réussite: {win_pct:.2f}%')
    
    if hasattr(trade_analysis, 'won') and win_trades > 0:
        print(f'Gain moyen des transactions gagnantes: {trade_analysis.won.pnl.average:.6f}')
    if hasattr(trade_analysis, 'lost') and loss_trades > 0:
        print(f'Perte moyenne des transactions perdantes: {trade_analysis.lost.pnl.average:.6f}')
    
    # Tracer le graphique si demandé
    if plot:
        print('Génération du graphique...')
        fig = cerebro.plot(style='candlestick', barup='green', bardown='red', 
                          volup='green', voldown='red', 
                          plotdist=0.1, width=16, height=9)
        
        # Sauvegarder le graphique
        try:
            from matplotlib import pyplot as plt
            plt.savefig('backtest_results.png')
            print('Graphique sauvegardé sous backtest_results.png')
        except Exception as e:
            print(f'Erreur lors de la sauvegarde du graphique: {e}')
    
    # Si un fichier temporaire a été créé, le supprimer
    if data_limit and os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass
    
    return {
        'final_value': final_value,
        'return_pct': return_pct,
        'sharpe': sharpe,
        'drawdown': drawdown,
        'total_trades': total_trades,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'win_pct': (win_trades / total_trades * 100) if total_trades > 0 else 0,
        'duration': duration
    }

if __name__ == '__main__':
    # Paramètres pour le backtest principal
    test_params = {
        'pivot_length': 10,
        'cooldown_bars': 5,
        'rr_ratio': 2.0,
        'show_logs': False,
        'log_interval': 10000,
        'sl_buffer': 0.0002,
    }
    
    # Exécuter le backtest avec un nombre limité de barres pour un test rapide
    results = run_backtest(
        csv_file="EURUSD_data_1M.csv",
        params=test_params,
        plot=True,
        data_limit=20000  # Utiliser les 20 000 premières barres
    )
    
    """
    # Pour exécuter un backtest sur toutes les données (décommenter pour utiliser)
    full_results = run_backtest(
        csv_file="EURUSD_data_1M.csv",
        params=test_params,
        plot=False,
        data_limit=None  # Utiliser toutes les données
    )
    """ 