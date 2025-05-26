#!/usr/bin/env python3
"""
MASTERTREND SIMPLE VISUAL STRATEGY
Version simplifiée avec visualisation corrigée
Objectif: Analyser visuellement les signaux sans erreurs
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import deque


class SimpleMasterTrendVisual(bt.Strategy):
    """
    Stratégie MasterTrend simplifiée avec visualisation
    """
    
    params = (
        # SuperTrend
        ('st_period', 10),
        ('st_multiplier', 2.5),
        
        # MACD
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        
        # RSI
        ('rsi_period', 14),
        
        # Risk Management
        ('position_size', 0.02),
        ('max_consecutive_losses', 3),
        
        # Visualisation
        ('save_plots', True),
    )
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # === INDICATEURS DE BASE ===
        
        # Prix
        self.hl2 = (self.data.high + self.data.low) / 2.0
        
        # ATR pour SuperTrend
        self.atr = bt.indicators.ATR(self.data, period=self.p.st_period)
        
        # Moyennes mobiles
        self.sma_20 = bt.indicators.SMA(self.data.close, period=20)
        self.sma_50 = bt.indicators.SMA(self.data.close, period=50)
        self.ema_12 = bt.indicators.EMA(self.data.close, period=12)
        self.ema_26 = bt.indicators.EMA(self.data.close, period=26)
        
        # MACD simplifié
        self.macd_line = self.ema_12 - self.ema_26
        self.macd_signal = bt.indicators.EMA(self.macd_line, period=self.p.macd_signal)
        self.macd_histogram = self.macd_line - self.macd_signal
        
        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        
        # Volume
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)
        
        # === VARIABLES D'ÉTAT ===
        
        # SuperTrend
        self.trend = 1
        self.supertrend_up = 0.0
        self.supertrend_dn = 0.0
        
        # Trading
        self.initial_cash = self.broker.getvalue()
        self.consecutive_losses = 0
        self.order = None
        
        # Données pour graphiques
        self.trade_signals = []
        self.equity_data = []
        self.datetime_data = []
        self.price_data = []
        self.indicator_data = {
            'sma_20': [],
            'sma_50': [],
            'rsi': [],
            'macd': [],
            'macd_signal': [],
            'macd_hist': [],
            'volume': [],
            'supertrend': []
        }
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🟢 BUY EXECUTED @ {order.executed.price:.5f}')
            else:
                self.log(f'🔴 SELL EXECUTED @ {order.executed.price:.5f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('❌ Order Canceled/Margin/Rejected')
            
        self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            if trade.pnl > 0:
                self.consecutive_losses = 0
                self.log(f'💰 PROFIT: ${trade.pnl:.2f}')
            else:
                self.consecutive_losses += 1
                self.log(f'💸 LOSS: ${trade.pnl:.2f} [Consecutive: {self.consecutive_losses}]')
    
    def update_supertrend(self):
        """Mise à jour SuperTrend"""
        if len(self.data) < 2:
            return
        
        try:
            basic_up = self.hl2[0] - self.p.st_multiplier * self.atr[0]
            basic_dn = self.hl2[0] + self.p.st_multiplier * self.atr[0]
            
            if len(self.data) == 2:  # Premier calcul
                self.supertrend_up = basic_up
                self.supertrend_dn = basic_dn
                self.trend = 1
            else:
                # Mise à jour SuperTrend
                if self.data.close[-1] > self.supertrend_up:
                    self.supertrend_up = max(basic_up, self.supertrend_up)
                else:
                    self.supertrend_up = basic_up
                    
                if self.data.close[-1] < self.supertrend_dn:
                    self.supertrend_dn = min(basic_dn, self.supertrend_dn)
                else:
                    self.supertrend_dn = basic_dn
                    
                # Détermination de la tendance
                if self.trend == -1 and self.data.close[0] > self.supertrend_dn:
                    self.trend = 1
                elif self.trend == 1 and self.data.close[0] < self.supertrend_up:
                    self.trend = -1
        except Exception as e:
            self.log(f"Erreur SuperTrend: {e}")
    
    def collect_data(self):
        """Collecter les données pour les graphiques"""
        try:
            # Données de base
            self.datetime_data.append(self.datas[0].datetime.datetime(0))
            self.price_data.append(self.data.close[0])
            self.equity_data.append(self.broker.getvalue())
            
            # Indicateurs
            self.indicator_data['sma_20'].append(self.sma_20[0])
            self.indicator_data['sma_50'].append(self.sma_50[0])
            self.indicator_data['rsi'].append(self.rsi[0])
            self.indicator_data['macd'].append(self.macd_line[0])
            self.indicator_data['macd_signal'].append(self.macd_signal[0])
            self.indicator_data['macd_hist'].append(self.macd_histogram[0])
            self.indicator_data['volume'].append(self.data.volume[0])
            
            # SuperTrend
            if self.trend == 1:
                self.indicator_data['supertrend'].append(self.supertrend_up)
            else:
                self.indicator_data['supertrend'].append(self.supertrend_dn)
                
        except Exception as e:
            self.log(f"Erreur collecte données: {e}")
    
    def next(self):
        try:
            # Collecter les données
            self.collect_data()
            
            # Mise à jour SuperTrend
            self.update_supertrend()
            
            # Vérifier les règles de trading
            if self.consecutive_losses >= self.p.max_consecutive_losses:
                return
            
            # === SIGNAUX DE TRADING ===
            
            # MACD Crossover
            macd_bullish = (self.macd_line[0] > self.macd_signal[0] and 
                           self.macd_line[-1] <= self.macd_signal[-1])
            macd_bearish = (self.macd_line[0] < self.macd_signal[0] and 
                           self.macd_line[-1] >= self.macd_signal[-1])
            
            # Conditions de base
            long_condition = (self.trend == 1 and macd_bullish and 
                             self.rsi[0] < 70 and 
                             self.data.close[0] > self.sma_20[0])
            
            short_condition = (self.trend == -1 and macd_bearish and 
                              self.rsi[0] > 30 and 
                              self.data.close[0] < self.sma_20[0])
            
            # === EXÉCUTION DES TRADES ===
            
            if self.order:
                return
            
            if not self.position:
                if long_condition:
                    size = int(self.broker.getvalue() * self.p.position_size / self.data.close[0])
                    self.log(f'🚀 BUY SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.buy(size=max(1, size))
                    self.trade_signals.append({
                        'datetime': self.datas[0].datetime.datetime(0),
                        'price': self.data.close[0],
                        'type': 'BUY',
                        'trend': self.trend
                    })
                    
                elif short_condition:
                    size = int(self.broker.getvalue() * self.p.position_size / self.data.close[0])
                    self.log(f'🚀 SELL SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.sell(size=max(1, size))
                    self.trade_signals.append({
                        'datetime': self.datas[0].datetime.datetime(0),
                        'price': self.data.close[0],
                        'type': 'SELL',
                        'trend': self.trend
                    })
            
            else:  # Position ouverte
                # Sorties simples
                if self.position.size > 0:  # Position longue
                    if (self.rsi[0] > 75 or 
                        self.trend == -1 or 
                        macd_bearish):
                        self.log(f'💰 LONG EXIT @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                        
                elif self.position.size < 0:  # Position courte
                    if (self.rsi[0] < 25 or 
                        self.trend == 1 or 
                        macd_bullish):
                        self.log(f'💰 SHORT EXIT @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                        
        except Exception as e:
            self.log(f"Erreur dans next(): {e}")
    
    def create_plots(self, config_name):
        """Créer les graphiques simplifiés"""
        try:
            if not self.datetime_data or not self.price_data:
                self.log("Pas de données pour créer les graphiques")
                return
            
            # Configuration
            plt.style.use('dark_background')
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            fig.suptitle(f'MasterTrend Simple Visual - {config_name}', fontsize=16, fontweight='bold')
            
            # 1. Prix et moyennes mobiles
            ax1 = axes[0]
            ax1.plot(self.datetime_data, self.price_data, color='#00ff88', linewidth=2, label='Prix')
            
            if len(self.indicator_data['sma_20']) == len(self.datetime_data):
                ax1.plot(self.datetime_data, self.indicator_data['sma_20'], 
                        color='#ff6b6b', linewidth=1.5, label='SMA 20', alpha=0.7)
                ax1.plot(self.datetime_data, self.indicator_data['sma_50'], 
                        color='#4ecdc4', linewidth=1.5, label='SMA 50', alpha=0.7)
                ax1.plot(self.datetime_data, self.indicator_data['supertrend'], 
                        color='#ffaa00', linewidth=2, label='SuperTrend', alpha=0.8)
            
            # Signaux de trading
            for signal in self.trade_signals:
                color = '#00ff00' if signal['type'] == 'BUY' else '#ff0000'
                marker = '^' if signal['type'] == 'BUY' else 'v'
                ax1.scatter(signal['datetime'], signal['price'], 
                           color=color, s=100, marker=marker, zorder=5)
            
            ax1.set_title('Prix et Signaux', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Equity Curve
            ax2 = axes[1]
            if self.equity_data:
                returns = [(eq - self.initial_cash) / self.initial_cash * 100 for eq in self.equity_data]
                ax2.plot(self.datetime_data[:len(returns)], returns, 
                        color='#00aaff', linewidth=2)
                ax2.axhline(y=0, color='white', linestyle='--', alpha=0.5)
                ax2.set_title('Courbe d\'Équité (%)', fontweight='bold')
                ax2.set_ylabel('Rendement (%)')
                ax2.grid(True, alpha=0.3)
            
            # 3. RSI
            ax3 = axes[2]
            if len(self.indicator_data['rsi']) == len(self.datetime_data):
                ax3.plot(self.datetime_data, self.indicator_data['rsi'], 
                        color='#ffaa00', linewidth=2)
                ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Surachat')
                ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Survente')
                ax3.axhline(y=50, color='white', linestyle='-', alpha=0.3)
                ax3.set_title('RSI (14)', fontweight='bold')
                ax3.set_ylim(0, 100)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 4. MACD
            ax4 = axes[3]
            if len(self.indicator_data['macd']) == len(self.datetime_data):
                ax4.plot(self.datetime_data, self.indicator_data['macd'], 
                        color='#aa00ff', linewidth=2, label='MACD')
                ax4.plot(self.datetime_data, self.indicator_data['macd_signal'], 
                        color='orange', linewidth=1.5, label='Signal')
                ax4.bar(self.datetime_data, self.indicator_data['macd_hist'], 
                       color='gray', alpha=0.6, label='Histogramme')
                ax4.axhline(y=0, color='white', linestyle='-', alpha=0.3)
                ax4.set_title('MACD', fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            # Format des dates
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Sauvegarde
            if self.p.save_plots:
                filename = f'mastertrend_simple_{config_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
                self.log(f'📊 Graphique sauvegardé: {filename}')
            
            plt.show()
            
        except Exception as e:
            self.log(f"Erreur création graphiques: {e}")
    
    def stop(self):
        """Statistiques finales"""
        try:
            final_value = self.broker.getvalue()
            total_return = (final_value - self.initial_cash) / self.initial_cash * 100
            
            self.log(f'=== RÉSULTATS MASTERTREND SIMPLE VISUAL ===')
            self.log(f'💰 Capital Initial: ${self.initial_cash:.2f}')
            self.log(f'💰 Capital Final: ${final_value:.2f}')
            self.log(f'📈 Rendement Total: {total_return:.2f}%')
            self.log(f'🎯 Signaux Générés: {len(self.trade_signals)}')
            
            # Analyse des signaux
            buy_signals = [s for s in self.trade_signals if s['type'] == 'BUY']
            sell_signals = [s for s in self.trade_signals if s['type'] == 'SELL']
            
            self.log(f'🟢 Signaux BUY: {len(buy_signals)}')
            self.log(f'🔴 Signaux SELL: {len(sell_signals)}')
            
            # Évaluation
            if total_return >= 5:
                self.log('\n🏆 EXCELLENT! Stratégie très performante!')
            elif total_return >= 2:
                self.log('\n✅ BON! Stratégie efficace')
            elif total_return >= 0:
                self.log('\n⚠️  MOYEN. Optimisation possible')
            else:
                self.log('\n❌ INSUFFISANT. Révision nécessaire')
                
        except Exception as e:
            self.log(f"Erreur dans stop(): {e}")


def check_data_file(filename):
    """Vérifier si le fichier de données existe"""
    return os.path.exists(filename)


if __name__ == '__main__':
    # Test avec visualisation simplifiée
    test_configs = [
        ("EURUSD_1M", "EURUSD_data_1M.csv", bt.TimeFrame.Minutes, 1),
    ]
    
    for config_name, filename, timeframe, compression in test_configs:
        print(f'\n{"="*60}')
        print(f'🚀 TESTING MASTERTREND SIMPLE VISUAL - {config_name}')
        print(f'📊 Mode: VISUALISATION SIMPLIFIÉE')
        print(f'🎯 Capital Initial: $10,000')
        print('='*60)
        
        if not check_data_file(filename):
            print(f'❌ FICHIER MANQUANT: {filename}')
            continue
        
        try:
            # Configuration
            cerebro = bt.Cerebro()
            
            # Données (limiter pour test rapide)
            data = bt.feeds.GenericCSVData(
                dataname=filename,
                dtformat=('%Y-%m-%d %H:%M:%S'),
                datetime=0, open=1, high=2, low=3, close=4, volume=5,
                timeframe=timeframe, compression=compression,
                openinterest=-1, headers=True, separator=',',
                fromdate=datetime.datetime(2025, 5, 22, 3, 0),  # Limiter les données
                todate=datetime.datetime(2025, 5, 22, 6, 0)
            )
            
            cerebro.adddata(data)
            cerebro.addstrategy(SimpleMasterTrendVisual)
            
            # Configuration
            cerebro.broker.setcash(10000.0)
            cerebro.broker.setcommission(commission=0.0001)
            
            # Analyseurs
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            results = cerebro.run()
            strat = results[0]
            
            # Créer les graphiques
            strat.create_plots(config_name)
            
            # Analyse des résultats
            trades = strat.analyzers.trades.get_analysis()
            returns = strat.analyzers.returns.get_analysis()
            
            total_trades = getattr(getattr(trades, 'total', None), 'closed', 0)
            total_return = getattr(returns, 'rtot', 0) * 100
            
            print(f'\n🔥 RÉSULTATS {config_name}:')
            print(f'📊 Total Trades: {total_trades}')
            print(f'📈 Rendement Total: {total_return:.2f}%')
            print(f'📊 Graphiques générés avec succès!')
                
        except Exception as e:
            print(f'❌ ERREUR sur {config_name}: {e}')
    
    print(f'\n{"="*80}')
    print('🎨 MASTERTREND SIMPLE VISUAL - Analyse graphique terminée!')
    print('📊 Vérifiez les fichiers PNG générés pour l\'analyse détaillée.')
    print('='*80) 