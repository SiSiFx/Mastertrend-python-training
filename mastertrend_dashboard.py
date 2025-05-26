#!/usr/bin/env python3
"""
MASTERTREND DASHBOARD
Dashboard interactif complet pour l'analyse de la stratégie MasterTrend
Objectif: Analyse visuelle avancée avec métriques détaillées
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class MasterTrendDashboard(bt.Strategy):
    """
    Stratégie MasterTrend avec dashboard complet
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
        ('stop_loss_pct', 0.01),
        ('take_profit_pct', 0.02),
        
        # Dashboard
        ('save_dashboard', True),
        ('show_interactive', True),
    )
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # === INDICATEURS ===
        
        # Prix
        self.hl2 = (self.data.high + self.data.low) / 2.0
        
        # ATR
        self.atr = bt.indicators.ATR(self.data, period=self.p.st_period)
        
        # Moyennes mobiles
        self.sma_20 = bt.indicators.SMA(self.data.close, period=20)
        self.sma_50 = bt.indicators.SMA(self.data.close, period=50)
        self.ema_12 = bt.indicators.EMA(self.data.close, period=12)
        self.ema_26 = bt.indicators.EMA(self.data.close, period=26)
        
        # MACD
        self.macd_line = self.ema_12 - self.ema_26
        self.macd_signal = bt.indicators.EMA(self.macd_line, period=self.p.macd_signal)
        self.macd_histogram = self.macd_line - self.macd_signal
        
        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        
        # Bollinger Bands
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20)
        
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
        self.peak_value = self.initial_cash
        self.max_drawdown = 0.0
        
        # Statistiques détaillées
        self.trade_history = []
        self.signal_history = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.win_streak = 0
        self.loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        
        # Données pour dashboard
        self.dashboard_data = {
            'datetime': [],
            'price': [],
            'equity': [],
            'drawdown': [],
            'sma_20': [],
            'sma_50': [],
            'bb_upper': [],
            'bb_middle': [],
            'bb_lower': [],
            'rsi': [],
            'macd': [],
            'macd_signal': [],
            'macd_hist': [],
            'volume': [],
            'volume_ma': [],
            'supertrend': [],
            'trend': [],
            'atr': []
        }
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🟢 BUY EXECUTED @ {order.executed.price:.5f} | Size: {order.executed.size}')
            else:
                self.log(f'🔴 SELL EXECUTED @ {order.executed.price:.5f} | Size: {order.executed.size}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('❌ Order Canceled/Margin/Rejected')
            
        self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            # Calculer les métriques du trade
            trade_data = {
                'datetime': self.datas[0].datetime.datetime(0),
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl / self.broker.getvalue() * 100,
                'size': trade.size,
                'entry_price': trade.price,
                'exit_price': self.data.close[0],
                'duration': len(trade),
                'is_long': trade.size > 0
            }
            
            self.trade_history.append(trade_data)
            
            if trade.pnl > 0:
                self.consecutive_losses = 0
                self.win_streak += 1
                self.loss_streak = 0
                self.max_win_streak = max(self.max_win_streak, self.win_streak)
                self.log(f'💰 PROFIT: ${trade.pnl:.2f} ({trade.pnl/self.broker.getvalue()*100:.2f}%)')
            else:
                self.consecutive_losses += 1
                self.loss_streak += 1
                self.win_streak = 0
                self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)
                self.log(f'💸 LOSS: ${trade.pnl:.2f} ({trade.pnl/self.broker.getvalue()*100:.2f}%) [Consecutive: {self.consecutive_losses}]')
    
    def update_supertrend(self):
        """Mise à jour SuperTrend"""
        if len(self.data) < 2:
            return
        
        try:
            basic_up = self.hl2[0] - self.p.st_multiplier * self.atr[0]
            basic_dn = self.hl2[0] + self.p.st_multiplier * self.atr[0]
            
            if len(self.data) == 2:
                self.supertrend_up = basic_up
                self.supertrend_dn = basic_dn
                self.trend = 1
            else:
                if self.data.close[-1] > self.supertrend_up:
                    self.supertrend_up = max(basic_up, self.supertrend_up)
                else:
                    self.supertrend_up = basic_up
                    
                if self.data.close[-1] < self.supertrend_dn:
                    self.supertrend_dn = min(basic_dn, self.supertrend_dn)
                else:
                    self.supertrend_dn = basic_dn
                    
                if self.trend == -1 and self.data.close[0] > self.supertrend_dn:
                    self.trend = 1
                elif self.trend == 1 and self.data.close[0] < self.supertrend_up:
                    self.trend = -1
        except Exception as e:
            self.log(f"Erreur SuperTrend: {e}")
    
    def collect_dashboard_data(self):
        """Collecter toutes les données pour le dashboard"""
        try:
            current_equity = self.broker.getvalue()
            
            # Mettre à jour le peak et drawdown
            if current_equity > self.peak_value:
                self.peak_value = current_equity
            
            current_drawdown = (self.peak_value - current_equity) / self.peak_value * 100
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Collecter les données
            self.dashboard_data['datetime'].append(self.datas[0].datetime.datetime(0))
            self.dashboard_data['price'].append(self.data.close[0])
            self.dashboard_data['equity'].append(current_equity)
            self.dashboard_data['drawdown'].append(current_drawdown)
            self.dashboard_data['sma_20'].append(self.sma_20[0])
            self.dashboard_data['sma_50'].append(self.sma_50[0])
            self.dashboard_data['bb_upper'].append(self.bb.top[0])
            self.dashboard_data['bb_middle'].append(self.bb.mid[0])
            self.dashboard_data['bb_lower'].append(self.bb.bot[0])
            self.dashboard_data['rsi'].append(self.rsi[0])
            self.dashboard_data['macd'].append(self.macd_line[0])
            self.dashboard_data['macd_signal'].append(self.macd_signal[0])
            self.dashboard_data['macd_hist'].append(self.macd_histogram[0])
            self.dashboard_data['volume'].append(self.data.volume[0])
            self.dashboard_data['volume_ma'].append(self.volume_ma[0])
            self.dashboard_data['atr'].append(self.atr[0])
            self.dashboard_data['trend'].append(self.trend)
            
            # SuperTrend
            if self.trend == 1:
                self.dashboard_data['supertrend'].append(self.supertrend_up)
            else:
                self.dashboard_data['supertrend'].append(self.supertrend_dn)
                
        except Exception as e:
            self.log(f"Erreur collecte données: {e}")
    
    def next(self):
        try:
            # Collecter les données
            self.collect_dashboard_data()
            
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
            
            # Bollinger Bands
            bb_squeeze = (self.bb.top[0] - self.bb.bot[0]) / self.bb.mid[0] < 0.02
            price_near_bb_lower = self.data.close[0] <= self.bb.bot[0] * 1.001
            price_near_bb_upper = self.data.close[0] >= self.bb.top[0] * 0.999
            
            # Volume confirmation
            volume_above_avg = self.data.volume[0] > self.volume_ma[0] * 1.2
            
            # Conditions avancées
            long_condition = (
                self.trend == 1 and 
                macd_bullish and 
                self.rsi[0] < 70 and 
                self.data.close[0] > self.sma_20[0] and
                not bb_squeeze and
                volume_above_avg
            )
            
            short_condition = (
                self.trend == -1 and 
                macd_bearish and 
                self.rsi[0] > 30 and 
                self.data.close[0] < self.sma_20[0] and
                not bb_squeeze and
                volume_above_avg
            )
            
            # === EXÉCUTION DES TRADES ===
            
            if self.order:
                return
            
            if not self.position:
                if long_condition:
                    size = int(self.broker.getvalue() * self.p.position_size / self.data.close[0])
                    self.log(f'🚀 BUY SIGNAL @ {self.data.close[0]:.5f} | RSI: {self.rsi[0]:.1f} | Trend: {self.trend}')
                    self.order = self.buy(size=max(1, size))
                    
                    # Enregistrer le signal
                    self.signal_history.append({
                        'datetime': self.datas[0].datetime.datetime(0),
                        'type': 'BUY',
                        'price': self.data.close[0],
                        'rsi': self.rsi[0],
                        'macd': self.macd_line[0],
                        'trend': self.trend,
                        'volume_ratio': self.data.volume[0] / self.volume_ma[0]
                    })
                    
                elif short_condition:
                    size = int(self.broker.getvalue() * self.p.position_size / self.data.close[0])
                    self.log(f'🚀 SELL SIGNAL @ {self.data.close[0]:.5f} | RSI: {self.rsi[0]:.1f} | Trend: {self.trend}')
                    self.order = self.sell(size=max(1, size))
                    
                    # Enregistrer le signal
                    self.signal_history.append({
                        'datetime': self.datas[0].datetime.datetime(0),
                        'type': 'SELL',
                        'price': self.data.close[0],
                        'rsi': self.rsi[0],
                        'macd': self.macd_line[0],
                        'trend': self.trend,
                        'volume_ratio': self.data.volume[0] / self.volume_ma[0]
                    })
            
            else:  # Position ouverte
                # Stop Loss et Take Profit
                if self.position.size > 0:  # Position longue
                    entry_price = self.position.price
                    stop_loss = entry_price * (1 - self.p.stop_loss_pct)
                    take_profit = entry_price * (1 + self.p.take_profit_pct)
                    
                    if (self.data.close[0] <= stop_loss or 
                        self.data.close[0] >= take_profit or
                        self.rsi[0] > 80 or 
                        self.trend == -1):
                        self.log(f'💰 LONG EXIT @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                        
                elif self.position.size < 0:  # Position courte
                    entry_price = self.position.price
                    stop_loss = entry_price * (1 + self.p.stop_loss_pct)
                    take_profit = entry_price * (1 - self.p.take_profit_pct)
                    
                    if (self.data.close[0] >= stop_loss or 
                        self.data.close[0] <= take_profit or
                        self.rsi[0] < 20 or 
                        self.trend == 1):
                        self.log(f'💰 SHORT EXIT @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                        
        except Exception as e:
            self.log(f"Erreur dans next(): {e}")
    
    def create_dashboard(self, config_name):
        """Créer le dashboard complet"""
        try:
            if not self.dashboard_data['datetime']:
                self.log("Pas de données pour créer le dashboard")
                return
            
            # Configuration du style
            plt.style.use('dark_background')
            sns.set_palette("husl")
            
            # Créer la figure avec subplots
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 3, height_ratios=[2, 1, 1, 1], width_ratios=[2, 1, 1])
            
            # Titre principal
            fig.suptitle(f'📊 MasterTrend Dashboard - {config_name}', 
                        fontsize=20, fontweight='bold', y=0.98)
            
            # === 1. GRAPHIQUE PRINCIPAL (Prix + Indicateurs) ===
            ax_main = fig.add_subplot(gs[0, :2])
            
            # Prix et moyennes mobiles
            ax_main.plot(self.dashboard_data['datetime'], self.dashboard_data['price'], 
                        color='#00ff88', linewidth=2, label='Prix', alpha=0.9)
            ax_main.plot(self.dashboard_data['datetime'], self.dashboard_data['sma_20'], 
                        color='#ff6b6b', linewidth=1.5, label='SMA 20', alpha=0.7)
            ax_main.plot(self.dashboard_data['datetime'], self.dashboard_data['sma_50'], 
                        color='#4ecdc4', linewidth=1.5, label='SMA 50', alpha=0.7)
            ax_main.plot(self.dashboard_data['datetime'], self.dashboard_data['supertrend'], 
                        color='#ffaa00', linewidth=2, label='SuperTrend', alpha=0.8)
            
            # Bollinger Bands
            ax_main.fill_between(self.dashboard_data['datetime'], 
                               self.dashboard_data['bb_upper'], 
                               self.dashboard_data['bb_lower'],
                               alpha=0.1, color='purple', label='Bollinger Bands')
            
            # Signaux de trading
            for signal in self.signal_history:
                color = '#00ff00' if signal['type'] == 'BUY' else '#ff0000'
                marker = '^' if signal['type'] == 'BUY' else 'v'
                ax_main.scatter(signal['datetime'], signal['price'], 
                              color=color, s=150, marker=marker, zorder=5,
                              edgecolors='white', linewidth=1)
            
            ax_main.set_title('Prix et Signaux de Trading', fontweight='bold', fontsize=14)
            ax_main.legend(loc='upper left')
            ax_main.grid(True, alpha=0.3)
            
            # === 2. EQUITY CURVE ===
            ax_equity = fig.add_subplot(gs[1, :2])
            
            if self.dashboard_data['equity']:
                returns = [(eq - self.initial_cash) / self.initial_cash * 100 
                          for eq in self.dashboard_data['equity']]
                ax_equity.plot(self.dashboard_data['datetime'], returns, 
                             color='#00aaff', linewidth=2, label='Rendement (%)')
                ax_equity.axhline(y=0, color='white', linestyle='--', alpha=0.5)
                ax_equity.fill_between(self.dashboard_data['datetime'], returns, 0,
                                     where=[r >= 0 for r in returns], 
                                     color='green', alpha=0.3, interpolate=True)
                ax_equity.fill_between(self.dashboard_data['datetime'], returns, 0,
                                     where=[r < 0 for r in returns], 
                                     color='red', alpha=0.3, interpolate=True)
            
            ax_equity.set_title('Courbe d\'Équité', fontweight='bold')
            ax_equity.set_ylabel('Rendement (%)')
            ax_equity.grid(True, alpha=0.3)
            
            # === 3. RSI ===
            ax_rsi = fig.add_subplot(gs[2, :2])
            ax_rsi.plot(self.dashboard_data['datetime'], self.dashboard_data['rsi'], 
                       color='#ffaa00', linewidth=2)
            ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax_rsi.axhline(y=50, color='white', linestyle='-', alpha=0.3)
            ax_rsi.fill_between(self.dashboard_data['datetime'], 70, 100, alpha=0.2, color='red')
            ax_rsi.fill_between(self.dashboard_data['datetime'], 0, 30, alpha=0.2, color='green')
            ax_rsi.set_title('RSI (14)', fontweight='bold')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True, alpha=0.3)
            
            # === 4. MACD ===
            ax_macd = fig.add_subplot(gs[3, :2])
            ax_macd.plot(self.dashboard_data['datetime'], self.dashboard_data['macd'], 
                        color='#aa00ff', linewidth=2, label='MACD')
            ax_macd.plot(self.dashboard_data['datetime'], self.dashboard_data['macd_signal'], 
                        color='orange', linewidth=1.5, label='Signal')
            
            # Histogramme coloré
            colors = ['green' if h >= 0 else 'red' for h in self.dashboard_data['macd_hist']]
            ax_macd.bar(self.dashboard_data['datetime'], self.dashboard_data['macd_hist'], 
                       color=colors, alpha=0.6, width=0.8)
            
            ax_macd.axhline(y=0, color='white', linestyle='-', alpha=0.3)
            ax_macd.set_title('MACD', fontweight='bold')
            ax_macd.legend()
            ax_macd.grid(True, alpha=0.3)
            
            # === 5. MÉTRIQUES DE PERFORMANCE ===
            ax_metrics = fig.add_subplot(gs[0, 2])
            ax_metrics.axis('off')
            
            # Calculer les métriques
            final_equity = self.dashboard_data['equity'][-1] if self.dashboard_data['equity'] else self.initial_cash
            total_return = (final_equity - self.initial_cash) / self.initial_cash * 100
            
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] < 0]) if (total_trades - winning_trades) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Texte des métriques
            metrics_text = f"""
📊 MÉTRIQUES DE PERFORMANCE

💰 Capital Initial: ${self.initial_cash:,.2f}
💰 Capital Final: ${final_equity:,.2f}
📈 Rendement Total: {total_return:.2f}%
📉 Drawdown Max: {self.max_drawdown:.2f}%

🎯 Total Trades: {total_trades}
✅ Trades Gagnants: {winning_trades}
❌ Trades Perdants: {total_trades - winning_trades}
🎯 Taux de Réussite: {win_rate:.1f}%

💰 Gain Moyen: ${avg_win:.2f}
💸 Perte Moyenne: ${avg_loss:.2f}
⚡ Profit Factor: {profit_factor:.2f}

🔥 Série Gagnante Max: {self.max_win_streak}
❄️ Série Perdante Max: {self.max_loss_streak}

🎯 Signaux Générés: {len(self.signal_history)}
🟢 Signaux BUY: {len([s for s in self.signal_history if s['type'] == 'BUY'])}
🔴 Signaux SELL: {len([s for s in self.signal_history if s['type'] == 'SELL'])}
            """
            
            ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8))
            
            # === 6. DISTRIBUTION DES TRADES ===
            ax_dist = fig.add_subplot(gs[1, 2])
            
            if self.trade_history:
                pnls = [t['pnl'] for t in self.trade_history]
                ax_dist.hist(pnls, bins=min(10, len(pnls)), alpha=0.7, color='skyblue', edgecolor='black')
                ax_dist.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax_dist.set_title('Distribution P&L', fontweight='bold')
                ax_dist.set_xlabel('P&L ($)')
                ax_dist.set_ylabel('Fréquence')
                ax_dist.grid(True, alpha=0.3)
            
            # === 7. DRAWDOWN ===
            ax_dd = fig.add_subplot(gs[2, 2])
            
            if self.dashboard_data['drawdown']:
                ax_dd.fill_between(self.dashboard_data['datetime'], 
                                 self.dashboard_data['drawdown'], 0,
                                 color='red', alpha=0.6)
                ax_dd.set_title('Drawdown (%)', fontweight='bold')
                ax_dd.set_ylabel('Drawdown (%)')
                ax_dd.grid(True, alpha=0.3)
            
            # === 8. VOLUME ===
            ax_vol = fig.add_subplot(gs[3, 2])
            
            if self.dashboard_data['volume']:
                ax_vol.bar(self.dashboard_data['datetime'], self.dashboard_data['volume'], 
                          alpha=0.6, color='gray', width=0.8)
                ax_vol.plot(self.dashboard_data['datetime'], self.dashboard_data['volume_ma'], 
                           color='orange', linewidth=2, label='Volume MA')
                ax_vol.set_title('Volume', fontweight='bold')
                ax_vol.legend()
                ax_vol.grid(True, alpha=0.3)
            
            # Format des dates pour tous les axes
            for ax in [ax_main, ax_equity, ax_rsi, ax_macd, ax_dd, ax_vol]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Sauvegarde
            if self.p.save_dashboard:
                filename = f'mastertrend_dashboard_{config_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
                self.log(f'📊 Dashboard sauvegardé: {filename}')
            
            if self.p.show_interactive:
                plt.show()
            
        except Exception as e:
            self.log(f"Erreur création dashboard: {e}")
    
    def stop(self):
        """Statistiques finales et création du dashboard"""
        try:
            final_value = self.broker.getvalue()
            total_return = (final_value - self.initial_cash) / self.initial_cash * 100
            
            self.log(f'=== RÉSULTATS MASTERTREND DASHBOARD ===')
            self.log(f'💰 Capital Initial: ${self.initial_cash:.2f}')
            self.log(f'💰 Capital Final: ${final_value:.2f}')
            self.log(f'📈 Rendement Total: {total_return:.2f}%')
            self.log(f'📉 Drawdown Max: {self.max_drawdown:.2f}%')
            self.log(f'🎯 Total Trades: {len(self.trade_history)}')
            self.log(f'🎯 Signaux Générés: {len(self.signal_history)}')
            
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
    # Test avec dashboard complet
    test_configs = [
        ("EURUSD_1M", "EURUSD_data_1M.csv", bt.TimeFrame.Minutes, 1),
    ]
    
    for config_name, filename, timeframe, compression in test_configs:
        print(f'\n{"="*80}')
        print(f'🚀 TESTING MASTERTREND DASHBOARD - {config_name}')
        print(f'📊 Mode: DASHBOARD COMPLET')
        print(f'🎯 Capital Initial: $10,000')
        print('='*80)
        
        if not check_data_file(filename):
            print(f'❌ FICHIER MANQUANT: {filename}')
            continue
        
        try:
            # Configuration
            cerebro = bt.Cerebro()
            
            # Données (période étendue pour plus d'analyse)
            data = bt.feeds.GenericCSVData(
                dataname=filename,
                dtformat=('%Y-%m-%d %H:%M:%S'),
                datetime=0, open=1, high=2, low=3, close=4, volume=5,
                timeframe=timeframe, compression=compression,
                openinterest=-1, headers=True, separator=',',
                fromdate=datetime.datetime(2025, 5, 22, 3, 0),
                todate=datetime.datetime(2025, 5, 22, 12, 0)  # Période plus longue
            )
            
            cerebro.adddata(data)
            cerebro.addstrategy(MasterTrendDashboard)
            
            # Configuration
            cerebro.broker.setcash(10000.0)
            cerebro.broker.setcommission(commission=0.0001)
            
            # Analyseurs
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            
            results = cerebro.run()
            strat = results[0]
            
            # Créer le dashboard
            strat.create_dashboard(config_name)
            
            # Analyse des résultats
            trades = strat.analyzers.trades.get_analysis()
            returns = strat.analyzers.returns.get_analysis()
            drawdown = strat.analyzers.drawdown.get_analysis()
            
            total_trades = getattr(getattr(trades, 'total', None), 'closed', 0)
            total_return = getattr(returns, 'rtot', 0) * 100
            max_dd = getattr(drawdown, 'max', {}).get('drawdown', 0)
            
            print(f'\n🔥 RÉSULTATS {config_name}:')
            print(f'📊 Total Trades: {total_trades}')
            print(f'📈 Rendement Total: {total_return:.2f}%')
            print(f'📉 Drawdown Max: {max_dd:.2f}%')
            print(f'📊 Dashboard généré avec succès!')
                
        except Exception as e:
            print(f'❌ ERREUR sur {config_name}: {e}')
    
    print(f'\n{"="*80}')
    print('🎨 MASTERTREND DASHBOARD - Analyse complète terminée!')
    print('📊 Vérifiez les fichiers PNG générés pour l\'analyse détaillée.')
    print('='*80) 