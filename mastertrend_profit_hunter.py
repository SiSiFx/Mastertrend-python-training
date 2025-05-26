#!/usr/bin/env python3
"""
MASTERTREND PROFIT HUNTER
Version ultra-agressive pour maximiser les profits
Objectif: Générer 15-30% de rendement minimum
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
from collections import deque


class ProfitHunterMasterTrend(bt.Strategy):
    """
    Stratégie MasterTrend AGRESSIVE pour maximiser les profits
    """
    
    params = (
        # SuperTrend AGRESSIF
        ('st_period', 8),            # Plus réactif
        ('st_multiplier', 2.2),      # Plus sensible
        
        # MACD AGRESSIF
        ('macd_fast', 8),            # Plus rapide
        ('macd_slow', 18),           # Plus rapide
        ('macd_signal', 6),          # Plus rapide
        
        # Williams Fractals AGRESSIFS
        ('williams_left', 1),        # Plus réactif
        ('williams_right', 1),       # Plus réactif
        ('williams_buffer', 0.02),   # Buffer minimal
        
        # P1 Indicator AGRESSIF
        ('p1_e31', 3),              # Plus rapide
        ('p1_m', 6),                # Plus rapide
        ('p1_l31', 10),             # Plus rapide
        
        # StopGap Filter PERMISSIF
        ('sg_ema_short', 6),         # Plus rapide
        ('sg_ema_long', 15),         # Plus rapide
        ('sg_lookback', 3),          # Plus court
        ('sg_median_period', 8),     # Plus court
        ('sg_multiplier', 0.8),      # Plus permissif
        
        # Filtres ASSOUPLIS pour plus de trades
        ('rsi_period', 10),          # Plus réactif
        ('rsi_oversold', 20),        # Plus permissif
        ('rsi_overbought', 80),      # Plus permissif
        ('volume_ma_period', 15),    # Plus court
        ('min_volume_ratio', 0.8),   # Plus permissif
        
        # Risk Management AGRESSIF
        ('max_daily_loss', 0.08),    # 8% perte journalière max
        ('max_total_loss', 0.15),    # 15% perte totale max
        ('profit_target', 0.30),     # 30% objectif de profit
        ('position_size', 0.05),     # 5% par trade (AGRESSIF)
        ('max_consecutive_losses', 5), # Plus de tolérance
        
        # Sessions ÉTENDUES
        ('trading_start', datetime.time(8, 0)),   # Plus tôt
        ('trading_end', datetime.time(18, 0)),    # Plus tard
        
        # Filtres de marché PERMISSIFS
        ('min_atr_ratio', 0.3),      # Plus permissif
        ('max_atr_ratio', 3.0),      # Plus permissif
        ('trend_strength_period', 10), # Plus court
        ('min_trend_strength', 0.3),   # Plus permissif
        
        # NOUVEAUX PARAMÈTRES AGRESSIFS
        ('use_leverage', True),       # Utiliser l'effet de levier
        ('leverage_factor', 2.0),     # Facteur de levier
        ('scalping_mode', True),      # Mode scalping
        ('quick_exit_rsi', 85),       # Sortie rapide si RSI extrême
        ('quick_exit_profit', 0.02),  # Sortie rapide à 2% de profit
        ('martingale_mode', True),    # Mode martingale après pertes
        ('martingale_multiplier', 1.5), # Multiplicateur martingale
        ('trend_following', True),    # Suivre la tendance aggressivement
        ('breakout_trading', True),   # Trading de breakout
        ('news_trading', True),       # Trading sur les nouvelles (volatilité)
    )
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # === INDICATEURS AGRESSIFS ===
        
        # SuperTrend agressif
        self.hl2 = (self.data.high + self.data.low) / 2.0
        self.atr = bt.indicators.ATR(self.data, period=self.p.st_period)
        self.atr_ma = bt.indicators.SMA(self.atr, period=10)  # Plus court
        
        # Variables SuperTrend
        self.basic_up = self.hl2 - self.p.st_multiplier * self.atr
        self.basic_dn = self.hl2 + self.p.st_multiplier * self.atr
        
        # MACD agressif
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        
        # RSI agressif
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        
        # Volume agressif
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
        
        # Williams Fractals agressifs
        self.fractal_high = bt.indicators.Highest(self.data.high, period=self.p.williams_left + self.p.williams_right + 1)
        self.fractal_low = bt.indicators.Lowest(self.data.low, period=self.p.williams_left + self.p.williams_right + 1)
        
        # P1 Indicator agressif
        self.hilow = (self.data.high - self.data.low) * 100
        self.openclose = (self.data.close - self.data.open) * 100
        self.spreadv = self.openclose * self.data.close
        
        # Période plus courte pour plus de réactivité
        self.pt_approx = bt.indicators.SMA(self.spreadv, period=50)
        
        self.ema_e31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_e31)
        self.ema_m = bt.indicators.EMA(self.pt_approx, period=self.p.p1_m)
        self.ema_l31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_l31)
        
        self.a1 = self.ema_l31 - self.ema_m
        self.b1 = self.ema_e31 - self.ema_m
        self.p1 = self.a1 + self.b1
        
        # StopGap Filter permissif
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_long)
        self.highest_sg = bt.indicators.Highest(self.data.high, period=self.p.sg_lookback)
        self.lowest_sg = bt.indicators.Lowest(self.data.low, period=self.p.sg_lookback)
        
        # Indicateurs additionnels pour l'agressivité
        self.close_ma = bt.indicators.SMA(self.data.close, period=self.p.trend_strength_period)
        self.momentum = bt.indicators.Momentum(self.data.close, period=5)
        self.stoch = bt.indicators.Stochastic(self.data, period=8)
        
        # Détection de breakout
        self.bb = bt.indicators.BollingerBands(self.data.close, period=15, devfactor=1.8)
        
        # === VARIABLES D'ÉTAT AGRESSIVES ===
        self.trend = 1
        self.final_up = 0.0
        self.final_dn = 0.0
        
        # Williams Stops
        self.williams_long_stop = None
        self.williams_short_stop = None
        self.williams_long_active = False
        self.williams_short_active = False
        
        # Tracking agressif
        self.initial_cash = self.broker.getvalue()
        self.daily_start_cash = self.initial_cash
        self.peak_value = self.initial_cash
        self.current_date = None
        
        # Trade Management agressif
        self.order = None
        self.position_entry_price = None
        self.consecutive_losses = 0
        self.last_trade_profit = 0
        self.martingale_size = 1.0
        
        # StopGap median calculation
        self.candle_sizes = deque(maxlen=self.p.sg_median_period)
        
        # Statistiques agressives
        self.buy_signals = []
        self.sell_signals = []
        self.trade_history = []
        self.total_signals = 0
        self.profitable_signals = 0
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🟢 BUY EXECUTED @ {order.executed.price:.5f} (Size: {order.executed.size})')
                self.position_entry_price = order.executed.price
            else:
                self.log(f'🔴 SELL EXECUTED @ {order.executed.price:.5f} (Size: {order.executed.size})')
                self.position_entry_price = order.executed.price
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('❌ Order Canceled/Margin/Rejected')
            
        self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.last_trade_profit = trade.pnl
            if trade.pnl > 0:
                self.consecutive_losses = 0
                self.martingale_size = 1.0  # Reset martingale
                self.profitable_signals += 1
                self.log(f'💰 PROFIT: ${trade.pnl:.2f} (ROI: {trade.pnl/abs(trade.value)*100:.1f}%)')
            else:
                self.consecutive_losses += 1
                if self.p.martingale_mode:
                    self.martingale_size *= self.p.martingale_multiplier
                    self.martingale_size = min(self.martingale_size, 3.0)  # Limite à 3x
                self.log(f'💸 LOSS: ${trade.pnl:.2f} (Consecutive: {self.consecutive_losses})')
            
            # Enregistrer le trade
            self.trade_history.append({
                'profit': trade.pnl,
                'duration': trade.barlen,
                'entry_price': self.position_entry_price,
                'roi': trade.pnl/abs(trade.value)*100 if trade.value != 0 else 0
            })
    
    def update_supertrend(self):
        """Mise à jour SuperTrend agressif"""
        if len(self.data) < 2:
            return
            
        if self.data.close[-1] > self.final_up:
            self.final_up = max(self.basic_up[0], self.final_up)
        else:
            self.final_up = self.basic_up[0]
            
        if self.data.close[-1] < self.final_dn:
            self.final_dn = min(self.basic_dn[0], self.final_dn)
        else:
            self.final_dn = self.basic_dn[0]
            
        prev_trend = self.trend
        if self.trend == -1 and self.data.close[0] > self.final_dn:
            self.trend = 1
        elif self.trend == 1 and self.data.close[0] < self.final_up:
            self.trend = -1
    
    def update_williams_stops(self):
        """Mise à jour Williams Stops agressifs"""
        if len(self.data) > self.p.williams_left + self.p.williams_right:
            center_idx = self.p.williams_right
            
            # High fractal (plus agressif)
            is_high_fractal = True
            center_high = self.data.high[-center_idx]
            
            for i in range(self.p.williams_left + self.p.williams_right + 1):
                if i != center_idx and self.data.high[-i] >= center_high:
                    is_high_fractal = False
                    break
            
            if is_high_fractal:
                self.williams_short_stop = center_high * (1 + self.p.williams_buffer / 100)
                self.williams_short_active = True
            
            # Low fractal (plus agressif)
            is_low_fractal = True
            center_low = self.data.low[-center_idx]
            
            for i in range(self.p.williams_left + self.p.williams_right + 1):
                if i != center_idx and self.data.low[-i] <= center_low:
                    is_low_fractal = False
                    break
            
            if is_low_fractal:
                self.williams_long_stop = center_low * (1 - self.p.williams_buffer / 100)
                self.williams_long_active = True
    
    def calculate_stopgap_filter(self):
        """Calcul du filtre StopGap permissif"""
        trend_up = self.ema_short[0] > self.ema_long[0]
        trend_down = self.ema_short[0] < self.ema_long[0]
        
        candle_size = abs(self.data.high[0] - self.data.low[0])
        self.candle_sizes.append(candle_size)
        
        if len(self.candle_sizes) == self.p.sg_median_period:
            median = np.median(list(self.candle_sizes))
        else:
            median = candle_size
        
        stop_gap = 0.0
        if trend_down and self.data.high[0] < self.highest_sg[-1]:
            stop_gap = abs(self.highest_sg[-1] - self.data.low[0])
        elif trend_up and self.data.low[0] > self.lowest_sg[-1]:
            stop_gap = abs(self.data.high[0] - self.lowest_sg[-1])
        
        # Plus permissif
        return stop_gap > (median * self.p.sg_multiplier)
    
    def check_breakout_conditions(self):
        """Détection de breakout pour trades agressifs"""
        if not self.p.breakout_trading:
            return False, False
        
        # Breakout Bollinger Bands
        bb_breakout_up = self.data.close[0] > self.bb.top[0]
        bb_breakout_down = self.data.close[0] < self.bb.bot[0]
        
        # Volume breakout
        volume_breakout = self.data.volume[0] > self.volume_ma[0] * 1.5
        
        # Momentum breakout
        momentum_up = self.momentum[0] > 0.0005
        momentum_down = self.momentum[0] < -0.0005
        
        long_breakout = bb_breakout_up and volume_breakout and momentum_up
        short_breakout = bb_breakout_down and volume_breakout and momentum_down
        
        return long_breakout, short_breakout
    
    def check_scalping_conditions(self):
        """Conditions de scalping rapide"""
        if not self.p.scalping_mode:
            return False, False
        
        # RSI rapide
        rsi_long = self.rsi[0] < 35 and self.rsi[-1] >= 35
        rsi_short = self.rsi[0] > 65 and self.rsi[-1] <= 65
        
        # Stochastic rapide
        stoch_long = self.stoch.percK[0] < 25 and self.stoch.percK[-1] >= 25
        stoch_short = self.stoch.percK[0] > 75 and self.stoch.percK[-1] <= 75
        
        # Momentum rapide
        momentum_long = self.momentum[0] > self.momentum[-1] and self.momentum[0] > 0
        momentum_short = self.momentum[0] < self.momentum[-1] and self.momentum[0] < 0
        
        scalp_long = (rsi_long or stoch_long) and momentum_long
        scalp_short = (rsi_short or stoch_short) and momentum_short
        
        return scalp_long, scalp_short
    
    def check_aggressive_rules(self):
        """Vérification des règles agressives"""
        current_value = self.broker.getvalue()
        dt = self.datas[0].datetime.datetime(0)
        
        # Reset daily tracking
        if self.current_date != dt.date():
            self.current_date = dt.date()
            self.daily_start_cash = current_value
        
        # Update peak
        self.peak_value = max(self.peak_value, current_value)
        
        # Check consecutive losses (plus tolérant)
        if self.consecutive_losses >= self.p.max_consecutive_losses:
            self.log(f'⚠️  MAX CONSECUTIVE LOSSES HIT: {self.consecutive_losses}')
            return False
        
        # Check daily loss limit (plus tolérant)
        daily_pnl = (current_value - self.daily_start_cash) / self.initial_cash
        if daily_pnl < -self.p.max_daily_loss:
            self.log(f'⚠️  DAILY LOSS LIMIT HIT: {daily_pnl*100:.2f}%')
            return False
        
        # Check total drawdown (plus tolérant)
        total_dd = (self.peak_value - current_value) / self.peak_value
        if total_dd > self.p.max_total_loss:
            self.log(f'⚠️  TOTAL DRAWDOWN LIMIT HIT: {total_dd*100:.2f}%')
            return False
        
        # Check profit target
        total_profit = (current_value - self.initial_cash) / self.initial_cash
        if total_profit >= self.p.profit_target:
            self.log(f'🎯 PROFIT TARGET REACHED: {total_profit*100:.2f}%')
            return False
        
        return True
    
    def is_trading_session(self):
        """Vérification session de trading étendue"""
        dt = self.datas[0].datetime.datetime(0)
        current_time = dt.time()
        return self.p.trading_start <= current_time <= self.p.trading_end
    
    def calculate_aggressive_position_size(self):
        """Calcul de la taille de position agressive"""
        account_value = self.broker.getvalue()
        
        # Taille de base agressive
        base_size = self.p.position_size
        
        # Appliquer martingale si activé
        if self.p.martingale_mode:
            base_size *= self.martingale_size
        
        # Appliquer leverage si activé
        if self.p.use_leverage:
            base_size *= self.p.leverage_factor
        
        risk_amount = account_value * base_size
        
        # Utiliser Williams Stop pour calculer la taille
        if self.williams_long_stop:
            stop_distance = abs(self.data.close[0] - self.williams_long_stop)
        elif self.williams_short_stop:
            stop_distance = abs(self.data.close[0] - self.williams_short_stop)
        else:
            stop_distance = self.data.close[0] * 0.005  # 0.5% par défaut
        
        if stop_distance > 0:
            size = risk_amount / stop_distance
            # Limite plus élevée pour l'agressivité
            max_size = account_value * 0.3 / self.data.close[0]  # Maximum 30% du capital
            size = min(size, max_size)
            return max(1, int(size))
        return 1
    
    def next(self):
        # Vérification des règles agressives
        if not self.check_aggressive_rules():
            if self.position:
                self.close()
            return
        
        # Vérification session de trading
        if not self.is_trading_session():
            return
        
        # Mise à jour des indicateurs
        self.update_supertrend()
        self.update_williams_stops()
        
        # Calcul des signaux agressifs
        
        # MACD Crossovers (plus sensibles)
        crossmacdbear = (self.macd.macd[0] > 0.00005 and self.macd.macd[-1] <= 0.00005)
        crossmacd = (self.macd.macd[0] < -0.00005 and self.macd.macd[-1] >= -0.00005)
        
        # P1 Conditions (plus permissives)
        b1_ge_p1 = self.b1[0] >= self.p1[0] * 1.02  # 2% de marge seulement
        b1_le_p1 = self.b1[0] <= self.p1[0] * 0.98  # 2% de marge seulement
        
        # StopGap Filter
        stopgap_ok = self.calculate_stopgap_filter()
        
        # Conditions de breakout et scalping
        breakout_long, breakout_short = self.check_breakout_conditions()
        scalp_long, scalp_short = self.check_scalping_conditions()
        
        # === CONDITIONS AGRESSIVES ===
        
        long_condition = (
            (self.williams_long_active and crossmacdbear and b1_ge_p1 and self.trend == 1 and stopgap_ok) or
            breakout_long or
            scalp_long
        )
        
        short_condition = (
            (self.williams_short_active and crossmacd and b1_le_p1 and self.trend == -1 and stopgap_ok) or
            breakout_short or
            scalp_short
        )
        
        # === EXÉCUTION DES TRADES AGRESSIFS ===
        
        if self.order:
            return
        
        if not self.position:
            if long_condition:
                self.total_signals += 1
                size = self.calculate_aggressive_position_size()
                self.log(f'🚀 AGGRESSIVE BUY @ {self.data.close[0]:.5f} (Size: {size}, RSI: {self.rsi[0]:.1f})')
                self.order = self.buy(size=size)
                self.buy_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
                
            elif short_condition:
                self.total_signals += 1
                size = self.calculate_aggressive_position_size()
                self.log(f'🚀 AGGRESSIVE SELL @ {self.data.close[0]:.5f} (Size: {size}, RSI: {self.rsi[0]:.1f})')
                self.order = self.sell(size=size)
                self.sell_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
        
        else:  # Position ouverte - gestion agressive
            if self.position.size > 0:  # Position longue
                # Quick exit sur profit
                if (self.position_entry_price and 
                    self.data.close[0] >= self.position_entry_price * (1 + self.p.quick_exit_profit)):
                    self.log(f'💰 QUICK PROFIT EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Quick exit sur RSI extrême
                elif self.rsi[0] > self.p.quick_exit_rsi:
                    self.log(f'📈 RSI EXIT @ {self.data.close[0]:.5f} (RSI: {self.rsi[0]:.1f})')
                    self.order = self.close()
                # Stop loss Williams
                elif (self.williams_long_stop and 
                      self.data.low[0] <= self.williams_long_stop):
                    self.log(f'🛑 LONG STOP @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Signal de sortie MACD
                elif crossmacd:
                    self.log(f'🔄 LONG EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                    
            elif self.position.size < 0:  # Position courte
                # Quick exit sur profit
                if (self.position_entry_price and 
                    self.data.close[0] <= self.position_entry_price * (1 - self.p.quick_exit_profit)):
                    self.log(f'💰 QUICK PROFIT EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Quick exit sur RSI extrême
                elif self.rsi[0] < (100 - self.p.quick_exit_rsi):
                    self.log(f'📉 RSI EXIT @ {self.data.close[0]:.5f} (RSI: {self.rsi[0]:.1f})')
                    self.order = self.close()
                # Stop loss Williams
                elif (self.williams_short_stop and 
                      self.data.high[0] >= self.williams_short_stop):
                    self.log(f'🛑 SHORT STOP @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Signal de sortie MACD
                elif crossmacdbear:
                    self.log(f'🔄 SHORT EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
    
    def stop(self):
        """Statistiques finales agressives"""
        final_value = self.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        max_dd = (self.peak_value - final_value) / self.peak_value * 100
        
        self.log(f'=== RÉSULTATS PROFIT HUNTER ===')
        self.log(f'💰 Capital Initial: ${self.initial_cash:.2f}')
        self.log(f'💰 Capital Final: ${final_value:.2f}')
        self.log(f'📈 Rendement Total: {total_return:.2f}%')
        self.log(f'📉 Drawdown Max: {max_dd:.2f}%')
        self.log(f'🎯 Signaux Générés: {self.total_signals}')
        self.log(f'🟢 Signaux BUY: {len(self.buy_signals)}')
        self.log(f'🔴 Signaux SELL: {len(self.sell_signals)}')
        self.log(f'📊 Trades Total: {len(self.trade_history)}')
        self.log(f'💸 Pertes Consécutives Max: {self.consecutive_losses}')
        
        # Analyse des trades
        if self.trade_history:
            profitable_trades = [t for t in self.trade_history if t['profit'] > 0]
            win_rate = len(profitable_trades) / len(self.trade_history) * 100
            avg_profit = sum(t['profit'] for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0
            avg_loss = sum(t['profit'] for t in self.trade_history if t['profit'] < 0) / len([t for t in self.trade_history if t['profit'] < 0]) if any(t['profit'] < 0 for t in self.trade_history) else 0
            avg_roi = sum(t['roi'] for t in self.trade_history) / len(self.trade_history)
            
            self.log(f'🎯 Taux de Réussite: {win_rate:.1f}%')
            self.log(f'💰 Profit Moyen: ${avg_profit:.2f}')
            self.log(f'💸 Perte Moyenne: ${avg_loss:.2f}')
            self.log(f'📊 ROI Moyen: {avg_roi:.2f}%')
            if avg_loss != 0:
                profit_factor = abs(avg_profit / avg_loss)
                self.log(f'⚡ Profit Factor: {profit_factor:.2f}')
        
        # Évaluation finale
        if total_return >= 15:
            self.log('🏆 EXCELLENT! Objectif de profit atteint!')
        elif total_return >= 8:
            self.log('✅ BON! Profit satisfaisant')
        elif total_return >= 3:
            self.log('⚠️  MOYEN. Peut mieux faire')
        else:
            self.log('❌ INSUFFISANT. Stratégie à revoir')


if __name__ == '__main__':
    # Configuration agressive
    cerebro = bt.Cerebro()
    
    # Données
    data = bt.feeds.GenericCSVData(
        dataname="EURUSD_data_15M.csv",
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes, compression=15,
        openinterest=-1, headers=True, separator=','
    )
    
    cerebro.adddata(data)
    cerebro.addstrategy(ProfitHunterMasterTrend)
    
    # Configuration agressive
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)
    
    # Analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print('🚀 MASTERTREND PROFIT HUNTER - DÉMARRAGE')
    print('💰 Objectif: 15-30% de rendement minimum')
    print('⚡ Mode: ULTRA-AGRESSIF')
    print('🎯 Capital Initial: $10,000')
    print('=' * 50)
    
    try:
        results = cerebro.run()
        strat = results[0]
        
        # Analyse détaillée des résultats
        trades = strat.analyzers.trades.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        
        print('\n🔥 ANALYSE PROFIT HUNTER:')
        if hasattr(trades, 'total') and trades.total.closed > 0:
            print(f'📊 Total Trades: {trades.total.closed}')
            print(f'🏆 Trades Gagnants: {trades.won.total if hasattr(trades, "won") else 0}')
            print(f'💸 Trades Perdants: {trades.lost.total if hasattr(trades, "lost") else 0}')
            if hasattr(trades, 'won') and trades.won.total > 0:
                win_rate = trades.won.total / trades.total.closed * 100
                print(f'🎯 Taux de Réussite: {win_rate:.1f}%')
                
                if hasattr(trades.won, 'pnl') and hasattr(trades.lost, 'pnl'):
                    avg_win = trades.won.pnl.average if hasattr(trades.won.pnl, 'average') else 0
                    avg_loss = trades.lost.pnl.average if hasattr(trades.lost.pnl, 'average') else 0
                    if avg_loss != 0:
                        profit_factor = abs(avg_win / avg_loss)
                        print(f'⚡ Profit Factor: {profit_factor:.2f}')
        
        if hasattr(drawdown, 'max'):
            print(f'📉 Drawdown Maximum: {drawdown.max.drawdown:.2f}%')
            
        if hasattr(returns, 'rtot'):
            print(f'📈 Rendement Total: {returns.rtot*100:.2f}%')
            
        # Évaluation finale
        final_return = returns.rtot*100 if hasattr(returns, 'rtot') else 0
        if final_return >= 15:
            print('\n🏆 MISSION ACCOMPLIE! Profit Hunter a réussi!')
        elif final_return >= 8:
            print('\n✅ Bon résultat! Objectif partiellement atteint')
        else:
            print('\n⚠️  Résultat insuffisant. Optimisation nécessaire.')
            
    except Exception as e:
        print(f'❌ ERREUR LORS DE L\'EXÉCUTION: {e}')
        print('Vérifiez que le fichier EURUSD_data_15M.csv existe et a le bon format.') 