#!/usr/bin/env python3
"""
MASTERTREND BALANCED STRATEGY
Version équilibrée pour des profits consistants
Objectif: Générer 10-20% de rendement avec un bon ratio risque/récompense
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
from collections import deque


class BalancedMasterTrend(bt.Strategy):
    """
    Stratégie MasterTrend ÉQUILIBRÉE pour des profits consistants
    """
    
    params = (
        # SuperTrend ÉQUILIBRÉ
        ('st_period', 12),           # Équilibré
        ('st_multiplier', 2.5),      # Équilibré
        
        # MACD ÉQUILIBRÉ
        ('macd_fast', 10),           # Équilibré
        ('macd_slow', 22),           # Équilibré
        ('macd_signal', 8),          # Équilibré
        
        # Williams Fractals ÉQUILIBRÉS
        ('williams_left', 3),        # Équilibré
        ('williams_right', 3),       # Équilibré
        ('williams_buffer', 0.08),   # Buffer raisonnable
        
        # P1 Indicator ÉQUILIBRÉ
        ('p1_e31', 5),              # Équilibré
        ('p1_m', 9),                # Équilibré
        ('p1_l31', 14),             # Équilibré
        
        # StopGap Filter ÉQUILIBRÉ
        ('sg_ema_short', 8),         # Équilibré
        ('sg_ema_long', 18),         # Équilibré
        ('sg_lookback', 5),          # Équilibré
        ('sg_median_period', 10),    # Équilibré
        ('sg_multiplier', 1.0),      # Équilibré
        
        # Filtres ÉQUILIBRÉS
        ('rsi_period', 14),          # Standard
        ('rsi_oversold', 30),        # Standard
        ('rsi_overbought', 70),      # Standard
        ('volume_ma_period', 20),    # Standard
        ('min_volume_ratio', 1.0),   # Standard
        
        # Risk Management ÉQUILIBRÉ
        ('max_daily_loss', 0.05),    # 5% perte journalière max
        ('max_total_loss', 0.10),    # 10% perte totale max
        ('profit_target', 0.20),     # 20% objectif de profit
        ('position_size', 0.03),     # 3% par trade (ÉQUILIBRÉ)
        ('max_consecutive_losses', 4), # Limite raisonnable
        
        # Sessions ÉQUILIBRÉES
        ('trading_start', datetime.time(8, 0)),   # Heures normales
        ('trading_end', datetime.time(17, 0)),    # Heures normales
        
        # Filtres de marché ÉQUILIBRÉS
        ('min_atr_ratio', 0.5),      # Équilibré
        ('max_atr_ratio', 2.5),      # Équilibré
        ('trend_strength_period', 14), # Standard
        ('min_trend_strength', 0.5),   # Équilibré
        
        # PARAMÈTRES ÉQUILIBRÉS
        ('use_leverage', False),      # Pas de levier
        ('leverage_factor', 1.0),     # Pas de levier
        ('scalping_mode', False),     # Pas de scalping
        ('quick_exit_rsi', 75),       # Sortie modérée
        ('quick_exit_profit', 0.025), # Sortie à 2.5% de profit
        ('martingale_mode', False),   # Pas de martingale
        ('trend_following', True),    # Suivre la tendance
        ('breakout_trading', True),   # Trading de breakout modéré
        ('momentum_trading', True),   # Trading de momentum modéré
        
        # NOUVEAUX PARAMÈTRES D'ÉQUILIBRAGE
        ('signal_confirmation', True), # Confirmation des signaux
        ('trend_filter', True),       # Filtre de tendance
        ('volatility_filter', True),  # Filtre de volatilité
        ('time_filter', True),        # Filtre temporel
        ('correlation_threshold', 0.7), # Seuil de corrélation
    )
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # === INDICATEURS ÉQUILIBRÉS ===
        
        # SuperTrend équilibré
        self.hl2 = (self.data.high + self.data.low) / 2.0
        self.atr = bt.indicators.ATR(self.data, period=self.p.st_period)
        self.atr_ma = bt.indicators.SMA(self.atr, period=14)
        
        # Variables SuperTrend
        self.basic_up = self.hl2 - self.p.st_multiplier * self.atr
        self.basic_dn = self.hl2 + self.p.st_multiplier * self.atr
        
        # MACD équilibré
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        
        # RSI standard
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        
        # Volume standard
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
        
        # Williams Fractals équilibrés
        self.fractal_high = bt.indicators.Highest(self.data.high, period=self.p.williams_left + self.p.williams_right + 1)
        self.fractal_low = bt.indicators.Lowest(self.data.low, period=self.p.williams_left + self.p.williams_right + 1)
        
        # P1 Indicator équilibré
        self.hilow = (self.data.high - self.data.low) * 100
        self.openclose = (self.data.close - self.data.open) * 100
        self.spreadv = self.openclose * self.data.close
        
        # Période équilibrée
        self.pt_approx = bt.indicators.SMA(self.spreadv, period=50)
        
        self.ema_e31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_e31)
        self.ema_m = bt.indicators.EMA(self.pt_approx, period=self.p.p1_m)
        self.ema_l31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_l31)
        
        self.a1 = self.ema_l31 - self.ema_m
        self.b1 = self.ema_e31 - self.ema_m
        self.p1 = self.a1 + self.b1
        
        # StopGap Filter équilibré
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_long)
        self.highest_sg = bt.indicators.Highest(self.data.high, period=self.p.sg_lookback)
        self.lowest_sg = bt.indicators.Lowest(self.data.low, period=self.p.sg_lookback)
        
        # Indicateurs de tendance
        self.sma_20 = bt.indicators.SMA(self.data.close, period=20)
        self.sma_50 = bt.indicators.SMA(self.data.close, period=50)
        self.ema_9 = bt.indicators.EMA(self.data.close, period=9)
        self.ema_21 = bt.indicators.EMA(self.data.close, period=21)
        
        # Indicateurs de momentum
        self.momentum = bt.indicators.Momentum(self.data.close, period=10)
        self.stoch = bt.indicators.Stochastic(self.data, period=14)
        
        # Détection de breakout équilibrée
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2.0)
        
        # Indicateurs de volatilité
        self.volatility = bt.indicators.StdDev(self.data.close, period=20)
        self.volatility_ma = bt.indicators.SMA(self.volatility, period=10)
        
        # === VARIABLES D'ÉTAT ÉQUILIBRÉES ===
        self.trend = 1
        self.final_up = 0.0
        self.final_dn = 0.0
        
        # Williams Stops
        self.williams_long_stop = None
        self.williams_short_stop = None
        self.williams_long_active = False
        self.williams_short_active = False
        
        # Tracking équilibré
        self.initial_cash = self.broker.getvalue()
        self.daily_start_cash = self.initial_cash
        self.peak_value = self.initial_cash
        self.current_date = None
        
        # Trade Management équilibré
        self.order = None
        self.position_entry_price = None
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.last_trade_profit = 0
        
        # StopGap median calculation
        self.candle_sizes = deque(maxlen=self.p.sg_median_period)
        
        # Statistiques équilibrées
        self.buy_signals = []
        self.sell_signals = []
        self.trade_history = []
        self.total_signals = 0
        self.profitable_signals = 0
        
        # Variables d'équilibrage
        self.signal_strength = 0.0
        self.trend_strength = 0.0
        self.last_signal_time = None
        self.min_signal_interval = 4  # Minimum 4 barres entre signaux
    
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
                self.consecutive_wins += 1
                self.profitable_signals += 1
                roi = trade.pnl/abs(trade.value)*100 if trade.value != 0 else 0
                self.log(f'💰 PROFIT: ${trade.pnl:.2f} (ROI: {roi:.1f}%) [Win Streak: {self.consecutive_wins}]')
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.log(f'💸 LOSS: ${trade.pnl:.2f} (Consecutive: {self.consecutive_losses})')
            
            # Enregistrer le trade
            self.trade_history.append({
                'profit': trade.pnl,
                'duration': trade.barlen,
                'entry_price': self.position_entry_price,
                'roi': trade.pnl/abs(trade.value)*100 if trade.value != 0 else 0
            })
    
    def update_supertrend(self):
        """Mise à jour SuperTrend équilibré"""
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
        """Mise à jour Williams Stops équilibrés"""
        if len(self.data) > self.p.williams_left + self.p.williams_right:
            center_idx = self.p.williams_right
            
            # High fractal (équilibré)
            is_high_fractal = True
            center_high = self.data.high[-center_idx]
            
            for i in range(self.p.williams_left + self.p.williams_right + 1):
                if i != center_idx and self.data.high[-i] >= center_high:
                    is_high_fractal = False
                    break
            
            if is_high_fractal:
                self.williams_short_stop = center_high * (1 + self.p.williams_buffer / 100)
                self.williams_short_active = True
            
            # Low fractal (équilibré)
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
        """Calcul du filtre StopGap équilibré"""
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
        
        return stop_gap > (median * self.p.sg_multiplier)
    
    def check_trend_filter(self):
        """Vérification du filtre de tendance"""
        if not self.p.trend_filter:
            return True, True
        
        # Tendance haussière: EMAs alignées + prix au-dessus SMA20
        trend_up = (self.ema_9[0] > self.ema_21[0] and 
                   self.data.close[0] > self.sma_20[0] and
                   self.sma_20[0] > self.sma_50[0])
        
        # Tendance baissière: EMAs alignées + prix en-dessous SMA20
        trend_down = (self.ema_9[0] < self.ema_21[0] and 
                     self.data.close[0] < self.sma_20[0] and
                     self.sma_20[0] < self.sma_50[0])
        
        return trend_up, trend_down
    
    def check_volatility_filter(self):
        """Vérification du filtre de volatilité"""
        if not self.p.volatility_filter:
            return True
        
        # Volatilité normale (ni trop haute ni trop basse)
        volatility_ratio = self.volatility[0] / self.volatility_ma[0] if self.volatility_ma[0] > 0 else 1.0
        return 0.7 <= volatility_ratio <= 1.5
    
    def check_time_filter(self):
        """Vérification du filtre temporel"""
        if not self.p.time_filter:
            return True
        
        # Éviter les signaux trop rapprochés
        current_bar = len(self.data)
        if self.last_signal_time is not None:
            bars_since_last = current_bar - self.last_signal_time
            if bars_since_last < self.min_signal_interval:
                return False
        
        return True
    
    def calculate_signal_strength(self, long_signal, short_signal):
        """Calcul de la force du signal"""
        strength = 0.0
        
        if long_signal:
            # Force basée sur l'alignement des indicateurs
            if self.trend == 1:
                strength += 0.3
            if self.rsi[0] < 50:
                strength += 0.2
            if self.macd.macd[0] > self.macd.signal[0]:
                strength += 0.2
            if self.data.volume[0] > self.volume_ma[0]:
                strength += 0.2
            if self.momentum[0] > 0:
                strength += 0.1
                
        elif short_signal:
            # Force basée sur l'alignement des indicateurs
            if self.trend == -1:
                strength += 0.3
            if self.rsi[0] > 50:
                strength += 0.2
            if self.macd.macd[0] < self.macd.signal[0]:
                strength += 0.2
            if self.data.volume[0] > self.volume_ma[0]:
                strength += 0.2
            if self.momentum[0] < 0:
                strength += 0.1
        
        return strength
    
    def check_signal_confirmation(self, long_signal, short_signal):
        """Vérification de la confirmation des signaux"""
        if not self.p.signal_confirmation:
            return long_signal, short_signal
        
        # Calculer la force du signal
        signal_strength = self.calculate_signal_strength(long_signal, short_signal)
        
        # Exiger une force minimale
        min_strength = 0.6
        
        confirmed_long = long_signal and signal_strength >= min_strength
        confirmed_short = short_signal and signal_strength >= min_strength
        
        return confirmed_long, confirmed_short
    
    def check_balanced_rules(self):
        """Vérification des règles équilibrées"""
        current_value = self.broker.getvalue()
        dt = self.datas[0].datetime.datetime(0)
        
        # Reset daily tracking
        if self.current_date != dt.date():
            self.current_date = dt.date()
            self.daily_start_cash = current_value
        
        # Update peak
        self.peak_value = max(self.peak_value, current_value)
        
        # Check consecutive losses
        if self.consecutive_losses >= self.p.max_consecutive_losses:
            self.log(f'⚠️  MAX CONSECUTIVE LOSSES HIT: {self.consecutive_losses}')
            return False
        
        # Check daily loss limit
        if self.initial_cash == 0:
            daily_pnl = 0.0
        else:
            daily_pnl = (current_value - self.daily_start_cash) / self.initial_cash
        if daily_pnl < -self.p.max_daily_loss:
            self.log(f'⚠️  DAILY LOSS LIMIT HIT: {daily_pnl*100:.2f}%')
            return False
        
        # Check total drawdown
        if self.peak_value == 0:
            total_dd = 0.0
        else:
            total_dd = (self.peak_value - current_value) / self.peak_value
        if total_dd > self.p.max_total_loss:
            self.log(f'⚠️  TOTAL DRAWDOWN LIMIT HIT: {total_dd*100:.2f}%')
            return False
        
        # Check profit target
        if self.initial_cash == 0:
            total_profit = 0.0
        else:
            total_profit = (current_value - self.initial_cash) / self.initial_cash
        if total_profit >= self.p.profit_target:
            self.log(f'🎯 PROFIT TARGET REACHED: {total_profit*100:.2f}%')
            return False
        
        return True
    
    def is_trading_session(self):
        """Vérification session de trading équilibrée"""
        dt = self.datas[0].datetime.datetime(0)
        current_time = dt.time()
        return self.p.trading_start <= current_time <= self.p.trading_end
    
    def calculate_balanced_position_size(self):
        """Calcul de la taille de position équilibrée"""
        account_value = self.broker.getvalue()
        
        # Taille de base équilibrée
        base_size = self.p.position_size
        
        # Ajustement basé sur la performance récente (modéré)
        if self.consecutive_wins >= 3:
            base_size *= 1.1  # Augmentation modérée
        elif self.consecutive_losses >= 2:
            base_size *= 0.9  # Réduction modérée
        
        risk_amount = account_value * base_size
        
        # Utiliser Williams Stop pour calculer la taille
        if self.williams_long_stop:
            stop_distance = abs(self.data.close[0] - self.williams_long_stop)
        elif self.williams_short_stop:
            stop_distance = abs(self.data.close[0] - self.williams_short_stop)
        else:
            stop_distance = self.data.close[0] * 0.01  # 1% par défaut
        
        if stop_distance > 0:
            size = risk_amount / stop_distance
            # Limite raisonnable
            if self.data.close[0] > 0:
                max_size = account_value * 0.15 / self.data.close[0]  # Maximum 15% du capital
                size = min(size, max_size)
            return max(1, int(size))
        return 1
    
    def next(self):
        # Vérification des règles équilibrées
        if not self.check_balanced_rules():
            if self.position:
                self.close()
            return
        
        # Vérification session de trading
        if not self.is_trading_session():
            return
        
        # Mise à jour des indicateurs
        self.update_supertrend()
        self.update_williams_stops()
        
        # Calcul des signaux équilibrés
        
        # MACD Crossovers (équilibrés)
        crossmacdbear = (self.macd.macd[0] > self.macd.signal[0] and 
                        self.macd.macd[-1] <= self.macd.signal[-1])
        crossmacd = (self.macd.macd[0] < self.macd.signal[0] and 
                    self.macd.macd[-1] >= self.macd.signal[-1])
        
        # P1 Conditions (équilibrées)
        b1_ge_p1 = self.b1[0] >= self.p1[0] * 1.05  # 5% de marge
        b1_le_p1 = self.b1[0] <= self.p1[0] * 0.95  # 5% de marge
        
        # StopGap Filter
        stopgap_ok = self.calculate_stopgap_filter()
        
        # Filtres équilibrés
        trend_up, trend_down = self.check_trend_filter()
        volatility_ok = self.check_volatility_filter()
        time_ok = self.check_time_filter()
        
        # === CONDITIONS ÉQUILIBRÉES ===
        
        long_condition = (
            self.williams_long_active and 
            crossmacdbear and 
            b1_ge_p1 and 
            self.trend == 1 and 
            stopgap_ok and
            trend_up and
            volatility_ok and
            time_ok and
            self.rsi[0] < 60  # Éviter les zones de surachat
        )
        
        short_condition = (
            self.williams_short_active and 
            crossmacd and 
            b1_le_p1 and 
            self.trend == -1 and 
            stopgap_ok and
            trend_down and
            volatility_ok and
            time_ok and
            self.rsi[0] > 40  # Éviter les zones de survente
        )
        
        # Confirmation des signaux
        long_condition, short_condition = self.check_signal_confirmation(long_condition, short_condition)
        
        # === EXÉCUTION DES TRADES ÉQUILIBRÉS ===
        
        if self.order:
            return
        
        if not self.position:
            if long_condition:
                self.total_signals += 1
                self.last_signal_time = len(self.data)
                size = self.calculate_balanced_position_size()
                self.log(f'🚀 BALANCED BUY @ {self.data.close[0]:.5f} (Size: {size}, RSI: {self.rsi[0]:.1f})')
                self.order = self.buy(size=size)
                self.buy_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
                
            elif short_condition:
                self.total_signals += 1
                self.last_signal_time = len(self.data)
                size = self.calculate_balanced_position_size()
                self.log(f'🚀 BALANCED SELL @ {self.data.close[0]:.5f} (Size: {size}, RSI: {self.rsi[0]:.1f})')
                self.order = self.sell(size=size)
                self.sell_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
        
        else:  # Position ouverte - gestion équilibrée
            if self.position.size > 0:  # Position longue
                # Quick exit sur profit
                if (self.position_entry_price and 
                    self.data.close[0] >= self.position_entry_price * (1 + self.p.quick_exit_profit)):
                    self.log(f'💰 PROFIT EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Exit sur RSI extrême
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
                # Sortie sur changement de tendance
                elif self.trend == -1:
                    self.log(f'📉 TREND CHANGE EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                    
            elif self.position.size < 0:  # Position courte
                # Quick exit sur profit
                if (self.position_entry_price and 
                    self.data.close[0] <= self.position_entry_price * (1 - self.p.quick_exit_profit)):
                    self.log(f'💰 PROFIT EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Exit sur RSI extrême
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
                # Sortie sur changement de tendance
                elif self.trend == 1:
                    self.log(f'📈 TREND CHANGE EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
    
    def stop(self):
        """Statistiques finales équilibrées"""
        final_value = self.broker.getvalue()
        
        # Protection contre division par zéro
        if self.initial_cash == 0:
            total_return = 0.0
        else:
            total_return = (final_value - self.initial_cash) / self.initial_cash * 100
            
        if self.peak_value == 0:
            max_dd = 0.0
        else:
            max_dd = (self.peak_value - final_value) / self.peak_value * 100
        
        self.log(f'=== RÉSULTATS MASTERTREND ÉQUILIBRÉ ===')
        self.log(f'💰 Capital Initial: ${self.initial_cash:.2f}')
        self.log(f'💰 Capital Final: ${final_value:.2f}')
        self.log(f'📈 Rendement Total: {total_return:.2f}%')
        self.log(f'📉 Drawdown Max: {max_dd:.2f}%')
        self.log(f'🎯 Signaux Générés: {self.total_signals}')
        self.log(f'🟢 Signaux BUY: {len(self.buy_signals)}')
        self.log(f'🔴 Signaux SELL: {len(self.sell_signals)}')
        self.log(f'📊 Trades Total: {len(self.trade_history)}')
        self.log(f'💸 Pertes Consécutives Max: {self.consecutive_losses}')
        
        # Analyse des trades avec protection contre division par zéro
        if self.trade_history and len(self.trade_history) > 0:
            profitable_trades = [t for t in self.trade_history if t['profit'] > 0]
            losing_trades = [t for t in self.trade_history if t['profit'] < 0]
            
            win_rate = len(profitable_trades) / len(self.trade_history) * 100
            avg_profit = sum(t['profit'] for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0
            avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            avg_roi = sum(t['roi'] for t in self.trade_history) / len(self.trade_history)
            
            self.log(f'🎯 Taux de Réussite: {win_rate:.1f}%')
            self.log(f'💰 Profit Moyen: ${avg_profit:.2f}')
            self.log(f'💸 Perte Moyenne: ${avg_loss:.2f}')
            self.log(f'📊 ROI Moyen: {avg_roi:.2f}%')
            if avg_loss != 0:
                profit_factor = abs(avg_profit / avg_loss)
                self.log(f'⚡ Profit Factor: {profit_factor:.2f}')
        
        # Évaluation finale
        if total_return >= 10:
            self.log('🏆 EXCELLENT! Objectif de profit atteint!')
        elif total_return >= 5:
            self.log('✅ BON! Profit satisfaisant')
        elif total_return >= 2:
            self.log('⚠️  MOYEN. Peut mieux faire')
        else:
            self.log('❌ INSUFFISANT. Stratégie à revoir')


if __name__ == '__main__':
    # Configuration équilibrée
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
    cerebro.addstrategy(BalancedMasterTrend)
    
    # Configuration équilibrée
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)
    
    # Analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print('🚀 MASTERTREND BALANCED STRATEGY - DÉMARRAGE')
    print('💰 Objectif: 10-20% de rendement avec bon ratio risque/récompense')
    print('⚖️  Mode: ÉQUILIBRÉ')
    print('🎯 Capital Initial: $10,000')
    print('=' * 50)
    
    try:
        results = cerebro.run()
        strat = results[0]
        
        # Analyse détaillée des résultats
        trades = strat.analyzers.trades.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        
        print('\n🔥 ANALYSE MASTERTREND BALANCED:')
        if hasattr(trades, 'total') and trades.total.closed > 0:
            print(f'📊 Total Trades: {trades.total.closed}')
            print(f'🏆 Trades Gagnants: {trades.won.total if hasattr(trades, "won") else 0}')
            print(f'💸 Trades Perdants: {trades.lost.total if hasattr(trades, "lost") else 0}')
            if hasattr(trades, 'won') and trades.won.total > 0 and trades.total.closed > 0:
                win_rate = trades.won.total / trades.total.closed * 100
                print(f'🎯 Taux de Réussite: {win_rate:.1f}%')
                
                if hasattr(trades.won, 'pnl') and hasattr(trades.lost, 'pnl'):
                    avg_win = trades.won.pnl.average if hasattr(trades.won.pnl, 'average') else 0
                    avg_loss = trades.lost.pnl.average if hasattr(trades.lost.pnl, 'average') else 0
                    if avg_loss != 0 and avg_loss is not None:
                        profit_factor = abs(avg_win / avg_loss)
                        print(f'⚡ Profit Factor: {profit_factor:.2f}')
        
        if hasattr(drawdown, 'max'):
            print(f'📉 Drawdown Maximum: {drawdown.max.drawdown:.2f}%')
            
        if hasattr(returns, 'rtot'):
            print(f'📈 Rendement Total: {returns.rtot*100:.2f}%')
            
        # Évaluation finale
        final_return = returns.rtot*100 if hasattr(returns, 'rtot') else 0
        if final_return >= 10:
            print('\n🏆 MISSION ACCOMPLIE! MasterTrend Balanced a réussi!')
        elif final_return >= 5:
            print('\n✅ Bon résultat! Objectif partiellement atteint')
        else:
            print('\n⚠️  Résultat insuffisant. Optimisation nécessaire.')
            
    except Exception as e:
        print(f'❌ ERREUR LORS DE L\'EXÉCUTION: {e}')
        print('Vérifiez que le fichier EURUSD_data_15M.csv existe et a le bon format.') 