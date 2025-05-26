#!/usr/bin/env python3
"""
MASTERTREND VISUAL STRATEGY
Version avec visualisation graphique complète
Objectif: Analyser visuellement les signaux et la performance
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


class VisualMasterTrend(bt.Strategy):
    """
    Stratégie MasterTrend avec visualisation graphique
    """
    
    params = (
        # SuperTrend optimisé
        ('st_period', 10),
        ('st_multiplier', 2.5),
        
        # MACD optimisé
        ('macd_fast', 10),
        ('macd_slow', 22),
        ('macd_signal', 8),
        
        # Williams Fractals optimisés
        ('williams_left', 3),
        ('williams_right', 3),
        ('williams_buffer', 0.08),
        
        # P1 Indicator optimisé
        ('p1_e31', 5),
        ('p1_m', 9),
        ('p1_l31', 14),
        
        # StopGap Filter optimisé
        ('sg_ema_short', 8),
        ('sg_ema_long', 18),
        ('sg_lookback', 5),
        ('sg_median_period', 10),
        
        # Risk Management
        ('max_daily_loss', 0.05),
        ('max_total_loss', 0.12),
        ('profit_target', 0.20),
        ('position_size_base', 0.03),
        ('max_consecutive_losses', 4),
        
        # Trading Sessions
        ('trading_start', datetime.time(7, 0)),
        ('trading_end', datetime.time(18, 0)),
        
        # Features
        ('adaptive_mode', True),
        ('market_regime_detection', True),
        ('min_signal_strength', 0.5),
        ('min_bars_between_signals', 4),
        ('use_regime_exits', True),
        ('require_volume_confirmation', False),
        ('use_quick_exits', True),
        ('use_trailing_stops', True),
        
        # Visualisation
        ('save_plots', True),
        ('plot_signals', True),
        ('plot_indicators', True),
    )
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # === INDICATEURS OPTIMISÉS ===
        
        # Prix
        self.hl2 = (self.data.high + self.data.low) / 2.0
        
        # ATR pour volatilité
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.atr_ma = bt.indicators.SMA(self.atr, period=20)
        
        # Moyennes mobiles
        self.sma_20 = bt.indicators.SMA(self.data.close, period=20)
        self.sma_50 = bt.indicators.SMA(self.data.close, period=50)
        self.ema_9 = bt.indicators.EMA(self.data.close, period=9)
        self.ema_21 = bt.indicators.EMA(self.data.close, period=21)
        
        # MACD
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        
        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.rsi_fast = bt.indicators.RSI(self.data.close, period=7)
        
        # Volume
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)
        
        # Bollinger Bands
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2.0)
        
        # Volatilité
        self.volatility = bt.indicators.StdDev(self.data.close, period=20)
        
        # Momentum
        self.momentum = bt.indicators.Momentum(self.data.close, period=10)
        
        # Williams %R
        self.williams_r = bt.indicators.WilliamsR(self.data, period=14)
        
        # === VARIABLES D'ÉTAT ===
        
        # SuperTrend
        self.trend = 1
        self.final_up = 0.0
        self.final_dn = 0.0
        
        # Williams Stops
        self.williams_long_stop = None
        self.williams_short_stop = None
        self.williams_long_active = False
        self.williams_short_active = False
        
        # P1 Indicator
        self.hilow = (self.data.high - self.data.low) * 100
        self.openclose = (self.data.close - self.data.open) * 100
        self.spreadv = self.openclose * self.data.close
        self.pt_approx = bt.indicators.SMA(self.spreadv, period=50)
        
        # EMAs P1
        self.ema_e31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_e31)
        self.ema_m = bt.indicators.EMA(self.pt_approx, period=self.p.p1_m)
        self.ema_l31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_l31)
        
        self.a1 = self.ema_l31 - self.ema_m
        self.b1 = self.ema_e31 - self.ema_m
        self.p1 = self.a1 + self.b1
        
        # StopGap
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_long)
        self.highest_sg = bt.indicators.Highest(self.data.high, period=self.p.sg_lookback)
        self.lowest_sg = bt.indicators.Lowest(self.data.low, period=self.p.sg_lookback)
        
        # === DÉTECTION DE RÉGIME ===
        self.market_regime = "NEUTRAL"
        self.regime_confidence = 0.0
        
        # === VARIABLES DE TRADING ===
        self.initial_cash = self.broker.getvalue()
        self.daily_start_cash = self.initial_cash
        self.peak_value = self.initial_cash
        self.current_date = None
        
        # Trade Management
        self.order = None
        self.position_entry_price = None
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Statistiques pour graphiques
        self.buy_signals = []
        self.sell_signals = []
        self.trade_history = []
        self.total_signals = 0
        self.equity_curve = []
        self.regime_history = []
        self.signal_strength_history = []
        
        # Adaptation
        self.position_size = self.p.position_size_base
        self.last_signal_time = None
        self.signal_strength = 0.0
        
        # StopGap median calculation
        self.candle_sizes = deque(maxlen=self.p.sg_median_period)
        
        # Performance tracking
        self.performance_by_regime = {
            "TRENDING_UP": {"trades": 0, "profit": 0.0},
            "TRENDING_DOWN": {"trades": 0, "profit": 0.0},
            "RANGING": {"trades": 0, "profit": 0.0},
            "NEUTRAL": {"trades": 0, "profit": 0.0}
        }
        
        # Trailing stops
        self.trailing_stop_long = None
        self.trailing_stop_short = None
        
        # Données pour graphiques
        self.price_data = []
        self.datetime_data = []
        self.indicator_data = {
            'sma_20': [],
            'sma_50': [],
            'ema_9': [],
            'ema_21': [],
            'bb_upper': [],
            'bb_lower': [],
            'bb_middle': [],
            'rsi': [],
            'macd': [],
            'macd_signal': [],
            'macd_hist': [],
            'volume': [],
            'atr': []
        }
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🟢 BUY EXECUTED @ {order.executed.price:.5f} [Regime: {self.market_regime}, Strength: {self.signal_strength:.2f}]')
                self.position_entry_price = order.executed.price
                if self.p.use_trailing_stops:
                    self.trailing_stop_long = order.executed.price * 0.99
            else:
                self.log(f'🔴 SELL EXECUTED @ {order.executed.price:.5f} [Regime: {self.market_regime}, Strength: {self.signal_strength:.2f}]')
                self.position_entry_price = order.executed.price
                if self.p.use_trailing_stops:
                    self.trailing_stop_short = order.executed.price * 1.01
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('❌ Order Canceled/Margin/Rejected')
            
        self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            # Reset trailing stops
            self.trailing_stop_long = None
            self.trailing_stop_short = None
            
            # Enregistrer performance par régime
            if self.market_regime in self.performance_by_regime:
                self.performance_by_regime[self.market_regime]["trades"] += 1
                self.performance_by_regime[self.market_regime]["profit"] += trade.pnl
            
            if trade.pnl > 0:
                self.consecutive_losses = 0
                self.consecutive_wins += 1
                roi = trade.pnl/abs(trade.value)*100 if trade.value != 0 else 0
                self.log(f'💰 PROFIT: ${trade.pnl:.2f} (ROI: {roi:.1f}%) [Regime: {self.market_regime}, Streak: {self.consecutive_wins}]')
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.log(f'💸 LOSS: ${trade.pnl:.2f} [Regime: {self.market_regime}, Consecutive: {self.consecutive_losses}]')
            
            # Enregistrer le trade
            self.trade_history.append({
                'datetime': self.datas[0].datetime.datetime(0),
                'profit': trade.pnl,
                'duration': trade.barlen,
                'entry_price': self.position_entry_price,
                'exit_price': self.data.close[0],
                'roi': trade.pnl/abs(trade.value)*100 if trade.value != 0 else 0,
                'regime': self.market_regime,
                'signal_strength': self.signal_strength,
                'type': 'LONG' if trade.size > 0 else 'SHORT'
            })
    
    def safe_divide(self, numerator, denominator, default=0.0):
        """Division sécurisée"""
        try:
            if denominator == 0 or denominator is None or np.isnan(denominator) or np.isinf(denominator):
                return default
            result = numerator / denominator
            if np.isnan(result) or np.isinf(result):
                return default
            return result
        except:
            return default
    
    def detect_market_regime(self):
        """Détection optimisée du régime de marché"""
        if len(self.data) < 50:
            return
        
        try:
            # Analyse de tendance
            price_vs_sma20 = self.safe_divide(
                (self.data.close[0] - self.sma_20[0]), 
                self.sma_20[0], 
                0.0
            ) * 100
            
            # Alignement des EMAs
            ema_bullish = self.ema_9[0] > self.ema_21[0]
            ema_bearish = self.ema_9[0] < self.ema_21[0]
            
            # Momentum
            momentum_positive = self.momentum[0] > 0
            momentum_negative = self.momentum[0] < 0
            
            # RSI pour range detection
            rsi_middle = 40 <= self.rsi[0] <= 60
            
            # Volatilité
            volatility_high = self.volatility[0] > np.mean([self.volatility[-i] for i in range(1, min(21, len(self.data)))])
            
            # Détermination du régime
            if ema_bullish and momentum_positive and price_vs_sma20 > 0.5:
                self.market_regime = "TRENDING_UP"
                self.regime_confidence = 0.8
            elif ema_bearish and momentum_negative and price_vs_sma20 < -0.5:
                self.market_regime = "TRENDING_DOWN"
                self.regime_confidence = 0.8
            elif rsi_middle and not volatility_high:
                self.market_regime = "RANGING"
                self.regime_confidence = 0.6
            else:
                self.market_regime = "NEUTRAL"
                self.regime_confidence = 0.5
                
        except Exception as e:
            self.log(f"Erreur détection régime: {e}")
            self.market_regime = "NEUTRAL"
            self.regime_confidence = 0.5
    
    def calculate_signal_strength(self, long_signal, short_signal):
        """Calcul optimisé de la force du signal"""
        strength = 0.0
        
        try:
            if long_signal:
                if self.trend == 1:
                    strength += 0.15
                if self.macd.macd[0] > self.macd.signal[0]:
                    strength += 0.15
                if self.rsi[0] < 65:
                    strength += 0.10
                
                volume_ratio = self.safe_divide(self.data.volume[0], self.volume_ma[0], 1.0)
                if volume_ratio > 1.1:
                    strength += 0.20
                
                if self.williams_r[0] < -60:
                    strength += 0.10
                bb_range = self.bb.top[0] - self.bb.bot[0]
                if bb_range > 0:
                    bb_position = self.safe_divide(
                        (self.data.close[0] - self.bb.bot[0]), 
                        bb_range, 
                        0.5
                    ) * 100
                    if bb_position < 40:
                        strength += 0.10
                
                if self.momentum[0] > 0:
                    strength += 0.10
                if self.rsi_fast[0] > self.rsi_fast[-1]:
                    strength += 0.10
                    
            elif short_signal:
                if self.trend == -1:
                    strength += 0.15
                if self.macd.macd[0] < self.macd.signal[0]:
                    strength += 0.15
                if self.rsi[0] > 35:
                    strength += 0.10
                
                volume_ratio = self.safe_divide(self.data.volume[0], self.volume_ma[0], 1.0)
                if volume_ratio > 1.1:
                    strength += 0.20
                
                if self.williams_r[0] > -40:
                    strength += 0.10
                bb_range = self.bb.top[0] - self.bb.bot[0]
                if bb_range > 0:
                    bb_position = self.safe_divide(
                        (self.data.close[0] - self.bb.bot[0]), 
                        bb_range, 
                        0.5
                    ) * 100
                    if bb_position > 60:
                        strength += 0.10
                
                if self.momentum[0] < 0:
                    strength += 0.10
                if self.rsi_fast[0] < self.rsi_fast[-1]:
                    strength += 0.10
            
            if self.market_regime in ["TRENDING_UP", "TRENDING_DOWN"]:
                strength += 0.1 * self.regime_confidence
            
            return min(1.0, strength)
            
        except Exception as e:
            self.log(f"Erreur calcul signal strength: {e}")
            return 0.0
    
    def get_regime_specific_signals(self):
        """Signaux spécifiques au régime optimisés"""
        long_signals = []
        short_signals = []
        
        try:
            if self.market_regime == "TRENDING_UP":
                if (self.data.close[0] > self.ema_21[0] and 
                    self.rsi[0] < 55 and self.rsi[-1] >= 55):
                    long_signals.append("TREND_PULLBACK")
                
                if (self.data.close[0] > self.data.high[-1] and
                    self.volume_ma[0] > 0 and
                    self.data.volume[0] > self.volume_ma[0] * 1.2):
                    long_signals.append("TREND_BREAKOUT")
            
            elif self.market_regime == "TRENDING_DOWN":
                if (self.data.close[0] < self.ema_21[0] and 
                    self.rsi[0] > 45 and self.rsi[-1] <= 45):
                    short_signals.append("TREND_PULLBACK")
                
                if (self.data.close[0] < self.data.low[-1] and
                    self.volume_ma[0] > 0 and
                    self.data.volume[0] > self.volume_ma[0] * 1.2):
                    short_signals.append("TREND_BREAKDOWN")
            
            elif self.market_regime == "RANGING":
                bb_range = self.bb.top[0] - self.bb.bot[0]
                if bb_range > 0:
                    bb_position = self.safe_divide(
                        (self.data.close[0] - self.bb.bot[0]), 
                        bb_range, 
                        0.5
                    ) * 100
                    
                    if bb_position < 30 and self.rsi[0] < 40:
                        long_signals.append("RANGE_REVERSAL")
                    elif bb_position > 70 and self.rsi[0] > 60:
                        short_signals.append("RANGE_REVERSAL")
            
            if (self.data.close[0] < self.data.close[-5] and 
                self.rsi[0] > self.rsi[-5] and 
                self.rsi[0] < 35):
                long_signals.append("RSI_DIVERGENCE")
            elif (self.data.close[0] > self.data.close[-5] and 
                  self.rsi[0] < self.rsi[-5] and 
                  self.rsi[0] > 65):
                short_signals.append("RSI_DIVERGENCE")
                        
        except Exception as e:
            self.log(f"Erreur signaux régime: {e}")
        
        return long_signals, short_signals
    
    def check_filters(self):
        """Vérification des filtres optimisés"""
        try:
            current_bar = len(self.data)
            if self.last_signal_time is not None:
                bars_since_last = current_bar - self.last_signal_time
                if bars_since_last < self.p.min_bars_between_signals:
                    return False
            
            if self.consecutive_losses >= 3:
                return False
            
            return True
        except:
            return False
    
    def update_supertrend(self):
        """Mise à jour SuperTrend optimisée"""
        if len(self.data) < 2:
            return
        
        try:
            basic_up = self.hl2[0] - self.p.st_multiplier * self.atr[0]
            basic_dn = self.hl2[0] + self.p.st_multiplier * self.atr[0]
            
            if self.data.close[-1] > self.final_up:
                self.final_up = max(basic_up, self.final_up)
            else:
                self.final_up = basic_up
                
            if self.data.close[-1] < self.final_dn:
                self.final_dn = min(basic_dn, self.final_dn)
            else:
                self.final_dn = basic_dn
                
            if self.trend == -1 and self.data.close[0] > self.final_dn:
                self.trend = 1
            elif self.trend == 1 and self.data.close[0] < self.final_up:
                self.trend = -1
        except Exception as e:
            self.log(f"Erreur SuperTrend: {e}")
    
    def update_williams_stops(self):
        """Mise à jour Williams Stops optimisée"""
        try:
            if len(self.data) > self.p.williams_left + self.p.williams_right:
                center_idx = self.p.williams_right
                
                is_high_fractal = True
                center_high = self.data.high[-center_idx]
                
                for i in range(self.p.williams_left + self.p.williams_right + 1):
                    if i != center_idx and self.data.high[-i] >= center_high:
                        is_high_fractal = False
                        break
                
                if is_high_fractal:
                    self.williams_short_stop = center_high * (1 + self.p.williams_buffer / 100)
                    self.williams_short_active = True
                
                is_low_fractal = True
                center_low = self.data.low[-center_idx]
                
                for i in range(self.p.williams_left + self.p.williams_right + 1):
                    if i != center_idx and self.data.low[-i] <= center_low:
                        is_low_fractal = False
                        break
                
                if is_low_fractal:
                    self.williams_long_stop = center_low * (1 - self.p.williams_buffer / 100)
                    self.williams_long_active = True
        except Exception as e:
            self.log(f"Erreur Williams Stops: {e}")
    
    def calculate_stopgap_filter(self):
        """Calcul du filtre StopGap optimisé"""
        try:
            trend_up = self.ema_short[0] > self.ema_long[0]
            trend_down = self.ema_short[0] < self.ema_long[0]
            
            candle_size = abs(self.data.high[0] - self.data.low[0])
            self.candle_sizes.append(candle_size)
            
            if len(self.candle_sizes) >= 5:
                median = np.median(list(self.candle_sizes))
            else:
                median = candle_size
            
            stop_gap = 0.0
            if trend_down and self.data.high[0] < self.highest_sg[-1]:
                stop_gap = abs(self.highest_sg[-1] - self.data.low[0])
            elif trend_up and self.data.low[0] > self.lowest_sg[-1]:
                stop_gap = abs(self.data.high[0] - self.lowest_sg[-1])
            
            multiplier = 1.0
            if self.market_regime == "VOLATILE":
                multiplier = 0.8
            elif self.market_regime == "RANGING":
                multiplier = 1.2
            
            return stop_gap > (median * multiplier) if median > 0 else True
        except Exception as e:
            self.log(f"Erreur StopGap: {e}")
            return True
    
    def update_trailing_stops(self):
        """Mise à jour des trailing stops"""
        if not self.p.use_trailing_stops:
            return
            
        try:
            if self.position.size > 0 and self.trailing_stop_long is not None:
                new_stop = self.data.close[0] * 0.99
                if new_stop > self.trailing_stop_long:
                    self.trailing_stop_long = new_stop
                    
            elif self.position.size < 0 and self.trailing_stop_short is not None:
                new_stop = self.data.close[0] * 1.01
                if new_stop < self.trailing_stop_short:
                    self.trailing_stop_short = new_stop
        except:
            pass
    
    def check_rules(self):
        """Vérification des règles de trading optimisées"""
        try:
            current_value = self.broker.getvalue()
            dt = self.datas[0].datetime.datetime(0)
            
            if self.current_date != dt.date():
                self.current_date = dt.date()
                self.daily_start_cash = current_value
            
            self.peak_value = max(self.peak_value, current_value)
            
            if self.consecutive_losses >= self.p.max_consecutive_losses:
                self.log(f'⚠️  MAX CONSECUTIVE LOSSES: {self.consecutive_losses}')
                return False
            
            if self.initial_cash > 0:
                daily_pnl = self.safe_divide(
                    (current_value - self.daily_start_cash), 
                    self.initial_cash, 
                    0.0
                )
                if daily_pnl < -self.p.max_daily_loss:
                    self.log(f'⚠️  DAILY LOSS LIMIT: {daily_pnl*100:.2f}%')
                    return False
            
            if self.peak_value > 0:
                total_dd = self.safe_divide(
                    (self.peak_value - current_value), 
                    self.peak_value, 
                    0.0
                )
                if total_dd > self.p.max_total_loss:
                    self.log(f'⚠️  TOTAL DRAWDOWN LIMIT: {total_dd*100:.2f}%')
                    return False
            
            if self.initial_cash > 0:
                total_profit = self.safe_divide(
                    (current_value - self.initial_cash), 
                    self.initial_cash, 
                    0.0
                )
                if total_profit >= self.p.profit_target:
                    self.log(f'🎯 PROFIT TARGET REACHED: {total_profit*100:.2f}%')
                    return False
            
            return True
        except Exception as e:
            self.log(f"Erreur check rules: {e}")
            return False
    
    def is_trading_session(self):
        """Vérification session de trading"""
        try:
            dt = self.datas[0].datetime.datetime(0)
            current_time = dt.time()
            return self.p.trading_start <= current_time <= self.p.trading_end
        except:
            return True
    
    def calculate_position_size(self):
        """Calcul de la taille de position optimisée"""
        try:
            account_value = self.broker.getvalue()
            
            base_size = self.position_size
            
            if self.consecutive_wins >= 3:
                base_size *= 1.2
            elif self.consecutive_wins >= 2:
                base_size *= 1.1
            elif self.consecutive_losses >= 2:
                base_size *= 0.7
            elif self.consecutive_losses >= 1:
                base_size *= 0.85
            
            base_size *= (0.6 + 0.4 * self.regime_confidence)
            base_size *= (0.4 + 0.6 * self.signal_strength)
            
            if self.atr_ma[0] > 0:
                volatility_ratio = self.atr[0] / self.atr_ma[0]
                if volatility_ratio > 1.5:
                    base_size *= 0.8
                elif volatility_ratio < 0.8:
                    base_size *= 1.2
            
            risk_amount = account_value * base_size
            
            if self.williams_long_stop and self.data.close[0] > 0:
                stop_distance = abs(self.data.close[0] - self.williams_long_stop)
            elif self.williams_short_stop and self.data.close[0] > 0:
                stop_distance = abs(self.data.close[0] - self.williams_short_stop)
            else:
                stop_distance = self.data.close[0] * 0.008
            
            if stop_distance > 0 and self.data.close[0] > 0:
                size = self.safe_divide(risk_amount, stop_distance, 1.0)
                max_pct = 0.15 if self.market_regime in ["TRENDING_UP", "TRENDING_DOWN"] else 0.10
                max_size = self.safe_divide(account_value * max_pct, self.data.close[0], 1.0)
                size = min(size, max_size)
                return max(1, int(size))
            return 1
        except Exception as e:
            self.log(f"Erreur position size: {e}")
            return 1
    
    def collect_data_for_plots(self):
        """Collecter les données pour les graphiques"""
        try:
            # Données de prix et datetime
            self.datetime_data.append(self.datas[0].datetime.datetime(0))
            self.price_data.append({
                'open': self.data.open[0],
                'high': self.data.high[0],
                'low': self.data.low[0],
                'close': self.data.close[0]
            })
            
            # Equity curve
            self.equity_curve.append(self.broker.getvalue())
            
            # Régime et force du signal
            self.regime_history.append(self.market_regime)
            self.signal_strength_history.append(self.signal_strength)
            
            # Indicateurs
            self.indicator_data['sma_20'].append(self.sma_20[0])
            self.indicator_data['sma_50'].append(self.sma_50[0])
            self.indicator_data['ema_9'].append(self.ema_9[0])
            self.indicator_data['ema_21'].append(self.ema_21[0])
            self.indicator_data['bb_upper'].append(self.bb.top[0])
            self.indicator_data['bb_lower'].append(self.bb.bot[0])
            self.indicator_data['bb_middle'].append(self.bb.mid[0])
            self.indicator_data['rsi'].append(self.rsi[0])
            self.indicator_data['macd'].append(self.macd.macd[0])
            self.indicator_data['macd_signal'].append(self.macd.signal[0])
            self.indicator_data['macd_hist'].append(self.macd.histo[0])
            self.indicator_data['volume'].append(self.data.volume[0])
            self.indicator_data['atr'].append(self.atr[0])
            
        except Exception as e:
            self.log(f"Erreur collecte données: {e}")
    
    def next(self):
        try:
            # Collecter les données pour les graphiques
            self.collect_data_for_plots()
            
            # Détection du régime de marché
            self.detect_market_regime()
            
            # Vérification des règles
            if not self.check_rules():
                if self.position:
                    self.close()
                return
            
            # Vérification session de trading
            if not self.is_trading_session():
                return
            
            # Mise à jour des indicateurs
            self.update_supertrend()
            self.update_williams_stops()
            self.update_trailing_stops()
            
            # === SIGNAUX DE BASE ===
            
            crossmacdbear = (self.macd.macd[0] > self.macd.signal[0] and 
                            self.macd.macd[-1] <= self.macd.signal[-1])
            crossmacd = (self.macd.macd[0] < self.macd.signal[0] and 
                        self.macd.macd[-1] >= self.macd.signal[-1])
            
            p1_margin = 0.03
            b1_ge_p1 = self.b1[0] >= self.p1[0] * (1 + p1_margin) if self.p1[0] != 0 else False
            b1_le_p1 = self.b1[0] <= self.p1[0] * (1 - p1_margin) if self.p1[0] != 0 else False
            
            stopgap_ok = self.calculate_stopgap_filter()
            
            regime_long_signals, regime_short_signals = self.get_regime_specific_signals()
            
            # === CONDITIONS OPTIMISÉES ===
            
            base_long = (self.williams_long_active and crossmacdbear and b1_ge_p1 and 
                        self.trend == 1 and stopgap_ok)
            base_short = (self.williams_short_active and crossmacd and b1_le_p1 and 
                         self.trend == -1 and stopgap_ok)
            
            alt_long = (self.trend == 1 and self.rsi[0] < 35 and 
                       self.data.close[0] > self.ema_9[0])
            alt_short = (self.trend == -1 and self.rsi[0] > 65 and 
                        self.data.close[0] < self.ema_9[0])
            
            long_condition = base_long or alt_long or len(regime_long_signals) > 0
            short_condition = base_short or alt_short or len(regime_short_signals) > 0
            
            self.signal_strength = self.calculate_signal_strength(long_condition, short_condition)
            
            signal_quality_ok = self.signal_strength >= self.p.min_signal_strength
            filters_ok = self.check_filters()
            
            volume_ok = True
            if self.p.require_volume_confirmation:
                volume_ratio = self.safe_divide(self.data.volume[0], self.volume_ma[0], 1.0)
                volume_ok = volume_ratio > 1.05
            
            long_condition = long_condition and signal_quality_ok and filters_ok and volume_ok
            short_condition = short_condition and signal_quality_ok and filters_ok and volume_ok
            
            # === EXÉCUTION DES TRADES ===
            
            if self.order:
                return
            
            if not self.position:
                if long_condition:
                    self.total_signals += 1
                    self.last_signal_time = len(self.data)
                    size = self.calculate_position_size()
                    signal_type = regime_long_signals[0] if regime_long_signals else ("ALT" if alt_long else "BASE")
                    self.log(f'🚀 BUY @ {self.data.close[0]:.5f} [Regime: {self.market_regime}, Signal: {signal_type}, Strength: {self.signal_strength:.2f}]')
                    self.order = self.buy(size=size)
                    self.buy_signals.append({
                        'datetime': self.datas[0].datetime.datetime(0),
                        'price': self.data.close[0],
                        'signal_type': signal_type,
                        'strength': self.signal_strength,
                        'regime': self.market_regime
                    })
                    
                elif short_condition:
                    self.total_signals += 1
                    self.last_signal_time = len(self.data)
                    size = self.calculate_position_size()
                    signal_type = regime_short_signals[0] if regime_short_signals else ("ALT" if alt_short else "BASE")
                    self.log(f'🚀 SELL @ {self.data.close[0]:.5f} [Regime: {self.market_regime}, Signal: {signal_type}, Strength: {self.signal_strength:.2f}]')
                    self.order = self.sell(size=size)
                    self.sell_signals.append({
                        'datetime': self.datas[0].datetime.datetime(0),
                        'price': self.data.close[0],
                        'signal_type': signal_type,
                        'strength': self.signal_strength,
                        'regime': self.market_regime
                    })
            
            else:  # Position ouverte - gestion optimisée
                if self.position.size > 0:  # Position longue
                    if (self.p.use_quick_exits and self.position_entry_price and 
                        self.data.close[0] >= self.position_entry_price * 1.015):
                        self.log(f'💰 QUICK PROFIT EXIT @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                    elif (self.p.use_trailing_stops and self.trailing_stop_long and 
                          self.data.low[0] <= self.trailing_stop_long):
                        self.log(f'🛑 TRAILING STOP @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                    elif self.p.use_regime_exits and self.market_regime == "RANGING":
                        bb_range = self.bb.top[0] - self.bb.bot[0]
                        if bb_range > 0:
                            bb_position = self.safe_divide(
                                (self.data.close[0] - self.bb.bot[0]), 
                                bb_range, 
                                0.5
                            ) * 100
                            if bb_position > 70:
                                self.log(f'💰 RANGE EXIT @ {self.data.close[0]:.5f}')
                                self.order = self.close()
                    elif self.rsi[0] > 75:
                        self.log(f'📈 RSI EXIT @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                    elif (self.williams_long_stop and 
                          self.data.low[0] <= self.williams_long_stop):
                        self.log(f'🛑 LONG STOP @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                    elif crossmacd or self.trend == -1:
                        self.log(f'🔄 LONG EXIT @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                        
                elif self.position.size < 0:  # Position courte
                    if (self.p.use_quick_exits and self.position_entry_price and 
                        self.data.close[0] <= self.position_entry_price * 0.985):
                        self.log(f'💰 QUICK PROFIT EXIT @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                    elif (self.p.use_trailing_stops and self.trailing_stop_short and 
                          self.data.high[0] >= self.trailing_stop_short):
                        self.log(f'🛑 TRAILING STOP @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                    elif self.p.use_regime_exits and self.market_regime == "RANGING":
                        bb_range = self.bb.top[0] - self.bb.bot[0]
                        if bb_range > 0:
                            bb_position = self.safe_divide(
                                (self.data.close[0] - self.bb.bot[0]), 
                                bb_range, 
                                0.5
                            ) * 100
                            if bb_position < 30:
                                self.log(f'💰 RANGE EXIT @ {self.data.close[0]:.5f}')
                                self.order = self.close()
                    elif self.rsi[0] < 25:
                        self.log(f'📉 RSI EXIT @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                    elif (self.williams_short_stop and 
                          self.data.high[0] >= self.williams_short_stop):
                        self.log(f'🛑 SHORT STOP @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                    elif crossmacdbear or self.trend == 1:
                        self.log(f'🔄 SHORT EXIT @ {self.data.close[0]:.5f}')
                        self.order = self.close()
                        
        except Exception as e:
            self.log(f"Erreur dans next(): {e}")
    
    def create_plots(self, config_name):
        """Créer les graphiques de visualisation"""
        try:
            if not self.datetime_data or not self.price_data:
                self.log("Pas de données pour créer les graphiques")
                return
            
            # Configuration des graphiques
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(20, 16))
            
            # Couleurs
            colors = {
                'price': '#00ff88',
                'sma20': '#ff6b6b',
                'sma50': '#4ecdc4',
                'ema9': '#ffe66d',
                'ema21': '#ff8b94',
                'bb_upper': '#a8e6cf',
                'bb_lower': '#a8e6cf',
                'bb_middle': '#88d8c0',
                'buy': '#00ff00',
                'sell': '#ff0000',
                'equity': '#00aaff',
                'rsi': '#ffaa00',
                'macd': '#aa00ff',
                'volume': '#666666'
            }
            
            # 1. Graphique principal - Prix et indicateurs
            ax1 = plt.subplot(4, 2, (1, 2))
            
            # Prix
            closes = [p['close'] for p in self.price_data]
            ax1.plot(self.datetime_data, closes, color=colors['price'], linewidth=2, label='Prix', alpha=0.9)
            
            # Moyennes mobiles
            if len(self.indicator_data['sma_20']) == len(self.datetime_data):
                ax1.plot(self.datetime_data, self.indicator_data['sma_20'], color=colors['sma20'], linewidth=1.5, label='SMA 20', alpha=0.7)
                ax1.plot(self.datetime_data, self.indicator_data['sma_50'], color=colors['sma50'], linewidth=1.5, label='SMA 50', alpha=0.7)
                ax1.plot(self.datetime_data, self.indicator_data['ema_9'], color=colors['ema9'], linewidth=1, label='EMA 9', alpha=0.6)
                ax1.plot(self.datetime_data, self.indicator_data['ema_21'], color=colors['ema21'], linewidth=1, label='EMA 21', alpha=0.6)
            
            # Bollinger Bands
            if len(self.indicator_data['bb_upper']) == len(self.datetime_data):
                ax1.plot(self.datetime_data, self.indicator_data['bb_upper'], color=colors['bb_upper'], linewidth=1, alpha=0.5)
                ax1.plot(self.datetime_data, self.indicator_data['bb_lower'], color=colors['bb_lower'], linewidth=1, alpha=0.5)
                ax1.fill_between(self.datetime_data, self.indicator_data['bb_upper'], self.indicator_data['bb_lower'], 
                               color=colors['bb_upper'], alpha=0.1)
            
            # Signaux d'achat et de vente
            for signal in self.buy_signals:
                ax1.scatter(signal['datetime'], signal['price'], color=colors['buy'], s=100, marker='^', 
                          label='Signal BUY' if signal == self.buy_signals[0] else "", zorder=5)
                ax1.annotate(f"BUY\n{signal['signal_type']}\nS:{signal['strength']:.2f}", 
                           (signal['datetime'], signal['price']), 
                           xytext=(10, 20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['buy'], alpha=0.7),
                           fontsize=8, color='black')
            
            for signal in self.sell_signals:
                ax1.scatter(signal['datetime'], signal['price'], color=colors['sell'], s=100, marker='v', 
                          label='Signal SELL' if signal == self.sell_signals[0] else "", zorder=5)
                ax1.annotate(f"SELL\n{signal['signal_type']}\nS:{signal['strength']:.2f}", 
                           (signal['datetime'], signal['price']), 
                           xytext=(10, -30), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['sell'], alpha=0.7),
                           fontsize=8, color='white')
            
            ax1.set_title(f'MasterTrend Visual - {config_name} - Prix et Signaux', fontsize=16, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 2. Equity Curve
            ax2 = plt.subplot(4, 2, 3)
            if self.equity_curve:
                returns = [(eq - self.initial_cash) / self.initial_cash * 100 for eq in self.equity_curve]
                ax2.plot(self.datetime_data[:len(returns)], returns, color=colors['equity'], linewidth=2)
                ax2.axhline(y=0, color='white', linestyle='--', alpha=0.5)
                ax2.set_title('Courbe d\'Équité (%)', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylabel('Rendement (%)')
            
            # 3. RSI
            ax3 = plt.subplot(4, 2, 4)
            if len(self.indicator_data['rsi']) == len(self.datetime_data):
                ax3.plot(self.datetime_data, self.indicator_data['rsi'], color=colors['rsi'], linewidth=2)
                ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Surachat')
                ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Survente')
                ax3.axhline(y=50, color='white', linestyle='-', alpha=0.3)
                ax3.set_title('RSI (14)', fontsize=12, fontweight='bold')
                ax3.set_ylim(0, 100)
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)
            
            # 4. MACD
            ax4 = plt.subplot(4, 2, 5)
            if len(self.indicator_data['macd']) == len(self.datetime_data):
                ax4.plot(self.datetime_data, self.indicator_data['macd'], color=colors['macd'], linewidth=2, label='MACD')
                ax4.plot(self.datetime_data, self.indicator_data['macd_signal'], color='orange', linewidth=1.5, label='Signal')
                ax4.bar(self.datetime_data, self.indicator_data['macd_hist'], color='gray', alpha=0.6, label='Histogramme')
                ax4.axhline(y=0, color='white', linestyle='-', alpha=0.3)
                ax4.set_title('MACD', fontsize=12, fontweight='bold')
                ax4.legend(fontsize=8)
                ax4.grid(True, alpha=0.3)
            
            # 5. Volume
            ax5 = plt.subplot(4, 2, 6)
            if len(self.indicator_data['volume']) == len(self.datetime_data):
                ax5.bar(self.datetime_data, self.indicator_data['volume'], color=colors['volume'], alpha=0.7)
                ax5.set_title('Volume', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3)
            
            # 6. Régimes de marché
            ax6 = plt.subplot(4, 2, 7)
            if self.regime_history:
                regime_colors = {'TRENDING_UP': 'green', 'TRENDING_DOWN': 'red', 'RANGING': 'blue', 'NEUTRAL': 'gray'}
                regime_values = {'TRENDING_UP': 1, 'TRENDING_DOWN': -1, 'RANGING': 0, 'NEUTRAL': 0.5}
                
                regime_nums = [regime_values.get(r, 0) for r in self.regime_history]
                for i, (dt, regime) in enumerate(zip(self.datetime_data[:len(self.regime_history)], self.regime_history)):
                    color = regime_colors.get(regime, 'gray')
                    ax6.scatter(dt, regime_nums[i], color=color, s=20, alpha=0.7)
                
                ax6.set_title('Régimes de Marché', fontsize=12, fontweight='bold')
                ax6.set_ylim(-1.5, 1.5)
                ax6.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax6.set_yticklabels(['TRENDING_DOWN', '', 'RANGING', 'NEUTRAL', 'TRENDING_UP'])
                ax6.grid(True, alpha=0.3)
            
            # 7. Force des signaux
            ax7 = plt.subplot(4, 2, 8)
            if self.signal_strength_history:
                ax7.plot(self.datetime_data[:len(self.signal_strength_history)], self.signal_strength_history, 
                        color='yellow', linewidth=2)
                ax7.axhline(y=self.p.min_signal_strength, color='red', linestyle='--', alpha=0.7, 
                          label=f'Seuil min ({self.p.min_signal_strength})')
                ax7.set_title('Force des Signaux', fontsize=12, fontweight='bold')
                ax7.set_ylim(0, 1)
                ax7.legend(fontsize=8)
                ax7.grid(True, alpha=0.3)
            
            # Ajustement de la mise en page
            plt.tight_layout()
            
            # Sauvegarde
            if self.p.save_plots:
                filename = f'mastertrend_visual_{config_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
                self.log(f'📊 Graphique sauvegardé: {filename}')
            
            plt.show()
            
        except Exception as e:
            self.log(f"Erreur création graphiques: {e}")
    
    def stop(self):
        """Statistiques finales avec graphiques"""
        try:
            final_value = self.broker.getvalue()
            
            total_return = self.safe_divide(
                (final_value - self.initial_cash), 
                self.initial_cash, 
                0.0
            ) * 100
                
            max_dd = self.safe_divide(
                (self.peak_value - final_value), 
                self.peak_value, 
                0.0
            ) * 100
            
            self.log(f'=== RÉSULTATS MASTERTREND VISUAL ===')
            self.log(f'💰 Capital Initial: ${self.initial_cash:.2f}')
            self.log(f'💰 Capital Final: ${final_value:.2f}')
            self.log(f'📈 Rendement Total: {total_return:.2f}%')
            self.log(f'📉 Drawdown Max: {max_dd:.2f}%')
            self.log(f'🎯 Signaux Générés: {self.total_signals}')
            self.log(f'🟢 Signaux BUY: {len(self.buy_signals)}')
            self.log(f'🔴 Signaux SELL: {len(self.sell_signals)}')
            self.log(f'📊 Trades Total: {len(self.trade_history)}')
            
            # Analyse par régime
            self.log(f'\n=== PERFORMANCE PAR RÉGIME ===')
            for regime, stats in self.performance_by_regime.items():
                if stats["trades"] > 0:
                    avg_profit = self.safe_divide(stats["profit"], stats["trades"], 0.0)
                    self.log(f'{regime}: {stats["trades"]} trades, Profit Moyen: ${avg_profit:.2f}')
            
            # Analyse des trades
            if self.trade_history:
                profitable_trades = [t for t in self.trade_history if t['profit'] > 0]
                losing_trades = [t for t in self.trade_history if t['profit'] < 0]
                
                win_rate = self.safe_divide(len(profitable_trades), len(self.trade_history), 0.0) * 100
                avg_profit = self.safe_divide(
                    sum(t['profit'] for t in profitable_trades), 
                    len(profitable_trades), 
                    0.0
                ) if profitable_trades else 0
                avg_loss = self.safe_divide(
                    sum(t['profit'] for t in losing_trades), 
                    len(losing_trades), 
                    0.0
                ) if losing_trades else 0
                avg_roi = self.safe_divide(
                    sum(t['roi'] for t in self.trade_history), 
                    len(self.trade_history), 
                    0.0
                )
                avg_signal_strength = self.safe_divide(
                    sum(t.get('signal_strength', 0) for t in self.trade_history), 
                    len(self.trade_history), 
                    0.0
                )
                
                self.log(f'\n=== ANALYSE DÉTAILLÉE ===')
                self.log(f'🎯 Taux de Réussite: {win_rate:.1f}%')
                self.log(f'💰 Profit Moyen: ${avg_profit:.2f}')
                self.log(f'💸 Perte Moyenne: ${avg_loss:.2f}')
                self.log(f'📊 ROI Moyen: {avg_roi:.2f}%')
                self.log(f'⚡ Force Signal Moyenne: {avg_signal_strength:.2f}')
                if avg_loss != 0:
                    profit_factor = abs(self.safe_divide(avg_profit, avg_loss, 0.0))
                    self.log(f'⚡ Profit Factor: {profit_factor:.2f}')
            
            # Évaluation finale
            if total_return >= 10:
                self.log('\n🏆 EXCELLENT! Stratégie MasterTrend Visual très performante!')
            elif total_return >= 5:
                self.log('\n✅ BON! Stratégie MasterTrend Visual efficace')
            elif total_return >= 2:
                self.log('\n⚠️  MOYEN. Optimisation possible')
            else:
                self.log('\n❌ INSUFFISANT. Révision nécessaire')
                
        except Exception as e:
            self.log(f"Erreur dans stop(): {e}")


def check_data_file(filename):
    """Vérifier si le fichier de données existe"""
    return os.path.exists(filename)


if __name__ == '__main__':
    # Test avec visualisation
    test_configs = [
        ("EURUSD_1M", "EURUSD_data_1M.csv", bt.TimeFrame.Minutes, 1),
        ("EURUSD_15M", "EURUSD_data_15M.csv", bt.TimeFrame.Minutes, 15),
    ]
    
    for config_name, filename, timeframe, compression in test_configs:
        print(f'\n{"="*60}')
        print(f'🚀 TESTING MASTERTREND VISUAL - {config_name}')
        print(f'📊 Mode: VISUALISATION COMPLÈTE')
        print(f'🎯 Capital Initial: $10,000')
        print('='*60)
        
        if not check_data_file(filename):
            print(f'❌ FICHIER MANQUANT: {filename}')
            continue
        
        try:
            # Configuration
            cerebro = bt.Cerebro()
            
            # Données
            data = bt.feeds.GenericCSVData(
                dataname=filename,
                dtformat=('%Y-%m-%d %H:%M:%S'),
                datetime=0, open=1, high=2, low=3, close=4, volume=5,
                timeframe=timeframe, compression=compression,
                openinterest=-1, headers=True, separator=','
            )
            
            cerebro.adddata(data)
            cerebro.addstrategy(VisualMasterTrend)
            
            # Configuration
            cerebro.broker.setcash(10000.0)
            cerebro.broker.setcommission(commission=0.0001)
            
            # Analyseurs
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            results = cerebro.run()
            strat = results[0]
            
            # Créer les graphiques
            strat.create_plots(config_name)
            
            # Analyse des résultats
            trades = strat.analyzers.trades.get_analysis()
            drawdown = strat.analyzers.drawdown.get_analysis()
            returns = strat.analyzers.returns.get_analysis()
            
            total_trades = getattr(getattr(trades, 'total', None), 'closed', 0)
            win_rate = 0
            profit_factor = 0
            
            if hasattr(trades, 'won') and hasattr(trades, 'lost') and total_trades > 0:
                wins = getattr(getattr(trades, 'won', None), 'total', 0)
                win_rate = wins / total_trades * 100 if total_trades > 0 else 0
                
                avg_win = getattr(getattr(getattr(trades, 'won', None), 'pnl', None), 'average', 0)
                avg_loss = getattr(getattr(getattr(trades, 'lost', None), 'pnl', None), 'average', 0)
                if avg_loss != 0:
                    profit_factor = abs(avg_win / avg_loss)
            
            max_drawdown = getattr(getattr(drawdown, 'max', None), 'drawdown', 0)
            total_return = getattr(returns, 'rtot', 0) * 100
            
            print(f'\n🔥 RÉSULTATS {config_name}:')
            print(f'📊 Total Trades: {total_trades}')
            print(f'🎯 Taux de Réussite: {win_rate:.1f}%')
            print(f'⚡ Profit Factor: {profit_factor:.2f}')
            print(f'📉 Drawdown Max: {max_drawdown:.2f}%')
            print(f'📈 Rendement Total: {total_return:.2f}%')
            print(f'📊 Graphiques générés avec succès!')
            
            # Ne tester qu'une paire pour éviter trop de graphiques
            break
                
        except Exception as e:
            print(f'❌ ERREUR sur {config_name}: {e}')
    
    print(f'\n{"="*80}')
    print('🎨 MASTERTREND VISUAL - Analyse graphique terminée!')
    print('📊 Vérifiez les fichiers PNG générés pour l\'analyse détaillée.')
    print('='*80) 