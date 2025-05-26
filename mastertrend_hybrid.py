#!/usr/bin/env python3
"""
MASTERTREND HYBRID STRATEGY
Version hybride combinant les meilleurs aspects de toutes les approches
Objectif: S'adapter automatiquement aux conditions de marché pour maximiser les profits
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
from collections import deque


class HybridMasterTrend(bt.Strategy):
    """
    Stratégie MasterTrend HYBRIDE - Adaptation automatique aux conditions de marché
    """
    
    params = (
        # SuperTrend ADAPTATIF
        ('st_period_base', 10),      # Période de base
        ('st_multiplier_base', 2.5), # Multiplicateur de base
        
        # MACD ADAPTATIF
        ('macd_fast_base', 10),      # Base rapide
        ('macd_slow_base', 22),      # Base lente
        ('macd_signal_base', 8),     # Base signal
        
        # Williams Fractals ADAPTATIFS
        ('williams_left', 3),        # Équilibré
        ('williams_right', 3),       # Équilibré
        ('williams_buffer_base', 0.08), # Buffer de base
        
        # P1 Indicator ADAPTATIF
        ('p1_e31_base', 5),         # Base rapide
        ('p1_m_base', 9),           # Base moyenne
        ('p1_l31_base', 14),        # Base lente
        
        # StopGap Filter ADAPTATIF
        ('sg_ema_short_base', 8),    # Base courte
        ('sg_ema_long_base', 18),    # Base longue
        ('sg_lookback_base', 5),     # Base lookback
        ('sg_median_period_base', 10), # Base médiane
        
        # Risk Management ADAPTATIF
        ('max_daily_loss', 0.08),    # 8% perte journalière max
        ('max_total_loss', 0.15),    # 15% perte totale max
        ('profit_target', 0.25),     # 25% objectif de profit
        ('position_size_base', 0.04), # 4% par trade de base
        ('max_consecutive_losses', 5), # Limite adaptative
        
        # Sessions ADAPTATIVES
        ('trading_start', datetime.time(7, 0)),   # Étendue
        ('trading_end', datetime.time(18, 0)),    # Étendue
        
        # PARAMÈTRES HYBRIDES
        ('adaptive_mode', True),      # Mode adaptatif
        ('market_regime_detection', True), # Détection de régime
        ('volatility_adaptation', True),   # Adaptation à la volatilité
        ('trend_strength_adaptation', True), # Adaptation à la force de tendance
        ('volume_adaptation', True),  # Adaptation au volume
        ('time_adaptation', True),    # Adaptation temporelle
        
        # Modes de trading
        ('scalping_enabled', True),   # Scalping en haute volatilité
        ('swing_enabled', True),      # Swing en tendance forte
        ('breakout_enabled', True),   # Breakout en consolidation
        ('mean_reversion_enabled', True), # Mean reversion en range
        
        # Seuils adaptatifs
        ('volatility_threshold_low', 0.5),   # Seuil volatilité basse
        ('volatility_threshold_high', 2.0),  # Seuil volatilité haute
        ('trend_strength_threshold', 0.6),   # Seuil force de tendance
        ('volume_threshold', 1.2),           # Seuil volume
        
        # Optimisations
        ('dynamic_sizing', True),     # Taille dynamique
        ('adaptive_stops', True),     # Stops adaptatifs
        ('signal_filtering', True),   # Filtrage des signaux
        ('regime_switching', True),   # Changement de régime
    )
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # === INDICATEURS DE BASE ===
        
        # Prix et volume
        self.hl2 = (self.data.high + self.data.low) / 2.0
        self.hlc3 = (self.data.high + self.data.low + self.data.close) / 3.0
        self.ohlc4 = (self.data.open + self.data.high + self.data.low + self.data.close) / 4.0
        
        # ATR pour volatilité
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.atr_ma = bt.indicators.SMA(self.atr, period=20)
        
        # Indicateurs de tendance multiples
        self.sma_10 = bt.indicators.SMA(self.data.close, period=10)
        self.sma_20 = bt.indicators.SMA(self.data.close, period=20)
        self.sma_50 = bt.indicators.SMA(self.data.close, period=50)
        self.sma_100 = bt.indicators.SMA(self.data.close, period=100)
        
        self.ema_9 = bt.indicators.EMA(self.data.close, period=9)
        self.ema_21 = bt.indicators.EMA(self.data.close, period=21)
        self.ema_50 = bt.indicators.EMA(self.data.close, period=50)
        
        # MACD adaptatif (sera ajusté dynamiquement)
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast_base,
            period_me2=self.p.macd_slow_base,
            period_signal=self.p.macd_signal_base
        )
        
        # RSI pour momentum
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.rsi_fast = bt.indicators.RSI(self.data.close, period=7)
        
        # Stochastic pour oscillation
        self.stoch = bt.indicators.Stochastic(self.data, period=14)
        self.stoch_fast = bt.indicators.Stochastic(self.data, period=7)
        
        # Volume
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)
        self.volume_ma_fast = bt.indicators.SMA(self.data.volume, period=10)
        
        # Bollinger Bands pour volatilité et breakouts
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2.0)
        self.bb_squeeze = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=1.5)
        
        # Indicateurs de volatilité
        self.volatility = bt.indicators.StdDev(self.data.close, period=20)
        self.volatility_ma = bt.indicators.SMA(self.volatility, period=10)
        
        # Momentum
        self.momentum = bt.indicators.Momentum(self.data.close, period=10)
        self.momentum_fast = bt.indicators.Momentum(self.data.close, period=5)
        
        # Williams %R
        self.williams_r = bt.indicators.WilliamsR(self.data, period=14)
        
        # === VARIABLES D'ÉTAT HYBRIDES ===
        
        # SuperTrend adaptatif
        self.trend = 1
        self.final_up = 0.0
        self.final_dn = 0.0
        self.st_period = self.p.st_period_base
        self.st_multiplier = self.p.st_multiplier_base
        
        # Williams Stops adaptatifs
        self.williams_long_stop = None
        self.williams_short_stop = None
        self.williams_long_active = False
        self.williams_short_active = False
        self.williams_buffer = self.p.williams_buffer_base
        
        # P1 Indicator adaptatif
        self.hilow = (self.data.high - self.data.low) * 100
        self.openclose = (self.data.close - self.data.open) * 100
        self.spreadv = self.openclose * self.data.close
        self.pt_approx = bt.indicators.SMA(self.spreadv, period=50)
        
        # EMAs P1 (seront ajustées dynamiquement)
        self.ema_e31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_e31_base)
        self.ema_m = bt.indicators.EMA(self.pt_approx, period=self.p.p1_m_base)
        self.ema_l31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_l31_base)
        
        self.a1 = self.ema_l31 - self.ema_m
        self.b1 = self.ema_e31 - self.ema_m
        self.p1 = self.a1 + self.b1
        
        # StopGap adaptatif
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_short_base)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_long_base)
        self.highest_sg = bt.indicators.Highest(self.data.high, period=self.p.sg_lookback_base)
        self.lowest_sg = bt.indicators.Lowest(self.data.low, period=self.p.sg_lookback_base)
        
        # === DÉTECTION DE RÉGIME DE MARCHÉ ===
        self.market_regime = "NEUTRAL"  # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
        self.regime_confidence = 0.0
        self.volatility_regime = "NORMAL"  # LOW, NORMAL, HIGH
        self.trend_strength = 0.0
        self.volume_regime = "NORMAL"  # LOW, NORMAL, HIGH
        
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
        self.last_trade_profit = 0
        
        # Statistiques
        self.buy_signals = []
        self.sell_signals = []
        self.trade_history = []
        self.total_signals = 0
        self.profitable_signals = 0
        
        # Adaptation
        self.position_size = self.p.position_size_base
        self.last_signal_time = None
        self.min_signal_interval = 3
        
        # StopGap median calculation
        self.candle_sizes = deque(maxlen=self.p.sg_median_period_base)
        
        # Régime tracking
        self.regime_history = deque(maxlen=50)
        self.performance_by_regime = {
            "TRENDING_UP": {"trades": 0, "profit": 0.0},
            "TRENDING_DOWN": {"trades": 0, "profit": 0.0},
            "RANGING": {"trades": 0, "profit": 0.0},
            "VOLATILE": {"trades": 0, "profit": 0.0}
        }
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🟢 BUY EXECUTED @ {order.executed.price:.5f} (Size: {order.executed.size}) [Regime: {self.market_regime}]')
                self.position_entry_price = order.executed.price
            else:
                self.log(f'🔴 SELL EXECUTED @ {order.executed.price:.5f} (Size: {order.executed.size}) [Regime: {self.market_regime}]')
                self.position_entry_price = order.executed.price
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('❌ Order Canceled/Margin/Rejected')
            
        self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.last_trade_profit = trade.pnl
            
            # Enregistrer performance par régime
            if self.market_regime in self.performance_by_regime:
                self.performance_by_regime[self.market_regime]["trades"] += 1
                self.performance_by_regime[self.market_regime]["profit"] += trade.pnl
            
            if trade.pnl > 0:
                self.consecutive_losses = 0
                self.consecutive_wins += 1
                self.profitable_signals += 1
                roi = trade.pnl/abs(trade.value)*100 if trade.value != 0 else 0
                self.log(f'💰 PROFIT: ${trade.pnl:.2f} (ROI: {roi:.1f}%) [Regime: {self.market_regime}, Streak: {self.consecutive_wins}]')
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.log(f'💸 LOSS: ${trade.pnl:.2f} [Regime: {self.market_regime}, Consecutive: {self.consecutive_losses}]')
            
            # Enregistrer le trade
            self.trade_history.append({
                'profit': trade.pnl,
                'duration': trade.barlen,
                'entry_price': self.position_entry_price,
                'roi': trade.pnl/abs(trade.value)*100 if trade.value != 0 else 0,
                'regime': self.market_regime,
                'volatility_regime': self.volatility_regime
            })
    
    def detect_market_regime(self):
        """Détection avancée du régime de marché"""
        if len(self.data) < 50:
            return
        
        # === ANALYSE DE TENDANCE ===
        
        # Alignement des moyennes mobiles
        ma_alignment_up = (self.ema_9[0] > self.ema_21[0] > self.ema_50[0] and
                          self.sma_20[0] > self.sma_50[0])
        ma_alignment_down = (self.ema_9[0] < self.ema_21[0] < self.ema_50[0] and
                            self.sma_20[0] < self.sma_50[0])
        
        # Force de la tendance
        price_vs_ma = (self.data.close[0] - self.sma_20[0]) / self.sma_20[0] * 100
        trend_momentum = self.momentum[0] / self.data.close[0] * 100
        
        # Calcul de la force de tendance
        self.trend_strength = abs(price_vs_ma) + abs(trend_momentum)
        
        # === ANALYSE DE VOLATILITÉ ===
        
        volatility_ratio = self.volatility[0] / self.volatility_ma[0] if self.volatility_ma[0] > 0 else 1.0
        atr_ratio = self.atr[0] / self.atr_ma[0] if self.atr_ma[0] > 0 else 1.0
        
        # Régime de volatilité
        if volatility_ratio < self.p.volatility_threshold_low:
            self.volatility_regime = "LOW"
        elif volatility_ratio > self.p.volatility_threshold_high:
            self.volatility_regime = "HIGH"
        else:
            self.volatility_regime = "NORMAL"
        
        # === ANALYSE DE VOLUME ===
        
        volume_ratio = self.data.volume[0] / self.volume_ma[0] if self.volume_ma[0] > 0 else 1.0
        
        if volume_ratio < 0.8:
            self.volume_regime = "LOW"
        elif volume_ratio > self.p.volume_threshold:
            self.volume_regime = "HIGH"
        else:
            self.volume_regime = "NORMAL"
        
        # === DÉTERMINATION DU RÉGIME ===
        
        # Tendance forte
        if (ma_alignment_up and self.trend_strength > self.p.trend_strength_threshold and
            price_vs_ma > 1.0):
            self.market_regime = "TRENDING_UP"
            self.regime_confidence = min(1.0, self.trend_strength / 2.0)
            
        elif (ma_alignment_down and self.trend_strength > self.p.trend_strength_threshold and
              price_vs_ma < -1.0):
            self.market_regime = "TRENDING_DOWN"
            self.regime_confidence = min(1.0, self.trend_strength / 2.0)
            
        # Marché volatile
        elif self.volatility_regime == "HIGH" and atr_ratio > 1.5:
            self.market_regime = "VOLATILE"
            self.regime_confidence = min(1.0, volatility_ratio / 2.0)
            
        # Marché en range
        else:
            self.market_regime = "RANGING"
            self.regime_confidence = 1.0 - min(1.0, self.trend_strength / 2.0)
        
        # Enregistrer l'historique
        self.regime_history.append(self.market_regime)
    
    def adapt_parameters(self):
        """Adaptation des paramètres selon le régime de marché"""
        if not self.p.adaptive_mode:
            return
        
        # === ADAPTATION SELON LE RÉGIME ===
        
        if self.market_regime == "TRENDING_UP" or self.market_regime == "TRENDING_DOWN":
            # Paramètres pour tendance forte
            self.st_period = max(8, self.p.st_period_base - 2)
            self.st_multiplier = self.p.st_multiplier_base * 0.9
            self.position_size = self.p.position_size_base * 1.2
            self.min_signal_interval = 2
            
        elif self.market_regime == "VOLATILE":
            # Paramètres pour haute volatilité
            self.st_period = self.p.st_period_base + 2
            self.st_multiplier = self.p.st_multiplier_base * 1.3
            self.position_size = self.p.position_size_base * 0.8
            self.min_signal_interval = 1
            
        elif self.market_regime == "RANGING":
            # Paramètres pour marché en range
            self.st_period = self.p.st_period_base + 4
            self.st_multiplier = self.p.st_multiplier_base * 1.1
            self.position_size = self.p.position_size_base * 0.9
            self.min_signal_interval = 5
        
        # === ADAPTATION À LA VOLATILITÉ ===
        
        if self.p.volatility_adaptation:
            if self.volatility_regime == "HIGH":
                self.williams_buffer = self.p.williams_buffer_base * 1.5
            elif self.volatility_regime == "LOW":
                self.williams_buffer = self.p.williams_buffer_base * 0.7
            else:
                self.williams_buffer = self.p.williams_buffer_base
        
        # === ADAPTATION AU VOLUME ===
        
        if self.p.volume_adaptation and self.volume_regime == "HIGH":
            self.position_size *= 1.1
        elif self.p.volume_adaptation and self.volume_regime == "LOW":
            self.position_size *= 0.9
    
    def update_supertrend(self):
        """Mise à jour SuperTrend adaptatif"""
        if len(self.data) < 2:
            return
        
        # Utiliser les paramètres adaptatifs
        basic_up = self.hl2[0] - self.st_multiplier * self.atr[0]
        basic_dn = self.hl2[0] + self.st_multiplier * self.atr[0]
        
        if self.data.close[-1] > self.final_up:
            self.final_up = max(basic_up, self.final_up)
        else:
            self.final_up = basic_up
            
        if self.data.close[-1] < self.final_dn:
            self.final_dn = min(basic_dn, self.final_dn)
        else:
            self.final_dn = basic_dn
            
        prev_trend = self.trend
        if self.trend == -1 and self.data.close[0] > self.final_dn:
            self.trend = 1
        elif self.trend == 1 and self.data.close[0] < self.final_up:
            self.trend = -1
    
    def update_williams_stops(self):
        """Mise à jour Williams Stops adaptatifs"""
        if len(self.data) > self.p.williams_left + self.p.williams_right:
            center_idx = self.p.williams_right
            
            # High fractal
            is_high_fractal = True
            center_high = self.data.high[-center_idx]
            
            for i in range(self.p.williams_left + self.p.williams_right + 1):
                if i != center_idx and self.data.high[-i] >= center_high:
                    is_high_fractal = False
                    break
            
            if is_high_fractal:
                self.williams_short_stop = center_high * (1 + self.williams_buffer / 100)
                self.williams_short_active = True
            
            # Low fractal
            is_low_fractal = True
            center_low = self.data.low[-center_idx]
            
            for i in range(self.p.williams_left + self.p.williams_right + 1):
                if i != center_idx and self.data.low[-i] <= center_low:
                    is_low_fractal = False
                    break
            
            if is_low_fractal:
                self.williams_long_stop = center_low * (1 - self.williams_buffer / 100)
                self.williams_long_active = True
    
    def calculate_stopgap_filter(self):
        """Calcul du filtre StopGap adaptatif"""
        trend_up = self.ema_short[0] > self.ema_long[0]
        trend_down = self.ema_short[0] < self.ema_long[0]
        
        candle_size = abs(self.data.high[0] - self.data.low[0])
        self.candle_sizes.append(candle_size)
        
        if len(self.candle_sizes) == self.p.sg_median_period_base:
            median = np.median(list(self.candle_sizes))
        else:
            median = candle_size
        
        stop_gap = 0.0
        if trend_down and self.data.high[0] < self.highest_sg[-1]:
            stop_gap = abs(self.highest_sg[-1] - self.data.low[0])
        elif trend_up and self.data.low[0] > self.lowest_sg[-1]:
            stop_gap = abs(self.data.high[0] - self.lowest_sg[-1])
        
        # Ajustement selon le régime
        multiplier = 1.0
        if self.market_regime == "VOLATILE":
            multiplier = 0.8  # Plus permissif en volatilité
        elif self.market_regime == "RANGING":
            multiplier = 1.2  # Plus strict en range
        
        return stop_gap > (median * multiplier)
    
    def get_regime_specific_signals(self):
        """Signaux spécifiques au régime de marché"""
        long_signals = []
        short_signals = []
        
        # === SIGNAUX POUR TENDANCE HAUSSIÈRE ===
        if self.market_regime == "TRENDING_UP" and self.p.swing_enabled:
            # Pullback dans tendance haussière
            if (self.data.close[0] > self.ema_21[0] and
                self.rsi[0] < 50 and self.rsi[-1] >= 50 and
                self.macd.macd[0] > self.macd.signal[0]):
                long_signals.append("TREND_PULLBACK")
        
        # === SIGNAUX POUR TENDANCE BAISSIÈRE ===
        elif self.market_regime == "TRENDING_DOWN" and self.p.swing_enabled:
            # Pullback dans tendance baissière
            if (self.data.close[0] < self.ema_21[0] and
                self.rsi[0] > 50 and self.rsi[-1] <= 50 and
                self.macd.macd[0] < self.macd.signal[0]):
                short_signals.append("TREND_PULLBACK")
        
        # === SIGNAUX POUR MARCHÉ VOLATILE ===
        elif self.market_regime == "VOLATILE" and self.p.scalping_enabled:
            # Scalping en volatilité
            if (self.rsi_fast[0] < 30 and self.rsi_fast[-1] >= 30 and
                self.data.volume[0] > self.volume_ma[0] * 1.2):
                long_signals.append("VOLATILITY_SCALP")
            elif (self.rsi_fast[0] > 70 and self.rsi_fast[-1] <= 70 and
                  self.data.volume[0] > self.volume_ma[0] * 1.2):
                short_signals.append("VOLATILITY_SCALP")
        
        # === SIGNAUX POUR MARCHÉ EN RANGE ===
        elif self.market_regime == "RANGING" and self.p.mean_reversion_enabled:
            # Mean reversion
            bb_lower_touch = self.data.low[0] <= self.bb.bot[0] and self.data.close[0] > self.bb.bot[0]
            bb_upper_touch = self.data.high[0] >= self.bb.top[0] and self.data.close[0] < self.bb.top[0]
            
            if bb_lower_touch and self.williams_r[0] < -80:
                long_signals.append("MEAN_REVERSION")
            elif bb_upper_touch and self.williams_r[0] > -20:
                short_signals.append("MEAN_REVERSION")
        
        # === SIGNAUX DE BREAKOUT (TOUS RÉGIMES) ===
        if self.p.breakout_enabled:
            # Breakout Bollinger Bands avec volume
            bb_breakout_up = (self.data.close[0] > self.bb.top[0] and 
                             self.data.close[-1] <= self.bb.top[-1] and
                             self.data.volume[0] > self.volume_ma[0] * 1.3)
            bb_breakout_down = (self.data.close[0] < self.bb.bot[0] and 
                               self.data.close[-1] >= self.bb.bot[-1] and
                               self.data.volume[0] > self.volume_ma[0] * 1.3)
            
            if bb_breakout_up:
                long_signals.append("BREAKOUT")
            elif bb_breakout_down:
                short_signals.append("BREAKOUT")
        
        return long_signals, short_signals
    
    def check_hybrid_rules(self):
        """Vérification des règles hybrides"""
        current_value = self.broker.getvalue()
        dt = self.datas[0].datetime.datetime(0)
        
        # Reset daily tracking
        if self.current_date != dt.date():
            self.current_date = dt.date()
            self.daily_start_cash = current_value
        
        # Update peak
        self.peak_value = max(self.peak_value, current_value)
        
        # Check consecutive losses (adaptatif selon le régime)
        max_losses = self.p.max_consecutive_losses
        if self.market_regime == "VOLATILE":
            max_losses += 2  # Plus de tolérance en volatilité
        
        if self.consecutive_losses >= max_losses:
            self.log(f'⚠️  MAX CONSECUTIVE LOSSES HIT: {self.consecutive_losses} [Regime: {self.market_regime}]')
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
        """Vérification session de trading"""
        dt = self.datas[0].datetime.datetime(0)
        current_time = dt.time()
        return self.p.trading_start <= current_time <= self.p.trading_end
    
    def calculate_dynamic_position_size(self):
        """Calcul de la taille de position dynamique"""
        account_value = self.broker.getvalue()
        
        # Taille de base adaptée
        base_size = self.position_size
        
        # Ajustement basé sur la performance récente
        if self.consecutive_wins >= 3:
            base_size *= 1.15
        elif self.consecutive_losses >= 2:
            base_size *= 0.85
        
        # Ajustement basé sur la confiance du régime
        base_size *= (0.8 + 0.4 * self.regime_confidence)
        
        # Ajustement basé sur la volatilité
        if self.volatility_regime == "HIGH":
            base_size *= 0.8
        elif self.volatility_regime == "LOW":
            base_size *= 1.1
        
        risk_amount = account_value * base_size
        
        # Utiliser Williams Stop pour calculer la taille
        if self.williams_long_stop:
            stop_distance = abs(self.data.close[0] - self.williams_long_stop)
        elif self.williams_short_stop:
            stop_distance = abs(self.data.close[0] - self.williams_short_stop)
        else:
            stop_distance = self.data.close[0] * 0.008  # 0.8% par défaut
        
        if stop_distance > 0:
            size = risk_amount / stop_distance
            # Limite selon le régime
            max_pct = 0.2 if self.market_regime == "VOLATILE" else 0.15
            if self.data.close[0] > 0:
                max_size = account_value * max_pct / self.data.close[0]
                size = min(size, max_size)
            return max(1, int(size))
        return 1
    
    def next(self):
        # Détection du régime de marché
        self.detect_market_regime()
        
        # Adaptation des paramètres
        self.adapt_parameters()
        
        # Vérification des règles hybrides
        if not self.check_hybrid_rules():
            if self.position:
                self.close()
            return
        
        # Vérification session de trading
        if not self.is_trading_session():
            return
        
        # Mise à jour des indicateurs
        self.update_supertrend()
        self.update_williams_stops()
        
        # === SIGNAUX DE BASE ===
        
        # MACD Crossovers
        crossmacdbear = (self.macd.macd[0] > self.macd.signal[0] and 
                        self.macd.macd[-1] <= self.macd.signal[-1])
        crossmacd = (self.macd.macd[0] < self.macd.signal[0] and 
                    self.macd.macd[-1] >= self.macd.signal[-1])
        
        # P1 Conditions
        p1_margin = 0.03 if self.market_regime == "VOLATILE" else 0.05
        b1_ge_p1 = self.b1[0] >= self.p1[0] * (1 + p1_margin)
        b1_le_p1 = self.b1[0] <= self.p1[0] * (1 - p1_margin)
        
        # StopGap Filter
        stopgap_ok = self.calculate_stopgap_filter()
        
        # Signaux spécifiques au régime
        regime_long_signals, regime_short_signals = self.get_regime_specific_signals()
        
        # === CONDITIONS HYBRIDES ===
        
        # Conditions de base
        base_long = (self.williams_long_active and crossmacdbear and b1_ge_p1 and 
                    self.trend == 1 and stopgap_ok)
        base_short = (self.williams_short_active and crossmacd and b1_le_p1 and 
                     self.trend == -1 and stopgap_ok)
        
        # Conditions finales
        long_condition = base_long or len(regime_long_signals) > 0
        short_condition = base_short or len(regime_short_signals) > 0
        
        # Filtre temporel
        current_bar = len(self.data)
        if self.last_signal_time is not None:
            bars_since_last = current_bar - self.last_signal_time
            if bars_since_last < self.min_signal_interval:
                long_condition = False
                short_condition = False
        
        # === EXÉCUTION DES TRADES ===
        
        if self.order:
            return
        
        if not self.position:
            if long_condition:
                self.total_signals += 1
                self.last_signal_time = current_bar
                size = self.calculate_dynamic_position_size()
                signal_type = regime_long_signals[0] if regime_long_signals else "BASE"
                self.log(f'🚀 HYBRID BUY @ {self.data.close[0]:.5f} (Size: {size}) [Regime: {self.market_regime}, Signal: {signal_type}]')
                self.order = self.buy(size=size)
                self.buy_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
                
            elif short_condition:
                self.total_signals += 1
                self.last_signal_time = current_bar
                size = self.calculate_dynamic_position_size()
                signal_type = regime_short_signals[0] if regime_short_signals else "BASE"
                self.log(f'🚀 HYBRID SELL @ {self.data.close[0]:.5f} (Size: {size}) [Regime: {self.market_regime}, Signal: {signal_type}]')
                self.order = self.sell(size=size)
                self.sell_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
        
        else:  # Position ouverte - gestion hybride
            if self.position.size > 0:  # Position longue
                # Quick exit adaptatif
                quick_profit = 0.02 if self.market_regime == "VOLATILE" else 0.025
                if (self.position_entry_price and 
                    self.data.close[0] >= self.position_entry_price * (1 + quick_profit)):
                    self.log(f'💰 QUICK PROFIT EXIT @ {self.data.close[0]:.5f} [Regime: {self.market_regime}]')
                    self.order = self.close()
                # RSI exit adaptatif
                elif self.rsi[0] > (75 if self.market_regime == "VOLATILE" else 70):
                    self.log(f'📈 RSI EXIT @ {self.data.close[0]:.5f} (RSI: {self.rsi[0]:.1f})')
                    self.order = self.close()
                # Williams stop
                elif (self.williams_long_stop and 
                      self.data.low[0] <= self.williams_long_stop):
                    self.log(f'🛑 LONG STOP @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Signal de sortie
                elif crossmacd or self.trend == -1:
                    self.log(f'🔄 LONG EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                    
            elif self.position.size < 0:  # Position courte
                # Quick exit adaptatif
                quick_profit = 0.02 if self.market_regime == "VOLATILE" else 0.025
                if (self.position_entry_price and 
                    self.data.close[0] <= self.position_entry_price * (1 - quick_profit)):
                    self.log(f'💰 QUICK PROFIT EXIT @ {self.data.close[0]:.5f} [Regime: {self.market_regime}]')
                    self.order = self.close()
                # RSI exit adaptatif
                elif self.rsi[0] < (25 if self.market_regime == "VOLATILE" else 30):
                    self.log(f'📉 RSI EXIT @ {self.data.close[0]:.5f} (RSI: {self.rsi[0]:.1f})')
                    self.order = self.close()
                # Williams stop
                elif (self.williams_short_stop and 
                      self.data.high[0] >= self.williams_short_stop):
                    self.log(f'🛑 SHORT STOP @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Signal de sortie
                elif crossmacdbear or self.trend == 1:
                    self.log(f'🔄 SHORT EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
    
    def stop(self):
        """Statistiques finales hybrides"""
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
        
        self.log(f'=== RÉSULTATS MASTERTREND HYBRIDE ===')
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
                avg_profit = stats["profit"] / stats["trades"]
                self.log(f'{regime}: {stats["trades"]} trades, Profit Moyen: ${avg_profit:.2f}')
        
        # Analyse des trades
        if self.trade_history and len(self.trade_history) > 0:
            profitable_trades = [t for t in self.trade_history if t['profit'] > 0]
            losing_trades = [t for t in self.trade_history if t['profit'] < 0]
            
            win_rate = len(profitable_trades) / len(self.trade_history) * 100
            avg_profit = sum(t['profit'] for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0
            avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            avg_roi = sum(t['roi'] for t in self.trade_history) / len(self.trade_history)
            
            self.log(f'\n=== ANALYSE DÉTAILLÉE ===')
            self.log(f'🎯 Taux de Réussite: {win_rate:.1f}%')
            self.log(f'💰 Profit Moyen: ${avg_profit:.2f}')
            self.log(f'💸 Perte Moyenne: ${avg_loss:.2f}')
            self.log(f'📊 ROI Moyen: {avg_roi:.2f}%')
            if avg_loss != 0:
                profit_factor = abs(avg_profit / avg_loss)
                self.log(f'⚡ Profit Factor: {profit_factor:.2f}')
        
        # Évaluation finale
        if total_return >= 20:
            self.log('\n🏆 EXCELLENT! Stratégie hybride très performante!')
        elif total_return >= 10:
            self.log('\n✅ BON! Stratégie hybride efficace')
        elif total_return >= 5:
            self.log('\n⚠️  MOYEN. Optimisation possible')
        else:
            self.log('\n❌ INSUFFISANT. Révision nécessaire')


if __name__ == '__main__':
    # Configuration hybride
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
    cerebro.addstrategy(HybridMasterTrend)
    
    # Configuration
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)
    
    # Analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print('🚀 MASTERTREND HYBRID STRATEGY - DÉMARRAGE')
    print('🧬 Mode: ADAPTATIF HYBRIDE')
    print('🎯 Capital Initial: $10,000')
    print('🔄 Adaptation automatique aux régimes de marché')
    print('=' * 50)
    
    try:
        results = cerebro.run()
        strat = results[0]
        
        # Analyse détaillée des résultats
        trades = strat.analyzers.trades.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        
        print('\n🔥 ANALYSE MASTERTREND HYBRID:')
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
        if final_return >= 15:
            print('\n🏆 MISSION ACCOMPLIE! Stratégie hybride excellente!')
        elif final_return >= 8:
            print('\n✅ Bon résultat! Stratégie hybride efficace')
        else:
            print('\n⚠️  Résultat à améliorer. Ajustements possibles.')
            
    except Exception as e:
        print(f'❌ ERREUR LORS DE L\'EXÉCUTION: {e}')
        print('Vérifiez que le fichier EURUSD_data_15M.csv existe et a le bon format.') 