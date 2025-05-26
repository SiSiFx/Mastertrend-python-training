#!/usr/bin/env python3
"""
MASTERTREND HYBRID IMPROVED STRATEGY
Version hybride améliorée avec meilleure gestion des marchés en range
Objectif: S'adapter intelligemment aux conditions de marché
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
from collections import deque


class ImprovedHybridMasterTrend(bt.Strategy):
    """
    Stratégie MasterTrend HYBRIDE AMÉLIORÉE - Gestion intelligente des régimes de marché
    """
    
    params = (
        # SuperTrend ADAPTATIF
        ('st_period_base', 12),      # Plus stable
        ('st_multiplier_base', 2.8), # Plus conservateur
        
        # MACD ADAPTATIF
        ('macd_fast_base', 12),      # Plus stable
        ('macd_slow_base', 26),      # Standard
        ('macd_signal_base', 9),     # Standard
        
        # Williams Fractals ADAPTATIFS
        ('williams_left', 4),        # Plus stable
        ('williams_right', 4),       # Plus stable
        ('williams_buffer_base', 0.1), # Plus conservateur
        
        # P1 Indicator ADAPTATIF
        ('p1_e31_base', 7),         # Plus stable
        ('p1_m_base', 12),          # Plus stable
        ('p1_l31_base', 18),        # Plus stable
        
        # StopGap Filter ADAPTATIF
        ('sg_ema_short_base', 10),   # Plus stable
        ('sg_ema_long_base', 21),    # Plus stable
        ('sg_lookback_base', 8),     # Plus stable
        ('sg_median_period_base', 14), # Plus stable
        
        # Risk Management CONSERVATEUR
        ('max_daily_loss', 0.03),    # 3% perte journalière max
        ('max_total_loss', 0.08),    # 8% perte totale max
        ('profit_target', 0.15),     # 15% objectif de profit
        ('position_size_base', 0.02), # 2% par trade de base
        ('max_consecutive_losses', 3), # Limite stricte
        
        # Sessions OPTIMALES
        ('trading_start', datetime.time(8, 0)),   # Heures actives
        ('trading_end', datetime.time(16, 0)),    # Heures actives
        
        # PARAMÈTRES HYBRIDES AMÉLIORÉS
        ('adaptive_mode', True),      # Mode adaptatif
        ('market_regime_detection', True), # Détection de régime
        ('volatility_adaptation', True),   # Adaptation à la volatilité
        ('trend_strength_adaptation', True), # Adaptation à la force de tendance
        ('volume_adaptation', True),  # Adaptation au volume
        ('time_adaptation', True),    # Adaptation temporelle
        
        # Modes de trading SÉLECTIFS
        ('ranging_mode_enabled', True),   # Mode range amélioré
        ('trending_mode_enabled', True),  # Mode tendance
        ('breakout_mode_enabled', True),  # Mode breakout
        ('avoid_whipsaws', True),         # Éviter les faux signaux
        
        # Seuils adaptatifs AMÉLIORÉS
        ('volatility_threshold_low', 0.3),   # Plus strict
        ('volatility_threshold_high', 1.8),  # Plus strict
        ('trend_strength_threshold', 0.8),   # Plus strict
        ('volume_threshold', 1.5),           # Plus strict
        ('ranging_threshold', 0.4),          # Seuil pour détecter le range
        
        # Filtres de qualité
        ('min_signal_strength', 0.7),    # Force minimale du signal
        ('min_bars_between_signals', 8), # Minimum de barres entre signaux
        ('require_volume_confirmation', True), # Confirmation volume
        ('require_momentum_alignment', True),  # Alignement momentum
        ('use_regime_specific_exits', True),   # Sorties spécifiques au régime
    )
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # === INDICATEURS DE BASE AMÉLIORÉS ===
        
        # Prix et volume
        self.hl2 = (self.data.high + self.data.low) / 2.0
        self.hlc3 = (self.data.high + self.data.low + self.data.close) / 3.0
        self.ohlc4 = (self.data.open + self.data.high + self.data.low + self.data.close) / 4.0
        
        # ATR pour volatilité
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.atr_ma = bt.indicators.SMA(self.atr, period=21)
        self.atr_std = bt.indicators.StdDev(self.atr, period=21)
        
        # Indicateurs de tendance multiples
        self.sma_20 = bt.indicators.SMA(self.data.close, period=20)
        self.sma_50 = bt.indicators.SMA(self.data.close, period=50)
        self.sma_100 = bt.indicators.SMA(self.data.close, period=100)
        self.sma_200 = bt.indicators.SMA(self.data.close, period=200)
        
        self.ema_12 = bt.indicators.EMA(self.data.close, period=12)
        self.ema_26 = bt.indicators.EMA(self.data.close, period=26)
        self.ema_50 = bt.indicators.EMA(self.data.close, period=50)
        
        # MACD standard
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast_base,
            period_me2=self.p.macd_slow_base,
            period_signal=self.p.macd_signal_base
        )
        
        # RSI pour momentum
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.rsi_ma = bt.indicators.SMA(self.rsi, period=5)
        
        # Stochastic pour oscillation
        self.stoch = bt.indicators.Stochastic(self.data, period=14)
        
        # Volume
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=21)
        self.volume_ratio = self.data.volume / self.volume_ma
        
        # Bollinger Bands pour volatilité et range detection
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2.0)
        self.bb_width = (self.bb.top - self.bb.bot) / self.bb.mid * 100
        self.bb_position = (self.data.close - self.bb.bot) / (self.bb.top - self.bb.bot) * 100
        
        # Indicateurs de volatilité
        self.volatility = bt.indicators.StdDev(self.data.close, period=20)
        self.volatility_ma = bt.indicators.SMA(self.volatility, period=10)
        self.volatility_ratio = self.volatility / self.volatility_ma
        
        # Momentum et force
        self.momentum = bt.indicators.Momentum(self.data.close, period=14)
        self.roc = bt.indicators.RateOfChange(self.data.close, period=10)
        
        # Williams %R
        self.williams_r = bt.indicators.WilliamsR(self.data, period=14)
        
        # Donchian Channels pour range detection
        self.donchian_high = bt.indicators.Highest(self.data.high, period=20)
        self.donchian_low = bt.indicators.Lowest(self.data.low, period=20)
        self.donchian_mid = (self.donchian_high + self.donchian_low) / 2.0
        
        # === VARIABLES D'ÉTAT HYBRIDES AMÉLIORÉES ===
        
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
        
        # EMAs P1
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
        
        # === DÉTECTION DE RÉGIME DE MARCHÉ AMÉLIORÉE ===
        self.market_regime = "NEUTRAL"  # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, NEUTRAL
        self.regime_confidence = 0.0
        self.volatility_regime = "NORMAL"  # LOW, NORMAL, HIGH
        self.trend_strength = 0.0
        self.volume_regime = "NORMAL"  # LOW, NORMAL, HIGH
        self.ranging_score = 0.0
        
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
        self.signal_strength = 0.0
        
        # StopGap median calculation
        self.candle_sizes = deque(maxlen=self.p.sg_median_period_base)
        
        # Régime tracking amélioré
        self.regime_history = deque(maxlen=100)
        self.performance_by_regime = {
            "TRENDING_UP": {"trades": 0, "profit": 0.0, "win_rate": 0.0},
            "TRENDING_DOWN": {"trades": 0, "profit": 0.0, "win_rate": 0.0},
            "RANGING": {"trades": 0, "profit": 0.0, "win_rate": 0.0},
            "VOLATILE": {"trades": 0, "profit": 0.0, "win_rate": 0.0},
            "NEUTRAL": {"trades": 0, "profit": 0.0, "win_rate": 0.0}
        }
        
        # Filtres anti-whipsaw
        self.recent_signals = deque(maxlen=20)
        self.false_signal_count = 0
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'🟢 BUY EXECUTED @ {order.executed.price:.5f} (Size: {order.executed.size}) [Regime: {self.market_regime}, Strength: {self.signal_strength:.2f}]')
                self.position_entry_price = order.executed.price
            else:
                self.log(f'🔴 SELL EXECUTED @ {order.executed.price:.5f} (Size: {order.executed.size}) [Regime: {self.market_regime}, Strength: {self.signal_strength:.2f}]')
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
                
                # Calculer win rate
                regime_stats = self.performance_by_regime[self.market_regime]
                if regime_stats["trades"] > 0:
                    wins = sum(1 for t in self.trade_history if t.get('regime') == self.market_regime and t.get('profit', 0) > 0)
                    regime_stats["win_rate"] = wins / regime_stats["trades"] * 100
            
            if trade.pnl > 0:
                self.consecutive_losses = 0
                self.consecutive_wins += 1
                self.profitable_signals += 1
                roi = trade.pnl/abs(trade.value)*100 if trade.value != 0 else 0
                self.log(f'💰 PROFIT: ${trade.pnl:.2f} (ROI: {roi:.1f}%) [Regime: {self.market_regime}, Streak: {self.consecutive_wins}]')
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.false_signal_count += 1
                self.log(f'💸 LOSS: ${trade.pnl:.2f} [Regime: {self.market_regime}, Consecutive: {self.consecutive_losses}]')
            
            # Enregistrer le trade
            self.trade_history.append({
                'profit': trade.pnl,
                'duration': trade.barlen,
                'entry_price': self.position_entry_price,
                'roi': trade.pnl/abs(trade.value)*100 if trade.value != 0 else 0,
                'regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'signal_strength': self.signal_strength
            })
    
    def detect_market_regime_improved(self):
        """Détection améliorée du régime de marché"""
        if len(self.data) < 100:
            return
        
        # === ANALYSE DE TENDANCE AMÉLIORÉE ===
        
        # Alignement des moyennes mobiles avec pondération
        ma_score = 0.0
        
        # Score basé sur l'alignement des EMAs
        if self.ema_12[0] > self.ema_26[0] > self.ema_50[0]:
            ma_score += 3.0  # Tendance haussière forte
        elif self.ema_12[0] > self.ema_26[0]:
            ma_score += 1.0  # Tendance haussière faible
        elif self.ema_12[0] < self.ema_26[0] < self.ema_50[0]:
            ma_score -= 3.0  # Tendance baissière forte
        elif self.ema_12[0] < self.ema_26[0]:
            ma_score -= 1.0  # Tendance baissière faible
        
        # Score basé sur la position du prix vs SMAs
        price_vs_sma20 = (self.data.close[0] - self.sma_20[0]) / self.sma_20[0] * 100
        price_vs_sma50 = (self.data.close[0] - self.sma_50[0]) / self.sma_50[0] * 100
        
        # Force de la tendance
        self.trend_strength = abs(price_vs_sma20) + abs(self.momentum[0] / self.data.close[0] * 100)
        
        # === ANALYSE DE RANGE AMÉLIORÉE ===
        
                # Score de range basé sur plusieurs facteurs
        self.ranging_score = 0.0
        
        # 1. Bollinger Bands width (plus étroit = plus de range)
        bb_width_sma = bt.indicators.SMA(self.bb_width, period=50)[0]
        bb_width_norm = self.bb_width[0] / bb_width_sma if bb_width_sma > 0 else 1.0
        if bb_width_norm < 0.8:
            self.ranging_score += 2.0
        elif bb_width_norm < 1.0:
            self.ranging_score += 1.0
        
        # 2. Position dans les Bollinger Bands (oscillation)
        bb_pos_changes = 0
        for i in range(1, min(10, len(self.data))):
            if len(self.data) > i:
                bb_range_prev = self.bb.top[-i] - self.bb.bot[-i]
                bb_range_curr = self.bb.top[-i+1] - self.bb.bot[-i+1]
                if bb_range_prev > 0 and bb_range_curr > 0:
                    prev_pos = (self.data.close[-i] - self.bb.bot[-i]) / bb_range_prev * 100
                    curr_pos = (self.data.close[-i+1] - self.bb.bot[-i+1]) / bb_range_curr * 100
                    if (prev_pos > 70 and curr_pos < 30) or (prev_pos < 30 and curr_pos > 70):
                        bb_pos_changes += 1
        
        if bb_pos_changes >= 2:
            self.ranging_score += 2.0
        elif bb_pos_changes >= 1:
            self.ranging_score += 1.0
        
        # 3. Donchian Channel analysis
        if self.donchian_mid[0] > 0:
            donchian_range = (self.donchian_high[0] - self.donchian_low[0]) / self.donchian_mid[0] * 100
            if donchian_range < 2.0:  # Range étroit
                self.ranging_score += 1.5
        
        # 4. RSI oscillation
        rsi_in_middle = 40 <= self.rsi[0] <= 60
        if rsi_in_middle:
            self.ranging_score += 1.0
        
        # === ANALYSE DE VOLATILITÉ AMÉLIORÉE ===
        
        volatility_ratio = self.volatility_ratio[0] if self.volatility_ratio[0] > 0 else 1.0
        atr_ratio = self.atr[0] / self.atr_ma[0] if self.atr_ma[0] > 0 else 1.0
        
        # Régime de volatilité
        if volatility_ratio < self.p.volatility_threshold_low:
            self.volatility_regime = "LOW"
        elif volatility_ratio > self.p.volatility_threshold_high:
            self.volatility_regime = "HIGH"
        else:
            self.volatility_regime = "NORMAL"
        
        # === ANALYSE DE VOLUME ===
        
        volume_ratio = self.volume_ratio[0] if self.volume_ratio[0] > 0 else 1.0
        
        if volume_ratio < 0.7:
            self.volume_regime = "LOW"
        elif volume_ratio > self.p.volume_threshold:
            self.volume_regime = "HIGH"
        else:
            self.volume_regime = "NORMAL"
        
        # === DÉTERMINATION DU RÉGIME FINAL ===
        
        # Priorité au range si score élevé
        if self.ranging_score >= 4.0:
            self.market_regime = "RANGING"
            self.regime_confidence = min(1.0, self.ranging_score / 6.0)
            
        # Tendance forte
        elif ma_score >= 2.0 and self.trend_strength > self.p.trend_strength_threshold and price_vs_sma20 > 1.0:
            self.market_regime = "TRENDING_UP"
            self.regime_confidence = min(1.0, (ma_score + self.trend_strength) / 6.0)
            
        elif ma_score <= -2.0 and self.trend_strength > self.p.trend_strength_threshold and price_vs_sma20 < -1.0:
            self.market_regime = "TRENDING_DOWN"
            self.regime_confidence = min(1.0, (abs(ma_score) + self.trend_strength) / 6.0)
            
        # Marché volatile
        elif self.volatility_regime == "HIGH" and atr_ratio > 1.5:
            self.market_regime = "VOLATILE"
            self.regime_confidence = min(1.0, volatility_ratio / 2.0)
            
        # Marché neutre
        else:
            self.market_regime = "NEUTRAL"
            self.regime_confidence = 0.5
        
        # Enregistrer l'historique
        self.regime_history.append(self.market_regime)
    
    def calculate_signal_strength_improved(self, long_signal, short_signal):
        """Calcul amélioré de la force du signal"""
        strength = 0.0
        
        if long_signal:
            # 1. Alignement des indicateurs (30%)
            if self.trend == 1:
                strength += 0.15
            if self.macd.macd[0] > self.macd.signal[0]:
                strength += 0.10
            if self.rsi[0] < 60:  # Pas de surachat
                strength += 0.05
            
            # 2. Confirmation volume (20%)
            if self.volume_ratio[0] > 1.2:
                strength += 0.15
            elif self.volume_ratio[0] > 1.0:
                strength += 0.05
            
            # 3. Momentum (20%)
            if self.momentum[0] > 0:
                strength += 0.10
            if self.roc[0] > 0:
                strength += 0.10
            
            # 4. Position dans BB (15%)
            if self.bb_position[0] < 30:  # Proche du bas
                strength += 0.15
            elif self.bb_position[0] < 50:
                strength += 0.10
            
            # 5. Williams %R (15%)
            if self.williams_r[0] < -70:  # Survente
                strength += 0.15
            elif self.williams_r[0] < -50:
                strength += 0.10
                
        elif short_signal:
            # 1. Alignement des indicateurs (30%)
            if self.trend == -1:
                strength += 0.15
            if self.macd.macd[0] < self.macd.signal[0]:
                strength += 0.10
            if self.rsi[0] > 40:  # Pas de survente
                strength += 0.05
            
            # 2. Confirmation volume (20%)
            if self.volume_ratio[0] > 1.2:
                strength += 0.15
            elif self.volume_ratio[0] > 1.0:
                strength += 0.05
            
            # 3. Momentum (20%)
            if self.momentum[0] < 0:
                strength += 0.10
            if self.roc[0] < 0:
                strength += 0.10
            
            # 4. Position dans BB (15%)
            if self.bb_position[0] > 70:  # Proche du haut
                strength += 0.15
            elif self.bb_position[0] > 50:
                strength += 0.10
            
            # 5. Williams %R (15%)
            if self.williams_r[0] > -30:  # Surachat
                strength += 0.15
            elif self.williams_r[0] > -50:
                strength += 0.10
        
        # Bonus pour régime favorable
        if self.market_regime in ["TRENDING_UP", "TRENDING_DOWN"] and (long_signal or short_signal):
            strength += 0.1 * self.regime_confidence
        elif self.market_regime == "RANGING" and self.p.ranging_mode_enabled:
            strength += 0.05 * self.regime_confidence
        
        return min(1.0, strength)
    
    def get_regime_specific_signals_improved(self):
        """Signaux améliorés spécifiques au régime de marché"""
        long_signals = []
        short_signals = []
        
        # === SIGNAUX POUR TENDANCE HAUSSIÈRE ===
        if self.market_regime == "TRENDING_UP" and self.p.trending_mode_enabled:
            # Pullback de qualité dans tendance haussière
            pullback_conditions = (
                self.data.close[0] > self.ema_26[0] and  # Au-dessus de la tendance
                self.rsi[0] < 50 and self.rsi[-1] >= 50 and  # RSI pullback
                self.macd.macd[0] > self.macd.signal[0] and  # MACD positif
                self.bb_position[0] < 40  # Dans la partie basse des BB
            )
            if pullback_conditions:
                long_signals.append("TREND_PULLBACK")
        
        # === SIGNAUX POUR TENDANCE BAISSIÈRE ===
        elif self.market_regime == "TRENDING_DOWN" and self.p.trending_mode_enabled:
            # Pullback de qualité dans tendance baissière
            pullback_conditions = (
                self.data.close[0] < self.ema_26[0] and  # En-dessous de la tendance
                self.rsi[0] > 50 and self.rsi[-1] <= 50 and  # RSI pullback
                self.macd.macd[0] < self.macd.signal[0] and  # MACD négatif
                self.bb_position[0] > 60  # Dans la partie haute des BB
            )
            if pullback_conditions:
                short_signals.append("TREND_PULLBACK")
        
        # === SIGNAUX POUR MARCHÉ EN RANGE ===
        elif self.market_regime == "RANGING" and self.p.ranging_mode_enabled:
            # Mean reversion de qualité
            range_long_conditions = (
                self.bb_position[0] < 20 and  # Très proche du bas
                self.williams_r[0] < -80 and  # Très survente
                self.rsi[0] < 30 and  # RSI survente
                self.volume_ratio[0] > 1.0  # Volume confirmé
            )
            
            range_short_conditions = (
                self.bb_position[0] > 80 and  # Très proche du haut
                self.williams_r[0] > -20 and  # Très surachat
                self.rsi[0] > 70 and  # RSI surachat
                self.volume_ratio[0] > 1.0  # Volume confirmé
            )
            
            if range_long_conditions:
                long_signals.append("RANGE_REVERSAL")
            elif range_short_conditions:
                short_signals.append("RANGE_REVERSAL")
        
        # === SIGNAUX DE BREAKOUT (TOUS RÉGIMES) ===
        if self.p.breakout_mode_enabled:
            # Breakout avec confirmation multiple
            breakout_up_conditions = (
                self.data.close[0] > self.bb.top[0] and  # Breakout BB
                self.data.close[-1] <= self.bb.top[-1] and  # Nouveau breakout
                self.volume_ratio[0] > 1.5 and  # Volume fort
                self.rsi[0] > 50 and  # Momentum positif
                self.atr_ratio[0] > 1.1  # Volatilité en expansion
            )
            
            breakout_down_conditions = (
                self.data.close[0] < self.bb.bot[0] and  # Breakout BB
                self.data.close[-1] >= self.bb.bot[-1] and  # Nouveau breakout
                self.volume_ratio[0] > 1.5 and  # Volume fort
                self.rsi[0] < 50 and  # Momentum négatif
                self.atr_ratio[0] > 1.1  # Volatilité en expansion
            )
            
            if breakout_up_conditions:
                long_signals.append("BREAKOUT")
            elif breakout_down_conditions:
                short_signals.append("BREAKOUT")
        
        return long_signals, short_signals
    
    def check_anti_whipsaw_filters(self):
        """Filtres anti-whipsaw améliorés"""
        if not self.p.avoid_whipsaws:
            return True
        
        # 1. Éviter les signaux trop fréquents
        current_bar = len(self.data)
        if self.last_signal_time is not None:
            bars_since_last = current_bar - self.last_signal_time
            if bars_since_last < self.p.min_bars_between_signals:
                return False
        
        # 2. Éviter les signaux en cas de trop de faux signaux récents
        if self.false_signal_count > 5 and len(self.trade_history) > 0:
            recent_trades = self.trade_history[-5:]
            recent_losses = sum(1 for t in recent_trades if t['profit'] < 0)
            if recent_losses >= 4:  # 4 pertes sur 5 derniers trades
                return False
        
        # 3. Éviter les signaux en volatilité extrême sans confirmation
        if self.volatility_regime == "HIGH" and self.volume_regime == "LOW":
            return False
        
        # 4. Éviter les signaux contradictoires avec le régime
        if self.market_regime == "RANGING" and self.ranging_score < 3.0:
            return False
        
        return True
    
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
            multiplier = 0.7  # Plus permissif en volatilité
        elif self.market_regime == "RANGING":
            multiplier = 1.3  # Plus strict en range
        
        return stop_gap > (median * multiplier)
    
    def check_improved_rules(self):
        """Vérification des règles améliorées"""
        current_value = self.broker.getvalue()
        dt = self.datas[0].datetime.datetime(0)
        
        # Reset daily tracking
        if self.current_date != dt.date():
            self.current_date = dt.date()
            self.daily_start_cash = current_value
            self.false_signal_count = 0  # Reset daily
        
        # Update peak
        self.peak_value = max(self.peak_value, current_value)
        
        # Check consecutive losses (strict)
        if self.consecutive_losses >= self.p.max_consecutive_losses:
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
        """Calcul de la taille de position dynamique améliorée"""
        account_value = self.broker.getvalue()
        
        # Taille de base adaptée
        base_size = self.position_size
        
        # Ajustement basé sur la performance récente (conservateur)
        if self.consecutive_wins >= 2:
            base_size *= 1.1
        elif self.consecutive_losses >= 1:
            base_size *= 0.8
        
        # Ajustement basé sur la confiance du régime
        base_size *= (0.7 + 0.3 * self.regime_confidence)
        
        # Ajustement basé sur la force du signal
        base_size *= (0.5 + 0.5 * self.signal_strength)
        
        # Ajustement basé sur la volatilité
        if self.volatility_regime == "HIGH":
            base_size *= 0.7
        elif self.volatility_regime == "LOW":
            base_size *= 1.2
        
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
            # Limite conservatrice
            max_pct = 0.1  # Maximum 10% du capital
            if self.data.close[0] > 0:
                max_size = account_value * max_pct / self.data.close[0]
                size = min(size, max_size)
            return max(1, int(size))
        return 1
    
    def next(self):
        # Détection du régime de marché améliorée
        self.detect_market_regime_improved()
        
        # Vérification des règles améliorées
        if not self.check_improved_rules():
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
        
        # P1 Conditions (plus strictes)
        p1_margin = 0.08  # Marge plus large
        b1_ge_p1 = self.b1[0] >= self.p1[0] * (1 + p1_margin)
        b1_le_p1 = self.b1[0] <= self.p1[0] * (1 - p1_margin)
        
        # StopGap Filter
        stopgap_ok = self.calculate_stopgap_filter()
        
        # Signaux spécifiques au régime améliorés
        regime_long_signals, regime_short_signals = self.get_regime_specific_signals_improved()
        
        # === CONDITIONS AMÉLIORÉES ===
        
        # Conditions de base (plus strictes)
        base_long = (self.williams_long_active and crossmacdbear and b1_ge_p1 and 
                    self.trend == 1 and stopgap_ok)
        base_short = (self.williams_short_active and crossmacd and b1_le_p1 and 
                     self.trend == -1 and stopgap_ok)
        
        # Conditions finales
        long_condition = base_long or len(regime_long_signals) > 0
        short_condition = base_short or len(regime_short_signals) > 0
        
        # Calcul de la force du signal
        self.signal_strength = self.calculate_signal_strength_improved(long_condition, short_condition)
        
        # Filtres de qualité
        signal_quality_ok = self.signal_strength >= self.p.min_signal_strength
        anti_whipsaw_ok = self.check_anti_whipsaw_filters()
        
        # Appliquer tous les filtres
        long_condition = long_condition and signal_quality_ok and anti_whipsaw_ok
        short_condition = short_condition and signal_quality_ok and anti_whipsaw_ok
        
        # === EXÉCUTION DES TRADES ===
        
        if self.order:
            return
        
        if not self.position:
            if long_condition:
                self.total_signals += 1
                self.last_signal_time = len(self.data)
                size = self.calculate_dynamic_position_size()
                signal_type = regime_long_signals[0] if regime_long_signals else "BASE"
                self.log(f'🚀 IMPROVED BUY @ {self.data.close[0]:.5f} (Size: {size}) [Regime: {self.market_regime}, Signal: {signal_type}, Strength: {self.signal_strength:.2f}]')
                self.order = self.buy(size=size)
                self.buy_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
                
            elif short_condition:
                self.total_signals += 1
                self.last_signal_time = len(self.data)
                size = self.calculate_dynamic_position_size()
                signal_type = regime_short_signals[0] if regime_short_signals else "BASE"
                self.log(f'🚀 IMPROVED SELL @ {self.data.close[0]:.5f} (Size: {size}) [Regime: {self.market_regime}, Signal: {signal_type}, Strength: {self.signal_strength:.2f}]')
                self.order = self.sell(size=size)
                self.sell_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
        
        else:  # Position ouverte - gestion améliorée
            if self.position.size > 0:  # Position longue
                # Exit adaptatif selon le régime
                if self.p.use_regime_specific_exits:
                    if self.market_regime == "RANGING":
                        # Exit rapide en range
                        if self.bb_position[0] > 75 or self.williams_r[0] > -25:
                            self.log(f'💰 RANGE EXIT @ {self.data.close[0]:.5f}')
                            self.order = self.close()
                            return
                    elif self.market_regime in ["TRENDING_UP", "TRENDING_DOWN"]:
                        # Exit sur changement de tendance
                        if self.trend == -1:
                            self.log(f'🔄 TREND CHANGE EXIT @ {self.data.close[0]:.5f}')
                            self.order = self.close()
                            return
                
                # Exits standards
                if (self.position_entry_price and 
                    self.data.close[0] >= self.position_entry_price * 1.02):  # 2% profit
                    self.log(f'💰 PROFIT EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                elif self.rsi[0] > 75:
                    self.log(f'📈 RSI EXIT @ {self.data.close[0]:.5f} (RSI: {self.rsi[0]:.1f})')
                    self.order = self.close()
                elif (self.williams_long_stop and 
                      self.data.low[0] <= self.williams_long_stop):
                    self.log(f'🛑 LONG STOP @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                elif crossmacd:
                    self.log(f'🔄 LONG EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                    
            elif self.position.size < 0:  # Position courte
                # Exit adaptatif selon le régime
                if self.p.use_regime_specific_exits:
                    if self.market_regime == "RANGING":
                        # Exit rapide en range
                        if self.bb_position[0] < 25 or self.williams_r[0] < -75:
                            self.log(f'💰 RANGE EXIT @ {self.data.close[0]:.5f}')
                            self.order = self.close()
                            return
                    elif self.market_regime in ["TRENDING_UP", "TRENDING_DOWN"]:
                        # Exit sur changement de tendance
                        if self.trend == 1:
                            self.log(f'🔄 TREND CHANGE EXIT @ {self.data.close[0]:.5f}')
                            self.order = self.close()
                            return
                
                # Exits standards
                if (self.position_entry_price and 
                    self.data.close[0] <= self.position_entry_price * 0.98):  # 2% profit
                    self.log(f'💰 PROFIT EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                elif self.rsi[0] < 25:
                    self.log(f'📉 RSI EXIT @ {self.data.close[0]:.5f} (RSI: {self.rsi[0]:.1f})')
                    self.order = self.close()
                elif (self.williams_short_stop and 
                      self.data.high[0] >= self.williams_short_stop):
                    self.log(f'🛑 SHORT STOP @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                elif crossmacdbear:
                    self.log(f'🔄 SHORT EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
    
    def stop(self):
        """Statistiques finales améliorées"""
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
        
        self.log(f'=== RÉSULTATS MASTERTREND HYBRIDE AMÉLIORÉ ===')
        self.log(f'💰 Capital Initial: ${self.initial_cash:.2f}')
        self.log(f'💰 Capital Final: ${final_value:.2f}')
        self.log(f'📈 Rendement Total: {total_return:.2f}%')
        self.log(f'📉 Drawdown Max: {max_dd:.2f}%')
        self.log(f'🎯 Signaux Générés: {self.total_signals}')
        self.log(f'🟢 Signaux BUY: {len(self.buy_signals)}')
        self.log(f'🔴 Signaux SELL: {len(self.sell_signals)}')
        self.log(f'📊 Trades Total: {len(self.trade_history)}')
        
        # Analyse par régime améliorée
        self.log(f'\n=== PERFORMANCE PAR RÉGIME ===')
        for regime, stats in self.performance_by_regime.items():
            if stats["trades"] > 0:
                avg_profit = stats["profit"] / stats["trades"]
                self.log(f'{regime}: {stats["trades"]} trades, Profit Moyen: ${avg_profit:.2f}, Win Rate: {stats["win_rate"]:.1f}%')
        
        # Analyse des trades
        if self.trade_history and len(self.trade_history) > 0:
            profitable_trades = [t for t in self.trade_history if t['profit'] > 0]
            losing_trades = [t for t in self.trade_history if t['profit'] < 0]
            
            win_rate = len(profitable_trades) / len(self.trade_history) * 100
            avg_profit = sum(t['profit'] for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0
            avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            avg_roi = sum(t['roi'] for t in self.trade_history) / len(self.trade_history)
            avg_signal_strength = sum(t.get('signal_strength', 0) for t in self.trade_history) / len(self.trade_history)
            
            self.log(f'\n=== ANALYSE DÉTAILLÉE ===')
            self.log(f'🎯 Taux de Réussite: {win_rate:.1f}%')
            self.log(f'💰 Profit Moyen: ${avg_profit:.2f}')
            self.log(f'💸 Perte Moyenne: ${avg_loss:.2f}')
            self.log(f'📊 ROI Moyen: {avg_roi:.2f}%')
            self.log(f'⚡ Force Signal Moyenne: {avg_signal_strength:.2f}')
            if avg_loss != 0:
                profit_factor = abs(avg_profit / avg_loss)
                self.log(f'⚡ Profit Factor: {profit_factor:.2f}')
        
        # Évaluation finale
        if total_return >= 10:
            self.log('\n🏆 EXCELLENT! Stratégie hybride améliorée très performante!')
        elif total_return >= 5:
            self.log('\n✅ BON! Stratégie hybride améliorée efficace')
        elif total_return >= 2:
            self.log('\n⚠️  MOYEN. Optimisation possible')
        else:
            self.log('\n❌ INSUFFISANT. Révision nécessaire')


if __name__ == '__main__':
    # Test sur différentes timeframes
    timeframes = [
        ("15M", "EURUSD_data_15M.csv", bt.TimeFrame.Minutes, 15),
        ("1M", "EURUSD_data_1M.csv", bt.TimeFrame.Minutes, 1),
    ]
    
    for tf_name, filename, timeframe, compression in timeframes:
        print(f'\n{"="*60}')
        print(f'🚀 TESTING MASTERTREND HYBRID IMPROVED - {tf_name}')
        print(f'🧬 Mode: ADAPTATIF HYBRIDE AMÉLIORÉ')
        print(f'🎯 Capital Initial: $10,000')
        print(f'📊 Timeframe: {tf_name}')
        print('='*60)
        
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
            cerebro.addstrategy(ImprovedHybridMasterTrend)
            
            # Configuration
            cerebro.broker.setcash(10000.0)
            cerebro.broker.setcommission(commission=0.0001)
            
            # Analyseurs
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            results = cerebro.run()
            strat = results[0]
            
            # Analyse détaillée des résultats
            trades = strat.analyzers.trades.get_analysis()
            drawdown = strat.analyzers.drawdown.get_analysis()
            returns = strat.analyzers.returns.get_analysis()
            
            print(f'\n🔥 ANALYSE MASTERTREND HYBRID IMPROVED ({tf_name}):')
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
            if final_return >= 8:
                print(f'\n🏆 EXCELLENT! Stratégie hybride améliorée performante sur {tf_name}!')
            elif final_return >= 3:
                print(f'\n✅ Bon résultat sur {tf_name}!')
            else:
                print(f'\n⚠️  Résultat à améliorer sur {tf_name}.')
                
        except Exception as e:
            print(f'❌ ERREUR LORS DE L\'EXÉCUTION sur {tf_name}: {e}')
            print(f'Vérifiez que le fichier {filename} existe et a le bon format.') 