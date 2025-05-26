#!/usr/bin/env python3
"""
MASTERTREND PROFIT HUNTER - OPTIMIZED VERSION
Version ultra-optimisée pour maximiser les profits
Objectif: Générer 15-30% de rendement minimum
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
from collections import deque


class OptimizedProfitHunterMasterTrend(bt.Strategy):
    """
    Stratégie MasterTrend OPTIMISÉE pour maximiser les profits
    """
    
    params = (
        # SuperTrend OPTIMISÉ
        ('st_period', 6),            # Plus réactif
        ('st_multiplier', 1.8),      # Plus sensible
        
        # MACD OPTIMISÉ
        ('macd_fast', 6),            # Plus rapide
        ('macd_slow', 15),           # Plus rapide
        ('macd_signal', 4),          # Plus rapide
        
        # Williams Fractals OPTIMISÉS
        ('williams_left', 2),        # Équilibré
        ('williams_right', 2),       # Équilibré
        ('williams_buffer', 0.05),   # Buffer plus agressif
        
        # P1 Indicator OPTIMISÉ
        ('p1_e31', 2),              # Très rapide
        ('p1_m', 4),                # Très rapide
        ('p1_l31', 8),              # Très rapide
        
        # StopGap Filter OPTIMISÉ
        ('sg_ema_short', 4),         # Très rapide
        ('sg_ema_long', 12),         # Rapide
        ('sg_lookback', 2),          # Très court
        ('sg_median_period', 6),     # Très court
        ('sg_multiplier', 0.6),      # Très permissif
        
        # Filtres OPTIMISÉS
        ('rsi_period', 8),           # Très réactif
        ('rsi_oversold', 25),        # Plus permissif
        ('rsi_overbought', 75),      # Plus permissif
        ('volume_ma_period', 10),    # Plus court
        ('min_volume_ratio', 0.7),   # Plus permissif
        
        # Risk Management OPTIMISÉ
        ('max_daily_loss', 0.12),    # 12% perte journalière max
        ('max_total_loss', 0.20),    # 20% perte totale max
        ('profit_target', 0.35),     # 35% objectif de profit
        ('position_size', 0.08),     # 8% par trade (TRÈS AGRESSIF)
        ('max_consecutive_losses', 6), # Plus de tolérance
        
        # Sessions OPTIMISÉES
        ('trading_start', datetime.time(6, 0)),   # Très tôt
        ('trading_end', datetime.time(20, 0)),    # Très tard
        
        # Filtres de marché OPTIMISÉS
        ('min_atr_ratio', 0.2),      # Très permissif
        ('max_atr_ratio', 4.0),      # Très permissif
        ('trend_strength_period', 8), # Plus court
        ('min_trend_strength', 0.2),   # Très permissif
        
        # PARAMÈTRES ULTRA-AGRESSIFS
        ('use_leverage', True),       # Utiliser l'effet de levier
        ('leverage_factor', 2.5),     # Facteur de levier plus élevé
        ('scalping_mode', True),      # Mode scalping
        ('quick_exit_rsi', 80),       # Sortie rapide moins stricte
        ('quick_exit_profit', 0.015), # Sortie rapide à 1.5% de profit
        ('martingale_mode', True),    # Mode martingale après pertes
        ('martingale_multiplier', 1.8), # Multiplicateur martingale plus élevé
        ('trend_following', True),    # Suivre la tendance aggressivement
        ('breakout_trading', True),   # Trading de breakout
        ('momentum_trading', True),   # Trading de momentum
        ('volatility_trading', True), # Trading de volatilité
        
        # NOUVEAUX PARAMÈTRES D'OPTIMISATION
        ('dynamic_sizing', True),     # Taille dynamique basée sur volatilité
        ('adaptive_stops', True),     # Stops adaptatifs
        ('multi_timeframe', True),    # Analyse multi-timeframe
        ('correlation_filter', True), # Filtre de corrélation
        ('volatility_breakout', True), # Breakout de volatilité
    )
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # === INDICATEURS OPTIMISÉS ===
        
        # SuperTrend optimisé
        self.hl2 = (self.data.high + self.data.low) / 2.0
        self.atr = bt.indicators.ATR(self.data, period=self.p.st_period)
        self.atr_ma = bt.indicators.SMA(self.atr, period=8)  # Plus court
        
        # Variables SuperTrend
        self.basic_up = self.hl2 - self.p.st_multiplier * self.atr
        self.basic_dn = self.hl2 + self.p.st_multiplier * self.atr
        
        # MACD optimisé
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        
        # RSI optimisé
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        
        # Volume optimisé
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
        
        # Williams Fractals optimisés
        self.fractal_high = bt.indicators.Highest(self.data.high, period=self.p.williams_left + self.p.williams_right + 1)
        self.fractal_low = bt.indicators.Lowest(self.data.low, period=self.p.williams_left + self.p.williams_right + 1)
        
        # P1 Indicator optimisé
        self.hilow = (self.data.high - self.data.low) * 100
        self.openclose = (self.data.close - self.data.open) * 100
        self.spreadv = self.openclose * self.data.close
        
        # Période plus courte pour plus de réactivité
        self.pt_approx = bt.indicators.SMA(self.spreadv, period=30)  # Plus court
        
        self.ema_e31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_e31)
        self.ema_m = bt.indicators.EMA(self.pt_approx, period=self.p.p1_m)
        self.ema_l31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_l31)
        
        self.a1 = self.ema_l31 - self.ema_m
        self.b1 = self.ema_e31 - self.ema_m
        self.p1 = self.a1 + self.b1
        
        # StopGap Filter optimisé
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_long)
        self.highest_sg = bt.indicators.Highest(self.data.high, period=self.p.sg_lookback)
        self.lowest_sg = bt.indicators.Lowest(self.data.low, period=self.p.sg_lookback)
        
        # Indicateurs additionnels optimisés
        self.close_ma = bt.indicators.SMA(self.data.close, period=self.p.trend_strength_period)
        self.momentum = bt.indicators.Momentum(self.data.close, period=3)  # Plus rapide
        self.stoch = bt.indicators.Stochastic(self.data, period=6)  # Plus rapide
        
        # Détection de breakout optimisée
        self.bb = bt.indicators.BollingerBands(self.data.close, period=12, devfactor=1.5)  # Plus sensible
        
        # Indicateurs de volatilité
        self.volatility = bt.indicators.StdDev(self.data.close, period=10)
        self.volatility_ma = bt.indicators.SMA(self.volatility, period=5)
        
        # Multi-timeframe (simulation avec moyennes mobiles)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=5)
        self.ema_medium = bt.indicators.EMA(self.data.close, period=15)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=30)
        
        # === VARIABLES D'ÉTAT OPTIMISÉES ===
        self.trend = 1
        self.final_up = 0.0
        self.final_dn = 0.0
        
        # Williams Stops
        self.williams_long_stop = None
        self.williams_short_stop = None
        self.williams_long_active = False
        self.williams_short_active = False
        
        # Tracking optimisé
        self.initial_cash = self.broker.getvalue()
        self.daily_start_cash = self.initial_cash
        self.peak_value = self.initial_cash
        self.current_date = None
        
        # Trade Management optimisé
        self.order = None
        self.position_entry_price = None
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.last_trade_profit = 0
        self.martingale_size = 1.0
        
        # StopGap median calculation
        self.candle_sizes = deque(maxlen=self.p.sg_median_period)
        
        # Statistiques optimisées
        self.buy_signals = []
        self.sell_signals = []
        self.trade_history = []
        self.total_signals = 0
        self.profitable_signals = 0
        
        # Variables d'optimisation
        self.volatility_threshold = 0.001
        self.momentum_threshold = 0.0003
        self.trend_strength = 0.0
    
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
                self.martingale_size = 1.0  # Reset martingale
                self.profitable_signals += 1
                roi = trade.pnl/abs(trade.value)*100 if trade.value != 0 else 0
                self.log(f'💰 PROFIT: ${trade.pnl:.2f} (ROI: {roi:.1f}%) [Win Streak: {self.consecutive_wins}]')
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                if self.p.martingale_mode:
                    self.martingale_size *= self.p.martingale_multiplier
                    self.martingale_size = min(self.martingale_size, 4.0)  # Limite à 4x
                self.log(f'💸 LOSS: ${trade.pnl:.2f} (Consecutive: {self.consecutive_losses})')
            
            # Enregistrer le trade
            self.trade_history.append({
                'profit': trade.pnl,
                'duration': trade.barlen,
                'entry_price': self.position_entry_price,
                'roi': trade.pnl/abs(trade.value)*100 if trade.value != 0 else 0
            })
    
    def update_supertrend(self):
        """Mise à jour SuperTrend optimisé"""
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
        """Mise à jour Williams Stops optimisés"""
        if len(self.data) > self.p.williams_left + self.p.williams_right:
            center_idx = self.p.williams_right
            
            # High fractal (optimisé)
            is_high_fractal = True
            center_high = self.data.high[-center_idx]
            
            for i in range(self.p.williams_left + self.p.williams_right + 1):
                if i != center_idx and self.data.high[-i] >= center_high:
                    is_high_fractal = False
                    break
            
            if is_high_fractal:
                self.williams_short_stop = center_high * (1 + self.p.williams_buffer / 100)
                self.williams_short_active = True
            
            # Low fractal (optimisé)
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
        """Calcul du filtre StopGap optimisé"""
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
    
    def check_volatility_conditions(self):
        """Vérification des conditions de volatilité"""
        if not self.p.volatility_trading:
            return False, False
        
        # Volatilité élevée
        high_volatility = self.volatility[0] > self.volatility_ma[0] * 1.3
        
        # Expansion de volatilité
        volatility_expansion = self.volatility[0] > self.volatility[-1] * 1.2
        
        # Breakout de Bollinger Bands avec volatilité
        bb_squeeze = (self.bb.top[0] - self.bb.bot[0]) < (self.bb.top[-5] - self.bb.bot[-5]) * 0.8
        bb_expansion = (self.bb.top[0] - self.bb.bot[0]) > (self.bb.top[-1] - self.bb.bot[-1]) * 1.1
        
        volatility_long = high_volatility and volatility_expansion and self.data.close[0] > self.bb.mid[0]
        volatility_short = high_volatility and volatility_expansion and self.data.close[0] < self.bb.mid[0]
        
        return volatility_long, volatility_short
    
    def check_momentum_conditions(self):
        """Vérification des conditions de momentum"""
        if not self.p.momentum_trading:
            return False, False
        
        # Momentum fort
        strong_momentum_up = self.momentum[0] > self.momentum_threshold and self.momentum[0] > self.momentum[-1]
        strong_momentum_down = self.momentum[0] < -self.momentum_threshold and self.momentum[0] < self.momentum[-1]
        
        # Alignement des moyennes mobiles
        ma_alignment_up = self.ema_fast[0] > self.ema_medium[0] > self.ema_slow[0]
        ma_alignment_down = self.ema_fast[0] < self.ema_medium[0] < self.ema_slow[0]
        
        momentum_long = strong_momentum_up and ma_alignment_up
        momentum_short = strong_momentum_down and ma_alignment_down
        
        return momentum_long, momentum_short
    
    def check_breakout_conditions(self):
        """Détection de breakout optimisée"""
        if not self.p.breakout_trading:
            return False, False
        
        # Breakout Bollinger Bands
        bb_breakout_up = self.data.close[0] > self.bb.top[0] and self.data.close[-1] <= self.bb.top[-1]
        bb_breakout_down = self.data.close[0] < self.bb.bot[0] and self.data.close[-1] >= self.bb.bot[-1]
        
        # Volume breakout
        volume_breakout = self.data.volume[0] > self.volume_ma[0] * 1.3
        
        # Momentum breakout
        momentum_up = self.momentum[0] > 0.0003
        momentum_down = self.momentum[0] < -0.0003
        
        # ATR breakout
        atr_expansion = self.atr[0] > self.atr_ma[0] * 1.2
        
        long_breakout = bb_breakout_up and volume_breakout and momentum_up and atr_expansion
        short_breakout = bb_breakout_down and volume_breakout and momentum_down and atr_expansion
        
        return long_breakout, short_breakout
    
    def check_scalping_conditions(self):
        """Conditions de scalping optimisées"""
        if not self.p.scalping_mode:
            return False, False
        
        # RSI rapide avec divergence
        rsi_oversold_bounce = self.rsi[0] > 30 and self.rsi[-1] <= 30 and self.rsi[-2] <= self.rsi[-1]
        rsi_overbought_drop = self.rsi[0] < 70 and self.rsi[-1] >= 70 and self.rsi[-2] >= self.rsi[-1]
        
        # Stochastic rapide
        stoch_oversold = self.stoch.percK[0] > 25 and self.stoch.percK[-1] <= 25
        stoch_overbought = self.stoch.percK[0] < 75 and self.stoch.percK[-1] >= 75
        
        # Momentum rapide
        momentum_reversal_up = self.momentum[0] > 0 and self.momentum[-1] <= 0
        momentum_reversal_down = self.momentum[0] < 0 and self.momentum[-1] >= 0
        
        scalp_long = (rsi_oversold_bounce or stoch_oversold) and momentum_reversal_up
        scalp_short = (rsi_overbought_drop or stoch_overbought) and momentum_reversal_down
        
        return scalp_long, scalp_short
    
    def check_aggressive_rules(self):
        """Vérification des règles optimisées"""
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
        if self.initial_cash == 0:
            daily_pnl = 0.0
        else:
            daily_pnl = (current_value - self.daily_start_cash) / self.initial_cash
        if daily_pnl < -self.p.max_daily_loss:
            self.log(f'⚠️  DAILY LOSS LIMIT HIT: {daily_pnl*100:.2f}%')
            return False
        
        # Check total drawdown (plus tolérant)
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
        """Vérification session de trading optimisée"""
        dt = self.datas[0].datetime.datetime(0)
        current_time = dt.time()
        return self.p.trading_start <= current_time <= self.p.trading_end
    
    def calculate_dynamic_position_size(self):
        """Calcul de la taille de position dynamique optimisée"""
        account_value = self.broker.getvalue()
        
        # Taille de base optimisée
        base_size = self.p.position_size
        
        # Ajustement basé sur la performance récente
        if self.consecutive_wins >= 3:
            base_size *= 1.2  # Augmenter après des gains
        elif self.consecutive_losses >= 2:
            base_size *= 0.8  # Réduire après des pertes
        
        # Ajustement basé sur la volatilité
        if self.p.dynamic_sizing:
            volatility_ratio = self.volatility[0] / self.volatility_ma[0] if self.volatility_ma[0] > 0 else 1.0
            if volatility_ratio > 1.5:
                base_size *= 1.3  # Augmenter en haute volatilité
            elif volatility_ratio < 0.7:
                base_size *= 0.9  # Réduire en basse volatilité
        
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
            stop_distance = self.data.close[0] * 0.003  # 0.3% par défaut (plus agressif)
        
        if stop_distance > 0:
            size = risk_amount / stop_distance
            # Limite plus élevée pour l'optimisation
            if self.data.close[0] > 0:
                max_size = account_value * 0.4 / self.data.close[0]  # Maximum 40% du capital
                size = min(size, max_size)
            return max(1, int(size))
        return 1
    
    def next(self):
        # Vérification des règles optimisées
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
        
        # Calcul des signaux optimisés
        
        # MACD Crossovers (plus sensibles)
        crossmacdbear = (self.macd.macd[0] > 0.00003 and self.macd.macd[-1] <= 0.00003)
        crossmacd = (self.macd.macd[0] < -0.00003 and self.macd.macd[-1] >= -0.00003)
        
        # P1 Conditions (plus permissives)
        b1_ge_p1 = self.b1[0] >= self.p1[0] * 1.01  # 1% de marge seulement
        b1_le_p1 = self.b1[0] <= self.p1[0] * 0.99  # 1% de marge seulement
        
        # StopGap Filter
        stopgap_ok = self.calculate_stopgap_filter()
        
        # Conditions optimisées
        breakout_long, breakout_short = self.check_breakout_conditions()
        scalp_long, scalp_short = self.check_scalping_conditions()
        volatility_long, volatility_short = self.check_volatility_conditions()
        momentum_long, momentum_short = self.check_momentum_conditions()
        
        # === CONDITIONS ULTRA-OPTIMISÉES ===
        
        long_condition = (
            (self.williams_long_active and crossmacdbear and b1_ge_p1 and self.trend == 1 and stopgap_ok) or
            breakout_long or
            scalp_long or
            volatility_long or
            momentum_long
        )
        
        short_condition = (
            (self.williams_short_active and crossmacd and b1_le_p1 and self.trend == -1 and stopgap_ok) or
            breakout_short or
            scalp_short or
            volatility_short or
            momentum_short
        )
        
        # === EXÉCUTION DES TRADES OPTIMISÉS ===
        
        if self.order:
            return
        
        if not self.position:
            if long_condition:
                self.total_signals += 1
                size = self.calculate_dynamic_position_size()
                self.log(f'🚀 OPTIMIZED BUY @ {self.data.close[0]:.5f} (Size: {size}, RSI: {self.rsi[0]:.1f})')
                self.order = self.buy(size=size)
                self.buy_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
                
            elif short_condition:
                self.total_signals += 1
                size = self.calculate_dynamic_position_size()
                self.log(f'🚀 OPTIMIZED SELL @ {self.data.close[0]:.5f} (Size: {size}, RSI: {self.rsi[0]:.1f})')
                self.order = self.sell(size=size)
                self.sell_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
        
        else:  # Position ouverte - gestion optimisée
            if self.position.size > 0:  # Position longue
                # Quick exit sur profit (plus agressif)
                if (self.position_entry_price and 
                    self.data.close[0] >= self.position_entry_price * (1 + self.p.quick_exit_profit)):
                    self.log(f'💰 QUICK PROFIT EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Quick exit sur RSI extrême
                elif self.rsi[0] > self.p.quick_exit_rsi:
                    self.log(f'📈 RSI EXIT @ {self.data.close[0]:.5f} (RSI: {self.rsi[0]:.1f})')
                    self.order = self.close()
                # Stop loss Williams adaptatif
                elif (self.williams_long_stop and 
                      self.data.low[0] <= self.williams_long_stop):
                    self.log(f'🛑 LONG STOP @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Signal de sortie MACD
                elif crossmacd:
                    self.log(f'🔄 LONG EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Sortie sur momentum négatif
                elif self.momentum[0] < -self.momentum_threshold:
                    self.log(f'📉 MOMENTUM EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                    
            elif self.position.size < 0:  # Position courte
                # Quick exit sur profit (plus agressif)
                if (self.position_entry_price and 
                    self.data.close[0] <= self.position_entry_price * (1 - self.p.quick_exit_profit)):
                    self.log(f'💰 QUICK PROFIT EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Quick exit sur RSI extrême
                elif self.rsi[0] < (100 - self.p.quick_exit_rsi):
                    self.log(f'📉 RSI EXIT @ {self.data.close[0]:.5f} (RSI: {self.rsi[0]:.1f})')
                    self.order = self.close()
                # Stop loss Williams adaptatif
                elif (self.williams_short_stop and 
                      self.data.high[0] >= self.williams_short_stop):
                    self.log(f'🛑 SHORT STOP @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Signal de sortie MACD
                elif crossmacdbear:
                    self.log(f'🔄 SHORT EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Sortie sur momentum positif
                elif self.momentum[0] > self.momentum_threshold:
                    self.log(f'📈 MOMENTUM EXIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
    
    def stop(self):
        """Statistiques finales optimisées"""
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
        
        self.log(f'=== RÉSULTATS PROFIT HUNTER OPTIMISÉ ===')
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
        if total_return >= 15:
            self.log('🏆 EXCELLENT! Objectif de profit atteint!')
        elif total_return >= 8:
            self.log('✅ BON! Profit satisfaisant')
        elif total_return >= 3:
            self.log('⚠️  MOYEN. Peut mieux faire')
        else:
            self.log('❌ INSUFFISANT. Stratégie à revoir')


if __name__ == '__main__':
    # Configuration optimisée
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
    cerebro.addstrategy(OptimizedProfitHunterMasterTrend)
    
    # Configuration optimisée
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)
    
    # Analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print('🚀 MASTERTREND PROFIT HUNTER OPTIMIZED - DÉMARRAGE')
    print('💰 Objectif: 15-30% de rendement minimum')
    print('⚡ Mode: ULTRA-OPTIMISÉ')
    print('🎯 Capital Initial: $10,000')
    print('=' * 50)
    
    try:
        results = cerebro.run()
        strat = results[0]
        
        # Analyse détaillée des résultats
        trades = strat.analyzers.trades.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        
        print('\n🔥 ANALYSE PROFIT HUNTER OPTIMIZED:')
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
            print('\n🏆 MISSION ACCOMPLIE! Profit Hunter Optimized a réussi!')
        elif final_return >= 8:
            print('\n✅ Bon résultat! Objectif partiellement atteint')
        else:
            print('\n⚠️  Résultat insuffisant. Optimisation supplémentaire nécessaire.')
            
    except Exception as e:
        print(f'❌ ERREUR LORS DE L\'EXÉCUTION: {e}')
        print('Vérifiez que le fichier EURUSD_data_15M.csv existe et a le bon format.') 