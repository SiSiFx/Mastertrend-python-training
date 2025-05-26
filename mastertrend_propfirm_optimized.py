#!/usr/bin/env python3
"""
MASTERTREND PropFirm Strategy - Version OPTIMISÉE
Améliorations pour réduire le drawdown et augmenter le taux de réussite
Optimisée spécifiquement pour passer les challenges des prop firms
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
from collections import deque


class OptimizedPropFirmMasterTrend(bt.Strategy):
    """
    Stratégie MasterTrend OPTIMISÉE pour prop firms
    Filtres renforcés et gestion de risque améliorée
    """
    
    # === PARAMÈTRES OPTIMISÉS POUR PROP FIRMS ===
    params = (
        # SuperTrend (ajusté pour moins de faux signaux)
        ('st_period', 12),           # Réduit de 14 à 12 pour plus de réactivité
        ('st_multiplier', 2.8),      # Augmenté de 2.5 à 2.8 pour équilibrer
        
        # MACD (ajusté pour plus de précision)
        ('macd_fast', 10),           # Augmenté de 8 à 10
        ('macd_slow', 24),           # Augmenté de 21 à 24
        ('macd_signal', 7),          # Augmenté de 5 à 7
        
        # Williams Fractals (moins conservateur)
        ('williams_left', 2),        # Réduit de 3 à 2
        ('williams_right', 2),       # Réduit de 3 à 2
        ('williams_buffer', 0.05),   # Réduit de 0.1% à 0.05%
        
        # P1 Indicator (moins strict)
        ('p1_e31', 6),              # Réduit de 8 à 6
        ('p1_m', 11),               # Réduit de 13 à 11
        ('p1_l31', 18),             # Réduit de 21 à 18
        
        # StopGap Filter (moins restrictif)
        ('sg_ema_short', 9),         # Augmenté de 8 à 9
        ('sg_ema_long', 22),         # Réduit de 25 à 22
        ('sg_lookback', 5),          # Réduit de 6 à 5
        ('sg_median_period', 12),    # Réduit de 15 à 12
        ('sg_multiplier', 1.2),      # Réduit de 1.5 à 1.2 pour moins de restriction
        
        # Filtres additionnels (assouplis)
        ('rsi_period', 14),          
        ('rsi_oversold', 25),        # Réduit de 30 à 25
        ('rsi_overbought', 75),      # Augmenté de 70 à 75
        ('volume_ma_period', 20),    
        ('min_volume_ratio', 1.0),   # Réduit de 1.2 à 1.0 pour accepter plus de trades
        
        # Prop Firm Risk Management (équilibré)
        ('max_daily_loss', 0.018),   # Augmenté de 1.5% à 1.8%
        ('max_total_loss', 0.045),   # Augmenté de 4% à 4.5%
        ('profit_target', 0.08),     
        ('position_size', 0.01),     # Augmenté de 0.8% à 1%
        ('max_consecutive_losses', 3), 
        
        # Sessions (moins restrictives)
        ('trading_start', datetime.time(9, 30)),  # Retour à 9h30
        ('trading_end', datetime.time(16, 0)),    # Retour à 16h00
        
        # Filtres de marché (assouplis)
        ('min_atr_ratio', 0.4),      # Réduit de 0.5 à 0.4
        ('max_atr_ratio', 2.5),      # Augmenté de 2.0 à 2.5
        ('trend_strength_period', 15), # Réduit de 20 à 15
        ('min_trend_strength', 0.5),   # Réduit de 0.6 à 0.5
    )
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # === INDICATEURS PRINCIPAUX ===
        
        # SuperTrend (optimisé)
        self.hl2 = (self.data.high + self.data.low) / 2.0
        self.atr = bt.indicators.ATR(self.data, period=self.p.st_period)
        self.atr_ma = bt.indicators.SMA(self.atr, period=20)  # Moyenne ATR
        
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
        
        # RSI pour filtrer les zones extrêmes
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        
        # Volume filter
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
        
        # Williams Fractals (optimisés)
        self.fractal_high = bt.indicators.Highest(self.data.high, period=self.p.williams_left + self.p.williams_right + 1)
        self.fractal_low = bt.indicators.Lowest(self.data.low, period=self.p.williams_left + self.p.williams_right + 1)
        
        # P1 Indicator (optimisé)
        self.hilow = (self.data.high - self.data.low) * 100
        self.openclose = (self.data.close - self.data.open) * 100
        self.spreadv = self.openclose * self.data.close
        
        # Cumulative sum avec période plus longue
        self.pt_approx = bt.indicators.SMA(self.spreadv, period=75)
        
        self.ema_e31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_e31)
        self.ema_m = bt.indicators.EMA(self.pt_approx, period=self.p.p1_m)
        self.ema_l31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_l31)
        
        self.a1 = self.ema_l31 - self.ema_m
        self.b1 = self.ema_e31 - self.ema_m
        self.p1 = self.a1 + self.b1
        
        # StopGap Filter (optimisé)
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_long)
        self.highest_sg = bt.indicators.Highest(self.data.high, period=self.p.sg_lookback)
        self.lowest_sg = bt.indicators.Lowest(self.data.low, period=self.p.sg_lookback)
        
        # Trend Strength Indicator
        self.close_ma = bt.indicators.SMA(self.data.close, period=self.p.trend_strength_period)
        
        # === VARIABLES D'ÉTAT ===
        self.trend = 1
        self.final_up = 0.0
        self.final_dn = 0.0
        
        # Williams Stops
        self.williams_long_stop = None
        self.williams_short_stop = None
        self.williams_long_active = False
        self.williams_short_active = False
        
        # Prop Firm Tracking
        self.initial_cash = self.broker.getvalue()
        self.daily_start_cash = self.initial_cash
        self.peak_value = self.initial_cash
        self.current_date = None
        
        # Trade Management
        self.order = None
        self.position_entry_price = None
        self.consecutive_losses = 0
        self.last_trade_profit = 0
        
        # StopGap median calculation
        self.candle_sizes = deque(maxlen=self.p.sg_median_period)
        
        # Signaux et statistiques
        self.buy_signals = []
        self.sell_signals = []
        self.trade_history = []
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED @ {order.executed.price:.5f}')
                self.position_entry_price = order.executed.price
            else:
                self.log(f'SELL EXECUTED @ {order.executed.price:.5f}')
                self.position_entry_price = order.executed.price
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.last_trade_profit = trade.pnl
            if trade.pnl < 0:
                self.consecutive_losses += 1
                self.log(f'LOSS: {trade.pnl:.2f} (Consecutive: {self.consecutive_losses})')
            else:
                self.consecutive_losses = 0
                self.log(f'PROFIT: {trade.pnl:.2f}')
            
            # Enregistrer le trade
            self.trade_history.append({
                'profit': trade.pnl,
                'duration': trade.barlen,
                'entry_price': self.position_entry_price
            })
    
    def update_supertrend(self):
        """Mise à jour SuperTrend optimisée"""
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
        """Mise à jour Williams Stops optimisée"""
        if len(self.data) > self.p.williams_left + self.p.williams_right:
            center_idx = self.p.williams_right
            
            # High fractal (plus strict)
            is_high_fractal = True
            center_high = self.data.high[-center_idx]
            
            for i in range(self.p.williams_left + self.p.williams_right + 1):
                if i != center_idx and self.data.high[-i] >= center_high:
                    is_high_fractal = False
                    break
            
            if is_high_fractal:
                self.williams_short_stop = center_high * (1 + self.p.williams_buffer / 100)
                self.williams_short_active = True
            
            # Low fractal (plus strict)
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
        
        # Appliquer le multiplicateur pour être plus restrictif
        return stop_gap > (median * self.p.sg_multiplier)
    
    def calculate_trend_strength(self):
        """Calcul de la force de tendance"""
        if len(self.data) < self.p.trend_strength_period:
            return 0.5
        
        # Compter les barres dans la direction de la tendance
        trend_bars = 0
        for i in range(self.p.trend_strength_period):
            if self.trend == 1 and self.data.close[-i] > self.close_ma[-i]:
                trend_bars += 1
            elif self.trend == -1 and self.data.close[-i] < self.close_ma[-i]:
                trend_bars += 1
        
        return trend_bars / self.p.trend_strength_period
    
    def check_market_conditions(self):
        """Vérification des conditions de marché"""
        # Filtre ATR (éviter les marchés trop volatils ou trop calmes)
        atr_ratio = self.atr[0] / self.atr_ma[0] if self.atr_ma[0] > 0 else 1.0
        if atr_ratio < self.p.min_atr_ratio or atr_ratio > self.p.max_atr_ratio:
            return False
        
        # Filtre volume
        if self.data.volume[0] > 0 and self.volume_ma[0] > 0:
            volume_ratio = self.data.volume[0] / self.volume_ma[0]
            if volume_ratio < self.p.min_volume_ratio:
                return False
        
        # Force de tendance
        trend_strength = self.calculate_trend_strength()
        if trend_strength < self.p.min_trend_strength:
            return False
        
        return True
    
    def check_prop_firm_rules(self):
        """Vérification des règles prop firm optimisées"""
        current_value = self.broker.getvalue()
        dt = self.datas[0].datetime.datetime(0)
        
        # Reset daily tracking
        if self.current_date != dt.date():
            self.current_date = dt.date()
            self.daily_start_cash = current_value
            # Reset consecutive losses daily
            if self.consecutive_losses > 0:
                self.log(f'Daily reset - Consecutive losses: {self.consecutive_losses}')
        
        # Update peak
        self.peak_value = max(self.peak_value, current_value)
        
        # Check consecutive losses
        if self.consecutive_losses >= self.p.max_consecutive_losses:
            self.log(f'MAX CONSECUTIVE LOSSES HIT: {self.consecutive_losses}')
            return False
        
        # Check daily loss limit
        daily_pnl = (current_value - self.daily_start_cash) / self.initial_cash
        if daily_pnl < -self.p.max_daily_loss:
            self.log(f'DAILY LOSS LIMIT HIT: {daily_pnl*100:.2f}%')
            return False
        
        # Check total drawdown
        total_dd = (self.peak_value - current_value) / self.peak_value
        if total_dd > self.p.max_total_loss:
            self.log(f'TOTAL DRAWDOWN LIMIT HIT: {total_dd*100:.2f}%')
            return False
        
        # Check profit target
        total_profit = (current_value - self.initial_cash) / self.initial_cash
        if total_profit >= self.p.profit_target:
            self.log(f'PROFIT TARGET REACHED: {total_profit*100:.2f}%')
            return False
        
        return True
    
    def is_trading_session(self):
        """Vérification session de trading optimisée"""
        dt = self.datas[0].datetime.datetime(0)
        current_time = dt.time()
        return self.p.trading_start <= current_time <= self.p.trading_end
    
    def next(self):
        # Vérification des règles prop firm
        if not self.check_prop_firm_rules():
            if self.position:
                self.close()
            return
        
        # Vérification session de trading
        if not self.is_trading_session():
            return
        
        # Vérification des conditions de marché
        if not self.check_market_conditions():
            return
        
        # Mise à jour des indicateurs
        self.update_supertrend()
        self.update_williams_stops()
        
        # Calcul des signaux avec filtres renforcés
        
        # MACD Crossovers
        crossmacdbear = (self.macd.macd[0] > 0 and self.macd.macd[-1] <= 0)
        crossmacd = (self.macd.macd[0] < 0 and self.macd.macd[-1] >= 0)
        
        # P1 Conditions (moins strictes)
        b1_ge_p1 = self.b1[0] >= self.p1[0] * 1.05  # Réduit de 10% à 5% de marge
        b1_le_p1 = self.b1[0] <= self.p1[0] * 0.95  # Réduit de 10% à 5% de marge
        
        # StopGap Filter
        stopgap_ok = self.calculate_stopgap_filter()
        
        # Filtres RSI (assouplis)
        rsi_long_ok = self.rsi[0] < self.p.rsi_overbought and self.rsi[0] > 20  # Seuil minimum abaissé
        rsi_short_ok = self.rsi[0] > self.p.rsi_oversold and self.rsi[0] < 80   # Seuil maximum relevé
        
        # === CONDITIONS OPTIMISÉES ===
        
        long_condition = (
            self.williams_long_active and 
            crossmacdbear and 
            b1_ge_p1 and 
            self.trend == 1 and 
            stopgap_ok and
            rsi_long_ok and
            self.consecutive_losses < 3  # Augmenté de 2 à 3
        )
        
        short_condition = (
            self.williams_short_active and 
            crossmacd and 
            b1_le_p1 and 
            self.trend == -1 and 
            stopgap_ok and
            rsi_short_ok and
            self.consecutive_losses < 3  # Augmenté de 2 à 3
        )
        
        # === EXÉCUTION DES TRADES ===
        
        if self.order:
            return
        
        if not self.position:
            if long_condition:
                size = self.calculate_position_size()
                # Vérification de sécurité : ne jamais risquer plus de 0.5% du capital
                max_position_value = self.broker.getvalue() * 0.005  # 0.5% du capital
                position_value = size * self.data.close[0]
                if position_value <= max_position_value:
                    self.log(f'BUY SIGNAL @ {self.data.close[0]:.5f} (Size: {size}, RSI: {self.rsi[0]:.1f}, Trend: {self.calculate_trend_strength():.2f})')
                    self.order = self.buy(size=size)
                    self.buy_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
                else:
                    self.log(f'BUY SIGNAL REJECTED - Position too large: ${position_value:.2f} > ${max_position_value:.2f}')
                
            elif short_condition:
                size = self.calculate_position_size()
                # Vérification de sécurité : ne jamais risquer plus de 0.5% du capital
                max_position_value = self.broker.getvalue() * 0.005  # 0.5% du capital
                position_value = size * self.data.close[0]
                if position_value <= max_position_value:
                    self.log(f'SELL SIGNAL @ {self.data.close[0]:.5f} (Size: {size}, RSI: {self.rsi[0]:.1f}, Trend: {self.calculate_trend_strength():.2f})')
                    self.order = self.sell(size=size)
                    self.sell_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
                else:
                    self.log(f'SELL SIGNAL REJECTED - Position too large: ${position_value:.2f} > ${max_position_value:.2f}')
        
        else:  # Position ouverte - gestion optimisée des stops
            if self.position.size > 0:  # Position longue
                # Stop loss moins serré
                if (self.williams_long_stop and 
                    self.data.low[0] <= self.williams_long_stop * 0.995):  # 0.5% de buffer
                    self.log(f'LONG STOP HIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                elif crossmacd:
                    self.log(f'LONG EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Take profit moins agressif
                elif self.rsi[0] > 80:  # Augmenté de 75 à 80
                    self.log(f'LONG TAKE PROFIT (RSI: {self.rsi[0]:.1f}) @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                    
            elif self.position.size < 0:  # Position courte
                if (self.williams_short_stop and 
                    self.data.high[0] >= self.williams_short_stop * 1.005):  # 0.5% de buffer
                    self.log(f'SHORT STOP HIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                elif crossmacdbear:
                    self.log(f'SHORT EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                # Take profit moins agressif
                elif self.rsi[0] < 20:  # Réduit de 25 à 20
                    self.log(f'SHORT TAKE PROFIT (RSI: {self.rsi[0]:.1f}) @ {self.data.close[0]:.5f}')
                    self.order = self.close()
    
    def calculate_position_size(self):
        """Calcul de la taille de position optimisé avec protection stricte"""
        account_value = self.broker.getvalue()
        
        # Réduction moins agressive après des pertes consécutives
        risk_reduction = 1.0
        if self.consecutive_losses >= 2:
            risk_reduction = 0.7  # Réduire de 30% au lieu de 50% après 2 pertes
        elif self.consecutive_losses >= 1:
            risk_reduction = 0.85  # Réduire de 15% après 1 perte
        
        # Limiter strictement le risque à 0.3% du capital par trade (plus conservateur)
        max_risk_per_trade = account_value * 0.003  # 0.3% maximum
        risk_amount = min(account_value * self.p.position_size * risk_reduction, max_risk_per_trade)
        
        # Utiliser Williams Stop pour calculer la taille, mais avec des limites plus strictes
        if self.williams_long_stop:
            stop_distance = abs(self.data.close[0] - self.williams_long_stop)
        elif self.williams_short_stop:
            stop_distance = abs(self.data.close[0] - self.williams_short_stop)
        else:
            stop_distance = self.data.close[0] * 0.02  # 2% par défaut pour plus de sécurité
        
        # S'assurer que le stop distance n'est pas trop petit
        min_stop_distance = self.data.close[0] * 0.005  # Minimum 0.5%
        stop_distance = max(stop_distance, min_stop_distance)
        
        if stop_distance > 0:
            size = risk_amount / stop_distance
            # Limiter la taille absolue pour éviter les positions trop importantes
            max_size = account_value * 0.005 / self.data.close[0]  # Maximum 0.5% du capital en valeur
            size = min(size, max_size)
            return max(1, int(size))
        return 1
    
    def stop(self):
        """Statistiques finales optimisées"""
        final_value = self.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        max_dd = (self.peak_value - final_value) / self.peak_value * 100
        
        self.log(f'=== RÉSULTATS FINAUX OPTIMISÉS ===')
        self.log(f'Capital Initial: ${self.initial_cash:.2f}')
        self.log(f'Capital Final: ${final_value:.2f}')
        self.log(f'Rendement Total: {total_return:.2f}%')
        self.log(f'Drawdown Max: {max_dd:.2f}%')
        self.log(f'Signaux BUY: {len(self.buy_signals)}')
        self.log(f'Signaux SELL: {len(self.sell_signals)}')
        self.log(f'Trades Total: {len(self.trade_history)}')
        self.log(f'Pertes Consécutives Max: {self.consecutive_losses}')
        
        # Analyse des trades
        if self.trade_history:
            profitable_trades = [t for t in self.trade_history if t['profit'] > 0]
            win_rate = len(profitable_trades) / len(self.trade_history) * 100
            avg_profit = sum(t['profit'] for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0
            avg_loss = sum(t['profit'] for t in self.trade_history if t['profit'] < 0) / len([t for t in self.trade_history if t['profit'] < 0]) if any(t['profit'] < 0 for t in self.trade_history) else 0
            
            self.log(f'Taux de Réussite: {win_rate:.1f}%')
            self.log(f'Profit Moyen: ${avg_profit:.2f}')
            self.log(f'Perte Moyenne: ${avg_loss:.2f}')
            if avg_loss != 0:
                profit_factor = abs(avg_profit / avg_loss)
                self.log(f'Profit Factor: {profit_factor:.2f}')
        
        # Vérification des objectifs prop firm
        if total_return >= self.p.profit_target * 100:
            self.log('✅ OBJECTIF PROFIT ATTEINT - PROP FIRM CHALLENGE RÉUSSI!')
        elif max_dd <= self.p.max_total_loss * 100:
            self.log('✅ DRAWDOWN SOUS CONTRÔLE - RÈGLES PROP FIRM RESPECTÉES')
        else:
            self.log('❌ ÉCHEC DU CHALLENGE PROP FIRM')


if __name__ == '__main__':
    # Configuration pour prop firm challenge optimisé
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
    cerebro.addstrategy(OptimizedPropFirmMasterTrend)
    
    # Configuration prop firm optimisée
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)
    
    # Analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print('🚀 DÉMARRAGE DU CHALLENGE PROP FIRM OPTIMISÉ')
    print('Capital Initial: $10,000')
    print('Objectif: +8% ($800) - Plus conservateur')
    print('Limite DD: -4% ($400) - Plus strict')
    print('Limite Journalière: -1.5% ($150) - Plus strict')
    print('Max Pertes Consécutives: 3')
    
    try:
        results = cerebro.run()
        strat = results[0]
        
        # Analyse détaillée des résultats
        trades = strat.analyzers.trades.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        
        print('\n📊 ANALYSE DÉTAILLÉE OPTIMISÉE:')
        if hasattr(trades, 'total') and trades.total.closed > 0:
            print(f'Total Trades: {trades.total.closed}')
            print(f'Trades Gagnants: {trades.won.total if hasattr(trades, "won") else 0}')
            print(f'Trades Perdants: {trades.lost.total if hasattr(trades, "lost") else 0}')
            if hasattr(trades, 'won') and trades.won.total > 0:
                win_rate = trades.won.total / trades.total.closed * 100
                print(f'Taux de Réussite: {win_rate:.1f}%')
                
                if hasattr(trades.won, 'pnl') and hasattr(trades.lost, 'pnl'):
                    avg_win = trades.won.pnl.average if hasattr(trades.won.pnl, 'average') else 0
                    avg_loss = trades.lost.pnl.average if hasattr(trades.lost.pnl, 'average') else 0
                    if avg_loss != 0:
                        profit_factor = abs(avg_win / avg_loss)
                        print(f'Profit Factor: {profit_factor:.2f}')
        
        if hasattr(drawdown, 'max'):
            print(f'Drawdown Maximum: {drawdown.max.drawdown:.2f}%')
            
        if hasattr(returns, 'rtot'):
            print(f'Rendement Total: {returns.rtot*100:.2f}%')
            
    except Exception as e:
        print(f'❌ ERREUR LORS DE L\'EXÉCUTION: {e}')
        print('Vérifiez que le fichier EURUSD_data_15M.csv existe et a le bon format.') 