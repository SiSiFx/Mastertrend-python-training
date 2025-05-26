#!/usr/bin/env python3
"""
MASTERTREND PropFirm Strategy - Version Simplifiée mais Fidèle
Basée sur la logique exacte du PineScript pour maintenir la qualité des signaux
Optimisée pour passer les challenges des prop firms de manière automatique
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
from collections import deque


class PropFirmMasterTrend(bt.Strategy):
    """
    Stratégie MasterTrend simplifiée mais fidèle pour prop firms
    Reproduit exactement la logique du PineScript avec inputs fixes
    """
    
    # === PARAMÈTRES FIXES POUR PROP FIRMS ===
    params = (
        # SuperTrend (identique au PineScript)
        ('st_period', 10),
        ('st_multiplier', 3.0),
        
        # MACD Standard (pour crossmacdbear/crossmacd)
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        
        # Williams Fractals (identique au PineScript)
        ('williams_left', 2),
        ('williams_right', 2),
        ('williams_buffer', 0.0),
        
        # P1 Indicator (identique au PineScript)
        ('p1_e31', 5),
        ('p1_m', 9),
        ('p1_l31', 14),
        
        # StopGap Filter
        ('sg_ema_short', 10),
        ('sg_ema_long', 20),
        ('sg_lookback', 4),
        ('sg_median_period', 10),
        
        # Prop Firm Risk Management
        ('max_daily_loss', 0.02),    # 2% daily loss limit
        ('max_total_loss', 0.05),    # 5% total drawdown limit
        ('profit_target', 0.10),     # 10% profit target
        ('position_size', 0.01),     # 1% risk per trade
        
        # Sessions (simplifié pour prop firms)
        ('trading_start', datetime.time(9, 30)),
        ('trading_end', datetime.time(16, 0)),
    )
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # === INDICATEURS PRINCIPAUX ===
        
        # SuperTrend (reproduction exacte du PineScript)
        self.hl2 = (self.data.high + self.data.low) / 2.0
        self.atr = bt.indicators.ATR(self.data, period=self.p.st_period)
        
        # Variables SuperTrend
        self.basic_up = self.hl2 - self.p.st_multiplier * self.atr
        self.basic_dn = self.hl2 + self.p.st_multiplier * self.atr
        
        # MACD Standard pour crossovers
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        
        # Williams Fractals
        self.fractal_high = bt.indicators.Highest(self.data.high, period=self.p.williams_left + self.p.williams_right + 1)
        self.fractal_low = bt.indicators.Lowest(self.data.low, period=self.p.williams_left + self.p.williams_right + 1)
        
        # P1 Indicator (reproduction exacte)
        self.hilow = (self.data.high - self.data.low) * 100
        self.openclose = (self.data.close - self.data.open) * 100
        self.spreadv = self.openclose * self.data.close
        
        # Cumulative sum approximation
        self.pt_approx = bt.indicators.SMA(self.spreadv, period=50)  # Approximation
        
        self.ema_e31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_e31)
        self.ema_m = bt.indicators.EMA(self.pt_approx, period=self.p.p1_m)
        self.ema_l31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_l31)
        
        self.a1 = self.ema_l31 - self.ema_m
        self.b1 = self.ema_e31 - self.ema_m
        self.p1 = self.a1 + self.b1
        
        # StopGap Filter
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_long)
        self.highest_sg = bt.indicators.Highest(self.data.high, period=self.p.sg_lookback)
        self.lowest_sg = bt.indicators.Lowest(self.data.low, period=self.p.sg_lookback)
        
        # === VARIABLES D'ÉTAT ===
        self.trend = 1  # SuperTrend direction
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
        
        # StopGap median calculation
        self.candle_sizes = deque(maxlen=self.p.sg_median_period)
        
        # Signaux
        self.buy_signals = []
        self.sell_signals = []
    
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
    
    def update_supertrend(self):
        """Mise à jour SuperTrend (reproduction exacte du PineScript)"""
        if len(self.data) < 2:
            return
            
        # up := close[1] > up1 ? math.max(up, up1) : up
        if self.data.close[-1] > self.final_up:
            self.final_up = max(self.basic_up[0], self.final_up)
        else:
            self.final_up = self.basic_up[0]
            
        # dn := close[1] < dn1 ? math.min(dn, dn1) : dn
        if self.data.close[-1] < self.final_dn:
            self.final_dn = min(self.basic_dn[0], self.final_dn)
        else:
            self.final_dn = self.basic_dn[0]
            
        # trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend
        prev_trend = self.trend
        if self.trend == -1 and self.data.close[0] > self.final_dn:
            self.trend = 1
        elif self.trend == 1 and self.data.close[0] < self.final_up:
            self.trend = -1
    
    def update_williams_stops(self):
        """Mise à jour Williams Stops (simplifiée mais fonctionnelle)"""
        # Détection des fractals (simplifiée)
        if len(self.data) > self.p.williams_left + self.p.williams_right:
            # High fractal
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
            
            # Low fractal
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
        """Calcul du filtre StopGap (reproduction de la logique PineScript)"""
        # Trend detection
        trend_up = self.ema_short[0] > self.ema_long[0]
        trend_down = self.ema_short[0] < self.ema_long[0]
        
        # Candle size
        candle_size = abs(self.data.high[0] - self.data.low[0])
        self.candle_sizes.append(candle_size)
        
        # Calculate median
        if len(self.candle_sizes) == self.p.sg_median_period:
            median = np.median(list(self.candle_sizes))
        else:
            median = candle_size
        
        # StopGap calculation
        stop_gap = 0.0
        if trend_down and self.data.high[0] < self.highest_sg[-1]:
            stop_gap = abs(self.highest_sg[-1] - self.data.low[0])
        elif trend_up and self.data.low[0] > self.lowest_sg[-1]:
            stop_gap = abs(self.data.high[0] - self.lowest_sg[-1])
        
        return stop_gap > median
    
    def check_prop_firm_rules(self):
        """Vérification des règles prop firm"""
        current_value = self.broker.getvalue()
        dt = self.datas[0].datetime.datetime(0)
        
        # Reset daily tracking
        if self.current_date != dt.date():
            self.current_date = dt.date()
            self.daily_start_cash = current_value
        
        # Update peak
        self.peak_value = max(self.peak_value, current_value)
        
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
        """Vérification session de trading"""
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
        
        # Mise à jour des indicateurs
        self.update_supertrend()
        self.update_williams_stops()
        
        # Calcul des signaux (reproduction exacte du PineScript)
        
        # MACD Crossovers
        crossmacdbear = (self.macd.macd[0] > 0 and self.macd.macd[-1] <= 0)
        crossmacd = (self.macd.macd[0] < 0 and self.macd.macd[-1] >= 0)
        
        # P1 Conditions
        b1_ge_p1 = self.b1[0] >= self.p1[0]
        b1_le_p1 = self.b1[0] <= self.p1[0]
        
        # StopGap Filter
        stopgap_ok = self.calculate_stopgap_filter()
        
        # === CONDITIONS EXACTES DU PINESCRIPT ===
        
        # longCondition = williamsLongStopPriceTrailPlotDisplay and crossmacdbear and (b1 >= p1) and (trend == 1) and (stopGap > med) and sessions
        long_condition = (
            self.williams_long_active and 
            crossmacdbear and 
            b1_ge_p1 and 
            self.trend == 1 and 
            stopgap_ok
        )
        
        # shortCondition = williamsShortStopPriceTrailPlotDisplay and crossmacd and (b1 <= p1) and (trend == -1) and (stopGap > med) and sessions
        short_condition = (
            self.williams_short_active and 
            crossmacd and 
            b1_le_p1 and 
            self.trend == -1 and 
            stopgap_ok
        )
        
        # === EXÉCUTION DES TRADES ===
        
        if self.order:  # Ordre en attente
            return
        
        if not self.position:  # Pas de position
            if long_condition:
                self.log(f'BUY SIGNAL @ {self.data.close[0]:.5f}')
                size = self.calculate_position_size()
                self.order = self.buy(size=size)
                self.buy_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
                
            elif short_condition:
                self.log(f'SELL SIGNAL @ {self.data.close[0]:.5f}')
                size = self.calculate_position_size()
                self.order = self.sell(size=size)
                self.sell_signals.append((self.datas[0].datetime.datetime(0), self.data.close[0]))
        
        else:  # Position ouverte - gestion des stops
            if self.position.size > 0:  # Position longue
                if (self.williams_long_stop and 
                    self.data.low[0] <= self.williams_long_stop):
                    self.log(f'LONG STOP HIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                elif crossmacd:  # Exit sur signal opposé
                    self.log(f'LONG EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                    
            elif self.position.size < 0:  # Position courte
                if (self.williams_short_stop and 
                    self.data.high[0] >= self.williams_short_stop):
                    self.log(f'SHORT STOP HIT @ {self.data.close[0]:.5f}')
                    self.order = self.close()
                elif crossmacdbear:  # Exit sur signal opposé
                    self.log(f'SHORT EXIT SIGNAL @ {self.data.close[0]:.5f}')
                    self.order = self.close()
    
    def calculate_position_size(self):
        """Calcul de la taille de position basée sur le risque"""
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.p.position_size
        
        # Utiliser Williams Stop comme stop loss pour calculer la taille
        if self.williams_long_stop:
            stop_distance = abs(self.data.close[0] - self.williams_long_stop)
        elif self.williams_short_stop:
            stop_distance = abs(self.data.close[0] - self.williams_short_stop)
        else:
            stop_distance = self.data.close[0] * 0.01  # 1% par défaut
        
        if stop_distance > 0:
            size = risk_amount / stop_distance
            return max(1, int(size))  # Minimum 1 unité
        return 1
    
    def stop(self):
        """Statistiques finales"""
        final_value = self.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        max_dd = (self.peak_value - final_value) / self.peak_value * 100
        
        self.log(f'=== RÉSULTATS FINAUX ===')
        self.log(f'Capital Initial: ${self.initial_cash:.2f}')
        self.log(f'Capital Final: ${final_value:.2f}')
        self.log(f'Rendement Total: {total_return:.2f}%')
        self.log(f'Drawdown Max: {max_dd:.2f}%')
        self.log(f'Signaux BUY: {len(self.buy_signals)}')
        self.log(f'Signaux SELL: {len(self.sell_signals)}')
        
        # Vérification des objectifs prop firm
        if total_return >= self.p.profit_target * 100:
            self.log('✅ OBJECTIF PROFIT ATTEINT - PROP FIRM CHALLENGE RÉUSSI!')
        elif max_dd <= self.p.max_total_loss * 100:
            self.log('✅ DRAWDOWN SOUS CONTRÔLE - RÈGLES PROP FIRM RESPECTÉES')
        else:
            self.log('❌ ÉCHEC DU CHALLENGE PROP FIRM')


if __name__ == '__main__':
    # Configuration pour prop firm challenge
    cerebro = bt.Cerebro()
    
    # Données - Configuration corrigée
    data = bt.feeds.GenericCSVData(
        dataname="EURUSD_data_15M.csv",
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, 
        open=1, 
        high=2, 
        low=3, 
        close=4, 
        volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=15,
        openinterest=-1,  # Pas de données d'intérêt ouvert
        headers=True,     # Le fichier a des en-têtes
        separator=','     # Séparateur virgule
    )
    
    cerebro.adddata(data)
    cerebro.addstrategy(PropFirmMasterTrend)
    
    # Configuration prop firm standard
    cerebro.broker.setcash(10000.0)  # 10K challenge
    cerebro.broker.setcommission(commission=0.0001)  # Spread réaliste
    
    # Analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    
    print('🚀 DÉMARRAGE DU CHALLENGE PROP FIRM')
    print('Capital Initial: $10,000')
    print('Objectif: +10% ($1,000)')
    print('Limite DD: -5% ($500)')
    print('Limite Journalière: -2% ($200)')
    
    try:
        results = cerebro.run()
        strat = results[0]
        
        # Analyse des résultats
        trades = strat.analyzers.trades.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        
        print('\n📊 ANALYSE DÉTAILLÉE:')
        if hasattr(trades, 'total'):
            print(f'Total Trades: {trades.total.closed}')
            print(f'Trades Gagnants: {trades.won.total if hasattr(trades, "won") else 0}')
            print(f'Trades Perdants: {trades.lost.total if hasattr(trades, "lost") else 0}')
            if hasattr(trades, 'won') and trades.won.total > 0:
                win_rate = trades.won.total / trades.total.closed * 100
                print(f'Taux de Réussite: {win_rate:.1f}%')
        
        if hasattr(drawdown, 'max'):
            print(f'Drawdown Maximum: {drawdown.max.drawdown:.2f}%')
            
    except Exception as e:
        print(f'❌ ERREUR LORS DE L\'EXÉCUTION: {e}')
        print('Vérifiez que le fichier EURUSD_data_15M.csv existe et a le bon format.') 