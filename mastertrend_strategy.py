#!/usr/bin/env python3
"""
MASTERTREND Strategy (convertie de PineScript vers Python/Backtrader)
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
import pytz
import collections
import sys # Add import sys
import types  # for dummy broker namespace


# ===== INDICATEURS PERSONNALISÉS =====

# Indicateur JMA (Jurik Moving Average)
class JMA(bt.Indicator):
    """Jurik Moving Average"""
    
    lines = ('jma',)
    params = (
        ('period', 7),
        ('phase', 50),
        ('power', 2),
    )
    
    def __init__(self):
        self.phaseRatio = 0.5 if self.p.phase < -100 else 2.5 if self.p.phase > 100 else self.p.phase / 100 + 1.5
        self.beta = 0.45 * (self.p.period - 1) / (0.45 * (self.p.period - 1) + 2)
        self.alpha = pow(self.beta, self.p.power)
        
        self.e0 = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        self.jma = 0.0
    
    def next(self):
        # e0 := (1 - alpha) * source + alpha * nz(e0[1])
        self.e0 = (1 - self.alpha) * self.data[0] + self.alpha * self.e0
        
        # e1 := (source - e0) * (1 - beta) + beta * nz(e1[1])
        self.e1 = (self.data[0] - self.e0) * (1 - self.beta) + self.beta * self.e1
        
        # e2 := (e0 + phaseRatio * e1 - nz(jma[1])) * math.pow(1 - alpha, 2) + math.pow(alpha, 2) * nz(e2[1])
        self.e2 = (self.e0 + self.phaseRatio * self.e1 - self.jma) * pow(1 - self.alpha, 2) + pow(self.alpha, 2) * self.e2
        
        # jma := e2 + nz(jma[1])
        self.jma = self.e2 + self.jma
        self.lines.jma[0] = self.jma


# JMACD (MACD basé sur JMA au lieu de EMA)
class JMACD(bt.Indicator):
    """MACD basé sur Jurik Moving Average"""
    
    lines = ('macd', 'signal', 'histo')
    params = (
        ('fastlen', 12),
        ('slowlen', 26),
        ('siglen', 9),
    )
    
    def __init__(self):
        self.dif = JMA(self.data, period=self.p.fastlen, phase=0, power=1.0)
        self.dea = JMA(self.data, period=self.p.slowlen, phase=0, power=1.0)
        self.macd = self.dif - self.dea
        
        # signal = ta.alma(macd, siglen, 0.85, 6)
        self.signal = bt.indicators.ALMA(self.macd, period=self.p.siglen, offset=0.85, sigma=6)
        
        # histo = (macd - signal) * 2
        self.histo = (self.macd - self.signal) * 2
        
        # Assigner aux lignes de l'indicateur
        self.lines.macd = self.macd
        self.lines.signal = self.signal
        self.lines.histo = self.histo


# Indicateur de pivots williamsFractal
class WilliamsFractal(bt.Indicator):
    """Indicator: William's Fractals"""
    
    lines = ('is_high_fractal', 'high_fractal_price', 'is_low_fractal', 'low_fractal_price')
    params = (
        ('left_bars', 2),
        ('right_bars', 2),
    )
    
    plotinfo = dict(subplot=False)

    def __init__(self):
        self.addminperiod(self.p.left_bars + self.p.right_bars + 1)
        self.plotlines.is_high_fractal = False
        self.plotlines.high_fractal_price = False
        self.plotlines.is_low_fractal = False
        self.plotlines.low_fractal_price = False

    def next(self):
        # Default to no fractal this bar
        self.lines.is_high_fractal[0] = 0
        self.lines.high_fractal_price[0] = float('nan')
        self.lines.is_low_fractal[0] = 0
        self.lines.low_fractal_price[0] = float('nan')

        # Check for high fractal confirmation
        # A high fractal is at high[-p.right_bars] if it's the highest over the window
        # The fractal is confirmed on the current bar [0]
        is_high = True
        # Potential fractal high value is at index -self.p.right_bars
        potential_high_val = self.data.high[-self.p.right_bars]
        
        # Check left side (excluding the pivot bar itself)
        for i in range(self.p.right_bars + 1, self.p.right_bars + self.p.left_bars + 1):
            if self.data.high[-i] > potential_high_val: # Use > for highs
                is_high = False
                break
        if is_high:
            # Check right side (excluding the pivot bar itself)
            for i in range(0, self.p.right_bars): # up to, but not including, -self.p.right_bars
                if self.data.high[-i] >= potential_high_val: # Use >= for highs, if current is higher, it's not it.
                    is_high = False
                    break
        
        # Check for low fractal confirmation
        is_low = True
        potential_low_val = self.data.low[-self.p.right_bars]

        # Check left side
        for i in range(self.p.right_bars + 1, self.p.right_bars + self.p.left_bars + 1):
            if self.data.low[-i] < potential_low_val: # Use < for lows
                is_low = False
                break
        if is_low:
            # Check right side
            for i in range(0, self.p.right_bars):
                if self.data.low[-i] <= potential_low_val: # Use <= for lows
                    is_low = False
                    break

        if is_high:
            # Suppress if previous bar was also a high fractal (Pine logic: isWilliamsHigh := isWilliamsHigh[1] ? false : isWilliamsHigh)
            # This check is a bit tricky with how bt.Indicator state works. 
            # For simplicity, we'll rely on the fact that a new true high will naturally make the previous one not the highest in the window.
            # A more direct port of the suppression might need self.lines.is_high_fractal[-1].
            if self.lines.is_high_fractal[-1] == 0: # only set if previous wasn't a fractal
                 self.lines.is_high_fractal[0] = 1
                 self.lines.high_fractal_price[0] = potential_high_val
        
        if is_low:
            if self.lines.is_low_fractal[-1] == 0: # only set if previous wasn't a fractal
                self.lines.is_low_fractal[0] = 1
                self.lines.low_fractal_price[0] = potential_low_val


# Détecteur de session
class SessionIndicator(bt.Indicator):
    """Détecte si on est dans une session de trading spécifique"""
    
    lines = ('in_session',)
    params = (
        ('session', '0300-1200'),  # Format: "HHMM-HHMM"
        ('days', '1234567'),       # Format: "1234567" (lundi à dimanche)
        ('tz', 'America/New_York'),
    )
    
    def __init__(self):
        # Extraire heures et minutes de début/fin
        session_start, session_end = self.p.session.split('-')
        self.start_hour = int(session_start[:2])
        self.start_minute = int(session_start[2:]) if len(session_start) > 2 else 0
        self.end_hour = int(session_end[:2])
        self.end_minute = int(session_end[2:]) if len(session_end) > 2 else 0
        
        # Convertir la chaîne de jours en liste d'entiers (0=lundi, 6=dimanche)
        self.session_days = [int(d) % 7 for d in self.p.days]
        
        # Timezone
        self.timezone = pytz.timezone(self.p.tz)
    
    def next(self):
        # Obtenir la date et l'heure actuelle
        current_dt = self.data.datetime.datetime(0)
        if not current_dt.tzinfo:
            # Si la date n'a pas de timezone, on la considère comme étant dans la timezone spécifiée
            current_dt = self.timezone.localize(current_dt)
        
        # Vérifier si on est dans un jour valide (0=lundi, 6=dimanche)
        day_of_week = current_dt.weekday()
        if day_of_week not in self.session_days:
            self.lines.in_session[0] = 0
            return
        
        # Vérifier si on est dans l'intervalle horaire
        current_time = current_dt.time()
        session_start = datetime.time(self.start_hour, self.start_minute)
        session_end = datetime.time(self.end_hour, self.end_minute)
        
        if session_start <= session_end:
            # Session normale (ex: 09:00-17:00)
            in_session = session_start <= current_time <= session_end
        else:
            # Session qui traverse minuit (ex: 22:00-04:00)
            in_session = current_time >= session_start or current_time <= session_end
        
        self.lines.in_session[0] = 1 if in_session else 0


# Indicateur SuperTrend
class SuperTrendIndicator(bt.Indicator):
    """SuperTrend Indicator"""
    lines = ('supertrend', 'trend_direction', 'final_up', 'final_down')
    params = (
        ('period', 10),
        ('multiplier', 3),
    )

    def __init__(self):
        self.src = (self.data.high + self.data.low) / 2.0
        self.atr = bt.indicators.ATR(self.data, period=self.p.period)
        self.addminperiod(self.p.period) # Wait for ATR to have a valid value

        # To suppress plotting of intermediate lines by default
        self.plotlines.final_up = False
        self.plotlines.final_down = False
        # self.plotlines.trend_direction = False # Might be useful for debugging

    def nextstart(self):
        # Initial calculation for the first valid bar (after ATR period)
        current_basic_up = self.src[0] - self.p.multiplier * self.atr[0]
        current_basic_down = self.src[0] + self.p.multiplier * self.atr[0]

        # PineScript initializes trend = 1.
        # For the very first calculation, up1 = up, dn1 = dn, trend[1] = trend (1)
        # This means the first supertrend value is based on the initial trend assumption.
        self.lines.trend_direction[0] = 1
        self.lines.final_up[0] = current_basic_up
        self.lines.final_down[0] = current_basic_down # Calculated but not used if trend starts up
        self.lines.supertrend[0] = self.lines.final_up[0] # If trend is 1, ST is final_up

    def next(self):
        # Previous values from the indicator's own lines
        prev_trend = self.lines.trend_direction[-1]
        prev_final_up = self.lines.final_up[-1]
        prev_final_down = self.lines.final_down[-1]
        
        # prev_bar_close is Pine's close[1] in the context of current ST calculation
        prev_bar_close = self.data.close[-1]

        # Basic up/down for the current bar [0]
        current_basic_up = self.src[0] - self.p.multiplier * self.atr[0]
        current_basic_down = self.src[0] + self.p.multiplier * self.atr[0]

        # Pine: up := close[1] > up1 ? math.max(up, up1) : up
        # up1 is prev_final_up, up (without [1]) is current_basic_up
        if prev_bar_close > prev_final_up:
            self.lines.final_up[0] = max(current_basic_up, prev_final_up)
        else:
            self.lines.final_up[0] = current_basic_up

        # Pine: dn := close[1] < dn1 ? math.min(dn, dn1) : dn
        # dn1 is prev_final_down, dn (without [1]) is current_basic_down
        if prev_bar_close < prev_final_down:
            self.lines.final_down[0] = min(current_basic_down, prev_final_down)
        else:
            self.lines.final_down[0] = current_basic_down
        
        # Pine: trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend
        # 'close' is current bar's close (self.data.close[0])
        # 'dn1' is prev_final_down, 'up1' is prev_final_up
        current_trend = prev_trend
        if prev_trend == -1 and self.data.close[0] > prev_final_down:
            current_trend = 1
        elif prev_trend == 1 and self.data.close[0] < prev_final_up:
            current_trend = -1
        self.lines.trend_direction[0] = current_trend

        # Set the SuperTrend line value for the current bar
        if current_trend == 1:
            self.lines.supertrend[0] = self.lines.final_up[0]
        else:
            self.lines.supertrend[0] = self.lines.final_down[0]


# Helper Indicator for Cumulative Sum
class CumulativeSumIndicator(bt.Indicator):
    lines = ('cumsum',)
    # No params needed if source is passed directly at instantiation
    # params = (('source_line', None),) # Not using params directly for source line
    plotinfo = dict(subplot=False) # Usually not plotted directly

    def __init__(self, source_line):
        # The source_line is passed directly, not via params.p
        # This is a common pattern for helper indicators that are not meant to be generic params-driven.
        self.source_line = source_line # Keep a reference
        self.running_sum = 0.0
        # Ensure lines are defined before use by other indicators if this itself is nested.
        # Here, cumsum is the output line of this indicator.

    def next(self):
        val = self.source_line[0]
        if math.isnan(val):
            val = 0.0 # Or handle as per strategy needs
        self.running_sum += val
        self.lines.cumsum[0] = self.running_sum


# Indicateur P1 (complex volume/price analysis)
class P1Indicator(bt.Indicator):
    lines = ('p1_value', 'b1_value', 'bullish_rule', 'bearish_rule', 'pt_line')
    params = (
        ('e31', 5),
        ('m', 9),
        ('l31', 14),
    )
    plotinfo = dict(subplot=True)

    def __init__(self):
        # Spread calculation
        self.spreadv_line = (self.datas[0].close - self.datas[0].open) * 100.0 * self.datas[0].close
        
        # Cumulative sum of spreadv_line
        cum = CumulativeSumIndicator(source_line=self.spreadv_line)
        # Base pt_line
        self.l.pt_line = self.spreadv_line + cum.l.cumsum

        # EMAs on pt_line using own params
        ema_l31 = bt.indicators.EMA(self.l.pt_line, period=self.p.l31)
        ema_m = bt.indicators.EMA(self.l.pt_line, period=self.p.m)
        ema_e31 = bt.indicators.EMA(self.l.pt_line, period=self.p.e31)

        # Compute values
        a1 = ema_l31 - ema_m
        self.l.b1_value = ema_e31 - ema_m
        self.l.p1_value = a1 + self.l.b1_value
        self.l.bullish_rule = self.l.b1_value >= self.l.p1_value
        self.l.bearish_rule = self.l.b1_value <= self.l.p1_value

        # Minimum bars for the longest EMA
        self.addminperiod(self.p.l31)


# Indicateur Stop Gap Median Filter
class StopGapMedFilterIndicator(bt.Indicator):
    lines = ('filter_passed', 'actual_median')
    params = (
        ('ema_short_period', 10),
        ('ema_long_period', 20),
        ('highest_lowest_period', 4),
        ('median_lookback', 10),
    )
    plotinfo = dict(subplot=True)

    def __init__(self):
        # EMAs for trend detection
        self.ema_short = bt.indicators.EMA(self.datas[0].close, period=self.p.ema_short_period)
        self.ema_long = bt.indicators.EMA(self.datas[0].close, period=self.p.ema_long_period)

        self.trend_up = self.ema_short > self.ema_long
        self.trend_down = self.ema_short < self.ema_long

        # Candle size for stop-gap calculation
        self.candle_size = abs(self.datas[0].high - self.datas[0].low)
        
        # Highest/Lowest over lookback
        self.highest_val = bt.indicators.Highest(self.datas[0].high, period=self.p.highest_lowest_period)
        self.lowest_val = bt.indicators.Lowest(self.datas[0].low, period=self.p.highest_lowest_period)
        
        # Rolling window for median
        self.median_window = collections.deque(maxlen=self.p.median_lookback)
        # self.plotlines.actual_median = True  # disabled faulty plot assignment

    def next(self):
        current_trend_up = self.trend_up[0] > 0
        current_trend_down = self.trend_down[0] > 0
        current_high = self.datas[0].high[0]
        current_low = self.datas[0].low[0]
        prev_highest_val = self.highest_val[-1]
        prev_lowest_val = self.lowest_val[-1]

        stop_gap = 0.0
        if current_trend_down and current_high < prev_highest_val:
            stop_gap = abs(prev_highest_val - current_low)
        elif current_trend_up and current_low > prev_lowest_val:
            stop_gap = abs(current_high - prev_lowest_val)
        
        current_candle_size = self.candle_size[0]
        if not math.isnan(current_candle_size):
            self.median_window.append(current_candle_size)

        calculated_median_val = float('nan')
        if len(self.median_window) == self.p.median_lookback:
            sorted_window = sorted(list(self.median_window))
            mid_idx = len(sorted_window) // 2
            if len(sorted_window) % 2 == 0:
                calculated_median_val = (sorted_window[mid_idx - 1] + sorted_window[mid_idx]) / 2.0
            else:
                calculated_median_val = sorted_window[mid_idx]
        
        self.lines.actual_median[0] = calculated_median_val
        current_median_value = self.lines.actual_median[0]
        
        if not math.isnan(stop_gap) and not math.isnan(current_median_value):
            self.lines.filter_passed[0] = stop_gap > current_median_value
        else:
            self.lines.filter_passed[0] = False


# Stratégie MASTERTREND
class MasterTrendStrategy(bt.Strategy):
    params = (
        # SuperTrend defaults
        ('supertrend_period', 10),
        ('supertrend_multiplier', 3),
        
        # PineScript input() defaults for MACD
        ('macd_fast', 13),
        ('macd_slow', 26),
        ('macd_signal', 9),

        # PineScript defaults for P1Indicator
        ('p1_e31', 5),
        ('p1_m', 9),
        ('p1_l31', 14),

        # PineScript defaults for StopGapMedFilterIndicator
        ('sg_ema_short_period', 10),
        ('sg_ema_long_period', 20),
        ('sg_highest_lowest_period', 4),
        ('sg_median_lookback', 10),

        # Test EMA default (for quick EMA sanity check)
        ('test_ema_period', 14),

        # Williams Fractal defaults (align with Pine's inputWilliams... for trailing stop)
        ('left_range', 2),    # Corresponds to inputWilliamsLeftRange
        ('right_range', 2),   # Corresponds to inputWilliamsRightRange
        ('williams_stop_buffer', 0.0),  # Corresponds to inputWilliamsStopBufferInput
        
        # Sessions (unchanged)
        ('london_session', True), ('london_hours', '0300-1200'),
        ('ny_session', True), ('ny_hours', '0800-1700'),
        ('tokyo_session', True), ('tokyo_hours', '2000-0400'),
        ('sydney_session', False), ('sydney_hours', '1700-0200'),
    )
    
    def log(self, txt, dt=None):
        """Logging fonction"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} {txt}')
    
    def __init__(self):
        # Dummy broker for standalone RL use
        self.cash = 0.0
        self.broker = types.SimpleNamespace(
            getvalue=lambda: self.cash,
            setcash=lambda c: setattr(self, 'cash', c)
        )
        # Initialisation des principales variables et indicateurs
        
        # --- MACD --- (Standard MACD for crossmacdbear/crossmacd signals from Pine)
        self.standard_macd_val = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        ).macd
        
        # --- SuperTrend ---
        self.supertrend = SuperTrendIndicator(
            self.data,
            period=self.p.supertrend_period,
            multiplier=self.p.supertrend_multiplier
        )
        
        # --- Williams Fractals ---
        self.fractals = WilliamsFractal(self.data, left_bars=self.p.left_range, right_bars=self.p.right_range)
        
        # --- Indicateurs de session ---
        self.london_ind = SessionIndicator(self.data, session=self.p.london_hours, days='1234567')
        self.ny_ind = SessionIndicator(self.data, session=self.p.ny_hours, days='1234567')
        self.tokyo_ind = SessionIndicator(self.data, session=self.p.tokyo_hours, days='1234567')
        self.sydney_ind = SessionIndicator(self.data, session=self.p.sydney_hours, days='1234567')
                
        # --- P1 Indicator ---
        self.p1_indicator = P1Indicator()
        
        # --- StopGapMedFilter Indicator ---
        self.stop_gap_filter = StopGapMedFilterIndicator()

        # --- TEST EMA --- (To check if basic EMA on data works)
        self.test_ema = bt.indicators.EMA(
            self.data.close,
            period=self.p.test_ema_period,
            plotname=f'TestEMA{self.p.test_ema_period}'
        )
        
        # --- Autres variables ---
        self.order = None
        self.buyprice = None
        self.sellprice = None
        
        # Williams Trailing Stop Variables - initialized here, managed in next()
        self.williams_low_price_buffered = float('nan')
        self.williams_high_price_buffered = float('nan')
        
        self.persisted_williams_long_stop = float('nan') # Corresponds to williamsLongStopPrice in Pine
        self.persisted_williams_short_stop = float('nan')# Corresponds to williamsShortStopPrice in Pine

        self.williams_long_stop_trail = float('nan')    # williamsLongStopPriceTrail
        self.williams_short_stop_trail = float('nan')   # williamsShortStopPriceTrail

        # Flip logic state
        self.flip_is_long = True  # Start true as per Pine: var bool _isLong = true
        self.flip_is_short = True # Start true as per Pine: var bool _isShort = true
        
        # Plottable lines (will be na if not active)
        self.williams_long_stop_trail_plot = float('nan')
        self.williams_short_stop_trail_plot = float('nan')

        # For f_getFlipResetWilliams logic
        self.last_confirmed_low_fractal_bar_index = -1
        self.last_confirmed_high_fractal_bar_index = -1

        # --- Lines for plotting Williams Stops from Strategy ---
        self.plot_williams_long_stop = bt.LineSeries(plotname='Williams Long Stop (Strat)')
        self.plot_williams_long_stop.plotinfo.color = 'blue'
        self.plot_williams_long_stop.plotinfo.linestyle = '--'
        
        self.plot_williams_short_stop = bt.LineSeries(plotname='Williams Short Stop (Strat)')
        self.plot_williams_short_stop.plotinfo.color = 'magenta'
        self.plot_williams_short_stop.plotinfo.linestyle = '--'

        # Compteurs pour statistiques
        self.sum_long_bars = 0.0
        self.sum_short_bars = 0.0
        self.sum_long_distance = 0.0
        self.sum_short_distance = 0.0
        self.count_long = 0
        self.count_short = 0
        self.current_count = 0
        self.start_price = 0.0
        self.in_long = False
        self.in_short = False
        # Lists to store buy and sell signals as (timestamp, price)
        self.buy_signals = []
        self.sell_signals = []
        # Unified trade log: list of (timestamp, side, price)
        self.trade_log = []
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return  # Attendre
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.5f}')
                self.buyprice = order.executed.price
            else:
                self.log(f'SELL EXECUTED, {order.executed.price:.5f}')
                self.sellprice = order.executed.price
                
            self.bar_executed = len(self)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        
        self.log(f'TRADE P/L: GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
    
    def update_williams_stops(self):
        # This function will now implement the full trailing logic.
        # It's called from next() before trading decisions.
        current_bar_index = len(self.data) -1 # Current bar index (0-based)

        # 1. Get current fractal signals (isWilliamsHigh, williamsHighPrice, etc. in Pine)
        is_new_low_fractal = self.fractals.is_low_fractal[0] == 1
        new_low_fractal_price = self.fractals.low_fractal_price[0]
        is_new_high_fractal = self.fractals.is_high_fractal[0] == 1
        new_high_fractal_price = self.fractals.high_fractal_price[0]

        if is_new_low_fractal:
            self.last_confirmed_low_fractal_bar_index = current_bar_index
        if is_new_high_fractal:
            self.last_confirmed_high_fractal_bar_index = current_bar_index

        # 2. Add buffer (f_addPercentBuffer)
        current_williams_low_price_buffered = float('nan')
        if not math.isnan(new_low_fractal_price):
            current_williams_low_price_buffered = new_low_fractal_price * (1 - self.p.williams_stop_buffer / 100)

        current_williams_high_price_buffered = float('nan')
        if not math.isnan(new_high_fractal_price):
            current_williams_high_price_buffered = new_high_fractal_price * (1 + self.p.williams_stop_buffer / 100)
            
        # 3. Persist and Reset (f_persistAndReset for williamsLongStopPrice, williamsShortStopPrice)
        # persisted_williams_long_stop
        if is_new_low_fractal and not math.isnan(current_williams_low_price_buffered):
            self.persisted_williams_long_stop = current_williams_low_price_buffered
        elif math.isnan(self.persisted_williams_long_stop) and not math.isnan(current_williams_low_price_buffered) : # Initial case if first fractal is low
             self.persisted_williams_long_stop = current_williams_low_price_buffered
        # else: it persists from self.persisted_williams_long_stop (from previous bar, handled by assignment from trail later if needed)

        # persisted_williams_short_stop
        if is_new_high_fractal and not math.isnan(current_williams_high_price_buffered):
            self.persisted_williams_short_stop = current_williams_high_price_buffered
        elif math.isnan(self.persisted_williams_short_stop) and not math.isnan(current_williams_high_price_buffered): # Initial case
            self.persisted_williams_short_stop = current_williams_high_price_buffered
        # else: it persists

        # Initialize trails if they are nan (first run or after reset)
        if math.isnan(self.williams_long_stop_trail) and not math.isnan(self.persisted_williams_long_stop):
            self.williams_long_stop_trail = self.persisted_williams_long_stop
        if math.isnan(self.williams_short_stop_trail) and not math.isnan(self.persisted_williams_short_stop):
            self.williams_short_stop_trail = self.persisted_williams_short_stop
            
        # 4. Trail (f_trail)
        # Trail the high (short) stop down
        if not math.isnan(self.persisted_williams_short_stop):
            if math.isnan(self.williams_short_stop_trail) or self.persisted_williams_short_stop <= self.williams_short_stop_trail : # or self.persisted_williams_short_stop >= self.williams_short_stop_trail[-1] (Pine)
                self.williams_short_stop_trail = self.persisted_williams_short_stop
            # else: self.williams_short_stop_trail remains self.williams_short_stop_trail (trails down)

        # Trail the low (long) stop up
        if not math.isnan(self.persisted_williams_long_stop):
            if math.isnan(self.williams_long_stop_trail) or self.persisted_williams_long_stop >= self.williams_long_stop_trail: # or self.persisted_williams_long_stop <= self.williams_long_stop_trail[-1] (Pine)
                self.williams_long_stop_trail = self.persisted_williams_long_stop
            # else: self.williams_long_stop_trail remains self.williams_long_stop_trail (trails up)

        # 5. Flip Logic (f_flip)
        # These are previous states for flip logic
        prev_is_long = self.flip_is_long
        prev_is_short = self.flip_is_short

        flip_long_now = False
        flip_short_now = False

        # Using close for flip trigger as per Pine: f_flip("Close", ...)
        flip_long_source = self.data.close[0]
        flip_short_source = self.data.close[0] # Pine uses low for short flip source with "Wick"

        if prev_is_short and not math.isnan(self.williams_short_stop_trail) and flip_long_source > self.williams_short_stop_trail:
            flip_long_now = True
        
        if prev_is_long and not math.isnan(self.williams_long_stop_trail) and flip_short_source < self.williams_long_stop_trail:
            flip_short_now = True

        # Edge case for simultaneous flips (Pine logic)
        if flip_short_now and flip_long_now:
            if self.data.close[0] > self.williams_long_stop_trail : # check against long trail first
                flip_short_now = False # Prioritize long
            elif self.data.close[0] < self.williams_short_stop_trail: # then short trail
                flip_long_now = False # Prioritize short
            # If close is between, both might remain true, Pine handles this by subsequent _isLong/_isShort logic

        # Update _isLong and _isShort states
        if flip_long_now:
            self.flip_is_long = True
            self.flip_is_short = False
        elif flip_short_now:
            self.flip_is_long = False
            self.flip_is_short = True
        # else: they persist their previous state (self.flip_is_long = self.flip_is_long etc)
        
        # --- f_getFlipResetWilliamsLong / f_getFlipResetWilliamsShort Logic --- 
        # This is the more complex reset logic from Pine.
        # If we flipped to long (is_long and not prev_is_long)
        reset_long_to = self.persisted_williams_long_stop # Default to current persisted stop
        if self.flip_is_long and not prev_is_long:
            if self.last_confirmed_low_fractal_bar_index != -1: # If a low fractal has ever been confirmed
                # Pine: _barIndexWhenLastFractalConfirmed = ta.valuewhen(_isWilliamsLow, bar_index, 0)
                # Pine: _barsSinceLastFractalConfirmed = bar_index - _barIndexWhenLastFractalConfirmed
                # Pine: int _barsToGoBack = _barsSinceLastFractalConfirmed + _williamsRightRange
                # _williamsRightRange is self.p.right_range for the fractal indicator
                bars_since_last_low_fractal_confirmed = current_bar_index - self.last_confirmed_low_fractal_bar_index
                bars_to_go_back_for_low = bars_since_last_low_fractal_confirmed + self.p.right_range # self.p.right_range from strategy params
                
                lowest_low_in_lookback = self.data.low[0] # Start with current low
                # Loop from 1 bar ago up to bars_to_go_back_for_low (inclusive of current bar effectively if bars_to_go_back is 0)
                # Max lookback should not exceed available data. Clamped to current_bar_index.
                # The loop in Pine `for i = 0 to _barsToGoBack by 1` refers to `low[i]` where i is offset from current.
                # So, self.data.low[-i] for i from 0 to bars_to_go_back_for_low
                actual_loop_limit = min(bars_to_go_back_for_low, current_bar_index) 

                for i in range(actual_loop_limit + 1): # +1 because range is exclusive at end, Pine is inclusive
                    if -i < 0: # Ensure index is valid for self.data.low (i.e., not too far back)
                         # Check if data available at self.data.low[-i]
                        if len(self.data.low) > i:
                             lowest_low_in_lookback = min(lowest_low_in_lookback, self.data.low[-i])
                
                if not math.isnan(lowest_low_in_lookback):
                    reset_long_to = lowest_low_in_lookback * (1 - self.p.williams_stop_buffer / 100)
            # If no low fractal ever confirmed, reset_long_to remains self.persisted_williams_long_stop (which might be NaN initially)

        # If we flipped to short (is_short and not prev_is_short)
        reset_short_to = self.persisted_williams_short_stop # Default
        if self.flip_is_short and not prev_is_short:
            if self.last_confirmed_high_fractal_bar_index != -1:
                bars_since_last_high_fractal_confirmed = current_bar_index - self.last_confirmed_high_fractal_bar_index
                # Pine uses _williamsLeftRange for short, but it's symmetrical in this WilliamsFractal impl. Using right_range for consistency with above.
                # The original Pine has: f_getFlipResetWilliamsShort(_shortTrail, _buffer, _isWilliamsHigh, _williamsLeftRange)
                # Our self.p.left_range / self.p.right_range are params for WilliamsFractal indicator itself. Strategy passes these.
                bars_to_go_back_for_high = bars_since_last_high_fractal_confirmed + self.p.left_range 
                
                highest_high_in_lookback = self.data.high[0]
                actual_loop_limit_high = min(bars_to_go_back_for_high, current_bar_index)

                for i in range(actual_loop_limit_high + 1):
                    if -i < 0:
                        if len(self.data.high) > i:
                            highest_high_in_lookback = max(highest_high_in_lookback, self.data.high[-i])
                
                if not math.isnan(highest_high_in_lookback):
                    reset_short_to = highest_high_in_lookback * (1 + self.p.williams_stop_buffer / 100)
        # --- End of f_getFlipResetWilliams --- 

        # Reset trails on flip using the more accurate reset_to values
        if self.flip_is_long and not prev_is_long: # Flipped to long
            self.williams_long_stop_trail = reset_long_to
        
        if self.flip_is_short and not prev_is_short: # Flipped to short
            self.williams_short_stop_trail = reset_short_to

        # Update plottable trails
        if self.flip_is_long:
            self.williams_long_stop_trail_plot = self.williams_long_stop_trail
        else:
            self.williams_long_stop_trail_plot = float('nan')

        if self.flip_is_short:
            self.williams_short_stop_trail_plot = self.williams_short_stop_trail
        else:
            self.williams_short_stop_trail_plot = float('nan')
            
        # Pine: Show both if _isLong and _isShort (initial state)
        if self.flip_is_long and self.flip_is_short:
             self.williams_long_stop_trail_plot = self.williams_long_stop_trail
             self.williams_short_stop_trail_plot = self.williams_short_stop_trail

        # Update the plottable lines
        self.plot_williams_long_stop[0] = self.williams_long_stop_trail_plot
        self.plot_williams_short_stop[0] = self.williams_short_stop_trail_plot

    def is_in_session(self):
        # Vérifie si nous sommes dans une session de trading active
        in_london = self.p.london_session and self.london_ind[0] > 0
        in_ny = self.p.ny_session and self.ny_ind[0] > 0
        in_tokyo = self.p.tokyo_session and self.tokyo_ind[0] > 0
        in_sydney = self.p.sydney_session and self.sydney_ind[0] > 0
        
        return in_london or in_ny or in_tokyo or in_sydney
    
    def next(self):
        # Only SuperTrend flips (validBuyEntry/validSellEntry) within 9:30–16:00
        dt = self.datas[0].datetime.datetime(0)
        # Time filter (9:30 to 16:00)
        if not (datetime.time(9, 30) <= dt.time() <= datetime.time(16, 0)):
            return
        # SuperTrend flip detection
        prev_tr = self.supertrend.trend_direction[-1]
        curr_tr = self.supertrend.trend_direction[0]
        # BUY on flip -1->1
        if prev_tr == -1 and curr_tr == 1:
            self.trade_log.append((dt, 'BUY', self.data.close[0]))
                self.order = self.buy()
        # SELL on flip 1->-1
        elif prev_tr == 1 and curr_tr == -1:
            self.trade_log.append((dt, 'SELL', self.data.close[0]))
                self.order = self.sell()
    
    def stop(self):
        # Statistiques finales
        avg_long_bars = self.sum_long_bars / self.count_long if self.count_long > 0 else 0
        avg_short_bars = self.sum_short_bars / self.count_short if self.count_short > 0 else 0
        avg_long_distance = self.sum_long_distance / self.count_long if self.count_long > 0 else 0
        avg_short_distance = self.sum_short_distance / self.count_short if self.count_short > 0 else 0
        
        # Calculer les pourcentages
        avg_long_percent = (avg_long_distance / self.start_price) * 100 if self.start_price > 0 else 0
        avg_short_percent = (avg_short_distance / self.start_price) * 100 if self.start_price > 0 else 0
        
        # Afficher les résultats
        self.log(f"Performance LONG: {self.count_long} trades, Barres: {avg_long_bars:.2f}, Distance: {avg_long_percent:.2f}%")
        self.log(f"Performance SHORT: {self.count_short} trades, Barres: {avg_short_bars:.2f}, Distance: {avg_short_percent:.2f}%")

        # Print recorded signals
        print('\nBuy Signals:')
        for ts, price in self.buy_signals:
            print(f'{ts} BUY @ {price:.5f}')

        print('Sell Signals:')
        for ts, price in self.sell_signals:
            print(f'{ts} SELL @ {price:.5f}')

        # Unified Trade Log
        print('\nTrade Log:')
        print('Idx\tTime\t\tSide\tPrice')
        for idx, (ts, side, price) in enumerate(self.trade_log, start=1):
            time_str = ts.strftime('%Y-%m-%d %H:%M')
            print(f'{idx}\t{time_str}\t{side}\t{price:.5f}')

    def next_bar(self, bar):
        """
        Manually feed a single OHLCV bar into the strategy.
        bar: dict with keys ['datetime','open','high','low','close','volume']
        """
        # Accumulate history
        if not hasattr(self, '_rl_history'):
            self._rl_history = []
        self._rl_history.append(bar)
        # Build a temporary DataFrame feed
        import pandas as pd
        from backtrader import Cerebro
        from backtrader.feeds import PandasData

        df = pd.DataFrame(self._rl_history)
        # Create and run a fresh Cerebro for this slice
        cerebro = Cerebro(stdstats=False)
        feed = PandasData(dataname=df,
                          datetime='datetime', open='open', high='high',
                          low='low', close='close', volume='volume')
        cerebro.adddata(feed)
        # Restore cash
        cerebro.broker.setcash(self.cash)
        # Add same strategy class
        cerebro.addstrategy(self.__class__)
        # Run only the next bar (Cerebro will process all bars in df)
        strat = cerebro.run()[0]
        # Update cash
        self.cash = cerebro.broker.getvalue()
        return self.cash


if __name__ == '__main__':
    # Configuration et exécution de la stratégie
    cerebro = bt.Cerebro()
    
    # --- IMPORTANT: Set your desired backtest date here ---
    # Replace YYYY, M, D with the year, month, and day you want to test
    # For example, for May 15, 2023:
    # target_year = 2023
    # target_month = 5
    # target_day = 15
    
    # PLEASE SPECIFY THE DATE YOU WANT TO TEST:
    target_year = 2025  # <-- REPLACE WITH YOUR DESIRED YEAR
    target_month = 5    # <-- REPLACE WITH YOUR DESIRED MONTH
    target_day = 16     # <-- REPLACE WITH YOUR DESIRED DAY (backtest on May 16)

    from_date = datetime.datetime(target_year, target_month, target_day, 0, 0, 0)
    # To backtest a single full day, todate should be the start of the next day.
    to_date = datetime.datetime(target_year, target_month, target_day) + datetime.timedelta(days=1)

    # Ajouter les données
    data = bt.feeds.GenericCSVData(
        dataname="EURUSD_data_1M.csv",
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1,
        fromdate=from_date, # Added fromdate
        todate=to_date      # Added todate
    )
    
    cerebro.adddata(data)
    
    # Ajouter la stratégie
    cerebro.addstrategy(MasterTrendStrategy)
    
    # Configurer le capital initial et les commissions
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)  # 0.01%
    
    # Ajouter des analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    # Observer pour afficher les signaux d'achat/vente sur le graphique
    cerebro.addobserver(bt.observers.BuySell)
    
    # Exécuter la stratégie
    print('Capital initial: %.2f' % cerebro.broker.getvalue())
    
    results = cerebro.run()
    strat = results[0]
    
    # Afficher les résultats
    print('Capital final: %.2f' % cerebro.broker.getvalue())
    print('Rendement: %.2f%%' % ((cerebro.broker.getvalue() / 10000.0 - 1.0) * 100))
    
    # Statistiques des trades
    trades = strat.analyzers.trades.get_analysis()
    
    print('\nAnalyse des transactions:')
    print('Total des trades:', trades.total.closed if hasattr(trades, 'total') else 0)
    print('Trades gagnants:', trades.won.total if hasattr(trades, 'won') else 0)
    print('Trades perdants:', trades.lost.total if hasattr(trades, 'lost') else 0)
    
    if hasattr(trades, 'won') and trades.won.total > 0:
        print('Gain moyen des trades gagnants:', trades.won.pnl.average)
    if hasattr(trades, 'lost') and trades.lost.total > 0:
        print('Perte moyenne des trades perdants:', trades.lost.pnl.average)
    
    # Sharpe Ratio
    sharpe = strat.analyzers.sharpe.get_analysis()
    print('\nSharpe Ratio:', sharpe['sharperatio'] if 'sharperatio' in sharpe else "N/A")
    
    # Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    print('Drawdown max: %.2f%%' % drawdown.max.drawdown if hasattr(drawdown, 'max') else "N/A")
    
    # Tracer le graphique
    # Décommentez pour afficher le graphique 
    cerebro.plot(style='candlestick') # Uncommented to show plot 