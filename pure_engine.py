import math
import datetime
from collections import deque

class MasterTrendEngine:
    """
    Pure-Python engine for the MasterTrend strategy.
    Call next_bar() with raw OHLCV to get long/short signals.
    """
    def __init__(self,
                 supertrend_period=10,
                 supertrend_multiplier=3,
                 left_range=2,
                 right_range=2,
                 williams_stop_buffer=0.0,
                 macd_fast=13,
                 macd_slow=26,
                 macd_signal=9,
                 p1_e31=5,
                 p1_m=9,
                 p1_l31=14,
                 sg_ema_short=10,
                 sg_ema_long=20,
                 sg_hl_period=4,
                 sg_median_lookback=10,
                 window=None):
        # Parameters
        self.sp = supertrend_period
        self.sm = supertrend_multiplier
        self.lr = left_range
        self.rr = right_range
        self.wsb = williams_stop_buffer
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.p1_e31 = p1_e31
        self.p1_m = p1_m
        self.p1_l31 = p1_l31
        self.sg_ema_short = sg_ema_short
        self.sg_ema_long = sg_ema_long
        self.sg_hl = sg_hl_period
        self.sg_med = sg_median_lookback

        # Buffers for price history
        max_history = max(self.sp+1, self.lr+self.rr+1, self.sg_med, self.macd_slow+1)
        self.opens  = deque(maxlen=max_history)
        self.highs  = deque(maxlen=max_history)
        self.lows   = deque(maxlen=max_history)
        self.closes = deque(maxlen=max_history)
        self.vols   = deque(maxlen=max_history)
        self.dts    = deque(maxlen=max_history)

        # Indicator state placeholders
        self.atr = None
        self.prev_tr = None
        self.basic_up = None
        self.basic_dn = None
        self.final_up = None
        self.final_dn = None
        self.trend_dir = 1  # +1 up, -1 down
        # Jurik MA state
        self.jma_e0 = 0.0
        self.jma_e1 = 0.0
        self.jma_e2 = 0.0
        self.jma_val = 0.0
        # MACD state: use simple EMA tracking
        self.ema_fast = None
        self.ema_slow = None
        self.ema_signal = None
        # Williams fractal state
        # P1 and StopGap state
        self.sg_window = []
        # Trend session ignored in pure engine

        # Position state
        self.in_long = False
        self.in_short = False
        # Fractal history and trailing-stop state
        self.fr_high_flags = deque(maxlen=max_history)
        self.fr_low_flags = deque(maxlen=max_history)
        # flip-state: start both sides active
        self._is_long = True
        self._is_short = True
        # trailing stop levels and plots
        self.williams_long_stop = None
        self.williams_short_stop = None
        self.williams_long_trail_plot = None
        self.williams_short_trail_plot = None

    def next_bar(self, bar):
        """
        Process one bar: bar = dict with keys:
            'datetime', 'open','high','low','close','volume'
        Returns: (long_signal:bool, short_signal:bool)
        """
        # 1) Append to history
        dt = bar['datetime'] if isinstance(bar['datetime'], datetime.datetime) else datetime.datetime.strptime(bar['datetime'], '%Y-%m-%d %H:%M:%S')
        self.dts.append(dt)
        self.opens.append(bar['open'])
        self.highs.append(bar['high'])
        self.lows.append(bar['low'])
        self.closes.append(bar['close'])
        self.vols.append(bar['volume'])

        # 2) Compute ATR
        self._update_atr()
        # 3) Compute SuperTrend
        self._update_supertrend()
        # 4) Compute other indicators
        self._update_jma()
        self._update_macd()
        self._update_p1()
        self._update_stopgap()

        # 5) Compute Williams fractals
        self._update_fractals()

        # 6) Combine signals
        long_signal = False
        short_signal = False
        # Example placeholder logic:
        if self.trend_dir == 1 and not self.in_long:
            long_signal = True
            self.in_long = True
            self.in_short = False
        elif self.trend_dir == -1 and not self.in_short:
            short_signal = True
            self.in_short = True
            self.in_long = False

        return long_signal, short_signal

    def _update_atr(self):
        if len(self.closes) < 2:
            return
        high = self.highs[-1]
        low = self.lows[-1]
        prev_close = self.closes[-2]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if self.atr is None:
            self.atr = tr
        else:
            self.atr = (self.atr * (self.sp - 1) + tr) / self.sp

    def _update_supertrend(self):
        # Check we have enough bars
        if len(self.closes) < self.sp:
            return
        src = (self.highs[-1] + self.lows[-1]) / 2.0
        basic_up = src - self.sm * self.atr
        basic_dn = src + self.sm * self.atr
        if self.final_up is None:
            # initial
            self.final_up = basic_up
            self.final_dn = basic_dn
            self.trend_dir = 1
        else:
            prev_close = self.closes[-2]
            # final up
            if prev_close > self.final_up:
                self.final_up = max(basic_up, self.final_up)
            else:
                self.final_up = basic_up
            # final dn
            if prev_close < self.final_dn:
                self.final_dn = min(basic_dn, self.final_dn)
            else:
                self.final_dn = basic_dn
            # trend
            if self.trend_dir == -1 and self.closes[-1] > self.final_dn:
                self.trend_dir = 1
            elif self.trend_dir == 1 and self.closes[-1] < self.final_up:
                self.trend_dir = -1

    # EMA helper for simple EMA calculations
    def _ema(self, value, prev_ema, period):
        alpha = 2.0 / (period + 1)
        return value if prev_ema is None else alpha * value + (1 - alpha) * prev_ema

    def _update_jma(self):
        # Jurik Moving Average
        if len(self.closes) < getattr(self, 'jma_period', 0):
            return
        src = self.closes[-1]
        # Initialize JMA parameters if not done
        if not hasattr(self, 'jma_alpha'):
            # Default JMA params
            self.jma_period = 7
            self.jma_phase = 50
            self.jma_power = 2
            # Compute ratios
            self.jma_phase_ratio = 0.5 if self.jma_phase < -100 else 2.5 if self.jma_phase > 100 else self.jma_phase / 100 + 1.5
            self.jma_beta = 0.45 * (self.jma_period - 1) / (0.45 * (self.jma_period - 1) + 2)
            self.jma_alpha = pow(self.jma_beta, self.jma_power)
        # Update JMA states
        self.jma_e0 = (1 - self.jma_alpha) * src + self.jma_alpha * self.jma_e0
        self.jma_e1 = (src - self.jma_e0) * (1 - self.jma_beta) + self.jma_beta * self.jma_e1
        self.jma_e2 = (self.jma_e0 + self.jma_phase_ratio * self.jma_e1 - self.jma_val) * pow(1 - self.jma_alpha, 2) + pow(self.jma_alpha, 2) * self.jma_e2
        self.jma_val = self.jma_e2 + self.jma_val

    def _update_macd(self):
        # Standard EMA-based MACD
        if len(self.closes) < max(self.macd_fast, self.macd_slow, self.macd_signal):
            return
        price = self.closes[-1]
        self.ema_fast = self._ema(price, self.ema_fast, self.macd_fast)
        self.ema_slow = self._ema(price, self.ema_slow, self.macd_slow)
        dif = self.ema_fast - self.ema_slow
        self.ema_signal = self._ema(dif, self.ema_signal, self.macd_signal)
        self.macd_hist = (dif - self.ema_signal) * 2

    def _update_p1(self):
        # P1 Indicator logic
        spreadv = (self.closes[-1] - self.opens[-1]) * 100.0 * self.closes[-1]
        # Cumulative sum of spread
        if not hasattr(self, 'cum_spread'):
            self.cum_spread = 0.0
        self.cum_spread += spreadv
        pt_line = spreadv + self.cum_spread
        # EMA calculations
        if not hasattr(self, 'p1_ema_l31'):
            self.p1_ema_l31 = None
            self.p1_ema_m = None
            self.p1_ema_e31 = None
        self.p1_ema_l31 = self._ema(pt_line, self.p1_ema_l31, self.p1_l31)
        self.p1_ema_m  = self._ema(pt_line, self.p1_ema_m,  self.p1_m)
        self.p1_ema_e31= self._ema(pt_line, self.p1_ema_e31,self.p1_e31)
        a1 = self.p1_ema_l31 - self.p1_ema_m
        b1 = self.p1_ema_e31 - self.p1_ema_m
        self.p1 = a1 + b1
        self.b1 = b1
        self.p1_bullish = b1 >= self.p1
        self.p1_bearish = b1 <= self.p1

    def _update_stopgap(self):
        # StopGap Median Filter logic
        if len(self.closes) < self.sg_hl:
            return
        curr_high = self.highs[-1]
        curr_low  = self.lows[-1]
        # EMA trend detection
        if not hasattr(self, 'ema_sg_short_val'):
            self.ema_sg_short_val = None
            self.ema_sg_long_val  = None
        # Use close price for EMAs
        price = self.closes[-1]
        self.ema_sg_short_val = self._ema(price, self.ema_sg_short_val, self.sg_ema_short)
        self.ema_sg_long_val  = self._ema(price, self.ema_sg_long_val,  self.sg_ema_long)
        trend_up   = self.ema_sg_short_val > self.ema_sg_long_val
        trend_down = self.ema_sg_short_val < self.ema_sg_long_val
        # Highest/lowest over lookback period
        highest = max(self.highs)
        lowest  = min(self.lows)
        stop_gap = 0.0
        if trend_down and curr_high < highest:
            stop_gap = abs(highest - curr_low)
        elif trend_up and curr_low > lowest:
            stop_gap = abs(curr_high  - lowest)
        # Median lookback window
        if not hasattr(self, 'sg_median_window'):
            from collections import deque as _dq
            self.sg_median_window = _dq(maxlen=self.sg_med)
        self.sg_median_window.append(self.highs[-1] - self.lows[-1])
        median_val = 0.0
        if len(self.sg_median_window) == self.sg_med:
            sorted_w = sorted(self.sg_median_window)
            mid = len(sorted_w) // 2
            if len(sorted_w) % 2 == 0:
                median_val = (sorted_w[mid-1] + sorted_w[mid]) / 2.0
            else:
                median_val = sorted_w[mid]
        self.sg_filter_passed = stop_gap > median_val

    def _update_fractals(self):
        """
        Detect Williams high/low fractals with left/right range and apply optional buffer.
        """
        # Need enough history
        if len(self.highs) < self.lr + self.rr + 1:
            return
        # Bar index for fractal (rr bars ago)
        ph = self.highs[-1 - self.rr]
        pl = self.lows[-1 - self.rr]
        # High fractal
        is_high = True
        # Left side check
        for i in range(self.rr + 1, self.rr + self.lr + 1):
            if self.highs[-1 - i] > ph:
                is_high = False
                break
        if is_high:
            # Right side check (excluding pivot)
            for i in range(self.rr):
                if self.highs[-1 - i] >= ph:
                    is_high = False
                    break
        # Low fractal
        is_low = True
        for i in range(self.rr + 1, self.rr + self.lr + 1):
            if self.lows[-1 - i] < pl:
                is_low = False
                break
        if is_low:
            for i in range(self.rr):
                if self.lows[-1 - i] <= pl:
                    is_low = False
                    break
        # Append fractal flags to history
        self.fr_high_flags.append(is_high)
        self.fr_low_flags.append(is_low)
        # Store fractal flags and prices
        self.is_williams_high = is_high
        self.is_williams_low = is_low
        self.williams_high_price = ph if is_high else None
        self.williams_low_price = pl if is_low else None
        # Apply percentage buffer
        if self.wsb:
            if is_high:
                self.williams_high_price_buf = ph * (1 + self.wsb / 100.0)
            if is_low:
                self.williams_low_price_buf = pl * (1 - self.wsb / 100.0)
        # Persist raw stops on fractal (for reset)
        raw_long = getattr(self, 'williams_low_price_buf', pl) if is_low else None
        raw_short = getattr(self, 'williams_high_price_buf', ph) if is_high else None

        # Initialize stops on first fractal if not set
        if self.williams_long_stop is None and raw_long is not None:
            self.williams_long_stop = raw_long
        if self.williams_short_stop is None and raw_short is not None:
            self.williams_short_stop = raw_short
        # Flip/trailing-stop logic
        prev_long = self._is_long
        prev_short = self._is_short
        close_price = self.closes[-1]
        # Determine flip events
        flip_long_now = prev_short and self.williams_short_stop is not None and close_price > self.williams_short_stop
        flip_short_now = prev_long and self.williams_long_stop is not None and close_price < self.williams_long_stop
        # Update flip-state
        self._is_long = flip_long_now or (prev_long and not flip_short_now)
        self._is_short = flip_short_now or (prev_short and not flip_long_now)
        # Reset stops on flip
        if flip_long_now:
            self.williams_long_stop = self._get_reset_long()
        if flip_short_now:
            self.williams_short_stop = self._get_reset_short()
        # Determine display levels
        self.williams_long_trail_plot = self.williams_long_stop if self._is_long else None
        self.williams_short_trail_plot = self.williams_short_stop if self._is_short else None

    def _get_reset_long(self):
        """Compute trailing long stop reset based on last low fractal."""
        flags = list(self.fr_low_flags)
        if True not in flags or self.wsb is None:
            return self.williams_long_stop
        bars_since = flags[::-1].index(True)
        bars_to_go_back = bars_since + self.rr
        lows = list(self.lows)[-bars_to_go_back-1:]
        lowest_low = min(lows) if lows else self.williams_long_stop
        return lowest_low * (1 - self.wsb / 100.0)

    def _get_reset_short(self):
        """Compute trailing short stop reset based on last high fractal."""
        flags = list(self.fr_high_flags)
        if True not in flags or self.wsb is None:
            return self.williams_short_stop
        bars_since = flags[::-1].index(True)
        bars_to_go_back = bars_since + self.lr
        highs = list(self.highs)[-bars_to_go_back-1:]
        highest_high = max(highs) if highs else self.williams_short_stop
        return highest_high * (1 + self.wsb / 100.0)