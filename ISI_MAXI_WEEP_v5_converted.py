import backtrader as bt
import datetime
import pandas as pd # We will likely need pandas for data manipulation
# import pandas_ta as ta # For technical indicators, if needed and not in backtrader

# --- Helper Functions and Custom Indicators ---

def parse_tf_string(tf_string):
    """Converts a timeframe string (e.g., '60', 'D', 'W') to backtrader timeframe and compression."""
    if tf_string.isdigit(): # Minutes
        minutes = int(tf_string)
        if minutes >= 60: # Check for hours
            hours = minutes // 60
            if hours > 0 and minutes % 60 == 0: # Whole hours
                 # Backtrader doesn't have a direct 'Hours' timeframe for resampling, use Minutes
                 # For example, 240 minutes is simply bt.TimeFrame.Minutes, compression=240
                 pass # Handled by minutes directly
        return bt.TimeFrame.Minutes, minutes
    elif tf_string == 'D':
        return bt.TimeFrame.Days, 1
    elif tf_string == 'W':
        return bt.TimeFrame.Weeks, 1
    elif tf_string == 'M':
        return bt.TimeFrame.Months, 1
    # Add more cases if needed (e.g., 'H' for hours, though '60', '120', '240' are handled as minutes)
    raise ValueError(f"Unsupported timeframe string: {tf_string}")

class PivotHighPine(bt.Indicator):
    """Mimics PineScript's ta.pivothigh(source, leftbars, rightbars)"""
    lines = ('ph',)
    params = (('left', 5), ('right', 5), ('_datasrc', None),)
    plotinfo = dict(plot=False) # Usually these are intermediate, not plotted directly

    def __init__(self):
        self.addminperiod(self.p.left + self.p.right + 1)
        self.source = self.p._datasrc if self.p._datasrc is not None else self.data.high

    def next(self):
        if len(self.source) < (self.p.left + self.p.right + 1):
            self.lines.ph[0] = float('nan')
            return
        pivot_candidate_val = self.source[-self.p.right]
        is_pivot = True
        for i in range(1, self.p.left + 1):
            if pivot_candidate_val <= self.source[-self.p.right - i]: is_pivot = False; break
        if not is_pivot: self.lines.ph[0] = float('nan'); return
        for i in range(1, self.p.right + 1):
            if pivot_candidate_val <= self.source[-self.p.right + i]: is_pivot = False; break
        self.lines.ph[0] = pivot_candidate_val if is_pivot else float('nan')

class PivotLowPine(bt.Indicator):
    """Mimics PineScript's ta.pivotlow(source, leftbars, rightbars)"""
    lines = ('pl',)
    params = (('left', 5), ('right', 5), ('_datasrc', None),)
    plotinfo = dict(plot=False)

    def __init__(self):
        self.addminperiod(self.p.left + self.p.right + 1)
        self.source = self.p._datasrc if self.p._datasrc is not None else self.data.low

    def next(self):
        if len(self.source) < (self.p.left + self.p.right + 1):
            self.lines.pl[0] = float('nan')
            return
        pivot_candidate_val = self.source[-self.p.right]
        is_pivot = True
        for i in range(1, self.p.left + 1):
            if pivot_candidate_val >= self.source[-self.p.right - i]: is_pivot = False; break
        if not is_pivot: self.lines.pl[0] = float('nan'); return
        for i in range(1, self.p.right + 1):
            if pivot_candidate_val >= self.source[-self.p.right + i]: is_pivot = False; break
        self.lines.pl[0] = pivot_candidate_val if is_pivot else float('nan')

# Mimics PineScript's valuewhen function
class Valuewhen(bt.Indicator):
    lines = ('valuewhen',)
    params = (('condition', None), ('source', None), ('occurrence', 0),)
    
    def __init__(self):
        self.triggered = 0
        self.previous_values = []
        
    def next(self):
        if self.p.condition[0]:
            self.triggered = 1
            self.previous_values.insert(0, self.p.source[0])
            
        if self.triggered and self.p.occurrence < len(self.previous_values):
            self.lines.valuewhen[0] = self.previous_values[self.p.occurrence]
        else:
            self.lines.valuewhen[0] = float('nan')

class HTFPivotValue(bt.Indicator):
    """
    Calculates the persistent pivot value from a higher timeframe.
    Equivalent to PineScript's:
    ta.valuewhen(not na(ta.pivothigh(high, len, len)), high[len], 0)
    or ta.valuewhen(not na(ta.pivotlow(low, len, len)), low[len], 0)
    """
    lines = ('htf_pivot_value',)
    params = (
        ('pivot_len', 10),
        ('is_high', True),
        ('pivot_source', "High/Low"), # Can be "High/Low" or "Close"
        ('plot', True),
        ('_datasrc', None)
    )
    
    plotinfo = dict(subplot=False)
    
    def __init__(self):
        # Choose the source data for pivots based on pivot_source param
        datasource = self.datas[0] if self.p._datasrc is None else self.p._datasrc
        source = datasource.high if self.p.is_high else datasource.low
        
        if self.p.is_high:
            self.pivot_indicator = PivotHighPine(datasource, left=self.p.pivot_len, right=self.p.pivot_len)
        else:
            self.pivot_indicator = PivotLowPine(datasource, left=self.p.pivot_len, right=self.p.pivot_len)
        
        # Détection d'un pivot (non NaN)
        condition_line = self.pivot_indicator.lines.ph if self.p.is_high else self.pivot_indicator.lines.pl
        pivot_detected = bt.Or(condition_line > 0, condition_line < 0)  # Une valeur non-NaN sera soit positive soit négative
        
        # Remplacer l'appel à Valuewhen par notre propre implémentation
        self.lines.htf_pivot_value = Valuewhen(
            condition=pivot_detected,
            source=source,
            occurrence=0
        )
        if self.p.plot:
            self.plotinfo.plotmaster = self.data # Plot on the same chart as the data feed it's applied to

# Mimics WilliamsFractal indicator since it's not in backtrader
class WilliamsFractal(bt.Indicator):
    """Williams Fractals indicator"""
    lines = ('high', 'low')
    params = (('periods', 2),)
    
    def __init__(self):
        self.addminperiod(2 * self.p.periods + 1)
    
    def next(self):
        # High fractal detection
        high_middle = self.data.high[-self.p.periods]
        high_fractal = True
        for i in range(1, self.p.periods + 1):
            if (self.data.high[-self.p.periods - i] >= high_middle or
                self.data.high[-self.p.periods + i] >= high_middle):
                high_fractal = False
                break
                
        # Low fractal detection
        low_middle = self.data.low[-self.p.periods]
        low_fractal = True
        for i in range(1, self.p.periods + 1):
            if (self.data.low[-self.p.periods - i] <= low_middle or
                self.data.low[-self.p.periods + i] <= low_middle):
                low_fractal = False
                break
                
        self.lines.high[0] = high_middle if high_fractal else float('nan')
        self.lines.low[0] = low_middle if low_fractal else float('nan')

class SweepAndFractalStrategy(bt.Strategy):
    params = (
        # Sweep & Fractal Inputs
        ('htf_timeframe_str', '60'), 
        ('cooldown_bars', 10),      
        ('pivot_len', 10),          
        ('htf_only_breakouts', True),
        ('show_fractals', True), # Determines if fractal logic runs, plotting is separate 
        ('fractal_periods', 2),     # n in PineScript

        # MTF Trend Filter Inputs
        ('filter_lookback', 5),     
        ('filter_pivot_type', "High/Low"), 
        ('filter_tf1_str', '240'),  
        ('filter_tf2_str', 'D'),    
        ('filter_tf3_str', 'W'),    
        ('filter_use_tf1', False),   
        ('filter_use_tf2', False),   
        ('filter_use_tf3', False),  

        # Trend Table Display Inputs
        ('show_trend_table', True), # Enables logging of MTF trend summary
        ('table_location', 'Top-Right'), # Not used in backtrader plotting directly

        # Kill Zone Filter Inputs
        ('filter_by_kill_zone', False), 
        ('show_asia_kz_filter', False), 
        ('asia_kz_time', '2300-0355'), 
        ('show_london_kz_filter', False),
        ('london_kz_time', '0700-0955'),
        ('show_ny_am_kz_filter', False),
        ('ny_am_kz_time', '1430-1655'), 
        ('show_ny_pm_kz_filter', False),
        ('ny_pm_kz_time', '1930-2055'), 

        # Strategy Settings
        ('rr_ratio', 2.0),          
        ('use_stop_loss', True),    
        ('use_take_profit', True),  
    )

    lines = (
        'display_VLow', 'display_VHigh', 
        'bull_box_high', 'bull_box_low', 
        'bear_box_high', 'bear_box_low',
        'active_sl', 'active_tp',
    )

    def __init__(self):
        # En initialisation, l'accès aux données n'est pas encore disponible
        # self.log("Strategy Initialized") # Ligne commentée car elle cause une erreur
        print("Strategy Initialized") # Utilisation de print standard au lieu de self.log
        self.d_close = self.datas[0].close
        self.d_high = self.datas[0].high
        self.d_low = self.datas[0].low
        self.d_open = self.datas[0].open
        self.d_volume = self.datas[0].volume
        self.d_datetime = self.datas[0].datetime

        data_idx = 1
        self.htf_data = self.htf_VLow_indicator = self.htf_VHigh_indicator = None
        if self.p.htf_timeframe_str:
             self.htf_data = self.datas[data_idx]
             self.htf_VLow_indicator = HTFPivotValue(
                 self.htf_data, pivot_len=self.p.pivot_len, is_high=False, plot=False
             )
             self.htf_VHigh_indicator = HTFPivotValue(
                 self.htf_data, pivot_len=self.p.pivot_len, is_high=True, plot=False
             )
             data_idx +=1
        
        self.filter_tf1_data = self.filter_tf1_high_level = self.filter_tf1_low_level = None
        if self.p.filter_tf1_str and self.p.filter_use_tf1:
            self.filter_tf1_data = self.datas[data_idx]
            self.filter_tf1_high_level = HTFPivotValue(self.filter_tf1_data, pivot_len=self.p.filter_lookback, is_high=True, pivot_source=self.p.filter_pivot_type, plot=False)
            self.filter_tf1_low_level = HTFPivotValue(self.filter_tf1_data, pivot_len=self.p.filter_lookback, is_high=False, pivot_source=self.p.filter_pivot_type, plot=False)
            data_idx +=1

        self.filter_tf2_data = self.filter_tf2_high_level = self.filter_tf2_low_level = None
        if self.p.filter_tf2_str and self.p.filter_use_tf2:
            self.filter_tf2_data = self.datas[data_idx]
            self.filter_tf2_high_level = HTFPivotValue(self.filter_tf2_data, pivot_len=self.p.filter_lookback, is_high=True, pivot_source=self.p.filter_pivot_type, plot=False)
            self.filter_tf2_low_level = HTFPivotValue(self.filter_tf2_data, pivot_len=self.p.filter_lookback, is_high=False, pivot_source=self.p.filter_pivot_type, plot=False)
            data_idx +=1

        self.filter_tf3_data = self.filter_tf3_high_level = self.filter_tf3_low_level = None
        if self.p.filter_tf3_str and self.p.filter_use_tf3:
            self.filter_tf3_data = self.datas[data_idx]
            self.filter_tf3_high_level = HTFPivotValue(self.filter_tf3_data, pivot_len=self.p.filter_lookback, is_high=True, pivot_source=self.p.filter_pivot_type, plot=False)
            self.filter_tf3_low_level = HTFPivotValue(self.filter_tf3_data, pivot_len=self.p.filter_lookback, is_high=False, pivot_source=self.p.filter_pivot_type, plot=False)
            data_idx +=1

        self.fractal_indicator = WilliamsFractal(periods=self.p.fractal_periods)
        if not self.p.show_fractals:
             self.fractal_indicator.plotinfo.plot = False # Disable plotting if show_fractals is false

        # Initialize state variables
        self.prevVLow = float('nan'); self.prevVHigh = float('nan')
        self.displayVLow = float('nan'); self.displayVHigh = float('nan')
        # Ne pas initialiser les lignes directement - cela doit être fait dans next()
        # self.lines.display_VLow = float('nan'); self.lines.display_VHigh = float('nan') # Plot lines

        self.lastLowSweepBar = float('nan'); self.lastHighSweepBar = float('nan')
        self.sweptLow = False; self.sweptHigh = False

        self.bullActive = False; self.bullSweepBar = float('nan'); self.bullDone = False
        self.bullBreakoutDetected = False; self.bullBoxCreated = False
        self.foundBullHighFractal = False; self.bullHighFractalIndex = float('nan'); self.bullHighFractalLevel = float('nan')
        self.foundBullLowFractal = False; self.bullLowFractalIndex = float('nan'); self.bullLowFractalLevel = float('nan')
        # Ne pas initialiser les lignes directement
        # self.lines.bull_box_high = float('nan'); self.lines.bull_box_low = float('nan') # Plot lines

        self.bearActive = False; self.bearSweepBar = float('nan'); self.bearDone = False
        self.bearBreakoutDetected = False; self.bearBoxCreated = False
        self.foundBearHighFractal = False; self.bearHighFractalIndex = float('nan'); self.bearHighFractalLevel = float('nan')
        self.foundBearLowFractal = False; self.bearLowFractalIndex = float('nan'); self.bearLowFractalLevel = float('nan')
        # Ne pas initialiser les lignes directement
        # self.lines.bear_box_high = float('nan'); self.lines.bear_box_low = float('nan') # Plot lines

        self.slLevel = float('nan') 
        self.activeLongSL = float('nan'); self.activeLongTP = float('nan')
        self.activeShortSL = float('nan'); self.activeShortTP = float('nan')
        # Ne pas initialiser les lignes directement
        # self.lines.active_sl = float('nan'); self.lines.active_tp = float('nan') # Plot lines

        self.inUptrendF1 = False; self.inDowntrendF1 = False
        self.inUptrendF2 = False; self.inDowntrendF2 = False
        self.inUptrendF3 = False; self.inDowntrendF3 = False
        self.currentActiveKZ_name = "N/A"
        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.datetime(0)
        if isinstance(dt, float): dt = bt.num2date(dt)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status in [order.Completed]:
            if order.isbuy(): self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            else: self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]: self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
        # Reset active SL/TP lines after trade closes
        self.lines.active_sl[0] = float('nan')
        self.lines.active_tp[0] = float('nan')

    def _calculate_mtf_signal(self, main_data_high, main_data_low, mtf_high_level_line, mtf_low_level_line, current_in_uptrend):
        uptrend_signal_prev, downtrend_signal_prev = False, False
        new_in_uptrend = current_in_uptrend
        if len(main_data_high) > 1 and len(main_data_low) > 1 and not pd.isna(mtf_high_level_line[0]) and not pd.isna(mtf_low_level_line[0]):
             if main_data_high[-1] > mtf_high_level_line[0]: uptrend_signal_prev = True
             if main_data_low[-1] < mtf_low_level_line[0]: downtrend_signal_prev = True
        if uptrend_signal_prev: new_in_uptrend = True
        elif downtrend_signal_prev: new_in_uptrend = False
        return new_in_uptrend, not new_in_uptrend

    def _is_in_kill_zone_session(self, current_time_dt):
        if not self.p.filter_by_kill_zone: self.currentActiveKZ_name = "Disabled"; return True
        current_bar_time = current_time_dt.time()
        def parse_session(s_str): start, end = s_str.split('-'); return datetime.time(int(start[:2]), int(start[2:])), datetime.time(int(end[:2]), int(end[2:]))
        sessions = []
        if self.p.show_asia_kz_filter: sessions.append(("Asia", self.p.asia_kz_time))
        if self.p.show_london_kz_filter: sessions.append(("London", self.p.london_kz_time))
        if self.p.show_ny_am_kz_filter: sessions.append(("NY AM", self.p.ny_am_kz_time))
        if self.p.show_ny_pm_kz_filter: sessions.append(("NY PM", self.p.ny_pm_kz_time))
        for name, time_str in sessions:
            s, e = parse_session(time_str)
            if (s <= e and s <= current_bar_time <= e) or (s > e and (current_bar_time >= s or current_bar_time <= e)):
                self.currentActiveKZ_name = name; return True
        self.currentActiveKZ_name = "None"; return False

    def next(self):
        current_bar_idx = len(self.datas[0]); current_datetime_dt = self.data.datetime.datetime(0)
        # Mise à jour des lignes correctement
        self.lines.display_VLow[0] = self.displayVLow
        self.lines.display_VHigh[0] = self.displayVHigh
        self.lines.bull_box_high[0] = float('nan')
        self.lines.bull_box_low[0] = float('nan')
        self.lines.bear_box_high[0] = float('nan')
        self.lines.bear_box_low[0] = float('nan')
        
        if not self.position: # Only update SL/TP lines if no position (or after close)
            self.lines.active_sl[0] = float('nan')
            self.lines.active_tp[0] = float('nan')

        htfVLow_current, htfVHigh_current = float('nan'), float('nan')
        if self.htf_VLow_indicator: htfVLow_current = self.htf_VLow_indicator.lines.htf_pivot_value[0]
        if self.htf_VHigh_indicator: htfVHigh_current = self.htf_VHigh_indicator.lines.htf_pivot_value[0]
        lowPivotChanged = not pd.isna(htfVLow_current) and (pd.isna(self.prevVLow) or htfVLow_current != self.prevVLow)
        highPivotChanged = not pd.isna(htfVHigh_current) and (pd.isna(self.prevVHigh) or htfVHigh_current != self.prevVHigh)
        if lowPivotChanged: self.displayVLow = htfVLow_current
        if highPivotChanged: self.displayVHigh = htfVHigh_current
        if not pd.isna(htfVLow_current): self.prevVLow = htfVLow_current
        if not pd.isna(htfVHigh_current): self.prevVHigh = htfVHigh_current
        # Mise à jour des lignes à nouveau
        self.lines.display_VLow[0] = self.displayVLow
        self.lines.display_VHigh[0] = self.displayVHigh

        if self.filter_tf1_high_level: self.inUptrendF1, self.inDowntrendF1 = self._calculate_mtf_signal(self.d_high, self.d_low, self.filter_tf1_high_level.lines.htf_pivot_value, self.filter_tf1_low_level.lines.htf_pivot_value, self.inUptrendF1)
        if self.filter_tf2_high_level: self.inUptrendF2, self.inDowntrendF2 = self._calculate_mtf_signal(self.d_high, self.d_low, self.filter_tf2_high_level.lines.htf_pivot_value, self.filter_tf2_low_level.lines.htf_pivot_value, self.inUptrendF2)
        if self.filter_tf3_high_level: self.inUptrendF3, self.inDowntrendF3 = self._calculate_mtf_signal(self.d_high, self.d_low, self.filter_tf3_high_level.lines.htf_pivot_value, self.filter_tf3_low_level.lines.htf_pivot_value, self.inUptrendF3)
        self.bullFilterCondition = True
        if self.p.filter_use_tf1 and self.filter_tf1_data: self.bullFilterCondition = self.bullFilterCondition and self.inUptrendF1
        if self.p.filter_use_tf2 and self.filter_tf2_data: self.bullFilterCondition = self.bullFilterCondition and self.inUptrendF2
        if self.p.filter_use_tf3 and self.filter_tf3_data: self.bullFilterCondition = self.bullFilterCondition and self.inUptrendF3
        self.bearFilterCondition = True
        if self.p.filter_use_tf1 and self.filter_tf1_data: self.bearFilterCondition = self.bearFilterCondition and self.inDowntrendF1
        if self.p.filter_use_tf2 and self.filter_tf2_data: self.bearFilterCondition = self.bearFilterCondition and self.inDowntrendF2
        if self.p.filter_use_tf3 and self.filter_tf3_data: self.bearFilterCondition = self.bearFilterCondition and self.inDowntrendF3
        
        killZoneFilterActive = self._is_in_kill_zone_session(current_datetime_dt)

        isHighFractal, isLowFractal = False, False
        if self.p.show_fractals: # Only calculate if show_fractals is true
            if len(self.fractal_indicator.lines.high) > self.p.fractal_periods and not pd.isna(self.fractal_indicator.lines.high[-self.p.fractal_periods]): isHighFractal = True
            if len(self.fractal_indicator.lines.low) > self.p.fractal_periods and not pd.isna(self.fractal_indicator.lines.low[-self.p.fractal_periods]): isLowFractal = True

        if lowPivotChanged: self.sweptLow = False; self.lastLowSweepBar = float('nan') 
        if highPivotChanged: self.sweptHigh = False; self.lastHighSweepBar = float('nan')
        if not self.sweptLow and (pd.isna(self.lastLowSweepBar) or current_bar_idx >= self.lastLowSweepBar + self.p.cooldown_bars) and not pd.isna(self.displayVLow) and self.d_close[0] < self.displayVLow:
            self.sweptLow = True; self.lastLowSweepBar = current_bar_idx
        if not self.sweptHigh and (pd.isna(self.lastHighSweepBar) or current_bar_idx >= self.lastHighSweepBar + self.p.cooldown_bars) and not pd.isna(self.displayVHigh) and self.d_close[0] > self.displayVHigh:
            self.sweptHigh = True; self.lastHighSweepBar = current_bar_idx
        
        if lowPivotChanged: self.bullActive = False; self.bullDone = False; self.bullBreakoutDetected = False; self.bullBoxCreated = False; self.foundBullHighFractal = False; self.foundBullLowFractal = False
        if highPivotChanged: self.bearActive = False; self.bearDone = False; self.bearBreakoutDetected = False; self.bearBoxCreated = False; self.foundBearHighFractal = False; self.foundBearLowFractal = False

        if self.sweptLow and not self.bullActive and not self.bullBreakoutDetected: self.bullActive = True; self.bullSweepBar = current_bar_idx; self.foundBullHighFractal = False; self.bullHighFractalIndex = float('nan'); self.bullHighFractalLevel = float('nan'); self.foundBullLowFractal = False; self.bullLowFractalIndex = float('nan'); self.bullLowFractalLevel = float('nan'); self.bullDone = False; self.bullBoxCreated = False
        if self.sweptHigh and not self.bearActive and not self.bearBreakoutDetected: self.bearActive = True; self.bearSweepBar = current_bar_idx; self.foundBearHighFractal = False; self.bearHighFractalIndex = float('nan'); self.bearHighFractalLevel = float('nan'); self.foundBearLowFractal = False; self.bearLowFractalIndex = float('nan'); self.bearLowFractalLevel = float('nan'); self.bearDone = False; self.bearBoxCreated = False
        
        if self.p.show_fractals: # Fractal logic only if enabled
            fractal_check_bar_index = current_bar_idx - self.p.fractal_periods
            if self.bullActive and not self.bullBoxCreated and not self.foundBullHighFractal and isHighFractal and fractal_check_bar_index >= self.bullSweepBar: self.foundBullHighFractal = True; self.bullHighFractalIndex = fractal_check_bar_index; self.bullHighFractalLevel = self.fractal_indicator.lines.high[-self.p.fractal_periods]
            if self.bullActive and not self.bullBoxCreated and not self.foundBullLowFractal and isLowFractal and fractal_check_bar_index >= self.bullSweepBar: self.foundBullLowFractal = True; self.bullLowFractalIndex = fractal_check_bar_index; self.bullLowFractalLevel = self.fractal_indicator.lines.low[-self.p.fractal_periods]
            if self.bearActive and not self.bearBoxCreated and not self.foundBearHighFractal and isHighFractal and fractal_check_bar_index >= self.bearSweepBar: self.foundBearHighFractal = True; self.bearHighFractalIndex = fractal_check_bar_index; self.bearHighFractalLevel = self.fractal_indicator.lines.high[-self.p.fractal_periods]
            if self.bearActive and not self.bearBoxCreated and not self.foundBearLowFractal and isLowFractal and fractal_check_bar_index >= self.bearSweepBar: self.foundBearLowFractal = True; self.bearLowFractalIndex = fractal_check_bar_index; self.bearLowFractalLevel = self.fractal_indicator.lines.low[-self.p.fractal_periods]

        if self.bullActive and self.foundBullHighFractal and self.foundBullLowFractal and not self.bullBoxCreated: 
            self.bullBoxCreated = True
            self.lines.bull_box_high[0] = self.bullHighFractalLevel
            self.lines.bull_box_low[0] = self.bullLowFractalLevel
        if self.bearActive and self.foundBearHighFractal and self.foundBearLowFractal and not self.bearBoxCreated: 
            self.bearBoxCreated = True
            self.lines.bear_box_high[0] = self.bearHighFractalLevel
            self.lines.bear_box_low[0] = self.bearLowFractalLevel
            
        if self.position.size == 0:
            if self.bullActive and self.bullBoxCreated and not self.bullDone and not self.bullBreakoutDetected:
                if self.d_close[0] > self.bullHighFractalLevel:
                    if self.bullFilterCondition and killZoneFilterActive: 
                        self.activeLongSL = self.bullLowFractalLevel
                        self.activeLongTP = self.bullHighFractalLevel + (self.bullHighFractalLevel - self.activeLongSL) * self.p.rr_ratio
                        self.log(f"BUY SIGNAL: SL={self.activeLongSL:.5f}, TP={self.activeLongTP:.5f}, KZ:{self.currentActiveKZ_name}")
                        self.buy()
                        self.lines.active_sl[0] = self.activeLongSL
                        self.lines.active_tp[0] = self.activeLongTP
                        self.bullBreakoutDetected = True
                    else: 
                        self.log(f"Bull Breakout SKIPPED. Filter:{self.bullFilterCondition}, KZ:{killZoneFilterActive}({self.currentActiveKZ_name})")
                    self.bullDone = True
                    self.bullActive = False
                elif self.d_close[0] < self.bullLowFractalLevel: 
                    self.bullDone = True
                    self.bullActive = False
            
            if self.bearActive and self.bearBoxCreated and not self.bearDone and not self.bearBreakoutDetected:
                if self.d_close[0] < self.bearLowFractalLevel:
                    if self.bearFilterCondition and killZoneFilterActive: 
                        self.activeShortSL = self.bearHighFractalLevel
                        self.activeShortTP = self.bearLowFractalLevel - (self.activeShortSL - self.bearLowFractalLevel) * self.p.rr_ratio
                        self.log(f"SELL SIGNAL: SL={self.activeShortSL:.5f}, TP={self.activeShortTP:.5f}, KZ:{self.currentActiveKZ_name}")
                        self.sell()
                        self.lines.active_sl[0] = self.activeShortSL
                        self.lines.active_tp[0] = self.activeShortTP
                        self.bearBreakoutDetected = True
                    else: 
                        self.log(f"Bear Breakout SKIPPED. Filter:{self.bearFilterCondition}, KZ:{killZoneFilterActive}({self.currentActiveKZ_name})")
                    self.bearDone = True
                    self.bearActive = False
                elif self.d_close[0] > self.bearHighFractalLevel: 
                    self.bearDone = True
                    self.bearActive = False
        
        if self.position.size > 0: 
            if self.p.use_stop_loss and not pd.isna(self.activeLongSL) and self.d_low[0] <= self.activeLongSL: self.log(f"Long SL Hit: Low {self.d_low[0]:.5f} <= SL {self.activeLongSL:.5f}"); self.close()
            elif self.p.use_take_profit and not pd.isna(self.activeLongTP) and self.d_high[0] >= self.activeLongTP: self.log(f"Long TP Hit: High {self.d_high[0]:.5f} >= TP {self.activeLongTP:.5f}"); self.close()
        elif self.position.size < 0:
            if self.p.use_stop_loss and not pd.isna(self.activeShortSL) and self.d_high[0] >= self.activeShortSL: self.log(f"Short SL Hit: High {self.d_high[0]:.5f} >= SL {self.activeShortSL:.5f}"); self.close()
            elif self.p.use_take_profit and not pd.isna(self.activeShortTP) and self.d_low[0] <= self.activeShortTP: self.log(f"Short TP Hit: Low {self.d_low[0]:.5f} <= TP {self.activeShortTP:.5f}"); self.close()

        if self.p.show_trend_table: # Log MTF summary if enabled
            log_trend_F1 = "N/A" if not (self.p.filter_use_tf1 and self.filter_tf1_data) else ("Up" if self.inUptrendF1 else "Down" if self.inDowntrendF1 else "Ntrl")
            log_trend_F2 = "N/A" if not (self.p.filter_use_tf2 and self.filter_tf2_data) else ("Up" if self.inUptrendF2 else "Down" if self.inDowntrendF2 else "Ntrl")
            log_trend_F3 = "N/A" if not (self.p.filter_use_tf3 and self.filter_tf3_data) else ("Up" if self.inUptrendF3 else "Down" if self.inDowntrendF3 else "Ntrl")
            active_filters_str_parts = []
            if self.p.filter_use_tf1 and self.filter_tf1_data: active_filters_str_parts.append(f"{self.p.filter_tf1_str}:{log_trend_F1}")
            if self.p.filter_use_tf2 and self.filter_tf2_data: active_filters_str_parts.append(f"{self.p.filter_tf2_str}:{log_trend_F2}")
            if self.p.filter_use_tf3 and self.filter_tf3_data: active_filters_str_parts.append(f"{self.p.filter_tf3_str}:{log_trend_F3}")
            active_filters_summary = ", ".join(active_filters_str_parts) if active_filters_str_parts else "None Active"
            # Simplified log compared to Pine, focusing on core info
            self.log(f"--- Bar {current_bar_idx} End --- Trends: {active_filters_summary} --- KZ: {self.currentActiveKZ_name} ({('Active' if killZoneFilterActive else 'Inactive') if self.p.filter_by_kill_zone else 'Disabled'}) ---")


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # --- Data Loading Setup ---
    # This section is now configured to load EURUSD 1M data by default,
    # assuming you run the download_forex_data.py script (modified for 1M) first.
    default_forex_pair_data = "EURUSD_data_1M.csv" # Expected filename from download_forex_data.py with 1M interval
    datapath = default_forex_pair_data 

    # Parameters for GenericCSVData based on 1M data from download_forex_data.py
    # The download script formats datetime as '%Y-%m-%d %H:%M:%S'
    # For 1M data, timeframe is Minutes and compression is 1.
    csv_params = {
        'dataname': datapath,
        'dtformat': ('%Y-%m-%d %H:%M:%S'),
        'datetime': 0, 'open': 1, 'high': 2, 'low': 3, 'close': 4, 'volume': 5,
        'timeframe': bt.TimeFrame.Minutes, # For 1M data
        'compression': 1,                # 1 minute
        'openinterest': -1
    }
    
    # If you download daily data (e.g., interval="1d"), you would change timeframe and compression:
    # csv_params_daily = {
    #     'dataname': "EURUSD_data_1D.csv", # Adjust filename if needed
    #     'dtformat': ('%Y-%m-%d %H:%M:%S'), # download_forex_data.py also adds 00:00:00 for daily
    #     'timeframe': bt.TimeFrame.Days,
    #     'compression': 1,
    #     # ... other params same ...
    # }
    # And then use: data0 = bt.feeds.GenericCSVData(**csv_params_daily)

    try:
        data0 = bt.feeds.GenericCSVData(**csv_params)
        cerebro.adddata(data0)
        data_feeds_added = {"main": data0}
        print(f"Successfully loaded data from: {datapath}")
    except Exception as e:
        print(f"CRITICAL ERROR loading main data feed from '{datapath}': {e}")
        print("Please ensure you have run 'download_forex_data.py' first, or that the file exists and is correctly formatted.")
        print("If you used different settings in download_forex_data.py, adjust 'datapath' and 'csv_params' here accordingly.")
        exit()

    # --- Strategy Parameters ---
    # These are the default parameters from your PineScript.
    # You can override them here before running the backtest if desired.
    strategy_params = {
        'htf_timeframe_str': '60',      # 60 minutes (1H). Base data is 1M. HTF is 60M.
                                        # Change to '240' for 4H pivots, 'D' for daily pivots on HTF, etc.
        'cooldown_bars': 10,
        'pivot_len': 10,                # Réduit de 20 à 10
        'htf_only_breakouts': True,
        'show_fractals': True,
        'fractal_periods': 2,
        'filter_lookback': 5,
        'filter_pivot_type': "High/Low",
        'filter_tf1_str': '240',     # 240 minutes (4H). Base data is 1M. Filter TF1 is 240M.
        'filter_use_tf1': False,      # Désactivé pour simplifier
        'filter_tf2_str': 'D',       # Daily. Base data is 1M. Filter TF2 is Daily.
        'filter_use_tf2': False,      # Désactivé pour simplifier
        'filter_tf3_str': 'W',       # Weekly. Base data is 1M. Filter TF3 is Weekly.
        'filter_use_tf3': False,     # Désactivé
        'show_trend_table': True,
        'filter_by_kill_zone': False, # Désactivé pour simplifier
        'show_asia_kz_filter': False, 
        'asia_kz_time': '2300-0355',
        'show_london_kz_filter': False,
        'london_kz_time': '0700-0955',
        'show_ny_am_kz_filter': False,
        'ny_am_kz_time': '1430-1655',
        'show_ny_pm_kz_filter': False,
        'ny_pm_kz_time': '1930-2055',
        'rr_ratio': 2.0,
        'use_stop_loss': True,
        'use_take_profit': True,
    }

    # --- Resampling Data Feeds for HTF and MTF Filters ---
    # The order of adding resampled data matters for self.datas[idx] in strategy
    print("Resampling data feeds based on strategy parameters...")
    if strategy_params.get('htf_timeframe_str'):
        htf_tf, htf_comp = parse_tf_string(strategy_params['htf_timeframe_str'])
        # Avoid resampling if HTF is same as base data (e.g. base is 1H, htf_timeframe_str is '60')
        if not (csv_params['timeframe'] == htf_tf and csv_params['compression'] == htf_comp):
            cerebro.resampledata(data_feeds_added["main"], timeframe=htf_tf, compression=htf_comp, name="htf")
            print(f"Resampled main data to HTF: TF Name={strategy_params['htf_timeframe_str']}")
        else:
            # If HTF is same as base, strategy needs to access datas[0] for HTF indicators.
            # This is handled in strategy __init__ by checking if self.htf_data is self.datas[0]
            # The current logic assumes distinct datas objects. We might need to adjust __init__ or pass datas[0]
            # For now, if same, we simply don't add it to cerebro resample, strategy will use datas[0]
            # This actually simplifies things if htf_data == main_data
            # The current strategy structure assigns self.datas[1] etc. so this needs careful handling if we don't resample.
            # Easiest is to always resample, even if it's to the same timeframe, or adjust strategy init.
            # Let's resample to ensure self.datas[1] exists for htf if htf_timeframe_str is set.
            cerebro.resampledata(data_feeds_added["main"], timeframe=htf_tf, compression=htf_comp, name="htf")
            print(f"HTF is same as base data. Resampled as 'htf' for consistent data indexing in strategy.")


    active_filter_tfs = []
    if strategy_params.get('filter_tf1_str') and strategy_params.get('filter_use_tf1'):
        active_filter_tfs.append((strategy_params['filter_tf1_str'], "filter_tf1"))
    if strategy_params.get('filter_tf2_str') and strategy_params.get('filter_use_tf2'):
        active_filter_tfs.append((strategy_params['filter_tf2_str'], "filter_tf2"))
    if strategy_params.get('filter_tf3_str') and strategy_params.get('filter_use_tf3'):
        active_filter_tfs.append((strategy_params['filter_tf3_str'], "filter_tf3"))

    for tf_str, name in active_filter_tfs:
        tf, comp = parse_tf_string(tf_str)
        if not (csv_params['timeframe'] == tf and csv_params['compression'] == comp):
            cerebro.resampledata(data_feeds_added["main"], timeframe=tf, compression=comp, name=name)
            print(f"Resampled main data to {name}: TF Name={tf_str}")
        else:
            # Resample even if same TF to ensure consistent data indexing (self.datas[x]) in strategy
            cerebro.resampledata(data_feeds_added["main"], timeframe=tf, compression=comp, name=name)
            print(f"{name} is same as base data. Resampled as '{name}' for consistent data indexing.")
    print("Data resampling complete.")

    # Add strategy to Cerebro
    cerebro.addstrategy(SweepAndFractalStrategy, **strategy_params)

    # Brokerage Setup
    cerebro.broker.setcash(10000.0)
    # For Forex, spread is more common than commission. Commission can be 0 if spread is main cost.
    # cerebro.broker.setcommission(commission=0.00002) # Example: 0.002% (very small)
    # You can also try to simulate spread if your data doesn't include it:
    # cerebro.broker.setSpread(0.0001) # Example: 1 pip for EURUSD (0.0001)
    # For this example, let's use a small commission, assuming ECN-like broker.
    cerebro.broker.setcommission(commission=0.00002, margin=None, mult=1.0) # 0.002% commission

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days) # Correct class name is SharpeRatio, not Sharpe
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # --- Run the Backtest ---
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    # --- Print Analysis Results ---
    print('--- Analyzers Results ---')
    trade_analysis = strat.analyzers.trade.get_analysis()
    if trade_analysis:
        print(f"Total Trades: {trade_analysis.total.total}")
        print(f"Winning Trades: {trade_analysis.won.total}")
        print(f"Losing Trades: {trade_analysis.lost.total}")
        if trade_analysis.won.total > 0: print(f"Avg Win (pnl): {trade_analysis.won.pnl.average:.2f}")
        if trade_analysis.lost.total > 0: print(f"Avg Loss (pnl): {trade_analysis.lost.pnl.average:.2f}")
        # Add more details as needed from trade_analysis

    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    if sharpe_analysis and 'sharperatio' in sharpe_analysis: print(f"Sharpe Ratio: {sharpe_analysis['sharperatio']:.3f}")
    else: print("Sharpe Ratio: N/A")
    
    sqn_analysis = strat.analyzers.sqn.get_analysis()
    if sqn_analysis and 'sqn' in sqn_analysis: print(f"SQN: {sqn_analysis['sqn']:.2f}")
    else: print("SQN: N/A")

    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    if drawdown_analysis and hasattr(drawdown_analysis.max, 'drawdown'): 
        print(f"Max DrawDown: {drawdown_analysis.max.drawdown:.2f}% / Money: {drawdown_analysis.max.moneydown:.2f}")
    else: print("Max DrawDown: N/A")
    
    # --- Plotting (Optional) ---
    # Make sure you have matplotlib installed: pip install matplotlib
    # Then uncomment the line below:
    # cerebro.plot(style='candlestick', barup='green', bardown='red', volume=True) 