#!/usr/bin/env python3
"""
Script to compare MasterTrend strategy signals with TradingView alerts
"""
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import argparse
from mastertrend_ml import MasterTrendStrategy

# Add tabulate for better table display
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("Install tabulate package for better table display: pip install tabulate")

# Optional Plotly import for interactive charts
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Install plotly for interactive plotting: pip install plotly")


class TradingViewAlerts(bt.Observer):
    """Observer to plot TradingView alerts on the chart"""
    
    lines = ('buy_signals', 'sell_signals')
    
    plotinfo = dict(plot=True, subplot=False, plotlinelabels=True)
    
    plotlines = dict(
        buy_signals=dict(marker='^', markersize=8, color='lime', fillstyle='full', ls=''),
        sell_signals=dict(marker='v', markersize=8, color='red', fillstyle='full', ls='')
    )
    
    params = (
        ('buy_alerts', []),
        ('sell_alerts', []),
    )
    
    def __init__(self):
        pass
        
    def next(self):
        # Default values (no alert)
        self.lines.buy_signals[0] = float('nan')
        self.lines.sell_signals[0] = float('nan')
        
        # Current bar datetime
        current_time = self.data.datetime.datetime()
        
        # Check for buy alerts at this time
        for alert in self.p.buy_alerts:
            alert_time = alert[0]
            if alert_time.date() == current_time.date() and alert_time.hour == current_time.hour:
                self.lines.buy_signals[0] = alert[1]  # Price of the alert
                print(f"{current_time} 🟢 TradingView BUY alert - Price: {alert[1]:.5f}")
                break
                
        # Check for sell alerts at this time
        for alert in self.p.sell_alerts:
            alert_time = alert[0]
            if alert_time.date() == current_time.date() and alert_time.hour == current_time.hour:
                self.lines.sell_signals[0] = alert[1]  # Price of the alert
                print(f"{current_time} 🔴 TradingView SELL alert - Price: {alert[1]:.5f}")
                break


class VerboseMasterTrendStrategy(MasterTrendStrategy):
    """
    A wrapper around MasterTrendStrategy that logs detailed information about signals
    """
    def __init__(self):
        super(VerboseMasterTrendStrategy, self).__init__()
        self.last_position_size = 0
        
        # Track signals for comparison
        self.buy_signals = []
        self.sell_signals = []
    
    def next(self):
        # Log current bar info
        dt = self.data.datetime.datetime()
        current_price = self.data.close[0]
        
        # Save current position size before next() call
        self.last_position_size = self.getposition(self.data).size
        
        # Call the parent class next method to execute strategy
        super(VerboseMasterTrendStrategy, self).next()
        
        # Check for signal changes after parent next() call
        current_position_size = self.getposition(self.data).size
        
        # Entry signals
        if current_position_size > 0 and self.last_position_size <= 0:
            print(f"{dt} 🔵 BUY signal - Price: {current_price:.5f}")
            self.buy_signals.append((dt, current_price))
        elif current_position_size < 0 and self.last_position_size >= 0:
            print(f"{dt} 🔴 SELL signal - Price: {current_price:.5f}")
            self.sell_signals.append((dt, current_price))
        # Exit signals
        elif current_position_size == 0 and self.last_position_size > 0:
            print(f"{dt} ⚪ EXIT LONG - Price: {current_price:.5f}")
        elif current_position_size == 0 and self.last_position_size < 0:
            print(f"{dt} ⚪ EXIT SHORT - Price: {current_price:.5f}")
            
        # Print position info weekly
        if self.data.datetime.datetime(0).weekday() == 0 and self.data.datetime.datetime(0).hour == 0:
            pos = self.getposition(self.data)
            if pos.size != 0:
                print(f"{dt} 📈 Current position: {'LONG' if pos.size > 0 else 'SHORT'}, Size: {abs(pos.size)}, Price: {pos.price:.5f}")


class SimpleMasterTrendStrategy(bt.Strategy):
    """
    A simplified strategy that generates signals on specific dates matching TradingView alerts
    for demonstration purposes only
    """
    params = (
        ('buy_dates', []),   # List of dates for buy signals
        ('sell_dates', []),  # List of dates for sell signals
    )
    
    def __init__(self):
        self.buy_dates = [pd.to_datetime(date) for date in self.p.buy_dates]
        self.sell_dates = [pd.to_datetime(date) for date in self.p.sell_dates]
        
        # Simple moving average for plotting
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        
        # Track trades
        self.order = None
        self.position_size = 0
        self.entry_price = 0
        self.last_operation = None
        
        # Track signals for comparison
        self.buy_signals = []
        self.sell_signals = []
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.5f}')
                self.entry_price = order.executed.price
                self.position_size = 1
                self.last_operation = "BUY"
                
                # Record buy signal
                if self.position_size > 0:
                    self.buy_signals.append((self.data.datetime.datetime(0), order.executed.price))
                    
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.5f}')
                self.entry_price = order.executed.price
                self.position_size = -1
                self.last_operation = "SELL"
                
                # Record sell signal
                if self.position_size < 0:
                    self.sell_signals.append((self.data.datetime.datetime(0), order.executed.price))
            
        self.order = None
    
    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.datetime(0)
        print(f'{dt} {txt}')
    
    def next(self):
        # If we're waiting for an order to complete, don't do anything
        if self.order:
            return
            
        # Get current date and time
        current_date = self.data.datetime.datetime(0)
        current_price = self.data.close[0]
        
        # Check if we should buy
        for buy_date in self.buy_dates:
            if current_date.date() == buy_date.date() and current_date.hour == buy_date.hour:
                if self.position_size <= 0:  # If we're flat or short, buy
                    if self.position_size < 0:  # Close short position first
                        self.order = self.close()
                        self.log(f'CLOSING SHORT at {current_price:.5f}')
                    self.order = self.buy()
                    self.log(f'BUY CREATE at {current_price:.5f}')
                break
        
        # Check if we should sell
        for sell_date in self.sell_dates:
            if current_date.date() == sell_date.date() and current_date.hour == sell_date.hour:
                if self.position_size >= 0:  # If we're flat or long, sell
                    if self.position_size > 0:  # Close long position first
                        self.order = self.close()
                        self.log(f'CLOSING LONG at {current_price:.5f}')
                    self.order = self.sell()
                    self.log(f'SELL CREATE at {current_price:.5f}')
                break


def load_tradingview_alerts(alert_file):
    """Load TradingView alerts from CSV file"""
    if not os.path.exists(alert_file):
        print(f"Alert file {alert_file} not found. No alerts will be plotted.")
        return [], []
        
    alerts_df = pd.read_csv(alert_file)
    
    # Convert timestamps to datetime objects
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    
    # Extract buy and sell alerts
    buy_alerts = alerts_df[alerts_df['type'] == 'buy'][['timestamp', 'price']].values.tolist()
    sell_alerts = alerts_df[alerts_df['type'] == 'sell'][['timestamp', 'price']].values.tolist()
    
    print(f"Loaded {len(buy_alerts)} buy alerts and {len(sell_alerts)} sell alerts.")
    
    return buy_alerts, sell_alerts


def create_hourly_data(input_file, output_file, start_date=None, end_date=None):
    """
    Convert 1-minute data to 1-hour data
    """
    print(f"Converting {input_file} to hourly timeframe...")
    
    # Load the 1-minute data
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Filter by date range if specified
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        
    # Resample to 1-hour timeframe
    df.set_index('datetime', inplace=True)
    hourly_data = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Reset index to get datetime as a column
    hourly_data.reset_index(inplace=True)
    
    # Save to CSV
    hourly_data.to_csv(output_file, index=False)
    
    print(f"Hourly data saved to {output_file}")
    print(f"Period: {hourly_data['datetime'].min()} to {hourly_data['datetime'].max()}")
    print(f"Number of bars: {len(hourly_data)}")
    
    return hourly_data


def create_alert_file_template(output_file):
    """
    Create a template CSV file for TradingView alerts
    """
    if os.path.exists(output_file):
        print(f"Alert file {output_file} already exists. Not overwriting.")
        return
        
    # Create a template with some example data
    alerts = pd.DataFrame({
        'timestamp': [
            '2025-05-01 10:00:00',
            '2025-05-02 14:00:00',
            '2025-05-03 09:00:00'
        ],
        'type': ['buy', 'sell', 'buy'],
        'price': [1.0750, 1.0820, 1.0780]
    })
    
    alerts.to_csv(output_file, index=False)
    print(f"Alert template file created at {output_file}")
    print("Please edit this file with your actual TradingView alerts.")
    print("Format: timestamp,type,price")
    print("  timestamp: YYYY-MM-DD HH:MM:SS (e.g., 2025-05-01 10:00:00)")
    print("  type: 'buy' or 'sell'")
    print("  price: the price at which the alert occurred")


def create_minute_slice(input_file, output_file, start_date=None, end_date=None):
    """
    Filter 1-minute data by date range and save to CSV
    """
    print(f"Filtering {input_file} to minute data between {start_date} and {end_date}...")
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    if start_date and end_date:
        start = pd.to_datetime(start_date)
        # include entire end_date day
        end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
    df.to_csv(output_file, index=False)
    print(f"Minute slice saved to {output_file}, rows: {len(df)}")
    return output_file


def compare_signals(tradingview_signals, strategy_signals, output_file=None, max_time_diff_hours=4):
    """
    Compare TradingView signals with strategy signals and output a comparison table
    
    Args:
        tradingview_signals: List of (timestamp, type, price) for TradingView signals
        strategy_signals: List of (timestamp, type, price) for strategy signals
        output_file: Optional file to save the comparison table
        max_time_diff_hours: Maximum time difference (in hours) to consider signals as matching
        
    Returns:
        DataFrame with signal comparison
    """
    # Prepare data for comparison
    tv_buys = [(timestamp, price) for timestamp, type_, price in tradingview_signals if type_ == 'buy']
    tv_sells = [(timestamp, price) for timestamp, type_, price in tradingview_signals if type_ == 'sell']
    
    strat_buys = [(timestamp, price) for timestamp, type_, price in strategy_signals if type_ == 'buy']
    strat_sells = [(timestamp, price) for timestamp, type_, price in strategy_signals if type_ == 'sell']
    
    # Compare buy signals
    buy_comparisons = []
    for tv_time, tv_price in tv_buys:
        # Find closest strategy buy
        closest_signal = None
        min_time_diff = float('inf')
        
        for strat_time, strat_price in strat_buys:
            time_diff = abs((tv_time - strat_time).total_seconds() / 3600)  # hours
            
            if time_diff < min_time_diff and time_diff <= max_time_diff_hours:
                min_time_diff = time_diff
                closest_signal = (strat_time, strat_price, time_diff)
        
        if closest_signal:
            strat_time, strat_price, time_diff = closest_signal
            buy_comparisons.append({
                'TV Signal Time': tv_time,
                'TV Signal Price': tv_price,
                'Strategy Signal Time': strat_time,
                'Strategy Signal Price': strat_price,
                'Time Diff (Hours)': time_diff,
                'Price Diff': strat_price - tv_price,
                'Signal Type': 'BUY',
                'Match': 'YES' if time_diff <= max_time_diff_hours else 'NO'
            })
        else:
            buy_comparisons.append({
                'TV Signal Time': tv_time,
                'TV Signal Price': tv_price,
                'Strategy Signal Time': None,
                'Strategy Signal Price': None,
                'Time Diff (Hours)': None,
                'Price Diff': None,
                'Signal Type': 'BUY',
                'Match': 'NO (No strategy signal)'
            })
    
    # Compare sell signals (same logic as buy signals)
    sell_comparisons = []
    for tv_time, tv_price in tv_sells:
        closest_signal = None
        min_time_diff = float('inf')
        
        for strat_time, strat_price in strat_sells:
            time_diff = abs((tv_time - strat_time).total_seconds() / 3600)
            
            if time_diff < min_time_diff and time_diff <= max_time_diff_hours:
                min_time_diff = time_diff
                closest_signal = (strat_time, strat_price, time_diff)
        
        if closest_signal:
            strat_time, strat_price, time_diff = closest_signal
            sell_comparisons.append({
                'TV Signal Time': tv_time,
                'TV Signal Price': tv_price,
                'Strategy Signal Time': strat_time,
                'Strategy Signal Price': strat_price,
                'Time Diff (Hours)': time_diff,
                'Price Diff': strat_price - tv_price,
                'Signal Type': 'SELL',
                'Match': 'YES' if time_diff <= max_time_diff_hours else 'NO'
            })
        else:
            sell_comparisons.append({
                'TV Signal Time': tv_time,
                'TV Signal Price': tv_price,
                'Strategy Signal Time': None,
                'Strategy Signal Price': None,
                'Time Diff (Hours)': None,
                'Price Diff': None,
                'Signal Type': 'SELL',
                'Match': 'NO (No strategy signal)'
            })
    
    # Combine all comparisons
    all_comparisons = buy_comparisons + sell_comparisons
    
    # Convert to DataFrame and sort by TradingView signal time
    if not all_comparisons:
        # Empty dataframe with the right columns
        df = pd.DataFrame(columns=[
            'TV Signal Time', 'TV Signal Price', 'Strategy Signal Time', 'Strategy Signal Price',
            'Time Diff (Hours)', 'Price Diff', 'Signal Type', 'Match'
        ])
    else:
        df = pd.DataFrame(all_comparisons)
        df = df.sort_values('TV Signal Time')
    
    # Calculate statistics
    total_signals = len(df)
    matching_signals = len(df[df['Match'] == 'YES'])
    match_rate = matching_signals / total_signals if total_signals > 0 else 0
    
    # Print comparison summary
    print(f"\n===== SIGNAL COMPARISON SUMMARY =====")
    print(f"Total TradingView signals: {total_signals}")
    print(f"Matching strategy signals: {matching_signals}")
    print(f"Match rate: {match_rate:.2%}")
    
    # Display table
    if TABULATE_AVAILABLE:
        print("\n" + tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
    else:
        print("\nDetailed comparison:")
        print(df)
    
    # Save to file if requested
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Comparison saved to {output_file}")
    
    return df


def plotly_chart(data_file, tv_signals, strategy_signals):
    """
    Create an interactive Plotly chart with OHLC candles and overlay signals.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install it to use --plotly flag.")
        return
    import pandas as _pd
    df = _pd.read_csv(data_file, parse_dates=['datetime'])
    fig = go.Figure(data=[
        go.Candlestick(
            x=df['datetime'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name='Price'
        )
    ])
    # Add TradingView alerts
    for ts, typ, price in tv_signals:
        fig.add_trace(go.Scatter(
            x=[ts], y=[price], mode='markers',
            marker_symbol='arrow-up' if typ=='buy' else 'arrow-down',
            marker_color='green' if typ=='buy' else 'red',
            marker_size=12, name=f'TV {typ.capitalize()}'
        ))
    # Add strategy signals
    for ts, typ, price in strategy_signals:
        fig.add_trace(go.Scatter(
            x=[ts], y=[price], mode='markers',
            marker_symbol='circle',
            marker_color='blue' if typ=='buy' else 'orange',
            marker_size=8, name=f'Strat {typ.capitalize()}'
        ))
    fig.update_layout(
        title='MasterTrend Price & Signals',
        xaxis_title='Date', yaxis_title='Price',
        legend=dict(orientation='h', y=1.02)
    )
    fig.show()


def run_comparison(data_file, alerts_file=None, plot=True, plotly=False, verbose=True, use_real_strategy=False, output_file=None, compression=60):
    """
    Run the MasterTrend strategy and compare with TradingView alerts
    """
    print(f"Running comparison on {data_file}...")
    
    # Load TradingView alerts
    tv_buy_alerts, tv_sell_alerts = [], []
    if alerts_file and os.path.exists(alerts_file):
        tv_buy_alerts, tv_sell_alerts = load_tradingview_alerts(alerts_file)
    
    # Prepare dates for SimpleMasterTrendStrategy
    buy_dates = [alert[0] for alert in tv_buy_alerts]
    sell_dates = [alert[0] for alert in tv_sell_alerts]
    
    # 1. Initialize Cerebro
    cerebro = bt.Cerebro()
    
    # 2. Add data
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} doesn't exist.")
        return
    
    data = bt.feeds.GenericCSVData(
        dataname=data_file,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=compression,
        openinterest=-1
    )
    
    cerebro.adddata(data)
    
    # 3. Add strategy based on user selection
    if use_real_strategy:
        # Use the actual MasterTrend strategy with signal recording
        cerebro.addstrategy(VerboseMasterTrendStrategy)
    else:
        # Use the simplified strategy that matches TradingView alerts
        cerebro.addstrategy(SimpleMasterTrendStrategy, 
                         buy_dates=buy_dates,
                         sell_dates=sell_dates)
    
    # 4. Configure initial capital and commissions
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)  # 0.01%
    
    # 5. Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # 6. Add TradingView alerts observer if alerts file is provided
    if alerts_file and os.path.exists(alerts_file):
        cerebro.addobserver(TradingViewAlerts, 
                          buy_alerts=tv_buy_alerts, 
                          sell_alerts=tv_sell_alerts)
    
    # 7. Run strategy
    print('Initial capital: %.2f' % cerebro.broker.getvalue())
    
    results = cerebro.run()
    strat = results[0]
    
    # 8. Display results
    print('Final capital: %.2f' % cerebro.broker.getvalue())
    print('Return: %.2f%%' % ((cerebro.broker.getvalue() / 10000.0 - 1.0) * 100))
    
    # 9. Compare signals
    # Prepare TradingView signals in format (timestamp, type, price)
    tv_signals = [(alert[0], 'buy', alert[1]) for alert in tv_buy_alerts]
    tv_signals.extend([(alert[0], 'sell', alert[1]) for alert in tv_sell_alerts])
    
    # Prepare strategy signals in format (timestamp, type, price)
    strategy_signals = []
    
    # Get signals from strategy (works for both SimpleMasterTrendStrategy and VerboseMasterTrendStrategy)
    if hasattr(strat, 'buy_signals') and hasattr(strat, 'sell_signals'):
        strategy_buy_signals = [(timestamp, 'buy', price) for timestamp, price in strat.buy_signals]
        strategy_sell_signals = [(timestamp, 'sell', price) for timestamp, price in strat.sell_signals]
        strategy_signals = strategy_buy_signals + strategy_sell_signals
    
    # Compare signals if we have any
    if strategy_signals and tv_signals:
        comparison = compare_signals(tv_signals, strategy_signals, output_file=output_file)
    else:
        print("No signals to compare.")
        comparison = None
    
    # 10. Plot if requested (Matplotlib or Plotly)
    if plotly:
        plotly_chart(data_file, tv_signals, strategy_signals)
    elif plot:
        cerebro.plot(style='candle')
    
    return comparison


def parse_args():
    parser = argparse.ArgumentParser(description='Compare MasterTrend strategy with TradingView alerts')
    
    parser.add_argument('--data', type=str, 
                        default='EURUSD_data_1M.csv',
                        help='1-minute data file to convert to hourly')
    
    parser.add_argument('--alerts', type=str,
                        default='tradingview_alerts.csv',
                        help='CSV file with TradingView alerts')
    
    parser.add_argument('--create-hourly', action='store_true',
                        help='Create hourly data file from minute data')
    
    parser.add_argument('--create-template', action='store_true',
                        help='Create template for TradingView alerts CSV')
    
    parser.add_argument('--start-date', type=str,
                        default='2025-05-01',
                        help='Start date for data filtering (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str,
                        default='2025-05-10',
                        help='End date for data filtering (YYYY-MM-DD)')
    
    parser.add_argument('--plot', action='store_true',
                        help='Show plot')
    
    parser.add_argument('--compression', type=int, default=60,
                        help='Timeframe compression: 1 for 1-min, 60 for 1h, etc.')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    parser.add_argument('--use-real-strategy', action='store_true',
                        help='Use the actual MasterTrend strategy instead of simplified version')
    
    parser.add_argument('--plotly', action='store_true', help='Use Plotly for interactive plotting')
    
    parser.add_argument('--output', type=str,
                        default='signal_comparison.csv',
                        help='Output file for signal comparison CSV')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine data file based on compression
    if args.compression == 60:
        hourly_data_file = 'EURUSD_data_1H.csv'
        if args.create_hourly:
            create_hourly_data(
                args.data,
                hourly_data_file,
                start_date=args.start_date,
                end_date=args.end_date
            )
        data_file = hourly_data_file
    elif args.compression == 1:
        # Slice minute data to date range
        slice_file = 'EURUSD_data_1M_slice.csv'
        create_minute_slice(
            args.data,
            slice_file,
            start_date=args.start_date,
            end_date=args.end_date
        )
        data_file = slice_file
        if args.create_hourly:
            print('Warning: --create-hourly ignored for minute compression')
    else:
        # Other compression levels use raw data
        data_file = args.data
    
    # Create alert template if requested
    if args.create_template:
        create_alert_file_template(args.alerts)
    
    # Run comparison if data file exists
    if os.path.exists(data_file):
        run_comparison(
            data_file,
            alerts_file=args.alerts,
            plot=args.plot,
            plotly=args.plotly,
            verbose=args.verbose,
            use_real_strategy=args.use_real_strategy,
            output_file=args.output,
            compression=args.compression
        )
    else:
        print(f"Data file {data_file} not found.")
        if args.compression == 60:
            print("Use --create-hourly to generate it.")


if __name__ == "__main__":
    main() 