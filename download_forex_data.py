import yfinance as yf
import pandas as pd
import datetime

# --- Parameters to configure ---
# Forex pairs on Yahoo Finance are often suffixed with "=X"
# Common pairs: "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"
ticker_symbol = "EURUSD=X"  # Target Forex pair
start_date = "2022-01-01"   # Desired start date for historical data
end_date = "2023-12-31"     # Desired end date for historical data (yfinance downloads up to, but not including, end_date for daily/intraday)
                            # To include 2023-12-31, you might set end_date to 2024-01-01 for daily. For intraday, check yfinance behavior.
interval = "1m"             # Data granularity: "1d" (daily), "1h" (hourly), "15m", "5m", "1m"
                            # Availability for intraday data (especially for long periods) can be limited.
                            # For "1m" data, yfinance typically provides only the last 7 days.
output_filename_prefix = ticker_symbol.replace("=X", "") # e.g., "EURUSD"
# --- End of parameters ---

print(f"Attempting to download data for {ticker_symbol} from {start_date} to {end_date} with interval {interval}...")

try:
    data = yf.download(tickers=ticker_symbol, start=start_date, end=end_date, interval=interval)

    if data.empty:
        print(f"No data received for {ticker_symbol}. Reasons could be:\n"
              f"- Incorrect ticker symbol (should it have '=X'?).\n"
              f"- No data available for the specified period or interval.\n"
              f"- yfinance limitations (e.g., 1m data only for last 7 days, intraday for specific date ranges limited to 60 days).")
    else:
        print("Data downloaded successfully. Processing...")

        # Ensure the index is datetime
        data.index = pd.to_datetime(data.index)

        # Rename columns to lowercase to match backtrader script expectations
        data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
            # 'Adj Close' is often present but not used by our script; can be ignored.
        }, inplace=True)

        # Prepare data for CSV: GenericCSVData expects a 'datetime' column.
        # If the index is already a DatetimeIndex (which yfinance usually provides), we can reset it.
        data_for_csv = data.reset_index()
        # yfinance might name the reset index column 'Datetime' or 'index'. Rename it to 'datetime'.
        first_col_name = data_for_csv.columns[0]
        if first_col_name.lower() in ['datetime', 'date', 'index', 'time']: # Check common names
             data_for_csv.rename(columns={first_col_name: 'datetime'}, inplace=True)
        else:
            print(f"Warning: The first column after reset_index was named '{first_col_name}'. Attempting to use it as datetime.")
            data_for_csv.rename(columns={first_col_name: 'datetime'}, inplace=True)


        # Format the 'datetime' column string for consistency, matching dtformat in backtrader script.
        # dtformat=('%Y-%m-%d %H:%M:%S')
        if interval == "1d":
            # For daily data, yfinance might not include time. Add 00:00:00.
            data_for_csv['datetime'] = pd.to_datetime(data_for_csv['datetime']).dt.strftime('%Y-%m-%d 00:00:00')
        else:
            # For intraday data, ensure timezone is removed (naive datetime) if present.
            if hasattr(data_for_csv['datetime'].dt, 'tz') and data_for_csv['datetime'].dt.tz is not None:
                data_for_csv['datetime'] = data_for_csv['datetime'].dt.tz_localize(None)
            data_for_csv['datetime'] = pd.to_datetime(data_for_csv['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Select and order the required columns
        columns_required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        missing_cols = [col for col in columns_required if col not in data_for_csv.columns]
        if missing_cols:
            print(f"Error: Missing required columns after processing: {missing_cols}")
            print(f"Available columns: {data_for_csv.columns.tolist()}")
        else:
            data_to_save = data_for_csv[columns_required]

            # Generate output filename
            # Replace 'm' with 'M' in interval for filename aesthetics if minutes, or use 'D' for daily, 'H' for hourly.
            interval_label = interval
            if 'm' in interval: interval_label = interval.replace('m', 'M')
            elif 'h' in interval: interval_label = interval.replace('h', 'H')
            elif 'd' in interval: interval_label = interval.upper()

            output_path = f"{output_filename_prefix}_data_{interval_label}.csv" # e.g., EURUSD_data_1H.csv

            data_to_save.to_csv(output_path, index=False)
            print(f"Forex data successfully saved to: {output_path}")
            print("--- First 5 rows of the saved CSV: ---")
            print(data_to_save.head())
            print("-----------------------------------------")
            print(f"Reminder: Update 'datapath' in your backtrader script to '{output_path}'.")
            print(f"And ensure 'timeframe' and 'compression' in GenericCSVData match the interval '{interval}'.")
            if interval == "1h":
                 print("For 1H data, use: timeframe=bt.TimeFrame.Minutes, compression=60")
            elif interval == "1d":
                 print("For 1D data, use: timeframe=bt.TimeFrame.Days, compression=1")
            elif "m" in interval:
                minutes = int(interval.replace("m",""))
                print(f"For {interval} data, use: timeframe=bt.TimeFrame.Minutes, compression={minutes}")


except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check your internet connection, ticker symbol, date range, and interval.")
    print("For intraday data with yfinance, very long date ranges (more than 60-90 days typically) might not be supported for intervals shorter than 1d.") 