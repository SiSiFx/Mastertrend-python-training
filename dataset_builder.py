#!/usr/bin/env python3
import pandas as pd
import numpy as np
from collections import deque
import math

# === Feature computation functions ===

def compute_atr(df, length=14):
    # True range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['tr'] = df[['tr1','tr2','tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(length).mean()
    return df


def compute_supertrend(df, atr_len=14, multiplier=3.0):
    src = (df['high'] + df['low']) / 2
    df['basic_up'] = src - multiplier * df['atr']
    df['basic_dn'] = src + multiplier * df['atr']
    fUp = [np.nan] * len(df)
    fDn = [np.nan] * len(df)
    trend = [1] * len(df)
    for i in range(1, len(df)):
        fUp[i] = max(df['basic_up'].iloc[i], fUp[i-1]) if df['close'].iloc[i-1] > fUp[i-1] else df['basic_up'].iloc[i]
        fDn[i] = min(df['basic_dn'].iloc[i], fDn[i-1]) if df['close'].iloc[i-1] < fDn[i-1] else df['basic_dn'].iloc[i]
        prev_trend = trend[i-1]
        if prev_trend == -1 and df['close'].iloc[i] > fDn[i-1]:
            trend[i] = 1
        elif prev_trend == 1 and df['close'].iloc[i] < fUp[i-1]:
            trend[i] = -1
        else:
            trend[i] = prev_trend
    df['supertrend'] = trend
    return df


def compute_jma(series, length=7, phase=50, power=2.0):
    # Jurik Moving Average
    phase_ratio = phase/100 + 1.5
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** power
    e0 = 0.0
    e1 = 0.0
    e2 = 0.0
    jma = []
    prev_jma = 0.0
    for v in series:
        e0 = (1 - alpha) * v + alpha * e0
        e1 = (v - e0) * (1 - beta) + beta * e1
        e2 = (e0 + phase_ratio * e1 - prev_jma) * (1 - alpha)**2 + alpha**2 * e2
        curr = e2 + prev_jma
        jma.append(curr)
        prev_jma = curr
    return pd.Series(jma, index=series.index)


def compute_jmacd(df, fast=12, slow=26, sig=9):
    df['jma_fast'] = compute_jma(df['hlc3'], fast)
    df['jma_slow'] = compute_jma(df['hlc3'], slow)
    df['jmacd'] = df['jma_fast'] - df['jma_slow']
    df['jmacd_signal'] = df['jmacd'].ewm(span=sig, adjust=False).mean()
    df['jmacd_histo'] = df['jmacd'] - df['jmacd_signal']
    return df


def compute_p1(df, e31=5, m=9, l31=14):
    # P1 indicator from spreadv + cum sum
    df['spreadv'] = (df['close'] - df['open']) * 100 * df['close']
    df['cumsum_spreadv'] = df['spreadv'].cumsum()
    df['pt'] = df['spreadv'] + df['cumsum_spreadv']
    df['ema_l31'] = df['pt'].ewm(span=l31, adjust=False).mean()
    df['ema_m'] = df['pt'].ewm(span=m, adjust=False).mean()
    df['ema_e31'] = df['pt'].ewm(span=e31, adjust=False).mean()
    df['a1'] = df['ema_l31'] - df['ema_m']
    df['b1'] = df['ema_e31'] - df['ema_m']
    df['p1'] = df['a1'] + df['b1']
    return df


def compute_stop_gap(df, hl_period=4, med_period=10):
    # Stop-gap median filter
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['trend_up'] = df['ema10'] > df['ema20']
    df['trend_down'] = df['ema10'] < df['ema20']
    df['highest_hl'] = df['high'].rolling(hl_period).max()
    df['lowest_hl'] = df['low'].rolling(hl_period).min()
    df['stop_gap'] = np.where(
        (df['trend_down'] & (df['high'] < df['highest_hl'])),
        df['highest_hl'] - df['low'],
        np.where(
            (df['trend_up'] & (df['low'] > df['lowest_hl'])),
            df['high'] - df['lowest_hl'],
            0.0
        )
    )
    df['median_hl'] = (df['high'] - df['low']).rolling(med_period).median()
    df['stop_gap_pass'] = df['stop_gap'] > df['median_hl']
    return df


def build_dataset(csv_file="EURUSD_data_1M.csv", output_file="dataset.csv"):
    df = pd.read_csv(csv_file, parse_dates=[0], names=["datetime","open","high","low","close","volume"], header=0)
    df.set_index('datetime', inplace=True)
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    # ATR and SuperTrend
    df = compute_atr(df, 14)
    df = compute_supertrend(df, 14, 3.0)
    # Target flip
    df['prev_trend'] = df['supertrend'].shift(1)
    df['target'] = 0
    df.loc[(df['prev_trend']==-1)&(df['supertrend']==1), 'target'] = 1
    df.loc[(df['prev_trend']==1)&(df['supertrend']==-1), 'target'] = -1
    # Features
    df = compute_jmacd(df)
    df = compute_p1(df)
    df = compute_stop_gap(df)
    # Drop NaNs
    df.dropna(inplace=True)
    # Save dataset
    df.to_csv(output_file, index=True)
    print(f"Dataset saved to {output_file}, shape={df.shape}")

if __name__ == '__main__':
    build_dataset() 