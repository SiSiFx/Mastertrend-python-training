#!/usr/bin/env python3
"""
CONSERVATIVE PROP FIRM TRADING SYSTEM
=====================================

Ultra-conservative trading system designed to meet strict prop firm requirements.
Focus on capital preservation with modest but consistent returns.

Key Features:
- Ultra-low risk per trade (0.25% max)
- Strict drawdown controls
- Conservative position sizing
- Multiple safety mechanisms
- Realistic trading costs

Author: AI Trading Assistant
Version: 2.0 Conservative
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ConservativePropFirmSystem:
    def __init__(self):
        """Initialize the conservative prop firm trading system"""
        self.api_key = "3240e71f449497ac70a3abce8989ef0a-631d288e0d87be637ba3a3a41ff998fd"
        self.data_dir = "automated_forex_data"
        
        # Ultra-conservative prop firm parameters
        self.initial_balance = 100000  # $100k challenge
        self.profit_target = 0.08      # 8% profit target
        self.max_drawdown = 0.04       # 4% max drawdown (stricter than 5%)
        self.daily_loss_limit = 0.03   # 3% daily loss limit (stricter than 5%)
        
        # Ultra-conservative trading parameters
        self.risk_per_trade = 0.0025   # 0.25% risk per trade (very conservative)
        self.max_positions = 2         # Maximum 2 concurrent positions
        self.min_win_rate = 0.45       # Minimum 45% win rate required
        
        # Trading costs (realistic)
        self.spreads = {
            'EUR_USD': 0.8, 'GBP_USD': 1.2, 'USD_JPY': 0.9,
            'AUD_USD': 1.1, 'EUR_JPY': 1.4, 'GBP_JPY': 1.8
        }
        self.commission_per_lot = 3.50  # $3.50 per standard lot round trip
        self.slippage_pips = 0.3       # 0.3 pips average slippage
        
        # Results storage
        self.results = {}
        
    def load_data(self, pair, timeframe):
        """Load market data for a currency pair and timeframe"""
        try:
            filename = f"{pair}_{timeframe}.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"❌ Data file not found: {filepath}")
                return None
                
            df = pd.read_csv(filepath)
            
            # Ensure proper column names
            if 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['time'])
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            else:
                print(f"❌ No time column found in {filename}")
                return None
                
            # Ensure OHLC columns exist
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                print(f"❌ Missing OHLC columns in {filename}")
                return None
                
            df = df.sort_values('datetime').reset_index(drop=True)
            print(f"✅ Loaded {len(df)} bars for {pair} {timeframe}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data for {pair} {timeframe}: {e}")
            return None
    
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def conservative_trend_strategy(self, df):
        """Ultra-conservative trend following strategy"""
        try:
            # Calculate indicators with longer periods for stability
            df['ema_fast'] = self.calculate_ema(df['close'], 21)  # Longer periods
            df['ema_slow'] = self.calculate_ema(df['close'], 55)
            df['ema_filter'] = self.calculate_ema(df['close'], 100)  # Long-term filter
            df['rsi'] = self.calculate_rsi(df['close'], 21)  # Longer RSI
            
            # Calculate MACD for confirmation
            macd_line, signal_line, histogram = self.calculate_macd(df['close'], 12, 26, 9)
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            
            # Generate ultra-conservative signals
            df['signal'] = 0
            
            # Buy signal: Multiple confirmations required
            buy_condition = (
                (df['ema_fast'] > df['ema_slow']) &  # Fast EMA above slow
                (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) &  # Crossover
                (df['close'] > df['ema_filter']) &  # Above long-term trend
                (df['rsi'] > 40) & (df['rsi'] < 65) &  # RSI in safe zone
                (df['macd'] > df['macd_signal']) &  # MACD confirmation
                (df['macd'] > 0)  # MACD above zero line
            )
            
            # Sell signal: Multiple confirmations required
            sell_condition = (
                (df['ema_fast'] < df['ema_slow']) &  # Fast EMA below slow
                (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)) &  # Crossover
                (df['close'] < df['ema_filter']) &  # Below long-term trend
                (df['rsi'] > 35) & (df['rsi'] < 60) &  # RSI in safe zone
                (df['macd'] < df['macd_signal']) &  # MACD confirmation
                (df['macd'] < 0)  # MACD below zero line
            )
            
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
            return df
            
        except Exception as e:
            print(f"❌ Error in conservative trend strategy: {e}")
            return df
    
    def conservative_mean_reversion_strategy(self, df):
        """Ultra-conservative mean reversion strategy"""
        try:
            # Calculate indicators
            df['rsi'] = self.calculate_rsi(df['close'], 14)
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(df['close'], 20, 2.5)  # Wider bands
            
            df['bb_upper'] = upper_band
            df['bb_middle'] = middle_band
            df['bb_lower'] = lower_band
            
            # Calculate additional filters
            df['ema_200'] = self.calculate_ema(df['close'], 200)  # Long-term trend filter
            
            # Generate ultra-conservative signals
            df['signal'] = 0
            
            # Buy signal: Extreme oversold with trend confirmation
            buy_condition = (
                (df['rsi'] < 25) &  # Very oversold
                (df['close'] <= df['bb_lower']) &  # Touch lower band
                (df['close'].shift(1) > df['bb_lower'].shift(1)) &  # First touch
                (df['close'] > df['ema_200']) &  # Above long-term trend
                (df['rsi'].shift(1) >= 25)  # RSI was not oversold before
            )
            
            # Sell signal: Extreme overbought with trend confirmation
            sell_condition = (
                (df['rsi'] > 75) &  # Very overbought
                (df['close'] >= df['bb_upper']) &  # Touch upper band
                (df['close'].shift(1) < df['bb_upper'].shift(1)) &  # First touch
                (df['close'] < df['ema_200']) &  # Below long-term trend
                (df['rsi'].shift(1) <= 75)  # RSI was not overbought before
            )
            
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
            return df
            
        except Exception as e:
            print(f"❌ Error in conservative mean reversion strategy: {e}")
            return df
    
    def calculate_conservative_position_size(self, balance, risk_per_trade, stop_loss_pips, pair):
        """Calculate ultra-conservative position size"""
        try:
            # Convert pips to price for position sizing
            if 'JPY' in pair:
                pip_value = 0.01  # For JPY pairs
            else:
                pip_value = 0.0001  # For other pairs
            
            # Risk amount in dollars
            risk_amount = balance * risk_per_trade
            
            # Position size calculation
            position_size = risk_amount / (stop_loss_pips * pip_value * 100000)  # 100k = 1 standard lot
            
            # Ultra-conservative limits
            position_size = min(position_size, 0.1)  # Max 0.1 lots (very small)
            position_size = max(position_size, 0.01)  # Minimum 0.01 lots
            
            return round(position_size, 2)
            
        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            return 0.01
    
    def backtest_conservative_strategy(self, df, pair, strategy_name):
        """Backtest with ultra-conservative risk management"""
        try:
            balance = self.initial_balance
            equity = balance
            peak_equity = balance
            max_drawdown = 0
            trades = []
            daily_returns = []
            consecutive_losses = 0
            max_consecutive_losses = 3  # Stop after 3 consecutive losses
            
            position = 0  # 0 = no position, 1 = long, -1 = short
            entry_price = 0
            entry_time = None
            stop_loss = 0
            take_profit = 0
            position_size = 0
            
            # Get spread and costs
            spread_pips = self.spreads.get(pair, 1.0)
            
            for i in range(1, len(df)):
                current_price = df.iloc[i]['close']
                current_time = df.iloc[i]['datetime']
                signal = df.iloc[i]['signal']
                
                # Check for exit conditions first
                if position != 0:
                    exit_trade = False
                    exit_price = current_price
                    exit_reason = ""
                    
                    # Check stop loss and take profit
                    if position == 1:  # Long position
                        if current_price <= stop_loss:
                            exit_trade = True
                            exit_price = stop_loss
                            exit_reason = "Stop Loss"
                        elif current_price >= take_profit:
                            exit_trade = True
                            exit_price = take_profit
                            exit_reason = "Take Profit"
                    else:  # Short position
                        if current_price >= stop_loss:
                            exit_trade = True
                            exit_price = stop_loss
                            exit_reason = "Stop Loss"
                        elif current_price <= take_profit:
                            exit_trade = True
                            exit_price = take_profit
                            exit_reason = "Take Profit"
                    
                    # Check for opposite signal
                    if signal != 0 and signal != position:
                        exit_trade = True
                        exit_reason = "Signal Reversal"
                    
                    # Exit trade if conditions met
                    if exit_trade:
                        # Calculate P&L
                        if position == 1:  # Long
                            pips_gained = (exit_price - entry_price) * (10000 if 'JPY' not in pair else 100)
                        else:  # Short
                            pips_gained = (entry_price - exit_price) * (10000 if 'JPY' not in pair else 100)
                        
                        # Apply spread cost
                        pips_gained -= spread_pips
                        
                        # Apply slippage
                        pips_gained -= self.slippage_pips
                        
                        # Calculate monetary P&L
                        pip_value = 10 if 'JPY' not in pair else 1000  # Per standard lot
                        trade_pnl = pips_gained * pip_value * position_size
                        
                        # Apply commission
                        trade_pnl -= self.commission_per_lot * position_size
                        
                        # Update balance
                        balance += trade_pnl
                        equity = balance
                        
                        # Track consecutive losses
                        if trade_pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0
                        
                        # Record trade
                        trade_duration = (current_time - entry_time).total_seconds() / 3600  # hours
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'pair': pair,
                            'direction': 'Long' if position == 1 else 'Short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'pips': pips_gained,
                            'pnl': trade_pnl,
                            'balance': balance,
                            'duration_hours': trade_duration,
                            'exit_reason': exit_reason
                        })
                        
                        # Reset position
                        position = 0
                        entry_price = 0
                        entry_time = None
                        
                        # Update drawdown
                        if equity > peak_equity:
                            peak_equity = equity
                        current_drawdown = (peak_equity - equity) / peak_equity
                        max_drawdown = max(max_drawdown, current_drawdown)
                        
                        # Stop trading if max drawdown exceeded
                        if max_drawdown > self.max_drawdown:
                            print(f"    ⚠️ Max drawdown exceeded, stopping trading")
                            break
                
                # Check for new entry signals (with additional safety checks)
                if (position == 0 and signal != 0 and 
                    consecutive_losses < max_consecutive_losses and
                    max_drawdown < self.max_drawdown * 0.8):  # Only trade if well below max DD
                    
                    # Calculate ATR for stop loss
                    atr = self.calculate_atr(df['high'], df['low'], df['close']).iloc[i]
                    if pd.isna(atr) or atr == 0:
                        atr = 0.001  # Default ATR
                    
                    # Ultra-conservative stop loss and take profit
                    atr_multiplier = 1.5  # Tighter stop loss
                    stop_loss_pips = atr * (10000 if 'JPY' not in pair else 100) * atr_multiplier
                    take_profit_pips = stop_loss_pips * 3  # 3:1 reward to risk
                    
                    # Calculate position size
                    position_size = self.calculate_conservative_position_size(
                        balance, self.risk_per_trade, stop_loss_pips, pair)
                    
                    # Enter position
                    position = signal
                    entry_price = current_price
                    entry_time = current_time
                    
                    if position == 1:  # Long
                        stop_loss = entry_price - (stop_loss_pips / (10000 if 'JPY' not in pair else 100))
                        take_profit = entry_price + (take_profit_pips / (10000 if 'JPY' not in pair else 100))
                    else:  # Short
                        stop_loss = entry_price + (stop_loss_pips / (10000 if 'JPY' not in pair else 100))
                        take_profit = entry_price - (take_profit_pips / (10000 if 'JPY' not in pair else 100))
                
                # Track daily returns
                if i > 0:
                    daily_return = (equity - self.initial_balance) / self.initial_balance
                    daily_returns.append(daily_return)
            
            # Calculate performance metrics
            total_trades = len(trades)
            if total_trades == 0:
                return None
            
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            total_return = (balance - self.initial_balance) / self.initial_balance
            
            # Strict prop firm compliance check
            prop_firm_compliant = (
                total_return >= self.profit_target and
                max_drawdown <= self.max_drawdown and
                win_rate >= self.min_win_rate
            )
            
            results = {
                'strategy': strategy_name,
                'pair': pair,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'final_balance': balance,
                'max_drawdown': max_drawdown,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                'prop_firm_compliant': prop_firm_compliant,
                'consecutive_losses': consecutive_losses,
                'trades': trades[-10:] if len(trades) > 10 else trades  # Last 10 trades
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error in backtest: {e}")
            return None
    
    def run_conservative_analysis(self):
        """Run conservative analysis focusing on capital preservation"""
        print("🛡️ STARTING CONSERVATIVE PROP FIRM SYSTEM ANALYSIS")
        print("=" * 60)
        print("🎯 Focus: Capital preservation with modest returns")
        print("📊 Risk per trade: 0.25% | Max drawdown: 4% | Min win rate: 45%")
        print("=" * 60)
        
        # Focus on most stable pairs and timeframes
        pairs = ['EUR_USD', 'GBP_USD', 'AUD_USD']  # Major pairs only
        timeframes = ['1hour']  # Longer timeframes for stability
        strategies = [
            ('Conservative Trend', self.conservative_trend_strategy),
            ('Conservative Mean Reversion', self.conservative_mean_reversion_strategy)
        ]
        
        all_results = []
        
        for pair in pairs:
            for timeframe in timeframes:
                print(f"\n📊 Analyzing {pair} {timeframe}...")
                
                # Load data
                df = self.load_data(pair, timeframe)
                if df is None:
                    continue
                
                for strategy_name, strategy_func in strategies:
                    print(f"  🔄 Testing {strategy_name}...")
                    
                    try:
                        # Apply strategy
                        df_strategy = df.copy()
                        df_strategy = strategy_func(df_strategy)
                        
                        # Backtest
                        results = self.backtest_conservative_strategy(df_strategy, pair, f"{strategy_name}_{timeframe}")
                        
                        if results:
                            results['timeframe'] = timeframe
                            all_results.append(results)
                            
                            # Print detailed summary
                            print(f"    ✅ {results['total_trades']} trades")
                            print(f"    📈 {results['win_rate']:.1%} win rate")
                            print(f"    💰 {results['total_return']:.1%} return")
                            print(f"    📉 {results['max_drawdown']:.1%} max DD")
                            print(f"    💎 Profit factor: {results['profit_factor']:.2f}")
                            
                            if results['prop_firm_compliant']:
                                print(f"    🎯 ✅ PROP FIRM COMPLIANT!")
                            else:
                                print(f"    ❌ Not compliant")
                        else:
                            print(f"    ❌ No valid results")
                            
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
        
        # Analyze results
        if all_results:
            # Sort by compliance first, then by return
            compliant_strategies = [r for r in all_results if r['prop_firm_compliant']]
            non_compliant_strategies = [r for r in all_results if not r['prop_firm_compliant']]
            
            compliant_strategies.sort(key=lambda x: x['total_return'], reverse=True)
            non_compliant_strategies.sort(key=lambda x: x['total_return'], reverse=True)
            
            all_results = compliant_strategies + non_compliant_strategies
            
            self.results = {
                'analysis_time': datetime.now().isoformat(),
                'total_strategies_tested': len(all_results),
                'prop_firm_compliant_count': len(compliant_strategies),
                'best_overall': all_results[0] if all_results else None,
                'best_compliant': compliant_strategies[0] if compliant_strategies else None,
                'all_results': all_results,
                'summary': self.generate_conservative_summary(all_results, compliant_strategies)
            }
            
            # Save results
            self.save_results()
            
            # Print final summary
            self.print_conservative_summary()
            
        else:
            print("❌ No valid results generated")
    
    def generate_conservative_summary(self, all_results, compliant_strategies):
        """Generate conservative analysis summary"""
        if not all_results:
            return "No results to analyze"
        
        # Calculate statistics
        avg_return = np.mean([r['total_return'] for r in all_results])
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in all_results])
        avg_profit_factor = np.mean([r['profit_factor'] for r in all_results])
        
        summary = {
            'total_strategies': len(all_results),
            'compliant_strategies': len(compliant_strategies),
            'compliance_rate': len(compliant_strategies) / len(all_results) if all_results else 0,
            'average_return': avg_return,
            'average_win_rate': avg_win_rate,
            'average_drawdown': avg_drawdown,
            'average_profit_factor': avg_profit_factor
        }
        
        return summary
    
    def save_results(self):
        """Save results to JSON file"""
        try:
            filename = "CONSERVATIVE_PROP_FIRM_RESULTS.json"
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def print_conservative_summary(self):
        """Print comprehensive conservative summary"""
        print("\n" + "=" * 80)
        print("🛡️ CONSERVATIVE PROP FIRM SYSTEM - FINAL RESULTS")
        print("=" * 80)
        
        if not self.results:
            print("❌ No results to display")
            return
        
        summary = self.results['summary']
        
        print(f"📊 CONSERVATIVE ANALYSIS OVERVIEW:")
        print(f"   • Total Strategies Tested: {summary['total_strategies']}")
        print(f"   • Prop Firm Compliant: {summary['compliant_strategies']} ({summary['compliance_rate']:.1%})")
        print(f"   • Average Return: {summary['average_return']:.1%}")
        print(f"   • Average Win Rate: {summary['average_win_rate']:.1%}")
        print(f"   • Average Max Drawdown: {summary['average_drawdown']:.1%}")
        print(f"   • Average Profit Factor: {summary['average_profit_factor']:.2f}")
        
        if self.results['best_compliant']:
            best = self.results['best_compliant']
            print(f"\n🎯 ✅ BEST PROP FIRM COMPLIANT STRATEGY:")
            print(f"   • Strategy: {best['strategy']} on {best['pair']} {best['timeframe']}")
            print(f"   • Total Return: {best['total_return']:.1%}")
            print(f"   • Win Rate: {best['win_rate']:.1%}")
            print(f"   • Max Drawdown: {best['max_drawdown']:.1%}")
            print(f"   • Total Trades: {best['total_trades']}")
            print(f"   • Profit Factor: {best['profit_factor']:.2f}")
            print(f"   • Final Balance: ${best['final_balance']:,.2f}")
            print(f"   • ✅ READY FOR PROP FIRM CHALLENGE!")
        else:
            print(f"\n❌ NO PROP FIRM COMPLIANT STRATEGIES FOUND")
            
            if self.results['best_overall']:
                best = self.results['best_overall']
                print(f"\n🚀 BEST OVERALL STRATEGY (not compliant):")
                print(f"   • Strategy: {best['strategy']} on {best['pair']} {best['timeframe']}")
                print(f"   • Total Return: {best['total_return']:.1%}")
                print(f"   • Win Rate: {best['win_rate']:.1%}")
                print(f"   • Max Drawdown: {best['max_drawdown']:.1%}")
        
        print(f"\n💡 CONSERVATIVE RECOMMENDATIONS:")
        if summary['compliant_strategies'] > 0:
            print(f"   ✅ Found {summary['compliant_strategies']} prop firm compliant strategies!")
            print(f"   ✅ These strategies prioritize capital preservation")
            print(f"   ✅ Ready for live prop firm challenge")
            print(f"   ✅ Start with demo trading for 1-2 weeks")
        else:
            print(f"   ⚠️  Even conservative approach didn't meet requirements")
            print(f"   ⚠️  Consider even lower risk (0.1% per trade)")
            print(f"   ⚠️  Focus on longer timeframes (4H, Daily)")
            print(f"   ⚠️  Test with smaller profit targets (5-6%)")
        
        print(f"\n📈 NEXT STEPS:")
        print(f"   1. Review detailed results in CONSERVATIVE_PROP_FIRM_RESULTS.json")
        print(f"   2. Demo trade the best strategy for 2 weeks minimum")
        print(f"   3. If consistent, apply to prop firm")
        print(f"   4. Start with smallest challenge size available")
        
        print("=" * 80)

def main():
    """Main execution function"""
    print("🛡️ CONSERVATIVE AUTOMATED PROP FIRM TRADING SYSTEM")
    print("🎯 Ultra-conservative approach for prop firm success")
    print("💰 Focus: Capital preservation over aggressive returns")
    print("📊 Using real Oanda market data")
    print()
    
    # Initialize system
    system = ConservativePropFirmSystem()
    
    # Run conservative analysis
    system.run_conservative_analysis()
    
    print("\n🎉 Conservative analysis complete!")

if __name__ == "__main__":
    main() 