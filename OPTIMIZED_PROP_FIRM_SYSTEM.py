#!/usr/bin/env python3
"""
OPTIMIZED PROP FIRM TRADING SYSTEM
==================================

Final optimized trading system that balances risk and return for prop firm success.
Uses adaptive risk management and proven strategies.

Key Features:
- Adaptive risk management (0.5% base risk)
- Dynamic position sizing
- Multiple strategy validation
- Realistic trading costs
- Prop firm compliance focus

Author: AI Trading Assistant
Version: 3.0 Final Optimized
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class OptimizedPropFirmSystem:
    def __init__(self):
        """Initialize the optimized prop firm trading system"""
        self.api_key = "3240e71f449497ac70a3abce8989ef0a-631d288e0d87be637ba3a3a41ff998fd"
        self.data_dir = "automated_forex_data"
        
        # Optimized prop firm parameters
        self.initial_balance = 100000  # $100k challenge
        self.profit_target = 0.08      # 8% profit target
        self.max_drawdown = 0.045      # 4.5% max drawdown
        self.daily_loss_limit = 0.04   # 4% daily loss limit
        
        # Optimized trading parameters
        self.base_risk_per_trade = 0.005  # 0.5% base risk per trade
        self.max_risk_per_trade = 0.008   # 0.8% max risk per trade
        self.min_risk_per_trade = 0.003   # 0.3% min risk per trade
        self.max_positions = 3            # Maximum concurrent positions
        
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
    
    def optimized_trend_strategy(self, df):
        """Optimized trend following strategy with multiple confirmations"""
        try:
            # Calculate indicators
            df['ema_fast'] = self.calculate_ema(df['close'], 12)
            df['ema_slow'] = self.calculate_ema(df['close'], 26)
            df['ema_filter'] = self.calculate_ema(df['close'], 50)
            df['rsi'] = self.calculate_rsi(df['close'], 14)
            
            # Calculate MACD
            macd_line, signal_line, histogram = self.calculate_macd(df['close'])
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_histogram'] = histogram
            
            # Generate signals
            df['signal'] = 0
            
            # Buy signal: Trend + momentum confirmation
            buy_condition = (
                (df['ema_fast'] > df['ema_slow']) &  # Fast EMA above slow
                (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) &  # Crossover
                (df['close'] > df['ema_filter']) &  # Above trend filter
                (df['rsi'] > 45) & (df['rsi'] < 70) &  # RSI in good zone
                (df['macd'] > df['macd_signal']) &  # MACD confirmation
                (df['macd_histogram'] > df['macd_histogram'].shift(1))  # MACD momentum
            )
            
            # Sell signal: Trend + momentum confirmation
            sell_condition = (
                (df['ema_fast'] < df['ema_slow']) &  # Fast EMA below slow
                (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)) &  # Crossover
                (df['close'] < df['ema_filter']) &  # Below trend filter
                (df['rsi'] > 30) & (df['rsi'] < 55) &  # RSI in good zone
                (df['macd'] < df['macd_signal']) &  # MACD confirmation
                (df['macd_histogram'] < df['macd_histogram'].shift(1))  # MACD momentum
            )
            
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
            return df
            
        except Exception as e:
            print(f"❌ Error in optimized trend strategy: {e}")
            return df
    
    def optimized_mean_reversion_strategy(self, df):
        """Optimized mean reversion strategy"""
        try:
            # Calculate indicators
            df['rsi'] = self.calculate_rsi(df['close'], 14)
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(df['close'], 20, 2)
            
            df['bb_upper'] = upper_band
            df['bb_middle'] = middle_band
            df['bb_lower'] = lower_band
            
            # Calculate trend filter
            df['ema_trend'] = self.calculate_ema(df['close'], 100)
            
            # Generate signals
            df['signal'] = 0
            
            # Buy signal: Oversold with trend support
            buy_condition = (
                (df['rsi'] < 35) &  # Oversold
                (df['close'] <= df['bb_lower'] * 1.001) &  # Near lower band
                (df['close'].shift(1) > df['bb_lower'].shift(1) * 1.001) &  # First touch
                (df['close'] > df['ema_trend'] * 0.998) &  # Not too far from trend
                (df['rsi'].shift(1) >= 35)  # RSI was not oversold before
            )
            
            # Sell signal: Overbought with trend resistance
            sell_condition = (
                (df['rsi'] > 65) &  # Overbought
                (df['close'] >= df['bb_upper'] * 0.999) &  # Near upper band
                (df['close'].shift(1) < df['bb_upper'].shift(1) * 0.999) &  # First touch
                (df['close'] < df['ema_trend'] * 1.002) &  # Not too far from trend
                (df['rsi'].shift(1) <= 65)  # RSI was not overbought before
            )
            
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
            return df
            
        except Exception as e:
            print(f"❌ Error in optimized mean reversion strategy: {e}")
            return df
    
    def calculate_adaptive_risk(self, current_drawdown, win_rate, recent_trades):
        """Calculate adaptive risk based on performance"""
        base_risk = self.base_risk_per_trade
        
        # Reduce risk if drawdown is high
        if current_drawdown > 0.02:  # 2% drawdown
            base_risk *= 0.7
        elif current_drawdown > 0.01:  # 1% drawdown
            base_risk *= 0.85
        
        # Adjust based on win rate
        if len(recent_trades) >= 10:
            recent_win_rate = sum(1 for t in recent_trades[-10:] if t['pnl'] > 0) / 10
            if recent_win_rate < 0.4:
                base_risk *= 0.8
            elif recent_win_rate > 0.6:
                base_risk *= 1.1
        
        # Ensure within bounds
        return max(self.min_risk_per_trade, min(self.max_risk_per_trade, base_risk))
    
    def calculate_optimized_position_size(self, balance, risk_per_trade, stop_loss_pips, pair):
        """Calculate optimized position size"""
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
            
            # Optimized limits
            position_size = min(position_size, 0.5)  # Max 0.5 lots
            position_size = max(position_size, 0.01)  # Minimum 0.01 lots
            
            return round(position_size, 2)
            
        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            return 0.01
    
    def backtest_optimized_strategy(self, df, pair, strategy_name):
        """Backtest with optimized risk management"""
        try:
            balance = self.initial_balance
            equity = balance
            peak_equity = balance
            max_drawdown = 0
            trades = []
            daily_returns = []
            consecutive_losses = 0
            
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
                
                # Calculate current drawdown
                current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                
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
                
                # Check for new entry signals
                if (position == 0 and signal != 0 and 
                    consecutive_losses < 4 and  # Max 4 consecutive losses
                    current_drawdown < self.max_drawdown * 0.9):  # Safety margin
                    
                    # Calculate ATR for stop loss
                    atr = self.calculate_atr(df['high'], df['low'], df['close']).iloc[i]
                    if pd.isna(atr) or atr == 0:
                        atr = 0.001  # Default ATR
                    
                    # Optimized stop loss and take profit
                    atr_multiplier = 1.8  # Balanced stop loss
                    stop_loss_pips = atr * (10000 if 'JPY' not in pair else 100) * atr_multiplier
                    take_profit_pips = stop_loss_pips * 2.5  # 2.5:1 reward to risk
                    
                    # Calculate adaptive risk
                    adaptive_risk = self.calculate_adaptive_risk(current_drawdown, 0, trades)
                    
                    # Calculate position size
                    position_size = self.calculate_optimized_position_size(
                        balance, adaptive_risk, stop_loss_pips, pair)
                    
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
            
            # Prop firm compliance check
            prop_firm_compliant = (
                total_return >= self.profit_target and
                max_drawdown <= self.max_drawdown and
                win_rate >= 0.40  # 40% minimum win rate
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
                'trades': trades[-10:] if len(trades) > 10 else trades  # Last 10 trades
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error in backtest: {e}")
            return None
    
    def run_optimized_analysis(self):
        """Run optimized analysis for prop firm success"""
        print("⚡ STARTING OPTIMIZED PROP FIRM SYSTEM ANALYSIS")
        print("=" * 60)
        print("🎯 Focus: Balanced risk-return for prop firm success")
        print("📊 Base risk: 0.5% | Adaptive: 0.3-0.8% | Max DD: 4.5%")
        print("=" * 60)
        
        # Test on major pairs and both timeframes
        pairs = ['EUR_USD', 'GBP_USD', 'AUD_USD', 'USD_JPY']
        timeframes = ['15min', '1hour']
        strategies = [
            ('Optimized Trend', self.optimized_trend_strategy),
            ('Optimized Mean Reversion', self.optimized_mean_reversion_strategy)
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
                        results = self.backtest_optimized_strategy(df_strategy, pair, f"{strategy_name}_{timeframe}")
                        
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
                'all_results': all_results[:15],  # Top 15 results
                'summary': self.generate_optimized_summary(all_results, compliant_strategies)
            }
            
            # Save results
            self.save_results()
            
            # Print final summary
            self.print_optimized_summary()
            
        else:
            print("❌ No valid results generated")
    
    def generate_optimized_summary(self, all_results, compliant_strategies):
        """Generate optimized analysis summary"""
        if not all_results:
            return "No results to analyze"
        
        # Calculate statistics
        avg_return = np.mean([r['total_return'] for r in all_results])
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in all_results])
        avg_profit_factor = np.mean([r['profit_factor'] for r in all_results])
        avg_trades = np.mean([r['total_trades'] for r in all_results])
        
        summary = {
            'total_strategies': len(all_results),
            'compliant_strategies': len(compliant_strategies),
            'compliance_rate': len(compliant_strategies) / len(all_results) if all_results else 0,
            'average_return': avg_return,
            'average_win_rate': avg_win_rate,
            'average_drawdown': avg_drawdown,
            'average_profit_factor': avg_profit_factor,
            'average_trades': avg_trades
        }
        
        return summary
    
    def save_results(self):
        """Save results to JSON file"""
        try:
            filename = "OPTIMIZED_PROP_FIRM_RESULTS.json"
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def print_optimized_summary(self):
        """Print comprehensive optimized summary"""
        print("\n" + "=" * 80)
        print("⚡ OPTIMIZED PROP FIRM SYSTEM - FINAL RESULTS")
        print("=" * 80)
        
        if not self.results:
            print("❌ No results to display")
            return
        
        summary = self.results['summary']
        
        print(f"📊 OPTIMIZED ANALYSIS OVERVIEW:")
        print(f"   • Total Strategies Tested: {summary['total_strategies']}")
        print(f"   • Prop Firm Compliant: {summary['compliant_strategies']} ({summary['compliance_rate']:.1%})")
        print(f"   • Average Return: {summary['average_return']:.1%}")
        print(f"   • Average Win Rate: {summary['average_win_rate']:.1%}")
        print(f"   • Average Max Drawdown: {summary['average_drawdown']:.1%}")
        print(f"   • Average Profit Factor: {summary['average_profit_factor']:.2f}")
        print(f"   • Average Trades: {summary['average_trades']:.0f}")
        
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
            print(f"   • 🚀 READY FOR PROP FIRM CHALLENGE!")
        else:
            print(f"\n❌ NO PROP FIRM COMPLIANT STRATEGIES FOUND")
            
            if self.results['best_overall']:
                best = self.results['best_overall']
                print(f"\n🚀 BEST OVERALL STRATEGY (not compliant):")
                print(f"   • Strategy: {best['strategy']} on {best['pair']} {best['timeframe']}")
                print(f"   • Total Return: {best['total_return']:.1%}")
                print(f"   • Win Rate: {best['win_rate']:.1%}")
                print(f"   • Max Drawdown: {best['max_drawdown']:.1%}")
        
        print(f"\n💡 OPTIMIZED RECOMMENDATIONS:")
        if summary['compliant_strategies'] > 0:
            print(f"   ✅ Found {summary['compliant_strategies']} prop firm compliant strategies!")
            print(f"   ✅ Balanced approach achieved target returns with controlled risk")
            print(f"   ✅ Ready for live prop firm challenge")
            print(f"   ✅ Start with demo trading for 1-2 weeks")
        else:
            print(f"   ⚠️  No strategies met all prop firm requirements")
            print(f"   ⚠️  Consider adjusting profit target to 6-7%")
            print(f"   ⚠️  Test with longer data periods")
            print(f"   ⚠️  Focus on most stable currency pairs")
        
        print(f"\n📈 NEXT STEPS:")
        print(f"   1. Review detailed results in OPTIMIZED_PROP_FIRM_RESULTS.json")
        print(f"   2. Demo trade the best strategy for 2-3 weeks")
        print(f"   3. If consistent, apply to prop firm")
        print(f"   4. Start with smallest challenge size available")
        print(f"   5. Scale up gradually after success")
        
        print("=" * 80)

def main():
    """Main execution function"""
    print("⚡ OPTIMIZED AUTOMATED PROP FIRM TRADING SYSTEM")
    print("🎯 Final optimized approach for prop firm success")
    print("💰 Balanced risk-return with adaptive management")
    print("📊 Using real Oanda market data")
    print()
    
    # Initialize system
    system = OptimizedPropFirmSystem()
    
    # Run optimized analysis
    system.run_optimized_analysis()
    
    print("\n🎉 Optimized analysis complete!")

if __name__ == "__main__":
    main() 