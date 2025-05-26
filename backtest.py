import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from env import MasterTrendEnv

# Load the trained model
model = PPO.load('mt_agent_updated_v3.zip')  # Test with existing model first

# Initialize environment with historical data
env = MasterTrendEnv(csv_path='EURUSD_data_15M.csv', window_size=60, init_cash=10000,
                     commission=0.0001, slippage=0.0, max_daily_dd=0.02,
                     holding_cost=0.1, trade_reward=0.1, use_dd_penalty=True)

# Reset environment
obs = env.reset()
done = False
info_history = []
action_history = []
entry_price = None
trade_log = []
step_details = []
position_transition_log = []
current_trade = None

# Run backtest
while not done:
    action, _states = model.predict(obs, deterministic=True)
    raw_action = action  # Log the raw prediction from the model
    print(f"Step {len(step_details) + 1}: Raw Model Prediction = {raw_action}")
    pre_position = env.position
    pre_equity = env.equity
    obs, reward, done, info = env.step(action)
    post_position = env.position
    post_equity = env.equity
    cost_incurred = pre_equity - post_equity if reward < 0 else 0.0
    info_history.append(info)
    action_scalar = int(action) if isinstance(action, np.ndarray) else action
    action_str = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}[action_scalar]
    step_details.append({
        'step': len(step_details) + 1,
        'action': action_str,
        'pre_position': pre_position,
        'post_position': post_position,
        'pre_equity': pre_equity,
        'post_equity': post_equity,
        'reward': reward,
        'cost_incurred': cost_incurred
    })
    # Track trade entry and exit
    if pre_position != post_position:
        if post_position == 1:  # Entering long position
            entry_price = env.df.iloc[env.idx - 1]['open'] * (1 + env.slippage)
            current_trade = {'entry_step': len(step_details), 'entry_price': entry_price, 'entry_position': post_position, 'entry_equity': post_equity, 'entry_time': str(env.df.iloc[env.idx - 1]['datetime'])}
            position_transition_log.append({'step': len(step_details), 'timestamp': str(env.df.iloc[env.idx - 1]['datetime']), 'transition': f'{pre_position} to {post_position}', 'price': entry_price, 'equity': post_equity, 'type': 'ENTRY_LONG'})
        elif post_position == -1:  # Entering short position
            entry_price = env.df.iloc[env.idx - 1]['open'] * (1 - env.slippage)
            current_trade = {'entry_step': len(step_details), 'entry_price': entry_price, 'entry_position': post_position, 'entry_equity': post_equity, 'entry_time': str(env.df.iloc[env.idx - 1]['datetime'])}
            position_transition_log.append({'step': len(step_details), 'timestamp': str(env.df.iloc[env.idx - 1]['datetime']), 'transition': f'{pre_position} to {post_position}', 'price': entry_price, 'equity': post_equity, 'type': 'ENTRY_SHORT'})
        elif post_position == 0 and current_trade is not None:  # Exiting position
            exit_price = env.df.iloc[env.idx - 1]['open']
            exit_time = str(env.df.iloc[env.idx - 1]['datetime'])
            if current_trade['entry_position'] == 1:  # Exiting long
                pnl = exit_price - current_trade['entry_price']
                position_transition_log.append({'step': len(step_details), 'timestamp': exit_time, 'transition': f'{pre_position} to {post_position}', 'price': exit_price, 'equity': post_equity, 'type': 'EXIT_LONG', 'pnl': pnl})
                trade_log.append({
                    'entry_step': current_trade['entry_step'],
                    'exit_step': len(step_details),
                    'entry_time': current_trade['entry_time'],
                    'exit_time': exit_time,
                    'entry_price': current_trade['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'entry_equity': current_trade['entry_equity'],
                    'exit_equity': post_equity,
                    'position_type': 'LONG'
                })
            elif current_trade['entry_position'] == -1:  # Exiting short
                pnl = current_trade['entry_price'] - exit_price
                position_transition_log.append({'step': len(step_details), 'timestamp': exit_time, 'transition': f'{pre_position} to {post_position}', 'price': exit_price, 'equity': post_equity, 'type': 'EXIT_SHORT', 'pnl': pnl})
                trade_log.append({
                    'entry_step': current_trade['entry_step'],
                    'exit_step': len(step_details),
                    'entry_time': current_trade['entry_time'],
                    'exit_time': exit_time,
                    'entry_price': current_trade['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'entry_equity': current_trade['entry_equity'],
                    'exit_equity': post_equity,
                    'position_type': 'SHORT'
                })
            current_trade = None
    if action_str == 'BUY' and info['position'] == 1 and entry_price is None:
        entry_price = env.df.iloc[env.idx - 1]['open'] * (1 + env.slippage)
    elif action_str != 'BUY' and info['position'] == 0 and entry_price is not None:
        exit_price = env.df.iloc[env.idx - 1]['open']
        pnl = exit_price - entry_price
        trade_log.append({'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl, 'equity': info['equity']})
        entry_price = None
    elif action_str == 'SELL' and info['position'] == -1 and entry_price is None:
        entry_price = env.df.iloc[env.idx - 1]['open'] * (1 - env.slippage)
    elif action_str != 'SELL' and info['position'] == 0 and entry_price is not None:
        exit_price = env.df.iloc[env.idx - 1]['open']
        pnl = entry_price - exit_price
        trade_log.append({'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl, 'equity': info['equity']})
        entry_price = None
    action_history.append({'action': action_str, 'equity': info['equity'], 'position': info['position']})

# Compile results
df_results = pd.DataFrame(info_history)
df_actions = pd.DataFrame(action_history)
df_trades = pd.DataFrame(trade_log)
df_steps = pd.DataFrame(step_details)
df_transitions = pd.DataFrame(position_transition_log)
print('Position Transition Summary:')
if not df_transitions.empty:
    print(f"Total Position Transitions: {len(df_transitions)}")
    print(f"Entry Long Transitions: {len(df_transitions[df_transitions['type'] == 'ENTRY_LONG'])}")
    print(f"Entry Short Transitions: {len(df_transitions[df_transitions['type'] == 'ENTRY_SHORT'])}")
    print(f"Exit Long Transitions: {len(df_transitions[df_transitions['type'] == 'EXIT_LONG'])}")
    print(f"Exit Short Transitions: {len(df_transitions[df_transitions['type'] == 'EXIT_SHORT'])}")
    if 'pnl' in df_transitions.columns:
        exit_transitions = df_transitions[df_transitions['type'].str.contains('EXIT')]
        if not exit_transitions.empty:
            print(f"Average P&L on Exits: {exit_transitions['pnl'].mean():.4f}")
            print(f"Total P&L from Exits: {exit_transitions['pnl'].sum():.4f}")
    df_transitions.to_csv('backtest_transitions.csv', index=False)
    print('Position transition details saved to backtest_transitions.csv')
else:
    print('No position transitions occurred during the backtest.')
print('Step-by-Step Analysis:')
if not df_steps.empty:
    print(f"Total Steps: {len(df_steps)}")
    print(f"Steps with Position Change: {len(df_steps[df_steps['pre_position'] != df_steps['post_position']])}")
    print(f"Steps with Cost Incurred: {len(df_steps[df_steps['cost_incurred'] > 0])}")
    print(f"Total Cost Incurred: {df_steps['cost_incurred'].sum():.4f}")
    df_steps.to_csv('backtest_steps.csv', index=False)
    print('Step details saved to backtest_steps.csv')
else:
    print('No step details available.')
print('Trade Summary:')
if not df_trades.empty:
    print(f"Total Trades: {len(df_trades)}")
    print(f"Average P&L per Trade: {df_trades['pnl'].mean():.4f}")
    print(f"Total P&L: {df_trades['pnl'].sum():.4f}")
    print(f"Long Trades: {len(df_trades[df_trades['position_type'] == 'LONG'])}")
    print(f"Short Trades: {len(df_trades[df_trades['position_type'] == 'SHORT'])}")
    long_trades = df_trades[df_trades['position_type'] == 'LONG']
    short_trades = df_trades[df_trades['position_type'] == 'SHORT']
    if not long_trades.empty:
        print(f"Average P&L per Long Trade: {long_trades['pnl'].mean():.4f}")
        print(f"Total P&L from Long Trades: {long_trades['pnl'].sum():.4f}")
    if not short_trades.empty:
        print(f"Average P&L per Short Trade: {short_trades['pnl'].mean():.4f}")
        print(f"Total P&L from Short Trades: {short_trades['pnl'].sum():.4f}")
    df_trades.to_csv('backtest_trades.csv', index=False)
    print('Trade details saved to backtest_trades.csv')
else:
    print('No trades were completed during the backtest.')
print('Action Summary:')
print(f"Total Actions: {len(df_actions)}")
print(f"HOLD Actions: {len(df_actions[df_actions['action'] == 'HOLD'])}")
print(f"BUY Actions: {len(df_actions[df_actions['action'] == 'BUY'])}")
print(f"SELL Actions: {len(df_actions[df_actions['action'] == 'SELL'])}")
df_actions.to_csv('backtest_actions.csv', index=False)
print('Action details saved to backtest_actions.csv')
print('Backtest Results:')
print(f"Final Equity: ${df_results['equity'].iloc[-1]:.2f}")
print(f"Max Drawdown: {df_results['drawdown'].max()*100:.2f}%")

# Check against prop firm metrics
initial_cash = env.init_cash
profit_target = env.profit_target
max_overall_dd = env.max_overall_dd
max_daily_dd = env.max_daily_dd
daily_loss_limit = env.daily_loss_limit

final_equity = df_results['equity'].iloc[-1]
overall_dd = (df_results['equity'].max() - df_results['equity'].min()) / df_results['equity'].max()
daily_dds = df_results.groupby(df_results.index // 1440)['equity'].apply(lambda x: (x.max() - x.min()) / x.max() if x.max() > 0 else 0)
max_daily_loss = df_results.groupby(df_results.index // 1440)['equity'].apply(lambda x: (x.iloc[0] - x.min()) / initial_cash if len(x) > 0 else 0).max()

print('\nProp Firm Metrics Check:')
print(f"Profit Target ({profit_target*100}%): {'Achieved' if final_equity >= initial_cash * (1 + profit_target) else 'Not Achieved'} (Final: {(final_equity/initial_cash - 1)*100:.2f}%)")
print(f"Overall Drawdown Limit ({max_overall_dd*100}%): {'Pass' if overall_dd <= max_overall_dd else 'Fail'} (Actual: {overall_dd*100:.2f}%)")
print(f"Max Daily Drawdown Limit ({max_daily_dd*100}%): {'Pass' if daily_dds.max() <= max_daily_dd else 'Fail'} (Actual: {daily_dds.max()*100:.2f}%)")
print(f"Daily Loss Limit ({daily_loss_limit*100}%): {'Pass' if max_daily_loss <= daily_loss_limit else 'Fail'} (Actual: {max_daily_loss*100:.2f}%)")

# Optional: Save results to CSV for further analysis
df_results.to_csv('backtest_results.csv', index=False)
print('Results saved to backtest_results.csv') 