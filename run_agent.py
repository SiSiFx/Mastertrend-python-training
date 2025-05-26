import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import MasterTrendEnv


def evaluate_model(csv_path, model_path, window_size=60, init_cash=10000, commission=0.0001, slippage=0.0, max_daily_dd=0.02):
    """
    Load a saved PPO model and evaluate it on fresh OHLCV data.
    Tracks equity curve, net PnL, and max drawdown.
    """
    # Initialize environment and model (P&L-based rewards)
    env = MasterTrendEnv(csv_path, window_size=window_size,
                           init_cash=init_cash,
                           commission=commission,
                           slippage=slippage,
                           max_daily_dd=max_daily_dd)
    model = PPO.load(model_path)

    obs = env.reset()
    done = False
    equity_curve = [env.equity]
    step_rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_rewards.append(reward)
        equity_curve.append(info.get('equity', env.equity))

    # Summary
    total_steps = len(step_rewards)
    net_pnl = equity_curve[-1] - init_cash
    # Compute max drawdown (peak-to-trough)
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq)
        max_dd = max(max_dd, dd)
    print(f"Total steps: {total_steps}")
    print(f"Final Equity: {equity_curve[-1]:.2f}")
    print(f"Net P&L: {net_pnl:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}")

    # Plot equity curve
    plt.figure(figsize=(10, 4))
    plt.plot(equity_curve, label='Equity Curve')
    plt.xlabel('Step')
    plt.ylabel('Equity')
    plt.title('Agent Equity Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Path to fresh test data and saved model
    test_csv = 'EURUSD_data_1M.csv'
    model_file = 'mt_agent.zip'
    evaluate_model(test_csv, model_file) 