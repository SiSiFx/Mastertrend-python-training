import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import MasterTrendEnv


def smoke_test_env():
    """Runs 100 random steps in the environment to ensure it works"""
    # Simplest settings: no commission, no slippage, no drawdown penalty for exploration
    env = MasterTrendEnv('EURUSD_data_1M.csv', commission=0.0, slippage=0.0, max_daily_dd=1.0,
                         holding_cost=0.1, trade_reward=0.1, use_dd_penalty=False)
    obs = env.reset()
    total_reward = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print('Smoke test complete: cumulative reward over 100 random steps =', total_reward)


if __name__ == '__main__':
    # 1) Smoke-test the environment
    smoke_test_env()

    # 2) Train a PPO agent
    # Create the environment
    env = make_vec_env(lambda: MasterTrendEnv(csv_path='EURUSD_data_15M.csv', window_size=60, init_cash=10000,
                                             commission=0.0001, slippage=0.0, max_daily_dd=0.02,
                                             holding_cost=0.1, trade_reward=0.1, use_dd_penalty=True),
                       n_envs=4)

    # Initialize the PPO model
    model = PPO('MultiInputPolicy', env, verbose=1, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2)

    # Train the model
    model.learn(total_timesteps=1000000, log_interval=10)

    # Save the trained model
    model.save('mt_agent_balanced_v4')

    print('Training completed. Model saved as mt_agent_balanced_v4.zip') 