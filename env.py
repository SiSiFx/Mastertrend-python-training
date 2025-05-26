import gym
import numpy as np
import pandas as pd
from gym import spaces
from pure_engine import MasterTrendEngine


class MasterTrendEnv(gym.Env):
    """
    Gym environment wrapping the MasterTrendStrategy for RL.
    State: last N bars of [open, high, low, close, volume]
    Actions: 0 = HOLD, 1 = BUY, 2 = SELL
    """
    metadata = {'render.modes': ['human']}

    def seed(self, seed=None):
        """
        Set the random seed for the environment.
        This is used to ensure reproducibility in random operations.
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def __init__(self, csv_path, window_size=60, init_cash=10000,
                 commission=0.0001, slippage=0.0, max_daily_dd=0.02,
                 holding_cost=0.1, trade_reward=0.1, use_dd_penalty=True):
        super().__init__()
        # Load data
        self.df = pd.read_csv(csv_path, parse_dates=['datetime'])
        self.window = window_size
        self.init_cash = init_cash
        # Trading cost & reward shaping parameters
        self.commission = commission
        self.slippage = slippage
        self.max_daily_dd = max_daily_dd
        self.max_overall_dd = 0.05  # 5% overall drawdown limit common in prop firms
        self.daily_loss_limit = 0.02  # 2% daily loss limit
        self.profit_target = 0.10     # 10% profit target to pass challenge
        self.holding_cost = holding_cost * 0.3  # Further reduce holding cost penalty to encourage holding positions once entered
        self.trade_reward = trade_reward * 10.0  # Drastically increase reward for taking trades to strongly encourage action
        self.use_dd_penalty = use_dd_penalty
        self.hold_penalty = 0.5   # Severely increase penalty for consecutive hold actions to heavily discourage inaction
        self.consecutive_holds = 0  # Track consecutive hold actions
        self.redundant_action_penalty = 0.02  # Small penalty for redundant BUY/SELL actions when already in position

        # Action space
        self.action_space = spaces.Discrete(3)
        # Observation: raw prices and engine indicators
        self.observation_space = spaces.Dict({
            'prices': spaces.Box(low=-np.inf, high=np.inf,
                                  shape=(window_size, 5), dtype=np.float32),
            'inds': spaces.Box(low=-np.inf, high=np.inf,
                               shape=(14,), dtype=np.float32),
        })

    def reset(self):
        # Reset index and strategy
        self.idx = self.window
        self.strategy = MasterTrendEngine()
        # Initialize trading state
        self.equity = self.init_cash
        self.peak_equity = self.init_cash  # Track peak for overall drawdown
        self.position = 0  # 0=flat, 1=long, -1=short
        self.prev_position = 0  # Track previous position for exit detection
        self.entry_price = None
        self.consecutive_holds = 0  # Reset hold counter
        # Daily drawdown tracking
        first_dt = self.df.iloc[self.idx]['datetime']
        self.current_date = first_dt.date()
        self.daily_high = self.equity
        # Prime pure-Python engine with initial window of bars
        for i in range(self.window):
            bar_i = self.df.iloc[i].to_dict()
            self.strategy.next_bar(bar_i)
        return self._get_full_obs()

    def _get_obs(self):
        slice_ = self.df.iloc[self.idx - self.window:self.idx]
        return slice_[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)

    def _get_full_obs(self):
        """Return both raw price window and the latest engine indicator features."""
        prices = self._get_obs()
        e = self.strategy
        inds = np.array([
            e.trend_dir,
            e.jma_val,
            getattr(e, 'macd_hist', 0.0),
            getattr(e, 'p1', 0.0),
            getattr(e, 'b1', 0.0),
            float(getattr(e, 'p1_bullish', False)),
            float(getattr(e, 'p1_bearish', False)),
            float(getattr(e, 'sg_filter_passed', False)),
            float(getattr(e, 'is_williams_high', False)),
            float(getattr(e, 'is_williams_low', False)),
            e.williams_long_stop or 0.0,
            e.williams_short_stop or 0.0,
            e.williams_long_trail_plot or 0.0,
            e.williams_short_trail_plot or 0.0
        ], dtype=np.float32)
        return {'prices': prices, 'inds': inds}

    def step(self, action):
        # Avoid out-of-bounds: if we've consumed all data, terminate
        if self.idx >= len(self.df):
            dd = (self.daily_high - self.equity) / self.daily_high if self.daily_high > 0 else 0
            return self._get_full_obs(), 0.0, True, {'equity': self.equity, 'drawdown': dd, 'position': self.position}

        # Consume one bar
        bar = self.df.iloc[self.idx].to_dict()
        # Run pure-Python engine on this bar to update indicators
        self.strategy.next_bar(bar)
        price_open = bar['open']
        price_close = bar['close']
        dt = bar['datetime']

        # Reset daily drawdown tracking at new day
        if dt.date() != self.current_date:
            self.current_date = dt.date()
            self.daily_high = self.equity

        prev_equity = self.equity
        self.prev_position = self.position  # Store previous position before action
        trade_bonus = 0.0
        # Execute action: long/short/hold
        if action == 1:  # BUY
            # Close short position if open
            if self.position == -1:
                pnl = (self.entry_price - price_open)
                cost = price_open * self.commission
                self.equity += pnl - cost
                self.position = 0
                trade_bonus += self.trade_reward
            # Open long if flat
            if self.position == 0:
                exec_price = price_open * (1 + self.slippage)
                cost = exec_price * self.commission
                self.entry_price = exec_price
                self.equity -= cost
                self.position = 1
                trade_bonus += self.trade_reward
        elif action == 2:  # SELL
            # Close long position if open
            if self.position == 1:
                pnl = (price_open - self.entry_price)
                cost = price_open * self.commission
                self.equity += pnl - cost
                self.position = 0
                trade_bonus += self.trade_reward
            # Open short if flat
            if self.position == 0:
                exec_price = price_open * (1 - self.slippage)
                cost = exec_price * self.commission
                self.entry_price = exec_price
                self.equity -= cost
                self.position = -1
                trade_bonus += self.trade_reward

        # Mark-to-market unrealized P&L
        if self.position == 1:
            # Calculate unrealized P&L but don't update entry_price
            unrealized_pnl = price_close - self.entry_price
            # Update equity with unrealized P&L (this will be reversed next step if position continues)
            self.equity = prev_equity + unrealized_pnl
        elif self.position == -1:
            # Calculate unrealized P&L but don't update entry_price  
            unrealized_pnl = self.entry_price - price_close
            # Update equity with unrealized P&L (this will be reversed next step if position continues)
            self.equity = prev_equity + unrealized_pnl

        # Calculate reward based on equity change
        equity_change = self.equity - prev_equity

        # Base reward on equity change
        reward = equity_change

        # Add penalties for costs
        if action == 1 and self.position != 1:  # Buy entry
            cost = price_open * self.commission
            reward -= cost
        elif action == 2 and self.position != -1:  # Sell entry
            cost = price_open * self.commission
            reward -= cost

        # Track consecutive holds and apply penalty for inaction
        if action == 0:  # HOLD
            self.consecutive_holds += 1
            if self.consecutive_holds > 5:  # Penalty after 5 consecutive holds
                reward -= self.hold_penalty * (self.consecutive_holds - 5)
        else:
            self.consecutive_holds = 0  # Reset on any trade action
            # Add a small bonus for taking any non-HOLD action to encourage trading
            reward += self.trade_reward * 0.5
            # Apply penalty for redundant actions
            if (action == 1 and self.position == 1) or (action == 2 and self.position == -1):
                reward -= self.redundant_action_penalty
        # Subtract holding cost if in position
        if self.position != 0:
            reward -= self.holding_cost
        # Update daily high and compute drawdown
        self.daily_high = max(self.daily_high, self.equity)
        self.peak_equity = max(self.peak_equity, self.equity)
        dd = (self.daily_high - self.equity) / self.daily_high
        overall_dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        daily_pnl = (self.equity - prev_equity) / self.init_cash
        # Check termination conditions
        done = self.idx >= len(self.df)
        # Drawdown penalty if enabled
        if self.use_dd_penalty and dd > self.max_daily_dd:
            reward -= (dd - self.max_daily_dd) * self.init_cash
            done = True
        # Additional penalties for prop firm rules
        if overall_dd > self.max_overall_dd:
            reward -= (overall_dd - self.max_overall_dd) * self.init_cash * 2  # Severe penalty for overall DD
            done = True
        if daily_pnl < -self.daily_loss_limit:
            reward -= abs(daily_pnl + self.daily_loss_limit) * self.init_cash
            done = True
        # Bonus for reaching profit target
        if self.equity >= self.init_cash * (1 + self.profit_target):
            reward += self.init_cash * 0.5  # Large bonus for passing challenge
            done = True
        # Bonus for profitable exits
        if self.position == 0 and self.prev_position != 0:  # Just exited a position
            if self.prev_position == 1:  # Exited long
                exit_price = price_open
                if hasattr(self, 'entry_price') and exit_price > self.entry_price:
                    profit = exit_price - self.entry_price
                    reward += profit * 1.0  # Moderate bonus proportional to profit to encourage profitable exits
                else:
                    # Small penalty for unprofitable exits to discourage bad timing
                    loss = self.entry_price - exit_price if hasattr(self, 'entry_price') else 0
                    reward -= loss * 0.2
            elif self.prev_position == -1:  # Exited short
                exit_price = price_open
                if hasattr(self, 'entry_price') and exit_price < self.entry_price:
                    profit = self.entry_price - exit_price
                    reward += profit * 1.0  # Moderate bonus proportional to profit to encourage profitable exits
                else:
                    # Small penalty for unprofitable exits to discourage bad timing
                    loss = exit_price - self.entry_price if hasattr(self, 'entry_price') else 0
                    reward -= loss * 0.2

        # Add penalty for holding positions too long without taking profit
        if self.position != 0:
            # Count how long we've been in this position (simplified approach)
            if hasattr(self, 'position_hold_count'):
                self.position_hold_count += 1
            else:
                self.position_hold_count = 1
            
            # Apply increasing penalty for very long holds (after 100 steps)
            if self.position_hold_count > 100:
                reward -= (self.position_hold_count - 100) * 0.001
        else:
            self.position_hold_count = 0

        self.idx += 1
        info = {'equity': self.equity, 'drawdown': dd, 'position': self.position}
        obs = self._get_full_obs()
        return obs, reward, done, info

    def render(self, mode='human'):
        pass 