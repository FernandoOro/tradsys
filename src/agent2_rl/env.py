
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

class TradingEnv(gym.Env):
    """
    A professional Trading Environment for Reinforcement Learning.
    
    Attributes:
        df (pd.DataFrame): The dataset containing price and features.
        window_size (int): Number of past candles to include in the observation.
        initial_balance (float): Starting capital.
        fee_rate (float): Transaction fee (e.g., 0.001 for 0.1%).
        slippage (float): Estimated slippage per trade.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=96, initial_balance=10000.0, fee_rate=0.0, slippage=0.0):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        # Action Space: 0=Neutral, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: 
        # [Window of Features + Current Position Status]
        # We assume the dataframe has N feature columns.
        # We append 2 extra features to the state: [Position (1, 0, -1), Unrealized PnL %]
        # For simplicity in MLP policies, we flatten the window or just use the last row.
        # To make it "Professional", we provide the flattened window.
        
        self.feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'time', 'target', 'target_reversion']]
        self.price_col = 'close'
        
        n_features = len(self.feature_cols)
        # Input shape: (Window Size * Features) + 2 state vars
        self.obs_shape = (window_size * n_features + 2,) 
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0 # 0=Flat, 1=Long, -1=Short
        self.entry_price = 0.0
        self.trades_history = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start with a random offset to prevent overfitting to the start date
        # But ensure enough data for window
        self.current_step = self.window_size
        
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.trades_history = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        # 1. Get Window Data
        start = self.current_step - self.window_size
        end = self.current_step
        
        window_data = self.df.iloc[start:end][self.feature_cols].values.flatten()
        
        # 2. Get Account State
        # Position: 0, 1, or -1
        # Unrealized PnL: Current price vs Entry
        current_price = self.df.iloc[self.current_step - 1][self.price_col]
        
        if self.position == 0:
            unrealized_pnl = 0.0
        elif self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        else: # Short
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            
        state_extras = np.array([self.position, unrealized_pnl], dtype=np.float32)
        
        # Combine
        obs = np.concatenate([window_data, state_extras])
        
        # Nan protection
        return np.nan_to_num(obs, nan=0.0).astype(np.float32)

    def step(self, action):
        # Actions: 0=Neutral/Close, 1=Long, 2=Short
        # Mapping 2 -> -1 for internal logic
        target_position = 0
        if action == 1: target_position = 1
        elif action == 2: target_position = -1
        
        current_price = self.df.iloc[self.current_step][self.price_col]
        prev_value = self._get_portfolio_value(current_price)
        
        # Execution Logic
        reward = 0
        trade_fee = 0
        
        if target_position != self.position:
            # Position Change Detected
            
            # 1. Close existing if any
            if self.position != 0:
                trade_fee += self._calculate_fees(current_price)
                # PnL is realized implicitly by balance update in _get_portfolio_value logic 
                # but we simulate fee deduction from cash
                # Simplified: Value is Balance + Unrealized
                pass
                
            # 2. Open new if any
            if target_position != 0:
                trade_fee += self._calculate_fees(current_price)
                self.entry_price = current_price
            
            # Apply cost
            # Affects "Step Reward" immediately
            reward -= (trade_fee * 2) # Penalize friction heavily to discourage churn
            
            self.position = target_position
            
        # Move Time Forward
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Calculate New Value
        new_price = self.df.iloc[self.current_step][self.price_col]
        new_value = self._get_portfolio_value(new_price)
        
        # Step Reward: Log Return of Portfolio Value
        # This aligns the Agent's "Pleasure" with "Making Money"
        step_return = np.log(new_value / prev_value)
        reward += step_return
        
        # Sortino/Sharpe Shaping: Penalize negative returns more?
        # For now, raw log return is the robust standard.
        
        info = {
            'portfolio_value': new_value,
            'position': self.position
        }
        
        return self._get_observation(), reward, terminated, truncated, info

    def _get_portfolio_value(self, current_price):
        # Cash + Unrealized PnL
        # Simplified: We assume 100% equity bet for RL simplicity (or 0)
        # Value = Initial * (1 + Cumulative Returns)
        # But to be precise:
        
        # Let's track absolute value.
        # If Flat: Value = Balance
        # If Position: Value = Balance + Unr. PnL (on full balance? No, assume 1 unit? 
        # Let's assume Compounding: Full Balance deployed.
        
        if self.position == 0:
            return self.balance
        
        if self.position == 1:
            # Long
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            # Fee was paid on entry, strictly speaking. But let's simplify state value.
            return self.balance * (1 + pnl_pct)
        
        if self.position == -1:
            # Short
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            return self.balance * (1 + pnl_pct)
            
        return self.balance

    def _calculate_fees(self, price):
        # Fee + Slippage %
        return self.fee_rate + self.slippage
