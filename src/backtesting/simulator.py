import vectorbt as vbt
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Simulator:
    """
    Backtesting Engine using vectorbt.
    """
    def __init__(self, fees: float = 0.001, slippage: float = 0.0005):
        self.fees = fees
        self.slippage = slippage
        
    def run_backtest(self, close_price: pd.Series, signals: pd.Series):
        """
        Runs a simulation based on signals (-1, 0, 1) or boolean entries.
        Args:
           close_price: Series of close prices.
           signals: Series of signals (1=Long, -1=Short, 0=Neutral/Halt).
        """
        # vectorbt 'from_signals' usually needs entries/exits booleans or size.
        # Let's clean signals first.
        
        # Entries: Signal == 1
        entries = signals == 1
        
        # Exits: Signal == -1 (Short) -> For Spot we just Sell. 
        # If Long/Short strategy, we use 'short_entries'.
        # Context says "Smart Spot Trading", so Shorting might naturally just mean selling to cash.
        exits = signals == -1 # OR signals == 0 if we want to exit on neutral.
        # Let's assume -1 means SELL everything.
        
        logger.info("Running VectorBT Backtest...")
        
        portfolio = vbt.Portfolio.from_signals(
            close_price,
            entries,
            exits,
            fees=self.fees,
            slippage=self.slippage,
            init_cash=10000,
            freq='1h' # Assumption
        )
        
        stats = portfolio.stats()
        logger.info("Backtest Results:")
        logger.info(f"Total Return: {stats['Total Return [%]']:.2f}%")
        logger.info(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        logger.info(f"Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
        
        return portfolio, stats

if __name__ == "__main__":
    # Smoke test
    logging.basicConfig(level=logging.INFO)
    price = pd.Series(np.random.randn(100) + 100, index=pd.date_range("2021-01-01", periods=100, freq='H'))
    signals = pd.Series(np.random.choice([0, 1, -1], 100), index=price.index)
    sim = Simulator()
    sim.run_backtest(price, signals)
