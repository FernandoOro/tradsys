import ccxt
import pandas as pd
import logging
import os
import time
from datetime import datetime, timezone
from src.config import config

logger = logging.getLogger(__name__)

class BinancePaperExecutor:
    """
    Handles interaction with Binance Testnet for Paper Trading.
    SAFE: Does not touch real funds (uses testnet URLs).
    """
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h"):
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Testnet Credentials
        api_key = os.getenv("BINANCE_TESTNET_KEY")
        secret = os.getenv("BINANCE_TESTNET_SECRET")
        
        if not api_key or not secret:
            logger.warning("⚠️ BINANCE_TESTNET_KEY/SECRET missing in .env! Read-Only Mode.")
        
        default_type = 'future' if config.EXCHANGE_ID in ['binanceusdm', 'binance_futures'] else 'spot'
        
        # FIX: If Futures (Real API for Data, Sim for Exec), DO NOT pass Testnet Keys.
        # Real API rejects Testnet Keys with "Invalid Api-Key ID".
        # Public data (OHLCV) does not require auth.
        if default_type == 'future':
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': default_type, 
                }
            })
        else:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': default_type, 
                }
            })
        
        # ACTIVATE TESTNET SANDBOX (Spot Only)
        if default_type == 'spot':
             self.exchange.set_sandbox_mode(True) 
        else:
             logger.warning("FUTURES SANDBOX NOT SUPPORTED. Using INTERNAL SIMULATION for execution.")

    def fetch_recent_metrics(self, limit=1000) -> pd.DataFrame:
        """
        Fetches the last N candles for BTC and ETH (Context).
        """
        try:
            # 1. Fetch Main Symbol (BTC)
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 2. Fetch Context Symbol (ETH)
            ohlcv_eth = self.exchange.fetch_ohlcv("ETH/USDT", self.timeframe, limit=limit)
            df_eth = pd.DataFrame(ohlcv_eth, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_eth['timestamp'] = pd.to_datetime(df_eth['timestamp'], unit='ms')
            df_eth.set_index('timestamp', inplace=True)
            
            # 3. Merge Context
            # We only need close_ETH for correlation features
            df['close_ETH'] = df_eth['close']
            
            # Reset index to keep 'timestamp' as column if needed, 
            # but FeatureEngineer.run() expects DatetimeIndex or handles it?
            # run_paper_1h.py sets index manually. Let's return with index or column?
            # Usage in run_paper_1h.py:
            # if 'timestamp' in df.columns: df = df.set_index('timestamp')
            # So returning with index is fine, or reset.
            # Let's reset to match original signature which returned 'timestamp' col.
            df.reset_index(inplace=True)

            # Add synthetic Time Features (required by model)
            # (Wait, FeatureEngineer adds these too. But we had them here originally)
            # Let's keep them here as safeguard or remove if FE adds them.
            # FE adds them in add_time_features. Redundant but harmless.
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def get_balance_usdt(self):
        # MOCK for Futures (Sim)
        if self.exchange.options['defaultType'] == 'future':
             return 10000.0 # Mock $10k
             
        try:
            bal = self.exchange.fetch_balance()
            return bal['total']['USDT']
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    def execute_order(self, side: str, amount_usdt: float):
        """
        Executes a MARKET order on Testnet.
        """
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']
            amount_btc = amount_usdt / price
            
            logger.info(f"🚀 EXECUTING {side.upper()} {amount_btc:.6f} BTC (~${amount_usdt:.2f}) on TESTNET...")
            
            # MOCK for Futures (Sim)
            if self.exchange.options['defaultType'] == 'future':
                logger.info("✅ SIMULATED Futures Order Executed (Internal).")
                return {
                    'id': f'sim_{int(time.time())}',
                    'price': price,
                    'amount': amount_btc,
                    'status': 'closed',
                    'filled': amount_btc
                }

            order = self.exchange.create_order(
                symbol=self.symbol,
                type='market',
                side=side,
                amount=amount_btc
            )
            logger.info(f"✅ Order Filled: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"❌ Execution Failed: {e}")
            return None

# Need numpy for sin/cos
import numpy as np
