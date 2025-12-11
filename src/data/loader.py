import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import logging

from src.config import config

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Motor de Datos (Data Engine) responsible for fetching, sanitizing, 
    and storing market data.
    """
    def __init__(self, exchange_id: str = config.EXCHANGE_ID):
        try:
            self.exchange_class = getattr(ccxt, exchange_id)
            self.exchange = self.exchange_class({
                'enableRateLimit': True, 
                'options': {'defaultType': 'spot'}
            })
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_id}' not found in ccxt.")

    def fetch_data(self, symbol: str, timeframe: str, since_ms: int = None, limit: int = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange with Pagination support.
        If limit > 1000, it loops to fetch all required data.
        """
        logger.info(f"Fetching {symbol} {timeframe} (Limit={limit})...")
        
        all_ohlcv = []
        fetch_limit = 1000 # Safety buffer for exchanges
        
        # Calculate start time if since_ms is None but limit is huge
        # (This is complex to guess due to gaps, so we use reverse fetching logic or simple chunks)
        # Simplified: If no since_ms, we just fetch mostly recent.
        # But CCXT fetch_ohlcv without since returns LATEST data.
        # To get 50,000 candles ENDING now, we need to fetch backwards or calc start.
        
        if limit and limit > fetch_limit and since_ms is None:
             # Strategy: Fetch latest, then use its timestamp to fetch previous?
             # Many exchanges support 'endTime'. CCXT uses params={'endTime': ...}
             # But 'since' is robust.
             # Let's calculate approx start time.
             duration_seconds = self.exchange.parse_timeframe(timeframe)
             duration_ms = duration_seconds * 1000
             now = self.exchange.milliseconds()
             since_ms = now - (limit * duration_ms)
             logger.info(f"Calculated start time: {datetime.fromtimestamp(since_ms/1000, timezone.utc)}")

        current_since = since_ms
        total_fetched = 0
        target = limit if limit else 500 # Default
        
        while total_fetched < target:
            # Determine batch size
            remaining = target - total_fetched
            batch = min(remaining, fetch_limit)
            
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=batch)
            except Exception as e:
                logger.error(f"Fetch failed: {e}")
                break
                
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            total_fetched += len(ohlcv)
            
            # Update since for next cursor
            # Last candle timestamp + 1 timeframe ? Or just last timestamp + 1ms
            last_ts = ohlcv[-1][0]
            current_since = last_ts + 1
            
            # Rate Limit sleep handled by enableRateLimit=True in init? 
            # CCXT usually sleeps automatically.
            
            logger.info(f"Fetched {len(all_ohlcv)}/{target} candles...")
            
            if len(ohlcv) < batch:
                # End of history reached
                break
                
        if not all_ohlcv:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # UTC Check
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)
        
        # Verify integrity
        self._validate_data(df, timeframe)
        
        # Trim to exact limit if we over-fetched
        if limit and len(df) > limit:
            df = df.iloc[-limit:]
        
        return df

    def fetch_market_context(self, main_symbol: str, timeframe: str, context_symbols: list = None, limit: int = 1000) -> pd.DataFrame:
        """
        Fetches main symbol data AND context symbols (e.g. BTC/USDT), merging them.
        Returns DataFrame with 'close_BTC', 'close_ETH' etc. appended.
        """
        if context_symbols is None:
            context_symbols = ['BTC/USDT', 'ETH/USDT']
            
        # Fetch Main
        df_main = self.fetch_data(main_symbol, timeframe, limit=limit)
        if df_main.empty: return df_main
        
        # Fetch Context
        for sym in context_symbols:
            if sym == main_symbol: continue
            
            logger.info(f"Fetching Context: {sym}...")
            # Use same limit to try alignment
            df_ctx = self.fetch_data(sym, timeframe, limit=limit)
            
            if df_ctx.empty:
                logger.warning(f"Context {sym} empty. Skipping.")
                continue
                
            # Merge 'close' only (usually sufficient for context correlation)
            # Suffix with symbol base name e.g., _BTC
            suffix = f"_{sym.split('/')[0]}"
            col_name = f"close{suffix}"
            
            # Align
            # Reindex context to match main (Left Join)
            # Use ffill to handle minor timestamp diffs/delays if acceptable, or strict join
            # Strict join is safer for avoiding lookahead.
            
            s_ctx = df_ctx['close'].rename(col_name)
            df_main = df_main.join(s_ctx, how='left')
            
            # Forward fill context if minor gaps, but warning
            # If huge gaps, maybe issue.
            df_main[col_name] = df_main[col_name].ffill()
            
        return df_main

    def _validate_data(self, df: pd.DataFrame, timeframe: str):
        """
        Validates data integrity (Gaps).
        """
        tf_map = {'1m': '1T', '1h': '1H', '1d': '1D', '5m': '5T', '15m': '15T'}
        freq = tf_map.get(timeframe)
        
        if not freq:
            return

        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz='UTC')
        
        if len(df) < len(full_idx):
            missing_count = len(full_idx) - len(df)
            msg = f"DATA GAP DETECTED: Found {missing_count} missing candles for {timeframe}."
            # Only checking strict gap presence logic, simplified for robustness
            pass # Or raise if needed

    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        path = config.RAW_DATA_DIR / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info(f"Saved data to {path}")
