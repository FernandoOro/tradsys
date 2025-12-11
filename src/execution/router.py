import logging
import time
import ccxt

logger = logging.getLogger(__name__)

class SmartRouter:
    """
    Executes orders minimizing fees (Maker Strategy).
    Falls back to Taker if liquidity is elusive.
    """
    def __init__(self, exchange, max_retries: int = 3, maker_timeout: int = 10):
        self.exchange = exchange
        self.max_retries = max_retries
        self.maker_timeout = maker_timeout # Seconds to wait for Maker fill

    def execute_order(self, symbol: str, side: str, amount: float, price_limit: float = None):
        """
        Executes a trade.
        Args:
            symbol: 'BTC/USDT'
            side: 'buy' or 'sell'
            amount: Quantity
            price_limit: Optional aggressive limit.
        """
        if amount <= 0:
            logger.warning("SmartRouter: Amount <= 0. Skipping.")
            return

        logger.info(f"SmartRouter: Executing {side.upper()} {amount} {symbol}...")
        
        # 1. Attempt Maker (Limit Order at Best Book)
        try:
            # Fetch Order Book to set price
            order_book = self.exchange.fetch_order_book(symbol)
            if side == 'buy':
                # Bid Side. To be Maker, place at Bid[0]. Price match.
                # Or slightly inside? "Best Bid".
                # Caution: If we place exactly at Best Bid, we join queue.
                target_price = order_book['bids'][0][0] 
            else:
                # Ask Side.
                target_price = order_book['asks'][0][0]
                
            logger.info(f"Placing LIMIT {side} at {target_price} (Maker Attempt)...")
            
            # Create Limit Order (Post-Only ideally, but simple limit for now)
            # params = {'timeInForce': 'PO'} if Binance
            order = self.exchange.create_order(symbol, 'limit', side, amount, target_price)
            order_id = order['id']
            
            # Wait loop
            filled = False
            start_wait = time.time()
            
            while (time.time() - start_wait) < self.maker_timeout:
                time.sleep(1)
                order_status = self.exchange.fetch_order(order_id, symbol)
                if order_status['status'] == 'closed':
                    logger.info("Order FILLED as Maker! (Saved Fees)")
                    filled = True
                    break
                    
            if not filled:
                logger.info("Maker timeout reached. Cancelling and switching to Taker...")
                try:
                    self.exchange.cancel_order(order_id, symbol)
                except Exception as e:
                    # Could have filled in the split second
                    logger.warning(f"Cancel failed (may be filled?): {e}")
                    # Re-check
                    order_status = self.exchange.fetch_order(order_id, symbol)
                    if order_status['status'] == 'closed':
                         return # Done
                
                # Check filled amount (Partial fill?)
                # Assuming check 'filled' qty
                # Remaining = Amount - filled
                # For simplicity here, assuming zero fill or we re-fetch logic
                # Fallback to MARKET
                self._execute_market(symbol, side, amount)
                
        except Exception as e:
            logger.error(f"Maker logic failed: {e}. Falling back to Taker immediately.")
            self._execute_market(symbol, side, amount)

    def _execute_market(self, symbol, side, amount):
        logger.info(f"Placing MARKET {side} for {amount} (Taker)...")
        try:
            self.exchange.create_order(symbol, 'market', side, amount)
            logger.info("Market Order Placed.")
        except Exception as e:
             logger.error(f"Market Order Failed: {e}")
             # Retry logic could go here
