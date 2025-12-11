import ccxt
import os
import sys
import logging
import json
from dotenv import load_dotenv

# Load env
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PANIC")

def get_exchange():
    exchange_id = os.getenv("EXCHANGE_ID", "binance")
    api_key = os.getenv("API_KEY")
    secret = os.getenv("SECRET_KEY")
    
    if not api_key or not secret:
        logger.error("API Keys missing.")
        sys.exit(1)
        
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True
    })
    return exchange

def panic_button():
    logger.critical("!!! PANIC BUTTON ACTIVATED !!!")
    logger.critical("INITIATING EMERGENCY LIQUIDATION SEQUENCE...")
    
    try:
        exchange = get_exchange()
        
        # 1. Cancel All Orders
        logger.info("Step 1: Cancelling ALL Open Orders...")
        try:
            exchange.cancel_all_orders()
            logger.info("✅ All orders cancelled.")
        except Exception as e:
            logger.error(f"❌ Failed to cancel orders: {e}")
            
        # 2. Fetch Positions and Close
        logger.info("Step 2: Closing ALL Positions...")
        try:
            # Different logic for Spot vs Futures.
            # Assuming Spot: Check Balances and Sell everything to USDT
            # Assuming Futures: Check Positions and Close
            
            # Simple Spot Logic for "Smart Spot"
            balance = exchange.fetch_balance()
            total_balance = balance['total']
            
            for currency, amount in total_balance.items():
                if currency == 'USDT' or amount <= 0:
                    continue
                
                # Check min notional / precise amount
                # Simplistic Market Sell
                symbol = f"{currency}/USDT"
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    price = ticker['last']
                    value = amount * price
                    if value > 10: # Min notion usually $10
                        logger.info(f"Selling {amount} {currency} (Value: ${value:.2f})...")
                        exchange.create_market_sell_order(symbol, amount)
                        logger.info(f"✅ Sold {currency}.")
                    else:
                        logger.info(f"Skipping {currency} (Dust).")
                except Exception as ex:
                    logger.error(f"Failed to sell {currency}: {ex}")

        except Exception as e:
            logger.error(f"❌ Failed to close positions: {e}")
            
    except Exception as e:
        logger.critical(f"FATAL ERROR in Panic Script: {e}")
    
    logger.critical("PANIC SEQUENCE COMPLETED. SHUTTING DOWN.")
    sys.exit(1)

if __name__ == "__main__":
    confirm = input("ARE YOU SURE YOU WANT TO LIQUIDATE EVERYTHING? (type 'YES'): ")
    if confirm == "YES":
        panic_button()
    else:
        print("Cancelled.")
