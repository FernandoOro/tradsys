import logging
import numpy as np

from src.config import config

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Smart Risk Management Module.
    Enforces Golden Rules for Position Sizing and Limits.
    """
    
    def __init__(self, target_risk_per_trade: float = None, max_leverage: float = None):
        # Use Config if not provided
        self.target_risk = target_risk_per_trade if target_risk_per_trade is not None else config.MAX_RISK_PER_TRADE
        self.max_leverage = max_leverage if max_leverage is not None else config.MAX_LEVERAGE
        self.stop_loss_atr_mult = config.STOP_LOSS_ATR_MULT

    def calculate_position_size(self, equity: float, predicted_atr: float, confidence: float, current_price: float) -> float:
        """
        Calculates position size based on Volatility and Confidence.
        Formula: Size = (Equity * TargetRisk) / (ATR * ConfidenceFactor)
        
        However, standard Kelly or Volatility sizing usually puts ATR in denominator relative to Price for Stop Loss distance.
        Risk Amount = Equity * TargetRisk
        Distance to Stop = ATR * k
        Size = Risk Amount / Distance to Stop
        
        Let's perform a robust calculation:
        1. Determine Stop Loss Distance based on ATR.
        2. Calculate Amount to Risk.
        3. Determine Size.
        4. Scale by Confidence [0, 1].
        """
        if confidence < 0.5:
             # Too low confidence, no trade
             return 0.0
             
        # Risk Amount (e.g. 1000 * 0.02 = $20 risk)
        risk_amount = equity * self.target_risk
        
        # Stop Distance (e.g. 2 * ATR)
        stop_distance = predicted_atr * self.stop_loss_atr_mult 
        if stop_distance == 0:
            return 0.0
            
        # Raw Size in Asset
        # If stop_distance is price diff ($100), Size = $20 / $100 = 0.2 BTC
        size_asset = risk_amount / stop_distance
        
        # Apply Confidence Scaling (Linear: 0.5->0, 1.0->1.0? Or just multiplier?)
        # Let's say Confidence 1.0 = Full Size. Confidence 0.5 = 0 size.
        # Scale = (Conf - 0.5) * 2
        scale_factor = max(0.0, (confidence - 0.5) * 2)
        
        final_size_asset = size_asset * scale_factor
        
        # Hard Limit Check: Max Leverage
        max_position_value_usd = equity * self.max_leverage
        final_position_value_usd = final_size_asset * current_price
        
        if final_position_value_usd > max_position_value_usd:
            logger.warning(f"Risk Manager: Cap at Max Leverage. Req: {final_position_value_usd:.2f}, Limit: {max_position_value_usd:.2f}")
            final_size_asset = max_position_value_usd / current_price
            
        return final_size_asset

    def check_circuit_breaker(self, daily_pnl_pct: float) -> bool:
        """
        Returns True if trading should STOP.
        Rule: Daily PnL < -3% -> Stop 24h.
        """
        if daily_pnl_pct < -0.03:
            logger.critical("CIRCUIT BREAKER TRIGGERED: Daily PnL < -3%. HALTING.")
            return True
        return False
