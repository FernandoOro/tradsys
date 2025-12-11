
import logging
from abc import ABC, abstractmethod
from src.config import config

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """
    Abstract Base Class for Trading Strategies.
    Encapsulates the decision logic: "To Trade or Not To Trade".
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def should_trade(self, confidence: float, regime: int, auditor_vetoed: bool) -> bool:
        """
        Determines if a trade should be executed given the signal context.
        Args:
            confidence: Model confidence (0.0 to 1.0)
            regime: HMM State (0, 1, 2...)
            auditor_vetoed: True if Auditor rejects the trade
        Returns:
            bool: True (Execute) / False (Skip)
        """
        pass

class SniperStrategy(Strategy):
    """
    High Precision, Low Frequency.
    Threshold: 0.95
    Regime: Block State 0 (Range/Bear)
    Auditor: Ignored (Confidence is high enough)
    """
    def __init__(self):
        super().__init__("SNIPER")
        self.threshold = 0.95

    def should_trade(self, confidence: float, regime: int, auditor_vetoed: bool) -> bool:
        if regime == 0:
            return False
        if confidence < self.threshold:
            return False
        return True

class AuditedStrategy(Strategy):
    """
    Balanced Frequency & Safety.
    Threshold: 0.75
    Regime: Block State 0
    Auditor: MUST Approve (Veto = Block)
    """
    def __init__(self):
        super().__init__("AUDITED")
        self.threshold = 0.75

    def should_trade(self, confidence: float, regime: int, auditor_vetoed: bool) -> bool:
        if regime == 0:
            return False
        if confidence < self.threshold:
            return False
        if auditor_vetoed:
            return False
        return True

class RecklessStrategy(Strategy):
    """
    High Frequency, Higher Risk.
    Threshold: 0.75
    Regime: Block State 0
    Auditor: Ignored
    """
    def __init__(self):
        super().__init__("RECKLESS")
        self.threshold = 0.75

    def should_trade(self, confidence: float, regime: int, auditor_vetoed: bool) -> bool:
        if regime == 0:
            return False
        if confidence < self.threshold:
            return False
        return True

class StrategyFactory:
    @staticmethod
    def get_strategy(profile_name: str = None) -> Strategy:
        if profile_name is None:
            profile_name = config.STRATEGY_PROFILE
            
        if profile_name == "SNIPER":
            return SniperStrategy()
        elif profile_name == "AUDITED":
            return AuditedStrategy()
        elif profile_name == "RECKLESS":
            return RecklessStrategy()
        else:
            logger.warning(f"Unknown Strategy Profile '{profile_name}'. Defaulting to AUDITED.")
            return AuditedStrategy()
