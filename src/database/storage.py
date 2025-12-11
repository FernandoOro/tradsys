from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging
from pathlib import Path
from src.config import config
from src.database.models import Base, Signal, Trade
import json

logger = logging.getLogger(__name__)

class StorageEngine:
    def __init__(self):
        self.db_path = config.get_db_path()
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Database connected at {self.db_path}")

    def store_signal(self, symbol: str, direction: float, confidence: float, features: dict, vetoed: bool = False):
        session = self.Session()
        try:
            signal = Signal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                features_json=json.dumps(features) if features else "{}",
                vetoed=vetoed
            )
            session.add(signal)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
        finally:
            session.close()

    def store_trade(self, symbol: str, side: str, size: float, price: float, status: str = "OPEN"):
        session = self.Session()
        try:
            trade = Trade(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=price,
                status=status
            )
            session.add(trade)
            session.commit()
            logger.info(f"Stored Trade: {side} {size} {symbol} @ {price}")
        except Exception as e:
            logger.error(f"Failed to store trade: {e}")
        finally:
            session.close()
            
    def get_trades(self, limit=100):
        session = self.Session()
        trades = session.query(Trade).order_by(Trade.timestamp.desc()).limit(limit).all()
        session.close()
        return trades
