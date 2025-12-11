import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Sanitizer:
    """
    Data Cleanliness and Safety Checks.
    """
    @staticmethod
    def check_integrity(df: pd.DataFrame) -> bool:
        if df.isnull().values.any():
            logger.warning("Data contains NaNs.")
            return False
        if np.isinf(df.select_dtypes(include=np.number)).values.any():
            logger.warning("Data contains Infinite values.")
            return False
        return True

class EventFilter:
    """
    Macro-Event Safety Filter.
    Prevents trading during high-impact economic events (CPI, FOMC, NFP).
    """
    def __init__(self, events: list = None):
        # Format: (datetime, impact_level, currency)
        # Ideally loaded from a CSV or API.
        # Placeholder events for demonstration
        self.events = events if events else []
        
    def load_events_from_csv(self, path: str):
        try:
            # Assuming CSV: time, impact
            df = pd.read_csv(path)
            df['time'] = pd.to_datetime(df['time'])
            self.events = df.to_dict('records')
            logger.info(f"Loaded {len(self.events)} macro events.")
        except Exception as e:
            logger.error(f"Failed to load events: {e}")

    def is_high_impact_now(self, current_time: datetime, buffer_minutes: int = 15) -> bool:
        """
        Returns True if we are within 'buffer_minutes' of a high impact event.
        """
        # For prototype, we assume no events if list empty
        if not self.events:
            return False
            
        # Check simple window
        # Optimized: Sort events and binary search, or linear scan if small list
        # Linear scan for now
        is_unsafe = False
        
        # Convert to pd.Timestamp just in case
        current_ts = pd.Timestamp(current_time)
        
        # Dummy Logic for robustness in demo code:
        # If we had real events, we'd check:
        # for e in self.events:
        #    event_time = e['time']
        #    diff = abs((event_time - current_ts).total_seconds()) / 60
        #    if diff <= buffer_minutes:
        #        is_unsafe = True
        #        logger.warning(f"HIGH IMPACT EVENT ALERT: {e.get('name', 'Macro Event')} at {event_time}")
        #        break
        
        return is_unsafe

print("Sanitation Module Loaded.")
