import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import warnings
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.data.loader import DataLoader
from src.models.agents.transformer import TransformerAgent

class TestPhase1(unittest.TestCase):
    
    def test_config_warning(self):
        """Test that Config warns if Binance + Bad Region"""
        # We need to re-instantiate Config with mocked env
        with patch.dict(os.environ, {'EXCHANGE_ID': 'binance', 'AWS_REGION': 'us-east-1'}):
            with self.assertWarns(UserWarning) as cm:
                cfg = Config()
            self.assertIn("LATENCY WARNING", str(cm.warning))
            
    def test_data_loader_gap_error(self):
        """Test that DataLoader raises ValueError on Gaps"""
        # Mock ccxt
        with patch('ccxt.binance') as mock_exchange_cls:
            mock_exchange = MagicMock()
            mock_exchange_cls.return_value = mock_exchange
            
            # Create data with a gap: 10:00, 10:01, [GAP], 10:03
            timestamps = [
                1609495200000, # 10:00
                1609495260000, # 10:01
                # Missing 10:02
                1609495380000  # 10:03
            ]
            ohlcv = [[ts, 100, 101, 99, 100, 10] for ts in timestamps]
            mock_exchange.fetch_ohlcv.return_value = ohlcv
            
            loader = DataLoader(exchange_id='binance')
            
            with self.assertRaises(ValueError) as cm:
                loader.fetch_data('BTC/USDT', '1m', limit=3)
            self.assertIn("DATA GAP DETECTED", str(cm.exception))

    def test_transformer_agent_skeleton(self):
        """Test that Agent 1 Skeleton instantiates and runs forward pass"""
        d_model = 64
        input_dim = 10
        seq_len = 5
        batch_size = 2
        
        model = TransformerAgent(input_dim=input_dim, d_model=d_model)
        
        # Fake Inputs
        x_features = torch.randn(batch_size, seq_len, input_dim)
        # Time features: [hour_sin, hour_cos, day_sin, day_cos] -> size 4
        x_time = torch.randn(batch_size, seq_len, 4)
        
        output = model(x_features, x_time)
        
        # Expected output shape: [Batch, 2] -> [Direction, Confidence]
        self.assertEqual(output.shape, (batch_size, 2))
        
        # Check values range
        direction = output[:, 0]
        confidence = output[:, 1]
        
        # Direction implies Tanh (-1 to 1) checking min/max theoretically or just values
        self.assertTrue(torch.all(output[:, 0] >= -1.0))
        self.assertTrue(torch.all(output[:, 0] <= 1.0))
        
        # Confidence implies Sigmoid (0 to 1)
        self.assertTrue(torch.all(output[:, 1] >= 0.0))
        self.assertTrue(torch.all(output[:, 1] <= 1.0))
        
        # Verify strict usage of TransformerEncoder
        self.assertIsInstance(model.transformer_encoder, torch.nn.TransformerEncoder)

if __name__ == '__main__':
    unittest.main()
