"""Dummy tester for training without evaluation"""
from src.testers.base_tester import BaseTester
import torch
import numpy as np

class DummyTester(BaseTester):
    """A dummy tester that does nothing - allows training without evaluation"""
    
    def __init__(self):
        pass
    
    def test(self, dataloader, net):
        """Return dummy metrics"""
        return {
            'translation_error': 0.0,
            'rotation_error': 0.0,
            'total_loss': 0.0
        }
    
    def compute_trajectory(self, dataloader, net):
        """Return empty trajectory"""
        return np.zeros((1, 6))
    
    def save_results(self, results, output_dir):
        """Dummy save function"""
        pass