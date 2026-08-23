"""Unit tests for models inference module"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSwinTransitClassifier:
    """Test SwinTransitClassifier model initialization"""
    
    def test_swin_classifier_import(self):
        """Test that SwinTransitClassifier can be imported"""
        try:
            from src.models.inference import SwinTransitClassifier
            assert SwinTransitClassifier is not None
        except ImportError as e:
            pytest.fail(f"Failed to import SwinTransitClassifier: {e}")
    
    @pytest.mark.gpu
    def test_inference_engine_import(self):
        """Test that VikramadithyaInferenceEngine can be imported
        
        Marked as GPU test because it depends on PyTorch imports.
        """
        try:
            from src.models.inference import VikramadithyaInferenceEngine
            assert VikramadithyaInferenceEngine is not None
        except ImportError as e:
            pytest.fail(f"Failed to import VikramadithyaInferenceEngine: {e}")
