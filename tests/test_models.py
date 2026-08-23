"""Unit tests for models inference module"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSwinTransitClassifier:
    """Test SwinTransitClassifier model initialization"""
    
    def test_swin_classifier_import(self):
        """Test that SwinTransitClassifier can be imported
        
        Skipped if PyTorch is not available (expected in main CI).
        """
        try:
            from src.models.inference import SwinTransitClassifier
            assert SwinTransitClassifier is not None
        except ImportError as e:
            if "torch" in str(e).lower():
                pytest.skip("PyTorch not available (expected in main CI, run gpu-test.yml for full model testing)")
            else:
                pytest.fail(f"Failed to import SwinTransitClassifier: {e}")
    
    @pytest.mark.gpu
    def test_inference_engine_import(self):
        """Test that VikramadithyaInferenceEngine can be imported
        
        Marked as GPU test because it depends on PyTorch imports.
        Skipped in main CI, runs in gpu-test.yml.
        """
        try:
            from src.models.inference import VikramadithyaInferenceEngine
            assert VikramadithyaInferenceEngine is not None
        except ImportError as e:
            if "torch" in str(e).lower():
                pytest.skip("PyTorch not available (this is a GPU test, run gpu-test.yml)")
            else:
                pytest.fail(f"Failed to import VikramadithyaInferenceEngine: {e}")
