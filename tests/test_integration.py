"""Integration tests for INS Vikramadithya system"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestModuleIntegration:
    """Test that all modules can be imported and initialized together"""
    
    @pytest.mark.integration
    def test_all_modules_import(self):
        """Test that all core modules can be imported successfully"""
        try:
            from src.pipeline.ingestion import TESSIngestionEngine
            from src.models.inference import SwinTransitClassifier, VikramadithyaInferenceEngine
            from src.api.server import app
            
            assert TESSIngestionEngine is not None
            assert SwinTransitClassifier is not None
            assert VikramadithyaInferenceEngine is not None
            assert app is not None
            
            print("✓ All modules imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import modules: {e}")
    
    @pytest.mark.integration
    def test_package_structure(self):
        """Test that package structure is correct"""
        import src
        import src.pipeline
        import src.models
        import src.api
        
        assert hasattr(src, "__file__")
        assert Path(src.__file__).parent.name == "src"
