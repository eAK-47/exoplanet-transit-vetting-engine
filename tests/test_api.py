"""Unit tests for FastAPI server module"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFastAPIServer:
    """Test FastAPI server initialization"""
    
    def test_server_import(self):
        """Test that FastAPI app can be imported"""
        try:
            from src.api.server import app
            assert app is not None
        except ImportError as e:
            pytest.fail(f"Failed to import FastAPI app: {e}")
    
    def test_app_routes(self):
        """Test that app has expected routes"""
        from src.api.server import app
        
        # Check that routes are defined
        routes = [route.path for route in app.routes]
        assert "/" in routes or "/docs" in routes
