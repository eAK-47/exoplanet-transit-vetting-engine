"""Unit tests for pipeline ingestion module"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTESSIngestionEngine:
    """Test TESSIngestionEngine initialization and basic functionality"""
    
    def test_ingestion_engine_import(self):
        """Test that TESSIngestionEngine can be imported"""
        try:
            from src.pipeline.ingestion import TESSIngestionEngine
            assert TESSIngestionEngine is not None
        except ImportError as e:
            pytest.fail(f"Failed to import TESSIngestionEngine: {e}")
    
    def test_ingestion_engine_init(self):
        """Test TESSIngestionEngine initialization"""
        from src.pipeline.ingestion import TESSIngestionEngine
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = TESSIngestionEngine(cache_dir=tmpdir)
            assert engine is not None
            assert engine.cache_dir == tmpdir
            engine.close()
