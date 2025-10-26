"""
Pytest configuration to skip problematic tests on macOS.
"""
import pytest
import sys
import platform

def pytest_configure(config):
    """Configure pytest to skip Transformers tests on macOS."""
    if platform.system() == "Darwin":  # macOS
        config.addinivalue_line(
            "markers", "skip_macos: mark test to skip on macOS due to Transformers MPS issues"
        )

def pytest_collection_modifyitems(config, items):
    """Skip Transformers tests on macOS."""
    if platform.system() == "Darwin":  # macOS
        skip_macos = pytest.mark.skip(reason="Skipped on macOS due to Transformers MPS bus error")
        for item in items:
            # Skip tests that use Transformers (bot detection API tests)
            if "test_full_analysis_integration" in item.name or \
               "test_fast_analysis_only" in item.name or \
               "test_detailed_integration_with_output" in item.name:
                item.add_marker(skip_macos)
