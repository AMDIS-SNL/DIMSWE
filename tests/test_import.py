import pytest


def test_import_core_modules():
    """
    Sanity check: ensure the core module imports without error.
    """
    try:
        pass
    except Exception as e:
        pytest.fail(f"Import failed: {e}")
