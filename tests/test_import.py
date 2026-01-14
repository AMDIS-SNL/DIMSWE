import pytest


def test_import_core_modules():
    """
    Sanity check: ensure the core module imports without error.
    """
    try:
        import dimswe
        import dimswe.model
        import dimswe.dynamics
        import dimswe.physics
    except Exception as e:
        pytest.fail(f"Import failed: {e}")
