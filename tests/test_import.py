import pytest

def test_import_dimswe():
    """
    Sanity check: ensure the core module imports without error.
    """
    try:
        import src  # adjust to your actual module name
    except Exception as e:
        pytest.fail(f"Import failed: {e}")
