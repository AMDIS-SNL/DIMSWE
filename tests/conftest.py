"""Set writable, process-local cache paths before test-module imports."""

import os
from pathlib import Path
import tempfile


cache_override = os.environ.get("DIMSWE_TEST_CACHE_DIR")
if cache_override:
    test_cache_root = Path(cache_override).expanduser()
    os.environ["PYOP2_CACHE_DIR"] = str(test_cache_root / "pyop2")
    os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] = str(test_cache_root / "tsfc")
else:
    test_cache_root = Path(
        tempfile.mkdtemp(prefix=f"dimswe-tests-{os.getpid()}-")
    )
    os.environ.setdefault("PYOP2_CACHE_DIR", str(test_cache_root / "pyop2"))
    os.environ.setdefault(
        "FIREDRAKE_TSFC_KERNEL_CACHE_DIR", str(test_cache_root / "tsfc")
    )
