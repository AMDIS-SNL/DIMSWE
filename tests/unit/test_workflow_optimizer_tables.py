"""Check that the workflow's pure-Python optimizer-method tables match dimswe.optimize.

workflows/optimize_params.py validates optimizer specs without importing Firedrake, so it keeps
its own copy of the method tables used by dimswe.optimize.resolve_hessp_use. This test (which
needs Firedrake, to import dimswe) keeps the two in sync.
"""

from dimswe import optimize as dimswe_optimize
from workflows import optimize_params


def test_hessp_method_tables_match():
    assert set(optimize_params.HESSP_METHODS) == set(dimswe_optimize.HESSP_METHODS)
    assert set(optimize_params.HESSP_REQUIRED_METHODS) == set(dimswe_optimize.HESSP_REQUIRED_METHODS)
    assert set(optimize_params.BOUNDED_METHODS) == set(dimswe_optimize.BOUNDED_METHODS)


def test_workflow_validation_agrees_with_runtime_checks():
    # every method/bounds combination the workflow accepts for the coeffs target is also accepted
    # by the runtime check in optimize() (has_hessp is always true for the workflow's optimizer)
    for method in optimize_params.KNOWN_METHODS:
        if method in optimize_params.FULL_HESSIAN_METHODS:
            continue
        for use_bounds in (True, False):
            spec = {**optimize_params.DEFAULT_OPTIMIZER, "method": method, "use_bounds": use_bounds}
            try:
                optimize_params.validate_optimizer_spec(spec, "coeffs", {})
            except ValueError:
                continue
            dimswe_optimize.resolve_hessp_use(method, None, True, has_hessp=True, has_bounds=use_bounds)
