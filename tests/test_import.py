import pytest

def test_import_core_modules():
    """
    Sanity check: ensure the core module imports without error.
    """
    try:
      import dimswe
      import dimswe.models
      import dimswe.timestepping
      import dimswe.parameters
      import dimswe.logger
      import dimswe.output
      import dimswe.dynamics

      import dimswe.dissipation
      import dimswe.physics
      import dimswe.transport_operators
      import dimswe.entropies
      import dimswe.hamiltonians
      import dimswe.initial_conditions
      import dimswe.meshes
      import dimswe.metric_brackets
     # import dimswe.poisson_brackets
      import dimswe.variables
      import dimswe.diagnostics
      import dimswe.statistics
      import dimswe.plot_adv_dens

    except Exception as e:
        pytest.fail(f"Import failed: {e}")
