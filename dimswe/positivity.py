"""Positivity measures of densities: cell means and whole-cell lower bounds, and their domain extrema.

For every density, CellPositivity computes two DG0 fields:
- <dens>_cellmean: the cell mean;
- <dens>_cellmin: a lower bound of the density over the whole cell, its smallest Bernstein
  coefficient (exact for degree <= 1). Only available when the cells are axis-aligned boxes.
The diagnostics output these fields, and the statistics record their domain minima and maxima,
<dens>_cellmean_min, <dens>_cellmean_max, <dens>_cellmin_min and <dens>_cellmin_max.
"""
import numpy as np
from firedrake import (Function, FunctionSpace, LinearVariationalProblem, LinearVariationalSolver,
                       TestFunction, TrialFunction, inner)

from .bernstein import BernsteinCoefficients, field_degree, has_cell_box_geometry
from .parameters import overall_solver_parameters

MEASURES = ('cellmean', 'cellmin')


def positivity_field_name(density, measure):
    """Return the diagnostic name of a positivity field, for example Qc_cellmin."""
    return f'{density}_{measure}'


def positivity_statistic_name(density, measure, extremum):
    """Return the statistic name of a domain extremum, for example Qc_cellmin_min."""
    return f'{density}_{measure}_{extremum}'


class CellPositivity():
    """Cell means and whole-cell lower bounds (DG0 Functions) of the densities of a variable set."""

    def __init__(self, spaces, variableset, density_names, dx=None):
        self.spaces = spaces
        self.variableset = variableset
        self.density_names = list(density_names)
        self.dx = dx
        # without spaces (no mesh yet) the names are listed as if box cells were available
        self.whole_cell_minimum = spaces is None or has_cell_box_geometry(spaces)
        self.measures = MEASURES if self.whole_cell_minimum else ('cellmean',)
        self.fields = {}
        if spaces is not None:
            self.DG0 = FunctionSpace(spaces.mesh, 'DG', 0)
            for density in self.density_names:
                for measure in self.measures:
                    name = positivity_field_name(density, measure)
                    self.fields[name] = Function(self.DG0, name=name)

    def field_names(self):
        """Return the names of the positivity fields, in output order."""
        return [positivity_field_name(density, measure) for density in self.density_names for measure in self.measures]

    def statistic_names(self):
        """Return the names of the domain extrema of the positivity fields."""
        return [positivity_statistic_name(density, measure, extremum)
                for density in self.density_names for measure in self.measures for extremum in ('min', 'max')]

    def density_space(self, density):
        """Return the function space of a density of the variable set."""
        return self.variableset.spacelist[self.variableset.varlist.index(density)]

    def create(self, xn):
        """Set up the solvers computing the positivity fields of the densities in xn.

        With box cells, the Bernstein coefficients of each density are projected onto a DG0 vector
        space (a diagonal solve); the cell mean is their average and the lower bound their minimum.
        Otherwise the cell mean is the projection onto DG0.
        """
        self.solvers = {}
        self.coefficients = {}
        for density in self.density_names:
            if self.whole_cell_minimum:
                bernstein = BernsteinCoefficients(self.spaces, field_degree(self.density_space(density)))
                coefficients = Function(bernstein.space, name=density + '_bernstein')
                lhs, rhs = bernstein.coefficient_expressions(xn[density], TestFunction(bernstein.space), TrialFunction(bernstein.space))
                problem = LinearVariationalProblem(lhs, rhs, coefficients)
                self.coefficients[density] = (bernstein, coefficients)
            else:
                meanhat, meantrial = TestFunction(self.DG0), TrialFunction(self.DG0)
                problem = LinearVariationalProblem(inner(meanhat, meantrial)*self.dx, inner(meanhat, xn[density])*self.dx,
                                                   self.fields[positivity_field_name(density, 'cellmean')])
            self.solvers[density] = LinearVariationalSolver(problem, solver_parameters=overall_solver_parameters['positivitydiag'], options_prefix='positivitydiag')

    def compute(self):
        """Compute the positivity fields from the current state."""
        for density in self.density_names:
            self.solvers[density].solve()
            if density in self.coefficients:
                bernstein, coefficients = self.coefficients[density]
                self.fields[positivity_field_name(density, 'cellmean')].interpolate(bernstein.mean(coefficients))
                self.fields[positivity_field_name(density, 'cellmin')].interpolate(bernstein.minimum(coefficients))

    def domain_extrema(self):
        """Return {statistic name: value} with the domain min and max of every positivity field."""
        from mpi4py import MPI
        comm = self.spaces.mesh.comm
        extrema = {}
        for density in self.density_names:
            for measure in self.measures:
                values = self.fields[positivity_field_name(density, measure)].dat.data_ro
                local_min = float(np.min(values)) if values.size else np.inf
                local_max = float(np.max(values)) if values.size else -np.inf
                extrema[positivity_statistic_name(density, measure, 'min')] = comm.allreduce(local_min, op=MPI.MIN)
                extrema[positivity_statistic_name(density, measure, 'max')] = comm.allreduce(local_max, op=MPI.MAX)
        return extrema
