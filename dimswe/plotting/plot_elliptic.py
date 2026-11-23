"""Plots for elliptic Hodge Laplacian runs (TODO-elliptic §4.8).

- plot_elliptic(parameters): static plots of the fields in an elliptic run's sim.h5:
  1D line plots (u_h overlaid with u_exact when known, sigma, p, d u, errors); 2D scalar plots for
  scalar proxies (k = 0, 2) and quiver + magnitude plots for vector proxies (1-forms); 3D runs
  are VTK-only (no static plots).
- plot_convergence(summary): log-log error-vs-h plot of an elliptic_convergence.py summary with
  reference slopes.
- plot_elliptic_eigen(parameters): eigenvalue-index plot (computed vs exact) and mode plots
  (1D/2D) of an elliptic-eigen run.

Which fields exist and whether they are scalar or vector follows from the form degrees: sigma is
a (k-1)-form, u, p, u_exact and u_error are k-forms, sigma_error a (k-1)-form, du a (k+1)-form;
0- and n-forms have scalar proxies, all others vector proxies.
"""
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from firedrake import CheckpointFile, Function, FunctionSpace, inner, sqrt

from ..meshes import get_mesh_type_dimension
from .plotting import plot_scalar1D, plot_scalar2D, plot_vector2D_quiver

# Chart tokens (light mode) of the reference data-viz palette. Series colors follow the error
# quantity, in a fixed categorical order, whichever subset is present.
SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
TEXT_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
AXIS = '#c3c2b7'
ERROR_SERIES = {
    # name: (label, color, marker); markers are a secondary encoding next to color.
    'u_error': (r'$\|u - u_h\|$', '#2a78d6', 'o'),
    'sigma_error': (r'$\|\sigma - \sigma_h\|$', '#eb6834', 's'),
    'du_error': (r'$\|d(u - u_h)\|$', '#1baf7a', '^'),
    'dsigma_error': (r'$\|d(\sigma - \sigma_h)\|$', '#eda100', 'D'),
    'p_error': (r'$\|p - p_h\|$', '#e87ba4', 'v'),
}
# Expected L2 rate per error, as a function of (order r, form degree k) (TODO-elliptic §6).
EXPECTED_RATE = {
    'u_error': lambda r, k: r + 1 if k == 0 else r,
    'sigma_error': lambda r, k: r,
    'du_error': lambda r, k: r,
    'dsigma_error': lambda r, k: r,
}


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------

def elliptic_field_degrees(k):
    """Form degree of every field an elliptic run can write (see the module docstring)."""
    return {'sigma': k - 1, 'u': k, 'p': k, 'u_exact': k, 'u_error': k, 'sigma_error': k - 1, 'du': k + 1}


def is_vector_field(name, k, dim):
    """True if the proxy of field name is a vector (degree strictly between 0 and dim)."""
    degree = elliptic_field_degrees(k)[name]
    return 0 < degree < dim


def elliptic_fields(k, dim, has_harmonic, has_exact):
    """Names of the fields written by run_elliptic, in plotting order."""
    names = ['sigma'] if k > 0 else []
    names.append('u')
    if has_harmonic:
        names.append('p')
    if has_exact:
        names += ['u_exact', 'u_error'] + (['sigma_error'] if k > 0 else [])
    if k < dim:
        names.append('du')
    return names


def expected_rates(order, k):
    """Expected L2 rates {error name: rate} for spatial order r and form degree k."""
    return {name: rule(order, k) for name, rule in EXPECTED_RATE.items()}


def convergence_series(rows):
    """{error name: (nx list, error list)} with only positive, finite errors, in ERROR_SERIES order."""
    series = {}
    for name in ERROR_SERIES:
        points = [(row['nx'], row.get(name)) for row in rows if row.get(name) is not None]
        points = [(nx, e) for nx, e in points if e > 0.0 and math.isfinite(e)]
        if points:
            series[name] = ([p[0] for p in points], [p[1] for p in points])
    return series


# ---------------------------------------------------------------------------------------------
# Field plots
# ---------------------------------------------------------------------------------------------

def _style_axes(axes):
    """Recessive axes: hairline solid grid, muted spines and tick labels, light surface."""
    axes.set_facecolor(SURFACE)
    axes.grid(True, which='major', color=GRIDLINE, linewidth=0.8, linestyle='-')
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_color(AXIS)
    axes.tick_params(colors=TEXT_SECONDARY, which='both')
    axes.xaxis.label.set_color(TEXT_SECONDARY)
    axes.yaxis.label.set_color(TEXT_SECONDARY)
    axes.title.set_color(TEXT_PRIMARY)


def _plot_magnitude(func, base):
    """Plot |func| of a 2D vector proxy as base + '-mag.png'.

    The magnitude is interpolated into scalar DG: Firedrake's own vector tripcolor goes through
    the element's sub-elements, which RTCE/RTCF (non-vector elements) do not have.
    """
    degree = max(1, int(getattr(func.function_space().ufl_element(), 'embedded_superdegree', 2)))
    magnitude = Function(FunctionSpace(func.function_space().mesh(), 'DG', degree))
    magnitude.interpolate(sqrt(inner(func, func)))
    plot_scalar2D(magnitude, base + '-mag')


def _sample_1d(funcs, num_points=400):
    """Sample 1D Functions at evenly spaced points (point evaluation works for any element).

    Returns (x, [values per function]). Points stay a hair inside the mesh, so periodic meshes
    (whose last point coincides with the first) and round-off at the ends do not matter.
    """
    coords = funcs[0].function_space().mesh().coordinates.dat.data_ro.reshape(-1)
    lo, hi = float(coords.min()), float(coords.max())
    eps = 1e-9 * max(1.0, hi - lo)
    x = np.linspace(lo + eps, hi - eps, num_points)
    points = [[xi] for xi in x]
    return x, [np.asarray(f.at(points), dtype=float).reshape(-1) for f in funcs]


def _plot_1d_overlay(u_h, u_exact, path):
    """1D line plot of u_h with u_exact (two series, legend).

    Plotted with plain matplotlib from point samples: firedrake.pyplot.plot draws one line per
    cell and turns style keywords into per-curve lists, which breaks the legend.
    """
    x, (values_h, values_exact) = _sample_1d([u_h, u_exact])
    fig, axes = plt.subplots(facecolor=SURFACE)
    axes.plot(x, values_exact, color=TEXT_MUTED, linewidth=2, linestyle='--', label=r'$u$ (exact)')
    axes.plot(x, values_h, color=ERROR_SERIES['u_error'][1], linewidth=2, solid_capstyle='round',
              label=r'$u_h$')
    axes.set_xlabel('x')
    axes.set_title('u')
    _style_axes(axes)
    axes.legend(frameon=False, labelcolor=TEXT_SECONDARY)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_elliptic(parameters, outdir='.'):
    """Static plots of the fields of an elliptic run (parameters of that run).

    Reads parameters['output']['outfile_name'] + '.h5' and writes PNGs named after the fields
    into outdir; returns the list of written paths (empty for 3D runs, which are VTK-only).
    """
    dim = get_mesh_type_dimension(parameters['mesh']['type'])
    k = int(parameters['elliptic']['form_degree'])
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    h5_path = parameters['output']['outfile_name'] + '.h5'
    if dim is None or dim == 3:
        print('elliptic plots: ' + ('3D' if dim == 3 else 'file-mesh') + ' runs are VTK-only, see '
              + parameters['output']['outfile_name'] + '.pvd')
        return []

    with h5py.File(h5_path, 'r') as handle:
        has_harmonic = 'num_harmonic_forms' in handle and handle['num_harmonic_forms'][0] > 0
        has_exact = 'u_error' in handle
        for name in sorted(handle.keys()):
            if isinstance(handle[name], h5py.Dataset):
                print('statistic ' + name + ' = ' + str(np.array(handle[name]).reshape(-1)[0]))

    written = []
    with CheckpointFile(h5_path, 'r') as checkpoint:
        mesh = checkpoint.load_mesh()
        fields = {name: checkpoint.load_function(mesh, name, idx=0)
                  for name in elliptic_fields(k, dim, has_harmonic, has_exact)}
        for name, func in fields.items():
            base = str(outdir / name)
            if dim == 1:
                if name == 'u' and 'u_exact' in fields:
                    _plot_1d_overlay(func, fields['u_exact'], base + '.png')
                else:
                    plot_scalar1D(func, base)
                written.append(base + '.png')
            elif is_vector_field(name, k, dim):
                plot_vector2D_quiver(func, base)
                _plot_magnitude(func, base)
                written += [base + '.png', base + '-mag.png']
            else:
                plot_scalar2D(func, base)
                written.append(base + '.png')
    return written


# ---------------------------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------------------------

def plot_convergence(summary, path):
    """Log-log error vs h = 1/nx for an elliptic_convergence.py summary, with reference slopes.

    summary: the summary dict (or a path to summary.yaml). One series per error statistic
    (fixed color + marker per quantity, legend always shown; the summary.csv next to the
    summary is the table view). Dashed muted lines show O(h^p) for the expected rates, anchored
    below the finest-mesh errors. Returns the written path, or None if there is nothing to plot.
    """
    if not isinstance(summary, dict):
        import yaml
        summary = yaml.safe_load(Path(summary).read_text())
    series = convergence_series(summary.get('rows', []))
    if not series:
        return None
    order = int(summary.get('spatial-discretization', {}).get('order', 1))
    k = int(summary.get('elliptic', {}).get('form_degree', 0))

    fig, axes = plt.subplots(figsize=(6.4, 4.8), facecolor=SURFACE)
    finest = []
    for name, (nxs, errors) in series.items():
        label, color, marker = ERROR_SERIES[name]
        h = [1.0 / nx for nx in nxs]
        axes.loglog(h, errors, color=color, linewidth=2, marker=marker, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, solid_joinstyle='round',
                    solid_capstyle='round', label=label, zorder=3)
        finest.append((min(h), errors[int(np.argmin(h))], max(h)))

    # Reference slopes for the distinct expected rates of the plotted errors.
    rates = sorted({rate for name, rate in expected_rates(order, k).items() if name in series})
    h_min = min(f[0] for f in finest)
    h_max = max(f[2] for f in finest)
    anchor = 0.5 * min(f[1] for f in finest)
    for rate in rates:
        hs = np.array([h_min, h_max])
        axes.loglog(hs, anchor * (hs / h_min) ** rate, color=TEXT_MUTED, linewidth=1, linestyle='--', zorder=2)
        # Label just above the right end, right-aligned so it stays inside the axes.
        axes.annotate(r'$O(h^{%d})$' % rate, xy=(h_max, anchor * (h_max / h_min) ** rate), xytext=(-2, 4),
                      textcoords='offset points', va='bottom', ha='right', color=TEXT_SECONDARY, fontsize=9)

    elliptic = summary.get('elliptic', {})
    title = 'k = ' + str(k) + (', ' + str(elliptic['complex']) if elliptic.get('complex') else '') \
        + ', order ' + str(order)
    axes.set_title(title)
    axes.set_xlabel('h = 1 / nx')
    axes.set_ylabel(r'$L^2$ error')
    _style_axes(axes)
    axes.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc='best')
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------------
# Eigenproblems
# ---------------------------------------------------------------------------------------------

def _plot_field(func, name, k, dim, base):
    """One field plot (1D line, 2D scalar or quiver + magnitude); returns the written paths."""
    if dim == 1:
        plot_scalar1D(func, base)
        return [base + '.png']
    if is_vector_field(name, k, dim):
        plot_vector2D_quiver(func, base)
        _plot_magnitude(func, base)
        return [base + '.png', base + '-mag.png']
    plot_scalar2D(func, base)
    return [base + '.png']


def plot_eigenvalues(computed, exact, path, title=''):
    """Eigenvalue vs index: computed (filled markers) and exact (muted ring markers), legend."""
    fig, axes = plt.subplots(figsize=(6.4, 4.8), facecolor=SURFACE)
    index = np.arange(len(computed))
    if exact is not None and len(exact):
        axes.plot(np.arange(len(exact)), exact, linestyle='none', marker='o', markersize=11,
                  markerfacecolor='none', markeredgecolor=TEXT_MUTED, markeredgewidth=1.5, label='exact', zorder=2)
    axes.plot(index, computed, linestyle='none', marker='o', markersize=6, color=ERROR_SERIES['u_error'][1],
              markeredgecolor=SURFACE, markeredgewidth=1.0, label='computed', zorder=3)
    axes.set_xlabel('index')
    axes.set_ylabel(r'$\lambda$')
    axes.set_title(title)
    _style_axes(axes)
    axes.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_elliptic_eigen(parameters, outdir='.', max_modes=6):
    """Eigenvalue plot and plots of the first max_modes modes (1D/2D) of an elliptic-eigen run.

    Reads parameters['output']['outfile_name'] + '.h5'; mode i is checkpoint index i. Returns
    the written paths (only the eigenvalue plot in 3D).
    """
    dim = get_mesh_type_dimension(parameters['mesh']['type'])
    k = int(parameters['elliptic']['form_degree'])
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    h5_path = parameters['output']['outfile_name'] + '.h5'
    with h5py.File(h5_path, 'r') as handle:
        computed = np.array(handle['eigenvalues'])
        exact = np.array(handle['exact_eigenvalues']) if 'exact_eigenvalues' in handle else None
    elliptic = parameters['elliptic']
    title = 'k = ' + str(k) + (', ' + str(elliptic['complex']) if dim == 2 and elliptic.get('complex') else '')
    written = [str(plot_eigenvalues(computed, exact, outdir / 'eigenvalues.png', title))]
    if dim is None or dim == 3:
        return written
    names = (['sigma'] if k > 0 else []) + ['u']
    with CheckpointFile(h5_path, 'r') as checkpoint:
        mesh = checkpoint.load_mesh()
        for i in range(min(max_modes, len(computed))):
            for name in names:
                func = checkpoint.load_function(mesh, name, idx=i)
                written += _plot_field(func, name, k, dim, str(outdir / ('mode' + str(i) + '-' + name)))
    return written
