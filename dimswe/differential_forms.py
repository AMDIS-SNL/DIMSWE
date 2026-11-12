"""Differential-form utilities for the FEEC spaces of DIMSWE (see TODO-elliptic §1.2, §1.4, §4.3).

Forms are represented by their usual proxies (scalars for 0- and n-forms, vectors for the
intermediate degrees). In 2D the 1-form proxy depends on the complex:
- 'grad-curl': CG -grad-> Hcurl -curl2D-> DG
- 'curl-div' : CG -skewgrad-> Hdiv -div-> DG
In 1D (CG -d/dx-> DG) and 3D (CG -grad-> Hcurl -curl-> Hdiv -div-> DG) the complex argument is
ignored.

Operator table (TODO-elliptic §1.2). Each exterior derivative satisfies
(d w, z) = (w, delta z) + int_{boundary} B(w, z) ds:

| case (w is a k-form)     | d w                  | delta z (z a (k+1)-form)      | B(w, z)              |
|--------------------------|----------------------|-------------------------------|----------------------|
| 1D, k=0                  | dw/dx                | -dz/dx                        | w z n                |
| 2D grad-curl / 3D, k=0   | grad w               | -div z                        | w (z . n)            |
| 2D curl-div, k=0         | skewgrad w           | -curl2D z                     | w (n x z) (2D cross) |
| 2D grad-curl, k=1        | curl2D w             | -skewgrad z = (z_y, -z_x)     | (n x w) z            |
| 3D, k=1                  | curl w               | curl z                        | (n x w) . z          |
| 2D curl-div k=1, 3D k=2  | div w                | -grad z                       | (w . n) z            |

with skewgrad w = (-w_y, w_x) and curl2D w = w1_x - w0_y as in dimswe/ufl_helpers.py (whose API
is unchanged). Every sign is verified by tests/unit/test_differential_forms.py.

The functions here work on any UFL expressions of the right shape: discrete Functions, test and
trial functions, and closed-form expressions of SpatialCoordinate (used to build manufactured
solutions).
"""
from firedrake import as_vector, cross, curl, div, dot, grad
from ufl import as_ufl

from .meshes import DIRECTIONS
from .ufl_helpers import curl2D, skewgrad
from .variables import get_diff_form_space

# Allowed names for the 2D complexes.
COMPLEXES = ('grad-curl', 'curl-div')


def normalize_complex(dim, complex):
    """Validate the complex name for dimension dim and return it in canonical form.

    In 2D complex must be 'grad-curl' or 'curl-div'. In 1D and 3D it is irrelevant and None is
    returned (whatever was given), so downstream code can rely on a single value per case.
    """
    if dim not in (1, 2, 3):
        raise ValueError("dimension " + str(dim) + " is not supported (1, 2 or 3)")
    if dim != 2:
        return None
    if complex not in COMPLEXES:
        raise ValueError("2D complex must be one of " + str(list(COMPLEXES)) + ", got " + str(complex))
    return complex


def _check_degree(k, dim):
    """Raise ValueError unless 0 <= k <= dim."""
    if k < 0 or k > dim:
        raise ValueError("form degree " + str(k) + " is out of range for dimension " + str(dim))


def is_scalar_form(k, dim):
    """True if k-forms in dim dimensions have a scalar proxy (k = 0 or k = dim), else vector."""
    _check_degree(k, dim)
    return k == 0 or k == dim


def form_space(spaces, k, dim, complex):
    """Discrete space of k-forms from a DeRhamComplex (thin wrapper over get_diff_form_space).

    In 2D, 1-forms live in Hcurl for 'grad-curl' and in Hdiv for 'curl-div'.
    """
    complex = normalize_complex(dim, complex)
    _check_degree(k, dim)
    return get_diff_form_space(spaces, 'S', k, dim, complex == 'curl-div')


def _cross_2d(a, b):
    """Scalar 2D cross product a x b = a0 b1 - a1 b0."""
    return a[0] * b[1] - a[1] * b[0]


def exterior_derivative(w, k, dim, complex):
    """Exterior derivative d w of a k-form proxy w, as a (k+1)-form proxy (table above).

    Returns None for k = dim (d of a top form vanishes, and there is no (dim+1)-form proxy).
    """
    complex = normalize_complex(dim, complex)
    _check_degree(k, dim)
    if k == dim:
        return None
    if dim == 1:
        return w.dx(0)
    if k == 0:
        return skewgrad(w) if complex == 'curl-div' else grad(w)
    if dim == 2:
        return curl2D(w) if complex == 'grad-curl' else div(w)
    return curl(w) if k == 1 else div(w)


def codifferential(z, k, dim, complex):
    """Codifferential delta z of a k-form proxy z, as a (k-1)-form proxy (table above).

    delta_k is the formal L2 adjoint of d_{k-1}, i.e. (d w, z) = (w, delta z) for compactly
    supported w. Returns None for k = 0. Used for manufactured solutions (sigma = delta u) and the
    Hodge Laplacian; the discrete problem never applies it to finite element functions.
    """
    complex = normalize_complex(dim, complex)
    _check_degree(k, dim)
    if k == 0:
        return None
    if dim == 1:
        return -z.dx(0)
    if k == 1:
        return -curl2D(z) if complex == 'curl-div' else -div(z)
    if dim == 2:
        # k = 2: adjoint of curl2D (grad-curl) or of div (curl-div)
        return -skewgrad(z) if complex == 'grad-curl' else -grad(z)
    return curl(z) if k == 2 else -grad(z)


def hodge_laplacian_expr(u, k, dim, complex):
    """Symbolic Hodge Laplacian (d delta + delta d) u of a k-form proxy u.

    Absent terms are skipped (no d delta for k = 0, no delta d for k = dim). On flat Cartesian
    domains this equals minus the componentwise Laplacian of the proxy, which the tests check.
    Used to compute the source f from an exact solution.
    """
    terms = []
    sigma = codifferential(u, k, dim, complex)
    if sigma is not None:
        terms.append(exterior_derivative(sigma, k - 1, dim, complex))
    du = exterior_derivative(u, k, dim, complex)
    if du is not None:
        terms.append(codifferential(du, k + 1, dim, complex))
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return result


def boundary_pairing(w, z, k, dim, complex, n):
    """Boundary pairing B_k(w, z) of the integration-by-parts identity for d acting on k-forms.

    The identity is (d w, z) = (w, delta z) + int_{boundary} B_k(w, z) ds, with w a k-form proxy,
    z a (k+1)-form proxy and n the outward unit normal (FacetNormal); see the table above.

    For k = dim the exterior derivative vanishes, so the pairing is 0 (the Python integer, which
    callers can add to UFL forms or skip).
    """
    complex = normalize_complex(dim, complex)
    _check_degree(k, dim)
    if k == dim:
        return 0
    if dim == 1:
        return w * z * n[0]
    if k == 0:
        if complex == 'curl-div':
            return w * _cross_2d(n, z)
        # grad (2D grad-curl and 3D)
        return w * dot(z, n)
    if dim == 2:
        # k = 1
        if complex == 'grad-curl':
            return _cross_2d(n, w) * z
        return dot(w, n) * z
    # dim == 3
    if k == 1:
        return dot(cross(n, w), z)
    return dot(w, n) * z


def normalize_index_set(I, k, dim):
    """Canonical form of the index set I of the constant form dx^I = dx_{i1} ^ ... ^ dx_{ik}.

    I may contain direction names ('x', 'y', 'z') or axis numbers (0, 1, 2); the result is a
    tuple of direction names in increasing order. Raises ValueError unless I has k distinct
    entries within the first dim directions.
    """
    _check_degree(k, dim)
    names = []
    for i in I:
        name = DIRECTIONS[i] if isinstance(i, int) and 0 <= i < 3 else str(i).strip().lower()
        if name not in DIRECTIONS[:dim]:
            raise ValueError("index " + repr(i) + " is not a direction of a " + str(dim) + "D form")
        names.append(name)
    if len(set(names)) != len(names) or len(names) != k:
        raise ValueError("index set " + repr(tuple(I)) + " must have " + str(k) + " distinct directions")
    return tuple(d for d in DIRECTIONS[:dim] if d in names)


def constant_form_values(I, k, dim, complex):
    """Proxy values of the constant k-form dx^I (TODO-elliptic §1.4), as a float or tuple.

    I is taken in increasing direction order (see normalize_index_set), so for example in 3D
    {x, z} means dx ^ dz = -(dz ^ dx). Proxy map:
    - k = 0 and k = dim: 1, except 2D curl-div k = 2, where dx^dy -> -1;
    - Hcurl 1-forms (2D grad-curl, 3D): dx_i -> e_i;
    - 2D curl-div 1-forms (Hdiv proxy, consistent with skewgrad): dx -> (0, 1), dy -> (-1, 0);
    - 3D 2-forms (Hdiv): dy^dz -> e_x, dx^dz -> -e_y, dx^dy -> e_z.
    Every value is fixed by requiring d(x_{i1} dx^{I'}) = dx^I with the exterior derivatives of
    this module. For 2D curl-div that forces the sign of the 2-form: d(x dy) = div(x (-1, 0)) = -1
    (flipping the 1-form map instead would break d x = skewgrad x = dx).
    Pure Python (no Firedrake objects), so it is cheap to test.
    """
    complex = normalize_complex(dim, complex)
    I = normalize_index_set(I, k, dim)
    if k == 0:
        return 1.0
    if k == dim:
        return -1.0 if complex == 'curl-div' else 1.0
    if k == 1:
        axis = DIRECTIONS.index(I[0])
        if complex == 'curl-div':
            return (0.0, 1.0) if axis == 0 else (-1.0, 0.0)
        return tuple(1.0 if j == axis else 0.0 for j in range(dim))
    # 3D 2-forms: dx_i ^ dx_j (i < j) -> sign * e_m with (i, j, m) cyclic up to sign.
    proxies = {('y', 'z'): (1.0, 0.0, 0.0), ('x', 'z'): (0.0, -1.0, 0.0), ('x', 'y'): (0.0, 0.0, 1.0)}
    return proxies[I]


def constant_form_proxy(I, k, dim, complex):
    """UFL proxy of the constant k-form dx^I (scalar literal or as_vector of literals).

    See constant_form_values for the conventions. The result has no domain; combine it with mesh
    quantities (or project/interpolate it into a space) to use it in forms.
    """
    values = constant_form_values(I, k, dim, complex)
    if isinstance(values, tuple):
        return as_vector(values)
    return as_ufl(values)
