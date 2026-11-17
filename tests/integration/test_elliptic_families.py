"""MMS convergence of the elliptic problem for the FEEC families P-, P+, S (TODO-elliptic §4.9, §6).

Expected rates per family and form degree come from the L2 approximation order a(V) of each
space (EXPECTED below): full-degree spaces (CG, S, DG, DPC, N2curl, BDM, BDMCE, BDMCF) of degree
s give a = s + 1, trimmed spaces (N1curl, RT, RTCE/F, NCE/F, Sminus*) of degree r give a = r.
The asserted rates are then u: a(V^k); sigma: min(a(V^{k-1}), a(V^k)); du: a(V^{k+1});
d sigma: a(V^k) - which reproduces the Q rates of tests/integration/test_elliptic.py.
"""
import math

import pytest

from dimswe.elliptic import EllipticProblem
from dimswe.meshes import family_elements
from tests.conftest import make_elliptic_parameters

FULL_DEGREE = {'CG', 'S', 'DG', 'DPC', 'N2curl', 'BDM', 'BDMCE', 'BDMCF'}

# (family, dim, order, resolutions); 3D cases are slow.
FAMILY_SETUPS = [
    ('P-', 2, 2, (4, 8)), ('P+', 2, 2, (4, 8)), ('S', 2, 2, (4, 8)),
    ('P+', 2, 3, (3, 6)),
    # 3D tetrahedra need finer pairs than hexahedra to reach the asymptotic range
    # (P1 at n = 6 still only has 4.5 cells per sine wavelength in z, where Lz = 0.75).
    # No serendipity in 3D; S- is unusable (no FIAT dual basis).
    ('P-', 3, 1, (8, 16)), ('P+', 3, 3, (4, 8)),
]
BC_SETS = {2: ['essential', 'natural', 'mixed', 'periodic'], 3: ['essential', 'natural']}


def approximation_order(element):
    """L2 approximation order of an (element, degree, variant) table entry."""
    name, degree, _ = element
    return degree + 1 if name in FULL_DEGREE else degree


def space_order(family, dim, order, j, complex):
    """a(V^j) for the j-form space of the family (j outside [0, dim] -> None)."""
    if j < 0 or j > dim:
        return None
    table = family_elements(family, dim, order)
    if j == 0:
        return approximation_order(table['CG'])
    if j == dim:
        return approximation_order(table['DG'])
    key = 'Hcurl' if (dim == 3 and j == 1) or (dim == 2 and complex == 'grad-curl') else 'Hdiv'
    return approximation_order(table[key])


def expected_rates(family, dim, order, k, complex):
    """Expected L2 rates {statistic: rate} (see the module docstring)."""
    a = {j: space_order(family, dim, order, j, complex) for j in (k - 1, k, k + 1)}
    rates = {'u_error': a[k]}
    if k > 0:
        rates['sigma_error'] = min(a[k - 1], a[k])
        rates['dsigma_error'] = a[k]
    if k < dim:
        rates['du_error'] = a[k + 1]
    return rates


def cases():
    """All (family, dim, order, resolutions, complex, k, bcs) cases, 3D marked slow."""
    params = []
    for family, dim, order, ns in FAMILY_SETUPS:
        complexes = ['grad-curl', 'curl-div'] if dim == 2 else [None]
        for cx in complexes:
            for k in range(dim + 1):
                for bcs in BC_SETS[dim]:
                    marks = [pytest.mark.slow] if dim == 3 else []
                    case_name = "-".join([family, str(dim) + "D", "r" + str(order), cx or "", "k" + str(k), bcs])
                    params.append(pytest.param(family, dim, order, ns, cx, k, bcs, marks=marks, id=case_name))
    return params


@pytest.mark.parametrize("family, dim, order, ns, cx, k, bcs", cases())
def test_family_mms_convergence(family, dim, order, ns, cx, k, bcs):
    """Observed rates >= expected - 0.3 for every family, form degree and BC set."""
    stats = []
    for n in ns:
        model = EllipticProblem(make_elliptic_parameters(dim, cx, k, bcs, n=n, order=order, family=family))
        model.solve()
        model.create_statistics()
        model.compute_statistics()
        stats.append(model.get_statistics())
        assert stats[-1]['harmonic_orthogonality'][0] < 1e-8
    expected = expected_rates(family, dim, order, k, cx)
    errors = {name: (stats[0][name][0], stats[1][name][0]) for name in expected}
    for name, rate in expected.items():
        e0, e1 = errors[name]
        if e1 < 1e-11:
            continue  # exact up to round-off
        observed = math.log(e0 / e1) / math.log(ns[1] / ns[0])
        assert observed >= rate - 0.3, (name, observed, rate, errors)


def test_expected_rates_reproduce_q():
    """The approximation-order rule gives the Q rates used in test_elliptic.py."""
    assert expected_rates('Q', 2, 2, 0, 'grad-curl') == {'u_error': 3, 'du_error': 2}
    assert expected_rates('Q', 2, 2, 1, 'curl-div') == {'u_error': 2, 'sigma_error': 2, 'dsigma_error': 2,
                                                        'du_error': 2}
    assert expected_rates('Q', 3, 1, 3, None) == {'u_error': 1, 'sigma_error': 1, 'dsigma_error': 1}
    # P+ (full P_r): degrees drop by one per form degree.
    assert expected_rates('P+', 2, 2, 2, 'curl-div') == {'u_error': 1, 'sigma_error': 1, 'dsigma_error': 1}
