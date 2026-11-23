"""Unit tests for the pure-Python parts of dimswe/numerical_harmonic.py (TODO-elliptic §4.13)."""
import math

import pytest

from dimswe.numerical_harmonic import HARMONIC_METHODS, classify_zero_eigenvalues


def test_classify_counts_zero_eigenvalues_and_gap():
    """Zero eigenvalues are those below zero_tolerance * max; the gap is the next one over the largest zero."""
    count, gap = classify_zero_eigenvalues([1e-13, 3e-14, 9.87, 19.7, 39.5])
    assert count == 2 and math.isclose(gap, 9.87 / 1e-13)
    count, gap = classify_zero_eigenvalues([0.0, 4.0, 5.0])
    assert count == 1 and gap == float('inf')
    count, gap = classify_zero_eigenvalues([2.0, 3.0])
    assert count == 0


def test_classify_all_zero_and_expected():
    """All-zero spectra have no gap; a count different from the Betti number raises."""
    count, gap = classify_zero_eigenvalues([0.0, 1e-15])
    assert count == 2 and math.isnan(gap)
    classify_zero_eigenvalues([1e-14, 9.0], expected=1)
    with pytest.raises(ValueError, match="Betti number is 2"):
        classify_zero_eigenvalues([1e-14, 9.0, 10.0], expected=2)
    with pytest.raises(ValueError):
        classify_zero_eigenvalues([])


def test_absolute_zero_floor():
    """Round-off eigenvalues are zero even when nothing larger was computed."""
    count, gap = classify_zero_eigenvalues([1e-13, 2.0])
    assert count == 1
    count, gap = classify_zero_eigenvalues([3e-11, 4e-11])
    assert count == 2 and math.isnan(gap)


def test_methods():
    """Both methods are known."""
    assert set(HARMONIC_METHODS) == {'eigen', 'kernel'}
