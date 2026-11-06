"""Independent sanity checks for the optional sampling-analysis estimators."""

import numpy as np
import pytest
from scipy.signal import lfilter

from benchmarks.performance.hmr.tests.analyze_sampling import (
    batch_ess,
    core_transitions,
    correlation,
    observables,
    positive_phi_core_visits,
    split_rhat,
)


def test_correlated_chain_matches_known_ar1_inefficiency():
    # Stationary AR(1): rho[k]=a**k, g=(1+a)/(1-a).
    rng = np.random.default_rng(87231)
    a = 0.9
    x = lfilter([1], [1, -a], rng.normal(size=201000))[1000:]
    info, rho = correlation(x, 0.1)
    assert rho[0] == pytest.approx(1)
    assert info["g"] == pytest.approx((1 + a) / (1 - a), rel=0.15)
    assert info["ess"] == pytest.approx(len(x) / info["g"])
    assert info["tau_int_ps"] == pytest.approx(info["g"] * 0.1 / 2)
    thinned, _ = correlation(x[::5], 0.5)
    assert thinned["tau_int_ps"] == pytest.approx(info["tau_int_ps"], rel=0.15)
    batch = batch_ess(x, 0.1, 100)
    assert len(x) / batch == pytest.approx((1 + a) / (1 - a), rel=0.25)


def test_iid_and_constant_observable_do_not_invent_correlation_or_ess():
    x = np.random.default_rng(414).normal(size=100000)
    info, rho = correlation(x, 0.1)
    assert len(x) * 0.9 < info["ess"] <= len(x)
    # Check FFT's linear (not circular) covariance against a direct formula.
    y = x - x.mean()
    for lag in [1, 10, 999]:
        direct = np.dot(y[:-lag], y[lag:]) / np.dot(y, y)
        assert rho[lag] == pytest.approx(direct, abs=1e-12)
    constant, rho = correlation(np.ones(100), 0.1)
    assert constant["ess"] is None
    assert rho is None
    assert batch_ess(np.ones(10000), 0.1, 100) is None
    assert batch_ess(x[:100], 0.1, 100) is None
    with pytest.raises(ValueError):
        correlation(np.array([np.nan] * 30), 0.1)


def test_periodic_observables_and_core_alternations():
    phi = np.array([-2.5, -1.2, 1.4, 2.7])
    psi = np.array([-3.0, -1.0, 0.5, 2.8])
    plain = observables(phi, psi)
    shifted = observables(phi, psi + 2 * np.pi)
    for key in plain:
        np.testing.assert_allclose(plain[key], shifted[key], atol=1e-14)
    # Repeated core observations and gap observations do not add crossings.
    assert core_transitions(np.radians([0, 10, 70, 110, 120, 70, 20])) == 2
    assert (
        positive_phi_core_visits(
            np.radians([-60, 0, 60, 70, 170, -170, -60, 0, 60])
        )
        == 2
    )


def test_split_rhat_detects_separated_chains():
    rng = np.random.default_rng(34)
    well_mixed = rng.normal(size=(3, 10000))
    assert split_rhat(well_mixed) == pytest.approx(1, abs=0.01)
    separated = well_mixed + np.array([-2, 0, 2])[:, None]
    assert split_rhat(separated) > 2
