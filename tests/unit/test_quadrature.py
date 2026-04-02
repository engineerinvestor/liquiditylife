"""Tests for Gaussian quadrature."""

import math

import numpy as np

from liquiditylife.calibrations.registry import load_calibration
from liquiditylife.solve.quadrature import (
    QuadratureSpec,
    build_shock_grid,
    gauss_hermite_nodes,
)


class TestGaussHermite:
    def test_weights_sum_to_one(self) -> None:
        _, weights = gauss_hermite_nodes(5)
        assert math.isclose(weights.sum(), 1.0, abs_tol=1e-12)

    def test_mean_zero(self) -> None:
        nodes, weights = gauss_hermite_nodes(5)
        mean = np.sum(nodes * weights)
        assert math.isclose(mean, 0.0, abs_tol=1e-10)

    def test_variance_one(self) -> None:
        nodes, weights = gauss_hermite_nodes(7)
        var = np.sum(nodes**2 * weights)
        assert math.isclose(var, 1.0, abs_tol=1e-10)

    def test_node_count(self) -> None:
        nodes, weights = gauss_hermite_nodes(10)
        assert len(nodes) == 10
        assert len(weights) == 10


class TestShockGrid:
    def test_shock_grid_weights_sum_to_one(self) -> None:
        cal = load_calibration("toy_demo_small_grid")
        spec = QuadratureSpec(n_xi=3, n_ncf=3, n_eta=3, n_eps=3)
        sg = build_shock_grid(cal, spec)
        assert math.isclose(sg.weights.sum(), 1.0, abs_tol=1e-10)

    def test_shock_grid_shapes_consistent(self) -> None:
        cal = load_calibration("toy_demo_small_grid")
        spec = QuadratureSpec(n_xi=3, n_ncf=3, n_eta=3, n_eps=3)
        sg = build_shock_grid(cal, spec)
        n = len(sg.weights)
        assert len(sg.xi) == n
        assert len(sg.ncf) == n
        assert len(sg.ndr) == n
        assert len(sg.eta) == n
        assert len(sg.eps) == n

    def test_ndr_is_rho_times_xi(self) -> None:
        cal = load_calibration("toy_demo_small_grid")
        spec = QuadratureSpec(n_xi=3, n_ncf=3, n_eta=3, n_eps=3)
        sg = build_shock_grid(cal, spec)
        expected_ndr = cal.asset_returns.rho_cs * sg.xi
        np.testing.assert_allclose(sg.ndr, expected_ndr)
