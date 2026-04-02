"""Gaussian quadrature for integration over stochastic shocks."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from pydantic import BaseModel

from liquiditylife._types import ArrayFloat
from liquiditylife.calibrations.bundles import CalibrationBundle


class QuadratureSpec(BaseModel, frozen=True):
    """Number of quadrature nodes per shock dimension."""

    n_xi: int = 5
    """Nodes for the equity premium innovation."""

    n_eps: int = 3
    """Nodes for the transitory income shock."""

    n_eta: int = 5
    """Nodes for the persistent income shock."""

    n_ncf: int = 5
    """Nodes for cash-flow news (within each regime)."""


class ShockGrid(NamedTuple):
    """Precomputed quadrature nodes and weights for all shocks.

    Each array has shape ``(n_total_nodes,)`` where ``n_total_nodes``
    is the total number of integration points in the tensor product.
    """

    xi: ArrayFloat
    """Equity premium innovation nodes."""

    ncf: ArrayFloat
    """Cash-flow news nodes."""

    ndr: ArrayFloat
    """Discount-rate news nodes (derived from xi via rho_cs)."""

    eta: ArrayFloat
    """Persistent income shock nodes."""

    eps: ArrayFloat
    """Transitory income shock nodes."""

    weights: ArrayFloat
    """Combined quadrature weights (product of individual weights)."""


def gauss_hermite_nodes(n: int) -> tuple[ArrayFloat, ArrayFloat]:
    """Return Gauss-Hermite nodes and weights adapted for N(0,1).

    ``numpy.polynomial.hermite_e.hermegauss`` uses the probabilist's
    convention (weight function exp(-x^2/2)), so nodes are already
    standard-normal scale. Weights are normalised to sum to 1.
    """
    nodes, weights = hermegauss(n)
    nodes = np.array(nodes, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return nodes, weights


def build_shock_grid(cal: CalibrationBundle, spec: QuadratureSpec) -> ShockGrid:
    """Build the tensor-product shock grid for quadrature integration.

    Handles the crash/non-crash mixture for return news by computing
    nodes for both regimes and weighting by crash probability.
    """
    ar = cal.asset_returns
    ip = cal.income
    crash = ar.crash

    # 1. Equity premium innovation: xi ~ N(0, sigma_xi^2)
    xi_nodes_std, xi_weights = gauss_hermite_nodes(spec.n_xi)
    xi_nodes = xi_nodes_std * ar.sigma_xi

    # 2. Cash-flow news: mixture of crash/non-crash regimes
    ncf_nodes_std, ncf_weights_raw = gauss_hermite_nodes(spec.n_ncf)

    # Crash regime NCF nodes
    ncf_crash = crash.mu_cf_crash + ncf_nodes_std * crash.sigma_cf
    ncf_crash_w = ncf_weights_raw * crash.p_crash

    # Non-crash regime NCF nodes
    ncf_no_crash = crash.mu_cf_no_crash + ncf_nodes_std * crash.sigma_cf
    ncf_no_crash_w = ncf_weights_raw * (1.0 - crash.p_crash)

    # Combine crash and non-crash nodes
    ncf_all = np.concatenate([ncf_crash, ncf_no_crash])
    ncf_w_all = np.concatenate([ncf_crash_w, ncf_no_crash_w])
    ncf_w_all = ncf_w_all / ncf_w_all.sum()

    # 3. Persistent income shock: eta ~ N(0, sigma_eta^2)
    #    (mixture conditioning on return news is applied during Bellman evaluation)
    eta_nodes_std, eta_weights = gauss_hermite_nodes(spec.n_eta)
    eta_nodes = eta_nodes_std * ip.sigma_eta

    # 4. Transitory income shock: eps ~ N(0, sigma_eps^2)
    eps_nodes_std, eps_weights = gauss_hermite_nodes(spec.n_eps)
    eps_nodes = eps_nodes_std * ip.sigma_eps

    # Build tensor product
    xi_g, ncf_g, eta_g, eps_g = np.meshgrid(
        xi_nodes, ncf_all, eta_nodes, eps_nodes, indexing="ij"
    )
    w_xi, w_ncf, w_eta, w_eps = np.meshgrid(
        xi_weights, ncf_w_all, eta_weights, eps_weights, indexing="ij"
    )

    # Compute NDR from xi (deterministic mapping)
    ndr_g = ar.rho_cs * xi_g

    # Flatten
    combined_weights = w_xi * w_ncf * w_eta * w_eps
    combined_weights = combined_weights / combined_weights.sum()

    return ShockGrid(
        xi=xi_g.ravel(),
        ncf=ncf_g.ravel(),
        ndr=ndr_g.ravel(),
        eta=eta_g.ravel(),
        eps=eps_g.ravel(),
        weights=combined_weights.ravel(),
    )
