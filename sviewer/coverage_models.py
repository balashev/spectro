"""Source-weighted transmission for spatial column-density distributions N(x,y) = N_ref*G_N(x,y).

The fitted logN is the mean covered column (x-PL, radial PL), the covered-zone column (PPC) or the high-zone column (two-zone).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping
import re
import time

import numpy as np
from scipy.special import gammainc, gammaln, hyp1f1

from .source_models import source_from_fit, source_mode_name, source_wavelengths_from_observed


class CoverageDeadlineExceeded(RuntimeError):
    """Raised cooperatively when an SS73 comparison evaluation hits its deadline."""


def _check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= float(deadline):
        raise CoverageDeadlineExceeded("SS73 coverage evaluation deadline exceeded")


COVERAGE_MODE_NAMES = (
    "full",
    "ppc",
    "powerlaw",          # x/area law; mode id 2
    "powerlaw_ppc",      # x/area law + uncovered; mode id 3
    "two_zone",
    "cfov",
    "powerlaw_radial",
    "powerlaw_radial_ppc",
    "powerlaw_ppc_sysa",  # x-law: per-system a + per-LFR uncovered fraction; mode id 8
    "powerlaw_sysa_global",  # global x-law: one a per system, all transitions; Cf=1; mode id 9
)

COVERAGE_MODE_LABELS = (
    "Full coverage",
    "Partial covering",
    "Power-law inhomogeneous (x)",
    "Power-law + uncovered (x)",
    "Two-zone",
    "Overlap Cf (2 components)",
    "Radial power-law inhomogeneous",
    "Radial power-law + uncovered",
    "Power-law + uncovered (x; system a)",
    "Global system x-PL (all transitions; Cf=1)",
)


def coverage_mode_name(value) -> str:
    """Return a valid model name from a stored numeric/string value."""
    if isinstance(value, str):
        value = value.strip().lower()
        if value in COVERAGE_MODE_NAMES:
            return value
        try:
            value = int(round(float(value)))
        except (TypeError, ValueError):
            return "ppc"
    try:
        index = int(round(float(value)))
    except (TypeError, ValueError):
        return "ppc"
    return COVERAGE_MODE_NAMES[index] if 0 <= index < len(COVERAGE_MODE_NAMES) else "ppc"


def coverage_mode_index(value) -> int:
    return COVERAGE_MODE_NAMES.index(coverage_mode_name(value))


def _tau(tau):
    arr = np.asarray(tau, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Optical depth contains non-finite values")
    if np.any(arr < 0):
        raise ValueError("Optical depth must be non-negative")
    return arr


def _fraction(value, name):
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return value


def powerlaw_x_column_scale(u, log10_a):
    """x-PL column scaling (a+1)*u**a, normalized to unit mean over u in [0,1]."""
    u = np.asarray(u, dtype=float)
    log10_a = float(log10_a)
    if not np.isfinite(log10_a):
        raise ValueError("log10_a must be finite")
    a = 10.0 ** log10_a
    return (a + 1.0) * np.clip(u, 0.0, 1.0) ** a


def powerlaw_radial_column_scale(rho, log10_a):
    """Radial column scaling with unit circular area mean inside rho<=1."""
    rho = np.asarray(rho, dtype=float)
    log10_a = float(log10_a)
    if not np.isfinite(log10_a):
        raise ValueError("log10_a must be finite")
    a = 10.0 ** log10_a
    scale = np.zeros_like(rho, dtype=float)
    inside = (rho >= 0.0) & (rho <= 1.0)
    scale[inside] = 0.5 * (a + 1.0) * (a + 2.0) * (1.0 - rho[inside]) ** a
    return scale


def full_coverage(tau):
    return np.exp(-_tau(tau))


# Barlow & Sargent 1997, AJ 113, 136.
def partial_covering(tau, cf):
    """Homogeneous partial covering: I = (1-Cf) + Cf*exp(-tau)."""
    tau = _tau(tau)
    cf = _fraction(cf, "cf")
    return (1.0 - cf) + cf * np.exp(-tau)


def overlap_bounds(cf1, cf2):
    cf1 = _fraction(cf1, "cf1")
    cf2 = _fraction(cf2, "cf2")
    return max(0.0, cf1 + cf2 - 1.0), min(cf1, cf2)


def overlap_from_normalized(cf1, cf2, cfov_norm):
    q = _fraction(cfov_norm, "cfov_norm")
    lower, upper = overlap_bounds(cf1, cf2)
    return lower + q * (upper - lower)


def overlap_partial_covering(tau1, tau2, cf1, cf2, cfov_norm):
    """Two partially covering absorbers with arbitrary projected overlap."""
    tau1, tau2 = np.broadcast_arrays(_tau(tau1), _tau(tau2))
    cf1 = _fraction(cf1, "cf1")
    cf2 = _fraction(cf2, "cf2")
    cfov = overlap_from_normalized(cf1, cf2, cfov_norm)

    uncovered = 1.0 - cf1 - cf2 + cfov
    only1 = cf1 - cfov
    only2 = cf2 - cfov
    flux = (
        uncovered
        + only1 * np.exp(-tau1)
        + only2 * np.exp(-tau2)
        + cfov * np.exp(-(tau1 + tau2))
    )
    return np.clip(flux, 0.0, 1.0)


@lru_cache(maxsize=8)
def _legendre_nodes(n_nodes):
    if int(n_nodes) < 16:
        raise ValueError("At least 16 quadrature nodes are required")
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    return (x + 1.0) / 2.0, w / 2.0


def _powerlaw_moment(tau_max, a, moment=0, *, n_nodes=48, profile=None, profile_label=None):
    """Return J_m = integral_0^1 u**m exp(-tau_max*u**a) du = gamma(s, tau_max)/(a*tau_max**s), s=(m+1)/a.

    Falls back to the 1F1 form (Abramowitz & Stegun 1964, 6.5.12), then to quadrature."""
    total_started = time.perf_counter()
    tau_max = _tau(tau_max)
    a = float(a)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("power-law exponent a must be positive")
    m = float(moment)
    if m < 0:
        raise ValueError("moment must be non-negative")

    shape = (m + 1.0) / a
    zero_value = 1.0 / (m + 1.0)
    value = np.full_like(tau_max, zero_value, dtype=float)

    positive = tau_max > 1e-12
    gamma_failed = np.zeros_like(tau_max, dtype=bool)
    gamma_seconds = 0.0
    hyp_seconds = 0.0
    quadrature_seconds = 0.0
    n_hyp = 0
    n_quad = 0
    if np.any(positive):
        t = tau_max[positive]
        stage_started = time.perf_counter()
        reg_lower = gammainc(shape, t)
        gamma_ok = np.isfinite(reg_lower) & (reg_lower > 0.0)
        local_value = np.full_like(t, np.nan, dtype=float)
        if np.any(gamma_ok):
            tg = t[gamma_ok]
            pg = reg_lower[gamma_ok]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                log_value = (
                    gammaln(shape) + np.log(pg) - np.log(a)
                    - shape * np.log(tg)
                )
                local_value[gamma_ok] = np.exp(log_value)
        gamma_seconds += time.perf_counter() - stage_started
        value[positive] = local_value
        failed_local = ~gamma_ok
        if np.any(failed_local):
            positive_indices = np.flatnonzero(positive)
            gamma_failed.flat[positive_indices[failed_local]] = True

    bad = (gamma_failed | (~np.isfinite(value)) | (value < 0.0)
           | (value > zero_value * (1.0 + 1e-10)))

    if np.any(bad):
        n_hyp = int(np.count_nonzero(bad))
        stage_started = time.perf_counter()
        rescued = np.asarray(
            hyp1f1(shape, shape + 1.0, -tau_max[bad]) / (m + 1.0),
            dtype=float,
        )
        hyp_seconds += time.perf_counter() - stage_started
        value[bad] = rescued

    bad = ((~np.isfinite(value)) | (value < 0.0)
           | (value > zero_value * (1.0 + 1e-10)))
    if np.any(bad):
        n_quad = int(np.count_nonzero(bad))
        stage_started = time.perf_counter()
        u, w = _legendre_nodes(max(int(n_nodes), 64))
        tau_bad = np.asarray(tau_max[bad], dtype=float).reshape(-1)
        fallback = np.sum(
            (u ** m) * np.exp(-tau_bad[:, None] * u**a) * w,
            axis=-1,
        )
        quadrature_seconds += time.perf_counter() - stage_started
        value[bad] = fallback

    result = np.clip(value, 0.0, zero_value)
    if profile is not None:
        finite_tau = tau_max[np.isfinite(tau_max)]
        profile.setdefault("powerlaw_moments", []).append({
            "label": profile_label or f"J{int(m) if m.is_integer() else m}",
            "moment": m,
            "a": a,
            "shape": shape,
            "n_points": int(tau_max.size),
            "n_positive": int(np.count_nonzero(positive)),
            "n_hyp_rescue": n_hyp,
            "n_quad_fallback": n_quad,
            "tau_min": float(np.min(finite_tau)) if finite_tau.size else float("nan"),
            "tau_max": float(np.max(finite_tau)) if finite_tau.size else float("nan"),
            "gamma_seconds": gamma_seconds,
            "hyp_seconds": hyp_seconds,
            "quadrature_seconds": quadrature_seconds,
            "total_seconds": time.perf_counter() - total_started,
        })
    return result


# Power-law column distribution over the source: de Kool, Korista & Arav 2002, ApJ 580, 54;
# Arav et al. 2005, ApJ 620, 665.
def powerlaw_x_inhomogeneous(tau, log10_a, *, tau_kind="mean", n_nodes=48, profile=None):
    """Transmission for N(x)=N_max*x**a, 0<=x<=1; N_max=(a+1)*<N> for tau_kind="mean"."""
    tau = _tau(tau)
    log10_a = float(log10_a)
    if not np.isfinite(log10_a):
        raise ValueError("log10_a must be finite")
    a = 10.0 ** log10_a
    if tau_kind == "mean":
        tau_max = (a + 1.0) * tau
    elif tau_kind == "max":
        tau_max = tau
    else:
        raise ValueError("tau_kind must be 'mean' or 'max'")
    return np.clip(_powerlaw_moment(tau_max, a, 0, n_nodes=n_nodes, profile=profile, profile_label="J0"), 0.0, 1.0)


def powerlaw_radial_inhomogeneous(tau, log10_a, *, tau_kind="mean", n_nodes=48, profile=None):
    """Transmission for N(rho)=N_max*(1-rho)**a, rho=r/R_abs, with area weight 2*rho*d(rho).

    With tau_kind="mean", N_max=(a+1)*(a+2)*<N>/2."""
    tau = _tau(tau)
    log10_a = float(log10_a)
    if not np.isfinite(log10_a):
        raise ValueError("log10_a must be finite")
    a = 10.0 ** log10_a
    if tau_kind == "mean":
        tau_max = 0.5 * (a + 1.0) * (a + 2.0) * tau
    elif tau_kind == "max":
        tau_max = tau
    else:
        raise ValueError("tau_kind must be 'mean' or 'max'")

    j0 = _powerlaw_moment(tau_max, a, 0, n_nodes=n_nodes, profile=profile, profile_label="J0")
    j1 = _powerlaw_moment(tau_max, a, 1, n_nodes=n_nodes, profile=profile, profile_label="J1")
    transmission = 2.0 * (j0 - j1)
    bad = (~np.isfinite(transmission)) | (transmission < 0.0) | (transmission > 1.0)
    if np.any(bad):
        rho, w = _legendre_nodes(max(int(n_nodes), 64))
        fallback = np.sum(
            2.0 * rho * np.exp(-tau_max[..., None] * (1.0-rho)**a) * w,
            axis=-1,
        )
        transmission = np.where(bad, fallback, transmission)
    return np.clip(transmission, 0.0, 1.0)


def powerlaw_with_uncovered_fraction(
    tau, cf, log10_a, *, tau_kind="mean", n_nodes=48, geometry="radial", profile=None
):
    cf = _fraction(cf, "cf")
    if geometry == "radial":
        covered = powerlaw_radial_inhomogeneous(
            tau, log10_a, tau_kind=tau_kind, n_nodes=n_nodes, profile=profile
        )
    elif geometry == "x":
        covered = powerlaw_x_inhomogeneous(
            tau, log10_a, tau_kind=tau_kind, n_nodes=n_nodes, profile=profile
        )
    else:
        raise ValueError("geometry must be 'radial' or 'x'")
    return (1.0 - cf) + cf * covered

def powerlaw_system_product_with_uncovered(
    tau_by_system: Mapping[int, np.ndarray], cf, log10_a_by_system, *, profile=None
):
    """I=(1-Cf)+Cf*product_k P_x(tau_k,a_k): one exponent per velocity system, Cf per wavelength region."""
    cf = _fraction(cf, "cf")
    system_ids = sorted(tau_by_system)
    if not system_ids:
        raise ValueError("tau_by_system is empty")

    first = _tau(tau_by_system[system_ids[0]])
    covered = np.ones_like(first, dtype=float)
    for system_id in system_ids:
        if system_id not in log10_a_by_system:
            raise ValueError(
                f"Missing system power-law exponent for sys{system_id}; "
                f"expected covasys_{system_id}"
            )
        covered *= powerlaw_x_inhomogeneous(
            tau_by_system[system_id],
            log10_a_by_system[system_id],
            profile=profile,
        )
    return (1.0 - cf) + cf * covered


def two_zone(tau_high, cf_high, tau_ratio):
    """Two column-density zones; PPC is recovered when ``tau_ratio=0``."""
    tau_high = _tau(tau_high)
    cf_high = _fraction(cf_high, "cf_high")
    tau_ratio = _fraction(tau_ratio, "tau_ratio")
    return (
        (1.0 - cf_high) * np.exp(-tau_ratio * tau_high)
        + cf_high * np.exp(-tau_high)
    )




SPATIAL_REALIZATION_LABELS = {
    "full": "uniform full screen",
    "ppc": "centered-stripe PPC",
    "powerlaw": "stripe x-PL realization",
    "powerlaw_ppc": "stripe x-PL + uncovered realization",
    "powerlaw_ppc_sysa": "orthogonal-rank square realization",
    "powerlaw_sysa_global": "orthogonal-rank square realization",
    "powerlaw_radial": "radial N(R) realization",
    "powerlaw_radial_ppc": "radial N(R) + angular uncovered sector",
    "two_zone": "centered-stripe two-zone realization",
    "cfov": "overlapping-stripe realization",
}


def spatial_realization_label(mode):
    return SPATIAL_REALIZATION_LABELS[coverage_mode_name(mode)]


@lru_cache(maxsize=16)
def _legendre_rule(n_nodes):
    n_nodes = int(n_nodes)
    if n_nodes < 8:
        raise ValueError("spatial quadrature requires at least 8 nodes")
    return np.polynomial.legendre.leggauss(n_nodes)


def _interval_rule(left, right, n_nodes=22, zero_power=3.0):
    left = float(left)
    right = float(right)
    if right <= left:
        return np.empty(0), np.empty(0)
    nodes, weights = _legendre_rule(n_nodes)
    t = 0.5 * (nodes + 1.0)
    wt = 0.5 * weights
    tol = 1e-15
    if abs(left) <= tol and right > 0.0:
        x = right * t ** zero_power
        w = wt * right * zero_power * t ** (zero_power - 1.0)
        return x, w
    if abs(right) <= tol and left < 0.0:
        length = -left
        x = -length * t ** zero_power
        w = wt * length * zero_power * t ** (zero_power - 1.0)
        return x, w
    x = left + (right - left) * t
    w = (right - left) * wt
    return x, w


def _split_at_zero(left, right):
    if left < 0.0 < right:
        return [(left, 0.0), (0.0, right)]
    return [(left, right)]


def _rectangle_rule(x0, x1, y0=-1.0, y1=1.0, n_nodes=22):
    chunks = []
    for xa, xb in _split_at_zero(float(x0), float(x1)):
        x, wx = _interval_rule(xa, xb, n_nodes)
        for ya, yb in _split_at_zero(float(y0), float(y1)):
            y, wy = _interval_rule(ya, yb, n_nodes)
            if not x.size or not y.size:
                continue
            xx, yy = np.meshgrid(x, y, indexing="ij")
            ww = np.multiply.outer(wx, wy)
            chunks.append((xx.ravel(), yy.ravel(), ww.ravel()))
    if not chunks:
        return np.empty(0), np.empty(0), np.empty(0)
    return tuple(np.concatenate([chunk[i] for chunk in chunks]) for i in range(3))


def _source_weight(source, wavelength, x, y, area_weight):
    radius = np.hypot(x, y)
    return source.brightness(radius, wavelength) * area_weight


def _axis_rule(left, right, n_nodes=22):
    chunks = [_interval_rule(a, b, n_nodes) for a, b in _split_at_zero(float(left), float(right))]
    chunks = [(x, w) for x, w in chunks if x.size]
    if not chunks:
        return np.empty(0), np.empty(0)
    return np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])


def _stripe_transmitted_flux(
    source, wavelength, tau_by_system, scale_by_system,
    x0, x1, y0=-1.0, y1=1.0, n_nodes=22, deadline=None, chunk_size=256,
):
    """Integrate stripe models; the absorber depends only on x, so the source is integrated over y first."""
    _check_deadline(deadline)
    x, wx = _axis_rule(x0, x1, n_nodes)
    y, wy = _axis_rule(y0, y1, n_nodes)
    if not x.size or not y.size:
        n_wave = next(iter(tau_by_system.values())).size
        return np.zeros(n_wave, dtype=float), 0.0
    radius = np.hypot(x[:, None], y[None, :])
    marginal_weight = wx * (source.brightness(radius, wavelength) @ wy)
    scales = {system_id: scale_by_system(system_id, x, np.zeros_like(x)) for system_id in tau_by_system}
    n_wave = next(iter(tau_by_system.values())).size
    transmitted = np.empty(n_wave, dtype=float)
    step = max(1, int(chunk_size))
    for start in range(0, n_wave, step):
        _check_deadline(deadline)
        stop = min(start + step, n_wave)
        local_tau = np.zeros((stop - start, x.size), dtype=float)
        for system_id, tau in tau_by_system.items():
            local_tau += tau[start:stop, None] * scales[system_id][None, :]
        transmitted[start:stop] = np.exp(-np.clip(local_tau, 0.0, 1e6)) @ marginal_weight
    return transmitted, float(np.sum(marginal_weight))


def _rectangle_transmitted_flux(
    source, wavelength, tau_by_system, scale_by_system,
    x0, x1, y0=-1.0, y1=1.0, n_nodes=22, deadline=None, chunk_size=64,
):
    _check_deadline(deadline)
    x, y, area_weight = _rectangle_rule(x0, x1, y0, y1, n_nodes)
    if not x.size:
        n_wave = next(iter(tau_by_system.values())).size
        return np.zeros(n_wave, dtype=float), 0.0
    source_weight = _source_weight(source, wavelength, x, y, area_weight)
    scales = {system_id: scale_by_system(system_id, x, y) for system_id in tau_by_system}
    n_wave = next(iter(tau_by_system.values())).size
    transmitted = np.empty(n_wave, dtype=float)
    step = max(1, int(chunk_size))
    for start in range(0, n_wave, step):
        _check_deadline(deadline)
        stop = min(start + step, n_wave)
        local_tau = np.zeros((stop - start, x.size), dtype=float)
        for system_id, tau in tau_by_system.items():
            local_tau += tau[start:stop, None] * scales[system_id][None, :]
        transmitted[start:stop] = np.exp(-np.clip(local_tau, 0.0, 1e6)) @ source_weight
    return transmitted, float(np.sum(source_weight))


def _radial_transmitted_flux(source, wavelength, tau_total, log10_a, n_nodes=64, deadline=None, chunk_size=512):
    _check_deadline(deadline)
    r, wr = _interval_rule(0.0, 1.0, n_nodes)
    source_weight = 2.0 * np.pi * wr * r * source.brightness(r, wavelength)
    scale = powerlaw_radial_column_scale(r, log10_a)
    transmitted = np.empty(tau_total.size, dtype=float)
    step = max(1, int(chunk_size))
    for start in range(0, tau_total.size, step):
        _check_deadline(deadline)
        stop = min(start + step, tau_total.size)
        local_tau = tau_total[start:stop, None] * scale[None, :]
        transmitted[start:stop] = np.exp(-np.clip(local_tau, 0.0, 1e6)) @ source_weight
    return transmitted, float(np.sum(source_weight))


def _radial_offset_transmitted_flux(
    source, wavelength, tau_total, log10_a, offset, n_nodes=48, deadline=None, chunk_size=128,
):
    """Integrate a circular radial cloud centred at (offset*R_abs, 0) over a disk centred at (0, 0)."""
    _check_deadline(deadline)
    offset = float(offset)
    if offset < 0 or not np.isfinite(offset):
        raise ValueError('radial offset d/R_abs must be finite and >= 0')
    if offset == 0.0:
        return _radial_transmitted_flux(
            source, wavelength, tau_total, log10_a, n_nodes=max(48, n_nodes),
            deadline=deadline, chunk_size=chunk_size,
        )
    nr = max(32, int(n_nodes))
    nt = max(48, 2 * int(n_nodes))
    rn, rw = _legendre_rule(nr)
    r = 0.5 * (rn + 1.0)
    wr = 0.5 * rw
    theta = 2.0 * np.pi * (np.arange(nt, dtype=float) + 0.5) / nt
    # G_N is radial about the absorber centre: average the source brightness over
    # azimuth before exp(-tau*G_N), so the exponential is Nwave x Nr only.
    xx = offset + r[:, None] * np.cos(theta)[None, :]
    yy = r[:, None] * np.sin(theta)[None, :]
    angular_brightness = np.mean(source.brightness(np.hypot(xx, yy), wavelength), axis=1)
    source_weight = 2.0 * np.pi * wr * r * angular_brightness
    scale = powerlaw_radial_column_scale(r, log10_a)
    transmitted = np.empty(tau_total.size, dtype=float)
    step = max(1, int(chunk_size))
    for start in range(0, tau_total.size, step):
        _check_deadline(deadline)
        stop = min(start + step, tau_total.size)
        local_tau = tau_total[start:stop, None] * scale[None, :]
        transmitted[start:stop] = np.exp(-np.clip(local_tau, 0.0, 1e6)) @ source_weight
    return transmitted, float(np.sum(source_weight))


def _stripe_bounds(fraction):
    fraction = _fraction(fraction, "fraction")
    return -fraction, fraction


def _xpl_scale_for_bounds(x, left, right, log10_a):
    if right <= left:
        return np.zeros_like(x, dtype=float)
    u = (x - left) / (right - left)
    return powerlaw_x_column_scale(u, log10_a)


def _overlap_intervals(cf1, cf2, cfov_norm):
    cf1 = _fraction(cf1, "cf1")
    cf2 = _fraction(cf2, "cf2")
    cfov = overlap_from_normalized(cf1, cf2, cfov_norm)
    w1 = 2.0 * cf1
    w2 = 2.0 * cf2
    ov = 2.0 * cfov
    interval1 = (-1.0, -1.0 + w1)
    interval2 = (interval1[1] - ov, interval1[1] - ov + w2)
    cuts = sorted(set([-1.0, 1.0, *interval1, *interval2]))
    out = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        left = max(-1.0, left)
        right = min(1.0, right)
        if right <= left:
            continue
        mid = 0.5 * (left + right)
        in1 = interval1[0] <= mid <= interval1[1]
        in2 = interval2[0] <= mid <= interval2[1]
        if in1 or in2:
            out.append((left, right, in1, in2))
    return out


def spatial_region_transmission(mode, tau_by_system, params, source, wavelength, n_nodes=22, deadline=None):
    """Source-weighted transmission for the spatial N(x,y) of ``mode``.

    Moving quadrature boundaries keep the transmission smooth in Cf."""
    _check_deadline(deadline)
    mode = coverage_mode_name(mode)
    system_ids = sorted(tau_by_system)
    if not system_ids:
        raise ValueError("tau_by_system is empty")
    tau_arrays = {k: _tau(tau_by_system[k]) for k in system_ids}
    n_wave = tau_arrays[system_ids[0]].size
    tau_total = np.sum([tau_arrays[k] for k in system_ids], axis=0)
    total_flux = float(source.total_flux(wavelength))
    if not np.isfinite(total_flux) or total_flux <= 0:
        raise ValueError("invalid SS73 source normalization")

    if mode == "full":
        return np.exp(-tau_total)

    if mode in {"ppc", "powerlaw", "powerlaw_ppc", "two_zone"}:
        cf = 1.0 if mode in {"powerlaw"} else params.get("cf", 1.0)
        left, right = _stripe_bounds(cf)
        if mode == "ppc":
            scale_fun = lambda system_id, x, y: np.ones_like(x)
        elif mode in {"powerlaw", "powerlaw_ppc"}:
            scale_fun = lambda system_id, x, y: _xpl_scale_for_bounds(
                x, left, right, params["log10_a"]
            )
        else:
            scale_fun = lambda system_id, x, y: np.ones_like(x)
        transmitted, covered_flux = _stripe_transmitted_flux(
            source, wavelength, tau_arrays, scale_fun,
            left, right, -1.0, 1.0, n_nodes=n_nodes, deadline=deadline,
        )
        if mode == "two_zone":
            ratio = _fraction(params["tau_ratio"], "tau_ratio")
            low_all = np.exp(-ratio * tau_total) * total_flux
            low_covered = np.exp(-ratio * tau_total) * covered_flux
            return np.clip((low_all - low_covered + transmitted) / total_flux, 0.0, 1.0)
        return np.clip((total_flux - covered_flux + transmitted) / total_flux, 0.0, 1.0)

    if mode in {"powerlaw_radial", "powerlaw_radial_ppc"}:
        offset = float(params.get("offset", 0.0))
        transmitted, absorber_flux = _radial_offset_transmitted_flux(
            source, wavelength, tau_total, params["log10_a"], offset,
            n_nodes=max(48, n_nodes), deadline=deadline
        )
        covered_fraction = 1.0 if mode == "powerlaw_radial" else _fraction(params["cf"], "cf")
        absorbed_delta = covered_fraction * (transmitted - absorber_flux)
        return np.clip(1.0 + absorbed_delta / total_flux, 0.0, 1.0)

    if mode == "cfov":
        if len(system_ids) == 1:
            cf = params["cf1"] if system_ids[0] == 0 else params["cf2"]
            left, right = _stripe_bounds(cf)
            scale_fun = lambda system_id, x, y: np.ones_like(x)
            transmitted, covered_flux = _stripe_transmitted_flux(
                source, wavelength, tau_arrays, scale_fun,
                left, right, -1.0, 1.0, n_nodes=n_nodes, deadline=deadline,
            )
            return np.clip((total_flux - covered_flux + transmitted) / total_flux, 0.0, 1.0)
        if len(system_ids) != 2:
            raise ValueError("Overlap Cf supports exactly two active systems")
        result = np.full(n_wave, total_flux, dtype=float)
        for left, right, in1, in2 in _overlap_intervals(
            params["cf1"], params["cf2"], params["cfov_norm"]
        ):
            systems_here = []
            if in1:
                systems_here.append(system_ids[0])
            if in2:
                systems_here.append(system_ids[1])
            sub_tau = {k: tau_arrays[k] for k in systems_here}
            scale_fun = lambda system_id, x, y: np.ones_like(x)
            transmitted, region_flux = _stripe_transmitted_flux(
                source, wavelength, sub_tau, scale_fun,
                left, right, -1.0, 1.0, n_nodes=n_nodes, deadline=deadline,
            )
            result += transmitted - region_flux
        return np.clip(result / total_flux, 0.0, 1.0)

    if mode in {"powerlaw_ppc_sysa", "powerlaw_sysa_global"}:
        if len(system_ids) > 2:
            raise ValueError(
                "The explicit SS73 system-xPL realization currently supports at most two systems"
            )
        cf = params["cf"] if mode == "powerlaw_ppc_sysa" else 1.0
        half = np.sqrt(_fraction(cf, "cf"))
        loga = params["log10_a_by_system"]

        def scale_fun(system_id, x, y):
            order = system_ids.index(system_id)
            coord = x if order == 0 else y
            return _xpl_scale_for_bounds(coord, -half, half, loga[system_id])

        transmitted, covered_flux = _rectangle_transmitted_flux(
            source, wavelength, tau_arrays, scale_fun,
            -half, half, -half, half, n_nodes=n_nodes, deadline=deadline,
        )
        return np.clip((total_flux - covered_flux + transmitted) / total_flux, 0.0, 1.0)

    raise ValueError(f"Unknown coverage model: {mode}")


def spatial_column_scale_map(mode, params, x, y, system_id=None, system_order=0):
    """Return the dimensionless N/N_ref field for diagnostic maps; zero outside the absorber."""
    mode = coverage_mode_name(mode)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError('x and y map arrays must have the same shape')
    if mode == 'full':
        return np.ones_like(x)
    if mode in {'ppc', 'powerlaw', 'powerlaw_ppc', 'two_zone'}:
        cf = 1.0 if mode == 'powerlaw' else float(params.get('cf', 1.0))
        left, right = _stripe_bounds(cf)
        inside = (x >= left) & (x <= right) & (y >= -1.0) & (y <= 1.0)
        if mode == 'two_zone':
            out = np.full_like(x, float(params.get('tau_ratio', 0.1)))
            out[inside] = 1.0
            return out
        if mode == 'ppc':
            return inside.astype(float)
        out = np.zeros_like(x)
        out[inside] = _xpl_scale_for_bounds(x[inside], left, right, params.get('log10_a', 0.0))
        return out
    if mode in {'powerlaw_radial', 'powerlaw_radial_ppc'}:
        d = float(params.get('offset', 0.0))
        rho = np.hypot(x - d, y)
        inside = rho <= 1.0
        # In radial+uncovered mode Cf is a local mixture, not a sector, so G_N
        # remains the full radial field.
        out = np.zeros_like(x)
        out[inside] = powerlaw_radial_column_scale(rho[inside], params.get('log10_a', 0.0))
        return out
    if mode in {'powerlaw_ppc_sysa', 'powerlaw_sysa_global'}:
        cf = float(params.get('cf', 1.0)) if mode == 'powerlaw_ppc_sysa' else 1.0
        half = np.sqrt(_fraction(cf, 'cf'))
        coord = x if int(system_order) == 0 else y
        inside = (np.abs(x) <= half) & (np.abs(y) <= half)
        loga_by_system = params.get('log10_a_by_system', {})
        loga = loga_by_system.get(system_id, 0.0)
        out = np.zeros_like(x)
        out[inside] = _xpl_scale_for_bounds(coord[inside], -half, half, loga)
        return out
    return np.zeros_like(x)


def _wavelength_segments(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return []
    if x.size == 1:
        return [np.array([0], dtype=int)]
    dx = np.diff(x)
    good = dx[np.isfinite(dx) & (dx > 0)]
    if not good.size:
        return [np.arange(x.size)]
    typical = float(np.median(good))
    breaks = np.flatnonzero((dx <= 0) | (dx > 8.0 * typical)) + 1
    return [part for part in np.split(np.arange(x.size), breaks) if part.size]


def _chromatic_combine_split(x, total_by_system, full_by_system, region_by_system, fit):
    x = np.asarray(x, dtype=float)
    source = source_from_fit(fit)
    source_lambda, z_ref = source_wavelengths_from_observed(x, fit)
    mode = coverage_mode_name(getattr(getattr(fit, "covmode", None), "val", 1))
    result = np.ones_like(x, dtype=float)
    deadline = getattr(fit, "_comparison_deadline", None)
    _check_deadline(deadline)

    if not getattr(fit, "cf_fit", False):
        tau_total = np.zeros_like(x, dtype=float)
        for arr in total_by_system.values():
            tau_total += _tau(arr)
        return np.exp(-tau_total)

    segments = _wavelength_segments(x)

    def evaluate_segment(segment):
        _check_deadline(deadline)
        lam = float(np.exp(np.mean(np.log(source_lambda[segment]))))

        if mode == "powerlaw_sysa_global":
            tau_seg = {k: _tau(v)[segment] for k, v in total_by_system.items()}
            loga = {
                int(system_id): fit.getValue(f"covasys_{int(system_id)}")
                for system_id in tau_seg
                if hasattr(fit, f"covasys_{int(system_id)}")
            }
            params = {"cf": 1.0, "log10_a_by_system": loga}
            return segment, spatial_region_transmission(
                mode, tau_seg, params, source, lam, deadline=deadline
            )

        full_tau = np.zeros(segment.size, dtype=float)
        for arr in full_by_system.values():
            full_tau += _tau(arr)[segment]
        seg_flux = np.exp(-full_tau)

        for region, tau_by_system in sorted(region_by_system.items()):
            _check_deadline(deadline)
            if not hasattr(fit, f"cf_{region}"):
                tau_region = np.zeros(segment.size, dtype=float)
                for arr in tau_by_system.values():
                    tau_region += _tau(arr)[segment]
                seg_flux *= np.exp(-tau_region)
                continue
            cf_default = fit.getValue(f"cf_{region}")
            params = {
                "cf": cf_default,
                "log10_a": fit.getValue(f"cova_{region}") if hasattr(fit, f"cova_{region}") else 0.0,
                "offset": fit.getValue(f"covd_{region}") if hasattr(fit, f"covd_{region}") else 0.0,
                "log10_a_by_system": {
                    int(system_id): fit.getValue(f"covasys_{int(system_id)}")
                    for system_id in tau_by_system
                    if hasattr(fit, f"covasys_{int(system_id)}")
                },
                "tau_ratio": fit.getValue(f"covr_{region}") if hasattr(fit, f"covr_{region}") else 0.1,
                "cf1": fit.getValue(f"cov1_{region}") if hasattr(fit, f"cov1_{region}") else cf_default,
                "cf2": fit.getValue(f"cov2_{region}") if hasattr(fit, f"cov2_{region}") else cf_default,
                "cfov_norm": fit.getValue(f"covov_{region}") if hasattr(fit, f"covov_{region}") else 1.0,
            }
            tau_seg = {k: _tau(v)[segment] for k, v in tau_by_system.items()}
            seg_flux *= spatial_region_transmission(mode, tau_seg, params, source, lam, deadline=deadline)
        return segment, seg_flux

    for segment in segments:
        segment, seg_flux = evaluate_segment(segment)
        result[segment] = seg_flux

    profile = getattr(fit, "_coverage_profile_sink", None)
    if profile is not None:
        profile["source_model"] = "ss73"
        profile["source_beta"] = float(source.beta)
        profile["source_r_ref"] = float(source.r_ref)
        profile["source_lambda_ref"] = float(source.lambda_ref)
        profile["source_wavelength_redshift_proxy"] = z_ref
        profile["source_flux_enclosure"] = float(source.flux_enclosure())
        profile["source_normalization"] = "analytic infinite SS73 disk"
        profile["spatial_realization"] = spatial_realization_label(mode)
        profile["coverage_parallel_workers"] = 1
    return np.clip(result, 0.0, 1.0)


def region_transmission(mode, tau_by_system: Mapping[int, np.ndarray], params, profile=None):
    """Apply one coverage model to the optical depth in one Cf region.

    ``cfov`` needs two systems; a one-system call reduces to partial covering."""
    mode = coverage_mode_name(mode)
    if not tau_by_system:
        raise ValueError("tau_by_system is empty")

    system_ids = sorted(tau_by_system)
    tau_total = np.sum([_tau(tau_by_system[i]) for i in system_ids], axis=0)

    if mode == "full":
        return full_coverage(tau_total)
    if mode == "ppc":
        return partial_covering(tau_total, params["cf"])
    if mode == "powerlaw":
        return powerlaw_x_inhomogeneous(tau_total, params["log10_a"], profile=profile)
    if mode == "powerlaw_ppc":
        return powerlaw_with_uncovered_fraction(
            tau_total, params["cf"], params["log10_a"], geometry="x", profile=profile
        )
    if mode == "powerlaw_ppc_sysa":
        return powerlaw_system_product_with_uncovered(
            tau_by_system, params["cf"], params["log10_a_by_system"], profile=profile
        )
    if mode == "powerlaw_sysa_global":
        return powerlaw_system_product_with_uncovered(
            tau_by_system, 1.0, params["log10_a_by_system"], profile=profile
        )
    if mode == "powerlaw_radial":
        return powerlaw_radial_inhomogeneous(tau_total, params["log10_a"], profile=profile)
    if mode == "powerlaw_radial_ppc":
        return powerlaw_with_uncovered_fraction(
            tau_total, params["cf"], params["log10_a"], geometry="radial", profile=profile
        )
    if mode == "two_zone":
        return two_zone(tau_total, params["cf"], params["tau_ratio"])
    if mode == "cfov":
        if len(system_ids) == 1:
            cf = params["cf1"] if system_ids[0] == 0 else params["cf2"]
            return partial_covering(tau_total, cf)
        if len(system_ids) != 2:
            raise ValueError(
                "The overlap-Cf model currently supports exactly two active "
                f"systems per covering region; found {len(system_ids)}"
            )
        return overlap_partial_covering(
            tau_by_system[system_ids[0]],
            tau_by_system[system_ids[1]],
            params["cf1"],
            params["cf2"],
            params["cfov_norm"],
        )
    raise ValueError(f"Unknown coverage model: {mode}")


def parse_coverage_assignment(addinfo):
    """Parse ``cf.addinfo`` (``all_all``, ``sys0sys1_exp0``, ``sys0_exp_0``) into system ids and exposure.

    The exposure is ``None`` for all exposures."""
    text = str(addinfo or "all").strip()
    parts = text.split("_")
    system_part = parts[0] if parts and parts[0] else "all"
    if system_part == "all":
        systems = None
    else:
        systems = tuple(int(v) for v in re.findall(r"sys(\d+)", system_part))
        if not systems:
            systems = None

    exposure = None
    match = re.search(r"exp_?(\d+)", text)
    if match:
        exposure = int(match.group(1))
    return systems, exposure


def format_coverage_assignment(systems=None, exposure=None):
    if systems is None:
        system_part = "all"
    else:
        system_part = "".join(f"sys{int(i)}" for i in sorted(set(systems))) or "all"
    exposure_part = "all" if exposure is None else f"exp{int(exposure)}"
    return f"{system_part}_{exposure_part}"


def coverage_region_for_line(fit, line, exposure_index=None):
    """Return the matching covering-region id using cf.left and cf.right."""
    if not getattr(fit, "cf_fit", False):
        return None
    center = float(line.l()) * (1.0 + float(line.z))
    matched = None
    for i in range(getattr(fit, "cf_num", 0)):
        name = f"cf_{i}"
        if not hasattr(fit, name):
            continue
        cf = getattr(fit, name)
        systems, exposure = parse_coverage_assignment(getattr(cf, "addinfo", "all"))
        if systems is not None and int(line.sys) not in systems:
            continue
        if exposure is not None and (exposure_index is None or exposure != int(exposure_index)):
            continue
        left = -np.inf if getattr(cf, "left", None) is None else float(cf.left)
        right = np.inf if getattr(cf, "right", None) is None else float(cf.right)
        if left <= center <= right:
            matched = i
    return matched


def selected_fit_wavelength_bounds(spectra):
    """Return min/max wavelength of all currently selected finite fit pixels."""
    lo, hi = np.inf, -np.inf
    for spectrum in spectra:
        try:
            x = np.asarray(spectrum.spec.x(), dtype=float)
            mask = np.asarray(spectrum.fit_mask.x(), dtype=bool)
            if x.size != mask.size:
                x = np.asarray(spectrum.spec.norm.x, dtype=float)
                mask = np.asarray(spectrum.fit_mask.norm.x, dtype=bool)
            good = mask & np.isfinite(x)
            if np.any(good):
                lo = min(lo, float(np.min(x[good])))
                hi = max(hi, float(np.max(x[good])))
        except Exception:
            continue
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else None


def sync_single_coverage_region_to_fit(parent, pad_fraction=2e-6):
    """Extend a single common covering region over all selected fit windows."""
    fit = getattr(parent, "fit", None)
    spectra = getattr(parent, "s", None)
    if fit is None or spectra is None or not getattr(fit, "cf_fit", False) or getattr(fit, "cf_num", 0) != 1:
        return None
    if not hasattr(fit, "cf_0"):
        return None
    bounds = selected_fit_wavelength_bounds(spectra)
    if bounds is None:
        return None
    left, right = bounds
    pad = max((right - left) * float(pad_fraction), 1e-6)
    cf = fit.cf_0
    cf.left, cf.right = left - pad, right + pad
    try:
        if getattr(parent, "plot", None) is not None and len(parent.plot.pcRegions) == 1:
            region = parent.plot.pcRegions[0]
            region.x1, region.x2 = cf.left, cf.right
            region.redraw()
    except Exception:
        pass
    return cf.left, cf.right


def _line_region_id(line):
    """Return the first valid covering-region id attached to a Spectro line."""
    raw = getattr(line, "cf", -1)
    ids = np.atleast_1d(raw).astype(int)
    ids = ids[ids >= 0]
    return int(ids[0]) if ids.size else None


def assigned_coverage_region(line, fit=None):
    """Return the LFR id attached to ``line``, or ``None``.

    With ``fit``, region ids without a ``cf_i`` parameter mean full coverage."""
    region = _line_region_id(line)
    if region is not None and fit is not None and not hasattr(fit, f"cf_{region}"):
        return None
    return region


def split_line_optical_depths(x, lines, fit=None):
    """Return total, full-coverage and region optical depths by system.

    region_by_system[region][system] uses the resolved line.cf (later region wins)."""
    x = np.asarray(x)
    total_by_system = {}
    full_by_system = {}
    region_by_system = {}
    for line in lines:
        try:
            system_id = int(line.sys)
        except Exception:
            continue
        prof = _tau(line.profile)
        if prof.shape != x.shape:
            continue
        total_by_system.setdefault(system_id, np.zeros_like(x, dtype=float))
        total_by_system[system_id] += prof
        region = assigned_coverage_region(line, fit=fit)
        if region is None:
            full_by_system.setdefault(system_id, np.zeros_like(x, dtype=float))
            full_by_system[system_id] += prof
        else:
            region_by_system.setdefault(int(region), {})
            region_by_system[int(region)].setdefault(system_id, np.zeros_like(x, dtype=float))
            region_by_system[int(region)][system_id] += prof
    return total_by_system, full_by_system, region_by_system


def combine_split_optical_depths(x, total_by_system, full_by_system, region_by_system, fit):
    """Reconstruct the transmission from intrinsic pre-coverage, pre-LSF tau channels.

    Achromatic mode is analytic; SS73 mode uses the spatial column-density realizations."""
    x = np.asarray(x)
    if source_mode_name(getattr(getattr(fit, "srcmode", None), "val", 0)) == "ss73":
        return _chromatic_combine_split(
            x, total_by_system, full_by_system, region_by_system, fit
        )
    mode = coverage_mode_name(getattr(getattr(fit, "covmode", None), "val", 1))
    if not getattr(fit, "cf_fit", False):
        tau_total = np.zeros_like(x, dtype=float)
        for arr in total_by_system.values():
            tau_total += _tau(arr)
        return np.exp(-tau_total)

    if mode == "powerlaw_sysa_global":
        loga = {
            int(system_id): fit.getValue(f"covasys_{int(system_id)}")
            for system_id in total_by_system
            if hasattr(fit, f"covasys_{int(system_id)}")
        }
        return powerlaw_system_product_with_uncovered(total_by_system, 1.0, loga)

    full_tau = np.zeros_like(x, dtype=float)
    for arr in full_by_system.values():
        full_tau += _tau(arr)
    flux = np.exp(-full_tau)
    for region, tau_by_system in sorted(region_by_system.items()):
        if not hasattr(fit, f"cf_{region}"):
            tau_region_total = np.zeros_like(x, dtype=float)
            for arr in tau_by_system.values():
                tau_region_total += _tau(arr)
            flux *= np.exp(-tau_region_total)
            continue
        cf_default = fit.getValue(f"cf_{region}")
        params = {
            "cf": cf_default,
            "log10_a": fit.getValue(f"cova_{region}") if hasattr(fit, f"cova_{region}") else 0.0,
            "offset": fit.getValue(f"covd_{region}") if hasattr(fit, f"covd_{region}") else 0.0,
            "log10_a_by_system": {
                int(system_id): fit.getValue(f"covasys_{int(system_id)}")
                for system_id in tau_by_system
                if hasattr(fit, f"covasys_{int(system_id)}")
            },
            "tau_ratio": fit.getValue(f"covr_{region}") if hasattr(fit, f"covr_{region}") else 0.1,
            "cf1": fit.getValue(f"cov1_{region}") if hasattr(fit, f"cov1_{region}") else cf_default,
            "cf2": fit.getValue(f"cov2_{region}") if hasattr(fit, f"cov2_{region}") else cf_default,
            "cfov_norm": fit.getValue(f"covov_{region}") if hasattr(fit, f"covov_{region}") else 1.0,
        }
        flux *= region_transmission(mode, tau_by_system, params)
    return np.clip(flux, 0.0, 1.0)


def combine_line_profiles(x, lines, fit):
    """Combine intrinsic line.profile optical depths on x.

    line.sys is the velocity system, line.cf the covering region; unassigned lines are fully covered."""
    total_started = time.perf_counter()
    x = np.asarray(x)
    lines = list(lines)
    profile = getattr(fit, "_coverage_profile_sink", None)
    if not lines:
        if profile is not None:
            profile["coverage_total_seconds"] = time.perf_counter() - total_started
            profile["coverage_regions"] = []
        return np.ones_like(x, dtype=float)

    grouping_started = time.perf_counter()
    if not getattr(fit, "cf_fit", False):
        tau_total = np.sum([_tau(line.profile) for line in lines], axis=0)
        result = np.exp(-tau_total)
        if profile is not None:
            profile["coverage_grouping_seconds"] = time.perf_counter() - grouping_started
            profile["coverage_regions"] = []
            profile["coverage_total_seconds"] = time.perf_counter() - total_started
        return result

    mode_value = getattr(getattr(fit, "covmode", None), "val", 1)
    mode = coverage_mode_name(mode_value)
    total_by_system, full_by_system, region_tau = split_line_optical_depths(x, lines, fit=fit)

    if source_mode_name(getattr(getattr(fit, "srcmode", None), "val", 0)) == "ss73":
        result = combine_split_optical_depths(
            x, total_by_system, full_by_system, region_tau, fit
        )
        if profile is not None:
            profile["coverage_grouping_seconds"] = time.perf_counter() - grouping_started
            profile.setdefault("coverage_regions", [])
            profile["coverage_total_seconds"] = time.perf_counter() - total_started
        return result

    if mode == "powerlaw_sysa_global":
        loga = {
            int(system_id): fit.getValue(f"covasys_{int(system_id)}")
            for system_id in total_by_system
            if hasattr(fit, f"covasys_{int(system_id)}")
        }
        missing = [k for k in total_by_system if k not in loga]
        if missing:
            raise ValueError(
                "Global system x-PL is missing covasys parameters for systems "
                + ", ".join(str(k) for k in missing)
            )
        flux = powerlaw_system_product_with_uncovered(
            total_by_system, 1.0, loga, profile=profile
        )
        if profile is not None:
            profile["coverage_grouping_seconds"] = time.perf_counter() - grouping_started
            profile["coverage_regions"] = [{
                "region": "global",
                "mode": mode,
                "systems": list(sorted(total_by_system)),
                "n_points": int(x.size),
            }]
            profile["coverage_total_seconds"] = time.perf_counter() - total_started
        return np.clip(flux, 0.0, 1.0)

    full_tau = np.zeros_like(x, dtype=float)
    for arr in full_by_system.values():
        full_tau += arr
    grouping_seconds = time.perf_counter() - grouping_started
    flux = np.exp(-full_tau)
    region_profiles = []
    for region, tau_by_system in region_tau.items():
        cf_default = fit.getValue(f"cf_{region}")
        params = {
            "cf": cf_default,
            "log10_a": fit.getValue(f"cova_{region}") if hasattr(fit, f"cova_{region}") else 0.0,
            "offset": fit.getValue(f"covd_{region}") if hasattr(fit, f"covd_{region}") else 0.0,
            "log10_a_by_system": {
                int(system_id): fit.getValue(f"covasys_{int(system_id)}")
                for system_id in tau_by_system
                if hasattr(fit, f"covasys_{int(system_id)}")
            },
            "tau_ratio": fit.getValue(f"covr_{region}") if hasattr(fit, f"covr_{region}") else 0.1,
            "cf1": fit.getValue(f"cov1_{region}") if hasattr(fit, f"cov1_{region}") else cf_default,
            "cf2": fit.getValue(f"cov2_{region}") if hasattr(fit, f"cov2_{region}") else cf_default,
            "cfov_norm": fit.getValue(f"covov_{region}") if hasattr(fit, f"covov_{region}") else 1.0,
        }
        region_profile = {
            "region": int(region),
            "mode": mode,
            "systems": list(sorted(tau_by_system)),
            "n_points": int(x.size),
        }
        region_started = time.perf_counter()
        flux *= region_transmission(mode, tau_by_system, params, profile=region_profile)
        region_profile["total_seconds"] = time.perf_counter() - region_started
        region_profiles.append(region_profile)

    if profile is not None:
        profile["coverage_grouping_seconds"] = grouping_seconds
        profile["coverage_regions"] = region_profiles
        profile["coverage_total_seconds"] = time.perf_counter() - total_started
    return np.clip(flux, 0.0, 1.0)
