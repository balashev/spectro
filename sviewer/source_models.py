"""Source-brightness models used by the coverage forward model."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma, zeta


SOURCE_MODE_NAMES = ("achromatic", "ss73")


def source_model_label(value, beta=None) -> str:
    """Human-readable source label; beta=0.75 is the SS73 preset."""
    mode = source_mode_name(value)
    if mode == "achromatic":
        return "Achromatic"
    try:
        beta_value = float(beta)
    except (TypeError, ValueError):
        beta_value = None
    if beta_value is not None and np.isfinite(beta_value) and np.isclose(beta_value, 0.75, rtol=0.0, atol=1e-12):
        return "Power-law temperature disk / SS73 (beta=0.75)"
    return "Power-law temperature disk"


def source_mode_name(value) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        aliases = {
            "thin disk / ss73": "ss73",
            "thin disk": "ss73",
            "thin_disk": "ss73",
            "power-law temperature disk": "ss73",
            "powerlaw temperature disk": "ss73",
            "powerlaw_disk": "ss73",
            "power-law disk": "ss73",
        }
        text = aliases.get(text, text)
        if text in SOURCE_MODE_NAMES:
            return text
        try:
            value = int(round(float(value)))
        except (TypeError, ValueError):
            return "achromatic"
    try:
        index = int(round(float(value)))
    except (TypeError, ValueError):
        return "achromatic"
    return SOURCE_MODE_NAMES[index] if 0 <= index < len(SOURCE_MODE_NAMES) else "achromatic"


def source_mode_index(value) -> int:
    return SOURCE_MODE_NAMES.index(source_mode_name(value))


# R_lambda proportional to lambda^(1/beta) for T(R) ~ R^(-beta); beta = 0.75 is the
# thin accretion disk of Shakura & Sunyaev 1973, A&A 24, 337.
def disk_radius(wavelength, beta=0.75, r_ref=1.0, lambda_ref=2400.0):
    wavelength = np.asarray(wavelength, dtype=float)
    beta = float(beta)
    r_ref = float(r_ref)
    lambda_ref = float(lambda_ref)
    if beta <= 0 or r_ref <= 0 or lambda_ref <= 0:
        raise ValueError("beta, R_ref, and lambda_ref must be positive")
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("source wavelengths must be finite and positive")
    return r_ref * (wavelength / lambda_ref) ** (1.0 / beta)


def ss73_brightness_scaled_radius(s, beta=0.75):
    """Dimensionless I_lambda at s=R/R_lambda, up to a constant factor."""
    s = np.asarray(s, dtype=float)
    beta = float(beta)
    if beta <= 0:
        raise ValueError("beta must be positive")
    power = np.maximum(s, 0.0) ** beta
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = 1.0 / np.expm1(power)
    value = np.where(s == 0, np.inf, value)
    return value


def ss73_total_dimensionless_flux(beta=0.75):
    """Return integral_0^inf s/[exp(s^beta)-1] ds = Gamma(2/beta) zeta(2/beta)/beta (Gradshteyn & Ryzhik 3.411.1)."""
    beta = float(beta)
    if beta <= 0 or 2.0 / beta <= 1.0:
        raise ValueError("beta must satisfy 0 < beta < 2 for finite disk flux")
    q = 2.0 / beta
    return float(gamma(q) * zeta(q) / beta)


def ss73_flux_enclosure(s_max, beta=0.75):
    """Fraction of monochromatic flux inside s=R/R_lambda.

    Integrated in t=s**beta, F(<s) ~ integral t**(2/beta-1)/(exp(t)-1) dt, to resolve shallow temperature laws."""
    s_max = float(s_max)
    beta = float(beta)
    if s_max <= 0:
        return 0.0
    total = ss73_total_dimensionless_flux(beta)
    q = 2.0 / beta
    t_max = s_max ** beta

    def integrand(t):
        if t <= 0.0:
            return 0.0
        if t > 700.0:
            return 0.0
        return (t ** (q - 1.0)) / np.expm1(t) / beta

    enclosed = quad(integrand, 0.0, t_max, epsabs=1e-11, epsrel=1e-10, limit=200)[0]
    return float(np.clip(enclosed / total, 0.0, 1.0))


@lru_cache(maxsize=256)
def powerlaw_half_light_factor(beta=0.75):
    """Return R_half/R_lambda for T(R) proportional to R**(-beta) (Mortonson, Schechter & Wambsganss 2005, ApJ 628, 594)."""
    beta = float(beta)
    ss73_total_dimensionless_flux(beta)
    lo, hi = 1e-12, 1.0
    while ss73_flux_enclosure(hi, beta) < 0.5:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError("could not bracket the half-light radius")
    return float(brentq(lambda s: ss73_flux_enclosure(s, beta) - 0.5, lo, hi))


def half_light_from_r_lambda(r_lambda, beta=0.75):
    """Convert ``R_lambda/R_abs`` to ``R_half/R_abs``."""
    r_lambda = float(r_lambda)
    if r_lambda <= 0 or not np.isfinite(r_lambda):
        raise ValueError("R_lambda must be finite and positive")
    return r_lambda * powerlaw_half_light_factor(float(beta))


@lru_cache(maxsize=64)
def ss73_enclosure_radius(beta=0.75, epsilon=1e-4):
    beta = float(beta)
    epsilon = float(epsilon)
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0, 1)")
    target = 1.0 - epsilon
    lo, hi = 0.0, 1.0
    while ss73_flux_enclosure(hi, beta) < target:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError("could not enclose the requested SS73 flux")
    return float(brentq(lambda s: ss73_flux_enclosure(s, beta) - target, lo + 1e-12, hi))


@lru_cache(maxsize=64)
def _polar_rule(beta=0.75, epsilon=1e-4, radial_nodes=32, angular_nodes=32):
    beta = float(beta)
    radial_nodes = int(radial_nodes)
    angular_nodes = int(angular_nodes)
    if radial_nodes < 12 or angular_nodes < 12:
        raise ValueError("SS73 quadrature requires at least 12 radial and angular nodes")
    s_max = ss73_enclosure_radius(beta, epsilon)
    nodes, weights = np.polynomial.legendre.leggauss(radial_nodes)
    s = 0.5 * (nodes + 1.0) * s_max
    ws = 0.5 * s_max * weights
    radial_weight = ws * s / np.expm1(s ** beta)
    theta = 2.0 * np.pi * (np.arange(angular_nodes, dtype=float) + 0.5) / angular_nodes
    spatial_weight = np.repeat(radial_weight / angular_nodes, angular_nodes)
    spatial_weight /= np.sum(spatial_weight)
    s_grid = np.repeat(s, angular_nodes)
    theta_grid = np.tile(theta, radial_nodes)
    return s_grid, theta_grid, spatial_weight, s_max


class AchromaticSource:
    name = "achromatic"
    label = "Achromatic"

    def radius(self, wavelength):
        return np.ones_like(np.asarray(wavelength, dtype=float))


class ThinDiskSource:
    # Stored as ss73; beta=0.75 is the Shakura & Sunyaev 1973 (A&A 24, 337) disk.
    name = "ss73"

    @property
    def label(self):
        return source_model_label(self.name, self.beta)

    def __init__(
        self,
        beta=0.75,
        r_ref=1.0,
        lambda_ref=2400.0,
        enclosure_epsilon=1e-4,
        radial_nodes=32,
        angular_nodes=32,
    ):
        self.beta = float(beta)
        self.r_ref = float(r_ref)
        self.lambda_ref = float(lambda_ref)
        self.enclosure_epsilon = float(enclosure_epsilon)
        self.radial_nodes = int(radial_nodes)
        self.angular_nodes = int(angular_nodes)
        disk_radius(self.lambda_ref, self.beta, self.r_ref, self.lambda_ref)
        ss73_total_dimensionless_flux(self.beta)

    def radius(self, wavelength):
        return disk_radius(wavelength, self.beta, self.r_ref, self.lambda_ref)

    def flux_enclosure(self):
        s_max = ss73_enclosure_radius(self.beta, self.enclosure_epsilon)
        return ss73_flux_enclosure(s_max, self.beta)

    def flux_fraction_inside_radius(self, radius, wavelength):
        radius = float(radius)
        if radius <= 0:
            return 0.0
        r_lambda = float(self.radius(wavelength))
        return ss73_flux_enclosure(radius / r_lambda, self.beta)

    def brightness(self, radius, wavelength):
        radius = np.asarray(radius, dtype=float)
        r_lambda = float(self.radius(wavelength))
        return ss73_brightness_scaled_radius(radius / r_lambda, self.beta)

    def total_flux(self, wavelength):
        r_lambda = float(self.radius(wavelength))
        return 2.0 * np.pi * r_lambda * r_lambda * ss73_total_dimensionless_flux(self.beta)

    def quadrature(self, wavelength):
        """Return a flux-enclosed polar rule for maps and diagnostics."""
        wavelength = float(wavelength)
        r_lambda = float(self.radius(wavelength))
        s, theta, weight, _ = _polar_rule(
            self.beta,
            self.enclosure_epsilon,
            self.radial_nodes,
            self.angular_nodes,
        )
        r = r_lambda * s
        return r * np.cos(theta), r * np.sin(theta), weight.copy()


def source_from_fit(fit):
    mode = source_mode_name(getattr(getattr(fit, "srcmode", None), "val", 0))
    if mode == "achromatic":
        return AchromaticSource()
    return ThinDiskSource(
        beta=fit.getValue("srcbeta") if hasattr(fit, "srcbeta") else 0.75,
        r_ref=fit.getValue("srcrref") if hasattr(fit, "srcrref") else 1.0,
        lambda_ref=fit.getValue("srclref") if hasattr(fit, "srclref") else 2400.0,
    )


def source_wavelengths_from_observed(x, fit):
    """Use an absorber-rest wavelength proxy; only wavelength ratios matter here."""
    x = np.asarray(x, dtype=float)
    redshifts = []
    for system in getattr(fit, "sys", []):
        try:
            redshifts.append(float(system.z.val))
        except Exception:
            pass
    z_ref = float(np.median(redshifts)) if redshifts else 0.0
    return x / (1.0 + z_ref), z_ref
