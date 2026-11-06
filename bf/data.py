"""Scientific inputs. Arrays use explicit units; masks are True for excluded pixels."""
from dataclasses import dataclass, field
import hashlib
import numpy as np
from astropy import units as u


@dataclass(frozen=True)
class Transition:
    id: str
    species: str
    ion_stage: int
    wavelength: u.Quantity
    f: float
    lower: str
    upper: str | None = None
    gamma: u.Quantity | None = None
    energy: u.Quantity | None = None
    g: float | None = None
    source: str = "unspecified"
    version: str = "unspecified"

    def __post_init__(self):
        if not np.isfinite(self.wavelength.to_value(u.AA)) or self.wavelength <= 0*u.AA:
            raise ValueError("Positive finite rest wavelength required")
        if not np.isfinite(self.f) or self.f < 0:
            raise ValueError("Oscillator strength must be finite and nonnegative")
        if self.gamma is not None and (not np.isfinite(self.gamma.to_value(1/u.s)) or self.gamma < 0/u.s):
            raise ValueError("Nonnegative damping rate required")


@dataclass
class Spectrum:
    wavelength: u.Quantity
    flux: np.ndarray
    uncertainty: np.ndarray
    frame: str
    wavelength_medium: str = "unspecified"
    normalized: bool = True
    continuum: np.ndarray | float = 1.
    continuum_uncertainty: np.ndarray | float | None = None
    mask: np.ndarray | None = None
    resolution: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.wavelength = u.Quantity(self.wavelength).to(u.AA)
        self.flux = np.asarray(self.flux, float).copy()
        self.uncertainty = np.asarray(self.uncertainty, float).copy()
        n = len(self.wavelength)
        if self.flux.shape != (n,) or self.uncertainty.shape != (n,) or np.any(np.diff(self.wavelength.value) <= 0):
            raise ValueError("Spectrum requires matching ordered 1D arrays")
        if not np.isfinite(self.wavelength.value).all():
            raise ValueError("Nonfinite wavelength")
        self.mask = np.zeros(n, bool) if self.mask is None else np.asarray(self.mask, bool).copy()
        if self.mask.shape != (n,):
            raise ValueError("Mask shape mismatch")
        invalid = ~np.isfinite(self.flux) | ~np.isfinite(self.uncertainty) | (self.uncertainty <= 0)
        self.metadata = dict(self.metadata)
        self.metadata["invalid_pixels_masked"] = int(np.sum(invalid & ~self.mask))
        self.mask |= invalid
        for name in ("continuum", "continuum_uncertainty"):
            x = getattr(self, name)
            if x is not None:
                x = np.broadcast_to(np.asarray(x, float), (n,)).copy()
                if np.any(~np.isfinite(x)) or np.any(x < 0) or (name == "continuum" and np.any(x == 0)):
                    raise ValueError(f"Invalid {name}")
                setattr(self, name, x)
        if self.resolution is not None and self.resolution <= 0:
            raise ValueError("Positive resolving power required")

    def velocity(self, transition, redshift):
        return 299792.458 * (self.wavelength.to_value(u.AA) /
                            (transition.wavelength.to_value(u.AA)*(1+redshift))-1)


@dataclass
class BFData:
    transitions: tuple[Transition, ...]
    intensity: np.ndarray
    uncertainty: np.ndarray
    quality: tuple[str, ...]
    windows: tuple[tuple[float, float] | None, ...]
    method: str
    provenance: dict
    blended: tuple[bool | None, ...] = ()
    saturated: tuple[bool | None, ...] = ()
    levels: tuple[str, ...] | None = None
    covariance: np.ndarray | None = None

    def __post_init__(self):
        self.transitions = tuple(self.transitions)
        n = len(self.transitions)
        self.intensity = np.asarray(self.intensity, float).copy()
        self.uncertainty = np.asarray(self.uncertainty, float).copy()
        if n == 0 or self.intensity.shape != (n,) or self.uncertainty.shape != (n,):
            raise ValueError("Matching nonempty BF arrays required")
        if not np.isfinite(self.intensity).all() or not np.isfinite(self.uncertainty).all() or np.any(self.uncertainty <= 0):
            raise ValueError("Finite intensities and positive errors required")
        if len(self.quality) != n or len(self.windows) != n:
            raise ValueError("Every BF needs classification and window metadata")
        if len({t.id for t in self.transitions}) != n:
            raise ValueError("Transition IDs must be unique")
        self.levels = tuple(dict.fromkeys(t.lower for t in self.transitions)) if self.levels is None else tuple(self.levels)
        if len(set(self.levels)) != len(self.levels) or any(t.lower not in self.levels for t in self.transitions):
            raise ValueError("Invalid level mapping")
        self.blended = self.blended or (None,)*n
        self.saturated = self.saturated or (None,)*n
        if len(self.blended) != n or len(self.saturated) != n:
            raise ValueError("Flag length mismatch")
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, float).copy()
            if self.covariance.shape != (n, n) or not np.allclose(self.covariance, self.covariance.T):
                raise ValueError("Covariance must be symmetric")
            np.linalg.cholesky(self.covariance)
            if not np.allclose(np.diag(self.covariance), self.uncertainty**2):
                raise ValueError("Covariance diagonal disagrees with uncertainty")
        self.intensity.setflags(write=False)
        self.uncertainty.setflags(write=False)

    @property
    def j(self):
        return np.array([self.levels.index(t.lower) for t in self.transitions])

    @property
    def wavelength(self):
        return np.array([t.wavelength.to_value(u.AA) for t in self.transitions])

    @property
    def flambda(self):
        return self.wavelength*np.array([t.f for t in self.transitions])

    @property
    def fingerprint(self):
        import json
        values = {"ids": [t.id for t in self.transitions], "levels": self.levels,
                  "wavelength": self.wavelength.tolist(), "flambda": self.flambda.tolist(),
                  "y": self.intensity.tolist(), "sigma": self.uncertainty.tolist(),
                  "quality": self.quality, "windows": self.windows, "method": self.method,
                  "blended": self.blended, "saturated": self.saturated,
                  "covariance": None if self.covariance is None else self.covariance.tolist()}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def subset(self, indexes):
        ix = np.arange(len(self.intensity))[indexes]
        ix = np.atleast_1d(ix)
        def select(x): return tuple(x[i] for i in ix)
        return BFData(select(self.transitions), self.intensity[ix], self.uncertainty[ix],
                      select(self.quality), select(self.windows), self.method, dict(self.provenance),
                      select(self.blended), select(self.saturated), self.levels,
                      None if self.covariance is None else self.covariance[np.ix_(ix, ix)])
