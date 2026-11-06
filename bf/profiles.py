"""Velocities in km/s; wavelengths in Angstrom; columns in cm^-2."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad_vec
from numpy.polynomial.hermite import hermgauss

# line-centre optical depth tau0 = KTAU f lambda N / b, KTAU = sqrt(pi) e^2/(m_e c) for lambda in A, N in cm^-2, b in km/s
# (Spitzer 1978, Physical Processes in the Interstellar Medium)
KTAU = 1.497e-15
C_KMS = 299792.458


@dataclass
class GaussianLSF:
    resolving_power: float | np.ndarray            # common R, or one R per BF line (nodes then have shape (lines, quadrature))
    quadrature: int | None = None

    def __post_init__(self):
        if np.ndim(self.resolving_power): self.resolving_power = np.asarray(self.resolving_power, float)
        R = np.asarray(self.resolving_power)
        if R.ndim > 1 or not np.isfinite(R).all() or np.any(R <= 0) or (self.quadrature is not None and self.quadrature < 4):
            raise ValueError("Positive resolving power and >=4 quadrature nodes required")
        if self.quadrature is None:
            self.nodes = self.weights = None
            return
        x, w = hermgauss(self.quadrature)   # Gauss-Hermite nodes (Abramowitz & Stegun 1964, 25.4.46)
        self.nodes = np.sqrt(2)*self.sigma*x if R.ndim == 0 else np.sqrt(2)*self.sigma[:, None]*x
        self.weights = w/np.sqrt(np.pi)

    @property
    def sigma(self): return C_KMS/self.resolving_power/(2*np.sqrt(2*np.log(2)))

    def evaluate(self, intensity, velocity):
        """Integrate intensity directly, including blend/covering geometry before convolution."""
        if self.quadrature is None:
            return quad_vec(lambda x: intensity(np.asarray(velocity)+self.sigma*x)*np.exp(-x*x/2)/np.sqrt(2*np.pi),
                            -10, 10, epsabs=1e-11, epsrel=1e-10)[0]
        return np.asarray(intensity(np.asarray(velocity)[..., None]+self.nodes)) @ self.weights
