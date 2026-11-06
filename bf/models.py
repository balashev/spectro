"""Effective BF model and explicit spatial-overlap spectral model."""
from dataclasses import dataclass
import numpy as np
from .profiles import KTAU, GaussianLSF


@dataclass
class BFModel:
    data: object
    covering: object
    lsf: GaussianLSF | None = None

    def __post_init__(self):
        if self.lsf is not None and np.ndim(self.lsf.resolving_power) and len(self.lsf.resolving_power) != len(self.data.intensity):
            raise ValueError("Per-line resolving power needs one value per BF line")

    def predict(self, q, z, b):
        q, b = np.asarray(q), np.asarray(b)
        nl = len(self.data.levels)
        if q.shape != (nl,) or b.shape != (nl,) or not np.isfinite(b).all() or np.any(b <= 0):
            raise ValueError("One q and positive width per lower level required")
        tau = KTAU*10**q[self.data.j]*self.data.flambda
        if self.lsf is None: t = np.exp(-tau)
        elif self.lsf.quadrature is None:
            t = self.lsf.evaluate(lambda v: np.exp(-tau*np.exp(-(v/b[self.data.j])**2)), np.zeros(len(tau)))
        else:
            nodes = self.lsf.nodes if self.lsf.nodes.ndim == 2 else self.lsf.nodes[None, :]
            shape = np.exp(-(nodes/b[self.data.j, None])**2)
            t = np.exp(-tau[:, None]*shape) @ self.lsf.weights
        c = self.covering(z)
        return 1-c+c*t   # partial covering (Barlow & Sargent 1997, AJ 113, 136)
