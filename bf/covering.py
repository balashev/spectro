"""Logistic covering laws with an explicit level coordinate for M3/M4. The covering factor C enters as
BF = 1 - C + C exp(-tau) (Barlow & Sargent 1997, AJ 113, 136)."""
from dataclasses import dataclass
import numpy as np
from astropy import units as u
from scipy.special import expit, logit

MODEL_NAMES = {'M0': 'Single covering', 'M1': 'Covering per level', 'M2': 'Per level + wavelength trend',
               'M3': 'Excitation trend', 'M4': 'Excitation + wavelength trend'}
MODEL_FORMULAS = {'M0': 'one C for all lines: C = expit(A)',
                  'M1': 'free C_j per lower level: C_j = expit(A_j)',
                  'M2': 'C_j plus a common wavelength slope: C = expit(A_j + B_lambda*ln(lambda/lambda_0))',
                  'M3': 'C = expit(A + B_level*E_j/1000 cm^-1)',
                  'M4': 'M3 plus the wavelength slope: C = expit(A + B_level*E_j/1000 cm^-1 + B_lambda*ln(lambda/lambda_0))'}
WAVELENGTH_LAWS = ('M2', 'M4')   # laws with the wavelength term B_lambda*ln(lambda/lambda_0)
LEVEL_LAWS = ('M3', 'M4')        # excitation laws: the level term B_level*(level coordinate)


def display_name(code, with_code=False):
    """Display name of covering law `code` ('M0'..'M4'); with_code adds ' (M3)'."""
    return MODEL_NAMES[code]+(f' ({code})' if with_code else '')


@dataclass
class CoveringLaw:
    name: str
    matrix: np.ndarray
    names: tuple[str, ...]
    initial: np.ndarray
    description: str
    lambda_0: float | None = None   # reference wavelength (Angstrom) of the wavelength term; None without one

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, float)
        self.initial = np.asarray(self.initial, float)
        if self.matrix.ndim != 2 or self.matrix.shape[1] != len(self.names) or self.initial.shape != (len(self.names),):
            raise ValueError("Invalid covering design matrix")
        if not np.isfinite(self.matrix).all():
            raise ValueError("Nonfinite covering coordinates")

    @property
    def n_parameters(self): return len(self.names)

    def __call__(self, parameters):
        return expit(self.matrix @ np.asarray(parameters))


def level_coordinate(data, kind):
    """'index': absolute level number j from SPECTRO level names (FeII -> 0, FeIIjN -> N; position among present levels if
    the names do not all follow that pattern for one ion); 'energy': lower-level E_j / 1000 cm^-1."""
    if kind == 'index':
        import re
        m = [re.fullmatch(r'(.*?)(?:j(\d+))?', l) for l in data.levels]
        if len({x.group(1) for x in m}) == 1: return np.array([int(x.group(2) or 0) for x in m], dtype=float)
        return np.arange(len(data.levels), dtype=float)
    if kind != 'energy': raise ValueError("Level coordinate must be 'index', 'energy' or an array")
    energy = {}
    for t in data.transitions:
        if t.energy is None: raise ValueError("Energy coordinate requires Transition.energy for every line")
        e = t.energy.to_value(u.cm**-1, equivalencies=u.spectral())
        if energy.setdefault(t.lower, e) != e: raise ValueError(f"Inconsistent energy for level {t.lower}")
    return np.array([energy[level] for level in data.levels])/1000.


def reference_wavelength(data, lambda_0='auto'):
    """lambda_0 (Angstrom) of the wavelength term: 'auto' is the geometric mean of the rest wavelengths rounded to 10 A.
    Changing lambda_0 only shifts the intercepts (A, A_j) by B_lambda*ln(lambda_0'/lambda_0)."""
    if isinstance(lambda_0, str):
        if lambda_0 != 'auto': raise ValueError("lambda_0 must be 'auto' or a wavelength in Angstrom")
        return float(np.round(np.exp(np.mean(np.log(data.wavelength)))/10.)*10.)
    value = float(lambda_0)
    if not np.isfinite(value) or value <= 0: raise ValueError("lambda_0 must be a positive wavelength in Angstrom")
    return value


def covering_law(name, data, *, coordinate='index', lambda_0='auto'):
    """M3/M4 level coordinate: 'index', 'energy', or one value per level. lambda_0: reference wavelength
    of the M2/M4 wavelength term ('auto' or Angstrom, see reference_wavelength); stored as CoveringLaw.lambda_0."""
    j = data.j
    n, nl = len(j), len(data.levels)
    one = np.ones((n, 1))
    independent = np.eye(nl)[j]
    lam0 = reference_wavelength(data, lambda_0) if name in WAVELENGTH_LAWS else None
    wavelength = None if lam0 is None else np.log(data.wavelength/lam0)[:, None]
    a = logit(.58)
    if name == 'M0': X, names, start = one, ('A',), [a]
    elif name == 'M1': X, names, start = independent, tuple('A:'+s for s in data.levels), [a]*nl
    elif name == 'M2': X, names, start = np.c_[independent, wavelength], tuple('A:'+s for s in data.levels)+('B_lambda',), [a]*nl+[0.]
    elif name in ('M3','M4'):
        if isinstance(coordinate, str): coordinate = level_coordinate(data, coordinate)
        if np.shape(coordinate) != (nl,):
            raise ValueError("M3/M4 need one level coordinate per level")
        X, names, start = np.c_[one, np.asarray(coordinate)[j]], ('A','B_level'), [a,-.16]
        if name == 'M4': X, names, start = np.c_[X, wavelength], names+('B_lambda',), start+[0.]
    else: raise ValueError(name)
    text = "expit(X z)" + ("" if lam0 is None else f"; wavelength reference lambda_0 = {lam0:g} Angstrom")
    return CoveringLaw(name, X, names, start, text, lam0)
