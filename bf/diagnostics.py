"""Inspection quantities of a BF fit (forward model, problem with unpack(x) -> (q, covering parameters, b), fit result).
q_j = log10(N_j/b_j) with N in cm^-2 and b in km/s; wavelengths in Angstrom, E_j in cm^-1."""
import numpy as np
from astropy import units as u
from scipy.special import expit

from .data import BFData, Transition
from .models import BFModel
from .profiles import KTAU, GaussianLSF

HC_K = 1.4387769   # second radiation constant hc/k, cm K
DEEPER = 'deeper than the model allows (probable blend)'
SHALLOWER = 'shallower than predicted (check continuum/emission)'


def unpack(problem, result):
    """(q, covering parameters, b) at the fit optimum, as float arrays."""
    return tuple(np.asarray(a, float) for a in problem.unpack(np.asarray(result['parameters'], float)))


def propagate(problem, result, fn):
    """fn(x) at the optimum and its 1 sigma from the fit covariance (central differences, h = 1e-6 max(1, |x|), clipped to
    the bounds); NaN without a covariance."""
    x = np.asarray(result['parameters'], float)
    y = np.asarray(fn(x), float)
    cov = result.get('covariance')
    if cov is None: return y, np.full(y.shape, np.nan)
    J = np.empty(y.shape + (len(x),))
    for k in range(len(x)):
        h = 1e-6*max(1., abs(x[k])); xp, xm = x.copy(), x.copy()
        xp[k] = min(x[k]+h, problem.upper[k]); xm[k] = max(x[k]-h, problem.lower[k])
        J[..., k] = (np.asarray(fn(xp), float)-np.asarray(fn(xm), float))/(xp[k]-xm[k])
    return y, np.sqrt(np.maximum(np.einsum('...i,ij,...j->...', J, cov, J), 0.))


def transmission(forward, q, b):
    """Convolved line-centre transmission of every line of forward.data at full coverage (C = 1)."""
    return BFModel(forward.data, lambda z: 1., forward.lsf).predict(q, None, b)


def tau0(forward, q):
    """Line-centre optical depth of every line for columns q (the Gaussian tau0 of the BF model)."""
    return KTAU*10**np.asarray(q)[forward.data.j]*forward.data.flambda


def level_design(forward, level):
    """(design-matrix row, mean wavelength) of `level`: its covering-law row with the wavelength term (if the law has one)
    at the mean rest wavelength of the level's lines."""
    d, law = forward.data, forward.covering
    i = np.flatnonzero(d.j == d.levels.index(level))
    row, lam = law.matrix[i[0]].copy(), float(d.wavelength[i].mean())
    if 'B_lambda' in law.names:
        k = law.names.index('B_lambda')
        row[k] = law.matrix[i[0], k]+np.log(lam/d.wavelength[i[0]])
    return row, lam


def covering_curve(forward, problem, result, level, wavelength):
    """Fitted covering of `level` at rest wavelengths `wavelength` (Angstrom); constant for laws without a wavelength term."""
    d, law = forward.data, forward.covering
    row = level_design(forward, level)[0]
    X = np.repeat(row[None, :], len(np.atleast_1d(wavelength)), axis=0)
    if 'B_lambda' in law.names:
        k, i = law.names.index('B_lambda'), np.flatnonzero(d.j == d.levels.index(level))[0]
        X[:, k] = law.matrix[i, k]+np.log(np.atleast_1d(wavelength)/d.wavelength[i])
    return expit(X @ unpack(problem, result)[1])


def level_table(forward, problem, result, energy, g):
    """Per level of forward.data: E_j, g_j, b_j, log10 N_j/g_j and C_j (at the level's mean wavelength) with 1 sigma.
    energy, g: {level: E_j}, {level: g_j}."""
    levels = list(forward.data.levels)
    X = np.array([level_design(forward, l)[0] for l in levels])
    gj = np.array([g[l] for l in levels], float)

    def log_ng(x):
        q, _, b = problem.unpack(x)
        return np.asarray(q)+np.log10(b)-np.log10(gj)
    n, n_err = propagate(problem, result, log_ng)
    c, c_err = propagate(problem, result, lambda x: expit(X @ problem.unpack(x)[1]))
    return dict(level=levels, E_j=np.array([energy[l] for l in levels], float), g_j=gj, b=unpack(problem, result)[2],
                logN_g=n, logN_g_err=n_err, C=c, C_err=c_err,
                lambda_mean=np.array([level_design(forward, l)[1] for l in levels]))


def lte_log_ng(log_ng0, energy, temperature):
    """Boltzmann log10 N_j/g_j through log10 N_0/g_0 at E_0 = energy[0]."""
    e = np.asarray(energy, float)
    return log_ng0-HC_K*(e-e[0])/(temperature*np.log(10))


def _probe(data, level, flambda, wavelength):
    """BFData of synthetic lines of `level` at the given f*lambda (model curves only; intensities are placeholders)."""
    tr = tuple(Transition(id=f'probe{i}', species='probe', ion_stage=0, wavelength=wavelength*u.AA, f=float(x/wavelength),
                          lower=level) for i, x in enumerate(flambda))
    n = len(tr)
    return BFData(tr, np.full(n, .5), np.ones(n), ('probe',)*n, ((0., 0.),)*n, 'model curve', {}, levels=data.levels)


def curve_of_growth(forward, problem, result, level, flambda):
    """Model BF of `level` along f*lambda (Angstrom) for the fitted column and b: with the fitted covering at the level's
    mean wavelength (partial) and with C = 1 (full). The LSF is the median R of the level's lines."""
    d = forward.data
    q, z, b = unpack(problem, result)
    row, lam = level_design(forward, level)
    c = float(expit(row @ z))
    lsf = forward.lsf
    if lsf is not None:
        R = np.atleast_1d(lsf.resolving_power)
        R = R[d.j == d.levels.index(level)] if len(R) == len(d.intensity) else R
        lsf = GaussianLSF(float(np.median(R)), lsf.quadrature)
    t = BFModel(_probe(d, level, np.asarray(flambda, float), lam), lambda z: 1., lsf).predict(q, None, b)
    return dict(flambda=np.asarray(flambda, float), bf_partial=1-c+c*t, bf_full=t, C=c, lambda_mean=lam)


def implied_covering(forward, problem, result):
    """Per line: F_i (convolved line-centre transmission at full coverage for the fitted column and b), the implied covering
    C_i = (1 - BF_i)/(1 - F_i) (Barlow & Sargent 1997, AJ 113, 136) and sigma_C = sigma_i/(1 - F_i)."""
    q, _, b = unpack(problem, result)
    F = transmission(forward, q, b)
    d = forward.data
    with np.errstate(divide='ignore', invalid='ignore'):
        return dict(F=F, C=(1-d.intensity)/(1-F), sigma_C=d.uncertainty/(1-F))


def check_lines(data, make, n_sigma=4., tau_min=.01, cancelled=lambda: False, progress=lambda i, n: None):
    """Leave-one-line-out check, make(data) -> (forward, problem, result): z = (BF_obs - BF_pred)/sqrt(sigma_obs^2 + sigma_pred^2)
    from the refit without each line; |z| > n_sigma and tau0 < tau_min are flagged. None if cancelled()."""
    forward, problem, result = make(data)
    t0 = tau0(forward, unpack(problem, result)[0])
    n = len(data.intensity)
    z, pred, spred = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    flags = [[] for _ in range(n)]
    for i in range(n):
        if cancelled(): return None
        if np.sum(data.j == data.j[i]) > 1:
            _, _, r = make(data.subset(np.arange(n) != i))
            y, s = propagate(problem, r, lambda x: problem.predict(x)[i:i+1])
            pred[i], spred[i] = y[0], s[0]
            z[i] = (data.intensity[i]-pred[i])/np.hypot(data.uncertainty[i], np.nan_to_num(spred[i]))
            if z[i] < -n_sigma: flags[i].append(DEEPER)
            if z[i] > n_sigma: flags[i].append(SHALLOWER)
        else: flags[i].append('only line of its level: not refitted')
        if t0[i] < tau_min: flags[i].append(f'uninformative (tau0 = {t0[i]:.3g} < {tau_min:g})')
        progress(i+1, n)
    return dict(id=[t.id for t in data.transitions], level=[t.lower for t in data.transitions], lambda_rest=data.wavelength,
                z=z, bf_pred=pred, sigma_pred=spred, tau0=t0, flag=['; '.join(f) for f in flags])


def column_covering_grid(forward, problem, result, level, g, n=81, n_sigma=5.):
    """Delta chi2 of `level` (M1, b at its best value) on an n x n grid of (log10 N_j/g_j, C_j) within the optimum +- n_sigma
    sigma, clipped to the bounds; computed from the level's lines alone. delta_chi2[i, k] belongs to (logN_g[i], C[k])."""
    d = forward.data
    if forward.covering.name != 'M1': raise ValueError('Column-covering grids need Covering per level (M1)')
    if d.covariance is not None: raise ValueError('Levels are not independent with a BF covariance')
    j, names = d.levels.index(level), list(result['names'])
    iq, ia = names.index('q:'+level), names.index('A:'+level)
    x = np.asarray(result['parameters'], float)
    cov = result.get('covariance')
    err = np.full(len(x), np.nan) if cov is None else np.sqrt(np.maximum(np.diag(cov), 0.))
    lo, hi = problem.lower, problem.upper
    ok = lambda k: np.isfinite(err[k]) and err[k] > 0
    q_axis = np.linspace(*((max(lo[iq], x[iq]-n_sigma*err[iq]), min(hi[iq], x[iq]+n_sigma*err[iq])) if ok(iq)
                           else (lo[iq], hi[iq])), n)
    c0 = float(expit(x[ia]))
    c_lo, c_hi = float(expit(lo[ia])), float(expit(hi[ia]))
    if ok(ia): c_lo, c_hi = max(c_lo, c0-n_sigma*c0*(1-c0)*err[ia]), min(c_hi, c0+n_sigma*c0*(1-c0)*err[ia])
    c_axis = np.linspace(c_lo, c_hi, n)
    q, _, b = unpack(problem, result)
    m = d.j == j
    y, s = d.intensity[m], d.uncertainty[m]
    F = np.array([transmission(forward, np.where(np.arange(len(q)) == j, v, q), b)[m] for v in q_axis])
    chi = ((((y-(1-c_axis[None, :, None]+c_axis[None, :, None]*F[:, None, :]))/s)**2).sum(-1))
    best = float((((y-(1-c0+c0*transmission(forward, q, b)[m]))/s)**2).sum())
    shift = np.log10(b[j])-np.log10(g)
    return dict(level=level, logN_g=q_axis+shift, C=c_axis, delta_chi2=chi-min(best, chi.min()), b=float(b[j]),
                best_logN_g=float(x[iq]+shift), best_C=c0)
