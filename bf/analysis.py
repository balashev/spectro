"""BF analyses on in-memory BFData: fits, parameter errors, per-level covering, population-grid fits, model comparison and
leave-one-out. Physics settings (b, level_b, resolving_power, quadrature, coordinate, lambda_0) go to experiment.make_problem."""
import numpy as np
import pandas as pd
from astropy import units as u
from scipy.special import expit
from scipy.stats import chi2 as chi2_dist, norm
from . import experiment as exp
from .fitting import fit
from .atomic import fit_population_grid
from .covering import display_name

MODELS = ('M0', 'M1', 'M2', 'M3', 'M4')


def fit_case(model, width, coordinate, *, data, **kw):
    """(forward model, problem, fit result) for experiment.make_problem(model, width, ...); kw: b, level_b, resolving_power,
    quadrature, convolved, lambda_0."""
    forward, problem = exp.make_problem(model, width, data=data, coordinate=coordinate, **kw)
    return forward, problem, fit(problem)


def level_covering(forward, result):
    """Per-level covering at lambda_0 of the law (B_lambda term zero; forward.covering.lambda_0, None for laws without a
    wavelength term) with delta-method 1sigma: level, energy_cm1 (NaN for a line without Transition.energy), C, C_err."""
    d, law = forward.data, forward.covering
    names = list(result['names'])
    cols = [names.index(n) for n in law.names]
    first = [int(np.flatnonzero(d.j == i)[0]) for i in range(len(d.levels))]
    X = law.matrix[first].copy()
    if 'B_lambda' in law.names: X[:, law.names.index('B_lambda')] = 0.
    c = expit(X @ result['parameters'][cols])
    if result['covariance'] is None: err = np.full(len(c), np.nan)
    else:
        g = (c*(1-c))[:, None]*X
        err = np.sqrt(np.einsum('ij,jk,ik->i', g, result['covariance'][np.ix_(cols, cols)], g))
    energy = [np.nan if d.transitions[i].energy is None else d.transitions[i].energy.to_value(1/u.cm) for i in first]
    return pd.DataFrame(dict(level=d.levels, energy_cm1=energy, C=c, C_err=err))


def best_node(grid, afit):
    """{axis: value} at the best node."""
    return {k: float(v[i]) for (k, v), i in zip(grid.axes.items(), afit['best_index'])}


def grid_fit(grid, model, width, *, data, **kw):
    """Population-grid fit of `model`, `width` on `data` (kw as fit_case, including coordinate)."""
    forward, base = exp.make_problem(model, width, data=data, **kw)
    return fit_population_grid(forward, grid, fixed_b=base.metadata['fixed_b_kms'])


def chi2_map(grid, afit):
    """Delta chi2(log T, log n_e) relative to the grid minimum, minimised over any further axis (radiation)."""
    delta = afit['chi2_grid']-afit['chi2']
    delta = delta.reshape(delta.shape[:2]+(-1,)).min(axis=2)
    return dict(log_T=np.log10(grid.axes['temperature']), log_ne=np.log10(grid.axes['electron_density']), delta_chi2=delta,
                best=tuple(int(i) for i in afit['best_index'][:2]), chi2_min=afit['chi2'])


def _loo_rows(data, chi_full, refit):
    rows = []
    for i, tr in enumerate(data.transitions):
        keep = np.arange(len(data.intensity)) != i
        chi = refit(data.subset(keep))
        rows.append(dict(level=tr.lower, lambda_rest=tr.wavelength.value, chi2=chi, delta_chi2=chi_full-chi))
    return pd.DataFrame(rows)


def leave_one_out(model, width, coordinate, *, data, **kw):
    """Refit without each line (all starts, incl. the M1 seed on the reduced data); delta_chi2 = chi2(all) - chi2(without line)."""
    full = fit_case(model, width, coordinate, data=data, **kw)[2]['chi2']
    return _loo_rows(data, full, lambda d: fit_case(model, width, coordinate, data=d, **kw)[2]['chi2'])


def leave_one_out_grid(grid, model, width, *, data, **kw):
    """As leave_one_out, re-searching the whole population grid for every omitted line."""
    full = grid_fit(grid, model, width, data=data, **kw)['chi2']
    return _loo_rows(data, full, lambda d: grid_fit(grid, model, width, data=d, **kw)['chi2'])


def lr_sigma(dchi2, dof=1):
    """Likelihood-ratio significance in Gaussian sigma: two-sided normal quantile of the chi2(dof) tail probability of dchi2
    (Wilks 1938, Ann. Math. Stat. 9, 60); sqrt(dchi2) for dof = 1."""
    x = max(float(dchi2), 0.)
    return float(np.sqrt(x)) if dof == 1 else float(norm.isf(chi2_dist.sf(x, dof)/2))


def compare_models(data, widths=('common_fixed',), models=MODELS, coordinate='energy', **kw):
    """Covering laws x widths on BFData (kw as experiment.make_problem) sorted by dBIC; lr_sigma against M0 with the same width,
    dof = k - k(M0). Returns (table, {(code, width): (forward, problem, result)})."""
    fits = {(m, w): fit_case(m, w, coordinate, data=data, **kw) for w in widths for m in dict.fromkeys(('M0',)+tuple(models))}
    rows = []
    for (m, w), (_, _, r) in fits.items():
        if m not in models: continue
        r0 = fits['M0', w][2]
        dof = r['k']-r0['k']
        rows.append(dict(code=m, name=display_name(m), width=w, k=r['k'], chi2=r['chi2'], BIC=r['bic'], dof=dof,
                         lr_sigma=lr_sigma(r0['chi2']-r['chi2'], dof) if dof > 0 else np.nan))
    t = pd.DataFrame(rows)
    t.insert(6, 'dBIC', t.BIC-t.BIC.min())
    return t.sort_values('dBIC', kind='stable', ignore_index=True), fits


def line_loo(data, model, width, coordinate='energy', grid=None, **kw):
    """Leave-one-out on in-memory BFData: id, level, lambda_rest, chi2, delta_chi2 = chi2(all) - chi2(without the line);
    with `grid` (PopulationGrid) the grid is searched again for every omitted line."""
    if grid is None: t = leave_one_out(model, width, coordinate, data=data, **kw)
    else: t = leave_one_out_grid(grid, model, width, data=data, coordinate=coordinate, **kw)
    t.insert(0, 'id', [tr.id for tr in data.transitions])
    return t
