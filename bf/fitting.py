"""Bounded least-squares fits with a Gaussian likelihood, for BF values or spectral pixels."""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve_triangular


# AIC: Akaike 1974, IEEE Trans. Autom. Control 19, 716; AICc: Hurvich & Tsai 1989, Biometrika 76, 297;
# BIC: Schwarz 1978, Ann. Stat. 6, 461.
def information_criteria(chi2, n, k):
    if n < 1 or k < 0: raise ValueError("Invalid parameter or observation count")
    return dict(chi2=float(chi2), n=int(n), k=int(k), aic=float(chi2+2*k),
                aicc=float(chi2+2*k+2*k*(k+1)/(n-k-1)) if n > k+1 else None,
                bic=float(chi2+k*np.log(n)))


@dataclass
class FitProblem:
    predict: object
    y: np.ndarray
    sigma: np.ndarray
    names: tuple[str, ...]
    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    metadata: dict = field(default_factory=dict)
    starts: tuple = ()
    covariance: np.ndarray | None = None

    def __post_init__(self):
        for key in ('y','sigma','initial','lower','upper'):
            setattr(self, key, np.asarray(getattr(self, key), float))
        k = len(self.names)
        if any(x.shape != (k,) for x in (self.initial, self.lower, self.upper)) or np.any(self.lower >= self.upper):
            raise ValueError("Invalid parameter bounds")
        if self.y.ndim != 1 or self.y.size == 0 or self.sigma.shape != self.y.shape or np.any(self.sigma <= 0):
            raise ValueError("Invalid Gaussian observations")
        if not all(np.isfinite(x).all() for x in (self.y,self.sigma,self.initial,self.lower,self.upper)):
            raise ValueError("Finite inputs and bounds required")
        self.cholesky = None
        if self.covariance is not None:
            cov = np.asarray(self.covariance, float)
            if cov.shape != (len(self.y),len(self.y)) or not np.allclose(cov,cov.T):
                raise ValueError("Invalid covariance")
            if not np.allclose(np.diag(cov), self.sigma**2): raise ValueError("Error/covariance mismatch")
            self.cholesky = np.linalg.cholesky(cov)

    def residual(self, p):
        r = np.asarray(self.predict(p))-self.y
        if r.shape != self.y.shape or not np.isfinite(r).all(): raise ValueError("Invalid model predictions")
        return r/self.sigma if self.cholesky is None else solve_triangular(self.cholesky, r, lower=True)


def fit(problem, *, start=None, max_nfev=2500, tolerance=1e-10):
    """Trust-region fits from every start, with each x_scale in problem.metadata['x_scales'] (default 'jac'); lowest chi2 kept."""
    starts = [np.asarray(start)] if start is not None else [problem.initial, *problem.starts]
    scales = problem.metadata.get('x_scales', ('jac',))
    solutions = [least_squares(problem.residual, np.clip(x, problem.lower+1e-8, problem.upper-1e-8),
                    bounds=(problem.lower,problem.upper), x_scale=xs, max_nfev=max_nfev,
                    ftol=tolerance, xtol=tolerance, gtol=tolerance) for x in starts for xs in scales]
    sol = min(solutions, key=lambda s: float(s.fun@s.fun))
    s = np.linalg.svd(sol.jac, compute_uv=False)
    k = len(sol.x)
    rank = int(np.sum(s > s[0]*1e-7)) if len(s) and s[0] > 0 else 0
    boundary = (np.abs(sol.x-problem.lower) <= 1e-3) | (np.abs(sol.x-problem.upper) <= 1e-3)
    notes = list(problem.metadata.get('warnings', []))
    if not sol.success: notes.append('Optimizer did not converge')
    if np.any(boundary): notes.append('Parameters at bounds: regular BIC and local intervals unreliable')
    if rank < k: notes.append('Numerically rank-deficient Jacobian; individual parameters not identified')
    if problem.metadata.get('free_widths'): notes.append('Width identifiability requires closed profile likelihoods')
    covariance, correlation = None, None
    if rank == k:
        # Known observational variances: do not rescale by reduced chi-square.
        _, singular, vt = np.linalg.svd(sol.jac, full_matrices=False)
        covariance = (vt.T/singular**2)@vt
        errors = np.sqrt(np.diag(covariance))
        correlation = covariance/np.outer(errors,errors)
        if k > 1 and np.max(np.abs(correlation-np.eye(k))) > .95:
            notes.append('Strong local parameter correlations (>0.95)')
    return dict(parameters=sol.x, names=problem.names, prediction=problem.predict(sol.x),
                residuals=sol.fun, **information_criteria(sol.fun@sol.fun,len(sol.fun),k),
                success=bool(sol.success), message=sol.message, nfev=int(sol.nfev),
                optimality=float(sol.optimality), lower=problem.lower, upper=problem.upper,
                boundary=boundary, singular_values=s, rank=rank,
                covariance=covariance, correlation=correlation, warnings=notes,
                identifiability='not established; inspect profiles',
                metadata=dict(problem.metadata), optimizer=dict(method='trf', x_scale='jac',
                tolerance=tolerance,max_nfev=max_nfev,starts=[np.asarray(x).tolist() for x in starts]))


def m1_start(model, width, fixed_b, width_bounds=(3.,100.)):
    """M1 optimum on the same data, widths and LSF: (q, z, width parameters) with z = lstsq(X, logit C_i) for `model`'s law."""
    from scipy.special import logit
    from .covering import covering_law
    from .models import BFModel
    m1 = BFModel(model.data, covering_law('M1', model.data), model.lsf)
    p = bf_problem(m1, width, fixed_b=fixed_b, width_bounds=width_bounds, m1_seed=False)
    x = fit(p)['parameters']; nl = len(model.data.levels)
    c = np.clip(m1.covering(x[nl:2*nl]), 1e-6, 1-1e-6)
    z = np.clip(np.linalg.lstsq(model.covering.matrix, logit(c), rcond=None)[0], -12+1e-6, 12-1e-6)
    return x[:nl], z, x[2*nl:]


def bf_problem(model, width='common_fixed', *, fixed_b=None, width_bounds=(3.,100.), m1_seed=True):
    """Starts: (q, b) = (13.5, 18.5), (12.5, 8), (14.5, 45) with the law's initial covering, plus the M1 solution (m1_seed)."""
    d, c = model.data, model.covering
    nl, nc = len(d.levels), c.n_parameters
    if width not in ('common_fixed','species_fixed','common_free','individual_free'):
        raise ValueError(width)
    if fixed_b is None: raise ValueError("Supply fixed widths explicitly, including the baseline for free fits")
    b0 = np.broadcast_to(np.asarray(fixed_b, float), (nl,)).copy()
    if not np.isfinite(b0).all() or np.any(b0 <= 0): raise ValueError("Positive widths required")
    if width == 'common_fixed' and not np.all(b0 == b0[0]): raise ValueError("Common width must be common")
    if width_bounds[0] <= 0 or width_bounds[1] <= width_bounds[0]: raise ValueError("Invalid width bounds")
    nb = nl if width == 'individual_free' else int(width == 'common_free')
    def unpack(p):
        b = 10**p[-nl:] if nb == nl and nb > 1 else (np.full(nl, 10**p[-1]) if nb else b0)
        return p[:nl],p[nl:nl+nc],b
    def seed(q,b): return np.r_[np.full(nl,q),c.initial,np.log10(np.full(nb,b))]
    low = np.r_[np.full(nl,9.),np.full(nc,-12.),np.full(nb,np.log10(width_bounds[0]))]
    high = np.r_[np.full(nl,19.),np.full(nc,12.),np.full(nb,np.log10(width_bounds[1]))]
    names = tuple('q:'+s for s in d.levels)+c.names+(
        tuple('log_b:'+s for s in d.levels) if width == 'individual_free' else (('log_b',) if nb else ()))
    notes = ['Gaussian BF errors are conditional on the supplied measurement errors']
    if d.covariance is None: notes.append('Diagonal likelihood omits shared continuum and measurement correlations')
    if width == 'species_fixed': notes.append('Conditional on supplied widths; check independence from these observations')
    if model.lsf is not None: notes.append('Effective line-centre BF differs from the observed pixel-minimum estimator')
    meta = dict(mode='BF',dataset_id=d.fingerprint,width=width,free_widths=nb > 0,
                levels=list(d.levels),fixed_b_kms=b0.tolist(),covering=c.name,
                design_matrix=c.matrix.tolist(),covering_names=list(c.names),lambda_0=c.lambda_0,
                lsf=None if model.lsf is None else dict(resolving_power=model.lsf.resolving_power,quadrature=model.lsf.quadrature),
                opacity_coefficient=1.497e-15,provenance=d.provenance,warnings=notes,
                x_scales=('jac', 1.0))  # unit scaling as well: avoids Jacobian-scaling overshoot from q = 12.5
    starts = (seed(12.5,8.),seed(14.5,45.))
    if m1_seed and c.name != 'M1': starts += (np.concatenate(m1_start(model,width,b0,width_bounds)),)
    problem = FitProblem(lambda p: model.predict(*unpack(p)),d.intensity,d.uncertainty,names,
                         seed(13.5,18.5),low,high,meta,starts,d.covariance)
    problem.unpack = unpack
    return problem
