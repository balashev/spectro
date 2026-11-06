"""BF estimators on a normalized spectrum: unmasked pixels around the lowest one in a contiguous window (velocity or pixels).
Errors: sigma_mc (Monte Carlo over the window pixels, minimum reselected per draw) and sigma_catalogue (median pixel error)."""
import numpy as np

ESTIMATORS = ('median3', 'min', 'median', 'mean')
SIGMA_METHODS = ('mc', 'catalogue')


def _pick(flux, n):
    """Values and run positions of the n pixels centred on the lowest pixel (last axis); NaN / -1 outside the run or masked."""
    if n < 1 or n % 2 == 0: raise ValueError('N must be a positive odd number')
    flux = np.asarray(flux, float)
    m = flux.shape[-1]
    i = np.argmin(np.where(np.isnan(flux), np.inf, flux), axis=-1)
    idx = i[..., None]+np.arange(n)-n//2
    inside = (idx >= 0) & (idx < m)
    values = np.where(inside, np.take_along_axis(flux, np.clip(idx, 0, m-1), axis=-1), np.nan)
    return values, np.where(np.isnan(values), -1, idx)


def _median(values):
    """Median over the last axis ignoring NaN (mean of the two middle values for an even count)."""
    s = np.sort(values, axis=-1)
    k = np.sum(~np.isnan(values), axis=-1)[..., None]
    return ((np.take_along_axis(s, (k-1)//2, -1)+np.take_along_axis(s, k//2, -1))/2)[..., 0]


def bf_min(flux):
    """Lowest pixel: (value, run positions)."""
    v, ix = _pick(flux, 1)
    return v[..., 0], ix


def bf_median(flux, n=3):
    """Median of the n pixels centred on the lowest pixel: (value, run positions)."""
    v, ix = _pick(flux, n)
    return _median(v), ix


def bf_median3(flux):
    """Catalogue estimator: bf_median with n = 3."""
    return bf_median(flux, 3)


def bf_mean(flux, n=3):
    """Mean of the n pixels centred on the lowest pixel: (value, run positions)."""
    v, ix = _pick(flux, n)
    return np.nanmean(v, axis=-1), ix


def estimate(flux, estimator='median3', n=3):
    """(value, run positions) of `estimator` on flux (..., pixels); NaN marks masked pixels."""
    if estimator == 'median3': return bf_median3(flux)
    if estimator == 'min': return bf_min(flux)
    if estimator == 'median': return bf_median(flux, n)
    if estimator == 'mean': return bf_mean(flux, n)
    raise ValueError(f'Estimator must be one of {ESTIMATORS}')


def window_run(spectrum, transition, redshift, window=None, pixels=None):
    """Contiguous pixel indexes of the BF window: `window` (vmin, vmax) km/s around the line at redshift, or `pixels`."""
    if (window is None) == (pixels is None): raise ValueError('Give either a velocity window or a pixel run')
    if pixels is None:
        v = spectrum.velocity(transition, redshift)
        run = np.flatnonzero((v >= window[0]) & (v <= window[1]))
    else:
        run = np.sort(np.asarray(pixels, int))
        if run.size and (run[0] < 0 or run[-1] >= len(spectrum.flux)): raise ValueError('Pixel run outside the spectrum')
    if run.size == 0 or np.any(np.diff(run) != 1): raise ValueError('BF window must be a nonempty contiguous pixel run')
    return run


def measure(spectrum, transition, redshift, window=None, *, pixels=None, estimator='median3', n=3, sigma='mc',
            n_draws=2000, seed=0, rng=None):
    """BF of one line with sigma_mc (std of n_draws redraws, mc_bias = mean draw - BF) and sigma_catalogue; `sigma` picks 'sigma'."""
    if not spectrum.normalized: raise ValueError('Normalize spectrum explicitly before measuring BF')
    if sigma not in SIGMA_METHODS: raise ValueError(f'Sigma method must be one of {SIGMA_METHODS}')
    n = {'median3': 3, 'min': 1}.get(estimator, int(n))
    run = window_run(spectrum, transition, redshift, window, pixels)
    flux = np.where(spectrum.mask[run], np.nan, spectrum.flux[run])
    if np.isnan(flux).all(): raise ValueError('No unmasked pixel in the BF window')
    value, ix = estimate(flux, estimator, n)
    used = run[ix[ix >= 0]]
    sigma_catalogue = float(np.median(spectrum.uncertainty[used]))
    sigma_mc = mc_bias = np.nan
    if n_draws:
        noise = (np.random.default_rng(seed) if rng is None else rng).standard_normal((n_draws, run.size))
        draws = estimate(flux+spectrum.uncertainty[run]*noise, estimator, n)[0]
        sigma_mc, mc_bias = float(np.std(draws, ddof=1)), float(np.mean(draws)-value)
    v = spectrum.velocity(transition, redshift)[run[[0, -1]]]
    return dict(BF=float(value), sigma=sigma_mc if sigma == 'mc' else sigma_catalogue, sigma_mc=sigma_mc, mc_bias=mc_bias,
                sigma_catalogue=sigma_catalogue, estimator=estimator, n=n, pixel_indexes=used.tolist(),
                window=tuple(map(float, window if pixels is None else v)), window_pixels=(int(run[0]), int(run[-1])),
                sigma_method=sigma, n_draws=int(n_draws), seed=seed if rng is None else None)
