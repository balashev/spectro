"""Fe II population grids from spectro.pyratio: P0 electron collisions, P1 + CMB at the absorber redshift, P2 + AGN field.
grid() returns a PopulationGrid for atomic.fit_population_grid, cached in the folder `cache`."""
from pathlib import Path
import hashlib, json, os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # the spectro package (pyratio.py and its atomic data)
STAGES = ('P0', 'P1', 'P2')

# Default grid axes and solver settings.
T_AXIS = np.geomspace(10**3.75, 10**4.65, 21)
NE_AXIS = np.geomspace(10, 1e7, 25)
LOG_RAD = np.array([-6, -3, -2, -1, 0, 0.5, 1, 1.5, 2], float)
RETAINED = 13
T_CMB0 = 2.72548   # K, Fixsen 2009, ApJ 707, 916


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def levels():
    """Level metadata (id, energy_cm1, g) of every retained Fe II level of the package level table (experiment.ion_levels)."""
    from .experiment import ion_levels
    return [dict(id=k, energy_cm1=e, g=g) for k, (e, g) in ion_levels('FeII').items()]


def default_axes(stage):
    """Default grid axes for `stage`."""
    axes = {'temperature': T_AXIS, 'electron_density': NE_AXIS}
    return axes | ({'log_radiation': LOG_RAD} if stage == 'P2' else {})


def override(a_j5):
    return {} if a_j5 is None else {'A_j5_to_ground_override_s-1': float(a_j5)}


def set_pyratio_a(sp, upper, lower, value):
    """A(upper -> lower) in the solver's in-memory arrays; Aij is refreshed and B recomputed from A."""
    if sp.A[upper, lower] <= 0: raise ValueError('No radiative transition to override')
    sp.A[upper, lower] = value; sp.setAij(); sp.setBij()


def pyratio_engine(levs, stage, redshift, agn=None, a_j5=None):
    """(engine, FeII species, solver indices of levs, level metadata) as used for every grid node.
    redshift: absorber redshift; agn: Pyratio AGN normalization {'filter', 'mag'} (P2 only)."""
    from .. import pyratio as pr
    if Path(pr.__file__).resolve() != ROOT / 'pyratio.py':
        raise RuntimeError(f'Wrong pyratio imported: {pr.__file__}')
    assert pr.FEII_SPLUPS_BT_DESCALED is True   # Fe II descaling of Burgess & Tully 1992, A&A 254, 436
    pars = ['T', 'e'] + (['CMB'] if stage != 'P0' else []) + (['rad'] if stage == 'P2' else [])
    if stage == 'P2' and agn is None: raise ValueError('P2 needs the AGN normalization (agn)')
    kw = dict(z=redshift, pars=pars, EBL=False, CMB=False, pumping='full', radiation='full',
              sed_type='AGN' if stage == 'P2' else 'Draine')
    if stage == 'P2': kw['agn'] = agn
    engine = pr.pyratio(**kw)
    if stage == 'P2': engine.pars['UV'] = None  # explicit UV flag: enables UV pumping in balance()
    engine.add_spec('FeII', num=RETAINED); sp = engine.species['FeII']
    ids = []; meta = []
    for lev in levs:
        m = np.flatnonzero((np.abs(np.asarray(sp.E[:RETAINED]) - lev['energy_cm1']) < 1.) &
                           (np.abs(np.asarray(sp.g[:RETAINED]) - lev['g']) < 1e-6))
        if len(m) != 1: raise ValueError(f'Ambiguous Pyratio identity: {lev}')
        i = int(m[0]); ids.append(i); meta.append(dict(lev, solver_index=i, energy_cm1=float(sp.E[i]), g=float(sp.g[i])))
    if a_j5 is not None: set_pyratio_a(sp, ids[5], ids[0], a_j5)
    return engine, sp, ids, meta


def solve(levs, stage, *, redshift, agn=None, a_j5=None, axes=None, progress=None):
    """(axes, populations, metadata): native Pyratio balance() at every node, 13 retained levels; a_j5 overrides sp.A in memory.
    progress(done, total) is called after every node; an exception raised in it aborts the run."""
    if stage not in STAGES: raise ValueError(f'Stage must be one of {STAGES}')
    engine, sp, ids, meta = pyratio_engine(levs, stage, redshift, agn, a_j5)
    root = ROOT
    if stage != 'P0': engine.pars['CMB'].value = T_CMB0 * (1 + redshift)
    axes = {k: np.asarray(v, float) for k, v in (default_axes(stage) if axes is None else axes).items()}
    units = {'temperature': 'K', 'electron_density': 'cm-3'}
    if stage == 'P2': units['log_radiation'] = 'dex (native Pyratio rad)'
    if list(axes) != list(units): raise ValueError(f'Axes must be {list(units)}')
    cache = []
    if stage == 'P2':
        # Full radiation matrices do not depend on T, ne: compute once per field, reuse via native 'simple' hooks.
        for s in axes['log_radiation']:
            engine.pars['rad'].value = float(s)
            cache.append((engine.balance('FeII', debug='UV') / 10**s, engine.balance('FeII', debug='IR') / 10**s))
        engine.pumping = engine.radiation = 'simple'
    shape = tuple(len(a) for a in axes.values()); pop = np.empty(shape + (len(ids),)); res = []; cache_err = []
    total = int(np.prod(shape))
    for done, idx in enumerate(np.ndindex(shape), 1):
        engine.pars['T'].value = np.log10(axes['temperature'][idx[0]])
        engine.pars['e'].value = np.log10(axes['electron_density'][idx[1]])
        if stage == 'P2':
            engine.pars['rad'].value = float(axes['log_radiation'][idx[2]]); sp.pump_rate, sp.rad_rate = cache[idx[2]]
        full = np.asarray(engine.balance('FeII'), float); full /= full.sum()
        W = np.asarray(engine.balance('FeII', debug='total'), float)
        if not np.isfinite(W).all() or np.any(W < 0) or not np.isfinite(full).all() or np.any(full < 0):
            raise ValueError('Invalid native rates/populations')
        K = W.T - np.diag(W.sum(axis=1)); r = float(np.max(np.abs(K @ full)) / max(np.max(W), 1e-300)); res.append(r)
        if r > 1e-10: raise ValueError('Native populations do not satisfy equilibrium')
        sel = full[ids]; pop[idx] = sel / sel.sum()
        if cache and idx in ((0,) * 3, tuple(n // 2 for n in shape), tuple(n - 1 for n in shape)):
            engine.pumping = engine.radiation = 'full'
            direct = np.asarray(engine.balance('FeII'), float); direct /= direct.sum()
            cache_err.append(float(np.max(np.abs(full - direct))))
            np.testing.assert_allclose(full, direct, atol=1e-12, rtol=1e-10)
            engine.pumping = engine.radiation = 'simple'
        if progress is not None: progress(done, total)
    files = {str(p.relative_to(root)): sha(p) for folder in ('data/pyratio/FeII', 'data/AGN', 'data/SDSS', 'sviewer/data/SDSS')
             for p in sorted((root / folder).rglob('*')) if p.is_file()}
    for name in ('data/pyratio/QSO1_template_norm.sed', 'data/pyratio/Richards2006.dat'):
        if (root / name).is_file(): files[name] = sha(root / name)
    for name in ('a_unc.py', 'profiles.py', 'stats.py', 'sviewer/utils.py'):
        files[name] = sha(root / name)
    for p in sorted((root / 'data').glob('*AGN*')):
        if p.is_file(): files[str(p.relative_to(root))] = sha(p)
    metadata = dict(species='FeII', levels=meta, axis_units=units,
        radiation={'stage': stage, 'redshift': redshift, 'agn': agn if stage == 'P2' else None, 'constructor_CMB': False,
                   'explicit_CMB_K': T_CMB0 * (1 + redshift) if stage != 'P0' else None, 'EBL': False},
        solver={'name': 'Pyratio', 'source': str(root / 'pyratio.py'), 'source_sha256': sha(root / 'pyratio.py'),
                'FEII_SPLUPS_BT_DESCALED': True},
        atomic_data=files,
        settings={'retained_levels': RETAINED, 'pumping': 'full', 'radiation': 'full', 'electron_only': True}
                 | ({'cache_native_radiation_matrices': True} if stage == 'P2' else {}) | override(a_j5),
        normalization='Selected levels renormalized after native retained-level solve',
        validation={'fresh': True, 'max_scaled_equilibrium_residual': max(res), 'minimum_population': float(pop.min()),
                    **({'cached_vs_full_max_absolute': max(cache_err)} if cache_err else {}),
                    'independent_physical_validation': False,
                    'warnings': ['Single-intermediate cascade approximation; omitted high-level collision/cascade paths',
                                 'Fe II collision mapping verified only through first 16 levels; validity range logT=3.7-4.7']})
    return axes, pop, metadata


def cache_key(stage, axes, levs, redshift, agn=None, a_j5=None):
    """sha256 of the settings: stage, axes, levels, a_j5, fixed constants, pyratio.py, this module (solver settings), the Fe II
    atomic data and the AGN/SDSS files used by P2."""
    files = [sha(p) for folder in ('data/pyratio/FeII', 'data/SDSS', 'sviewer/data/SDSS') if (ROOT/folder).is_dir()
             for p in sorted((ROOT/folder).rglob('*')) if p.is_file()]
    files += [sha(ROOT/n) for n in ('data/pyratio/QSO1_template_norm.sed', 'data/pyratio/Richards2006.dat') if (ROOT/n).is_file()]
    key = dict(stage=stage, axes={k: np.asarray(v, float).tolist() for k, v in axes.items()}, levels=levs,
               a_j5=a_j5, redshift=redshift, agn=agn, retained=RETAINED, t_cmb0=T_CMB0,
               pyratio=sha(ROOT/'pyratio.py'), solver=sha(__file__), feii_data=files)
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def grid(stage, *, redshift, levs, cache, agn=None, a_j5=None, axes=None, progress=None):
    """PopulationGrid for `stage` (default axes unless given), read from or written to the folder `cache` (None: no cache).
    levs: level metadata [dict(id, energy_cm1, g)], e.g. levels(); redshift and agn as in pyratio_engine."""
    from .atomic import PopulationGrid
    axes = default_axes(stage) if axes is None else axes
    agn = agn if stage == 'P2' else None
    path = None if cache is None else Path(cache)/f'pyratio_{stage}_{cache_key(stage, axes, levs, redshift, agn, a_j5)[:20]}.npz'
    if path is not None and path.is_file():
        try: return PopulationGrid.load(path)
        except Exception: path.unlink(missing_ok=True)   # unreadable or partial file: recompute
    g = PopulationGrid(*solve(levs, stage, redshift=redshift, agn=agn, a_j5=a_j5, axes=axes, progress=progress))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
        g.save(tmp)
        os.replace(tmp, path)
    return g
