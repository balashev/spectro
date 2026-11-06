"""Portable population grids. Solvers run separately; their physics remains external."""
from dataclasses import dataclass
import hashlib,json
import numpy as np
from astropy import units as u
from .fitting import FitProblem,fit,information_criteria,m1_start


def identity(metadata):
    return hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass
class PopulationGrid:
    axes: dict[str,np.ndarray]
    populations: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.axes={k:np.asarray(v,float) for k,v in self.axes.items()}
        self.populations=np.asarray(self.populations,float)
        required={'species','levels','axis_units','radiation','solver','atomic_data','settings','normalization','validation'}
        if not required.issubset(self.metadata): raise ValueError('Incomplete atomic provenance')
        ids=[str(l['id']) for l in self.metadata['levels']]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate level identifiers')
        if set(self.axes)!=set(self.metadata['axis_units']): raise ValueError('Grid units missing')
        for unit in self.metadata['axis_units'].values():
            # 'dex (native Pyratio rad)' is the unit label of the Pyratio radiation axis
            u.Unit('dex' if unit=='dex (native Pyratio rad)' else unit)
        for k,a in self.axes.items():
            if a.ndim!=1 or len(a)<1 or not np.isfinite(a).all() or np.any(np.diff(a)<=0): raise ValueError(f'Invalid {k} axis')
        expected=tuple(len(a) for a in self.axes.values())+(len(ids),)
        p=self.populations
        if p.shape!=expected or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(axis=-1),1,rtol=0,atol=1e-8):
            raise ValueError('Invalid or unnormalized populations')
        for lev in self.metadata['levels']:
            if not {'energy_cm1','g','id'}.issubset(lev) or lev['g']<=0: raise ValueError('Level metadata missing')

    @property
    def model_id(self):
        return identity(dict(self.metadata,axes={k:a.tolist() for k,a in self.axes.items()}))

    def save(self,path):
        meta=dict(self.metadata,axis_order=list(self.axes))
        payload=dict(populations=self.populations,metadata=np.array(json.dumps(meta,sort_keys=True)),
                     model_id=np.array(self.model_id),population_sha256=np.array(hashlib.sha256(self.populations.tobytes()).hexdigest()))
        payload.update({'axis_'+k:v for k,v in self.axes.items()})
        np.savez_compressed(path,**payload)

    @classmethod
    def load(cls,path,*,expected_model_id=None):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));order=meta.pop('axis_order')
            grid=cls({k:z['axis_'+k].copy() for k in order},z['populations'].copy(),meta)
            if str(z['model_id'])!=grid.model_id or (expected_model_id is not None and grid.model_id!=expected_model_id):
                raise ValueError('Incompatible atomic model identity')
            if str(z['population_sha256'])!=hashlib.sha256(grid.populations.tobytes()).hexdigest(): raise ValueError('Population payload corrupted')
        return grid


def population_problem(model,populations,level_ids,*,fixed_b,width='fixed',width_bounds=(3.,100.),starts=()):
    """q_j = q0 + log10(p_j/p0) + log10(b0/b_j); no atomic indices inferred. `starts`: extra seeds besides q0 = 13.4."""
    ids=list(map(str,level_ids));d=model.data
    if len(set(ids))!=len(ids): raise ValueError('Duplicate atomic IDs')
    p=np.asarray(populations,float)[[ids.index(str(k)) for k in d.levels]]
    if not np.isfinite(p).all() or np.any(p<=0): raise ValueError('Positive mapped populations required')
    nl,nc=len(p),model.covering.n_parameters
    b0=np.broadcast_to(np.asarray(fixed_b,float),(nl,)).copy()
    if np.any(b0<=0):raise ValueError('Positive widths required')
    if width not in ('fixed','common_free','individual_free'):raise ValueError(width)
    nb=nl if width=='individual_free' else int(width=='common_free')
    def unpack(x):
        b=(10**x[-nl:] if nb==nl and nb>1 else np.full(nl,10**x[-1])) if nb else b0
        q=x[0]+np.log10(p/p[0])+np.log10(b[0]/b)
        return q,x[1:1+nc],b
    seed=np.r_[13.4,model.covering.initial,np.log10(np.full(nb,18.5))]
    lo=np.r_[9.,np.full(nc,-12.),np.full(nb,np.log10(width_bounds[0]))]
    hi=np.r_[19.,np.full(nc,12.),np.full(nb,np.log10(width_bounds[1]))]
    names=('q0',)+model.covering.names+tuple(f'log_b:{i}' for i in range(nb))
    meta=dict(mode='BF',dataset_id=d.fingerprint,covering=model.covering.name,width=width,
              parameterization='population_ratios',design_matrix=model.covering.matrix.tolist(),
              covering_names=list(model.covering.names),lambda_0=model.covering.lambda_0,levels=list(d.levels),
              lsf=None if model.lsf is None else dict(resolving_power=model.lsf.resolving_power,quadrature=model.lsf.quadrature),
              opacity_coefficient=1.497e-15,provenance=d.provenance,
              population_ratios=(p/p[0]).tolist(),level_ids=list(d.levels),fixed_b_kms=b0.tolist(),
              free_widths=nb>0,warnings=['Conditional on supplied populations; gas-grid searches must count searched coordinates'])
    problem=FitProblem(lambda x:model.predict(*unpack(x)),d.intensity,d.uncertainty,names,seed,lo,hi,meta,tuple(starts),d.covariance)
    problem.unpack=unpack
    return problem


def fit_population_grid(model,grid,*,fixed_b,width='fixed',m1_seed=True):
    """Reoptimize nuisance parameters at every node; count each varied grid axis once.
    Starts: q0 = 13.4 with the law's initial covering, plus (m1_seed) q0 = q_j0 and covering of the M1 solution."""
    shape=grid.populations.shape[:-1];chi=np.empty(shape);converged=np.empty(shape,bool)
    starts=()
    if m1_seed:
        q,z,b=m1_start(model,'species_fixed' if width=='fixed' else width,fixed_b)
        starts=(np.r_[q[0],z,b],)
    parameters=[];best=None;best_index=None
    ids=[l['id'] for l in grid.metadata['levels']]
    for index in np.ndindex(shape):
        problem=population_problem(model,grid.populations[index],ids,fixed_b=fixed_b,width=width,starts=starts)
        r=fit(problem,max_nfev=800,tolerance=1e-9)
        chi[index]=r['chi2'];converged[index]=r['success'];parameters.append(r['parameters'])
        if best is None or r['chi2']<best['chi2']:best=r;best_index=index
    gas_k=sum(n>1 for n in shape)
    boundary=any(n>1 and i in (0,n-1) for i,n in zip(best_index,shape))
    notes=['Nominal information criteria for a finite grid search; grid resolution is a numerical uncertainty']
    if boundary:notes.append('Best gas coordinate is at a grid edge; confidence region may be open')
    if not converged.all(): notes.append('Some node fits did not converge')
    return dict(chi2_grid=chi,parameters=np.asarray(parameters).reshape(shape+(-1,)),success_grid=converged,
                best_index=best_index,best_fit=best,grid_edge=boundary,
                **information_criteria(best['chi2'],best['n'],best['k']+gas_k),
                grid_model_id=grid.model_id,warnings=notes)
