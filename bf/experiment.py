"""Inputs of a BF line set: CSV rows of a line file, level data (E_j, g_j) from SPECTRO's own atomic data (spectro.pyratio), and
the forward model / fit problem for one covering law and width. The sample itself (lines, redshift, window) is chosen by the caller."""
from functools import lru_cache
from pathlib import Path
import csv
import re
import warnings
import numpy as np
from astropy import units as u
from .data import Transition
from .covering import covering_law
from .models import BFModel
from .profiles import GaussianLSF
from .fitting import bf_problem


def rows(folder,name):
    """Rows (dicts) of the CSV file `name` in `folder`."""
    with (Path(folder)/name).open() as f: return list(csv.DictReader(f))


@lru_cache(maxsize=None)
def ion_levels(name):
    """{level: (E_j cm^-1, g_j)} of ion `name` ('FeII') from the level table spectro.pyratio uses (its retained levels, named as
    SPECTRO names them: FeII, FeIIj1, ...; g = 2J + 1); {} if pyratio has no data for the ion."""
    from .. import pyratio
    engine = pyratio.pyratio(z=0.)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try: engine.add_spec(name)
        except KeyError: return {}
    sp = engine.species[name]
    return {level: (float(sp.E[i]), float(sp.g[i])) for i, level in enumerate(sp.names)}


def level_data(level):
    """dict(energy_cm1, g) of a SPECTRO level name ('FeIIj3': the ion is the name without jN) from ion_levels, or None if the
    package has no data for that level."""
    e = ion_levels(re.sub(r'j\d+$', '', level)).get(level)
    return None if e is None else dict(energy_cm1=e[0], g=e[1])


ROMAN={'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,'X':10}


def ion(level):
    """(ion, stage) of a SPECTRO level name: 'FeIIj3' -> ('Fe II', 2); stage 0 if the name has no roman numeral."""
    m=re.match(r'([A-Z][a-z]?)([IVX]+)(j\d+)?$',level)
    return (f'{m.group(1)} {m.group(2)}',ROMAN.get(m.group(2),0)) if m else (level,0)


def transition(row,energy_cm1=None):
    """Transition of a catalogue row (species = lower level, lambda_rest, f); id 'level:lambda_rest'."""
    name,stage=ion(row['species'])
    return Transition(id=f"{row['species']}:{row['lambda_rest']}",species=name,ion_stage=stage,
                      wavelength=float(row['lambda_rest'])*u.AA,f=float(row['f']),lower=row['species'],
                      energy=None if energy_cm1 is None else energy_cm1/u.cm,
                      source='line catalogue; upstream atomic compilation unspecified',version='unspecified')


def make_problem(model,width,*,data,b,resolving_power,coordinate,quadrature=32,convolved=True,level_b=None,lambda_0='auto'):
    """(forward model, fit problem) for covering law `model` and `width` on `data`; coordinate, lambda_0 as in covering_law.
    b, level_b: widths in km/s (common or per level); resolving_power: R or {transition id: R}; quadrature: None is adaptive."""
    law=covering_law(model,data,coordinate=coordinate,lambda_0=lambda_0)
    if isinstance(resolving_power,dict): resolving_power=[resolving_power[t.id] for t in data.transitions]
    forward=BFModel(data,law,GaussianLSF(resolving_power,quadrature) if convolved else None)
    if width=='species_fixed' and level_b is not None:
        missing=[l for l in data.levels if l not in level_b]
        if missing: raise ValueError(f"No fixed b for level(s) {', '.join(missing)}")
        widths=np.array([level_b[l] for l in data.levels],float)
    elif width=='species_fixed' and np.ndim(b)==0: raise ValueError('Fixed per-level widths need level_b = {level: b}')
    else: widths=np.broadcast_to(np.asarray(b,float),(len(data.levels),)).copy()
    problem=bf_problem(forward,width,fixed_b=widths)
    problem.metadata['level_coordinate']=coordinate if isinstance(coordinate,str) else 'user array'
    return forward,problem
