from copy import copy, deepcopy
from collections import OrderedDict
import gc
import numpy as np
from .utils import Timer
from ..a_unc import a
from ..atomic import abundance, doppler
from ..pyratio import pyratio
from .coverage_models import (
    coverage_mode_index,
    coverage_mode_name,
    coverage_region_for_line,
    format_coverage_assignment,
    parse_coverage_assignment,
)
from .source_models import source_mode_index, source_mode_name

class par:
    def __init__(self, parent, name, val, min, max, step, addinfo='', vary=True, fit=True, show=True, left=None, right=None):
        self.parent = parent
        self.name = name
        if 'cont' in self.name:
            self.dec = 4
        elif 'me' in self.name:
            self.dec = 3
        elif 'res' in self.name:
            self.dec = 1
        elif self.name in ['covmode', 'srcmode']:
            self.dec = 0
        elif self.name.startswith('cov') or self.name.startswith('src'):
            # Resolve small column ratios in the model editor.
            self.dec = 9
        elif 'cf' in self.name:
            self.dec = 3
        elif 'zero' in self.name:
            self.dec = 3
        elif 'displ' in self.name:
            self.dec = 1
        elif 'disps' in self.name:
            self.dec = 8
        elif 'dispz' in self.name:
            self.dec = 4
        elif 'sts' in self.name:
            self.dec = 3
        elif 'stN' in self.name:
            self.dec = 2
        else:
            d = {'z': 7, 'b': 3, 'N': 3, 'I': 3, 'turb': 3, 'kin': 2, 'mu': 8, 'iso': 3, 'hcont': 3,
                 'Ntot': 3, 'logn': 3, 'logT': 3, 'logf': 3, 'rad': 3, 'CMB': 3}
            self.dec = d[self.name]

        if self.name in ['N', 'Ntot', 'logn', 'logT', 'rad', 'iso', 'me', 'mu', 'sts', 'stNl', 'stNu']:
            self.form = 'l'
        else:
            self.form = 'd'

        if self.name in ['b', 'N', 'I']:
            self.sys = self.parent.parent
        elif self.name in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            self.sys = self.parent
        else:
            self.sys = None

        self.val = val
        self.min = min
        self.max = max
        self.step = step
        self.vary = vary
        self.addinfo = addinfo
        self.fit = fit
        self.fit_w = self.fit
        self.show = show
        self.left = left
        self.right = right
        self.unc = a()

    def set(self, val, attr='val', check=True):
        if attr == 'unc':
            if isinstance(val, (int, float)):
                setattr(self, attr, a(self.val, val, self.form))
            elif isinstance(val, str):
                setattr(self, attr, a(val, self.form))
            else:
                setattr(self, attr, val)
        else:
            setattr(self, attr, val)
            if attr == 'vary':
                self.fit = self.vary
        if attr == 'val' and check:
            return self.check_range()

    def check_range(self):

        if self.val < self.min:
            self.val = self.min
            return False

        if self.val > self.max:
            self.val = self.max
            return False

        return True

    def check(self):
        if self.addinfo == '':
            self.vary = True
        else:
            self.vary = False

    def duplicate(self, other):
        attrs = ['val', 'min', 'max', 'step', 'vary', 'addinfo', 'fit', 'fit_w']
        for attr in attrs:
            setattr(self, attr, getattr(other, attr))

    def copy(self):
        return par(self.parent, self.name, self.val, self.min, self.max, self.step, addinfo=self.addinfo, vary=self.vary, fit=self.fit, show=self.show)

    def latexname(self):
        pass

    def ref(self, val=None, attr='val'):
        # special function for lmfit to translate redshift to velocity space
        if self.name.startswith('z'):
            c = 299792.458
            if 1:
                if val is None:
                    self.saved = self.val
                    return c * (self.val - self.saved), c * (self.min - self.saved), c * (self.max - self.saved)
                else:
                    if attr in ['val', 'min', 'max']:
                        return self.saved + val / c
                    elif attr in ['step', 'unc']:
                        return val / c
            else:
                if val is None:
                    self.saved = self.val
                    return self.val / self.saved, self.min / self.saved, self.max / self.saved
                else:
                    if attr in ['val', 'min', 'max']:
                        return val * self.saved
                    elif attr in ['step', 'unc']:
                        return val * self.saved
        else:
            if val is None:
                return self.val, self.min, self.max
            else:
                return val

    def __repr__(self):
        s = self.name
        if self.name in ['z', 'b', 'N', 'I', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            s += '_' + str(self.sys.ind)
        if self.name in ['b', 'N', 'I']:
            s += '_' + self.parent.name
        return s

    def __str__(self):
        s = self.name
        if self.name in ['z', 'b', 'N', 'I', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            s += '_' + str(self.sys.ind)
        if self.name in ['b', 'N', 'I']:
            s += '_' + self.parent.name
        return s

    def str(self, attr=None):
        high_precision = self.name.startswith(('cf_', 'cov', 'src'))
        if attr is None:
            if high_precision:
                if self.name.startswith('cf_'):
                    return '{} {:.17g} {:.17g} {:.17g} {:.17g} {:1d} {}'.format(
                        self, float(self.val), float(self.left), float(self.right),
                        float(self.step), int(self.vary), self.addinfo,
                    )
                return '{} {:.17g} {:.17g} {:.17g} {:.17g} {:1d} {}'.format(
                    self, float(self.val), float(self.min), float(self.max),
                    float(self.step), int(self.vary), self.addinfo,
                )
            if 'cf' not in self.name:
                return '{1:} {2:.{0}f} {3:.{0}f} {4:.{0}f} {5:.{0}f} {6:1d} {7:s}'.format(self.dec, self, self.val, self.min, self.max, self.step, self.vary, self.addinfo)
            else:
                return '{1:} {2:.{0}f} {3:.{0}f} {4:.{0}f} {5:.{0}f} {6:1d} {7:s}'.format(self.dec, self, self.val, self.left, self.right, self.step, self.vary, self.addinfo)
        if attr == 'lmfit':
            return '{1:} {2:.{0}f} ± {3:.{0}f}'.format(self.dec, self, self.val, self.step)
        else:
            if high_precision and attr in {'val', 'min', 'max', 'step', 'left', 'right'}:
                return '{:.17g}'.format(float(getattr(self, attr)))
            return '{0:.{1}f}'.format(getattr(self, attr), self.dec)

    def fitres(self, latex=False, dec=None, showname=True, classview=False, aview=False):
        if self.unc is not None:
            if dec is None and not latex:
                d = np.asarray([self.unc.plus, self.unc.minus])
                if len(np.nonzero(d)[0]) > 0:
                    dec = int(np.round(np.abs(np.log10(np.min(d[np.nonzero(d)])))) + 1)
                else:
                    dec = self.dec
            if latex:
                if self.name in ['z']:
                    dec = int(np.round(np.abs(np.log10(np.min(np.asarray([self.unc.plus, self.unc.minus])[np.nonzero(np.asarray([self.unc.plus, self.unc.minus]))])))) + 1)
                    return '${0:.{3}f}(^{{+{1:d}}}_{{-{2:d}}})$'.format(self.unc.val, int(self.unc.plus*10**dec), int(self.unc.minus*10**dec), dec)
                else:
                    return self.unc.latex(f=dec, base=0)
                    #return '${0:.{3}f}^{{+{1:.{3}f}}}_{{-{2:.{3}f}}}$'.format(self.unc.val, self.unc.plus, self.unc.minus, dec)
            elif classview:
                if self.name in ['z']:
                    return 'co = sy(q, {0:.{2}f}, {1:d})'.format(self.unc.val, int(np.sqrt(self.unc.plus**2 + self.unc.minus**2) * 10 ** dec), dec)
                elif self.name in ['N']:
                    return "co.el('{0}', {1:.{4}f}, {2:.{4}f}, {3:.{4}f})".format(self.parent.name, self.unc.val, self.unc.plus, self.unc.minus, dec)
            elif aview:
                return '({0:.{3}f}, {1:.{3}f}, {2:.{3}f})'.format(self.unc.val, self.unc.plus, self.unc.minus, dec)
            else:
                return '{0} = {1:.{4}f} + {2:.{4}f} - {3:.{4}f}'.format(str(self), self.unc.val, self.unc.plus, self.unc.minus, dec)
        else:
            if dec is None:
                dec = self.dec
            if showname:
                return '{0} = {1:.{2}f}'.format(str(self), self.val, dec)
            else:
                return '{0:.{1}f}'.format(self.val, dec)

class fitSpecies:
    def __init__(self, parent, name=None, type='abs'):
        self.parent = parent
        self.name = name
        #self.mass = self.setmass(name)
        self.type = type
        self.b = par(self, 'b', 4, 0.5, 200, 0.5)
        self.N = par(self, 'N', 14, 10, 22, 0.2)
        self.I = par(self, 'I', 10, 0, 100, 0.2)

    def set_type(self, attr):
        if attr in ['N', 'I']:
            self.type = {'N': 'abs', 'I': 'em'}[attr]

    def duplicate(self, other):
        attrs = ['b', 'N', 'I']
        self.type = other.type
        for attr in attrs:
            getattr(self, attr).duplicate(getattr(other, attr))

class fitSystem:
    def __init__(self, parent, z=0.0):
        self.parent = parent
        self.z = par(self, 'z', z, z-0.001, z+0.001, 1e-7)
        #self.cons_vary = False
        #self.turb = par(self, 'turb', 5, 0.5, 20, 0.05, vary=self.cons_vary, fit=self.cons_vary)
        #self.kin = par(self, 'kin', 5e4, 1e4, 1e5, 1e3, vary=self.cons_vary, fit=self.cons_vary)
        self.sp = OrderedDict()
        self.total = OrderedDict()
        self.pr = None
        self.exclude = []

    def add(self, name):
        if name in 'turb':
            self.turb = par(self, 'turb', 5, 0.5, 20, 0.05)
        if name in 'kin':
            self.kin = par(self, 'kin', 5e3, 1e3, 3e4, 1e3)
        if name in 'Ntot':
            self.Ntot = par(self, 'Ntot', 14, 12, 22, 0.05)
        if name in 'logn':
            self.logn = par(self, 'logn', 2, -2, 5, 0.05)
        if name in 'logT':
            self.logT = par(self, 'logT', 2, 0.5, 5, 0.05)
        if name in 'logf':
            self.logf = par(self, 'logf', 0, -6, 0, 0.05)
        if name in 'rad':
            self.rad = par(self, 'rad', 0, -6, 30, 0.05)
        if name in 'CMB':
            self.CMB = par(self, 'CMB', 2.726 * (1 + self.z.val), 0, 50, 0.05)

    def remove(self, name):
        if name in ['turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            if hasattr(self, name):
                delattr(self, name)

    def addSpecies(self, name, dic='sp'):
        if name not in getattr(self, dic).keys():
            getattr(self, dic)[name] = fitSpecies(self, name)
            #if self.parent.parent is not None:
            #    self.parent.parent.console.exec_command('show ' + name)
            return True
        else:
            return False

    def duplicate(self, other):
        self.z.duplicate(other.z)
        attrs = ['turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']
        for attr in attrs:
            if hasattr(other, attr):
                self.add(attr)
                getattr(self, attr).duplicate(getattr(other, attr))
        self.sp = OrderedDict()
        for k, v in other.sp.items():
            self.sp[k] = fitSpecies(self, name=k, type=v.type)
            self.sp[k].duplicate(v)

    def zshift(self, v):
        self.z.val += float(v) / 299792.458 * (1 + self.z.val)

    def zrange(self, v):
        self.z.min = self.z.val - abs(float(v)) / 299792.458 * (1 + self.z.val)
        self.z.max = self.z.val + abs(float(v)) / 299792.458 * (1 + self.z.val)

    def N(self, sp):
        if sp in self.sp.keys():
            if hasattr(self.parent, 'me') and 'HI' != sp and 'HI' in self.sp.keys():
                return abundance(sp, self.sp['HI'].N.val, self.parent.me.val)
            else:
                return self.sp[sp].N.val

    def pyratio(self, init=False):
        #t = Timer('pyratio '+ str(self.parent.sys.index(self)))
        if init or self.pr is None:
            if self == self.parent.sys[0]:

                self.pr = pyratio(z=self.z.val, pumping='simple', radiation='simple', sed_type=self.rad.addinfo)

                #print('init', self.pr.pumping, self.pr.radiation,  self.pr.sed_type)
                d = {'CO': [-1, 10], 'CI': [-1, 3], 'FeII': [-1, 13], 'H2': [-1, 3]}
                for s in self.sp.keys():
                    if s.startswith('CO'):
                        d['CO'][0] = 0 if s[3:4].strip() == '' else max(d['CO'][0], int(s[3:4]))
                        pars = ['T', 'n', 'f', 'CMB']
                        self.pr = pyratio(z=self.z.val, radiation='full', sed_type='CMB')
                    if s.startswith('CI'):
                        d['CI'][0] = 0 if s[3:4].strip() == '' else max(d['CI'][0], int(s[3:4]))
                        pars = ['T', 'n', 'f', 'rad', 'CMB']
                    if 'FeII' in s:
                        d['FeII'][0] = 0 if s[5:6].strip() == '' else max(d['FeII'][0], int(s[5:6]))
                        pars = ['T', 'e', 'rad']
                    if 'H2' in s:
                        d['H2'][0] = 0 if s[3:4].strip() == '' else max(d['H2'][0], int(s[3:4]))
                        pars = ['T', 'n', 'f', 'rad']
                self.pr.set_pars(pars)
                for k, v in d.items():
                    if v[0] > -1:
                        self.pr.add_spec(k, num=v[1])
            else:
                self.pr = deepcopy(self.parent.sys[0].pr)

        #t.time('init')
        if self.pr is not None:
            self.pr.pars['T'].value = self.logT.val
            if 'n' in self.pr.pars.keys():
                self.pr.pars['n'].value = self.logn.val
            if 'e' in self.pr.pars.keys():
                self.pr.pars['e'].value = self.logn.val
            if 'f' in self.pr.pars.keys():
                self.pr.pars['f'].value = self.logf.val
            if 'rad' in self.pr.pars.keys():
                self.pr.pars['rad'].value = self.rad.val
            if 'CMB' in self.pr.pars.keys():
                self.pr.pars['CMB'].value = self.CMB.val
            for k in self.pr.species.keys():
                col = self.pr.predict(name=k, level=-1, logN=self.Ntot.val)
                for s in self.sp.keys():
                    if k in s and 'Ntot' in self.sp[s].N.addinfo:
                        self.sp[s].N.val = col[self.pr.species[k].names.index(s)]
        #t.time('predict')


    def __str__(self):
        return '{:.6f} '.format(self.z.val) + str(self.sp)

class cont():
    def __init__(self, parent):
        self.parent = parent
        self.left = 0
        self.right = 0
        self.disp = 0
        self.exp = 0
        self.num = 0
        self.pars = []

    def copy(self):
        c = cont(self.parent)
        c.left = self.left
        c.right = self.right
        c.num = self.num
        c.disp = self.disp
        c.pars = self.pars[:]
        return c

    def update(self):
        for i in range(self.num):
            s = '{0:7.1f}'.format(self.left).strip() + '..'
            s += '{0:7.1f}'.format(self.right).strip()
            s += '_exp_{0:d}'.format(self.exp).strip()
            s += '_{0:5.3f}'.format(self.disp).strip()
            self.parent.setValue(self.pars[i].name, s, attr='addinfo')
            #print(self.pars[i].addinfo)

    def fromInfo(self, info):
        s = info.split('_')
        if len(s) > 0:
            self.left = float(s[0].split('..')[0])
            self.right = float(s[0].split('..')[1])
        if len(s) > 2:
            self.exp = int(s[2])
        if len(s) > 3:
            self.disp = float(s[3])

class fitCont(list):
    def __init__(self, parent):
        super(fitCont).__init__()
        self.parent = parent

    def add(self, name):
        s = name.split('_')
        if len(s) > 1:
            if int(s[1]) <= len(self):
                if int(s[1]) == len(self):
                    self.append(cont(self.parent))
                ind = int(s[1])
                if int(s[2]) == len(self[ind].pars):
                    if int(s[2]) == 0:
                        setattr(self.parent, name, par(self, name, 1, 0, 2, 0.01, addinfo='0..0'))
                    else:
                        setattr(self.parent, name, par(self, name, 0, -0.5, 0.5, 0.01, addinfo='0..0'))
                    self[ind].pars.append(getattr(self.parent, name))
                #self[ind].left, self[ind].right = [float(r) for r in getattr(self.parent, name).addinfo.split('..')]
                self[ind].num = len(self[ind].pars)

    def remove(self, name):
        if isinstance(name, str):
            s = name.split('_')
            if len(s) > 1:
                ind = int(s[1])
                if ind < len(self):
                    if int(s[2]) < len(self[ind].pars):
                        self[ind].pars.pop()
                        self[ind].num = len(self[ind].pars)
                        if self[ind].num == 0:
                            self.pop()
                        delattr(self.parent, name)
        elif isinstance(name, int):
            for i in range(self[name].num-1, -1, -1):
                self[name].pars.pop()
                delattr(self.parent, 'cont_' + str(name) + '_' + str(i))
            self[name].num = 0
            self.pop()

        self.parent.cont_num = len(self)
        if len(self) == 0:
            self.parent.remove('hcont')


    def update(self):
        for c in self:
            c.update()

class disp():
    def __init__(self, parent):
        self.parent = parent
        self.left = 0
        self.right = 0
        self.exp = 0
        self.legacy = False
        self.names = ['displ', 'disps', 'dispz']
        self.pars = []

    def copy(self):
        c = disp(self.parent)
        c.left = self.left
        c.right = self.right
        c.exp = self.exp
        c.legacy = self.legacy
        c.pars = self.pars[:]
        return c

    def update(self):
        s = '{0:7.1f}'.format(self.left).strip() + '..'
        s += '{0:7.1f}'.format(self.right).strip()
        s += '_exp_{0:d}'.format(self.exp).strip()
        if self.legacy:
            s = 'exp_' + str(self.exp)
        self.ind = self.parent.disp.index(self)
        print(self.ind)
        for name in self.names:
            print(name + '_' + str(self.ind), s)
            self.parent.setValue(name + '_' + str(self.ind), s, attr='addinfo')
            print(self.parent.getValue(name + '_' + str(self.ind), 'addinfo'))

    def fromInfo(self, info):
        self.legacy = info.startswith('exp_')
        if self.legacy:
            self.left, self.right = -np.inf, np.inf
            self.exp = int(info.split('_')[1])
            return
        s = info.split('_')
        if len(s) > 0:
            self.left = float(s[0].split('..')[0])
            self.right = float(s[0].split('..')[1])
        if len(s) > 2:
            self.exp = int(s[2])

class fitDisp(list):
    def __init__(self, parent):
        super(fitDisp).__init__()
        self.parent = parent

    def add(self, name):
        s = name.split('_')
        if len(s) > 1:
            ind = int(s[1])
            if ind <= len(self):
                if ind == len(self):
                    self.append(disp(self.parent))
                if 'displ' in name:
                    setattr(self.parent, name, par(self.parent, name, 5000, 3000, 9000, 0.1, addinfo='exp_0'))
                    self.parent.setValue(name, 0, attr='vary')
                if 'disps' in name:
                    setattr(self.parent, name, par(self.parent, name, 0.0, -1, 1, 1e-3, addinfo='exp_0'))
                if 'dispz' in name:
                    setattr(self.parent, name, par(self.parent, name, 0.0, -1, 1, 1e-3, addinfo='exp_0'))
                #self[ind].pars.append(getattr(self.parent, name))
                #self[ind].left, self[ind].right = [float(r) for r in getattr(self.parent, name).addinfo.split('..')]

    def remove(self, name):
        if isinstance(name, str):
            s = name.split('_')
            if len(s) > 1:
                ind = int(s[1])
                if ind < len(self):
                    if int(s[2]) < len(self[ind].pars):
                        self[ind].pars.pop()
                        self[ind].num = len(self[ind].pars)
                        if self[ind].num == 0:
                            self.pop()
                        delattr(self.parent, name)
        elif isinstance(name, int):
            for attr in self[name].names:
                delattr(self.parent, attr + '_' + str(name))
            self.pop()

        self.parent.disp_num = len(self)
        self.update()

    def update(self):
        for c in self:
            c.update()

class fitPars:
    def __init__(self, parent):
        self.parent = parent
        self.sys = []
        self.total = fitSystem(self)
        self.total.ind = 'total'
        self.cont_num = 0
        self.cont = fitCont(self)
        self.me_num = 0
        self.res_num = 0
        self.cf_fit = False
        self.cf_num = 0
        self.covmode = par(self, 'covmode', 1, 0, 9, 1, vary=False, fit=False)
        self.srcmode = par(self, 'srcmode', 0, 0, 1, 1, vary=False, fit=False)
        self.srcbeta = par(self, 'srcbeta', 0.75, 0.25, 1.5, 0.01, vary=False)
        self.srcrref = par(self, 'srcrref', 1.0, 1e-4, 100.0, 0.05, vary=False)
        self.srclref = par(self, 'srclref', 2400.0, 100.0, 20000.0, 10.0, vary=False)
        self.disp_num = 0
        self.disp = fitDisp(self)
        self.stack_num = 0
        self.tieds = {}
        self._tie_saved_states = {}

    def coverage_model(self):
        return coverage_mode_name(self.covmode.val)

    def checkJulia(self):
        # Julia supports every covering mode with the achromatic source only.
        if self.source_model() != 'achromatic':
            raise ValueError('Julia supports the achromatic source only (all covering modes); use the regular Python fitter for the SS73 source')
        if any(getattr(self, 'displ_' + str(i)).addinfo.startswith('exp_') for i in range(self.disp_num)):
            raise ValueError('exp_N dispersion uses a different wavelength mapping; use the Python fitter or configure dispersion regions')
        for system in self.sys:
            if hasattr(system, 'Ntot') and getattr(system, 'pr', None) is not None and 'FeII' in system.pr.species:
                raise ValueError('Fe II population constraints require the validated Python radiation/rate engine; Julia can fit independent Fe II columns')

    def checkJuliaFit(self):
        # fitLM varies every fit&vary parameter, fromJulia maps results onto list_fit().
        if [p for p in self.list() if p.fit and p.vary] != self.list_fit():
            raise ValueError('Julia fits every varied parameter; fix the parameters inactive in the current coverage/source model')

    def set_coverage_model(self, value):
        self.covmode.val = coverage_mode_index(value)
        self.covmode.vary = False
        self.covmode.fit = False

    def source_model(self):
        return source_mode_name(self.srcmode.val)

    def set_source_model(self, value):
        self.srcmode.val = source_mode_index(value)
        self.srcmode.vary = False
        self.srcmode.fit = False

    def source_parameter_active(self, name):
        if name == 'srcmode':
            return False
        if name.startswith('src'):
            return (
                self.source_model() == 'ss73'
                and self.cf_fit
                and self.coverage_model() != 'full'
            )
        return True

    @staticmethod
    def coverage_parameter_names(ind):
        return [f'cova_{ind}', f'covd_{ind}', f'covr_{ind}', f'cov1_{ind}', f'cov2_{ind}', f'covov_{ind}']

    def ensure_system_powerlaw_parameters(self):
        """Create one x-PL exponent ``covasys_i`` per kinematic system.
        Unlike ``cova_i`` (one per LFR), it is shared by all regions or transitions of system i."""
        for system_id in range(len(self.sys)):
            name = f'covasys_{system_id}'
            if not hasattr(self, name):
                self.add(name)

    def ensure_coverage_parameters(self, ind, reset_vary=False):
        """Create all coverage parameters for one Cf region.
        Existing vary flags are kept unless ``reset_vary`` is True."""
        for name in self.coverage_parameter_names(ind):
            if not hasattr(self, name):
                self.add(name)
        self.configure_coverage_parameters(ind, reset_vary=reset_vary)

    def configure_coverage_parameters(self, ind=None, reset_vary=False):
        """Synchronize coverage parameters with the selected model.
        Vary flags change only with ``reset_vary``; inactive parameters are excluded by :meth:`coverage_parameter_active`."""
        active_by_mode = {
            'full': set(),
            'ppc': {'cf'},
            'powerlaw': {'cova'},
            'powerlaw_ppc': {'cf', 'cova'},
            'powerlaw_ppc_sysa': {'cf', 'covasys'},
            'powerlaw_sysa_global': {'covasys'},
            'powerlaw_radial': {'cova', 'covd'},
            'powerlaw_radial_ppc': {'cf', 'cova', 'covd'},
            'two_zone': {'cf', 'covr'},
            'cfov': {'cov1', 'cov2', 'covov'},
        }
        active = active_by_mode[self.coverage_model()]
        indices = range(self.cf_num) if ind is None else [ind]
        for i in indices:
            names = {
                'cf': f'cf_{i}',
                'cova': f'cova_{i}',
                'covd': f'covd_{i}',
                'covr': f'covr_{i}',
                'cov1': f'cov1_{i}',
                'cov2': f'cov2_{i}',
                'covov': f'covov_{i}',
            }
            for key, name in names.items():
                if hasattr(self, name) and reset_vary:
                    par_ = getattr(self, name)
                    par_.set((key in active) and key != 'covd', attr='vary')

        self.ensure_system_powerlaw_parameters()
        uses_system_a = 'covasys' in active
        if reset_vary:
            for system_id in range(len(self.sys)):
                par_ = getattr(self, f'covasys_{system_id}')
                par_.set(uses_system_a, attr='vary')

    def remove_coverage_parameters(self, ind):
        for name in self.coverage_parameter_names(ind):
            if hasattr(self, name):
                delattr(self, name)

    def add(self, name, addinfo=''):
        if name in 'mu':
            self.mu = par(self, 'mu', 1e-6, 1e-7, 5e-6, 1e-8)
        if 'me' in name:
            setattr(self, name, par(self, name, 0, -3, 1, 0.01))
        #print(name)
        if name in 'iso':
            #print('iso', name)
            if addinfo == 'D/H':
                self.iso = par(self, 'iso', -4.5, -5.4, -4, 0.01, addinfo=addinfo)
            if addinfo == '13C/12C':
                self.iso = par(self, 'iso', -2.0, -5, 0, 0.1, addinfo=addinfo)
        if 'res' in name:
            setattr(self, name, par(self, name, 45000, 1000, 60000, 1, addinfo='exp_0'))
        if 'cont' in name:
            self.cont.add(name)
        if name in 'hcont':
            self.hcont = par(self, 'hcont', 1, 0, 10, 0.1, vary=False)
        if name == 'covmode':
            self.covmode = par(self, 'covmode', 1, 0, 9, 1, vary=False, fit=False)
        if name == 'srcmode':
            self.srcmode = par(self, 'srcmode', 0, 0, 1, 1, vary=False, fit=False)
        if name == 'srcbeta':
            self.srcbeta = par(self, 'srcbeta', 0.75, 0.25, 1.5, 0.01, vary=False)
        if name == 'srcrref':
            self.srcrref = par(self, 'srcrref', 1.0, 1e-4, 100.0, 0.05, vary=False)
        if name == 'srclref':
            self.srclref = par(self, 'srclref', 2400.0, 100.0, 20000.0, 10.0, vary=False)
        if name.startswith('cova_'):
            setattr(self, name, par(self, name, 0.0, -2.0, 2.0, 0.05, vary=False))
        if name.startswith('covasys_'):
            setattr(self, name, par(self, name, 0.0, -2.0, 2.0, 0.05, vary=False))
        if name.startswith('covd_'):
            setattr(self, name, par(self, name, 0.0, 0.0, 3.0, 0.05, vary=False))
        if name.startswith('covr_'):
            setattr(self, name, par(self, name, 0.1, 0.0, 1.0, 0.01, vary=False))
        if name.startswith('cov1_') or name.startswith('cov2_'):
            ind = int(name.split('_')[1])
            initial = getattr(self, f'cf_{ind}').val if hasattr(self, f'cf_{ind}') else 0.5
            setattr(self, name, par(self, name, initial, 0.0, 1.0, 0.01, vary=False))
        if name.startswith('covov_'):
            setattr(self, name, par(self, name, 1.0, 0.0, 1.0, 0.01, vary=False))
        if name.startswith('cf_'):
            setattr(self, name, par(self, name, 0.1, 0, 1, 0.01, addinfo='all', left=3000, right=9000))
            self.ensure_coverage_parameters(int(name.split('_')[1]))
        if 'zero' in name:
            setattr(self, name, par(self, name, 0.0, 0, 1, 0.01, addinfo='all'))
        if 'disp' in name:
            self.disp.add(name)
        if 'sts' in name:
            setattr(self, name, par(self, name, -1.5, -2, -1, 0.01))
        if 'stNl' in name:
            setattr(self, name, par(self, name, 18, 16, 20, 0.05))
        if 'stNu' in name:
            setattr(self, name, par(self, name, 21, 20, 22, 0.05))

    def remove(self, name):
        if name.startswith('cf_'):
            self.remove_coverage_parameters(int(name.split('_')[1]))
        if name in ['mu', 'iso', 'hcont', 'covmode', 'srcmode', 'srcbeta', 'srcrref', 'srclref'] or name.startswith('cov') or any([x in name for x in ['me', 'res', 'cf', 'zero', 'disp', 'sts', 'stNu', 'stNl']]):
            if hasattr(self, name):
                delattr(self, name)
        if 'cont_' in name:
            self.cont.remove(name)
                #gc.collect()

    def addTieds(self, p1, p2):
        """Tie ``p1`` to ``p2`` while remembering p1's prior fit state."""
        try:
            par1 = self.getPar(p1)
            self.getPar(p2)
            if p1 not in self._tie_saved_states:
                self._tie_saved_states[p1] = (bool(par1.vary), bool(par1.fit))
            self.tieds[p1] = p2
            par1.set(False, attr='vary')
        except Exception:
            pass

    def replaceTieds(self, ties):
        """Replace the complete tie map.
        Untied parameters get back the vary/fit state saved by :meth:`addTieds`."""
        ties = dict(ties)
        old_sources = set(self.tieds)
        new_sources = set(ties)
        for p1 in old_sources - new_sources:
            try:
                par1 = self.getPar(p1)
                vary, fit = self._tie_saved_states.pop(p1, (True, True))
                par1.vary = bool(vary)
                par1.fit = bool(fit)
            except Exception:
                self._tie_saved_states.pop(p1, None)

        self.tieds = {}
        for p1, p2 in ties.items():
            self.addTieds(p1, p2)

    def addSys(self, ind=-1, z=None):
        if z is None:
            if len(self.sys) > 0:
                self.sys.append(fitSystem(self, 0))
                self.sys[-1].duplicate(self.sys[ind])
            else:
                self.sys.append(fitSystem(self, 0))
        else:
            self.sys.append(fitSystem(self, z))
        self.refreshSys()
        if self.cf_fit:
            self.ensure_system_powerlaw_parameters()
            self.configure_coverage_parameters()

    def delSys(self, ind=-1):

        if 1:
            if ind < len(self.sys):
                for i in range(ind, len(self.sys)-1):
                    self.swapSys(i, i+1)

            ind = len(self.sys)-1
        s = self.sys[ind]
        self.sys.remove(s)
        del s

        gc.collect()
        self.refreshSys()
        stale = f'covasys_{len(self.sys)}'
        if hasattr(self, stale):
            delattr(self, stale)
        if self.cf_fit:
            self.ensure_system_powerlaw_parameters()
            self.configure_coverage_parameters()

    def swapSys(self, i1, i2):
        if self.cf_fit:
            for i in range(self.cf_num):
                if hasattr(self, 'cf_' + str(i)):
                    p = getattr(self, 'cf_' + str(i))
                    systems, exposure = parse_coverage_assignment(p.addinfo)
                    if systems is not None:
                        swapped = []
                        for system_id in systems:
                            if system_id == i1:
                                system_id = i2
                            elif system_id == i2:
                                system_id = i1
                            swapped.append(system_id)
                        self.setValue(
                            'cf_' + str(i),
                            format_coverage_assignment(swapped, exposure),
                            'addinfo',
                        )
        if self.coverage_model() == 'cfov' and {i1, i2} == {0, 1}:
            for i in range(self.cf_num):
                n1, n2 = f'cov1_{i}', f'cov2_{i}'
                if hasattr(self, n1) and hasattr(self, n2):
                    p1, p2 = getattr(self, n1), getattr(self, n2)
                    for attr in ['val', 'min', 'max', 'step', 'vary', 'fit']:
                        v1, v2 = getattr(p1, attr), getattr(p2, attr)
                        setattr(p1, attr, v2)
                        setattr(p2, attr, v1)
        n1, n2 = f'covasys_{i1}', f'covasys_{i2}'
        if hasattr(self, n1) and hasattr(self, n2):
            p1, p2 = getattr(self, n1), getattr(self, n2)
            for attr in ['val', 'min', 'max', 'step', 'vary', 'fit']:
                v1, v2 = getattr(p1, attr), getattr(p2, attr)
                setattr(p1, attr, v2)
                setattr(p2, attr, v1)
        self.sys[i1], self.sys[i2] = self.sys[i2], self.sys[i1]
        self.refreshSys()

    def refreshSys(self):
        for i, s, in enumerate(self.sys):
            s.ind = i

    def setValue(self, name, val, attr='val', check=True):
        s = name.split('_')
        if attr in ['val', 'min', 'max', 'step', 'left', 'right']:
            val = float(val)
        elif attr in ['vary', 'fit']:
            val = int(val)

        if s[0] in ['mu', 'iso']:
            if not hasattr(self, s[0]):
                self.add(s[0])
            res = getattr(self, s[0]).set(val, attr, check=check)

        if s[0] in ['dtoh']:
            if not hasattr(self, 'iso'):
                self.add('iso', addinfo='D/H')
            res = getattr(self, 'iso').set(val, attr, check=check)

        if s[0] in ['me', 'res', 'cf', 'zero', 'displ', 'disps', 'dispz', 'hcont', 'sts', 'stNl', 'stNu',
                    'covmode', 'cova', 'covd', 'covasys', 'covr', 'cov1', 'cov2', 'covov',
                    'srcmode', 'srcbeta', 'srcrref', 'srclref']:
            if not hasattr(self, name):
                self.add(name)
            res = getattr(self, name).set(val, attr, check=check)
            if s[0] == 'covmode':
                self.covmode.val = coverage_mode_index(self.covmode.val)
                self.covmode.vary = False
                self.covmode.fit = False
            if s[0] == 'srcmode':
                self.srcmode.val = source_mode_index(self.srcmode.val)
                self.srcmode.vary = False
                self.srcmode.fit = False

        if s[0] in ['cont']:
            if not hasattr(self, name):
                self.cont.add(name)
            res = getattr(self, name).set(val, attr, check=check)

        if s[0] in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            while len(self.sys) <= int(s[1]):
                self.addSys()
            if s[0] in ['turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
                if not hasattr(self.sys[int(s[1])], s[0]):
                    self.sys[int(s[1])].add(s[0])
            res = getattr(self.sys[int(s[1])], s[0]).set(val, attr, check=check)

        if s[0] in ['b', 'N', 'I']:
            while len(self.sys) <= int(s[1]):
                self.addSys()
            self.sys[int(s[1])].addSpecies(s[2])
            res = getattr(self.sys[int(s[1])].sp[s[2]], s[0]).set(val, attr, check=check)
            if attr == 'val':
                self.sys[int(s[1])].sp[s[2]].set_type(s[0])

        if s[0] in ['displ', 'disps', 'dispz']:
            if attr == 'addinfo':
                self.disp[int(s[1])].fromInfo(val)

        return res

    def update(self, what='all', ind='all', redraw=True):

        if what in ['all', 'res']:
            if self.res_num > 0:
                for i in range(self.res_num):
                    if i < len(self.parent.s):
                        self.parent.s[int(getattr(self, 'res_'+str(i)).addinfo[4:])].set_resolution(self.getValue('res_'+str(i)))

        if what in ['all', 'cf']:
            if redraw and self.cf_fit:
                for i in range(self.cf_num):
                    try:
                        self.parent.plot.pcRegions[i].updateFromFit()
                    except:
                        pass

        for i, sys in enumerate(self.sys):
            if ind == 'all' or i == ind:
                for k, s in sys.sp.items():
                    if what in ['all', 'b', 'turb', 'kin']:
                        if s.b.addinfo != '' and s.b.addinfo != 'consist':
                            s.b.val = sys.sp[s.b.addinfo].b.val
                        elif s.b.addinfo == 'consist':
                            s.b.val = doppler(k, sys.turb.val, sys.kin.val)

                if what in ['all', 'Ncons', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
                    if hasattr(sys, 'Ntot'):
                        sys.pyratio()
                        what = 'all'

                for k, s in sys.sp.items():
                    #print(k, s.N.addinfo)
                    if what in ['all', 'me', 'iso']:
                        if 'me' in s.N.addinfo and 'HI' in sys.sp.keys():
                            if any([sp not in k for sp in ['DI', '13C']]) and self.me_num > 0:
                                sys.sp[k].N.val = abundance(k, sys.sp['HI'].N.val, getattr(self, s.N.addinfo).val)
                        if s.N.addinfo == 'iso':
                            #print(self.iso.addinfo)
                            if 'D/H' in self.iso.addinfo and 'HI' in sys.sp.keys():
                                if 'DI' in k:
                                    sys.sp[k].N.val = sys.sp['HI'].N.val + self.iso.val
                            if '13C/12C' in self.iso.addinfo:
                                if '13CI' in k and k.replace('13', '') in sys.sp.keys():
                                    sys.sp[k].N.val = sys.sp[k.replace('13', '')].N.val + self.iso.val
                                #print(k, '13CO' in k, k.replace('13', '') in sys.sp.keys(), sys.sp.keys())
                                if '13CO' in k and k.replace('13', '') in sys.sp.keys():
                                    #print(k.replace('13', ''), sys.sp[k.replace('13', '')].N.val)
                                    sys.sp[k].N.val = sys.sp[k.replace('13', '')].N.val + self.iso.val

        for k, v in self.tieds.items():
            self.setValue(k, self.getValue(v))

    def getPar(self, name):
        s = name.split('_')
        par = None
        if s[0] in ['mu', 'iso', 'hcont']:
            if hasattr(self, s[0]):
                par = getattr(self, s[0])

        if s[0] in ['me', 'cont', 'res', 'cf', 'zero', 'displ', 'disps', 'dispz', 'sts', 'stNl', 'stNu',
                    'covmode', 'cova', 'covd', 'covasys', 'covr', 'cov1', 'cov2', 'covov',
                    'srcmode', 'srcbeta', 'srcrref', 'srclref']:
            if hasattr(self, name):
                par = getattr(self, name)

        if s[0] in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            if len(self.sys) > int(s[1]) and hasattr(self.sys[int(s[1])], s[0]):
                par = getattr(self.sys[int(s[1])], s[0])

        if s[0] in ['b', 'N', 'I']:
            if len(self.sys) > int(s[1]) and s[2] in self.sys[int(s[1])].sp and hasattr(self.sys[int(s[1])].sp[s[2]], s[0]):
                par = getattr(self.sys[int(s[1])].sp[s[2]], s[0])

        if par is None:
            raise ValueError('Fit model has no {:} parameter'.format(name))
        else:
            return par

    def getValue(self, name, attr='val'):
        par = self.getPar(name)
        if par is None:
            raise ValueError('Fit model has no {:} parameter'.format(name))
        else:
            return getattr(par, attr)

    def list(self, ind=None):
        return list(self.pars(ind).values())

    def list_check(self):
        for par in self.list():
            par.check()

    def coverage_parameter_active(self, name):
        prefix = name.split('_')[0]
        active_by_mode = {
            'full': set(),
            'ppc': {'cf'},
            'powerlaw': {'cova'},
            'powerlaw_ppc': {'cf', 'cova'},
            'powerlaw_ppc_sysa': {'cf', 'covasys'},
            'powerlaw_sysa_global': {'covasys'},
            'powerlaw_radial': {'cova', 'covd'},
            'powerlaw_radial_ppc': {'cf', 'cova', 'covd'},
            'two_zone': {'cf', 'covr'},
            'cfov': {'cov1', 'cov2', 'covov'},
        }
        if prefix not in {'cf', 'cova', 'covd', 'covasys', 'covr', 'cov1', 'cov2', 'covov'}:
            return True
        if prefix == 'covd' and self.source_model() != 'ss73':
            return False
        return prefix in active_by_mode[self.coverage_model()]

    def list_fit(self):
        return [par for par in self.list()
                if par.fit & par.vary
                & self.coverage_parameter_active(str(par))
                & self.source_parameter_active(str(par))]

    def list_vary(self):
        return [par for par in self.list() if par.vary]

    def list_names(self):
        return [str(par) for par in self.list()]

    def list_species(self):
        species = []
        for sys in self.sys:
            for sp in sys.sp.keys():
                if sp not in species:
                    species.append(sp)
        return species

    def pars(self, ind=None):
        pars = OrderedDict()
        if ind in [None, -1]:
            pars[str(self.covmode)] = self.covmode
            for source_par in ['srcmode', 'srcbeta', 'srcrref', 'srclref']:
                p = getattr(self, source_par)
                pars[str(p)] = p
            for attr in ['mu', 'iso', 'hcont', 'zero']:
                if hasattr(self, attr):
                    p = getattr(self, attr)
                    pars[str(p)] = p
            if self.cont_num > 0:
                for i in range(self.cont_num):
                    for k in range(self.cont[i].num):
                        attr = 'cont_' + str(i) + '_' + str(k)
                        if hasattr(self, attr):
                            p = getattr(self, attr)
                            pars[str(p)] = p
            if self.me_num > 0:
                for i in range(self.me_num):
                    attr = 'me_' + str(i)
                    if hasattr(self, attr):
                        p = getattr(self, attr)
                        pars[str(p)] = p
            if self.res_num > 0:
                for i in range(self.res_num):
                    attr = 'res_' + str(i)
                    if hasattr(self, attr):
                        p = getattr(self, attr)
                        pars[str(p)] = p
            if self.cf_fit and self.cf_num > 0:
                for i in range(self.cf_num):
                    for attr in ['cf_' + str(i)] + self.coverage_parameter_names(i):
                        if hasattr(self, attr):
                            p = getattr(self, attr)
                            pars[str(p)] = p
                self.ensure_system_powerlaw_parameters()
                for system_id in range(len(self.sys)):
                    attr = f'covasys_{system_id}'
                    if hasattr(self, attr):
                        p = getattr(self, attr)
                        pars[str(p)] = p
            if self.disp_num > 0:
                for i in range(self.disp_num):
                    for attr in ['displ', 'disps', 'dispz']:
                        attr = attr + '_' + str(i)
                        if hasattr(self, attr):
                            p = getattr(self, attr)
                            pars[str(p)] = p
            if self.stack_num > 0:
                for i in range(self.stack_num):
                    for attr in ['sts', 'stNl', 'stNu']:
                        attr = attr + '_' + str(i)
                        if hasattr(self, attr):
                            p = getattr(self, attr)
                            pars[str(p)] = p
        if len(self.sys) > 0:
            for i, sys in enumerate(self.sys):
                if ind in [None, i]:
                    for attr in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
                        if hasattr(sys, attr):
                            p = getattr(sys, attr)
                            pars[str(p)] = p
                    for sp in sys.sp.values():
                        for attr in ['b'] + (sp.type == 'abs') * ['N'] + (sp.type == 'em') * ['I']:
                            if hasattr(sp, attr):
                                p = getattr(sp, attr)
                                pars[str(p)] = p
        return pars


    def list_total(self):
        pars = OrderedDict()
        for sys in self.sys:
            for k, v in sys.total.items():
                pars['_'.join(['N', str(self.sys.index(sys)), k])] = v.N
        for k, v in self.total.sp.items():
            pars['_'.join(['N', 'total', k])] = v.N

        return pars

    def readPars(self, name):
        if name.count('*') > 0:
            name = name.replace('*' * name.count('*'), 'j' + str(name.count('*')))
        s = name.split()
        attrs = ['val', 'min', 'max', 'step', 'vary', 'addinfo']
        if len(s) == len(attrs):
            s.append('')

        if 'cont_' in s[0]:
            self.cont_num = max(self.cont_num, int(s[0].split('_')[1]) + 1)

        if 'res' in s[0]:
            self.res_num = max(self.res_num, int(s[0][4:]) + 1)

        if 'iso' in s[0]:
            self.add(s[0], addinfo=s[6])

        if 'me' in s[0]:
            if s[0] == 'me':
                s[0] = 'me_0'
            self.me_num = max(self.me_num, int(s[0][3:]) + 1)

        if s[0].startswith('cf_'):
            self.cf_fit = True
            attrs = ['val', 'left', 'right', 'step', 'vary', 'addinfo']
            self.parent.plot.add_pcRegion()
        if s[0] in ['covmode', 'srcmode', 'srcbeta', 'srcrref', 'srclref'] or s[0].startswith(('cova_', 'covd_', 'covasys_', 'covr_', 'cov1_', 'cov2_', 'covov_')):
            if not hasattr(self, s[0]):
                self.add(s[0])

        if 'disp' in s[0]:
            self.disp_num = max(self.disp_num, int(s[0][6:]) + 1)

        if 'sts' in s[0]:
            self.stack_num = max(self.stack_num, int(s[0][4:]) + 1)

        for attr, val in zip(reversed(attrs), reversed(s[1:])):
            self.setValue(s[0], val, attr)
            if attr == 'val':
                self.setValue(s[0], float(s[4]), 'unc')
            if attr == 'addinfo' and 'cont_' in s[0]:
                self.cont[int(s[0].split('_')[1])].fromInfo(val)

        if s[0].startswith('cf_'):
            self.parent.plot.pcRegions[-1].updateFromFit()

    def showLines(self, sp=None):
        if sp is None:
            sp = list(set([s for sys in self.sys for s in sys.sp.keys()]))

        print('showLines start')
        if len(sp) > 0:
            self.parent.console.exec_command('add ' + ' '.join(sp))
        print('showLines finish')

    def fromLMfit(self, result, quiet=False):
        for p in result.params.keys():
            par = result.params[p]
            name = str(par.name).replace('l4', '****').replace('l3', '***').replace('l2', '**').replace('l1', '*')
            self.setValue(name, self.pars()[name].ref(par.value), 'val')
            if not quiet:
                print(p, par.stderr)
            if isinstance(self.pars()[name].ref(par.stderr, attr='unc'), float):
                self.setValue(name, self.pars()[name].ref(par.stderr, attr='unc'), 'unc')
                self.setValue(name, self.pars()[name].ref(par.stderr, attr='unc'), 'step')
            else:
                self.setValue(name, 0, 'unc')

    def fromJulia(self, res, unc):
        s = ''
        for i, p in enumerate(self.list_fit()):
            if not self.parent.blindMode:
                print(i, p, res[i])
            self.setValue(p.__str__(), res[i])
            self.setValue(p.__str__(), unc[i], 'unc')
            self.setValue(p.__str__(), unc[i], 'step')
            #s += p.__str__() + ': ' + str(self.getValue(p.__str__(), 'unc')) + '\n'
            s += p.str(attr='lmfit') + '\n'
        return s

    def save(self):
        self.saved = OrderedDict()
        for k, v in self.pars().items():
            p = copy(v)
            for attr in ['val', 'min', 'max', 'step', 'addinfo']:
                setattr(p, attr, copy(getattr(v, attr)))
            self.saved[k] = p

    def load(self):
        for k in self.pars().keys():
            for attr in ['val', 'min', 'max', 'step', 'addinfo']:
                self.setValue(k, getattr(self.saved[k], attr), attr)

    def shake(self, scale=1):
        for p in self.list_fit():
            self.setValue(str(p), p.val + np.random.normal(scale=scale) * p.step, 'val')

    def setSpecific(self):
        self.addSys(z=2.8083543)
        self.setValue('b_0_SiIV', 10.6)
        self.setValue('N_0_SiIV', 12.71)
        self.addSys(z=2.8085)
        self.setValue('b_1_SiIV', 7.6)
        self.setValue('N_1_SiIV', 12.71)

    def __str__(self):
        return '\n'.join([str(s) for s in self.sys])


# Classes for parallelization
class spectra(list):
    def __init__(self):
        super(spectra).__init__()

    def set_lines(self):
        for s in self:
            s.set_lines()

class spec:
    def __init__(self, parent, x, y, err):
        self.x = x
        self.y = y
        self.err = err
        self.fit = None

    def set_lines(self, tlim=0.01, all=True, debug=False):
        """
        Function to prepare lines to fit.
          - ind         : specify component to fit
          - all         : if all then look for all the lines located in the  normalized spectrum

        Prepared lines where fit will be calculated are stored in self.fit_lines list.
        """
        self.fit_lines = []

        if self.x.shape[0] > 0:
            for sys in self.fit.sys:
                for sp in sys.sp.keys():
                    lin = self.parent.atomic.list(sp)
                    for l in lin:
                        l.b = sys.sp[sp].b.val
                        l.logN = sys.sp[sp].N.val
                        l.z = sys.z.val
                        l.recalc = True
                        l.sys = sys.ind
                        l.range = tau(l, resolution=self.resolution()).getrange(tlim=tlim)
                        region = coverage_region_for_line(
                            self.parent.fit, l, exposure_index=self.ind()
                        )
                        l.cf = -1 if region is None else region
                        if all:
                            if any([x[0] < p < x[-1] for p in l.range]):
                                self.fit_lines += [l]
                        else:
                            if np.sum(np.where(np.logical_and(x >= l.range[0], x <= l.range[1]))) > 0:
                                self.fit_lines += [l]
        if debug:
            print('findFitLines', self.fit_lines, [l.cf for l in self.fit_lines])

    def chi(self):
        pass

class calc_fit:
    def __init__(self, parent):
        self.parent = parent

    def set_data(self, data):
        self.s = spectra(self)
        for d in data:
            self.s.append(spec(self, d[0], d[1], d[2]))

    def set_model(self, pars, fit=None):
        if fit is None:
            self.fit = fitPars(self.parent)
            self.fit.readPars(pars)
        else:
            self.fit = fit

    def set_lines(self, lines):
        self.lines = lines


