import collections
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
import os, sys
sys.path.append('C:/Science/python')
from spectro.atomic import Asplund2009

class par():
    """
    class for parameters handle:
        - name         : name of parameter, can be:
                            'n'    - number density
                            'T'    - temperature
                            'f'    - molecular fraction
                            'rad'  - Scaling constant of radiation filed iether:
                                        flux in the units of standard field (in case of sed_type: 'Habing', 'Draine', 'Mathis')
                                        flux at 1kpc distance for source specified by agn keyword (in case of sed_type: 'AGN', 'QSO', 'GRB')
                            'e'    - electron density
                            'CMB'  - CMB temperature
        - init         : init value
        - range        : range of parameter values, 2 element array
        - init_range   : range of initial parameter values for MCMC, 2 element array
        - prior        : prior values for the calculations, <a> object
        - label        : label on plot
    """

    def __init__(self, name, prior=None, parent=None):
        self.name = name
        self.prior = prior
        self.parent = parent

        # >>> metallicity
        if name == 'Z':
            self.init = 1
            self.range = [1e-5, 100]
            self.label = r'$\rm Z$'

        # >>> particle number density
        if name == 'n':
            self.init = 1
            self.range = [1e-2, 1e5]
            self.label = r'$n$'

        # >>> temperature
        if name == 'T':
            self.init = 5000
            self.range = [1, 1e5]
            self.label = r'$T$'

        # >>> UV radiation field (log)
        if name == 'UV':
            self.init = 1
            self.range = [1e-4, 1e6]
            self.label = r'$\xi$'

        # >>> Cosmic ray Ionization rate (primary per H atom, log in units 1e-16 s^-1)
        if name == 'CR':
            self.init = 1e-16
            self.range = [1e-20, 1e-13]
            self.label = r'$\zeta$'

        # >>> electron number density (log)
        if name == 'n_e':
            self.init = 1e-4
            self.range = [1e-10, 1e4]
            self.label = r'$n_{\rm e}$'
            self.fixed = None

        # >>> molecular fraction
        if name == 'mol':
            self.init = 0
            self.range = [0, 1]
            self.label = r'$f_{\rm H_2}$'
            self.fixed = None

        # >>> CMB temperature
        if name == 'CMB':
            self.init = 2.72548 * (1 + self.parent.z)
            self.range = [1, 20]
            self.label = r'T$_{\rm CMB}$'

        self.value = self.init

    def set(self, x):
        self.value = np.min([np.max([x, self.range[0]]), self.range[-1]])

    def show(self):
        print('parameter name: ', self.name)
        print('initial value: ', self.init)
        print('parameters range: ', self.range)
        print('prior:', self.prior)
        print('initial value range (for MCMC): ', self.init_range)

    def __repr__(self):
        return '{:s}: {:.2f}'.format(self.name, self.value)

    def __str__(self):
        return '{:s}: {:.2f}'.format(self.name, self.value)


class ISM():
    def __init__(self, **kwargs):
        if 'z' in kwargs.keys():
            self.z = kwargs['z']
        self.initialize_pars(**kwargs)
        self.heating_types = ['photoelectric', 'cosmicray']
        self.cooling_types = ['Lya', 'CII', 'OI', 'rec']

    def initialize_pars(self, **kwargs):
        self.pars = collections.OrderedDict()
        for name in ['Z', 'n', 'T', 'UV', 'CR', 'n_e', 'mol']:
            self.pars[name] = par(name, parent=self)
            if name in kwargs.keys():
                print(name, kwargs[name])
                self.pars[name].value = kwargs[name]

    def p(self, attr):
        return self.pars[attr].value

    def depletion(self, element, Z=None, kind='Bialy'):
        if Z == None:
            Z = self.p('Z')
        d = {'C': 0.53, 'O': 0.41, 'Si': 0.5}
        if kind == 'Bialy' and element in d.keys():
            return 1 - d[element] * self.dust_to_gas_ratio(Z=Z) / Z
        else:
            return 1

    def dust_to_gas_ratio(self, Z=None):
        Z_0, alpha = 0.2, 3
        if Z == None:
            Z = self.p('Z')
        if Z > Z_0:
            return Z
        else:
            return Z_0 * (Z / Z_0) ** alpha

    def abundance(self, element, Z=None, depl_kind='Bialy'):
        if Z == None:
            Z = self.p('Z')

        if element == 'dust':
            return self.dust_to_gas_ratio(Z=Z)
        else:
            return 10**Asplund2009(element)[0] * Z * self.depletion(element, kind=depl_kind)

    def alpha_rr(self):
        T4 = self.p('T') / 10000
        return 2.54e-13 * T4 ** (-0.8163 - 0.0208 * np.log(T4))

    def alpha_gr(self, n_e):
        psi = self.p('UV') * self.p('T') ** 0.5 / n_e
        c = [12.25, 8.07E-06, 1.378, 5.09E+02, 1.59E-02, 0.4723, 1.10E-05]
        return 1e-14 * c[0] / (1 + c[1] * psi ** c[2] * (1 + c[3] * self.p('T') ** c[4] * psi ** (-c[5] - c[6] * np.log(self.p('T'))))) * self.abundance('dust')

    def get_f_ioniz(self, x):
        k_eff = 1.67 * self.p('CR') * ((1 - self.p('mol')) + 0.1 * (self.p('mol')) / 2) / self.p('n')
        rhs = self.alpha_rr() * (self.abundance('C') + x) * x + self.alpha_gr(self.abundance('C') + x) * x
        return k_eff - rhs

    def x_e(self, method='Nelder-Mead', force=False):
        if self.pars['n_e'].fixed == None or force:
            return (self.abundance('C') + fsolve(self.get_f_ioniz, 1e-4, args=())[0])
            #return (self.abundance('C') + minimize(self.get_f_ioniz, 1e-3, args=(), method=method).x[0])
        else:
            return self.p('n_e') / self.p('n')

    def n_e(self, force=False):
        #print(self.p('n'), self.x_e())
        return self.x_e(force=force) * self.p('n')

    def g_PE(self):
        psi = self.p('UV') * self.p('T') ** 0.5 / self.n_e() / 0.5
        #return 0.049 / (1 + 5.9e-13 * psi ** 0.73) + 0.037 * (self.p('T') / 10 ** 4) ** 0.7 / (1 + 3.4e-4 * psi)
        return 0.049 / (1 + 3.9e-4 * psi ** 0.73) + 0.037 * (self.p('T') / 10 ** 4) ** 0.7 / (1 + 2.0e-4 * psi)
        #return 0.049 / (1 + (psi / 1925) ** 0.73) + 0.037 * (self.p('T') / 10 ** 4) ** 0.7 / (1 + psi / 5000)

    def heating(self, kind=''):
        if kind == 'photoelectric':
            return 2.2e-24 * self.abundance('dust') * self.p('n') * self.p('UV') * self.g_PE()

        if kind == 'cosmicray':
            x_e = self.x_e()
            return 1.03e-11 * self.p('n') * self.p('CR') * (1 + 4.06 * (x_e / (x_e + 0.07)) ** 0.5) + 4.6e-10 * self.p('CR') * x_e * self.p('n')

        return 0

    def cooling(self, kind=''):
        if kind == 'Lya':
            return 7.3e-19 * self.n_e() * self.p('n') * np.exp(-118400 / self.p('T'))

        if kind == 'CII':
            return 2.54e-14 * (2.8e-6 * self.n_e() / self.p('n') * self.p('T') ** -0.5 + 8e-10) * self.abundance('C') * np.exp(-92 / self.p('T')) * self.p('n') ** 2

        if kind == 'OI':
            return 1e-26 * self.p('T') ** 0.5 * self.abundance('O') * (24 * np.exp(-228 / self.p('T')) + 7 * np.exp(-326 / self.p('T'))) * self.p('n') ** 2

        if kind == 'rec':
            return 4.65e-30 * self.p('T') ** 0.94 * (self.p('UV') * self.p('T') ** 0.5 / self.n_e() / 0.5) ** (0.73 / self.p('T') ** 0.068) * self.n_e() * self.p('n') * 0.5

        return 0

    def thermal_rates(self, kind='', n=[], T=[]):
        rate = np.zeros_like(n)
        if len(n) == 0 and len(T) != 0:
            n = np.ones_like(T) * self.p('n')
        if len(T) == 0 and len(n) != 0:
            T = np.ones_like(T) * self.p('T')
        if len(n) == 0 and len(T) == 0:
            n, T = [self.p('n')], [self.p('T')]
        for i, ni, Ti in zip(range(len(n)), n, T):
            self.pars['n'].set(ni)
            self.pars['T'].set(Ti)
            if kind in self.cooling_types:
                rate[i] = self.cooling(kind=kind)
            elif kind in self.heating_types:
                rate[i] = self.heating(kind=kind)
        return rate

    def thermal_balance(self):
        eq = 0
        for h in self.heating_types:
            #print(h, self.heating(h))
            eq += self.heating(h)
        for c in self.cooling_types:
            #print(c, self.cooling(c))
            eq -= self.cooling(c)
        return np.abs(eq)

    def thermal_mini(self, x):
        self.pars['T'].set(x)
        self.pars['n_e'].fixed = self.n_e(force=True)
        return self.thermal_balance()


    def phase_diagram(self, n=None, method='Nelder-Mead'):
        if n == None:
            n = np.logspace(-2, 4, 100)
        T = np.zeros_like(n)
        for i, ni in enumerate(n):
            self.pars['n'].value = ni
            T[i] = minimize(self.thermal_mini, 100, args=(), method=method).x[0]

        # >> calc mask for thermally stable regions dP/dn > 0
        m = np.diff(np.concatenate([n*T, [(n*T)[-1]]])) >= 0

        return n, T, m


if __name__ == '__main__':
    print('executing main program code')
    ism = ISM(Z=1)
    print(ism.pars)

    # >>> check cooling heating rates:
    if 0:
        fig, ax = plt.subplots()

        n, T, m = ism.phase_diagram()
        for c in ism.cooling_types:
            ax.plot(np.log10(n), np.log10(ism.thermal_rates(kind=c, n=n, T=T) / n), ls='-', label=c)
        for h in ism.heating_types:
            ax.plot(np.log10(n), np.log10(ism.thermal_rates(kind=h, n=n, T=T) / n), ls='--', label=h)
            #print(c, ism.cooling_rate(kind=c, n=n, T=T))
        #for ni, Ti in zip()
        ax.set_xlim([-2, 4])
        ax.set_ylim([-30, -25])
        fig.legend()
        plt.show()
    if 0:
        for f in [0, 1]:
            ism.pars['mol'].value = f
            #print(f, ism.x_e(force=True))

    if 1:
        fig, ax = plt.subplots()
        cmap = cm.get_cmap('winter')
        Z_range = [0.001, 0.01, 0.2, 1]
        if 1:
            for Z in Z_range:
                ism.pars['Z'].value = Z
                for f in [0, 1]:
                    ism.pars['mol'].value = f
                    n, T, m = ism.phase_diagram()
                    P = T * n
                    print((np.log10(Z) - np.log10(np.min(Z_range))) / (np.log10(np.max(Z_range)) - np.log10(np.min(Z_range))))
                    c = cmap((np.log10(Z) - np.log10(np.min(Z_range))) / (np.log10(np.max(Z_range)) - np.log10(np.min(Z_range))))
                    ax.plot(np.log10(n), np.log10(P), '--', c=c, label=f"Z={Z}")
                    P[~m] = np.nan
                    ax.plot(np.log10(n), np.log10(P), c=c)

        fig.legend()
        plt.show()