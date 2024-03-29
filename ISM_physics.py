import collections
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))[:-8])
from spectro.atomic import Asplund2009
from spectro.pyratio import pyratio

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
            self.range = [1e-6, 100]
            self.label = r'$\rm Z$'

        # >>> particle number density
        if name == 'n':
            self.init = 1
            self.range = [1e-4, 1e7]
            self.label = r'$n$'

        # >>> temperature
        if name == 'T':
            self.init = 5000
            self.range = [3, 1e5]
            self.label = r'$T$'

        # >>> UV radiation field (log)
        if name == 'uv':
            self.init = 1
            self.range = [1e-4, 1e6]
            self.label = r'$\xi$'

        # >>> Cosmic ray Ionization rate (primary per H atom, log in units 1e-16 s^-1)
        if name == 'cr':
            self.init = 1e-16
            self.range = [1e-20, 1e-13]
            self.label = r'$\zeta$'

        # >>> PAH abundance (log in dimensionless units)
        if name == 'pah':
            self.init = 6e-7
            self.range = [1e-9, 1e-5]
            self.label = r'$PAH$'

        # >>> fraction of electron, i.e. n_e / n_H (log)
        if name == 'x_e':
            self.init = 1e-4
            self.range = [1e-10, 1]
            self.label = r'$x_{\rm e}$'
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

        # >>> Turbulence heating
        if name == 'turb':
            self.init = 1e-27
            self.range = [0, 1e-24]
            self.label = r'$\lambda_{\rm turb}$'

        self.value = self.init

    def set(self, x, attr='value'):
        if attr == 'value':
            self.value = np.min([np.max([x, self.range[0]]), self.range[-1]])
        else:
            setattr(self, attr, x)
        if attr == 'fixed':
            self.value = self.fixed

    def show(self):
        print('parameter name: ', self.name)
        print('initial value: ', self.init)
        print('parameters range: ', self.range)
        print('prior:', self.prior)
        print('initial value range (for MCMC): ', self.init_range)

    def draw(self, mu=0, sigma=0.5):
        self.set(self.init * 10 ** (np.random.normal(mu, sigma)))

    def __repr__(self):
        return '{:s}: {:.2f}'.format(self.name, self.value)

    def __str__(self):
        return '{:s}: {:.2f}'.format(self.name, self.value)


class ISM():
    def __init__(self, **kwargs):

        # the redshift determines the CMB intensity and impact on the cooling due to fine-structure excitation.
        if 'z' in kwargs.keys():
            self.z = kwargs['z']
        else:
            self.z = 0

        # Default values of dtg_cutoff and dtg_alpha is from the fit of Remy-Ruyer+2014.
        if 'dtg_cutoff' in kwargs.keys():
            self.dtg_cutoff = kwargs['dtg_cutoff']
        else:
            self.dtg_cutoff = 0.2
        if 'dtg_slope' in kwargs.keys():
            self.dtg_slope = kwargs['dtg_slope']
        else:
            self.dtg_slope = 3
        if 'dtg_disp' in kwargs.keys():
            self.dtg_disp = kwargs['dtg_disp']
        else:
            self.dtg_disp = 0
        self.initialize_pars(**kwargs)
        self.heating_types = ['photoelectric', 'cosmicray', 'photo_c', 'turb'] # , 'grav']
        self.cooling_types = ['Lya', 'CII_Klessen', 'OI_Klessen', 'rec']

        self.pr = pyratio(z=self.z, sed_type=None)
        self.pr.set_pars(['T', 'n', 'f', 'rad'])
        self.pr.set_fixed('f', -10)
        for sp in ['OI', 'CII', 'CI']:
            self.pr.add_spec(sp)
        #print(self.pr.species['CII'].coll_rate('H', 1, 0, 2))

    def initialize_pars(self, **kwargs):
        self.pars = collections.OrderedDict()
        for name in ['Z', 'n', 'T', 'uv', 'cr', 'pah', 'x_e', 'mol', 'turb']:
            self.pars[name] = par(name, parent=self)
            if name in kwargs.keys():
                self.pars[name].init = kwargs[name]
                self.pars[name].value = kwargs[name]

    def p(self, attr):
        return self.pars[attr].value

    def update_pars(self, **kwargs):
        for name in ['Z', 'n', 'T', 'uv', 'cr', 'pah', 'x_e', 'mol', 'turb']:
            if name in kwargs.keys():
                self.pars[name].set(kwargs[name])

        # >>> update ionization fraction if necessary:
        if self.pars['x_e'].fixed == None and 'x_e' not in kwargs.keys():
            self.pars['x_e'].set(self.x_e(calc=True))
        self.phi_pah = self.p('pah') / 6e-7

    def depletion(self, element, Z=None, kind='Bialy'):
        if Z == None:
            Z = self.p('Z')
        d = {'C': 0.53, 'O': 0.41, 'Si': 0.5}
        if kind == 'Bialy' and element in d.keys():
            return 1 - d[element] * self.dust_to_gas_ratio(Z=Z) / Z
        else:
            return 1

    def dust_to_gas_ratio(self, Z=None):
        if Z == None:
            Z = self.p('Z')
        if Z > self.dtg_cutoff:
            return Z * 10 ** (self.dtg_disp)
        else:
            return self.dtg_cutoff * (Z / self.dtg_cutoff) ** self.dtg_slope * 10 ** (self.dtg_disp)

    def ionization_state(self, species, n=0, T=4):
        #print(species, n, 1 / (1 + np.exp((n - 10**2)/3)) + 10**-4, (1 - 1 / (1 + np.exp((n - 10**2)/3))) * (1 / (1 + np.exp((10**n - 10**3)/3))) + 10**-4)
        if species == 'CII':
            return 1 / (1 + np.exp((n - 10 ** 4) / 3)) + 1e-4
        elif species == 'CI':
            return (1 - 1 / (1 + np.exp((n - 10 ** 4) / 3))) * (1 / (1 + np.exp((n - 10 ** 6) / 3))) + 1e-4
        elif species == 'CO':
            return 1 - (1 / (1 + np.exp((n - 10 ** 6) / 3)))
        else:
            return np.ones_like(n)

    def abundance(self, element, Z=None, depl_kind='Bialy'):
        if Z == None:
            Z = self.p('Z')

        if element == 'dust':
            return self.dust_to_gas_ratio(Z=Z)
        else:
            return 10 ** Asplund2009(element)[0] * Z * self.depletion(element, Z=Z, kind=depl_kind) * self.ionization_state(element, n=self.p('n'))

    def alpha_rr(self):
        T4 = self.p('T') / 10000
        return 2.54e-13 * T4 ** (-0.8163 - 0.0208 * np.log(T4))

    def alpha_gr(self, n_e):
        psi = self.p('uv') * self.p('T') ** 0.5 / n_e
        c = [12.25, 8.07E-06, 1.378, 5.09E+02, 1.59E-02, 0.4723, 1.10E-05]
        return 1e-14 * c[0] / (1 + c[1] * psi ** c[2] * (1 + c[3] * self.p('T') ** c[4] * psi ** (-c[5] - c[6] * np.log(self.p('T'))))) * self.abundance('dust')

    def get_f_ioniz(self, x):
        k_eff = 1.67 * self.p('cr') * ((1 - self.p('mol')) + 0.1 * (self.p('mol')) / 2) / self.p('n')
        rhs = self.alpha_rr() * (self.abundance('C') + x) * x + self.alpha_gr(self.abundance('C') + x) * x
        return k_eff - rhs

    def x_e(self, method='Nelder-Mead', calc=False):
        if calc:
            return min(self.abundance('C') + fsolve(self.get_f_ioniz, 1e-4, args=())[0], 0.1)
            #return (self.abundance('C') + minimize(self.get_f_ioniz, 1e-3, args=(), method=method).x[0])
        else:
            return self.pars['x_e'].value if self.pars['x_e'].fixed == None else self.pars['x_e'].fixed

    def n_e(self, calc=False):
        #print(self.p('x_e'), self.x_e() * self.p('n'))
        return self.x_e(calc=calc) * self.p('n')

    def g_PE(self):
        psi = self.p('uv') * self.p('T') ** 0.5 / self.n_e() / self.phi_pah
        #print(self.p('T'), self.p('T') ** 0.5 , self.n_e(), self.p('n'), psi)
        # Bialy+2019 - BE CAREFULL - A POSSIBLE TYPO:
        #return 0.049 / (1 + 5.9e-13 * psi ** 0.73) + 0.037 * (self.p('T') / 10 ** 4) ** 0.7 / (1 + 3.4e-4 * psi)
        # Wolfire+2003:
        return 0.049 / (1 + 4.0e-3 * psi ** 0.73) + 0.037 * (self.p('T') / 10 ** 4) ** 0.7 / (1 + 2.0e-4 * psi)
        #return 0.049 / (1 + (psi / 1925) ** 0.73) + 0.037 * (self.p('T') / 10 ** 4) ** 0.7 / (1 + psi / 5000)

    def heating(self, kind=''):
        """
        provides various heating rates in [erg cm-3 s-1]
        :param kind:   type of heating
        :return:       heating rate in [erg cm-3 s-1]
        """
        kind = kind.split('_') + ['']
        if kind[0] == 'photoelectric':
            return 2.2e-24 * self.abundance('dust') * self.p('n') * self.p('uv') * self.g_PE()

        if kind[0] == 'cosmicray':
            # >> from Bialy+2019
            x_e = self.x_e()
            #print(self.p('n'), x_e)
            return self.p('cr') * 6.43 * 1.602e-12 * (1 + 4.06 * (x_e / (x_e + 0.07)) ** 0.5) * self.p('n') * (1 + x_e)

        if kind[0] == 'photo_c':
            return 2e-22 * self.abundance('CII') * 1e-2 * self.p('n')

        if kind[0] == 'turb':
            if 0:
                return self.p('turb') * (self.p('n')) ** 0.5
            else:
                return self.p('turb') * self.p('n')  #* 1 / (1 + np.exp(-(np.log10(self.p('n')) - 1.5) / 0.3))

        if kind[0] == 'grav':
            return 2.6e-31 * self.p('n') * self.p('T') * 1 / (1 + np.exp(-(np.log10(self.p('n')) - 2) / 0.5))

        return 0

    def cooling(self, kind=''):
        """
        provides various colling rates in [erg cm-3 s-1]
        :param kind:   type of cooling
        :return:       colling rate in [erg cm-3 s-1]
        """
        kind = kind.split('_') + ['']
        if kind[0] == 'Lya':
            # from Klessen & Glover
            n_e = self.n_e()
            return 7.3e-19 * n_e * (self.p('n') - n_e) * np.exp(-118400 / self.p('T')) / (1 + (self.p('T') / 1e5) ** 0.5)

        if kind[0] == 'CII':
            if kind[1] in ['Klessen']:
                #return 2.88e-20 * self.p('n') * self.abundance('C') * (2 * np.exp(-91.21 / self.p('T'))) / (1 + 2e3 / self.p('n') + 2 * np.exp(-91.21 / self.p('T')))
                return 2.88e-20 * self.p('n') * self.abundance('CII') * (2 * np.exp(-91.21 / self.p('T'))) / (1 + self.pr.species['CII'].critical_density('H', 1, 0, np.log10(self.p('T'))) / self.p('n') + 2 * np.exp(-91.21 / self.p('T')))
            if kind[1] in ['Wolfire']:
                return (3.15e-27 * self.p('n') + 1.4e-24 * self.p('T') ** -0.5 * self.n_e()) * np.exp(-92 / self.p('T')) * self.p('Z') * self.p('n')
            if kind[1] in ['Bialy']:
                #print(self.n_e() / self.p('n'))
                return 2.54e-14 * (2.8e-6 * self.n_e() / self.p('n') * self.p('T') ** -0.5 + 8e-10) * self.abundance('CII') * np.exp(-92 / self.p('T')) * self.p('n') ** 2
            if kind[1] in ['pyratio', '']:
                return self.pr.calc_cooling(species='CII', n=np.log10(self.p('n')), T=np.log10(self.p('T'))) * self.abundance('CII') * self.p('n')
            if kind[1] in ['Barinovs']:
                return 1e-24 * np.exp(-91.2 / self.p('T')) * (16 + 0.344 * np.sqrt(self.p('T')) + 47.7 / self.p('T')) * self.abundance('CII') * self.p('n') * self.p('n')

        if kind[0] == 'CI':
            if kind[1] in ['pyratio', '']:
                return self.pr.calc_cooling(species='CI', n=np.log10(self.p('n')), T=np.log10(self.p('T'))) * self.abundance('CI') * self.p('n')

        if kind[0] == 'OI':
            if kind[1] in ['Wolfire']:
                return 2.5e-27 * (self.p('T') / 100) ** 0.4 * (np.exp(-228 / self.p('T'))) * self.p('Z') * self.p('n') ** 2
            if kind[1] in ['Klessen']:
                return 2.79e-18 * self.p('n') * self.abundance('O') * (3 / 5 * np.exp(-228 / self.p('T'))) / (1 + self.pr.species['OI'].critical_density('H', 1, 0, np.log10(self.p('T'))) / self.p('n') + 3 / 5 * np.exp(-228 / self.p('T')))
            if kind[1] in ['pyratio', '']:
                return self.pr.calc_cooling(species='OI', n=np.log10(self.p('n')), T=np.log10(self.p('T'))) * self.abundance('O') * self.p('n')

        if kind[0] == 'rec':
            return 4.65e-30 * self.p('T') ** 0.94 * (self.p('uv') * self.p('T') ** 0.5 / self.n_e() / self.phi_pah) ** (0.73 / self.p('T') ** 0.068) * self.n_e() * self.p('n') * self.phi_pah

        return 0

    def thermal_rates(self, kind='', n=[], T=[]):
        """
        Return the heating and cooling rates in [erg cm-3 s-1]
        Args:
            kind:
            n:
            T:

        Returns:

        """
        rate = np.zeros_like(n)
        if len(n) == 0 and len(T) != 0:
            n = np.ones_like(T) * self.p('n')
        if len(T) == 0 and len(n) != 0:
            T = np.ones_like(T) * self.p('T')
        if len(n) == 0 and len(T) == 0:
            n, T = [self.p('n')], [self.p('T')]
        for i, ni, Ti in zip(range(len(n)), n, T):
            self.update_pars(n=ni, T=Ti)
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
        self.update_pars(T=x)
        return self.thermal_balance()


    def phase_diagram(self, n=[], method='Nelder-Mead'):
        if len(n) == 0:
            n = np.logspace(-2, 4, 50)
        T = np.zeros_like(n)
        T[-1] = 1e4
        for i, ni in enumerate(n):
            self.pars['n'].value = ni
            self.update_pars(T=T[i-1])
            print(T[i-1], self.thermal_balance())
            T[i] = minimize(self.thermal_mini, T[i-1], args=(), method=method, options={'xatol': 0.1}).x[0]
            print(ni, T[i], self.thermal_balance())
        # >> calc mask for thermally stable regions dP/dn > 0
        m = np.diff(np.concatenate([n*T, [(n*T)[-1]]])) >= 0

        return n, T, m


if __name__ == '__main__':
    print('executing main program code')
    ism = ISM(z=0, Z=1, uv=8, cr=3e-15, pah=5.3e-7, turb=1e-27)
    #ism.pars['x_e'].set(1e-5, attr='fixed')
    # >>> check cooling heating rates:
    if 1:
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
        #ism.cooling_types.append('CII_Wolfire')
        #ism.cooling_types.append('CII_Klessen')
        #ism.cooling_types.append('CII_Barinovs')
        #ism.cooling_types.append('OI_Wolfire')
        #ism.cooling_types.append('OI_Klessen')

        n, T, m = ism.phase_diagram(n=np.logspace(-3, 4, 101))
        #n, T, m = ism.phase_diagram(n=np.logspace(-4, 4, 50))
        print(n, T)
        P = T * n
        ax[0].plot(np.log10(n), np.log10(T), '--', c='k')
        P[~m] = np.nan
        ax[0].plot(np.log10(n), np.log10(T), c='k')
        axd = ax[0].twinx()
        x_e = []
        for ni, Ti in zip(n, T):
            ism.update_pars(n=ni, T=Ti)
            x_e.append(ism.x_e())
        axd.plot(np.log10(n), np.log10(x_e), '--', color='dodgerblue')

        for c in ism.cooling_types:
            ax[1].plot(np.log10(n), np.log10(ism.thermal_rates(kind=c, n=n, T=T) / n), ls='-', label=f'${c}$')
        for h in ism.heating_types:
            ax[1].plot(np.log10(n), np.log10(ism.thermal_rates(kind=h, n=n, T=T) / n), ls='--', label=f'${h}$')
            #print(c, ism.cooling_rate(kind=c, n=n, T=T))
        #for ni, Ti in zip()
        ax[1].set_xlim([-4, 4.5])
        ax[1].set_ylim([-30, -23])
        fig.legend()
        plt.show()

    if 0:
        for f in [0, 1]:
            ism.pars['mol'].value = f
            #print(f, ism.x_e(calc=True))

    if 0:
        fig, ax = plt.subplots()
        cmap = cm.get_cmap('winter')
        if 0:
            Z_range = [0.001, 0.01, 0.1, 1]
            for Z in Z_range:
                ism.update_pars(Z=Z)
                c = cmap((np.log10(Z) - np.log10(np.min(Z_range))) / (np.log10(np.max(Z_range)) - np.log10(np.min(Z_range))))
                for f in [0]:
                    ism.update_pars(f=f)
                    n, T, m = ism.phase_diagram(n=np.logspace(-4, 4, 100))
                    P = T * n
                    ax.plot(np.log10(n), np.log10(P), '--', c=c, label=f"Z={Z}")
                    P[~m] = np.nan
                    ax.plot(np.log10(n), np.log10(P), c=c)
        if 1:
            uv_range = [0.3, 1, 3, 10]
            for uv in uv_range:
                ism.update_pars(uv=uv)
                c = cmap((np.log10(uv) - np.log10(np.min(uv_range))) / (np.log10(np.max(uv_range)) - np.log10(np.min(uv_range))))
                for f in [0]:
                    ism.update_pars(f=f)
                    n, T, m = ism.phase_diagram()
                    P = T * n
                    ax.plot(np.log10(n), np.log10(P), '--', c=c, label=f"UV={UV}")
                    P[~m] = np.nan
                    ax.plot(np.log10(n), np.log10(P), c=c)

        if 0:
            cr_range = [0.3e-16, 1e-16, 3e-16, 1e-15]
            uv_range = [0.3, 1, 3, 10]
            for cr, uv in zip(cr_range, uv_range):
                ism.update_pars(cr=cr, uv=uv)
                c = cmap((np.log10(uv) - np.log10(np.min(uv_range))) / (np.log10(np.max(uv_range)) - np.log10(np.min(uv_range))))
                for f in [0]:
                    ism.update_pars(f=f)
                    n, T, m = ism.phase_diagram()
                    P = T * n
                    ax.plot(np.log10(n), np.log10(P), '--', c=c, label=f"CR, UV={UV}")
                    P[~m] = np.nan
                    ax.plot(np.log10(n), np.log10(P), c=c)

        fig.legend()
        plt.show()

    if 0:
        ism = ISM(z=0, Z=1, uv=1, cr=1e-16, turb=1e-27, dtg_cutoff=0.1, dtg_slope=2.5, dtg_disp=0.4)
        x, y = [], []
        for Z in 10**(-2 * np.random.rand(100)):
            ism.update_pars(Z=Z)
            ism.dtg_disp = np.random.randn() * 0.4
            x.append(Z)
            y.append(ism.abundance("dust"))

        print(x, y)
        fig, ax = plt.subplots()
        ax.plot(np.log10(x), np.log10(162) - np.log10(y), 'ok')
        plt.show()

