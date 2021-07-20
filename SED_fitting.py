from astropy import constants as ac
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
from scipy.interpolate import interp1d

class photo_filter():
    def __init__(self, name):
        self.name = name
        self.load()
        self.set_zp()

    def load(self):
        self.data = np.genfromtxt(r'C:/science/python/spectro/data/SDSS/' + self.name + '.dat', comments='#', usecols=(0, 1), unpack=True)
        self.inter = interp1d(self.data[0], self.data[1], bounds_error=False, fill_value=0, assume_sorted=True)
        self.l_eff = np.sqrt(np.trapz(self.data[1] * self.data[0], x=self.data[0]) / np.trapz(self.data[1] / self.data[0], x=self.data[0])) * u.AA

    def set_zp(self):
        b = {'u': 1.4e-10, 'g': 0.9e-10, 'r': 1.2e-10, 'i': 1.8e-10, 'z': 7.4e-10, 'W1': 8.1787e-12, 'W2': 2.415e-12,
             'W3': 6.5151e-14, 'W4': 5.0901e-15}
        self.zp = b[self.name]

    def plot_filter(self, ax, scale=1):
        ax.plot(np.log10(self.data[0]), np.log10(self.data[1] * scale))


class spectrum():
    """
    This class is for working with SED, including units conversion and calculating the photometric filters fluxes
    """

    def __init__(self, x=None, y=None, template=None, x_unit=u.AA, y_unit=u.erg / u.cm ** 2 / u.s / u.AA, z=0):
        """
        parameters:
            - x        :  energy units of flux, e.g. wavelengths or frequencies
            - y        :  flux
            - x_unit   :  units of x
            - y_unit   :  units of y
            - z        :  redshift  (needed to estimate the luminosity)
        """
        self.x_unit = x_unit
        self.y_unit = y_unit
        if x != None:
            self.x = x * self.x_unit
        if y != None:
            self.flux = y * self.y_unit
        self.z = z
        self.load_template(template)
        self.load_filters()

    def load_template(self, template=None):
        if template != None:
            sp = np.genfromtxt(template, unpack=True, comments='#')
            self.x = sp[0] * (1 + self.z) * self.x_unit
            self.flux = sp[1] * self.y_unit

    def load_filters(self, fil=['u', 'g', 'r', 'i', 'z', 'W1', 'W2', 'W3', 'W4']):
        """
        loading filters data
        """
        self.filter = {}
        for f in fil:
            self.filter[f] = photo_filter(f)

    def convert(self, unit):
        """
        convert to specified units, e.g. 'A' -> 'Hz', 'erg/cm^2/s/A' -> 'Jy'
        """
        if unit.decompose().bases in [[u.m], [u.s]]:
            if unit.decompose().bases == self.x.unit.decompose().bases:
                return self.x.to(unit)
            else:
                return self.x.to(unit, equivalencies=u.spectral())
        if unit.decompose().bases in [[u.kg, u.m, u.s], [u.kg, u.s]]:
            if unit.decompose().bases == self.flux.unit.decompose().bases:
                return self.flux.to(unit)
            else:
                return self.flux.to(unit, equivalencies=u.spectral_density(self.x))

    def val(self, unit):
        """
        return the value of the spectrum quantities, can correspond either to x or y units.
        """
        return self.convert(unit=unit).value

    def add_ext(self, z_ext=0, Av=0, kind='SMC'):
        """
        add extinction with specified Av at given redshift
        parameters:
            - z_ext     :  redshift of extinction applied
            - Av        :  Av
            - kind      :  type of extinction curve, can be either 'SMC', 'LMC'
        """
        if kind in ['SMC', 'LMC']:
            et = {'SMC': 2, 'LMC': 6}
            data = np.genfromtxt('C:/science/python/spectro/sviewer/data/extinction.dat', skip_header=3,
                                 usecols=[0, et[kind]], unpack=True)
            inter = interp1d(data[0] * 1e4, data[1], bounds_error=False, fill_value=(7, 0))  # 'extrapolate',)
        self.flux *= np.exp(- 0.92 * Av * inter(self.convert(u.AA).value / (1 + z_ext)))

    def photo(self, f, scale=1.0, units=u.erg / u.cm ** 2 / u.s / u.AA):
        """
        calculate photometric flux at given filter
        paremeters:
            - f          :  filter name
            - scale      :  scaling of the filter
            - units      :  units of the fluxes to be returned
        return: l, flux
            - l          :  effective wavelenght in A
            - flux       :  flux at given filter
        """
        x = self.convert(u.AA)
        fl = np.trapz(self.filter[f].inter(x) * x * self.flux * scale, x=x) / np.trapz(self.filter[f].inter(x) * x, x=x)
        return fl.to(units, equivalencies=u.spectral_density(self.filter[f].l_eff))

    def mag(self, f):
        """
        Calculate the magnitude at specifed photometric filter.
        """
        mask = (self.convert(u.AA).value > self.filter[f].data[0][0]) * (
                    self.convert(u.AA).value < self.filter[f].data[0][-1])
        x = self.convert(u.AA)[mask].value
        flux = np.trapz(self.convert(u.erg / u.cm ** 2 / u.s / u.AA)[mask] * self.filter[f].inter(x), x=x) / np.trapz(
            self.filter[f].inter(x), x=x)

        if f in 'ugriz':
            return -2.5 / np.log(10) * (np.arcsinh(
                flux.to('Jy', equivalencies=u.spectral_density(self.filter[f].l_eff)) / u.AB.to('Jy') / u.Jy / 2 /
                self.filter[f].zp).value + np.log(self.filter[f].zp))

        elif f in 'W1W2W3W4':
            return -2.5 * np.log10(flux.value / self.filter[f].zp)

    def match_filter(self, f, mag=0):
        """
        find the scaling of the flux based on matching the photometric filter value
        """
        mask = (self.convert(u.AA).value > self.filter[f].data[0][0]) * (
                    self.convert(u.AA).value < self.filter[f].data[0][-1])
        x = self.convert(u.AA)[mask].value
        flux = np.trapz(self.convert(u.erg / u.cm ** 2 / u.s / u.AA)[mask] * self.filter[f].inter(x), x=x) / np.trapz(
            self.filter[f].inter(x), x=x)

        if f in 'ugriz':
            scale = u.AB.to('Jy') * u.Jy / flux.to('Jy', equivalencies=u.spectral_density(self.filter[f].l_eff)) * 2 * \
                    self.filter[f].zp * np.sinh(- mag / 2.5 * np.log(10) - np.log(self.filter[f].zp))

        elif f in 'W1W2W3W4':
            scale = 10 ** (- mag / 2.5) * self.filter[f].zp / flux.value

        return scale

    def calc_lum(self, l_min=8 * u.micron, l_max=1e3 * u.micron, solar=False, plot=False):
        """
        calculates the integrated luminosity based on the spectrum.
        parameters:
            - lmin, lmax    :  ranges for the integration
            - solar         :  if True, relative to Solar luminosity
            - plot          :  shoe the plot for specific luminosity
        """
        l_nu = self.convert(u.Jy) * 4 * np.pi * cosmo.luminosity_distance(self.z) ** 2 / (1 + self.z)
        m = (self.convert(u.AA) / (1 + self.z) > l_min) * (self.convert(u.AA) / (1 + self.z) < l_max)
        L_IR = np.trapz(l_nu[m], x=-self.convert(u.Hz)[m] * (1 + self.z))
        if plot:
            fig, ax = plt.subplots()
            ax.plot(np.log10(self.convert(u.Hz).value * (1 + self.z)), np.log10(l_nu.value))
            ax.set_ylabel('$\log F_\nu$\,\, [erg/s/Hz]')
            ax.set_xlabel('$\log \nu$\,\, [Hz]')
        if solar:
            L_IR /= ac.L_sun.to('erg/s')
            return L_IR.decompose()
        else:
            return L_IR.to('erg/s')
