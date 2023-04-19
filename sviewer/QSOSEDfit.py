import astropy.constants as ac
from astropy.cosmology import Planck15 #, FlatLambdaCDM, LambdaCDM
from astropy.io import fits
import astropy.units as u
from bisect import bisect
from collections import OrderedDict
import corner
from chainconsumer import ChainConsumer
import emcee
import json
import itertools
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import lmfit
import os
import pandas as pd
import pickle
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d, CubicSpline
from scipy.special import gamma, gammainc
import sys
import zeus


if __name__ in ["__main__", "__mp_main__"]:
    from a_unc import a
    from stats import distr1d
else:
    from ..a_unc import a
    from ..stats import distr1d, distr2d

# subclass JSONEncoder
class resEncoder(json.JSONEncoder):
    def default(self, o):
        return {k:o.__dict__[k] for k in ('val', 'plus', 'minus', 'type') if k in o.__dict__}

def smooth(x, window_len=11, window='hanning', mode='valid'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode=mode)[window_len-1:-window_len+1]
    return y

def add_LyaForest(x, z_em=0, factor=1, kind='trans'):
    """
    add absorptions by Lya forest, taking into account its redshift dependence
    parameters:
        - x         : the wavelength grid
        - z_em      : emission redshift of object, if == 0 then Nothing to add
        - factor    : scaling factor of Lya density
        - kind      : type of Lya forest consideration, can be:
                         - 'trans'   : add as transmitted flux
                         - 'lines'   : add as individual lines, randomly drawn
    return:
        - corr      : correction array at each x.
    """
    corr = np.ones_like(x)
    if kind == 'trans':
        trans = np.array([[3360, 3580, 3773, 4089, 4493, 4866, 5357, 6804, 7417, 7700],
                          [1.0, 0.931, 0.879, 0.823, 0.742, 0.663, 0.547, 0.203, 0.071, 0]])
        inter = interp1d(trans[0], trans[1], bounds_error=False, fill_value=(1, 0))
        mask = x < (z_em - 0.05 + 1) * 1215.67
        corr[mask] = inter(x[mask])

    return corr

class gline():
    """
    class saving filter data
    """
    def __init__(self, x=[], y=[], err=[], mask=[]):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.err = np.asarray(err)
        self.mask = np.asarray(mask)
        # self.x_s, self.y_s, self.err_s = self.x, self.y, self.err
        self.n = self.x.shape[0]

class Filter():
    def __init__(self, parent, name, value=None, flux=None, err=None, system='AB'):
        self.parent = parent
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.system = system
        self.correct_name()
        self.init_data()

        if value is not None:
            self.value, self.err = value, err
        elif flux is not None:
            self.value, self.err = self.get_value(flux=flux), self.get_value(flux+err) - self.get_value(flux)

        #print(name, self.value, self.err)
        if value is not None and err is not None:
            self.flux = self.get_flux(value) * 1e17
            self.err_flux = [(self.get_flux(value) - self.get_flux(value + err)) * 1e17, (self.get_flux(value - err) - self.get_flux(value)) * 1e17]
        self.x = self.data.x[:]
        self.calc_weight()

    def correct_name(self):
        d = {'K_VISTA': 'Ks_VISTA', 'Y': 'Y_VISTA', 'J': 'J_2MASS', 'H': 'H_2MASS', 'K': 'Ks_2MASS', 'Ks': 'Ks_2MASS', 'K_2MASS': 'Ks_2MASS'}
        if self.name in d.keys():
            self.name = d[self.name]

    def init_data(self):
        if self.name == 'u':
            self.m0 = 24.63
            self.b = 1.4e-10
        if self.name == 'g':
            self.m0 = 25.11
            self.b = 0.9e-10
        if self.name == 'r':
            self.m0 = 24.80
            self.b = 1.2e-10
        if self.name == 'i':
            self.m0 = 24.36
            self.b = 1.8e-10
        if self.name == 'z':
            self.m0 = 22.83
            self.b = 7.4e-10
        if self.name == 'NUV':
            self.m0 = 28.3
        if self.name == 'FUV':
            self.m0 = 28.3

        zp_vega = {'u': 3.75e-9, 'g': 5.45e-9, 'r': 2.5e-9, 'i': 1.39e-9, 'z': 8.39e-10,
                   'G': 2.5e-9, 'G_BP': 4.04e-9, 'G_RP': 1.29e-9,
                   'J_2MASS': 3.13e-10, 'H_2MASS': 1.13e-10, 'Ks_2MASS': 4.28e-11,
                   'Y_VISTA': 6.01e-10, 'J_VISTA': 2.98e-10, 'H_VISTA': 1.15e-10, 'Ks_VISTA': 4.41e-11,
                   'Z_UKIDSS': 8.71e-10, 'Y_UKIDSS': 5.81e-10, 'J_UKIDSS': 3.0e-10, 'H_UKIDSS': 1.17e-10, 'K_UKIDSS': 3.99e-11,
                   'W1': 8.18e-12, 'W2': 2.42e-12, 'W3': 6.52e-14, 'W4': 5.09e-15,
                   'NUV': 4.45e-9, 'FUV': 6.51e-9,
                }
        zp_ab = {'u': 8.36e-9, 'g': 4.99e-9, 'r': 2.89e-9, 'i': 1.96e-9, 'z': 1.37e-09,
                 'G': 3.19e-9, 'G_BP': 4.32e-9, 'G_RP': 1.88e-9,
                 'J_2MASS': 7.21e-10, 'H_2MASS': 4.05e-10, 'Ks_2MASS': 2.35e-10,
                 'Y_VISTA': 1.05e-9, 'J_VISTA': 6.98e-10, 'H_VISTA': 4.07e-10, 'Ks_VISTA': 2.37e-10,
                 'Z_UKIDSS': 1.39e-9, 'Y_UKIDSS': 1.026e-9, 'J_UKIDSS': 7.0e-10, 'H_UKIDSS': 4.11e-10, 'K_UKIDSS': 2.26e-10,
                 'W1': 9.9e-11, 'W2': 5.22e-11, 'W3': 9.36e-12, 'W4': 2.27e-12,
                 'NUV': 2.05e-8, 'FUV': 4.54e-8,
                }
        self.zp = {'Vega': zp_vega[self.name], 'AB': zp_ab[self.name]}

        colors = {'u': (23, 190, 207), 'g': (44, 160, 44), 'r': (214, 39, 40), 'i': (227, 119, 194), 'z': (31, 119, 180),
                  'G': (225, 168, 18), 'G_BP': (0, 123, 167), 'G_RP': (227, 66, 52),
                  'J_2MASS': (152, 255, 152), 'H_2MASS': (8, 232, 222), 'Ks_2MASS': (30, 144, 255),
                  'Y_VISTA': (212, 245, 70), 'J_VISTA': (142, 245, 142), 'H_VISTA': (18, 222, 212), 'Ks_VISTA': (20, 134, 245),
                  'Z_UKIDSS': (235, 255, 50), 'Y_UKIDSS': (202, 235, 80), 'J_UKIDSS': (132, 235, 132), 'H_UKIDSS': (18, 212, 222), 'K_UKIDSS': (10, 124, 255),
                  'W1': (231, 226, 83), 'W2': (225, 117, 24), 'W3': (227, 66, 52), 'W4': (199, 21, 133),
                  'NUV': (227, 66, 52), 'FUV': (0, 123, 167),
                  }
        self.color = colors[self.name]

        if self.name in ['u', 'g', 'r', 'i', 'z']:
            self.mag_type = 'Asinh'
        if self.name in ['NUV', 'FUV', 'G', 'G_BP', 'G_RP', 'J_2MASS', 'H_2MASS', 'Ks_2MASS',
                         'Z_UKIDSS', 'Y_UKIDSS', 'J_UKIDSS', 'H_UKIDSS', 'K_UKIDSS',
                         'Y_VISTA', 'J_VISTA', 'H_VISTA', 'Ks_VISTA', 'W1', 'W2', 'W3', 'W4']:
            self.mag_type = 'Pogson'

        self.data = None
        self.read_data()

    def read_data(self):
        if self.name in ['u', 'g', 'r', 'i', 'z']:
            data = np.genfromtxt(self.path + r'/data/Filters/' + self.name + '.dat',
                                 skip_header=6, usecols=(0, 1), unpack=True)
            self.data = gline(x=data[0], y=data[1])
        if self.name in ['G', 'G_BP', 'G_RP']:
            data = np.genfromtxt(self.path + r'/data/Filters/GaiaDR2_Passbands.dat',
                                 skip_header=0, usecols=(0, {'G': 1, 'G_BP': 3, 'G_RP': 5}[self.name]), unpack=True)
            self.data = gline(x=data[0][data[1] < 1] * 10, y=data[1][data[1] < 1])
        if self.name in ['J_2MASS', 'H_2MASS', 'Ks_2MASS']:
            data = np.genfromtxt(self.path + f'/data/Filters/2MASS_2MASS.{self.name}.dat'.replace('_2MASS.dat', '.dat'),
                skip_header=0, usecols=(0, 1), unpack=True)
            self.data = gline(x=data[0], y=data[1])
        if self.name in ['Y_VISTA', 'J_VISTA', 'H_VISTA', 'Ks_VISTA', 'K_VISTA']:
            data = np.genfromtxt(self.path + f'/data/Filters/Paranal_VISTA.{self.name}.dat'.replace('_VISTA.dat', '.dat'),
                skip_header=0, usecols=(0, 1), unpack=True)
            self.data = gline(x=data[0], y=data[1])
        if self.name in ['Z_UKIDSS', 'Y_UKIDSS', 'J_UKIDSS', 'H_UKIDSS', 'K_UKIDSS']:
            data = np.genfromtxt(self.path + f'/data/Filters/UKIRT_UKIDSS.{self.name}.dat'.replace('_UKIDSS.dat', '.dat'),
                skip_header=0, usecols=(0, 1), unpack=True)
            self.data = gline(x=data[0], y=data[1])
        if self.name in ['W1', 'W2', 'W3', 'W4']:
            data = np.genfromtxt(self.path + f'/data/Filters/WISE_WISE.{self.name}.dat',
                                 skip_header=0, usecols=(0, 1), unpack=True)
            self.data = gline(x=data[0], y=data[1])
        if self.name in ['NUV', 'FUV']:
            data = np.genfromtxt(self.path + f'/data/Filters/GALEX_GALEX.{self.name}.dat',
                                 skip_header=0, usecols=(0, 1), unpack=True)
            self.data = gline(x=data[0], y=data[1])

        #self.flux_0 = np.trapz(3.631e-29 * ac.c.to('Angstrom/s').value / self.data.x**2 * self.data.y, x=self.data.x)
        #self.flux_0 = np.trapz(3.631 * 3e-18 / self.data.x * self.data.y, x=self.data.x)

        self.flux_0 = 3.631e-20 # in erg/s/cm^2/Hz. This is standart calibration flux in maggies
        self.norm = np.trapz(self.data.x * self.data.y, x=self.data.x)
        self.ymax_pos = np.argmax(self.data.y)
        self.inter = interp1d(self.data.x, self.data.y, bounds_error=False, fill_value=0, assume_sorted=True)
        self.l_eff = np.sqrt(np.trapz(self.inter(self.data.x) * self.data.x, x=self.data.x) / np.trapz(self.inter(self.data.x) / self.data.x, x=self.data.x))
        z = cumtrapz(self.data.y[:], self.data.x)
        self.range = [self.data.x[np.argmin(np.abs(z - np.quantile(z, 0.05)))], self.data.x[np.argmin(np.abs(z - np.quantile(z, 0.95)))]]

        #print(self.name, self.l_eff)

    def get_value(self, x=None, y=None, flux=None, err=None, mask=None, system=None):
        """
        return magnitude in photometric filter. Important that flux should be in erg/s/cm^2/A
        Args:
            x:
            y:
            flux:

        Returns:

        """
        if system == None:
            system = self.system

        try:
            #print(self.name, self.mag_type)
            if self.mag_type == 'Asinh':
                #m0 = -2.5 / np.log(10) * (np.log(self.b))
                if x is None or y is None:
                    x, y = self.parent.s[self.parent.s.ind].spec.x(), self.parent.s[self.parent.s.ind].spec.y()
                if mask is None and x is not None:
                    mask = np.ones_like(x)
                mask = np.logical_and(mask, np.logical_and(x > self.data.x[0], x < self.data.x[-1]))
                if np.sum(mask) > 10:
                    x, y = x[mask], y[mask]
                    interx = x * self.inter(x)
                    flux = np.trapz(y * 1e-17 * interx , x=x) / np.trapz(interx, x=x) * self.l_eff ** 2 / ac.c.to('Angstrom/s').value
                    if self.name in ['u', 'g', 'r', 'i', 'z']:
                        value = - 2.5 / np.log(10) * np.arcsinh(flux / self.flux_0 / 2 / self.b) + self.m0
                    elif self.name in ['NUV', 'FUV']:
                        value = - 2.5 * np.log10(np.exp(1)) * np.arcsinh(flux / self.flux_0 / 1e-9 / 0.01) + self.m0
                else:
                    value = np.nan

            elif self.mag_type == 'Pogson':
                if flux is None:
                    if x is None or y is None:
                        x, y = self.parent.s[self.parent.s.ind].spec.x(), self.parent.s[self.parent.s.ind].spec.y()
                    mask = np.logical_and(x > self.data.x[0], x < self.data.x[-1])
                    x, y = x[mask], y[mask]
                    y[y < 0] = 0
                    interx = x * self.inter(x)
                    flux = np.trapz(y * 1e-17 * interx, x=x) / np.trapz(interx, x=x)
                value = - 2.5 * np.log10(flux / self.zp[system])
                #print(self.value)
        except:
            value = np.nan
        return value

    def get_flux(self, value, system=None):
        #print(self.name, self.mag_type)
        if system == None:
            system = self.system
        if self.mag_type == 'Asinh':
            if self.name in ['u', 'g', 'r', 'i', 'z']:
                flux = self.flux_0 * ac.c.to('Angstrom/s').value / self.l_eff ** 2 * 10 ** (-value / 2.5) * (1 - (10 ** (value / 2.5) * self.b) ** 2)
            elif self.name in ['NUV', 'FUV']:
                flux = self.flux_0 * 1e-9 * 0.01 * ac.c.to('Angstrom/s').value / self.l_eff ** 2 * np.sinh(-self.m0 - value / 2.5 / np.log10(np.exp(1)))
        elif self.mag_type == 'Pogson':
            flux = self.zp[system] * 10 ** (- value / 2.5)
        return flux

    def calc_weight(self, z=0):
        num = int((np.log(self.range[1]) - np.log(self.range[0])) / 0.001)
        x = np.exp(np.linspace(np.log(self.x[0]), np.log(self.x[-1]), num))
        if 'W' in self.name:
            self.weight = 1
        elif self.name in ['FUV', 'NUV']:
            self.weight = 20
        else:
            self.weight = 5 #np.sqrt(np.sum(self.filter.inter(x)) / np.max(self.filter.data.y))
        #self.weight = 1
        #print('weight:', self.name, self.weight)

class sed_template():
    def __init__(self, parent, name, x=None, y=None):
        self.parent = parent
        self.name = name
        self.load_data(smooth_window=self.parent.smooth_window, xmin=self.parent.xmin, xmax=self.parent.xmax, z=self.parent.z, x=x, y=y)

    def flux(self, x=None):
        if x is not None:
            return self.inter(x)
        else:
            return self.y

    def load_data(self, smooth_window=None, xmin=None, xmax=None, z=0, x=None, y=None):

        if x is None and y is None:
            if self.name in ['VandenBerk', 'HST', 'Slesing', 'power', 'composite']:
                self.type = 'qso'

                if self.name == 'VandenBerk':
                    self.x, self.y = np.genfromtxt(self.parent.path + r'/data/SDSS/medianQSO.dat', skip_header=2, unpack=True)
                elif self.name == 'HST':
                    self.x, self.y = np.genfromtxt(self.parent.path + r'/data/SDSS/hst_composite.dat', skip_header=2, unpack=True)
                elif self.name == 'Slesing':
                    self.x, self.y = np.genfromtxt(self.parent.path + r'/data/SDSS/Slesing2016.dat', skip_header=0, unpack=True, usecols=(0, 1))
                elif self.name == 'power':
                    self.x = np.linspace(500, 25000, 1000)
                    self.y = np.power(self.x / 2500, -1.9)
                    smooth_window = None
                elif self.name == 'composite':
                    if 1:
                        self.x, self.y = np.genfromtxt(self.parent.path + r'/data/SDSS/QSO_composite.dat', skip_header=0, unpack=True)
                        self.x, self.y = self.x[self.x > 0], self.y[self.x > 0]
                    else:
                        self.template_qso = np.genfromtxt(self.parent.path + r'/data/SDSS/Slesing2016.dat', skip_header=0, unpack=True)
                        self.template_qso = self.template_qso[:,
                                            np.logical_or(self.template_qso[1] != 0, self.template_qso[2] != 0)]
                        if 0:
                            x = self.template_qso[0][-1] + np.arange(1, int((25000 - self.template_qso[0][
                                -1]) / 0.4)) * 0.4
                            y = np.power(x / 2500, -1.9) * 6.542031
                            self.template_qso = np.append(self.template_qso, [x, y, y / 10], axis=1)
                        else:
                            if 1:
                                data = np.genfromtxt(self.parent.path + r'/data/SDSS/QSO1_template_norm.sed', skip_header=0, unpack=True)
                                data[0] = ac.c.cgs.value() / 10 ** data[0] * 1e8
                                #print(data[0])
                                m = (data[0] > self.template_qso[0][-1]) * (data[0] < self.wavemax)
                                self.template_qso = np.append(self.template_qso, [data[0][m],
                                                                                  data[1][m] * self.template_qso[1][
                                                                                      -1] / data[1][m][0],
                                                                                  data[1][m] / 30], axis=1)
                            else:
                                data = np.genfromtxt(self.parent.path + r'/data/SDSS/Richards2006.dat', skip_header=0, unpack=True)
                                m = (data[0] > self.template_qso[0][-1]) * (data[0] < self.wavemax)
                                self.template_qso = np.append(self.template_qso,
                                                              [data[0][m], data[1][m] * self.template_qso[1][-1] /
                                                               data[1][m][0], data[1][m] / 30], axis=1)
                            x = self.template_qso[0][0] + np.linspace(
                                int((self.wavemin - self.template_qso[0][0]) / 0.4), -1) * 0.4
                            y = np.power(x / self.template_qso[0][0], -1.0) * self.template_qso[1][0]
                            self.template_qso = np.append([x, y, y / 10], self.template_qso, axis=1)

            elif self.name in ['S0', 'Sa', 'Sb', 'Sc', 'Sd', 'Sdm', 'Ell2', 'Ell5', 'Ell13', 'Sey2', 'Sey18']:
                if 0:
                    if isinstance(self.template_name_gal, int):
                        f = fits.open(self.parent.path + f"/data/SDSS/spDR2-0{23 + self.template_name_gal}.fit")
                        self.template_gal = [
                            10 ** (f[0].header['COEFF0'] + f[0].header['COEFF1'] * np.arange(f[0].header['NAXIS1'])),
                            f[0].data[0]]

                        if smooth_window is not None:
                            self.template_gal[1] = smooth(self.template_gal[1], window_len=smooth_window,
                                                          window='hanning', mode='same')
                self.x, self.y = np.genfromtxt(self.parent.path + f'/data/SDSS/{self.name}_template_norm.sed', unpack=True)

                        # mask = (self.template_gal[0] > self.wavemin) * (self.template_gal[0] < self.wavemax)
                        # self.template_gal = [self.template_gal[0][mask], self.template_gal[1][mask]]

            elif self.name in ['torus']:
                self.x, self.y = np.genfromtxt(self.parent.path + r'/data/SDSS/torus.dat', unpack=True)
                self.x, self.y = ac.c.cgs.value / 10 ** self.x[::-1] * 1e8, self.y[::-1]

            elif self.name == 'Fe':
                self.x, self.y = np.genfromtxt(self.parent.path + r'/data/Models/Fe.dat', unpack=True)
                self.norm = np.mean(self.y[(self.x > 2490) & (self.x < 2510)])
                print('norm:', self.norm)
        else:
            self.x, self.y = x, y

        if xmin is not None or xmax is not None:
            mask = np.ones_like(self.x, dtype=bool)
            if xmin is not None:
                mask *= self.x > xmin
            if xmax is not None:
                mask *= self.x < xmax

            self.x, self.y = self.x[mask], self.y[mask]

        if z > 0:
            self.y *= add_LyaForest(self.x * (1 + z), z_em=z)

        if smooth_window is not None:
            self.y = smooth(self.y, window_len=smooth_window, window='hanning', mode='same')

        self.inter = interp1d(self.x, self.y, bounds_error=False, fill_value=0, assume_sorted=True)

class sed():
    def __init__(self, name, smooth_window=None, xmin=None, xmax=None, z=None):
        self.path = os.path.dirname(os.path.realpath(__file__)) 
        self.name = name
        self.smooth_window, self.xmin, self.xmax, self.z = smooth_window, xmin, xmax, z
        self.load_data()
        self.data = {}

    def load_data(self):
        self.models = []
        if self.name in ['bbb']:
            self.models.append(sed_template(self, 'composite'))
            self.values = [0]

        if self.name in ['tor']:
            torus = np.load(self.path + r'/data/Models/TORUS/silva_v1_files.npz')
            for i in range(len(torus['arr_0'])):
                self.models.append(sed_template(self, self.name,
                                                x=ac.c.cgs.value / 10 ** torus['arr_1'][i][::-1] * 1e8,
                                                y=(torus['arr_2'][i] * 10 ** (2 * torus['arr_1'][i]) * 1e41)[::-1]
                                                ))
            self.values = torus['arr_0']

        if self.name in ['host']:
            self.values = ['S0', 'Sa', 'Sb', 'Sc', 'Ell2', 'Ell5', 'Ell13'] #['S0', 'Sa', 'Sb', 'Sc', 'Sd', 'Sdm', 'Ell2', 'Ell5', 'Ell13', 'Sey2', 'Sey18']
            for n in self.values:
                self.models.append(sed_template(self, n))

        if self.name in ['gal']:
            self.values = []
            with open(self.path +  r'/data/Models/bc03_275templates.pickle', 'rb') as f:
                l = pickle.load(f)
                self.tau = pickle.load(f)
                self.tg = pickle.load(f)
                SED = pickle.load(f)
                #print(self.tau, self.tg)
                #print(tau.shape, tg.shape, SED.shape)
                for i in range(len(self.tau)):
                    for k in range(len(self.tg)):
                        self.values.append([self.tau[i], self.tg[k]])
                        self.models.append(sed_template(self, 'gal', x=l.value, y=SED[k, i, :].value * 1e5))
            self.n_tau = self.tau.shape[0]
            self.n_tg = self.tg.shape[0]

        elif self.name == 'Fe':
            self.models.append(sed_template(self, 'Fe'))
            self.values = [0]

        self.n = len(self.values)
        if self.n > 1:
            self.vary = 1

        #print(self.values)
        #print(self.models)
        #self.min, self.max = 1, len(torus['arr_0'])

    def set_data(self, kind, x):
        self.data[kind] = []
        for m in self.models:
            self.data[kind].append(m.flux(x / (1 + self.z)))

    def get_model_ind(self, params):
        if self.name == 'gal':
            return np.argmin(np.abs((params['host_tau'].value - self.tau.value))) * self.n_tg + np.argmin(np.abs((params['host_tg'].value - np.log10(self.tg.value))))

        if self.name == 'tor':
            return np.argmin(np.abs((params['tor_type'].value - np.arange(len(self.values)))))

class QSOSEDfit():
    def __init__(self, catalog='', plot=1, save=1, mcmc_steps=1000, anneal_steps=100, corr=10, verbose=1):
        """
        Args:
            catalog:        path to the file containing the Erosita catalog
            plot:           plot sed
            save:           save plot figures to <path>/QS/plots
            mcmc_steps:     number of steps in mcmc
            anneal_steps:    number of steps in simulated annealing
            corr:           the limit of the correlation time
            verbose:        verbose output
        """
        self.catalog = catalog
        self.plot = plot
        self.save = save
        self.mcmc_steps = mcmc_steps
        self.anneal_steps = anneal_steps
        self.corr = corr
        self.verbose = verbose
        self.initData()
        self.loadTable()
        #self.addSDSSQSO()
        #self.addCustom()

    def initData(self):
        self.addPhoto = 1
        self.hostExt = 1
        self.df = None
        self.ind = None
        self.corr_status = 0
        self.ext = {}
        self.filters = {}
        self.fig = None
        self.iters = 100

        self.template_name_qso = 'composite'
        self.filter_names = "FUV NUV u g r i z Y J H K W1 W2 W3 W4"
        self.wavemin = 310
        self.wavemax = 3e5
        # self.template_name_gal = "Sb_template_norm" #"Sey2_template_norm"
        # self.host_templates = ['S0', 'Sa', 'Sb', 'Sc', 'Ell2', 'Ell5', 'Ell13']

    def loadTable(self, recalc=False):
        self.df = pd.read_csv(self.catalog)
        try:
            self.df.rename(columns={'Z': 'z', 'ML_FLUX_0_ero': 'F_X_int'}, inplace=True)
        except:
            pass

    def prepare(self, ind):
        self.d = self.df.loc[ind]
        #print(self.df.loc[ind, 'z'])
        self.spec = self.loadSDSS(self.df.loc[ind, 'PLATE'], self.df.loc[ind, 'FIBERID'], self.df.loc[ind, 'MJD'],
                             Av_gal=self.df.loc[ind, 'Av_gal'])
        self.mask = self.calc_mask(self.spec, self.df.loc[ind, 'z'])

        #print(self.mask, np.sum(self.mask))
        if np.sum(self.mask) > 20:
            self.set_filters(ind)
            if self.plot:
                self.plot_spec()

            self.sm = [np.asarray(self.spec[0][self.mask], dtype=np.float64), self.spec[1][self.mask], self.spec[2][self.mask]]

            self.models = {}
            for name in ['bbb', 'tor', 'host', 'gal', 'Fe']:
                self.models[name] = sed(name=name, xmin=self.wavemin, xmax=self.wavemax, z=self.df.loc[ind, 'z'])
                self.models[name].set_data('spec', self.sm[0])
                for k, f in self.filters.items():
                    if self.filters[k].fit:
                        self.models[name].set_data(k, f.x)
                self.models[name].set_data('spec_full', self.spec[0])
            return True

    def set_filters(self, ind, names=None):
        self.photo = {}
        self.filters = {}
        if self.addPhoto:
            for k in ['FUV', 'NUV']:
                if (names is None and k in self.filter_names) or (names is not None and k in names):
                    if 0:
                        if not np.isnan(self.df[f'{k}mag'][ind]) and not np.isnan(self.df[f'e_{k}mag'][ind]):
                            self.filters[k] = Filter(self, k, value=self.df.loc[ind, f'{k}mag'], err=self.df.loc[ind, f'e_{k}mag'])
                    else:
                        if self.df.loc[ind, 'GALEX_MATCHED'] and not np.isnan(self.df.loc[ind, f'{k}mag']) and not np.isnan(self.df.loc[ind, f'e_{k}mag']):
                            f = Filter(self, k, value=-2.5 * np.log10(np.exp(1.0)) * np.arcsinh(self.df.loc[ind, f'{k}'] / 0.01) + 28.3,
                                                     err=-2.5 * np.log10(np.exp(1.0)) * (np.arcsinh(self.df.loc[ind, f'{k}'] / 0.01) - np.arcsinh((self.df.loc[ind, f'{k}'] + 1 / np.sqrt(self.df.loc[ind, f'{k}_IVAR'])) / 0.01)))

                            if f.range[0] > 912 * (self.df.loc[ind, 'z'] + 1):
                                self.filters[k] = f
                                self.photo[k] = 'GALEX'

            for i, k in enumerate(['u', 'g', 'r', 'i', 'z']):
                if (names is None and k in self.filter_names) or (names is not None and k in names):
                    if self.df.loc[ind, f'PSFMAG{i}'] != 0 and self.df.loc[ind, f'ERR_PSFMAG{i}'] != 0:
                        f = Filter(self, k, value=self.df.loc[ind, f'PSFMAG{i}'], err=self.df.loc[ind, f'ERR_PSFMAG{i}'])
                        if f.range[0] > 912 * (self.df.loc[ind, 'z'] + 1):
                            self.filters[k] = f
                            self.photo[k] = 'SDSS'

            for k in ['J', 'H', 'K']:
                if (names is None and k in self.filter_names) or (names is not None and k in names):
                    if self.df.loc[ind, k + 'RDFLAG'] == 2:
                        self.filters[k + '_2MASS'] = Filter(self, k + '_2MASS', value=self.df.loc[ind, k + 'MAG'], err=self.df.loc[ind, 'ERR_' + k + 'MAG'], system='Vega')
                        #self.photo['2MASS'] = self.photo['2MASS'] + [k] if '2MASS' in self.photo.keys() else [k]
                        self.photo[k+'_2MASS'] = '2MASS'

            for k in ['Y', 'J', 'H', 'K']:
                if (names is None and k in self.filter_names) or (names is not None and k in names):
                    if self.df['UKIDSS_MATCHED'][ind]:
                        self.filters[k + '_UKIDSS'] = Filter(self, k + '_UKIDSS', system='AB',
                                                             value=22.5 - 2.5 * np.log10(self.df.loc[ind, k + 'FLUX'] / 3.631e-32),
                                                             err=2.5 * np.log10(1 + self.df.loc[ind, k + 'FLUX_ERR'] / self.df.loc[ind, k + 'FLUX']))
                        #self.photo['UKIDSS'] = self.photo['UKIDSS'] + [k] if 'UKIDSS' in self.photo.keys() else [k]
                        self.photo[k+'_UKIDSS'] = 'UKIDSS'

            for k in ['W1', 'W2', 'W3', 'W4']:
                if (names is None and k in self.filter_names) or k in names:
                    if not np.isnan(self.df.loc[ind, k + 'MAG']) and not np.isnan(self.df.loc[ind, 'ERR_' + k + 'MAG']):
                        self.filters[k] = Filter(self, k, system='Vega',
                                                 value=self.df.loc[ind, k + 'MAG'], err=self.df.loc[ind, 'ERR_' + k + 'MAG'])
                        #self.photo['WISE'] = self.photo['WISE'] + [k] if 'WISE' in self.photo.keys() else [k]
                        self.photo[k] = 'WISE'

        for k, f in self.filters.items():
            self.filters[k].fit = False
            if f.range[0] > self.wavemin * (1 + self.df.loc[ind, 'z']) and f.range[1] < self.wavemax * (1 + self.df.loc[ind, 'z']) and np.isfinite(f.value) and np.isfinite(f.err):
                self.filters[k].fit = True
            #print(k, self.filters[k].fit)
        print(self.filters)

    def add_mask(self, name=None):
        if name is not None:
            self.mask = np.logical_or(self.mask, self.df['SDSS_NAME'] == name)
            #print(np.sum(self.mask))

    def ext_fm07(self, wave, Av, Rv=2.74, c1=-4.959, c2=2.264, c3=0.389, c4=0.319, c5=6.097, x0=4.592, gamma=0.922):
        """
        Return extinction for provided wavelengths
        Args:
            wave:      wavelengths in Angstrem
            Av:        visual extinction
            ...

        Returns: extinction
        """
        x = (1e4 / np.asarray(wave, dtype=np.float64))

        k = np.zeros_like(x)
        uv = (x >= 1.e4 / 2700.)

        # UV region
        def D(x, x0, gamma):
            return x ** 2 / ((x ** 2 - x0 ** 2) ** 2 + x ** 2 * gamma ** 2)

        k[uv] = c1 + c2 * x[uv] + c3 * D(x[uv], x0, gamma)
        k[uv * (x > c5)] += c4 * (x[uv * (x > c5)] - c5) ** 2

        # Anchors
        x_uv = 1.e4 / np.array([2700., 2600.])
        k_uv = c1 + c2 * x_uv + c3 * D(x_uv, x0, gamma)
        x_opt = 1.e4 / np.array([5530., 4000., 3300.])
        k_opt = np.array([0., 1.322, 2.055])
        x_ir = np.array([0., 0.25, 0.50, 0.75, 1.])
        k_ir = (-0.83 + 0.63 * Rv) * x_ir ** 1.84 - Rv

        k[~uv] = CubicSpline(np.concatenate(([x_ir, x_opt, x_uv])), np.concatenate((k_ir, k_opt, k_uv)))(x[~uv])
        #k[~uv] = interp1d(np.concatenate(([x_ir, x_opt, x_uv])), np.concatenate((k_ir, k_opt, k_uv)), kind='cubic', assume_sorted=True)(x[~uv])

        return Av * (k / Rv + 1)

    def extinction_Pervot(self, x, params, z_ext=0):
        return 10 ** (-0.4 * params.valuesdict()['Av'] * (1.39 * (1e4 / np.asarray(x, dtype=np.float64) * (1 + z_ext)) ** 1.2 - 0.38) / 2.74)

    def extinction_MW(self, x, Av=0):
        #return 10 ** (-0.4 * extinction.Fitzpatrick99(3.1 * 1.0)(np.asarray(x, dtype=np.float64), Av))
        #return 10 ** (-0.4 * Av * (1.39 * (1e4 / np.asarray(x, dtype=np.float64)) ** 1.2 - 0.38) / 2.74)
        return 10 ** (-0.4 * self.ext_fm07(x, Av=Av, Rv=3.01, c1=-0.175, c2=0.807, c3=2.991, c4=0.319, c5=6.097, x0=4.592, gamma=0.922))

    def extinction(self, wave, params, z_ext=0):
        """
        Return extinction for provided wavelengths
        Args:
            wave:         wavelengths in Angstrem
            z_ext:     redshift
            Av:        visual extinction

        Returns: extinction
        """
        Rv = 2.71 if 'Rv' not in params.keys() else params.valuesdict()['Rv']
        #c3 = 0.389 if 'c3' not in params.keys() else params.valuesdict()['c3']
        c3 = 0.389 if 'c3' not in params.keys() else params.valuesdict()['Abump'] * 2 * 0.922 / params.valuesdict()['EBV'] / np.pi
        #s = time.time()
        #self.extinction_Pervot(wave, params, z_ext=z_ext)
        #print('t_Pervot:', time.time() - s)
        #s = time.time()
        #10 ** (-0.4 * self.ext_fm07(wave * (1 + z_ext), Av=params.valuesdict()['Av'], Rv=Rv, c1=-4.959, c2=2.264, c3=0.389, c4=0.319, c5=6.097, x0=4.592, gamma=0.922))
        #print('t_FM07:', time.time() - s)
        #return self.extinction_Pervot(wave, params, z_ext=z_ext)
        return 10 ** (-0.4 * self.ext_fm07(wave * (1 + z_ext), Av=params.valuesdict()['EBV'] * Rv, Rv=Rv, c1=-4.959, c2=2.264, c3=c3, c4=0.319, c5=6.097, x0=4.592, gamma=0.922))

    def loadSDSS(self, plate, fiber, mjd, Av_gal=np.nan):
        filename = os.path.dirname(self.catalog) + '/spectra/spec-{0:04d}-{2:05d}-{1:04d}.fits'.format(int(plate), int(fiber), int(mjd))
        #print(filename)
        if os.path.exists(filename):
            qso = fits.open(filename)
            ext = self.extinction_MW(10 ** qso[1].data['loglam'][:], Av=Av_gal)
            mask = ~np.logical_and(np.logical_and(np.isfinite(qso[1].data['flux']), np.isfinite(np.sqrt(1.0 / qso[1].data['ivar']))), (ext != 0))
            #print('mask: ', np.sum(mask))
            #print(Av_gal, ext)
            return [10 ** qso[1].data['loglam'], qso[1].data['flux'] / ext, np.sqrt(1.0 / qso[1].data['ivar']) / ext, np.logical_and(mask, qso[1].data['and_mask'])]

    def spec_model(self, params, x):
        return None

    def model(self, params, x, dtype, mtype='total'):
        model = np.zeros_like(self.models['bbb'].data[dtype][0])
        if mtype in ['total', 'bbb']:
            model = self.models['bbb'].data[dtype][0] * params.valuesdict()['bbb_norm'] * self.extinction(x / (1 + self.d['z']), params)
        if mtype in ['total', 'tor'] and params.valuesdict()['tor_type'] > -1:
            model += self.models['tor'].data[dtype][self.models['tor'].get_model_ind(params)] * params.valuesdict()['tor_norm']
        if mtype in ['total', 'gal'] and params.valuesdict()['host_tau'] > -1 and params.valuesdict()['host_tg'] > -1:
            model += self.models['gal'].data[dtype][self.models['gal'].get_model_ind(params)] * params.valuesdict()['host_norm'] * self.extinction_MW(x / (1 + self.d['z']), Av=params.valuesdict()['host_Av'])
        #if params.valuesdict()['host_type'] > -1:
        #    model += self.models['host'].data[kind][params.valuesdict()['host_type']] * params.valuesdict()['host_norm'] * self.extinction(x / (1 + d['z']), Av=params.valuesdict()['host_Av'])
        return model

    def model_emcee(self, params, x, dtype, mtype='total'):
        for s in ['spec', 'spec_full']:
            self.photo[s] = 'spec'
        alpha = 10 ** params['alpha_' + self.photo[dtype]].value * (x / (1 + self.d['z']) / 2500) ** params['slope_' + self.photo[dtype]].value if self.photo[dtype] not in ['WISE', 'spec'] else 1
        #print(dtype, self.photo[dtype], alpha)
        if 'bbb_slope' in params.keys():
            alpha *= (x / (1 + self.d['z']) / 2500) ** params['bbb_slope'].value
        model = np.zeros_like(self.models['bbb'].data[dtype][0])
        if mtype in ['total', 'bbb']:
            model += self.models['bbb'].data[dtype][0] * alpha * params['bbb_norm'].value
        if mtype in ['total', 'Fe']:
            model += self.models['Fe'].data[dtype][0] * params['Fe_norm'].value * params['bbb_norm'].value
        if mtype in ['total', 'bbb', 'Fe']:
            model *= self.extinction(x / (1 + self.d['z']), params)
        if mtype in ['total', 'tor'] and params['tor_type'].value > -1:
            model += self.models['tor'].data[dtype][self.models['tor'].get_model_ind(params)] * params['tor_norm'].value
        if mtype in ['total', 'gal'] and params['host_tau'].value > -1 and params['host_tg'].value > -1:
            model += self.models['gal'].data[dtype][self.models['gal'].get_model_ind(params)] * params['host_norm'].value * self.extinction_MW(x / (1 + self.d['z']), Av=params['host_Av'].value)

        #if params.valuesdict()['host_type'] > -1:
        #    model += self.models['host'].data[kind][params.valuesdict()['host_type']] * params.valuesdict()['host_norm'] * self.extinction(x / (1 + d['z']), Av=params.valuesdict()['host_Av'])
        return model

    def calc_host_lum(self, params, wave=1700):
        """
        calculate the luminosity of the host galaxy either at some wavelenght, or bolometric
        Args:
            params: parameters dictionary for the model
            wave:   wavelength at which luminosity is calculated, if 'bol', 'total' calculate bolometric

        Returns: luminosity in [erg/s/AA] (if wave is float) or [erg/s] for bolometric

        """
        if isinstance(wave, (int, float)):
            lum = self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(wave) * \
                         params.valuesdict()['host_norm'] * self.extinction_MW(wave, Av=params.valuesdict()['host_Av']) * \
                         1e-17 * 4 * np.pi * Planck15.luminosity_distance(self.d['z']).to('cm').value ** 2
        elif wave in ['bol', 'total']:
            wave = self.models['gal'].models[self.models['gal'].get_model_ind(params)].x
            lum = np.trapz(self.models['gal'].models[self.models['gal'].get_model_ind(params)].y * params.valuesdict()['host_norm'] * self.extinction_MW(wave, Av=params.valuesdict()['host_Av']),
                         x=self.models['gal'].models[self.models['gal'].get_model_ind(params)].x)
            lum *= 1e-17 * 4 * np.pi * Planck15.luminosity_distance(self.d['z']).to('cm').value ** 2
        return lum

    def fcn2min(self, params):
        chi = (self.model(params, self.sm[0], 'spec') - self.sm[1]) / self.sm[2]
        for k, f in self.filters.items():
            if self.filters[k].fit and any([s in k for s in ['UKIDSS', 'W', '2MASS']]):
                chi = np.append(chi, [f.weight / f.err * (f.value - f.get_value(x=f.x, y=self.model(params, f.x, k)))])
        return chi

    def fcn2min_mcmc(self, params):
        #print(params)
        pars = self.set_params(params)
        #print(pars)
        #params['host_L'].value = self.calc_host_lum(params)[0]
        lp = self.ln_priors(pars)
        if not np.isfinite(lp):
            return -np.inf
        #print(lp, np.sum(np.power(self.ln_like(pars), 2)))
        return lp -.5 * np.sum(np.power(self.ln_like(pars), 2))

    def anneal_priors(self, params):
        #print(self.lum_prior(params), self.agn_host_prior(params), params)
        return 100 * self.lum_prior(params, kind='anneal') + 50 * self.agn_host_prior(params)

    def agn_host_prior(self, params):
        host = self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(1400) * params.valuesdict()['host_norm'] * self.extinction_MW(1400, Av=params.valuesdict()['host_Av'])
        agn = self.models['bbb'].models[0].flux(1400) * params.valuesdict()['bbb_norm'] * self.extinction(1400, params)
        #print(host, agn, - host / agn)
        return - host / agn

    def lum_prior(self, params, kind='mcmc'):
        alpha, tmin = -1.2, 0.0001
        C = (alpha + 1) / (gamma(alpha + 2) * (1 - gammainc(alpha + 2, tmin)) - (tmin) ** (alpha + 1) * np.exp(-tmin))
        M_UVs = -20.9 - 1.1 * (self.d['z'] - 1)
        #f = self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(1700) * params.valuesdict()['host_norm'] * self.extinction(1700, Av=params.valuesdict()['host_Av'])
        f = self.calc_host_lum(params, wave=1700)
        #print('f:', f)
        if f == 0 or -2.5 * np.log10(f * 9.64e-13) + 51.6 > M_UVs + 10:
            M_UV = M_UVs + 10
        else:
            M_UV = -2.5 * np.log10(f * 9.64e-13) + 51.6
        #print('M_UV:', M_UV, M_UVs)
        LLs = 10 ** (-0.4 * (M_UV - M_UVs))
        #print(LLs ** (alpha) * np.exp(-LLs))
        #print(C)
        #print(np.log(C * LLs ** (-1.2) * np.exp(-LLs)))
        if kind == 'mcmc':
            return np.log(C * LLs ** (-1.2) * np.exp(-LLs)) if C * LLs ** (-1.2) * np.exp(-LLs) < 1 else 0
        elif kind == 'anneal':
            return -LLs

    def ln_priors(self, params):
        prior = self.lum_prior(params, kind='mcmc')
        #print('lum_prior: ', self.lum_prior(params, kind='mcmc'))
        for p in params.values():
            if p.value <= p.min or p.value >= p.max:
                return -np.inf
            if p.name == 'bbb_slope':
                if p.value < 0:
                    prior -= .5 * ((p.value - 0) / 0.3) ** 2
                else:
                    prior -= .5 * ((p.value - 0) / 0.1) ** 2
            if p.name == 'Fe_norm':
                prior -= .5 * ((p.value - 0) / 0.2) ** 2
            if p.name == 'host_tau':
                prior -= (p.value - p.min) ** 2
            if p.name == 'host_tg':
                prior -= 10 ** (p.max - p.value) - 1
            if p.name == 'host_Av':
                prior -= (p.value - p.min) ** 2
            if p.name == 'Rv':
                prior -= .5 * ((p.value - 2.71) / 0.2) ** 2
            if p.name == 'EBV':
                #prior -= .5 * ((p.value - 0) / 0.05) ** 2
                prior -= 1.5 * np.log10(p.value) + ((p.value - 0) / 0.05) ** 2
            if 'alpha' in p.name:
                prior -= .5 * (p.value / params['sigma'].value) ** 2 + np.log(params['sigma'].value)
            if 'slope' in p.name:
                prior -= .5 * (p.value / 0.3) ** 2
            if p.name == 'sigma':
                prior -= .5 * ((p.value - 0.2) / 0.05) ** 2

            #print(p.name, prior)
        return prior

    def ln_like(self, params, plot=0):
        # print(params)
        # t = Timer()
        chi = (self.model_emcee(params, self.sm[0], 'spec') - self.sm[1]) / self.sm[2]
        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.sm[0], chi)
            fig, ax = plt.subplots()
        #if np.sum(np.isnan(chi)):
        #    print(self.sm[0][np.isnan(chi)])
        # t.time('spec')
        for k, f in self.filters.items():
            if self.filters[k].fit:
                chi = np.append(chi, [f.weight / f.err * (f.value - f.get_value(x=f.x, y=self.model_emcee(params, f.x, k)))])
                if plot:
                    print(k, [f.weight / f.err * (f.value - f.get_value(x=f.x, y=self.model_emcee(params, f.x, k)))])
                    ax.plot(np.mean(f.x), [f.weight / f.err * (f.value - f.get_value(x=f.x, y=self.model_emcee(params, f.x, k)))])
                #if np.isnan(chi[-1]):
                #    print(k)
                # t.time(f.name)
        if plot:
            plt.show()

        return chi

    def set_params(self, x):
        params = self.params
        for p, v in zip(params.keys(), x):
            params[p].value = v
            #print(p, v, params[p].value)
        return params

    def mcmc(self, params=None, tvary=True, dust='Rv', stat=1, method='zeus', nsteps=200):
        # import os
        # os.environ["OMP_NUM_THREADS"] = "1"
        # from multiprocessing import Pool

        #print(self.photo)

        new = True if params is None else False

        if new:
            norm_bbb = np.nanmean(self.sm[1]) / np.nanmean(self.models['bbb'].data['spec'])
            params = lmfit.Parameters()
            params.add('bbb_norm', value=norm_bbb, min=0, max=1e10)
            params.add('bbb_slope', value=0, min=-2, max=2)
            params.add('EBV', value=0.0, min=0.0, max=10)
            params.add('tor_type', value=10, vary=True, min=0, max=self.models['tor'].n - 1)
            params.add('tor_norm', value=norm_bbb / np.max(self.models['bbb'].data['spec'][0]) * np.max(self.models['tor'].data['spec'][params.valuesdict()['tor_type']]), min=0, max=1e10)
            params.add('host_tau', value=0.3, vary=True, min=self.models['gal'].tau[0].value, max=3)
            params.add('host_tg', value=np.log10(Planck15.age(self.d['z']).to('yr').value) - 1, vary=True, min=np.log10(self.models['gal'].tg[0].value), max=np.log10(Planck15.age(self.d['z']).to('yr').value))
            #print('age:', np.log10(Planck15.age(self.d['z']).to('yr').value))
            params.add('host_norm', value=np.nanmean(self.sm[1]) / np.nanmean(self.models['gal'].data['spec'][self.models['gal'].get_model_ind(params)]), min=0, max=1e10)
            params.add('host_Av', value=0.1, min=0, max=5.0)

        if params['host_norm'].value < params['bbb_norm'].value / 1000:
            params['host_norm'].value = params['bbb_norm'].value / 1000

        #print(self.extinction(2500, Av=params['Av'].value), np.log(self.extinction(1000, Av=params['Av'].value) / self.extinction(2500, Av=params['Av'].value)) / np.log(0.4))
        if params['EBV'].value < 0:
            params['bbb_norm'].value *= self.extinction(2500, params)
            params['bbb_slope'].value = np.log(self.extinction(1000, params) / self.extinction(2500, params)) / np.log(0.4)
            params['EBV'].value = 0.01

        cov_range = {'bbb_norm': [params['bbb_norm'].value / 30, params['bbb_norm'].value / 2], 'bbb_slope': [0.1, 0.1],
                     'EBV': [0.01, 0.3], 'Rv': [0.2, 0.2], # 'c3': [0.05, 0.3],
                     'tor_type': [1, 5], 'tor_norm': [params['tor_norm'].value / 10, params['tor_norm'].value / 2],
                     'host_tau': [0.05, 0.5], 'host_tg': [0.1, 1],
                     'host_norm': [params['host_norm'].value / 30, params['host_norm'].value / 2], 'host_Av': [0.01, 0.3]
                     }

        if not new:
            cov = {}
            params['host_Av'].max = 5
            #params['host_tau'].value = 0.1
            for k in cov_range.keys():
                params[k].vary = True
                if params[k].stderr is None:
                    params[k].stderr = np.median(cov_range[k])
                if params[k].vary == True:
                    cov[k] = max([cov_range[k][0], min([cov_range[k][1], params[k].stderr])])

        if 'Fe_norm' in params.keys():
            params['Fe_norm'].vary = True
            cov['Fe_norm'] = 0.2 #cov['bbb_norm'] / 3

        if 'Abump' in params.keys():
            params['Abump'].vary = True
            cov['Abump'] = 0.2

        #print(cov)
        #params.add('host_L', value=0, min=0, max=100, vary=False)
        #cov['host_L'] = 0.1

        if tvary:
            params.add('sigma', value=0.2, min=0.01, max=3)
            cov['sigma'] = 0.02
            #params.add('alpha_spec', value=0.0, min=-3, max=3)
            #cov['alpha_spec'] = params['sigma'].value / 10
            #print(self.filters)
            for p in set([v for k, v in self.photo.items() if self.filters[k].fit]):
                if p != 'WISE':
                    params.add('alpha_' + p, value=0, min=-3, max=3)
                    cov['alpha_' + p] = params['sigma'].value
                    params.add('slope_' + p, value=0, min=-1, max=1)
                    cov['slope_' + p] = 0.02

            for p in set(self.photo.values()):
                if p != 'WISE':
                    chi = []
                    for k, f in self.filters.items():
                        if self.photo[k] == p and self.filters[k].fit:
                            chi.append([f.value, f.get_value(x=f.x, y=self.model_emcee(params, f.x, k))])
                    if len(chi) > 0:
                        params['alpha_' + p].value = -(np.sum(np.asarray(chi), axis=0)[0] - np.sum(np.asarray(chi), axis=0)[1]) / len(np.sum(np.asarray(chi), axis=0)) / 2.5
        #print(params)
        #print(cov)

        nwalkers, ndims = 50, len(params)

        pos = np.asarray([params[p].value + np.random.randn(nwalkers) * cov[p] for p in params])
        for k, p in enumerate(params.values()):
            pos[k][pos[k] <= p.min] = p.min + (p.max - p.min) / 100
            pos[k][pos[k] >= p.max] = p.max - (p.max - p.min) / 100
        pos = np.transpose(pos)
        #print(pos)
        #print(pos.shape, nwalkers, )

        print("run mcmc for ", self.ind)
        if 0:
            result = lmfit.minimize(self.fcn2min_mcmc, method='emcee', params=params, progress=self.verbose,
                                    nwalkers=nwalkers, steps=steps, burn=int(steps / 2), thin=2, pos=pos,
                                    nan_policy='omit')
            flat_sample = result.flatchain
        else:
            self.params = params
            if method == 'emcee':
                sampler = emcee.EnsembleSampler(nwalkers, ndims, self.fcn2min_mcmc, moves=[(emcee.moves.DESnookerMove(), 0.5), (emcee.moves.StretchMove(), 0.5)])

                subiters = 3
                #try:
                for i in range(subiters):
                    # We'll track how the average autocorrelation time estimate changes
                    autocorr = np.empty(self.mcmc_steps)

                    # This will be useful to testing convergence
                    index, old_tau = 0, np.inf

                    iterations = self.mcmc_steps // 10 if i < subiters - 1 else self.mcmc_steps
                    for sm in sampler.sample(pos, iterations=iterations, skip_initial_state_check=True, progress=(self.verbose == 1)):
                        # Only check convergence every <n> steps
                        if sampler.iteration % self.corr:
                            continue

                        # Compute the autocorrelation time so far
                        # Using tol=0 means that we'll always get an estimate even
                        # if it isn't trustworthy
                        tau = sampler.get_autocorr_time(tol=0)
                        autocorr[index] = np.mean(tau)
                        # print(tau)
                        # print(np.abs(old_tau - tau) / tau)
                        index += 1

                        # Check convergence
                        converged = np.all(tau * self.corr < sampler.iteration)
                        converged &= np.all(np.abs(old_tau - tau) / tau < 1 / self.corr)
                        if converged:
                            break
                        old_tau = tau

                    if i < subiters - 1:
                        lnL = sampler.get_last_sample().log_prob
                        pos = sampler.get_last_sample().coords
                        inds = np.argwhere(lnL < np.quantile(lnL, 0.7) + 1.0 * (np.quantile(lnL, 0.7) - np.max(lnL)))
                        #print(np.quantile(lnL, 0.7) + 1.0 * (np.quantile(lnL, 0.7) - np.max(lnL)), len(inds))
                        if len(inds) > 0:
                            mask = np.ones(lnL.shape, dtype=np.bool)
                            mask[inds] = False
                            mpos = np.mean(pos[mask], axis=0)
                            for ind in inds[0]:
                                pos[ind, :] = mpos + (mpos - pos[mask, :][np.random.randint(np.sum(mask))]) * (0.5 * np.random.random(len(mpos)))
                                for k, p in enumerate(params.values()):
                                    pos[ind, k] = np.max([p.min, np.min([p.max, pos[ind, k]])])
                #except:
                #    print(self.ind, i, len(inds))
                #    print(mpos)
                #    print(mask)
                #    print(pos[mask, :][np.random.randint(np.sum(mask))])
                #    print(0.5 * np.random.random(len(mpos)))
                #    print(params, cov, pos)
                

                thinning = int(0.5 * np.nanmax([2, np.nanmin(sampler.get_autocorr_time(tol=0))]))
                #print(thinning)
                flat_sample = sampler.get_chain(discard=sampler.iteration // 2, thin=thinning, flat=True)
                ln_max = -np.max(sampler.get_log_prob(discard=sampler.iteration // 2, thin=thinning, flat=True))
                #print(flat_sample)
                #except:
                #    print("Problem with ", self.ind, self.d['SDSS_NAME'])

            elif method == 'zeus':
                sampler = zeus.EnsembleSampler(nwalkers, ndims, self.fcn2min_mcmc)

                cb0 = zeus.callbacks.AutocorrelationCallback(ncheck=50, dact=0.05, nact=50, discard=0.5)
                cb1 = zeus.callbacks.SplitRCallback(ncheck=50, epsilon=0.05, nsplits=2, discard=0.5)
                cb2 = zeus.callbacks.MinIterCallback(nmin=100)

                sampler.run_mcmc(pos, self.mcmc_steps, callbacks=[cb0, cb1, cb2])
                if self.verbose:
                    print(sampler.summary)

                if self.plot:
                    tau = cb0.estimates
                    R = cb1.estimates

                    N = np.arange(len(tau)) * 100

                    figqc, ax = plt.subplots(ncols=2, figsize=(12, 6))

                    ax[0].plot(N, tau, lw=2.5)
                    ax[0].set_title('Integrated Autocorrelation Time', fontsize=14)
                    ax[0].set_xlabel('Iterations', fontsize=14)
                    ax[0].set_ylabel(r'$\tau$', fontsize=14)

                    ax[1].plot(N, R, lw=2.5)
                    ax[1].set_title('Split-R Gelman-Rubin Statistic', fontsize=14)
                    ax[1].set_xlabel('Iterations', fontsize=14)
                    ax[1].set_ylabel(r'$R$', fontsize=14)

                    figqc.tight_layout()
                    if self.save:
                        figqc.savefig(os.path.dirname(self.catalog) + '/QC/plots/' + self.d['SDSS_NAME'] + '_conv.png', bbox_inches='tight', pad_inches=0.1)

                flat_sample = sampler.get_chain(discard=sampler.iteration // 2, thin=10, flat=True)
                ln_max = -np.max(sampler.get_log_prob(discard=sampler.iteration // 2, thin=10, flat=True))
        #self.calc_host_lum(result.flatchain)

        #print(sampler.get_log_prob(discard=sampler.iteration // 2, thin=thinning, flat=True))
        pars, flat_sample = list(params.valuesdict().keys()) + ['lnL'], np.append(flat_sample, sampler.get_log_prob(discard=sampler.iteration // 2, thin=thinning, flat=True)[:, np.newaxis], axis=1)
        flat_sample = flat_sample[np.isfinite(flat_sample[:, -1]), :]
        #print(pars)
        #print(flat_sample.shape)
        #print(flat_sample)

        if flat_sample.shape[0] > flat_sample.shape[1]:
            if self.plot:
                if 1:
                    fig = corner.corner(flat_sample, labels=[str(p).replace('_', ' ') for p in pars])
                    if self.save:
                        fig.savefig(os.path.dirname(self.catalog) + '/QC/plots/' + self.d['SDSS_NAME'] + '_mcmc.png', bbox_inches='tight', pad_inches=0.1)
                else:
                    c = ChainConsumer()
                    #print(np.asarray(result.flatchain))
                    #print([r.replace('_', ' ') for r in result.var_names])
                    c.add_chain(np.asarray(flat_sample), parameters=[r.replace('_', ' ') for r in params.valuesdict().keys()])
                    c.configure(summary=True, bins=1.4, cloud=True, sigmas=np.linspace(0, 2, 3), smooth=1,
                                colors="#673AB7", shade_alpha=1)
                    fig = c.plotter.plot(figsize=(20, 15))
                    if self.save:
                        fig.savefig(os.path.dirname(self.catalog) + '/QC/plots/' + self.d['SDSS_NAME'] + '_mcmc.png', bbox_inches='tight', pad_inches=0.1)

            # >>> statistical determination:
            print("calc stats for ", self.ind)
            if stat:
                k = int(len(pars)) + 5 #samples.shape[1]
                n_hor = int(k ** 0.5)
                n_hor = np.max([n_hor, 2])
                n_vert = k // n_hor + 1 if k % n_hor > 0 else k // n_hor
                n_vert = np.max([n_vert, 2])

                fig, ax = plt.subplots(nrows=n_vert, ncols=n_hor, figsize=(6 * n_vert, 4 * n_hor))
                k = 0
                res = {}
                for i, p in enumerate(pars + ['M_UV', 'L_host', 'Av', 'L_UV_ext', 'L_UV_corr']):
                    if p not in ['M_UV', 'L_host', 'Av', 'L_UV_ext', 'L_UV_corr']:
                        d = distr1d(flat_sample[:, i].flatten())
                    else:
                        s = []
                        for l in range(flat_sample.shape[0]):
                            for j, p1 in enumerate(params.keys()):
                                params[p1].value = flat_sample[l, j]
                            if p == 'L_host':
                                s.append(np.log10(self.calc_host_lum(params, wave='bol')))
                            elif p == 'M_UV':
                                s.append(-2.5 * np.log10(self.calc_host_lum(params, wave=1700) * 9.64e-13) + 51.6)
                            elif p == 'Av':
                                s.append(params['EBV'].value * params['Rv'].value)
                            elif p == 'L_UV_ext':
                                #print('L_UV_ext', self.d['L_UV'], self.d['L_UV_err'], self.extinction(2500, params), np.log10((self.d['L_UV'] + np.random.randn() * self.d['L_UV_err']) / self.extinction(2500, params)))
                                s.append(np.log10((self.d['L_UV'] + np.random.randn() * self.d['L_UV_err']) / self.extinction(2500, params)))
                            elif p == 'L_UV_corr':
                                sind = np.argmin(np.abs(self.sm[0] - 2500 * (1 + self.d['z'])))
                                scaling = (1e-17 * u.erg / u.cm ** 2 / u.AA / u.s).to(u.erg / u.cm ** 2 / u.s / u.Hz, equivalencies=u.spectral_density(2500 * u.AA * (1 + self.d['z']))).value
                                s.append(np.log10(self.models['bbb'].data['spec'][0][sind] * params['bbb_norm'].value * 0.85 * scaling * 4 * np.pi * Planck15.luminosity_distance(self.d['z']).to('cm').value ** 2 / (1 + self.d['z'])))
                        if np.sum(np.isfinite(s)) > 0:
                            d = distr1d(np.asarray(s)[np.isfinite(s)])
                        else:
                            d = None
                    if d is not None:
                        d.dopoint()
                        d.dointerval()
                        res[p] = a(d.point, d.interval[1] - d.point, d.point - d.interval[0])
                        f = np.asarray([res[p].plus, res[p].minus])
                        f = int(np.round(np.abs(np.log10(np.min(f[np.nonzero(f)])))) + 1)
                        #print(d.interval[0], x[1], d.interval[1], x[-2])
                        d.dointerval(conf=0.95)
                        if p not in ['lnL', 'M_UV', 'L_host', 'Av', 'L_UV_ext', 'L_UV_corr']:
                            if d.interval[0] <= (d.x[1] + params[p].min) / 2 and d.interval[1] < (d.x[-2] + params[p].max) / 2:
                                res[p] = a(d.interval[1], t='u')
                            elif d.interval[1] >= (d.x[-2] + params[p].max) / 2 and d.interval[0] > (d.x[1] + params[p].min) / 2:
                                res[p] = a(d.interval[0], t='l')
                        elif p not in ['lnL', 'L_UV_ext', 'L_UV_corr']:
                            if d.interval[0] <= d.x[2] and d.interval[1] < d.x[-3]:
                                res[p] = a(d.interval[1], t='u')
                            elif d.interval[1] >= d.x[-3] and d.interval[0] > d.x[2]:
                                res[p] = a(d.interval[0], t='l')
                        
                        #print(p, res[p].latex(f=f))
                        vert, hor = k // n_hor, k % n_hor
                        k += 1
                        d.plot(conf=0.683, ax=ax[vert, hor], ylabel='')
                        ax[vert, hor].yaxis.set_ticklabels([])
                        ax[vert, hor].yaxis.set_ticks([])
                        ax[vert, hor].text(.05, .9, str(p).replace('_', ' '), ha='left', va='top', transform=ax[vert, hor].transAxes)
                        ax[vert, hor].text(.95, .9, res[p].latex(f=f), ha='right', va='top', transform=ax[vert, hor].transAxes)
                        #ax[vert, hor].set_title(pars[i].replace('_', ' '))

                if self.save:
                    fig.savefig(os.path.dirname(self.catalog) + '/QC/plots/' + self.d['SDSS_NAME'] + '_post.png', bbox_inches='tight', pad_inches=0.1)

            else:
                res = None
        else:
            print('problem with MCMC chain:', self.ind)
            res = None

        return flat_sample, ln_max, params, res

    def anneal_fit(self, params=None):

        #print(self.models['gal'].tg)
        #print(self.models['gal'].tau)
        if params is None:
            norm_bbb = np.nanmean(self.sm[1]) / np.nanmean(self.models['bbb'].data['spec'][0])
            print('norm_bbb:', norm_bbb)
            params = lmfit.Parameters()
            params.add('bbb_norm', value=norm_bbb, min=0, max=1e10)
            params.add('bbb_slope', value=0, min=-2, max=2, vary=False)
            params.add('Fe_norm', value=0, min=-1, max=100, vary=False)
            params.add('EBV', value=0.02, min=0.0, max=10)
            params.add('Rv', value=2.74, min=0.5, max=6.0, vary=False)
            if self.d['z'] > 0.70:
                params.add('Abump', value=0.01, min=0.0, max=100.0, vary=False)
            params.add('tor_type', value=10, vary=False, min=0, max=self.models['tor'].n - 1)
            params.add('tor_norm', value=norm_bbb / 100 * self.models['bbb'].data['spec'][0][-1] * np.max(self.models['tor'].data['spec'][params.valuesdict()['tor_type']]), min=0, max=1e10)
            print(norm_bbb / 100 * self.models['bbb'].data['spec'][0][-1] * np.max(self.models['tor'].data['spec'][params.valuesdict()['tor_type']]))
            if 0:
                params.add('host_type', value=0, vary=False, min=0, max=self.models['host'].n - 1)
            else:
                #print('age:', np.log10(Planck15.age(self.d['z']).to('yr').value))
                params.add('host_tau', value=0.2, vary=False, min=self.models['gal'].tau[0].value, max=3)
                params.add('host_tg', value=np.log10(Planck15.age(self.d['z']).to('yr').value), vary=False, min=np.log10(self.models['gal'].tg[0].value), max=np.log10(Planck15.age(self.d['z']).to('yr').value))
            norm_gal = np.nanmean(self.sm[1]) / np.nanmean(self.models['gal'].data['spec'][233])
            params.add('host_norm', value=norm_gal, min=0, max=1e10)
            params.add('host_Av', value=0.1, min=0, max=1.0)

        anneal_pars = OrderedDict([('tor_type', [int, 5])]) #, ('host_tau', [float, 0.5])]) #, ('host_tg', [float, 0.5])])
        #self.ln_like(params, plot=1)
        def objective(best, anneal_pars, params):
            for i, (p, f) in enumerate(anneal_pars.items()):
                params[p].value = best[i]

            #print(params)
            minner = lmfit.Minimizer(self.fcn2min, params, nan_policy='propagate', calc_covar=True)
            result = minner.minimize(method='leastsq')
            #lmfit.report_fit(result)
            chi = self.fcn2min(result.params)
            #print(np.sum(chi ** 2) / (len(chi) - len(result.params)))
            #print(np.sum(chi ** 2), 100 * self.lum_prior(result.params, kind='anneal'), self.anneal_priors(result.params))
            return (np.sum(chi ** 2) - self.anneal_priors(result.params)), result #/ (len(chi) - len(result.params)), result

        def simulated_annealing(objective, params, anneal_pars, n_iterations=self.anneal_steps, temp=1000):
            # generate an initial point
            best = [f[0](params[p].value) for p, f in anneal_pars.items()]
            # evaluate the initial point
            best_eval, res = objective(best, anneal_pars, params)
            if self.verbose:
                print(best, best_eval)
            # current working solution
            curr, curr_eval = best, best_eval
            # run the algorithm
            for i in range(n_iterations):
                # take a step
                #candidate = curr + randn(len(bounds)) * step_size
                candidate = [f[0](params[p].value + np.random.randn() * f[1]) for p, f in anneal_pars.items()]
                candidate = [p.min * (c < p.min) + p.max * (c > p.max) + c * ((c >= p.min) * (c <= p.max)) for c, p in zip(candidate, [params[p] for p in anneal_pars.keys()])]
                # evaluate candidate point
                candidate_eval, res = objective(candidate, anneal_pars, params)
                # check for new best solution
                if candidate_eval < best_eval:
                    # store new best point
                    best, best_eval = candidate, candidate_eval
                    # report progress
                    if self.verbose:
                        print('>%d f(%s) = %.5f' % (i, best, best_eval))
                # difference between candidate and current point evaluation
                diff = candidate_eval - curr_eval
                # calculate temperature for current epoch
                t = temp / float((i + 1) / 100)
                # calculate metropolis acceptance criterion
                metropolis = np.exp(-diff / t)
                #print(diff, t, metropolis)
                # check if we should keep the new point
                if diff < 0 or np.random.rand() < metropolis:
                    # store the new current point
                    curr, curr_eval = candidate, candidate_eval
            #print('best:', best)
            return objective(best, anneal_pars, params)

        chi2_min, result = simulated_annealing(objective, params, anneal_pars)
        #print(result.params)
        if self.verbose:
            print(chi2_min, lmfit.report_fit(result))
        return result, chi2_min

    def fit(self, ind=None, method='zeus'):

        self.ind = ind
        self.d = self.df.loc[ind]
        res = None
        print(ind, self.d['SDSS_NAME'])

        if method == 'annealing' and self.hostExt and any([f in self.filters.keys() for f in ['J', 'H', 'K', 'W1', 'W2']]) and any([f in self.filters.keys() for f in ['W3', 'W4']]):
            print('anneal:', ind, self.d['SDSS_NAME'])
            result, chi2_min = self.anneal_fit()
            host_min = self.models['gal'].get_model_ind(result.params)
            #print(result.params)
            if self.verbose:
                print(chi2_min, lmfit.report_fit(result))

            if self.plot:
                self.plot_spec(params=result.params)

        elif method in ['emcee', 'zeus'] and self.hostExt and any([f in self.filters.keys() for f in ['J', 'H', 'K', 'W1', 'W2']]) and any([f in self.filters.keys() for f in ['W3', 'W4']]):

            print('mcmc:', ind, self.d['SDSS_NAME'])
            if 1:
                result, chi2_min = self.anneal_fit()
                flat_sample, chi2_min, params, res = self.mcmc(params=result.params, method=method, nsteps=self.mcmc_steps)
            else:
                flat_sample, chi2_min, params, res = self.mcmc(method=method, nsteps=self.mcmc_steps)

            #print(res)

            if res is None:
                print('problem with ', self.ind)

            # >>> plot results:
            #plot = 1
            if self.plot and res is not None:
                print("plot spectrum of ", self.ind)
                inds = np.random.randint(flat_sample.shape[0], size=200)

                total, bbb, tor, host = [], [], [], []
                s = {}
                for k in ['total', 'bbb', 'tor', 'host'] + list(self.filters.keys()):
                    s[k] = []
                for i in inds:
                    #params = result.params
                    for k, p in enumerate(params.keys()):
                        #print(ind, k, flat_sample[ind, k])
                        params[p].value = flat_sample[i, k]
                        #if p in ['tor_type', 'host_tau', 'host_tg']:
                        #    params[p].value = np.max([params[p].min, np.min([params[p].max, round(params[p].value)])])
                    #self.ln_like(params, plot=plot)
                    #plot = 0
                    s['total'].append(self.model_emcee(params, self.spec[0], 'spec_full', mtype='total'))
                    s['bbb'].append(self.models['bbb'].models[0].y * params['bbb_norm'].value * (self.models['bbb'].models[0].x / 2500) ** params['bbb_slope'].value * self.extinction(self.models['bbb'].models[0].x, params))
                    s['tor'].append(self.models['tor'].models[self.models['tor'].get_model_ind(params)].y * params['tor_norm'].value)
                    if self.hostExt:
                        s['host'].append(self.models['gal'].models[self.models['gal'].get_model_ind(params)].y * params['host_norm'].value * self.extinction_MW(self.models['gal'].models[self.models['gal'].get_model_ind(params)].x, Av=params['host_Av'].value))
                        #s['total'][-1] += self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(self.spec[0] / (1 + self.d['z'])) * params['host_norm'].value * self.extinction(self.spec[0] / (1 + self.d['z']), Av=params['host_Av'].value)

                    # >>> filters fluxes:
                    for k, f in self.filters.items():
                        if self.filters[k].fit:
                            s[k].append(self.model_emcee(params, f.x, k, mtype='total'))

                self.fig.axes[0].fill_between(self.models['bbb'].models[0].x, *np.quantile(np.asarray(s['bbb']), [0.05, 0.95], axis=0), lw=1, color='tab:blue', zorder=3, label='bbb', alpha=0.5)
                self.fig.axes[0].fill_between(self.models['tor'].models[self.models['tor'].get_model_ind(params)].x, *np.quantile(np.asarray(s['tor']), [0.05, 0.95], axis=0), lw=1, color='tab:green', zorder=3, label='tor', alpha=0.5)
                #ax.plot(tor.x, tor.y * params['tor_norm'].value, '--', color='tab:orange', zorder=2, label='composite', alpha=alpha)
                if self.hostExt:
                    self.fig.axes[0].fill_between(self.models['gal'].models[self.models['gal'].get_model_ind(params)].x, *np.quantile(np.asarray(s['host']), [0.05, 0.95], axis=0), lw=1, color='tab:purple', zorder=2, label='host galaxy', alpha=0.5)

                for k, f in self.filters.items():
                    if self.filters[k].fit:
                        #print(k, f.x / (1 + self.d['z']), *np.quantile(np.asarray(s[k]), [0.05, 0.95], axis=0))
                        self.fig.axes[0].fill_between(f.x / (1 + self.d['z']), *np.quantile(np.asarray(s[k]), [0.05, 0.95], axis=0), color=[c / 255 for c in f.color], lw=1, zorder=3, alpha=0.5)

                self.fig.axes[0].fill_between(self.spec[0] / (1 + self.d['z']), *np.quantile(np.asarray(s['total']), [0.05, 0.95], axis=0), lw=1, color='tab:red', zorder=3, label='total', alpha=0.5)

                title = "id={0:4d} {1:19s} ({2:5d} {3:5d} {4:4d}) z={5:5.3f} slope={6:4.2f} EBV={7:4.2f} chi2={8:4.2f}".format(ind, self.df.loc[ind, 'SDSS_NAME'], self.df.loc[ind, 'PLATE'], self.df.loc[ind, 'MJD'], self.df.loc[ind, 'FIBERID'], self.df.loc[ind, 'z'], params['bbb_slope'].value, params['EBV'].value, chi2_min)
                if 0 and self.hostExt:
                    # title += " fgal={1:4.2f} {0:s}".format(self.models['host'].values[host_min], self.df['f_host' + '_photo' * self.addPhoto.isChecked()][i])
                    title += " fgal={2:4.2f} tau={0:4.2f} tg={1:4.2f}".format(self.models['gal'].values[host_min][0], self.models['gal'].values[host_min][1], self.df['f_host' + '_photo' * self.addPhoto][i])
                self.fig.axes[0].set_title(title)

                if self.save:
                    self.fig.savefig(os.path.dirname(self.catalog) + '/QC/plots/' + self.df.loc[ind, 'SDSS_NAME'] + '_spec.png', bbox_inches='tight', pad_inches=0.1)

                if 0:
                    plt.show()

        return res

    def plot_spec(self, params=None, fig=None, alpha=1):
        if self.fig is None:
            self.fig, ax = plt.subplots(figsize=(20, 12))
        else:
            ax = self.fig.axes[0]

        if self.spec is not None and params is None:
            ax.plot(self.spec[0] / (1 + self.d['z']), self.spec[1], '-k', lw=.5, zorder=2, label='spectrum')
            for k, f in self.filters.items():
                print(f.name, f.l_eff, f.flux, f.err_flux)
                ax.errorbar([f.l_eff / (1 + self.d['z'])], [f.flux], yerr=[[f.err_flux[0]], [f.err_flux[1]]], marker='s', color=[c / 255 for c in f.color])

            if self.addPhoto:
                ax.set_xlim([8e2, 3e5])
            else:
                ax.set_xlim([np.min(self.spec[0] / (1 + self.d['z'])), np.max(self.spec[0] / (1 + self.d['z']))])
            ax.set_ylim([0.001, ax.get_ylim()[1]])

            if self.mask is not None:
                ymin, ymax = ax.get_ylim()[0] * np.ones_like(self.spec[0] / (1 + self.d['z'])), ax.get_ylim()[1] * np.ones_like(self.spec[0] / (1 + self.d['z']))
                ax.fill_between(self.spec[0] / (1 + self.d['z']), ymin, ymax, where=self.mask, color='tab:green', alpha=0.3, zorder=0)
            #fig.legend(loc=1, fontsize=16, borderaxespad=0)

            if self.addPhoto:
                ax.set_ylim(0.2, ax.get_ylim()[1])
                ax.set_xscale('log')
                ax.set_yscale('log')

        if params is not None:
            host_min = params['host_tau'].value * (params['host_tg'].max + 1) + params['host_tg'].value
            bbb, tor, host = self.models['bbb'].models[0], self.models['tor'].models[params['tor_type'].value], self.models['gal'].models[host_min]

            # >>> plot templates:
            if alpha == 1:
                ax.plot(bbb.x, bbb.y * params['bbb_norm'].value, '--', color='tab:blue', zorder=2, label='composite', alpha=alpha)
            ax.plot(bbb.x, bbb.y * params['bbb_norm'].value * self.extinction(bbb.x, params),
                    '-', color='tab:blue', zorder=3, label='comp with ext', alpha=alpha)
            ax.plot(tor.x, tor.y * params['tor_norm'].value, '--', color='tab:orange', zorder=2, label='composite', alpha=alpha)
            if self.hostExt:
                if alpha == 1:
                    ax.plot(host.x, host.y * params['host_norm'].value, '--', color='tab:purple', zorder=2, label='host galaxy', alpha=alpha)

                ax.plot(host.x, host.y * params['host_norm'].value * self.extinction_MW(host.x, Av=params['host_Av'].value), '-', color='tab:purple', zorder=2, label='host galaxy', alpha=alpha)

            # >>> plot filters fluxes:
            for k, f in self.filters.items():
                temp = bbb.flux(f.x / (1 + self.d['z'])) * self.extinction(f.x / (1 + self.d['z']), params) * params['bbb_norm'].value + tor.flux(f.x / (1 + self.d['z'])) * params['tor_norm'].value
                if self.hostExt:
                    temp += host.flux(f.x / (1 + self.d['z'])) * params['host_norm'].value * self.extinction_MW(f.x / (1 + self.d['z']), Av=params['host_Av'].value)
                ax.plot(f.x / (1 + self.d['z']), temp, '-', color=[c / 255 for c in f.color], lw=2, zorder=3, alpha=alpha)
                # ax.scatter(f.filter.l_eff, f.filter.get_value(x=f.x, y=temp * self.extinction(f.x * (1 + self.d['z']), Av=params['Av'].value) * params['norm'].value),
                #           s=20, marker='o', c=[c/255 for c in f.filter.color])

            # >>> total profile:
            temp = bbb.flux(self.spec[0] / (1 + self.d['z'])) * params['bbb_norm'].value * self.extinction(self.spec[0] / (1 + self.d['z']), params) + tor.flux(self.spec[0] / (1 + self.d['z'])) * params['tor_norm'].value
            if self.hostExt:
                temp += host.flux(self.spec[0] / (1 + self.d['z'])) * params['host_norm'].value * self.extinction_MW(self.spec[0] / (1 + self.d['z']), Av=params['host_Av'].value)

            ax.plot(self.spec[0] / (1 + self.d['z']), temp, '-', lw=2, color='tab:red', zorder=3, label='total profile', alpha=alpha)
            # print(np.sum(((temp - spec[1]) / spec[2])[mask] ** 2) / np.sum(mask))

    def expand_mask(self, mask, exp_pixel=1):
        m = np.copy(mask)
        for p in itertools.product(np.linspace(-exp_pixel, exp_pixel, 2*exp_pixel+1).astype(int), repeat=2):
            m1 = np.copy(mask)
            if p[0] < 0:
                m1 = np.insert(m1[:p[0]], [0]*np.abs(p[0]), 0, axis=0)
            if p[0] > 0:
                m1 = np.insert(m1[p[0]:], [m1.shape[0]-p[0]]*p[0], 0, axis=0)
            m = np.logical_or(m, m1)
        #print(np.sum(mask), np.sum(m))
        #print(np.where(mask)[0], np.where(m)[0])
        return m

    def sdss_mask(self, mask):
        m = np.asarray([[s == '1' for s in np.binary_repr(m, width=29)[::-1]] for m in mask])
        l = [20, 22, 23, 26]
        return np.sum(m[:, l], axis=1)

    def calc_mask(self, spec, z_em=0, iter=3, window=101, clip=3.0):
        mask = np.logical_not(self.sdss_mask(spec[3]))
        #print(np.sum(mask))

        mask *= spec[0] > 1280 * (1 + z_em)

        for i in range(iter):
            m = np.zeros_like(spec[0])
            if window > 0 and np.sum(mask) > window:
                if i > 0:
                    m[mask] = np.abs(sm - spec[1][mask]) / spec[2][mask] > clip
                    mask *= np.logical_not(self.expand_mask(m, exp_pixel=2))
                    #mask[mask] *= np.abs(sm - spec[1][mask]) / spec[2][mask] < clip
                sm = smooth(spec[1][mask], window_len=window, window='hanning', mode='same')

        mask = np.logical_not(self.expand_mask(np.logical_not(mask), exp_pixel=3))

        mask[mask] *= (spec[1][mask] > 0) * (spec[1][mask] / spec[2][mask] > 1)
        
        #print(np.sum(mask))
        # remove  emission lines regions
        if 0:
            # all emission lines
            windows = [[1295, 1320], [1330, 1360], [1375, 1430], [1500, 1600], [1625, 1700], [1740, 1760],
                       [1840, 1960], [2050, 2120], [2250, 2650], [2710, 2890], [2940, 2990], [3280, 3330],
                       [3820, 3920], [4200, 4680], [4780, 5080], [5130, 5400], [5500, 5620], [5780, 6020],
                       [6300, 6850], [7600, 8050], [8250, 8300], [8400, 8600], [9000, 9400], [9500, 9700],
                       [9950, 10200]]
        else:
            # only strong ones
            windows = [[1295, 1320], [1330, 1360], [1375, 1430], [1500, 1600], [1625, 1700], [1740, 1760],
                       [1840, 1960], [2050, 2120], [2250, 2400], #[2250, 2650], #[2710, 2890],
                       [2630, 2930], #[2940, 2990], [3280, 3330],
                       [3820, 3920], #[4200, 4680],
                       [4920, 5080], #[4780, 5080],
                       [5130, 5400], [5500, 5620], [5780, 6020],
                       [6300, 6850], [7600, 8050], [8250, 8300], [8400, 8600], [9000, 9400], [9500, 9700],
                       [9950, 10200]]
            # only strongest ones
            windows = [[1500, 1600], [1840, 1960], [2760, 2860], [4920, 5080],  # [4780, 5080],
                       [6300, 6850]]
        for w in windows:
            mask *= (spec[0] < w[0] * (1 + z_em)) + (spec[0] > w[1] * (1 + z_em))

        #print(np.sum(mask))
        # remove atmospheric absorption region
        windows = [[5560, 5600], [6865, 6930], [7580, 7690], [9300, 9600], [10150, 10400], [13200, 14600]]
        for w in windows:
            mask *= (spec[0] < w[0]) + (spec[0] > w[1])

        #print(np.sum(mask))
        return mask

class jsoncat():
    def __init__(self, path='', prefix=''):
        self.data = []
        self.inds = []
        self.path = path
        self.prefix = prefix
        self.load()

    def load(self):
        with open(self.path + '/data' + self.prefix + '.json', 'r') as f:
            try:
                self.data = json.load(f)
            except:
                pass
            self.inds = [r[0] for r in self.data]

    def save(self, ):
        with open(self.path + '/data' + self.prefix + '.json', 'w') as f:
            f.write(json.dumps(self.data, indent=4, cls=resEncoder))

    def add(self, res):
        for r in res:
            if r[1] is not None:
                if r[0] in self.inds:
                    self.data[self.inds.index(r[0])] = r
                else:
                    self.data.insert(bisect(self.inds, r[0]), r)
                    self.inds.insert(bisect(self.inds, r[0]), r[0])
            else:
                with open(self.path + '/missed.dat', 'r') as f:
                    d = np.sort([int(float(l.strip())) for l in f.readlines() if l.strip() != ''])
                if r[0] not in d:
                    d = np.append(d, r[0])
                    np.savetxt(self.path + '/missed.dat', np.sort(d).astype(int), fmt='%i')
               
    def to_numpy(self):
        for k, ind in enumerate(inds):
            for i, p in enumerate(pars):
                if p in res[ind][1].keys():
                    pass

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return self.data

def run_model(ind, catfile=None):
    print(ind)

    qso = QSOSEDfit(catalog=catfile, plot=1, mcmc_steps=3000, anneal_steps=100, save=1, corr=30, verbose=1)
    if qso.prepare(ind):
        res = qso.fit(ind, method='emcee')
    else:
        res = None
    return (int(ind), res)
        
if __name__ == "__main__":
    # necessary to add cwd to path when script run 
    # by slurm (since it executes a copy)
    sys.path.append(os.getcwd())

    catfile = '/home/balashev/science/Erosita/match2_DR14Q_add.csv'
    catfile = 'C:/science/Erosita/UV_Xray/match2_DR14Q_add.csv'
    catfile = 'C:/science/Erosita/UV_Xray/individual/individual_targets_add.csv'

    try:
        i1, i2 = int(sys.argv[1]), int(sys.argv[2])
    except:
        i1, i2 = 1, 1

    if 1:
        res = run_model(0, catfile=catfile)
        print(res)
    else:
        #pars = ['bbb_norm', 'Av', 'tor_type', 'tor_norm', 'host_tau', 'host_tg', 'host_norm', 'host_Av', 'sigma', 'alpha_GALEX', 'alpha_SDSS', 'alpha_2MASS', 'alpha_UKIDSS']
        num = 90
        calc = 1
        if calc:
            if 1:
                for i in range(7975 // num + 1):
                    if i % i2 + 1 == i1:
                        with Pool(num) as p:
                            res_new = p.map(run_model, np.arange(i * num, min((i + 1) * num, 7975))) #total number of AGNs 7975
                        print(res_new)
                        res = jsoncat(path=path)
                        res.add(res_new)
                        res.save()
            else:
                pass
                    #print(p, res[ind][1][p])
                #out[k, i * 3] = res[ind][1][p].value
                #if

    plt.show()
