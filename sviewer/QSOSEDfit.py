import astropy.constants as ac
import scipy.stats
from astropy.cosmology import Planck15 #, FlatLambdaCDM, LambdaCDM
from astropy.io import fits
import astropy.units as u
from bisect import bisect
from collections import OrderedDict
import corner
from copy import copy
from chainconsumer import ChainConsumer
import dynesty
from dynesty import plotting as dyplot
import emcee
from functools import partial
import json
import itertools
import matplotlib.pyplot as plt
#from nautilus import Prior, Sampler
from multiprocessing import Pool
import numpy as np
import lmfit
import os
import pandas as pd
import pickle
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d, CubicSpline
from scipy.special import gamma, gammainc
from scipy import stats as st
import sys
#import zeus

if __name__ in ["__main__", "__mp_main__"]:
    from a_unc import a
    from stats import distr1d
    from utils import Timer
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

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>
# >>>   SpectRes
# >>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

"""
SpectRes: A fast spectral resampling function.
Copyright (C) 2017  A. C. Carnall
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# Function for calculating the left hand side (lhs) positions and widths of the spectral bins from their central wavelengths.
def make_bins(wavelengths, make_rhs="False"):
    bin_widths = np.zeros(wavelengths.shape[0])

    # This option makes the final entry in the left hand sides array the right hand side of the final bin
    if make_rhs == "True":
        bin_lhs = np.zeros(wavelengths.shape[0] + 1)
        # The first lhs position is assumed to be as far from the first central wavelength as the rhs of the first bin.
        bin_lhs[0] = wavelengths[0] - (wavelengths[1] - wavelengths[0]) / 2
        bin_widths[-1] = (wavelengths[-1] - wavelengths[-2])
        bin_lhs[-1] = wavelengths[-1] + (wavelengths[-1] - wavelengths[-2]) / 2
        bin_lhs[1:-1] = (wavelengths[1:] + wavelengths[:-1]) / 2
        bin_widths[:-1] = bin_lhs[1:-1] - bin_lhs[:-2]

    # Otherwise just return the lhs positions of each bin
    else:
        bin_lhs = np.zeros(wavelengths.shape[0])
        bin_lhs[0] = wavelengths[0] - (wavelengths[1] - wavelengths[0]) / 2
        bin_widths[-1] = (wavelengths[-1] - wavelengths[-2])
        bin_lhs[1:] = (wavelengths[1:] + wavelengths[:-1]) / 2
        bin_widths[:-1] = bin_lhs[1:] - bin_lhs[:-1]

    return bin_lhs, bin_widths


# Function for performing spectral resampling on a spectrum or array of spectra.
def spectres(spec_wavs, spec_fluxes, resampling, spec_errs=None, filename=None):
    # Generate arrays of left hand side positions and widths for the old and new bins
    resampling = resampling[(resampling > spec_wavs[0]) * (resampling < spec_wavs[-1])]
    filter_lhs, filter_widths = make_bins(resampling, make_rhs="True")
    spec_lhs, spec_widths = make_bins(spec_wavs)

    # Check that the range of wavelengths to be resampled onto falls within the initial sampling region
    if filter_lhs[0] < spec_lhs[0] or filter_lhs[-1] > spec_lhs[-1]:
        print("Spec_lhs, filter_lhs, filter_rhs, spec_rhs ", spec_lhs[0], filter_lhs[0], filter_lhs[-1], spec_lhs[-1], filename)
        sys.exit("spectres was passed a spectrum which did not cover the full wavelength range of the specified filter curve.")

    # Generate output arrays to be populated
    if spec_fluxes.ndim == 1:
        resampled = np.zeros((resampling.shape[0]))

    elif spec_fluxes.ndim == 2:
        resampled = np.zeros((len(resampling), spec_fluxes.shape[1]))

    if spec_errs is not None:
        if spec_errs.shape != spec_fluxes.shape:
            sys.exit("If specified, spec_errs must be the same shape as spec_fluxes.")
        else:
            resampled_errs = np.copy(resampled)

    start = 0
    stop = 0

    # Calculate the new spectral flux and uncertainty values, loop over the new bins
    for j in range(len(filter_lhs) - 1):

        # Find the first old bin which is partially covered by the new bin
        while spec_lhs[start + 1] <= filter_lhs[j]:
            start += 1

        # Find the last old bin which is partially covered by the new bin
        while spec_lhs[stop + 1] < filter_lhs[j + 1]:
            stop += 1

        if spec_fluxes.ndim == 1:

            # If the new bin falls entirely within one old bin the are the same the new flux and new error are the same as for that bin
            if stop == start:

                resampled[j] = spec_fluxes[start]
                if spec_errs is not None:
                    resampled_errs[j] = spec_errs[start]

            # Otherwise multiply the first and last old bin widths by P_ij, all the ones in between have P_ij = 1
            else:

                start_factor = (spec_lhs[start + 1] - filter_lhs[j]) / (spec_lhs[start + 1] - spec_lhs[start])
                end_factor = (filter_lhs[j + 1] - spec_lhs[stop]) / (spec_lhs[stop + 1] - spec_lhs[stop])

                spec_widths[start] *= start_factor
                spec_widths[stop] *= end_factor

                # Populate the resampled spectrum and uncertainty arrays
                resampled[j] = np.sum(spec_widths[start:stop + 1] * spec_fluxes[start:stop + 1]) / np.sum(
                    spec_widths[start:stop + 1])

                if spec_errs is not None:
                    resampled_errs[j] = np.sqrt(
                        np.sum((spec_widths[start:stop + 1] * spec_errs[start:stop + 1]) ** 2)) / np.sum(
                        spec_widths[start:stop + 1])

                # Put back the old bin widths to their initial values for later use
                spec_widths[start] /= start_factor
                spec_widths[stop] /= end_factor


        # The same as above, except operates on each row of the array, resampling all of the input models
        elif spec_fluxes.ndim == 2:

            if stop == start:

                resampled[j, :] = spec_fluxes[start, :]
                if spec_errs is not None:
                    resampled_errs[j, :] = spec_errs[start, :]

            else:

                start_factor = (spec_lhs[start + 1] - filter_lhs[j]) / (spec_lhs[start + 1] - spec_lhs[start])
                end_factor = (filter_lhs[j + 1] - spec_lhs[stop]) / (spec_lhs[stop + 1] - spec_lhs[stop])

                spec_widths[start] *= start_factor
                spec_widths[stop] *= end_factor

                resampled[j, :] = np.sum(
                    np.expand_dims(spec_widths[start:stop + 1], axis=1) * spec_fluxes[start:stop + 1, :],
                    axis=0) / np.sum(spec_widths[start:stop + 1])

                if spec_errs is not None:
                    resampled_errs[j, :] = np.sqrt(
                        np.sum((np.expand_dims(spec_widths[start:stop + 1], axis=1) * spec_errs[start:stop + 1]) ** 2,
                               axis=0)) / np.sum(spec_widths[start:stop + 1])

                spec_widths[start] /= start_factor
                spec_widths[stop] /= end_factor

    # If errors were supplied return the resampled spectrum and error arrays
    if spec_errs is not None:
        return resampled, resampled_errs

    # Otherwise just return the resampled spectrum array
    else:
        return resampled

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

def weighted_quantile(values, quantiles, sample_weight=None, axis=0,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values, axis=axis)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

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

        colors = {'u': (23, 190, 207), 'g': (44, 160, 44), 'r': (255, 139, 0), 'i': (227, 119, 194), 'z': (153, 102, 204),
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
            self.weight = 1
        else:
            self.weight = 1
            #np.sqrt(np.sum(self.filter.inter(x)) / np.max(self.filter.data.y))
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

    def ext_fm07(self, wave, Rv=2.74, c1=-4.959, c2=2.264, c3=0.389, c4=0.319, c5=6.097, x0=4.592, gamma=0.922):
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

        return k
    def load_data(self, smooth_window=None, xmin=None, xmax=None, z=0, x=None, y=None):

        if x is None and y is None:
            if self.name in ['VandenBerk', 'HST', 'Selsing', 'power', 'composite']:
                self.type = 'qso'

                if self.name == 'VandenBerk':
                    self.x, self.y = np.genfromtxt(self.parent.path + r'/data/SDSS/medianQSO.dat', skip_header=2, unpack=True)
                elif self.name == 'HST':
                    self.x, self.y = np.genfromtxt(self.parent.path + r'/data/SDSS/hst_composite.dat', skip_header=2, unpack=True)
                elif self.name == 'Selsing':
                    self.x, self.y = np.genfromtxt(self.parent.path + r'/data/SDSS/Selsing2016.dat', skip_header=0, unpack=True, usecols=(0, 1))
                elif self.name == 'power':
                    self.x = np.linspace(500, 25000, 1000)
                    self.y = np.power(self.x / 2500, -1.9)
                    smooth_window = None
                elif self.name == 'composite':
                    if 1:
                        self.x, self.y = np.genfromtxt(self.parent.path + r'/data/SDSS/QSO_composite.dat', skip_header=0, unpack=True)
                        self.x, self.y = self.x[self.x > 0], self.y[self.x > 0]
                    else:
                        self.template_qso = np.genfromtxt(self.parent.path + r'/data/SDSS/Selsing2016.dat', skip_header=0, unpack=True)
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

            elif self.name == 'ext':
                self.x = np.logspace(2, 5, 100)
                self.y = self.ext_fm07(self.x, Rv=2.7, c1=-4.959, c2=2.264, c3=0, c4=0.319, c5=6.097, x0=4.592, gamma=0.922)

            elif self.name == 'ext_gal':
                self.x = np.logspace(2, 5, 100)
                self.y = self.ext_fm07(self.x, Rv=3.01, c1=-0.175, c2=0.807, c3=2.991, c4=0.319, c5=6.097, x0=4.592, gamma=0.922)

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
                self.tau = np.asarray([np.log10(p.value) for p in pickle.load(f)])
                self.tg = np.asarray([np.log10(p.value / 1e9) for p in pickle.load(f)])
                SED = pickle.load(f)
                #print(self.tau, self.tg)
                #print(self.tau.shape, self.tg.shape, SED.shape)
                for i in range(len(self.tau)):
                    for k in range(len(self.tg)):
                        self.values.append([self.tau[i], self.tg[k]])
                        self.models.append(sed_template(self, 'gal', x=l.value, y=SED[k, i, :].value * 1e5))
            self.n_tau = self.tau.shape[0]
            self.n_tg = self.tg.shape[0]

        if self.name == 'Fe':
            self.models.append(sed_template(self, 'Fe'))
            self.values = [0]

        if self.name == 'ext':
            self.models.append(sed_template(self, 'ext'))
            self.values = [0]

        if self.name == 'ext_gal':
            self.models.append(sed_template(self, 'ext_gal'))
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
            return np.argmin(np.abs((params['host_tau'].value - self.tau))) * self.n_tg + np.argmin(np.abs((params['host_tg'].value - self.tg)))

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
        self.iters = 1000

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
        if np.sum(self.mask) > 10:
            self.set_filters(ind)
            if self.plot:
                self.plot_spec()

            self.sm = [np.asarray(self.spec[0][self.mask], dtype=np.float64), self.spec[1][self.mask], self.spec[2][self.mask]]

            self.models = {}
            for name in ['bbb', 'tor', 'host', 'gal', 'Fe', 'ext', 'ext_gal']:
                self.models[name] = sed(name=name, xmin=self.wavemin, xmax=self.wavemax, z=self.df.loc[ind, 'z'])
                self.models[name].set_data('spec', self.sm[0])
                for k, f in self.filters.items():
                    if self.filters[k].fit:
                        self.models[name].set_data(k, f.x)
                self.models[name].set_data('spec_full', self.spec[0])

            return True

    def set_filters(self, ind, names=None, add_UV=False, add_SDSS=False):
        self.photo = {}
        self.filters = {}
        if self.addPhoto:
            if add_UV:
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

            if add_SDSS:
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

    def add_mask(self, name=None):
        if name is not None:
            self.mask = np.logical_or(self.mask, self.df['SDSS_NAME'] == name)
            #print(np.sum(self.mask))

    def D_bump(self, x, x0, gamma):
        return x ** 2 / ((x ** 2 - x0 ** 2) ** 2 + x ** 2 * gamma ** 2)

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

        k[uv] = c1 + c2 * x[uv] + c3 * self.D_bump(x[uv], x0, gamma)
        k[uv * (x > c5)] += c4 * (x[uv * (x > c5)] - c5) ** 2

        # Anchors
        x_uv = 1.e4 / np.array([2700., 2600.])
        k_uv = c1 + c2 * x_uv + c3 * self.D_bump(x_uv, x0, gamma)
        x_opt = 1.e4 / np.array([5530., 4000., 3300.])
        k_opt = np.array([0., 1.322, 2.055])
        x_ir = np.array([0., 0.25, 0.50, 0.75, 1.])
        k_ir = (-0.83 + 0.63 * Rv) * x_ir ** 1.84 - Rv

        k[~uv] = CubicSpline(np.concatenate(([x_ir, x_opt, x_uv])), np.concatenate((k_ir, k_opt, k_uv)))(x[~uv])
        #k[~uv] = interp1d(np.concatenate(([x_ir, x_opt, x_uv])), np.concatenate((k_ir, k_opt, k_uv)), kind='cubic', assume_sorted=True)(x[~uv])

        return Av * (k / Rv + 1)

    def extinction_Pervot(self, x, params, z_ext=0):
        return 10 ** (-0.4 * params['Av'].value * (1.39 * (1e4 / np.asarray(x, dtype=np.float64) * (1 + z_ext)) ** 1.2 - 0.38) / 2.74)

    def extinction_MW(self, x, Av=0):
        #return 10 ** (-0.4 * extinction.Fitzpatrick99(3.1 * 1.0)(np.asarray(x, dtype=np.float64), Av))
        #return 10 ** (-0.4 * Av * (1.39 * (1e4 / np.asarray(x, dtype=np.float64)) ** 1.2 - 0.38) / 2.74)
        return 10 ** (-0.4 * self.ext_fm07(x, Av=Av, Rv=3.01, c1=-0.175, c2=0.807, c3=2.991, c4=0.319, c5=6.097, x0=4.592, gamma=0.922))

    def extinction(self, wave, params, z_ext=0, kind='qso'):
        """
        Return extinction for provided wavelengths
        Args:
            wave:      wavelengths in Angstrem
            z_ext:     redshift
            params:    parameters for visual extinction

        Returns: extinction
        """
        Rv = 2.71 if 'Rv' not in params.keys() else params['Rv'].value
        #c3 = 0.389 if 'c3' not in params.keys() else params['c3'].value
        c3 = 0.389 if 'Abump' not in params.keys() else params['Abump'].value * 2 * 0.922 / params['EBV'].value / np.pi
        c3 = 0.0 if 'Abump' not in params.keys() else params['Abump'].value * 2 * 0.922 / params['EBV'].value / np.pi
        #s = time.time()
        #self.extinction_Pervot(wave, params, z_ext=z_ext)
        #print('t_Pervot:', time.time() - s)
        #s = time.time()
        #10 ** (-0.4 * self.ext_fm07(wave * (1 + z_ext), Av=params['Av'].value, Rv=Rv, c1=-4.959, c2=2.264, c3=0.389, c4=0.319, c5=6.097, x0=4.592, gamma=0.922))
        #print('t_FM07:', time.time() - s)
        #return self.extinction_Pervot(wave, params, z_ext=z_ext)
        #print(self.ext_fm07(wave * (1 + z_ext), Av=params['EBV'].value * Rv, Rv=Rv, c1=-4.959, c2=2.264, c3=c3, c4=0.319, c5=6.097, x0=4.592, gamma=0.922))
        return 10 ** (-0.4 * self.ext_fm07(wave * (1 + z_ext), Av=params['EBV'].value * Rv, Rv=Rv, c1=-4.959, c2=2.264, c3=c3, c4=0.319, c5=6.097, x0=4.592, gamma=0.922))

    def loadSDSS(self, plate, fiber, mjd, Av_gal=np.nan, rebin=11):
        filename = os.path.dirname(self.catalog) + '/spectra/spec-{0:04d}-{2:05d}-{1:04d}.fits'.format(int(plate), int(fiber), int(mjd))
        #print(filename)
        if os.path.exists(filename):
            qso = fits.open(filename)
            ext = self.extinction_MW(10 ** qso[1].data['loglam'][:], Av=Av_gal)
            mask = np.logical_and(np.logical_and(np.isfinite(qso[1].data['flux']), np.isfinite(np.sqrt(1.0 / qso[1].data['ivar']))), (ext != 0))
            #print('mask: ', np.sum(mask))
            #print(Av_gal, ext)
            x, y, err, mask = 10 ** qso[1].data['loglam'], qso[1].data['flux'] / ext, np.sqrt(1.0 / qso[1].data['ivar']) / ext, np.logical_and(mask, qso[1].data['and_mask'] == 0)
            if rebin > 1:
                y, err = spectres(x, y, x[int(rebin / 2) + 1:((len(x) // rebin) - 1) * rebin:rebin], err, filename=filename)
                err *= rebin
                mask = spectres(x, mask, x[int(rebin / 2) + 1:((len(x) // rebin) - 1) * rebin:rebin]) > 0.9
                x = x[int(rebin / 2) + 1:((len(x) // rebin) - 1) * rebin:rebin]
                #mask = np.sum(mask[int(rebin/2)+1:((len(mask) // rebin) - 1) * rebin + int(rebin/2)+2].reshape(len(mask) // rebin, rebin), axis=1) > 1
            return [x, y, err, mask]

    def spec_model(self, params, x):
        return None

    def model(self, params, x, dtype, mtype='total'):
        model = np.zeros_like(self.models['bbb'].data[dtype][0])
        if mtype in ['total', 'bbb']:
            model = self.models['bbb'].data[dtype][0] * 10 ** params['bbb_norm'].value * self.extinction(x / (1 + self.d['z']), params)
        if mtype in ['total', 'tor'] and params['tor_type'].value > -1:
            model += self.models['tor'].data[dtype][self.models['tor'].get_model_ind(params)] * 10 ** params['tor_norm'].value
        if mtype in ['total', 'gal'] and params['host_tau'].value > -1 and params['host_tg'].value > -1:
            model += self.models['gal'].data[dtype][self.models['gal'].get_model_ind(params)] * 10 ** params['host_norm'].value * self.extinction_MW(x / (1 + self.d['z']), Av=params['host_Av'].value)
        #if params['host_type'].value > -1:
        #    model += self.models['host'].data[kind][params['host_type'].value] * 10 ** params['host_norm'].value * self.extinction(x / (1 + d['z']), Av=params['host_Av'].value)
        return model

    def model_emcee_old(self, params, x, dtype, mtype='total'):
        for s in ['spec', 'spec_full']:
            self.photo[s] = 'spec'
        alpha = 10 ** params['alpha_' + self.photo[dtype]].value * (x / (1 + self.d['z']) / 2500) ** params['slope_' + self.photo[dtype]].value if self.photo[dtype] not in ['WISE', 'spec'] else 1
        #print(dtype, self.photo[dtype], alpha)
        if 'bbb_slope' in params.keys():
            alpha *= (x / (1 + self.d['z']) / 2500) ** params['bbb_slope'].value
        model = np.zeros_like(self.models['bbb'].data[dtype][0])
        if mtype in ['total', 'bbb']:
            model += self.models['bbb'].data[dtype][0] * alpha * 10 ** params['bbb_norm'].value
        if mtype in ['total', 'Fe']:
            model += self.models['Fe'].data[dtype][0] * params['Fe_norm'].value * 10 ** params['bbb_norm'].value
        if mtype in ['total', 'bbb', 'Fe']:
            model *= self.extinction(x / (1 + self.d['z']), params)
        if mtype in ['total', 'tor'] and params['tor_type'].value > -1:
            model += self.models['tor'].data[dtype][self.models['tor'].get_model_ind(params)] * 10 ** params['tor_norm'].value
        if mtype in ['total', 'gal'] and params['host_tau'].value > -1 and params['host_tg'].value > -1:
            model += self.models['gal'].data[dtype][self.models['gal'].get_model_ind(params)] * 10 ** params['host_norm'].value * self.extinction_MW(x / (1 + self.d['z']), Av=params['host_Av'].value)

        #if params['host_type'].value > -1:
        #    model += self.models['host'].data[kind][params['host_type'].value] * 10 ** params['host_norm'].value * self.extinction(x / (1 + d['z']), Av=params['host_Av'].value)
        return model

    def model_emcee(self, params, x, dtype, mtype='total', lnorm=2200):
        for s in ['spec', 'spec_full']:
            self.photo[s] = 'spec'
        alpha = 10 ** params['alpha_' + self.photo[dtype]].value * (x / (1 + self.d['z']) / lnorm) ** params['slope_' + self.photo[dtype]].value if self.photo[dtype] not in ['WISE', 'spec'] else 1
        #print(dtype, self.photo[dtype], alpha)
        if 'bbb_slope' in params.keys():
            alpha *= (x / (1 + self.d['z']) / lnorm) ** params['bbb_slope'].value
        model = np.zeros_like(self.models['bbb'].data[dtype][0])
        if mtype in ['total', 'bbb']:
            model += self.models['bbb'].data[dtype][0] * alpha * 10 ** params['bbb_norm'].value
        if mtype in ['total', 'Fe']:
            model += self.models['Fe'].data[dtype][0] * params['Fe_norm'].value * 10 ** params['bbb_norm'].value
        if mtype in ['total', 'bbb', 'Fe']:
            model *= 10 ** (-0.4 * params['EBV'].value * (self.models['ext'].data[dtype][0] + params['Rv'].value))
        if mtype in ['total', 'tor'] and params['tor_type'].value > -1:
            model += self.models['tor'].data[dtype][self.models['tor'].get_model_ind(params)] * 10 ** params['tor_norm'].value
        if mtype in ['total', 'gal'] and params['host_tau'].value > -1 and params['host_tg'].value > -1:
            model += self.models['gal'].data[dtype][self.models['gal'].get_model_ind(params)] * 10 ** params['host_norm'].value * 10 ** (-0.4 * params['host_Av'].value * (self.models['ext_gal'].data[dtype][0] / 2.7 + 1))

        #if params['host_type'].value > -1:
        #    model += self.models['host'].data[kind][params['host_type'].value] * 10 ** params['host_norm'].value * self.extinction(x / (1 + d['z']), Av=params['host_Av'].value)
        return model

    def model_emcee_slope(self, params, x, dtype, mtype='total'):
        for s in ['spec', 'spec_full']:
            self.photo[s] = 'spec'
        if self.photo[dtype] not in ['WISE', 'spec']:
            alpha = 10 ** params['alpha_' + self.photo[dtype]].value * (x / (1 + self.d['z']) / 2500) ** params['slope_' + self.photo[dtype]].value
            alpha = alpha[:,np.newaxis] * np.power((x / (1 + self.d['z']) / 2500)[:, np.newaxis], np.linspace(-0.5, 0.5, 10))
        else:
            alpha = np.power((x / (1 + self.d['z']) / 2500)[:, np.newaxis], np.linspace(-0.5, 0.5, 10))
        #print(dtype, self.photo[dtype])
        #print(alpha.shape)
        model = np.zeros_like(alpha)
        if mtype in ['total', 'bbb']:
            model += 10 ** params['bbb_norm'].value * self.models['bbb'].data[dtype][0][:, np.newaxis] * alpha
        if mtype in ['total', 'Fe']:
            model += self.models['Fe'].data[dtype][0][:, np.newaxis] * params['Fe_norm'].value * 10 ** params['bbb_norm'].value
        if mtype in ['total', 'bbb', 'Fe']:
            model *= 10 ** (-0.4 * params['EBV'].value * (self.models['ext'].data[dtype][0][:, np.newaxis] + params['Rv'].value))
        if mtype in ['total', 'tor'] and params['tor_type'].value > -1:
            model += self.models['tor'].data[dtype][self.models['tor'].get_model_ind(params)][:, np.newaxis] * 10 ** params['tor_norm'].value
        if mtype in ['total', 'gal'] and params['host_tau'].value > -1 and params['host_tg'].value > -1:
            model += self.models['gal'].data[dtype][self.models['gal'].get_model_ind(params)][:, np.newaxis] * 10 ** params['host_norm'].value * 10 ** (-0.4 * params['host_Av'].value * (self.models['ext_gal'].data[dtype][0][:, np.newaxis] / 2.7 + 1))

        #print(model.shape)
        #if params['host_type'].value > -1:
        #    model += self.models['host'].data[kind][params['host_type'].value] * 10 ** params['host_norm'].value * self.extinction(x / (1 + d['z']), Av=params['host_Av'].value)
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
            lum = self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(wave) * 10 ** params['host_norm'].value * self.extinction_MW(wave, Av=params['host_Av'].value) * \
                         1e-17 * 4 * np.pi * Planck15.luminosity_distance(self.d['z']).to('cm').value ** 2
        elif wave in ['bol', 'total']:
            wave = self.models['gal'].models[self.models['gal'].get_model_ind(params)].x
            lum = np.trapz(self.models['gal'].models[self.models['gal'].get_model_ind(params)].y * 10 ** params['host_norm'].value * self.extinction_MW(wave, Av=params['host_Av'].value),
                         x=self.models['gal'].models[self.models['gal'].get_model_ind(params)].x)
            lum *= 1e-17 * 4 * np.pi * Planck15.luminosity_distance(self.d['z']).to('cm').value ** 2
        return lum

    def fcn2min(self, params, outliers=True):
        chi = (self.model(params, self.sm[0], 'spec') - self.sm[1]) / self.sm[2]
        if outliers:
            chi = chi[np.abs(chi) < np.quantile(np.abs(chi), 0.95)]
        for k, f in self.filters.items():
            if self.filters[k].fit and any([s in k for s in ['UKIDSS', 'W', '2MASS']]):
                chi = np.append(chi, [f.weight / f.err * (f.value - f.get_value(x=f.x, y=self.model(params, f.x, k)))])
        return chi

    def fcn2min_mcmc(self, pars):
        #print(params)
        params = self.set_params(pars)
        #print(pars)
        #params['host_L'].value = self.calc_host_lum(params)[0]
        lp = self.ln_priors(params)
        if not np.isfinite(lp):
            return -np.inf
        #print(lp, np.sum(np.power(self.ln_like(pars), 2)))
        return lp - .5 * np.sum(np.power(self.ln_like(params), 2))
        #return lp -.5 * np.sum(np.power(self.ln_like_slope(pars), 2))

    def lnlike_nest(self, pars):
        if 1:
            #t = Timer()
            params = self.set_params(pars)
            y = self.ln_like(params)[:]
            #t.time('calc')
        else:
            y = np.asarray([(self.params[p].value - v) ** 2 for p, v in zip(self.params, pars)])
        return - .5 * np.sum(np.power(y[np.isfinite(y)], 2))

    def ptform(self, u):
        #print(self.params)
        #t = Timer()
        x = np.array(u)
        for i, p in enumerate(self.params.values()):
            if p.name == 'bbb_slope':
                x[i] = st.norm.ppf(u[i], loc=0.0, scale=0.3)
            elif p.name == 'Fe_norm':
                x[i] = st.norm.ppf(u[i], loc=0.0, scale=0.2)
            elif p.name == 'host_tau':
                x[i] = np.min([st.expon.ppf(u[i]), p.max])
            elif p.name == 'host_tg':
                x[i] = np.max([p.max - st.expon.ppf(u[i]), 0])
            elif p.name == 'host_Av':
                x[i] = st.expon.ppf(u[i], scale=0.2)
            elif p.name == 'Rv':
                x[i] = st.norm.ppf(u[i], loc=2.71, scale=0.2)
            elif p.name == 'EBV':
                x[i] = st.expon.ppf(u[i], scale=0.1)
            elif 'alpha' in p.name:
                x[i] = st.norm.ppf(u[i], loc=0.0, scale=0.3)
            elif 'slope' in p.name:
                x[i] = st.norm.ppf(u[i], loc=0.0, scale=0.3)
            elif p.name == 'sigma':
                x[i] = st.norm.ppf(u[i], loc=0.2, scale=0.05)
            #elif 'norm' in p.name:
            #    x[i] = st.norm.ppf(u[i], loc=p.value, scale=1)
                #print(print(p.name, p.value, p.stderr, u[i], x[i]))
            else:
                x[i] = p.min + (p.max - p.min) * u[i]
                #print(p.name, p.min, p.max, u[i], x[i])

            x[i] = min(max(x[i], p.min), p.max)
            #print(p.name, i, x[i])
        #print([[p.name, x[i]] for i, p in enumerate(self.params.values())])
        #input()
        #print(x)
        #t.time('priors')
        return x
        #return lp -.5 * np.sum(np.power(self.ln_like_slope(pars), 2))

    def anneal_priors(self, params):
        prior = 100 * self.lum_prior(params, kind='anneal') + 50 * self.agn_host_prior(params)
        #print(prior)
        if all([p in params.keys() for p in ['bbb_slope', 'EBV']]):
            prior += 1 - 10 ** (-3 * params['bbb_slope'].value * params['EBV'].value * 3.1)
        #print(params['bbb_slope'].value, params['EBV'].value, prior)
        #print(self.lum_prior(params), self.agn_host_prior(params), params)
        return 100 * self.lum_prior(params, kind='anneal') + 50 * self.agn_host_prior(params)

    def agn_host_prior(self, params):
        host = self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(1400) * 10 ** params['host_norm'].value * self.extinction_MW(1400, Av=params['host_Av'].value)
        agn = self.models['bbb'].models[0].flux(1400) * 10 ** params['bbb_norm'].value * self.extinction(1400, params)
        #print(host, agn, - host / agn)
        return - host / agn

    def lum_prior(self, params, kind='mcmc'):
        alpha, tmin = -1.2, 0.0001
        C = (alpha + 1) / (gamma(alpha + 2) * (1 - gammainc(alpha + 2, tmin)) - (tmin) ** (alpha + 1) * np.exp(-tmin))
        M_UVs = -20.9 - 1.1 * (self.d['z'] - 1)
        #f = self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(1700) * 10 ** params['host_norm'].value * self.extinction(1700, Av=params['host_Av'].value)
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
                    prior -= .5 * ((p.value + 0.1) / 0.3) ** 6
                else:
                    prior -= .5 * ((p.value + 0.1) / 0.3) ** 6
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
                prior -= 1.5 * np.log(10 ** p.value) #+ ((p.value - 0) / 0.05) ** 2
            if 'alpha' in p.name:
                prior -= .5 * (p.value / params['sigma'].value) ** 2 + np.log(params['sigma'].value)
            if 'slope' in p.name:
                prior -= .5 * (p.value / 0.3) ** 2
            if p.name == 'sigma':
                prior -= .5 * ((p.value - 0.2) / 0.05) ** 2

            #print(p.name, prior)
        return prior

    def ln_like(self, params, plot=0, outliers=False):
        # print(params)
        #t = Timer()
        #print(len(self.sm[0]))
        chi = (self.model_emcee(params, self.sm[0], 'spec') - self.sm[1]) / self.sm[2]
        if outliers:
            chi = chi[np.abs(chi) < np.quantile(np.abs(chi), 0.95)]
        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.sm[0], chi)
            fig, ax = plt.subplots()
        #if np.sum(np.isnan(chi)):
        #    print(self.sm[0][np.isnan(chi)])
        #t.time('spec')
        for k, f in self.filters.items():
            if self.filters[k].fit:
                chi = np.append(chi, [f.weight / f.err * (f.value - f.get_value(x=f.x, y=self.model_emcee(params, f.x, k)))])
                #print(chi[-1])
                if plot:
                    print(k, [f.weight / f.err * (f.value - f.get_value(x=f.x, y=self.model_emcee(params, f.x, k)))])
                    ax.plot(np.mean(f.x), [f.weight / f.err * (f.value - f.get_value(x=f.x, y=self.model_emcee(params, f.x, k)))])
                #if np.isnan(chi[-1]):
                #    print(k)
                #t.time(f.name)
        if plot:
            plt.show()

        return chi

    def ln_like_slope(self, params, plot=0, outliers=False):
        # print(params)
        # t = Timer()
        #print('model:', self.model_emcee_slope(params, self.sm[0], 'spec').shape)
        chi = (self.model_emcee_slope(params, self.sm[0], 'spec') - self.sm[1][:, np.newaxis]) / self.sm[2][:, np.newaxis]
        #print('chi:', chi.shape)
        #if np.sum(np.isnan(chi)):
        #    print(self.sm[0][np.isnan(chi)])
        # t.time('spec')
        for k, f in self.filters.items():
            if self.filters[k].fit:
                chi = np.append(chi, [f.weight / f.err * (f.value - f.get_value(x=f.x, y=y)) for y in np.transpose(self.model_emcee_slope(params, f.x, k))])

        #print('chi:', chi.shape)
        norm = np.exp(-((np.arange(-0.6, 0.4, 10) + 0.1) / 0.3) ** 2)
        lnlike = np.log10(np.sum(np.exp(-chi) * np.exp(-((np.arange(-0.6, 0.4, 10) + 0.1) / 0.3) ** 2)[np.newaxis, :], axis=0) / norm)
        #if outliers:
        #    lnlike = lnlike[np.abs(lnlike) < np.quantile(np.abs(lnlike), 0.95)]

        return lnlike

    def set_params(self, x):
        params = copy(self.params)
        for p, v in zip(params.keys(), x):
            params[p].value = v
            #print(p, v, params[p].value)
        return params

    def prepare_params(self, params=None, tvary=True):

        new = True if params is None else False

        if new:
            norm_bbb = np.log10(np.nanmean(self.sm[1]) / np.nanmean(self.models['bbb'].data['spec']))
            params = lmfit.Parameters()
            params.add('bbb_norm', value=norm_bbb, min=-3, max=3)
            params.add('bbb_slope', value=0, min=-2, max=2)
            params.add('Fe_norm', value=0, min=-1, max=100, vary=True)
            params.add('EBV', value=0.0, min=0.0, max=10)
            params.add('Rv', value=2.74, min=0.5, max=6.0, vary=True)
            params.add('tor_type', value=10, vary=True, min=0, max=self.models['tor'].n - 1)
            params.add('tor_norm', value=np.log10(norm_bbb / np.max(self.models['bbb'].data['spec'][0]) * np.max(self.models['tor'].data['spec'][params['tor_type'].value])), min=-3, max=2)
            #params.add('host_tau', value=0.3, vary=True, min=self.models['gal'].tau[0].value, max=Planck15.age(self.d['z']).to('Gyr').value)
            #params.add('host_tg', value=Planck15.age(self.d['z']).to('Gyr').value / 2, vary=True, min=self.models['gal'].tg[0].value, max=min(10, Planck15.age(self.d['z']).to('Gyr').value))
            params.add('host_tau', value=np.log10(0.2), min=self.models['gal'].tau[0],
                       max=np.log10(Planck15.age(self.d['z']).to('Gyr').value), vary=True)
            params.add('host_tg', value=np.log10(min(Planck15.age(self.d['z']).to('Gyr').value, 10) / 2),
                       min=self.models['gal'].tg[0], max=min(np.log10(Planck15.age(self.d['z']).to('Gyr').value), self.models['gal'].tg[-1]), vary=True)
            print(params['host_tg'])
            # print('age:', np.log10(Planck15.age(self.d['z']).to('yr').value))
            params.add('host_norm', value=np.log10(np.nanmean(self.sm[1]) / np.nanmean(
                self.models['gal'].data['spec'][self.models['gal'].get_model_ind(params)])), min=-4, max=2)
            params.add('host_Av', value=0.1, min=0, max=10.0)

        if params['host_norm'].value < params['bbb_norm'].value - 2:
            params['host_norm'].value = params['bbb_norm'].value - 2

        # print(self.extinction(2500, Av=params['Av'].value), np.log(self.extinction(1000, Av=params['Av'].value) / self.extinction(2500, Av=params['Av'].value)) / np.log(0.4))
        if params['EBV'].value < 0:
            params['bbb_norm'].value -= np.log10(self.extinction(2500, params))
            params['bbb_slope'].value = np.log(self.extinction(1000, params) / self.extinction(2500, params)) / np.log(
                0.4)
            params['EBV'].value = 0.01

        cov_range = {'bbb_norm': [0.1, 0.1], 'bbb_slope': [0.1, 0.1],
                     'EBV': [0.01, 0.1], 'Rv': [0.2, 0.2],  # 'c3': [0.05, 0.3],
                     'tor_type': [1, 5], 'tor_norm': [0.1, 0.1],
                     'host_tau': [0.05, 0.1], 'host_tg': [0.1, 0.5],
                     'host_norm': [0.1, 0.1],
                     'host_Av': [0.01, 0.2]
                     }
        cov = {}
        if not new:
            params['host_Av'].max = 5
            # params['host_tau'].value = 0.1
            for k in cov_range.keys():
                params[k].vary = True
                if params[k].stderr is None:
                    params[k].stderr = np.median(cov_range[k])
                if params[k].vary == True:
                    cov[k] = max([cov_range[k][0], min([cov_range[k][1], params[k].stderr])])
        else:
            for k in cov_range.keys():
                cov[k] = np.median(cov_range[k])

        if 'Fe_norm' in params.keys():
            params['Fe_norm'].vary = True
            cov['Fe_norm'] = 0.2  # cov['bbb_norm'] / 3

        if 'Abump' in params.keys():
            params['Abump'].vary = True
            cov['Abump'] = 0.2

        # print(cov)
        # params.add('host_L', value=0, min=0, max=100, vary=False)
        # cov['host_L'] = 0.1

        if tvary:
            params.add('sigma', value=0.2, min=0.01, max=3)
            cov['sigma'] = 0.02
            # params.add('alpha_spec', value=0.0, min=-3, max=3)
            # cov['alpha_spec'] = params['sigma'].value / 10
            # print(self.filters)
            # print(self.photo.items())
            for p in set([v for k, v in self.photo.items() if self.filters[k].fit]):
                if p != 'WISE':
                    if np.sum([p == self.photo[k] and self.filters[k].fit for k in self.filters.keys()]) > 1:
                        params.add('alpha_' + p, value=0, min=-3, max=3)
                        params.add('slope_' + p, value=0, min=-3, max=3)
                        x, y = [], []
                        for k, f in self.filters.items():
                            if p == self.photo[k] and self.filters[k].fit:
                                x.append(np.log10(f.l_eff))
                                y.append(np.log10(f.get_value(x=f.x, y=self.model_emcee(params, f.x, k)) / f.value))
                        if len(x) > 1:
                            res = scipy.stats.linregress(x, y)
                            params['alpha_' + p].value = res.intercept
                            params['slope_' + p].value = res.slope
                            cov['alpha_' + p] = res.intercept_stderr if res.intercept_stderr != 0 else params[
                                'sigma'].value
                            cov['slope_' + p] = res.stderr if res.stderr != 0 else 0.02
                    else:
                        params.add('alpha_' + p, value=0, min=-3, max=3)
                        params.add('slope_' + p, value=0, min=-3, max=3)
                        for k, f in self.filters.items():
                            if p == self.photo[k] and self.filters[k].fit:
                                params['alpha_' + p].value = np.log10(
                                    f.get_value(x=f.x, y=self.model_emcee(params, f.x, k)) / f.value)
                        cov['alpha_' + p] = 0.2
                        cov['slope_' + p] = 0.02
                    # cov['alpha_' + p] = params['sigma'].value
                    # cov['slope_' + p] = 0.02

        # for p in set(self.photo.values()):
        #    if p != 'WISE':
        #        chi = []
        #        for k, f in self.filters.items():
        #            if self.photo[k] == p and self.filters[k].fit:
        #                chi.append([f.value, f.get_value(x=f.x, y=self.model_emcee(params, f.x, k))])
        #        if len(chi) > 0:
        #            params['alpha_' + p].value = -(np.sum(np.asarray(chi), axis=0)[0] - np.sum(np.asarray(chi), axis=0)[1]) / len(np.sum(np.asarray(chi), axis=0)) / 2.5
        # print(params)
        # print(cov)

        return params, cov

    def mcmc(self, params=None, stat=1, method='zeus', calc=1, nsteps=200):
        # import os
        # os.environ["OMP_NUM_THREADS"] = "1"
        # from multiprocessing import Pool

        #print(self.photo)

        params, cov = self.prepare_params(params=params)

        if calc:
            nwalkers, ndims = 100, len(params)

            pos = np.asarray([params[p].value + np.random.randn(nwalkers) * cov[p] for p in params])
            for k, p in enumerate(params.values()):
                pos[k][pos[k] <= p.min] = p.min + (p.max - p.min) * np.random.rand(sum(pos[k] <= p.min))
                pos[k][pos[k] >= p.max] = p.max - (p.max - p.min) * np.random.rand(sum(pos[k] >= p.max))
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
                    sampler = emcee.EnsembleSampler(nwalkers, ndims, self.fcn2min_mcmc, moves=[(emcee.moves.DEMove(), 0.3), (emcee.moves.StretchMove(), 0.7)])

                    subiters = 5
                    #try:
                    for i in range(subiters):
                        # We'll track how the average autocorrelation time estimate changes
                        autocorr = np.empty(self.mcmc_steps)

                        # This will be useful to testing convergence
                        index, old_tau = 0, np.inf

                        iterations = self.mcmc_steps // 10 if i < subiters - 1 else self.mcmc_steps
                        for sm in sampler.sample(pos, iterations=iterations, skip_initial_state_check=False, progress=(self.verbose == 1)):
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
                            inds = np.argwhere(lnL < np.quantile(lnL, 0.5) + 1.0 * (np.quantile(lnL, 0.5) - np.max(lnL)))
                            #print("st:", np.max(lnL), len(inds), len(lnL))
                            for i in range(50):
                                inds = np.argwhere(lnL < np.max(lnL) - (np.max(lnL) - np.quantile(lnL, 0.9)) * (1.0 + i * 0.2))
                                #print("st:", i, len(inds), len(lnL), np.max(lnL))
                                if len(inds) < 0.5 * len(lnL):
                                    break
                            #print(np.quantile(lnL, 0.7) + 1.0 * (np.quantile(lnL, 0.7) - np.max(lnL)), len(inds))
                            #print("st:", np.max(lnL), len(inds), len(lnL))
                            if len(inds) > 0:
                                mask = np.ones(lnL.shape, dtype=bool)
                                mask[inds] = False
                                #mpos = np.mean(pos[mask], axis=0)
                                mpos = pos[np.argmax(lnL)]
                                for ind in inds:
                                    pos[ind[0], :] = mpos + (mpos - pos[mask, :][np.random.randint(np.sum(mask))]) * (0.5 * np.random.random(len(mpos)))
                                    for k, p in enumerate(params.values()):
                                        pos[ind[0], k] = np.max([p.min, np.min([p.max, pos[ind[0], k]])])
                                    #if np.any(np.isnan(pos[ind[0], :])):
                                    #    print("OOOOOOOOO:", np.argmax(lnL), mpos, pos[mask, :])
                            #print(pos)
                    #except:
                    #    print(self.ind, i, len(inds))
                    #    print(mpos)
                    #    print(mask)
                    #    print(pos[mask, :][np.random.randint(np.sum(mask))])
                    #    print(0.5 * np.random.random(len(mpos)))
                    #    print(params, cov, pos)


                    thinning = int(0.5 * np.nanmax([2, np.nanmin(sampler.get_autocorr_time(tol=0))]))
                    #print(thinning)
                    burnin = int(2 * sampler.iteration // 3)
                    flat_sample = sampler.get_chain(discard=burnin, thin=thinning, flat=True)
                    ln_max = -np.max(sampler.get_log_prob(discard=burnin, thin=thinning, flat=True))
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

                    burnin = int(2 * sampler.iteration // 3)
                    flat_sample = sampler.get_chain(discard=burnin, thin=10, flat=True)
                    ln_max = -np.max(sampler.get_log_prob(discard=burnin, thin=10, flat=True))
            #self.calc_host_lum(result.flatchain)

            #print(sampler.get_log_prob(discard=sampler.iteration // 2, thin=thinning, flat=True))
            pars, flat_sample = list(params.valuesdict().keys()) + ['lnL'], np.append(flat_sample, sampler.get_log_prob(discard=burnin, thin=thinning, flat=True)[:, np.newaxis], axis=1)
            flat_sample = flat_sample[np.isfinite(flat_sample[:, -1]), :]
            # >>> saving flat_samples"
            if self.save:
                with open(os.path.dirname(self.catalog) + '/QC/chains/' + self.d['SDSS_NAME'] + '.pickle', 'wb') as f:
                    pickle.dump([pars, flat_sample], f)

        else:
            with open(os.path.dirname(self.catalog) + '/QC/chains/' + self.d['SDSS_NAME'] + '.pickle', 'rb') as f:
                pars, flat_sample = pickle.load(f)
                ln_max = np.min(flat_sample[:, -1])
                print('ln_max:', ln_max)

        #print(pars)
        #print(flat_sample.shape)
        #print(flat_sample)
        #print(flat_sample[:, 13])
        #print(flat_sample[:, 14])

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
                        print(p)
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
                                s.append(np.log10((self.d['L_UV'] + np.random.randn() * self.d['L_UV_err']) / self.extinction(2500, params)))
                                #print(self.d['L_UV'], np.log10(self.d['L_UV'] + np.random.randn() * self.d['L_UV_err']), self.extinction(2500, params), s[-1])
                            elif p == 'L_UV_corr':
                                sind = np.argmin(np.abs(self.sm[0] - 2500 * (1 + self.d['z'])))
                                scaling = (1e-17 * u.erg / u.cm ** 2 / u.AA / u.s).to(u.erg / u.cm ** 2 / u.s / u.Hz, equivalencies=u.spectral_density(2500 * u.AA * (1 + self.d['z']))).value
                                s.append(np.log10(self.models['bbb'].data['spec'][0][sind] * 10 ** params['bbb_norm'].value * 0.85 * scaling * 4 * np.pi * Planck15.luminosity_distance(self.d['z']).to('cm').value ** 2 / (1 + self.d['z'])))
                        print(s)
                        if np.sum(np.isfinite(s)) > 0:
                            d = distr1d(np.asarray(s)[np.isfinite(s)])
                        else:
                            d = None
                    print(p, d)
                    if d is not None:
                        d.dopoint()
                        d.dointerval()
                        res[p] = a(d.point, d.interval[1] - d.point, d.point - d.interval[0])
                        #print(p, res[p])
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

    def objective(self, best, anneal_pars, params):
        for i, (p, f) in enumerate(anneal_pars.items()):
            params[p].value = best[i]

        # print(params)
        minner = lmfit.Minimizer(self.fcn2min, params, nan_policy='propagate', calc_covar=True, max_nfev=100)
        result = minner.minimize(method='leastsq')
        # lmfit.report_fit(result)
        chi = self.fcn2min(result.params)
        # print(np.sum(chi ** 2) / (len(chi) - len(result.params)))
        # print(np.sum(chi ** 2), 100 * self.lum_prior(result.params, kind='anneal'), self.anneal_priors(result.params))
        return (np.sum(chi ** 2) - self.anneal_priors(
            result.params)), result  # / (len(chi) - len(result.params)), result

    def simulated_annealing(self, params, anneal_pars, n_iterations=100, temp=1000):
        # generate an initial point
        best = [f[0](params[p].value) for p, f in anneal_pars.items()]
        # evaluate the initial point
        best_eval, res = self.objective(best, anneal_pars, params)
        if self.verbose:
            print(best, best_eval)
        # current working solution
        curr, curr_eval = best, best_eval
        # run the algorithm
        for i in range(n_iterations):
            # if i > n_iterations / 2:
            #    params['bbb_slope'].vary = True
            # print(i)
            # take a step
            # candidate = curr + randn(len(bounds)) * step_size
            candidate = [f[0](params[p].value + np.random.randn() * f[1]) for p, f in anneal_pars.items()]
            candidate = [p.min * (c < p.min) + p.max * (c > p.max) + c * ((c >= p.min) * (c <= p.max)) for c, p in
                         zip(candidate, [params[p] for p in anneal_pars.keys()])]
            # evaluate candidate point
            candidate_eval, res = self.objective(candidate, anneal_pars, params)
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
            # print(diff, t, metropolis)
            # check if we should keep the new point
            if diff < 0 or np.random.rand() < metropolis:
                # store the new current point
                curr, curr_eval = candidate, candidate_eval
        # print('best:', best)
        return self.objective(best, anneal_pars, params)

    def anneal_fit(self, params=None, anneal_steps=None, slope=True, vary={}):

        if anneal_steps == None:
            anneal_steps = self.anneal_steps
        #print(self.models['gal'].tg)
        #print(self.models['gal'].tau)
        var = {'bbb_norm': True, 'bbb_slope': False, 'Fe_norm': False,
             'EBV': True, 'Rv': False, 'tor_type': False, 'tor_norm': True, 'host_type': True,
             'host_tau': True, 'host_tg': True, 'host_norm': True, 'host_Av': True,
             }
        for k, v in vary.items():
            var[k] = v
        if params is None:
            norm_bbb = np.log10(np.nanmean(self.sm[1]) / np.nanmean(self.models['bbb'].data['spec'][0]))
            params = lmfit.Parameters()
            params.add('bbb_norm', value=norm_bbb, min=-3, max=3, vary=var['bbb_norm'])
            params.add('bbb_slope', value=-0.1, min=-2, max=2, vary=var['bbb_slope'])
            params.add('Fe_norm', value=0, min=-1, max=100, vary=var['Fe_norm'])
            params.add('EBV', value=0.01, min=0.0, max=10, vary=var['EBV'])
            params.add('Rv', value=2.74, min=0.5, max=6.0, vary=var['Rv'])
            #if self.d['z'] > 0.70:
            #    params.add('Abump', value=0.01, min=0.0, max=100.0, vary=False)
            params.add('tor_type', value=10, min=0, max=self.models['tor'].n - 1, vary=var['tor_type'])
            params.add('tor_norm', value=norm_bbb - 2 + np.log10(self.models['bbb'].data['spec'][0][-1] * np.max(self.models['tor'].data['spec'][params['tor_type'].value])), min=-3, max=2, vary=var['tor_norm'])
            if 0:
                params.add('host_type', value=0, min=0, max=self.models['host'].n - 1, vary=var['host_type'])
            else:
                #print('age:', np.log10(Planck15.age(self.d['z']).to('yr').value))
                params.add('host_tau', value=np.log10(0.2), min=self.models['gal'].tau[0], max=np.log10(Planck15.age(self.d['z']).to('Gyr').value), vary=var['host_tau'])
                params.add('host_tg', value=np.log10(min(Planck15.age(self.d['z']).to('Gyr').value, 10) / 2), min=self.models['gal'].tg[0], max=min(np.log10(Planck15.age(self.d['z']).to('Gyr').value), self.models['gal'].tg[-1]), vary=var['host_tg'])
                print('anneal', params['host_tg'], self.models['gal'].tg[0], min(np.log10(Planck15.age(self.d['z']).to('Gyr').value), self.models['gal'].tg[-1]))
            norm_gal = np.log10(np.nanmean(self.sm[1]) / np.nanmean(self.models['gal'].data['spec'][233]))
            params.add('host_norm', value=norm_gal, min=-3, max=2, vary=var['host_norm'])
            params.add('host_Av', value=0.5, min=0, max=10.0, vary=var['host_Av'])

        anneal_pars = OrderedDict([('tor_type', [int, 5])]) #, ('host_tau', [float, 0.5])]) #, ('host_tg', [float, 0.5])])
        #self.ln_like(params, plot=1)

        chi2_min, result = self.simulated_annealing(params, anneal_pars, n_iterations=anneal_steps)
        #print(result.params)
        if self.verbose:
            print(chi2_min, lmfit.report_fit(result))
        return result, chi2_min

    def fit(self, ind=None, method='zeus', calc=True):

        self.ind = ind
        self.d = self.df.loc[ind]
        res = None

        if method == 'annealing' and self.hostExt and any([f in self.filters.keys() for f in ['J_UKIDSS', 'H_UKIDSS', 'K_UKIDSS', 'J', 'H', 'K', 'W1', 'W2']]) and any([f in self.filters.keys() for f in ['W3', 'W4']]):
            print('anneal:', ind, self.d['SDSS_NAME'])
            result, chi2_min = self.anneal_fit()
            host_min = self.models['gal'].get_model_ind(result.params)
            #print(result.params)
            if self.verbose:
                print(chi2_min, lmfit.report_fit(result))

            if self.plot:
                self.plot_spec(params=result.params)

        elif method in ['nested', 'nested_dyn', 'nautilus']:
            result, chi2_min = self.anneal_fit(anneal_steps=30, vary={'host_norm': True, 'host_tau': False, 'host_tg': False, 'host_Av': False})
            #result, chi2_min = self.anneal_fit(anneal_steps=30)
            params, cov = self.prepare_params(result.params)
            #print(params, cov)
            self.params = params
            parnames = [p for p in self.params]
            ndim = len([p for p in params.values() if p.vary])
            t = Timer()
            #from multiprocessing import Pool
            #num_proc = 10
            #pool = Pool(processes=num_proc)
            quantiles = [0.16, 0.5, 0.84]
            res = {}
            if method in ['nested', 'nested_dyn']:
                if calc:
                    if 'dyn' in method:
                        sampler = dynesty.DynamicNestedSampler(self.lnlike_nest, self.ptform, ndim, nlive=100, bound='multi') #, pool=pool, queue_size=num_proc) #, bound='balls')
                    else:
                        sampler = dynesty.NestedSampler(self.lnlike_nest, self.ptform, ndim, nlive=2000, bound='multi')
                    sampler.run_nested(maxiter=self.mcmc_steps, checkpoint_file=os.path.dirname(self.catalog) + '/QC/dynesty/' + self.d['SDSS_NAME'] + '.save', print_progress=True,
                                       dlogz_init=0.1, maxiter_init=20000) #, maxcall=500000)
                else:
                    sampler = dynesty.DynamicNestedSampler.restore(os.path.dirname(self.catalog) + '/QC/dynesty/' + self.d['SDSS_NAME'] + '.save')
                    # resume
                    sampler.run_nested(resume=True)
                results = sampler.results
                print(sampler.results)
                t.time(f'sample time for {self.ind}')

                # Plot a summary of the run.
                rfig, raxes = dyplot.runplot(sampler.results)
                rfig.savefig(os.path.dirname(self.catalog) + '/QC/dynesty_plots/' + self.d['SDSS_NAME'] + '_run.png')
                # Plot traces and 1-D marginalized posteriors.
                tfig, taxes = dyplot.traceplot(sampler.results, quantiles=quantiles, show_titles=True, labels=parnames)
                tfig.savefig(os.path.dirname(self.catalog) + '/QC/dynesty_plots/' + self.d['SDSS_NAME'] + '_trace.png')
                # Plot the 2-D marginalized posteriors.
                cfig, caxes = dyplot.cornerplot(sampler.results, quantiles=quantiles, show_titles=True, labels=parnames)
                cfig.savefig(os.path.dirname(self.catalog) + '/QC/dynesty_plots/' + self.d['SDSS_NAME'] + '_corner.png')
                #t.time('plot')
                sample = results.samples[:]
                weights = results.importance_weights()
            elif  method in ['nautilus']:
                sampler = Sampler(self.ptform, self.lnlike_nest, n_live=1000, n_dim=ndim, filepath=os.path.dirname(self.catalog) + '/QC/nautilus/' + self.d['SDSS_NAME'] + '.hdf5', resume=not calc) #, pool=20)
                if calc:
                    sampler.run(verbose=True)

                t.time(f'sample time for {self.ind}')

                sample, weights, log_l = sampler.posterior()
                weights = np.exp(weights)
                corner.corner(sample, weights=weights, bins=20, labels=parnames, color='purple', plot_datapoints=False, range=np.repeat(0.999, ndim))
            if 1:
                # >>> statistical determination:
                #print("calc stats for ", self.ind)
                #print(params)
                for i, p in enumerate(list(params.keys()) + ['M_UV', 'L_host', 'Av', 'L_UV_ext', 'L_UV_corr']):
                    if p not in ['M_UV', 'L_host', 'Av', 'L_UV_ext', 'L_UV_corr']:
                        d = dynesty.utils.quantile(sample[:, i], quantiles, weights=weights)
                    else:
                        #print(p)
                        s = []
                        for l in range(sample.shape[0]):
                            for j, p1 in enumerate(params.keys()):
                                params[p1].value = sample[l, j]
                            if p == 'L_host':
                                s.append(np.log10(self.calc_host_lum(params, wave='bol')))
                            elif p == 'M_UV':
                                s.append(-2.5 * np.log10(self.calc_host_lum(params, wave=1700) * 9.64e-13) + 51.6)
                            elif p == 'Av':
                                s.append(params['EBV'].value * params['Rv'].value)
                            elif p == 'L_UV_ext':
                                s.append(np.log10(
                                    (self.d['L_UV'] + np.random.randn() * self.d['L_UV_err']) / self.extinction(
                                        2500, params)))
                                # print(self.d['L_UV'], np.log10(self.d['L_UV'] + np.random.randn() * self.d['L_UV_err']), self.extinction(2500, params), s[-1])
                            elif p == 'L_UV_corr':
                                sind = np.argmin(np.abs(self.sm[0] - 2500 * (1 + self.d['z'])))
                                scaling = (1e-17 * u.erg / u.cm ** 2 / u.AA / u.s).to(
                                    u.erg / u.cm ** 2 / u.s / u.Hz,
                                    equivalencies=u.spectral_density(2500 * u.AA * (1 + self.d['z']))).value
                                s.append(np.log10(self.models['bbb'].data['spec'][0][sind] * 10 ** params[
                                    'bbb_norm'].value * 0.85 * scaling * 4 * np.pi * Planck15.luminosity_distance(
                                    self.d['z']).to('cm').value ** 2 / (1 + self.d['z'])))
                        if np.sum(np.isfinite(s)) > 0:
                            d = dynesty.utils.quantile(np.asarray(s)[np.isfinite(s)], quantiles, weights=weights[np.isfinite(s)])
                        else:
                            d = None
                    #print(p, d)
                    if d is not None:
                        res[p] = a(d[1], d[2] - d[1], d[1] - d[0])
            #t.time('stats')
            #print('res:', res)
            # >>> plot spectra
            if self.plot:
                #print("plot spectrum of ", self.ind)
                inds = np.random.randint(sample.shape[0], size=400)
                inds = np.where(weights > np.max(weights) / 2)[0]
                inds = inds[np.random.randint(len(inds), size=400)]
                total, bbb, tor, host = [], [], [], []
                s = {}
                for k in ['total', 'bbb', 'tor', 'host'] + list(self.filters.keys()):
                    s[k] = []
                for i in inds:
                    # params = result.params
                    for k, p in enumerate(params.keys()):
                        # print(ind, k, flat_sample[ind, k])
                        params[p].value = sample[i, k]
                        # if p in ['tor_type', 'host_tau', 'host_tg']:
                        #    params[p].value = np.max([params[p].min, np.min([params[p].max, round(params[p].value)])])
                    # self.ln_like(params, plot=plot)
                    # plot = 0
                    s['total'].append(self.model_emcee(params, self.spec[0], 'spec_full', mtype='total'))
                    s['bbb'].append(self.models['bbb'].models[0].y * 10 ** params['bbb_norm'].value * (
                                self.models['bbb'].models[0].x / 2500) ** params['bbb_slope'].value * self.extinction(
                        self.models['bbb'].models[0].x, params))
                    # s['bbb'].append(self.models['bbb'].models[0].y * params['bbb_norm'].value * (self.models['bbb'].models[0].x / 2500) ** (-0.1) * self.extinction(self.models['bbb'].models[0].x, params))

                    s['tor'].append(
                        self.models['tor'].models[self.models['tor'].get_model_ind(params)].y * 10 ** params[
                            'tor_norm'].value)
                    if self.hostExt:
                        s['host'].append(
                            self.models['gal'].models[self.models['gal'].get_model_ind(params)].y * 10 ** params[
                                'host_norm'].value * self.extinction_MW(
                                self.models['gal'].models[self.models['gal'].get_model_ind(params)].x,
                                Av=params['host_Av'].value))
                        # s['total'][-1] += self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(self.spec[0] / (1 + self.d['z'])) * 10 ** params['host_norm'].value * self.extinction(self.spec[0] / (1 + self.d['z']), Av=params['host_Av'].value)

                    # >>> filters fluxes:
                    for k, f in self.filters.items():
                        if self.filters[k].fit:
                            s[k].append(self.model_emcee(params, f.x, k, mtype='total'))

                self.fig.axes[0].fill_between(self.models['bbb'].models[0].x,
                                              *np.quantile(np.asarray(s['bbb']), [0.05, 0.95], axis=0), lw=1,
                                              color='tab:blue', zorder=3, label='bbb', alpha=0.5)
                self.fig.axes[0].fill_between(self.models['tor'].models[self.models['tor'].get_model_ind(params)].x,
                                              *np.quantile(np.asarray(s['tor']), [0.05, 0.95], axis=0), lw=1,
                                              color='tab:green', zorder=3, label='tor', alpha=0.5)
                # ax.plot(tor.x, tor.y * 10 ** params['tor_norm'].value, '--', color='tab:orange', zorder=2, label='composite', alpha=alpha)
                if self.hostExt:
                    self.fig.axes[0].fill_between(self.models['gal'].models[self.models['gal'].get_model_ind(params)].x,
                                                  *np.quantile(np.asarray(s['host']), [0.05, 0.95], axis=0), lw=1,
                                                  color='tab:purple', zorder=2, label='host galaxy', alpha=0.5)

                for k, f in self.filters.items():
                    if self.filters[k].fit:
                        # print(k, f.x / (1 + self.d['z']), *np.quantile(np.asarray(s[k]), [0.05, 0.95], axis=0))
                        self.fig.axes[0].fill_between(f.x / (1 + self.d['z']),
                                                      *np.quantile(np.asarray(s[k]), [0.05, 0.95], axis=0),
                                                      color=[c / 255 for c in f.color], lw=1, zorder=3, alpha=0.5)

                self.fig.axes[0].fill_between(self.spec[0] / (1 + self.d['z']),
                                              *np.quantile(np.asarray(s['total']), [0.05, 0.95], axis=0), lw=1,
                                              color='tab:red', zorder=3, label='total', alpha=0.5)

                if 1:
                    title = "id={0:4d} {1:19s} ({2:5d} {3:5d} {4:4d}) z={5:5.3f}".format(
                        ind, self.df.loc[ind, 'SDSS_NAME'], self.df.loc[ind, 'PLATE'], self.df.loc[ind, 'MJD'],
                        self.df.loc[ind, 'FIBERID'], self.df.loc[ind, 'z'])
                          # params['bbb_slope'].value
                    if 1:
                        title += " slope={0:s} EBV={1:s} chi2={2:4.2f}".format(str(res['bbb_slope']).replace('in log format', ''), str(res['EBV']).replace('in log format', ''), chi2_min)
                    if 0 and self.hostExt:
                        # title += " fgal={1:4.2f} {0:s}".format(self.models['host'].values[host_min], self.df['f_host' + '_photo' * self.addPhoto.isChecked()][i])
                        title += " fgal={2:4.2f} tau={0:4.2f} tg={1:4.2f}".format(self.models['gal'].values[host_min][0],
                                                                                  self.models['gal'].values[host_min][1],
                                                                                  self.df['f_host' + '_photo' * self.addPhoto][i])
                    self.fig.axes[0].set_title(title)

                if self.save:
                    self.fig.savefig(os.path.dirname(self.catalog) + '/QC/dynesty_plots/' + self.df.loc[ind, 'SDSS_NAME'] + '_spec.png', bbox_inches='tight', pad_inches=0.1)

                # print(self.fig)
                # self.fig.show()
                # plt.close(self.fig)
            t.time('plot spectrum')
            return res

        elif method in ['nautilus']:
            result, chi2_min = self.anneal_fit(anneal_steps=30, vary={'host_norm': True, 'host_tau': False, 'host_tg': False, 'host_Av': False})
            # result, chi2_min = self.anneal_fit(anneal_steps=30)
            params, cov = self.prepare_params(result.params)
            # print(params, cov)
            self.params = params
            parnames = [p for p in self.params]
            ndim = len([p for p in params.values() if p.vary])
            t = Timer()

            sampler = Sampler(self.ptform, self.lnlike_nest, n_live=1000, n_dim=ndim, filepath=os.path.dirname(self.catalog) + '/QC/nautilus/' + self.d['SDSS_NAME'] + '.hdf5', resume=not calc)
            if calc:
                sampler.run(verbose=True)

            t.time(f'sample time for {self.ind}')

            points, log_w, log_l = sampler.posterior()
            corner.corner(points, weights=np.exp(log_w), bins=20, labels=parnames, color='purple',
                plot_datapoints=False, range=np.repeat(0.999, ndim))


        elif method in ['emcee', 'zeus'] and self.hostExt and any([f in self.filters.keys() for f in ['J_UKIDSS', 'H_UKIDSS', 'K_UKIDSS', 'J', 'H', 'K', 'W1', 'W2']]): #and any([f in self.filters.keys() for f in ['W3', 'W4']]):
            if calc:
                result, chi2_min = self.anneal_fit(vary={'host_norm': True, 'host_tau': False, 'host_tg': False, 'host_Av': False})
                flat_sample, chi2_min, params, res = self.mcmc(params=result.params, method=method, nsteps=self.mcmc_steps)
            else:
                result, chi2_min = self.anneal_fit(anneal_steps=1)
                flat_sample, chi2_min, params, res = self.mcmc(params=result.params, method=method, nsteps=self.mcmc_steps, calc=calc)
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
                    s['bbb'].append(self.models['bbb'].models[0].y * 10 ** params['bbb_norm'].value * (self.models['bbb'].models[0].x / 2500) ** params['bbb_slope'].value * self.extinction(self.models['bbb'].models[0].x, params))
                    #s['bbb'].append(self.models['bbb'].models[0].y * params['bbb_norm'].value * (self.models['bbb'].models[0].x / 2500) ** (-0.1) * self.extinction(self.models['bbb'].models[0].x, params))

                    s['tor'].append(self.models['tor'].models[self.models['tor'].get_model_ind(params)].y * 10 ** params['tor_norm'].value)
                    if self.hostExt:
                        s['host'].append(self.models['gal'].models[self.models['gal'].get_model_ind(params)].y * 10 ** params['host_norm'].value * self.extinction_MW(self.models['gal'].models[self.models['gal'].get_model_ind(params)].x, Av=params['host_Av'].value))
                        #s['total'][-1] += self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(self.spec[0] / (1 + self.d['z'])) * 10 ** params['host_norm'].value * self.extinction(self.spec[0] / (1 + self.d['z']), Av=params['host_Av'].value)

                    # >>> filters fluxes:
                    for k, f in self.filters.items():
                        if self.filters[k].fit:
                            s[k].append(self.model_emcee(params, f.x, k, mtype='total'))

                self.fig.axes[0].fill_between(self.models['bbb'].models[0].x, *np.quantile(np.asarray(s['bbb']), [0.05, 0.95], axis=0), lw=1, color='tab:blue', zorder=3, label='bbb', alpha=0.5)
                self.fig.axes[0].fill_between(self.models['tor'].models[self.models['tor'].get_model_ind(params)].x, *np.quantile(np.asarray(s['tor']), [0.05, 0.95], axis=0), lw=1, color='tab:green', zorder=3, label='tor', alpha=0.5)
                #ax.plot(tor.x, tor.y * 10 ** params['tor_norm'].value, '--', color='tab:orange', zorder=2, label='composite', alpha=alpha)
                if self.hostExt:
                    self.fig.axes[0].fill_between(self.models['gal'].models[self.models['gal'].get_model_ind(params)].x, *np.quantile(np.asarray(s['host']), [0.05, 0.95], axis=0), lw=1, color='tab:purple', zorder=2, label='host galaxy', alpha=0.5)

                for k, f in self.filters.items():
                    if self.filters[k].fit:
                        #print(k, f.x / (1 + self.d['z']), *np.quantile(np.asarray(s[k]), [0.05, 0.95], axis=0))
                        self.fig.axes[0].fill_between(f.x / (1 + self.d['z']), *np.quantile(np.asarray(s[k]), [0.05, 0.95], axis=0), color=[c / 255 for c in f.color], lw=1, zorder=3, alpha=0.5)

                self.fig.axes[0].fill_between(self.spec[0] / (1 + self.d['z']), *np.quantile(np.asarray(s['total']), [0.05, 0.95], axis=0), lw=1, color='tab:red', zorder=3, label='total', alpha=0.5)

                print(ind, self.df.loc[ind, 'PLATE'], self.df.loc[ind, 'MJD'], self.df.loc[ind, 'FIBERID'])
                title = "id={0:4d} {1:19s} ({2:5d} {3:5d} {4:4d}) z={5:5.3f} slope={6:s} EBV={7:s} chi2={8:4.2f}".format(ind, self.df.loc[ind, 'SDSS_NAME'], self.df.loc[ind, 'PLATE'], self.df.loc[ind, 'MJD'], self.df.loc[ind, 'FIBERID'], self.df.loc[ind, 'z'], str(res['bbb_slope']).replace('in log format', ''), str(res['EBV']).replace('in log format', ''), chi2_min) #params['bbb_slope'].value
                if 0 and self.hostExt:
                    # title += " fgal={1:4.2f} {0:s}".format(self.models['host'].values[host_min], self.df['f_host' + '_photo' * self.addPhoto.isChecked()][i])
                    title += " fgal={2:4.2f} tau={0:4.2f} tg={1:4.2f}".format(self.models['gal'].values[host_min][0], self.models['gal'].values[host_min][1], self.df['f_host' + '_photo' * self.addPhoto][i])
                self.fig.axes[0].set_title(title)

                if self.save:
                    self.fig.savefig(os.path.dirname(self.catalog) + '/QC/plots/' + self.df.loc[ind, 'SDSS_NAME'] + '_spec.png', bbox_inches='tight', pad_inches=0.1)

                #print(self.fig)
                #self.fig.show()
                plt.show()
                #plt.close(self.fig)

        return res

    def plot_spec(self, params=None, fig=None, alpha=1):
        if fig is None or self.fig is None:
            self.fig, ax = plt.subplots(figsize=(20, 12))
        else:
            ax = self.fig.axes[0]

        if self.spec is not None and params is None:
            ax.plot(self.spec[0] / (1 + self.d['z']), self.spec[1], '-k', lw=2, zorder=2, label='spectrum')
            for k, f in self.filters.items():
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
                ax.plot(bbb.x, bbb.y * 10 ** params['bbb_norm'].value, '--', color='tab:blue', zorder=2, label='composite', alpha=alpha)
            ax.plot(bbb.x, bbb.y * 10 ** params['bbb_norm'].value * self.extinction(bbb.x, params),
                    '-', color='tab:blue', zorder=3, label='comp with ext', alpha=alpha)
            ax.plot(tor.x, tor.y * 10 ** params['tor_norm'].value, '--', color='tab:orange', zorder=2, label='composite', alpha=alpha)
            if self.hostExt:
                if alpha == 1:
                    ax.plot(host.x, host.y * 10 ** params['host_norm'].value, '--', color='tab:purple', zorder=2, label='host galaxy', alpha=alpha)

                ax.plot(host.x, host.y * 10 ** params['host_norm'].value * self.extinction_MW(host.x, Av=params['host_Av'].value), '-', color='tab:purple', zorder=2, label='host galaxy', alpha=alpha)

            # >>> plot filters fluxes:
            for k, f in self.filters.items():
                temp = bbb.flux(f.x / (1 + self.d['z'])) * self.extinction(f.x / (1 + self.d['z']), params) * 10 ** params['bbb_norm'].value + tor.flux(f.x / (1 + self.d['z'])) * 10 ** params['tor_norm'].value
                if self.hostExt:
                    temp += host.flux(f.x / (1 + self.d['z'])) * 10 ** params['host_norm'].value * self.extinction_MW(f.x / (1 + self.d['z']), Av=params['host_Av'].value)
                ax.plot(f.x / (1 + self.d['z']), temp, '-', color=[c / 255 for c in f.color], lw=2, zorder=3, alpha=alpha)
                # ax.scatter(f.filter.l_eff, f.filter.get_value(x=f.x, y=temp * self.extinction(f.x * (1 + self.d['z']), Av=params['Av'].value) * params['norm'].value),
                #           s=20, marker='o', c=[c/255 for c in f.filter.color])

            # >>> total profile:
            temp = bbb.flux(self.spec[0] / (1 + self.d['z'])) * 10 ** params['bbb_norm'].value * self.extinction(self.spec[0] / (1 + self.d['z']), params) + tor.flux(self.spec[0] / (1 + self.d['z'])) * 10 ** params['tor_norm'].value
            if self.hostExt:
                temp += host.flux(self.spec[0] / (1 + self.d['z'])) * 10 ** params['host_norm'].value * self.extinction_MW(self.spec[0] / (1 + self.d['z']), Av=params['host_Av'].value)

            ax.plot(self.spec[0] / (1 + self.d['z']), temp, '-', lw=2, color='tab:red', zorder=3, label='total profile', alpha=alpha)
            # print(np.sum(((temp - spec[1]) / spec[2])[mask] ** 2) / np.sum(mask))

    def plot_spec_nest(self, results, fig=None, alpha=1):
        if fig is None or self.fig is None:
            self.fig, ax = plt.subplots(figsize=(20, 12))
        else:
            ax = self.fig.axes[0]

        host_min = params['host_tau'].value * (params['host_tg'].max + 1) + params['host_tg'].value
        bbb, tor, host = self.models['bbb'].models[0], self.models['tor'].models[params['tor_type'].value], self.models['gal'].models[host_min]

        # >>> plot templates:
        if alpha == 1:
            ax.plot(bbb.x, bbb.y * 10 ** params['bbb_norm'].value, '--', color='tab:blue', zorder=2, label='composite', alpha=alpha)
        ax.plot(bbb.x, bbb.y * 10 ** params['bbb_norm'].value * self.extinction(bbb.x, params),
                '-', color='tab:blue', zorder=3, label='comp with ext', alpha=alpha)
        ax.plot(tor.x, tor.y * 10 ** params['tor_norm'].value, '--', color='tab:orange', zorder=2, label='composite', alpha=alpha)
        if self.hostExt:
            if alpha == 1:
                ax.plot(host.x, host.y * 10 ** params['host_norm'].value, '--', color='tab:purple', zorder=2, label='host galaxy', alpha=alpha)

            ax.plot(host.x, host.y * 10 ** params['host_norm'].value * self.extinction_MW(host.x, Av=params['host_Av'].value), '-', color='tab:purple', zorder=2, label='host galaxy', alpha=alpha)

        # >>> plot filters fluxes:
        for k, f in self.filters.items():
            temp = bbb.flux(f.x / (1 + self.d['z'])) * self.extinction(f.x / (1 + self.d['z']), params) * 10 ** params['bbb_norm'].value + tor.flux(f.x / (1 + self.d['z'])) * 10 ** params['tor_norm'].value
            if self.hostExt:
                temp += host.flux(f.x / (1 + self.d['z'])) * 10 ** params['host_norm'].value * self.extinction_MW(f.x / (1 + self.d['z']), Av=params['host_Av'].value)
            ax.plot(f.x / (1 + self.d['z']), temp, '-', color=[c / 255 for c in f.color], lw=2, zorder=3, alpha=alpha)
            # ax.scatter(f.filter.l_eff, f.filter.get_value(x=f.x, y=temp * self.extinction(f.x * (1 + self.d['z']), Av=params['Av'].value) * params['norm'].value),
            #           s=20, marker='o', c=[c/255 for c in f.filter.color])

        # >>> total profile:
        temp = bbb.flux(self.spec[0] / (1 + self.d['z'])) * 10 ** params['bbb_norm'].value * self.extinction(self.spec[0] / (1 + self.d['z']), params) + tor.flux(self.spec[0] / (1 + self.d['z'])) * 10 ** params['tor_norm'].value
        if self.hostExt:
            temp += host.flux(self.spec[0] / (1 + self.d['z'])) * 10 ** params['host_norm'].value * self.extinction_MW(self.spec[0] / (1 + self.d['z']), Av=params['host_Av'].value)

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

    def calc_mask(self, spec, z_em=0, iter=3, window=7, clip=3.0, exp_pixel=1):
        if 0:
            mask = np.logical_not(self.sdss_mask(spec[3]))
        else:
            mask = spec[3]
        #print(np.sum(mask))

        mask *= spec[0] > 1280 * (1 + z_em)

        for i in range(iter):
            m = np.zeros_like(spec[0])
            if window > 0 and np.sum(mask) > window:
                if i > 0:
                    m[mask] = np.abs(sm - spec[1][mask]) / spec[2][mask] > clip
                    mask *= np.logical_not(self.expand_mask(m, exp_pixel=exp_pixel))
                    #mask[mask] *= np.abs(sm - spec[1][mask]) / spec[2][mask] < clip
                sm = smooth(spec[1][mask], window_len=window, window='hanning', mode='same')

        mask = np.logical_not(self.expand_mask(np.logical_not(mask), exp_pixel=exp_pixel))

        mask[mask] *= (spec[1][mask] > 0) * (spec[1][mask] / spec[2][mask] > 0.5)
        
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
                       [2690, 2880], #[2940, 2990], [3280, 3330],
                       [3820, 3920], [4240, 4440],
                       [4920, 5080], [4720, 5080],
                       [5130, 5400], [5500, 5620], [5780, 6020],
                       [6300, 6850], [7600, 8050], [8250, 8300], [8400, 8600], [9000, 9400], [9500, 9700],
                       [9950, 10200]]
            # only strongest ones
            #windows = [[1500, 1600], [1840, 1960], [2760, 2860], [4920, 5080],  # [4780, 5080],
            #           [6300, 6850]]
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

def worker_wrapper(arg):
    ind, catfile, mcmc_steps, anneal_steps = arg
    return run_model(ind, catfile=catfile, mcmc_steps=mcmc_steps, anneal_steps=anneal_steps, method='nautilus')

def run_model(ind, catfile=None, mcmc_steps=10000, anneal_steps=300, method='emcee', calc=1):
    print(ind)

    qso = QSOSEDfit(catalog=catfile, plot=1, mcmc_steps=mcmc_steps, anneal_steps=anneal_steps, save=1, corr=50, verbose=0)
    if qso.prepare(ind):
        res = qso.fit(ind, method=method, calc=calc)
    else:
        res = None
    return (int(ind), res)
        
if __name__ == "__main__":
    # necessary to add cwd to path when script run 
    # by slurm (since it executes a copy)
    sys.path.append(os.getcwd())

    #catfile = '/home/balashev/science/Erosita/match2_DR14Q_add.csv'
    catfile = 'C:/science/Erosita/UV_Xray/match2_DR14Q_add.csv'
    #catfile = 'C:/science/Erosita/UV_Xray/individual/individual_targets_add.csv'
    path = os.path.dirname(catfile)
    #print(path)

    try:
        i1, i2 = int(sys.argv[1]), int(sys.argv[2])
    except:
        i1, i2 = 1, 1

    if 0:
        ind = 5
        if 0:
            res = run_model(ind, catfile=catfile, mcmc_steps=500, anneal_steps=50, method='emcee')
        else:
            res = run_model(ind, catfile=catfile, method='nautilus', calc=1) # method='nested_dyn') #
        print(res)
    else:
        #pars = ['bbb_norm', 'Av', 'tor_type', 'tor_norm', 'host_tau', 'host_tg', 'host_norm', 'host_Av', 'sigma', 'alpha_GALEX', 'alpha_SDSS', 'alpha_2MASS', 'alpha_UKIDSS']
        num = 10
        calc = 1
        if calc:
            if 1:
                for i in range(1): # range(7975 // num + 1):
                    if i % i2 + 1 == i1:
                        with Pool(num) as p:
                            res_new = p.map(worker_wrapper, [(k, catfile, 10000, 300) for k in np.arange(i * num, min((i + 1) * num, 7975))]) #total number of AGNs 7975
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
