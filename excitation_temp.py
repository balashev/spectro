#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import corner
import emcee
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import os
from scipy.optimize import minimize
#import sys
#sys.path.append('C:/science/spectro/')
from a_unc import a
from stats import distr1d, distr2d

def column(matrix, attr):
    return np.asarray([getattr(row,attr) for row in matrix])

class ExcitationTemp():
    """
    class for fitting population of levels by single excitation temperature
                can be used for H2, CI, CO et al. molecules
    parameters:
        - species :    <str>
                         name of species
        - n       :    list of <a_unc> type objects
                         column densities on levels
                         
    example of usage:
        n = []
        species = 'H2'
        n.append(a(20.71, 0.02, 0.02))
        n.append(a(21.06, 0.02, 0.02))
        
        Temp = ExcitationTemp(species)
        Temp.calcTemp(n, plot=1, verbose=1)
        Temp.latex()
    """
    def __init__(self, species, n, debug=False):
        self.species = species
        self.n = copy.deepcopy(n)
        self.num = len(self.n)
        self.debug = debug
        self.sampler = None

        self.set_data()
        self.set_ratio()

    def set_data(self, num=None):
        """
        set atomic data
        """

        if num is not None:
            self.num = num

        if self.species == 'H2':
            
            self.g = np.array([(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(self.num)])

            #data = np.genfromtxt('data/H2/energy_X.dat', comments='#', unpack=True)
            data = np.array([0, 118.5, 354.35, 705.54])
            # transform energy from cm^-1 to Kelvins
            self.E = data[:self.num] / 0.695

        if self.species == 'CI':
            self.g = np.array([2*i+1 for i in range(self.num)])
            data = [0, 16.42, 43.41] # energy in Kelvins
            self.E = np.array(data[:self.num])
        
        if self.species == 'CO':
            self.g = np.array([2*i+1 for i in range(self.num)])
            self.E = np.arange(self.num) * (np.arange(self.num) + 1) * 1.93128087 / 0.695 - np.power(np.arange(self.num) * (np.arange(self.num) + 1), 2) * 6.12147e-6 / 0.695
            print('E', self.E)

        if self.species == 'FeII':
            self.g = np.asarray([9/2, 7/2, 5/2, 3/2, 10.695/2, 9/2, 7/2, 5/2, 3/2, 7/2, 5/2, 3/2, 1/2])[:num]
            self.E = np.asarray([0, 384.7872, 667.6829, 862.6118, 977.0498, 1872.5998, 2430.1369, 2837.9807, 3117.4877, 7955.3186, 8391.9554, 8680.4706, 8846.7837])
            # transform energy from cm^-1 to Kelvins
            self.E = self.E[:self.num] / 0.695

        if self.debug:
            print('stat weights:', self.g)
            print('Energies, K:', self.E)
    
        # apply log10(e) correction fro linear 
        # E = [e*np.log10(np.exp(1)) for e in E]

    def set_ratio(self):
        """
        set column densities minus stat weight
        """
        self.y = copy.deepcopy(self.n)

        for i in range(self.num):
            self.y[i].val -= np.log10(self.g[i])

    def Z(self, temp=None):
        """
        return stat. sum
        """
        if temp is None:
            if isinstance(self.temp, a):
                return sum(self.g * np.exp(-self.E / self.temp.val))
            if isinstance(self.temp, (int, float)):
                return sum(self.g * np.exp(-self.E / self.temp))
        else:
            return sum(self.g * np.exp(-self.E / temp))

    def model(self, theta=None):
        #print(theta, [self.ntot + np.log10(g / self.Z(self.temp) * np.exp(- e / self.temp)) for e, g in zip(self.E, self.g)])
        return [self.ntot + np.log10(g / self.Z(self.temp) * np.exp(- e / self.temp)) for e, g in zip(self.E, self.g)]

    def model_curve(self, x, ntot, temp):
        return [ntot + np.log10(g / self.Z(temp) * np.exp(- e / temp)) for e, g in zip(x, self.g)]

    def set_pars(self, theta):
        self.ntot = theta[0]
        self.temp = theta[1]

    def log_prior(self, theta):
        self.set_pars(theta)
        lp = 0. if 0 < self.ntot < np.inf else -np.inf
        lp += 0. if 0 < self.temp < np.inf else -np.inf
        return lp

    def log_like(self, theta):
        #print(theta, np.sum([n.lnL(m) for n, m in zip(self.n, self.model(theta))]))
        return np.sum([n.lnL(m) for n, m in zip(self.n, self.model(theta))])

    def log_prob(self, theta):
        lp = self.log_prior(theta)
        if np.isfinite(lp):
            lp += self.log_like(theta)
        return lp, *self.model(theta)

    def calc(self, method='freq', plot=True, verbose=True):
        """
        estimate excitation temperature for given column density
        parameters:
            - n           : column densities on levels
            - method       : type of calculation, can be from ['freq', 'bayes']
                                    - freq   : frequentist approach. Likelihood minimization
                                    - emcee  : Bayessian approach. emcee Affine-invariant posterior sampling
            - plot        : plot some results
        return:
            None
        """

        self.verbose = verbose

        if self.verbose:
            print('Excitation temperature estimate from column densitites')
            print('species:', self.species)
            print('number of levels:', self.num)
            print('excitation energies:', self.E)
            print('stat. weights:', self.g)
            print('N, cm^-2:', self.n)
            print('N tot, cm^-2:', sum(self.n))

        if method == 'freq':
            self.ntot = sum(self.n).log().val
            self.temp = -(self.E[1] - self.E[0]) / np.log(self.n[1].dec().val / self.n[0].dec().val * self.g[0] / self.g[1])
            print(self.ntot, self.temp)
            self.ntot, self.temp = self.ntot - 0.3, self.temp + 10
            print([n.log().val for n in self.n], [(n.log().plus + n.log().minus) / 2 for n in self.n])
            popt, pcov = opt.curve_fit(self.model_curve, xdata=self.E, ydata=[n.log().val for n in self.n], p0=[self.ntot, self.temp], sigma=[(n.log().plus + n.log().minus) / 2 for n in self.n])

            print(popt, pcov)
            if np.any(np.isinf(pcov)):
                print("number of parameters is equal to number of variables. The covariance was not constrained...")

            self.res = {'ntot': a(popt[0], np.sqrt(pcov[0, 0]), np.sqrt(pcov[0, 0]), 'd'), 'temp': a(popt[1], np.sqrt(pcov[1, 1]), np.sqrt(pcov[1, 1]), 'd')}
            print(self.res)

        if method == 'emcee':
            ndim, nwalkers, nsteps = 2, 10, 5000
            self.calc(method='freq')

            print("mins:", np.min([(n.minus + n.plus) for n in self.n]))
            sigma = [(self.res['ntot'].plus + self.res['ntot'].minus) / 2 if np.isfinite(self.res['ntot'].plus + self.res['ntot'].minus) else np.min([(n.minus + n.plus) for n in self.n]),
                     (self.res['temp'].plus + self.res['temp'].minus) / 2 if np.isfinite(self.res['temp'].plus + self.res['temp'].minus) else self.res['temp'].val / 10]

            print(sigma)
            start = np.c_[self.res['ntot'].val + np.random.randn(nwalkers) * sigma[0], self.res['temp'].val + np.random.randn(nwalkers) * sigma[1]]
            print(start)

            if 1:
                self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob)
                #self.sampler.run_mcmc(start, nsteps, progress=True)
                # We'll track how the average autocorrelation time estimate changes
                max_n = 100000
                ind = 0
                autocorr = np.empty(max_n)
                old_tau = np.inf

                # Now we'll sample for up to max_n steps
                for sample in self.sampler.sample(start, iterations=max_n, progress=True):
                    # Only check convergence every 100 steps
                    #print(self.sampler.iteration)
                    if self.sampler.iteration % 200:
                        continue

                    print(self.sampler.iteration)
                    # Compute the autocorrelation time so far
                    # Using tol=0 means that we'll always get an estimate even
                    # if it isn't trustworthy
                    tau = self.sampler.get_autocorr_time(tol=0)
                    autocorr[ind] = np.mean(tau)
                    ind += 1
                    #print(self.sampler.iteration, tau)

                    # Check convergence
                    converged = np.all(tau * 200 < self.sampler.iteration)
                    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                    if converged:
                        break
                    old_tau = tau

                tau = self.sampler.get_autocorr_time()
                print(tau)
                if all(np.isfinite(tau)):
                    burnin, thin = int(10 * np.max(tau)), int(0.5 * np.min(tau))

                    samples = self.sampler.get_chain(discard=burnin, flat=True, thin=thin)

                    for i, l in enumerate(['ntot', 'temp']):
                        d = distr1d(samples[:, i])
                        print(d.stats(latex=2))
                        #q = np.quantile(samples[:, i], [0.1585, 0.50, 0.8415])
                        #print(l + ": ${0:.{n}f}^{{+{1:.{n}f}}}_{{-{2:.{n}f}}}$".format(q[1], q[2] - q[1], q[1] - q[0], n=2))
                        self.res[l] = a(d.stats(latex=4).split()[2])
                    self.temp, self.ntot = self.res['temp'].val, self.res['ntot'].val

                    log_prob_samples = self.sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
                    cols = self.sampler.get_blobs(discard=burnin, flat=True, thin=thin)
                    for i in range(self.num):
                        d = distr1d(cols[:, i])
                        print(d.stats(latex=4))
                        #q = np.quantile(samples[:, i], [0.1585, 0.50, 0.8415])
                        #print(l + ": ${0:.{n}f}^{{+{1:.{n}f}}}_{{-{2:.{n}f}}}$".format(q[1], q[2] - q[1], q[1] - q[0], n=2))
                        self.res[f'col_{i}'] = a(d.stats(latex=4).split()[2])

                    print(self.res)

                    print("burn-in: {0}".format(burnin))
                    print("thin: {0}".format(thin))
                    print("flat chain shape: {0}".format(samples.shape))

                    all_samples = np.concatenate((np.concatenate((samples, log_prob_samples[:, None]), axis=1), cols), axis=1)

                    print(all_samples.shape)

                    labels = ["logNtot", "Texc", "log prob"] + [f"logN_{i}" for i in range(self.num)]

                    if plot:
                        fig = corner.corner(all_samples, labels=labels, quantiles=[0.1585, 0.5, 0.8415], show_titles=True,)
                        # Extract the axes
                        axes = np.array(fig.axes).reshape((all_samples.shape[1], all_samples.shape[1]))

                        # Loop over the diagonal
                        for i in range(ndim+1, all_samples.shape[1]):
                            ax = axes[i, i]
                            ax.axvline(self.n[i - ndim - 1].val, color="tomato")
                            ax.axvspan(self.n[i - ndim - 1].val - self.n[i - ndim - 1].minus, self.n[i - ndim - 1].val + self.n[i - ndim - 1].plus, color="tomato", alpha=0.3)

    def slope_to_temp(self, slope=None, zero=None):
        """
        transform slope and zero point to temp and total column density
        """
        print('slope_to_temp:', slope)
        if slope is None:
            self.temp = -np.log10(np.exp(1)) / self.slope
            self.Ntot = self.zero - np.log10(self.g[0]) + np.log10(self.Z())
        else:
            temp = -np.log10(np.exp(1)) / slope
            print(temp)
            print(self.Z(temp=temp))
            Ntot = zero - np.log10(self.g[0]) + np.log10(self.Z(temp=temp))
            return temp, Ntot
        
    def temp_to_slope(self, temp=None, Ntot=None):
        """
        transform slope and zero point to temp and total column density
        """
        if temp is None:
            self.slope = -np.log10(np.exp(1))/self.temp
            self.zero = self.ntot + np.log10(self.g[0]) - np.log10(self.Z())
        else:
            slope = -np.log10(np.exp(1))/temp
            zero = Ntot + np.log10(self.g[0]) - np.log10(self.Z(temp))
            return slope, zero

    def plot(self, ax=None, color='tomato', energy='K'):
        if ax is None:
            fig = plt.figure(figsize=(14, 7))
            self.dataplot = fig.add_subplot(121)
            self.regionplot = fig.add_subplot(122)

        self.plot_data(ax=self.dataplot if ax is None else ax, color=color, enegry=energy)
        self.plot_temp(ax=self.dataplot if ax is None else ax, color=color, enegry=energy)

        if ax is None:
            self.plot_post(ax=self.regionplot if ax is None else ax)

    def plot_data(self, ax=None, color='tomato', markersize=8, label=None, energy='K'):
        E = self.E if energy == 'K' else self.E * 0.695
        ax.errorbar(E, column(self.n,'val') - np.log10(self.g), yerr=[column(self.n,'plus'), column(self.n,'minus')],
                               fmt='o', color=color , ecolor='k', linewidth=0.5, markersize=markersize, label=label)
        ax.set_xlim(E[0] - 10, E[-1] + 20)

    def plot_temp(self, color='tomato', ax=None, label=None, energy='K'):
        E = self.E if energy == 'K' else self.E * 0.695

        if 'temp' in self.res.keys():
            if isinstance(self.res['temp'], a):
                self.ntot, self.temp = self.res['ntot'].val, self.res['temp'].val
            else:
                self.ntot, self.temp = self.res['ntot'], self.res['temp']
        ax.plot(E, self.model([self.ntot, self.temp]) - np.log10(self.g), '--', color=color, lw=2)

        if len(self.res.keys()) > 2:
            d = [[self.res[f"col_{i}"].val + self.res[f"col_{i}"].plus - np.log10(self.g[i]) for i in range(self.num)], [self.res[f"col_{i}"].val - self.res[f"col_{i}"].minus - np.log10(self.g[i]) for i in range(self.num)]]
            ax.fill_between(E, d[0], d[1], color=color, alpha=0.3, label=label)

        #print(self.E[-1] * 0.8, self.ntot + np.log10(1 / self.g[-1] * np.exp(-self.E[-1] * 0.8 / self.temp)) + 0.5)
        if np.all([np.isfinite(getattr(self.res['temp'], attr)) for attr in ['val', 'plus', 'minus']]):
            ax.text(E[-1] * 0.7, self.ntot + np.log10(1 / self.g[-1] * np.exp(-self.E[-1] * 0.7 / self.temp)) + 0.8, r"$T_{\rm exc} = " + "{0:s}".format(self.res['temp'].latex(f=2, base=0)[1:]), va='bottom', ha='left', color=color, fontsize=16)

    def plot_post(self, ax=None):
        if self.sampler != None:
            tau = self.sampler.get_autocorr_time()
            print(tau)
            burnin, thin = int(10 * np.max(tau)), int(0.5 * np.min(tau))
            # burnin, thin = 1000, 5

            samples = self.sampler.get_chain(discard=burnin, flat=True, thin=thin)
            #d = distr2d(samples[:, 0], samples[:, 1])
            #d.plot_contour(ax=self.regionplot, color_point=None)
            corner.hist2d(samples[:, 0], samples[:, 1])

        self.regionplot.set_xlabel(r'$\log N_{\rm tot}\,[\rm cm^{-2}]$')
        self.regionplot.set_ylabel(r'$T_{\rm exc}\rm\,K$')

    def latex(self, attr='temp', f=2, base=None):
        if hasattr(self, 'res') and attr in self.res.keys():
            return self.res[attr].dec().latex(f=f, base=base)

if __name__ == '__main__':

    n = []

    species = 'CO'

    if species == 'H2':
        n.append(a(17.57, 0.12, 0.12))
        n.append(a(17.53, 0.12, 0.12))
        #n.append(a(16.73, 0.22, 0.22))
        t = ExcitationTemp(species, n, debug=True)
        t.calc(method='emcee', plot=1, verbose=1)
        t.temp_to_slope()
        print(t.slope, t.zero)
        t.plot()

    elif species == 'CO':
        n.append(a(14.43, 0.12, 0.12))
        n.append(a(14.52, 0.08, 0.08))
        n.append(a(14.33, 0.06, 0.06))
        n.append(a(13.73, 0.05, 0.05))
        #n.append(a(13.14, 0.13, 0.13))
        t = ExcitationTemp(species, n, debug=True)
        t.calc(method='emcee', verbose=1)
        t.plot()
        #print(t.latex('temp'))

    elif species == 'FeII':
        print(Temp.col_dens(num=13, T=15000))
    #Temp.plot_temp(temp=9.9, Ntot=14.94, color='b')
    #Temp.latex()
    plt.show()
