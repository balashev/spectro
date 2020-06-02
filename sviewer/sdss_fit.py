# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:37:28 2016

@author: Serj
"""
from astropy.io import fits
from functools import partial
#from matplotlib.mlab import PCA as mPCA
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import pickle
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import minimize
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, FastICA

from ..excitation_temp import ExcitationTemp
from ..profiles import convolveflux
from ..stats import distr2d, distr1d
from .graphics import Spectrum, Speclist
from .utils import Timer, labelLine

class Stack():
    def __init__(self, num=None):
        self.n = num
        self.attrs = ['sig', 'sig_p', 'zero', 'cont', 'poly']
        self.sub = [''] # ['', '_w']  # ['', '_w', '_ston']
        for attr in self.attrs + ['mask']:
            for s in self.sub:
                setattr(self, attr + s, np.zeros(num))

    def masked(self):
        for attr in self.attrs:
            for s in self.sub:
                setattr(self, attr + s, getattr(self, attr + s) / getattr(self, 'mask' + s))

    def save(self, l, attrs=None, folder='C:/science/Kaminker/Lyforest/temp/', filename=''):
        if attrs is None:
            attrs = self.attrs + ['mask']
        for attr in attrs:
            for s in self.sub:
                np.savetxt(folder + filename + attr + s + '.dat', np.c_[l, getattr(self, attr + s)],
                           fmt='%18.6f')

def calc_SDSS_stack(self, show=['sig'], save=True, ra=None, dec=None, lmin=3.5400, lmax=4.0000, snr=2.5):
    """
    Subroutine to calculate SDSS QSO stack spectrum

    parameters:
        self     -  basic class which call this routine
        typ        -  what type of Stack
        ra         -  Right Ascension mask (tuple of two values e.g. (20, 40) or None)
        dec        -  Declination mask (tuple of two values e.g. (20, 40) or None)
        lmin       -  loglambda minimum boundary
        lmax       -  loglambda maximum boundary
        snr        -  SNR threshold
    """

    # >>> prepare stack class to write
    delta = 0.0001
    num = int((lmax-lmin)/delta+1)
    stack = Stack(num)
    l_min = 1041
    l_max = 1185
    calc_stack = 1
    for s in self.s:
        s.remove()
    self.s = Speclist(self)
    self.specview = 'line'
    self.s.append(Spectrum(self, name='stack_cont'))
    self.s.append(Spectrum(self, name='stack_poly'))
    self.s.append(Spectrum(self, name='stack_zero'))
    self.s.append(Spectrum(self, name='cont'))
    self.s.append(Spectrum(self, name='poly'))
    self.s.append(Spectrum(self, name='mask'))

    data = self.IGMspec['BOSS_DR12/meta']
    #print(data.dtype)

    # >>> apply mask:
    print(data.dtype, data['RA_GROUP'], data['DEC_GROUP'])
    mask = np.ones(len(data), dtype=bool)
    print(np.sum(mask))

    #ra, dec = (144, 145), (20, 25)
    if ra is not None:
        mask *= (data['RA_GROUP'] > ra[0]) * (data['RA_GROUP'] < ra[1])
    print(np.sum(mask), data['RA_GROUP'])
    if dec is not None:
        mask *= (data['DEC_GROUP'] > dec[0]) * (data['DEC_GROUP'] < dec[1])
    print(np.sum(mask), data['DEC_GROUP'])
    if snr is not None:
        mask *= data['SNR_SPEC'] > snr
    print(np.sum(mask))
    num = np.sum(mask)
    if 0:
        fig, ax = plt.subplots(subplot_kw=dict(projection='aitoff'))
        ax.scatter((data['RA_GROUP'][mask]/180-1)*np.pi, data['DEC_GROUP'][mask]/180*np.pi, s=5, marker='+')
        plt.grid(True)
        plt.show()
    else:
        for i, s, z in zip(range(np.sum(mask)), self.IGMspec['BOSS_DR12/spec'][mask], self.IGMspec['BOSS_DR12/meta']['Z_VI'][mask]):
            print(z, i, num)
            m = (((s['wave'] > (1+z)*1300) * (s['wave'] < (1+z)*1383)) | ((s['wave'] > (1+z)*1408) * (s['wave'] < (1+z)*1500)))
            if np.sum(m) > 0:
                ma = ~ (np.ma.masked_invalid(s['sig'][m]).mask | np.ma.masked_invalid(s['flux'][m]).mask)
                norm = np.mean(s['flux'][m][ma])
                sn = norm / np.sqrt(np.mean(s['sig'][m][ma]))
                m = (s['wave'] > (1+z)*l_min) * (s['wave'] < (1+z)*l_max)
                if np.sum(m) > 0 and sn > 1:
                    data = np.recarray((len(s['wave'][m]),), dtype=[('wave', float), ('flux', float), ('sig', float), ('cont', float), ('mask', int), ('corr', float)])
                    data['wave'] = np.log10(s['wave'][m])
                    data['flux'] = s['flux'][m]
                    data['sig'] = s['sig'][m]
                    data['cont'] = np.ones_like(data['wave'])
                    i_min = int(max(0, int(round((data['wave'][0] - self.LeeResid[0][0]) / delta))))
                    i_max = int(max(0, int(round((data['wave'][-1] - self.LeeResid[0][0]) / delta) + 1)))
                    #print(self.LeeResid[0][i_min], self.LeeResid[0][i_max], data['wave'][0], data['wave'][-1])
                    data['corr'] = self.LeeResid[1][i_min:i_max]
                    data['corr'] = np.ones_like(data['wave'])
                    data['mask'] = np.ones_like(data['wave'])
                    p = np.polyfit(data['wave'], data['flux'], 1)
                    #print(np.sum(m), data['wave'], data['flux'])
                    #p1 = np.polyfit(data['wave'], data['flux'], 8, w=np.power(data['sig'], -1))
                    poly = np.polyval(p, data['wave'])
                    #poly_sig = (p1[0] * data['wave'] + p1[1])

                    imin = int(round((data['wave'][0]-lmin) / delta))
                    imax = int(round((data['wave'][-1]-lmin) / delta) + 1)
                    w = 1. / (sn ** -2 + 0.1 ** 2)
                    stack.mask[imin:imax] += data['mask'] * w
                    stack.sig[imin:imax] += data['flux'] / data['corr'] / norm * data['mask'] * w
                    #stack.cont[imin:imax] += data['cont'] * data['mask']
                    #stack.poly[imin:imax] += poly * data['mask']
                    #stack.sig[imin:imax] += data['flux'] / data['corr'] / data['cont'] * data['mask']
                    #stack.sig_p[imin:imax] += data['flux'] / data['corr'] / poly * data['mask']
                    #stack.zero[imin:imax] += (data['flux'] / data['corr'] - poly) * data['mask'] / np.std((data['mask'] / data['corr'] - poly) * data['mask'], ddof=1)

                    #ston = np.mean(data['flux'] / data['sig'])

                    #stack.mask_w[imin:imax] += data['mask'] * ston
                    #stack.cont_w[imin:imax] += data['cont'] * data['mask'] * ston
                    #stack.poly_w[imin:imax] += poly * data['mask'] * ston
                    #stack.sig_w[imin:imax] += data['flux'] / data['corr'] / data['cont'] * data['mask'] * ston
                    #stack.sig_p_w[imin:imax] += data['flux'] / data['corr'] / poly * data['mask'] * ston
                    #stack.zero_w[imin:imax] += (data['flux'] / data['corr'] - poly) * data['mask'] * ston / np.std((data['flux'] / data['corr'] - poly) * data['mask'], ddof=1)
                    if 0:
                        fig, ax = plt.subplots()
                        ax.plot(data['wave'], data['flux'], label='flux')
                        ax.plot(data['wave'], data['sig'], label='flux/cont')
                        ax.plot(data['wave'], data['mask'], label='mask')
                        ax.plot(data['wave'], data['cont'], label='cont')
                        ax.plot(data['wave'], poly, label='poly')
                        ax.plot(data['wave'], data['corr'], label='corr')
                        ax.plot(data['wave'], (data['flux'] - poly) / np.std((data['flux'] - poly) * data['mask'], ddof=1), label='zerot')
                        ax.legend(loc='best')
                        plt.show()
    stack.masked()

    l = np.power(10, lmin + delta * np.arange(stack.n))
    if show is not None and len(show) > 0:
        for i, attr in enumerate(stack.attrs + ['mask']):
            self.s[i].set_data([l, getattr(stack, attr)])
        self.s.redraw()
        self.vb.enableAutoRange(axis=self.vb.XAxis)
        self.vb.setRange(yRange=(-2, 2))

    if save:
        filename = '{:.1f}_{:.1f}_'.format(float(ra[0]), float(ra[1])) if ra is not None else ''
        stack.save(l, attrs=['sig'], folder='C:/science/Kaminker/Lyforest/SDSS_Stripes/', filename=filename)

    return stack

def calc_SDSS_stack_stripes(self):
    self.LeeResid = np.loadtxt('C:/Science/SDSS/DR9_Lee/residcorr_v5_4_45.dat', unpack=True)
    #calc_SDSS_stack(self, show=None)
    #for ra in range(120, 240, 20):
    #    calc_SDSS_stack(self, ra=(ra, ra+20), dec=(0, 60), show=None)
    #for ra in range(120, 240, 3):
    #    calc_SDSS_stack(self, ra=(ra, ra+3), dec=(0, 60), show=None)
    sig = np.genfromtxt('C:/science/Kaminker/Lyforest/SDSS_Stripes/sig.dat', unpack=True)
    for ra in range(120, 240, 20):
        filename = '{:.1f}_{:.1f}_'.format(float(ra), float(ra+20))
        data = np.genfromtxt('C:/science/Kaminker/Lyforest/SDSS_Stripes/'+filename+'sig.dat', unpack=True)
        np.savetxt('C:/science/Kaminker/Lyforest/SDSS_Stripes/'+filename+'sig_res.dat', np.c_[data[0], data[1]/sig[1]])

def makeH2Stack(self, beta=-0.9, Nmin=16, Nmax=22, b=4, norm=0, load=True, draw=True):
    # >>> make/load templates:

    if not load:
        Temp = 100
        num = 4
        N_0643 = [18.22, 18.25, 16.62, 14.84, 13.94, 13.86]
        N_ref = [19.70, 20.00, 17.96, 17.76, 15.88, 15.17]
        Ngrid = np.logspace(Nmin, Nmax, 71)
        H2 = ExcitationTemp('H2')
        #H2.calcTemp(N_ref[:2], plot=1)
        self.atomic.readH2(j=num-1)
        self.console.exec_command('load H2_{:}'.format(num))
        for s in self.fit.sys[0].sp.values():
            s.b.val = b
        spec = None
        for Ntot in Ngrid:
            #N = H2.col_dens(num=num, Temp=Temp, Ntot=np.log10(Ntot))
            N = H2.rescale_col_dens(N_ref[:num], Ntot=np.log10(Ntot))
            print(np.log10(Ntot))
            for i, Ni in enumerate(N):
                self.fit.sys[0].sp['H2j{}'.format(i)].N.val = Ni
            self.generate(template='const', z=0, fit=True, xmin=950, xmax=1200, resolution=2000, snr=None,
                 lyaforest=0.0, lycutoff=False, Av=0.0, Av_bump=0.0, z_Av=0.0, redraw=False)
            y = self.s[self.s.ind].spec.y()
            if spec == None:
                spec = np.empty((0, len(y)))
            spec = np.append(spec, [y], axis=0)
            x = self.s[self.s.ind].spec.x()
            if not draw:
                self.s.remove()
        with open('C:/Temp/H2_{:.1f}.dat'.format(b), 'wb') as f:
            pickle.dump((np.log10(Ngrid), x, spec), f)
    else:
        Ngrid, x, spec = pickle.load(open('C:/Temp/H2_{:.1f}.dat'.format(b), 'rb'))

    # >>> make stack
    a = interp2d(Ngrid, x, spec.transpose(), fill_value=1)
    N = np.logspace(Nmin, Nmax, 500)
    if 1:
        L = a(np.log10(N), x) * N ** (beta)
        spec = np.trapz(L, x=N, axis=1)
        if beta == -1:
            spec *= (np.log(10.0 ** Nmax) - np.log(10.0 ** Nmin))
        else:
            spec *= (beta + 1) / (10 ** (Nmax * (beta + 1)) - 10 ** (Nmin * (beta + 1)))
        spec = 1 - (1 - spec) * 10 ** norm
        #print(spec.shape, spec)
    else:
        mask = x > 1045
        W = np.zeros_like(N)
        W = np.trapz(1- a(np.log10(N), x[mask]), x=x[mask], axis=0)
        #W *= (beta + 1) / (10 ** (Nmax * (beta + 1)) - 10 ** (Nmin * (beta + 1))) * N ** (beta)
        print(W)

    #spec = convolveflux(x, spec, res=1300, kind='direct')
    if draw:
        print(x, spec)
        self.importSpectrum('mock_b={:.2f}_min={:}_max={:}_b={:}'.format(beta, Nmin, Nmax, b), spec=[x, spec], append=True)
        if 1:
            xs = [1018.6, 1030.2, 1044.6, 1057.7, 1071.6, 1086.5, 1102.5, 1126.7]
            ys = self.s[-1].spec.y()[self.s[-1].spec.index(xs)]
            print(xs, ys)
            self.s[-1].spline.add(xs, ys)
            self.s[-1].calc_spline()
    else:
        return x, spec

def H2StackFit(self, Nmin=16, Nmax=22, b=4, ngrid=30, load=True, draw=True, name='.dat'):

    print(name, Nmin, Nmax, b)
    ind = 0
    ESDLA = '_ESDLA' if 'sstack' in name else ''
    if not load:
        if 'metal' in name:
            beta = np.linspace(-1.8, -1.3, ngrid)
            norm = np.log10(np.logspace(-1.3, -0.8, ngrid))
        elif 'sstack' in name:
            beta = np.linspace(-1.4, -0.8, ngrid)
            norm = np.log10(np.logspace(-0.7, -0.2, ngrid))
        elif 'NLR' in name:
            beta = np.linspace(-1.9, -0.9, ngrid)
            norm = np.log10(np.logspace(-1.2, -0.5, ngrid))
        else:
            beta = np.linspace(-1.6, -1.0, ngrid)
            norm = np.log10(np.logspace(-1.6, -1.2, ngrid))
        x, spec = makeH2Stack(self, draw=False)
        Beta, Norm = np.meshgrid(beta, norm)
        z = np.empty_like(Beta)
        for i, be in enumerate(beta):
            print(i)
            for k, n in enumerate(norm):
                #print(k)
                x, spec = makeH2Stack(self, beta=be, Nmin=Nmin, Nmax=Nmax, b=b, norm=n, draw=False)
                self.s[ind].set_fit(x=x, y=spec)
                z[k, i] = self.s[-1].chi2()

        with open('C:/Temp/H2stack_'+name.replace('.dat', '')+'_{:.1f}_{:.1f}_{:.1f}.dat'.format(Nmin, Nmax, b), 'wb') as f:
            pickle.dump((beta, norm, z), f)
    else:
        beta, norm, z = pickle.load(open('C:/Temp/H2stack_'+name.replace('.dat', '')+'_{:.1f}_{:.1f}_{:.1f}.dat'.format(Nmin, Nmax, b), 'rb'))
        #print(beta, norm, z)

    rvs = False
    if draw:
        # >>> mask some problem at beta=-1
        if 0:
            i = np.argwhere(beta == -1)[0]
            print(i)
            if len(i) > 0:
                beta = np.delete(beta, i[0])
                print(beta, z)
                z = np.delete(z, i[0], 1)
                print(z.shape)

        Beta, Norm = np.meshgrid(beta, norm)

        rescale = np.sqrt(np.nanmin(z.flatten()) / (np.sum(self.s[ind].fit_mask.x()) - 2))
        print(np.nanmin(z.flatten()),  np.sum(self.s[ind].fit_mask.x()))
        print('rescale', rescale)
        z = np.exp(- (z-np.min(z.flatten()))/rescale**2)
        d = distr2d(beta, norm, z, debug=False)
        d.normalize()
        d.dopoint()
        d1 = d.marginalize(over='x')
        d1.dopoint()
        d1.plot()
        bf_beta, bf_norm = d.point[0], d.point[1]
        print(bf_beta, bf_norm)
        if rvs:
            x, y = d.rvs(4000) #, xrange=(-2.5,-0.8), yrange=(-1.8, -1.0))
            print([(xi, yi, d.pdf(xi, yi)[0]) for xi, yi in zip(x,y)])
            np.savetxt('C:/science/papers/H2Stack/work/rvs.dat', np.c_[x, y])
        beta_min, norm_min = d.point[0], d.point[1]

        if 1:
            font, xlabel = 24, r'$\beta$'
            ax = d.plot_contour(conf_levels=[0.6827, 0.9973], xlabel=xlabel, ylabel=r'$\log$(r)', color='k',
                                color_point='red', cmap=None, alpha=0, colorbar=False, ls=['-', '--'], font=font)
            ax.set_xlim([-1.75, -0.45])
            ax.set_ylim([-1.7, -0.0])
            if rvs:
                if 0:
                    counts, xbins, ybins, image = plt.hist2d(x, y, bins=10)
                    ax.contour(counts, extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()], linewidths=2)
                else:
                    ax.scatter(x, y)

            # >>> add Noterdaeme2008 data
            if 0 and name=='stack_tot.dat':
                x = np.linspace(ax.axis()[0], ax.axis()[1], 10)
                if 0:
                    #y1 = np.ones_like(x) * np.log10(0.11)
                    #y2 = np.ones_like(x) * np.log10(0.08)
                    ax.fill_between(x, y2, y1, interpolate=True, color='steelblue', alpha=0.5, label='Noterdaeme et al. 2008')
                else:
                    y = np.ones_like(x) * np.log10(0.10)
                    ax.errorbar(x, y, yerr=0.025, uplims=True, color='steelblue', label='Noterdaeme et al. 2008', lw=1)
                # ax.axhline(np.log10(0.06), color='k', ls='--', lw=1)

            # >>> add Balashev2014 data
            if 0 and name=='stack_tot.dat':
                x = np.linspace(ax.axis()[0], ax.axis()[1], 10)
                y = np.ones_like(x) * np.log10(0.075)
                ax.errorbar(x, y, yerr=0.025, uplims=True, color='darkorange', label='Balashev et al. 2014', lw=1)
                # ax.axhline(np.log10(0.06), color='k', ls='--', lw=1)

            # >>> add Jorgenson2014 data
            if 0 and name=='stack_tot.dat':
                x = np.linspace(ax.axis()[0], ax.axis()[1], 10)
                y = np.ones_like(x) * np.log10(0.05)
                ax.errorbar(x, y, yerr=0.025, uplims=True, color='deeppink', label='Jorgenson et al. 2014', lw=1)
                # ax.axhline(np.log10(0.06), color='k', ls='--', lw=1)
            if 0:
                name = 'stack_tot.dat'
                beta, norm, z = pickle.load(open('C:/Temp/H2stack_' + name.replace('.dat', '') + '_{:.1f}_{:.1f}.dat'.format(Nmin, Nmax), 'rb'))
                rescale = np.sqrt(np.min(z.flatten()) / (np.sum(self.s[ind].fit_mask.x()) - 2))
                z = np.exp( -(z - np.min(z.flatten()))/rescale**2)
                d = distr2d(beta, norm, z, debug=True)
                d.normalize()
                d.dopoint()
                #d.plot_contour(conf_levels=[0.683], ax=ax, alpha=0, ls=['--'], colorbar=False)
                c = ax.contour(d.X, d.Y, d.z / d.zmax, levels=d.level(0.683) / d.zmax, colors='k', ls='--')
                for l in c.collections:
                    l.set_dashes(':')
                #ax.scatter(d.point[0], d.point[1], s=50, color='#222222', marker='*')

            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.tick_params(which='both', width=1)
            ax.tick_params(which='major', length=5)
            ax.tick_params(which='minor', length=3)
            ax.tick_params(axis='both', which='major', labelsize=font-2)
            ax.legend(loc='best', fontsize=font-2, frameon=None, facecolor=None, framealpha=0)
            plt.tight_layout()
            plt.savefig('C:/science/papers/H2Stack/figures/contour.pdf')
        if 0:
            print('maginalize:')
            d1 = d.marginalize('y')
            print('marg point:', d1.dopoint())
            for c in [0.683]:
                print('marg interval:', d1.interval(conf=c)[0])
                d1.plot(conf=c)
            d1 = d.marginalize('x')
            print('marg point:', np.power(10, d1.dopoint()))
            for c in [0.683]:
                print('marg interval:', np.power(10, d1.interval(conf=c)[0]))
                d1.plot(conf=c)
        plt.show()

    print(beta_min, Nmin)
    x, spec = makeH2Stack(self, beta=beta_min, Nmin=Nmin, Nmax=Nmax, b=b, norm=norm_min, draw=False)
    self.fit.setValue('cf_0', 1 - 10**norm_min)
    self.fit.setValue('cf_0', 'sys0_exp0', 'addinfo')
    #self.s[ind].spec.norm.err *= rescale
    self.s[ind].set_fit(x=x, y=spec)
    self.s.chi2()
    if 0:
        self.importSpectrum('best_fit={:.2f}_min={:.1f}_max={:.1f}'.format(bf_beta, Nmin, Nmax), spec=[self.s[ind].fit.x(), self.s[ind].fit.y()], append=True)
        self.s[-1].spec.norm = self.s[-1].spec.raw
    self.s.redraw()

    if draw and 0:
        self.showLines(show=False)
        print('C:/science/papers/H2Stack/work/fig_H2_Mass'+ESDLA)
        self.showlines.loadSettings('C:/science/papers/H2Stack/work/fig_H2_Mass'+ESDLA)
        self.regions = ['1044..1119']
        self.showlines.y_formatter = '%.2f'
        fig = self.showlines.showPlot(True, showH2=[0, 1, 2, 3])

def makeHIStack(self, beta=-1.5, N_g=None, Nmin=20.0, Nmax=22.0, load=True, draw=True):
    #>>> make/load templates:

    #load = False
    if not load:
        Ngrid = np.logspace(Nmin, Nmax, 201)
        self.console.exec_command('load HIStack')
        spec = None
        for N in Ngrid:
            print(np.log10(N))
            self.fit.sys[0].z.val = 0
            self.fit.sys[0].sp['HI'].N.val = np.log10(N)
            self.generate(template='const', z=0, fit=True, xmin=900, xmax=1500, resolution=1500, snr=None,
                 lyaforest=0.0, lycutoff=False, Av=0.0, Av_bump=0.0, z_Av=0.0, redraw=False)
            y = self.s[self.s.ind].spec.y()
            if spec == None:
                spec = np.empty((0, len(y)))
            spec = np.append(spec, [self.s[self.s.ind].spec.y()], axis=0)
        x = self.s[self.s.ind].spec.norm.x
        with open('C:/Temp/HI.dat', 'wb') as f:
            pickle.dump((np.log10(Ngrid), x, spec), f)
    else:
        Ngrid, x, spec = pickle.load(open('C:/Temp/HI.dat', 'rb'))

    # make stack
    a = interp2d(Ngrid, x, spec.transpose(), fill_value=1)
    N = np.logspace(Nmin, Nmax, 500)
    if N_g is None:
        L = a(np.log10(N), x) * N ** (beta)
    else:
        L = a(np.log10(N), x) * (N/10**float(N_g))**(beta) * np.exp(-N/10**float(N_g))
    spec = np.trapz(L, x=N, axis=1)
    if N_g is None:
        spec *= (beta + 1) / (10 ** (Nmax * (beta + 1)) - 10 ** (Nmin * (beta + 1)))
    else:
        spec /= np.trapz((N/10**float(N_g))**(beta) * np.exp(-N/10**float(N_g)), x=N)

    #spec = convolveflux(x, spec, res=800, kind='direct')
    if draw:
        self.importSpectrum('mock_b={:.2f}_min={:}_max={:}'.format(beta, Nmin, Nmax), spec=[x, spec], append=True)
    else:
        return x, spec

def HIStackFitPower(self, load=True, draw=True):

    #load = False
    if not load:
        beta = np.linspace(-1.85, -1.75, 100)
        z = np.empty_like(beta)
        for i, b in enumerate(beta):
            print(i)
            x, spec = makeHIStack(self, beta=b, Nmin=20, Nmax=22, draw=False)
            self.s[-1].set_fit(x=x, y=spec)
            z[i] = self.s[-1].chi2()
        with open('C:/Temp/HIstack.dat', 'wb') as f:
            pickle.dump((beta, z), f)
    else:
        beta, z = pickle.load(open('C:/Temp/HIstack.dat', 'rb'))
        print(beta, z)

    rescale = np.sqrt(np.min(z.flatten())/2)

    rescale = np.sqrt(np.min(z.flatten()) / (np.sum(self.s[-1].fit_mask.x())- 2))
    print('rescale', rescale)

    z = np.exp(- (z - np.min(z.flatten()))/rescale**2)
    d = distr1d(beta, z, debug=True)
    d.dopoint()
    d.dointerval()
    beta_min = d.point
    ax = d.plot(conf=0.683, color='dodgerblue', xlabel=r'$\beta$')
    ax.set_xlim([-1.84, -1.79])

    x, spec = makeHIStack(self, beta=beta_min, Nmin=20, Nmax=22, draw=False)
    self.s[-1].spec.norm.err *= rescale / 2
    self.s[-1].set_fit(x=x, y=spec)
    self.s.chi2()
    self.s.redraw()

    if draw:
        self.showLines(show=False)
        self.showlines.loadSettings('C:/science/papers/H2Stack/work/fig_HI_Mass')
        self.plot.regions.fromText('1196..1241')
        fig = self.showlines.showPlot(False)
        #eta = d.rvs(30)
        #print(beta)
        #for b in beta:
        #    x, spec = self.makeHIStack(beta=b, Nmin=20, Nmax=22, draw=False)
        #    fig.axes[0].plot(x, spec, '-b', lw=0.5, alpha=0.1)
        if 0:
            beta = d.dointerval()
            x, spec1 = makeHIStack(self, beta=beta[0][0], Nmin=20, Nmax=22, draw=False)
            x, spec2 = makeHIStack(self, beta=beta[0][1], Nmin=20, Nmax=22, draw=False)
            fig.axes[0].plot(x, spec1, '-', c='dodgerblue', lw=0.5)
            fig.axes[0].plot(x, spec2, '-', c='dodgerblue', lw=0.5)
            fig.axes[0].fill_between(x, spec1, spec2, facecolor='dodgerblue', alpha=0.3, zorder=0)

        # >>> add individual HI profiles
        if 1:
            for N, loc in zip([20, 20.5, 21, 21.5], [1207, 1205.7, 1203.8, 1201]):
                self.fit.setValue('z_0', 0)
                self.fit.setValue('N_0_HI', N)
                self.showFit()
                fig.axes[0].plot(self.s[-1].fit.x(), self.s[-1].fit.y(), '--k', lw=0.5, zorder=0)
                lines = plt.gca().get_lines()
                labelLine(lines[-1], loc, label='{:.1f}'.format(N), fontsize=18, zorder=1)
        ax2 = fig.add_axes([.70, .34, .27, .41])
        d.plot(conf=0.683, ax=ax2, color='dodgerblue', xlabel=r'$\beta$', fontsize=22)
        ax2.set_xlim([-1.85, -1.785])
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.xaxis.set_major_locator(MultipleLocator(0.02))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.yaxis.set_major_locator(MultipleLocator(100))
        #plt.tight_layout()
        plt.savefig('C:/science/papers/H2Stack/figures/sdssHI.pdf')
        plt.show()

def HIStackFitGamma(self, load=True, draw=True):

    #load = False
    if not load:
        beta = np.linspace(-2.2, -0.9, 20)
        norm = np.log10(np.logspace(21, 22, 20))
        x, spec = self.makeHIStack(N_g=21.5, draw=False)
        speci = np.empty([len(beta), len(norm), len(x)])
        for i, b in enumerate(beta):
            print(i)
            for k, n in enumerate(norm):
                #print(k)
                x, spec = self.makeHIStack(beta=b, N_g=n, Nmin=20, Nmax=22, draw=False)
                speci[k, i] = spec
        with open('C:/Temp/HIstackanalysis.dat', 'wb') as f:
            pickle.dump((beta, norm, x, speci), f)
    else:
        beta, norm, x, speci = pickle.load(open('C:/Temp/HIstackanalysis.dat', 'rb'))
        print(beta, norm, x)

    Beta, Norm = np.meshgrid(beta, norm)
    z = np.empty_like(Beta)
    for i, b in enumerate(beta):
        print(i)
        for k, n in enumerate(norm):
            # print(k)
            self.s[-1].set_fit(x=x, y=speci[k, i])
            z[k, i] = self.s[-1].chi2()
    rescale = np.sqrt(np.min(z.flatten())/2)

    if draw:
        rescale = np.sqrt(np.min(z.flatten()) / (np.sum(self.s[-1].fit_mask.x())- 2))
        print('rescale', rescale)
        #self.console.exec_command('rescale err {:f}'.format(rescale))
        z = np.exp(- z / np.min(z.flatten()))
        d = distr2d(beta, norm, z, debug=True)
        d.dopoint()
        beta_min, norm_min = d.point[0], d.point[1]
        #d.plot()
        ax = d.plot_contour(xlabel=r'$\beta$', ylabel=r'$N_g$')
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(which='both', width=1)
        ax.tick_params(which='major', length=5)
        ax.tick_params(which='minor', length=3)
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        #plt.savefig('C:/science/papers/H2Stack/work/Gamma.pdf')
        plt.show()
        if 0:
            d1 = d.marginalize('y')
            print('marg point:', d1.point())
            for c in [0.683]:
                print('marg interval:', d1.interval(conf=c))

    x, spec = self.makeHIStack(beta=beta_min, N_g=norm_min, Nmin=20, Nmax=22, draw=False)
    self.s[-1].spec.norm.err *= rescale
    self.s[-1].set_fit(x=x, y=spec)
    self.s.chi2()
    self.s.redraw()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def make_H2_template(self):
    self.console.exec_command('load H2_2')
    H2_range = np.linspace(20, 21, 1)
    spec = None
    for i, H2 in enumerate(H2_range):
        print(i)
        self.fit.sys[0].sp['H2j0'].N.val = H2 - 0.4
        self.fit.sys[0].sp['H2j1'].N.val = H2 - 0.1
        self.generate(template='const', z=0.0, fit=True,
                      xmin=910, xmax=1150, resolution=1800, lyaforest=0,
                      )
        if spec is None:
            x = self.s[self.s.ind].spec.x()
            spec = np.zeros([H2_range.shape[0], x.shape[0]])

        spec[i] = self.s[self.s.ind].spec.y()
    f = open('temp/H2_temp', "wb")
    pickle.dump(H2_range, f)
    pickle.dump(x, f)
    pickle.dump(spec, f)

def load_H2_template():
    f = open('temp/H2_temp', "rb")
    H2_range = pickle.load(f)
    x = pickle.load(f)
    spec = pickle.load(f)
    return H2_range, x, spec

def weight_H2(x):
    return 1 / (1 + np.exp((-x-2)*5)) * (1 + np.exp((x-2)))

def search_H2(self, z_abs=2.4, v_range=2000, HI=21.2, H2=19.8):
    mask = (self.s[self.s.ind].spec.x() < 1130 * (1 + z_abs)) * (self.s[self.s.ind].spec.x() > 911 * (1 + z_abs))
    x, y, err = self.s[self.s.ind].spec.x()[mask], self.s[self.s.ind].spec.y()[mask], self.s[self.s.ind].spec.err()[mask]
    if 0:
        make_H2_template(self)
    H2_range, x_s, spec = load_H2_template()
    print(H2_range)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    #ax.plot(x, y)
    for i, H2 in enumerate(H2_range):
        #ax.plot(x, spec[i])
        f = interp1d(x_s, spec[i], fill_value=1, bounds_error=False)
        mask = spec[i] < 0.95
        mask_inter = interp1d(x_s, mask, fill_value=0, bounds_error=False)
        v_grid = np.linspace(-v_range, v_range, 100)
        chi, sp, p = np.zeros_like(v_grid), np.zeros_like(v_grid), np.zeros_like(v_grid)
        for k, v in enumerate(v_grid):
            z = z_abs + (1 + z_abs) * v / 3e5
            m = mask_inter(x / (1 + z)) == 1
            if 1:
                sp[k], p[k] = spearmanr(y[m], f(x[m] / (1 + z)))
            if 1:
                res = (y[m] - f(x[m] / (1 + z))) / err[m]
                w = weight_H2(res)
                chi[k] = np.sum(res ** 2 * w) / np.sum(w)

        if 1:
            ax.plot(v_grid, sp, c='orangered', label=H2)
            ax2.plot(v_grid, np.log10(p), 'dodgerblue', label=H2)
            ax2.spines['right'].set_position(('outward', 0))
            ax2.set_frame_on(True)
            ax2.patch.set_visible(False)
            ax3.plot(v_grid, chi, 'forestgreen', label=H2)
            ax3.spines['right'].set_position(('outward', 60))
            ax3.set_frame_on(True)
            ax3.patch.set_visible(False)
        if 0:
            x1 = np.linspace(910, 1150, 200)
            ax.plot(x1, mask_inter(x1))
        ind = np.argmax(sp)
        self.setz_abs(z_abs + (1 + z_abs) * v_grid[ind] / 3e5)
    plt.show()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def emsvd(Y, k=None, tol=1E-3, maxiter=None):
    """
    Approximate SVD on data with missing values via expectation-maximization

    Inputs:
    -----------
    Y:          (nobs, ndim) data matrix, missing values denoted by NaN/Inf
    k:          number of singular values/vectors to find (default: k=ndim)
    tol:        convergence tolerance on change in trace norm
    maxiter:    maximum number of EM steps to perform (default: no limit)

    Returns:
    -----------
    Y_hat:      (nobs, ndim) reconstructed data matrix
    mu_hat:     (ndim,) estimated column means for reconstructed data
    U, s, Vt:   singular values and vectors (see np.linalg.svd and 
                scipy.sparse.linalg.svds for details)
    """

    if k is None:
        svdmethod = partial(np.linalg.svd, full_matrices=False)
    else:
        svdmethod = partial(svds, k=k)
    if maxiter is None:
        maxiter = np.inf

    # initialize the missing values to their respective column means
    mu_hat = np.nanmean(Y, axis=0, keepdims=1)
    valid = np.isfinite(Y)
    Y_hat = np.where(valid, Y, mu_hat)

    halt = False
    ii = 1
    v_prev = 0

    while not halt:

        # SVD on filled-in data
        U, s, Vt = svdmethod(Y_hat - mu_hat)

        # impute missing values
        Y_hat[~valid] = (U.dot(np.diag(s)).dot(Vt) + mu_hat)[~valid]

        # update bias parameter
        mu_hat = Y_hat.mean(axis=0, keepdims=1)

        # test convergence using relative change in trace norm
        v = s.sum()
        if ii >= maxiter or ((v - v_prev) / v_prev) < tol:
            halt = True
        ii += 1
        v_prev = v

    return Y_hat, mu_hat, U, s, Vt
    
class sdss():
    def __init__(self, parent, name, z_qso, plate, MJD, fiber):
        self.parent = parent
        self.name = str(name.decode("utf-8"))
        self.z_qso = float(z_qso)
        self.plate = plate
        self.MJD = MJD
        self.fiber = fiber
        self.set_filename()
        self.read_data()
        
    def set_filename(self):
        #self.SDSSfolder = 'D:/science/SDSS/DR12/ASCII/'
        #self.SDSSfolder = 'F:/DR12/data.sdss3.org/sas/dr12/boss/spectro/redux/'
        self.SDSSfolder = 'D:/science/SDSS/DR9_Lee/BOSSLyaDR9_spectra/'
        self.filename = self.SDSSfolder + str(self.plate) + '/' + 'speclya-{0}-{1}-{2}.fits'.format(self.plate, self.MJD, self.fiber)
        
    def read_data(self):
        hdulist = fits.open(self.filename)
        data = hdulist[1].data
        DR9 = 1
        if DR9:
            res_st = int((data.field('LOGLAM')[0] - self.parent.LeeResid[0][0])*10000)
            self.mask = data.field('OR_MASK') > 0
            self.l = 10**data.field('LOGLAM')
            self.f = data.field('FLUX')
            #cont = (data.field('CONT') * self.LeeResid[1][res_st:res_st+len(l)]) #/ data.field('DLA_CORR')
            self.sig = np.divide(np.power(data.field('IVAR'), -0.5), data.field('NOISE_CORR'))
        else:
            self.l = 10**data.field('loglam')
            self.f = data.field('flux')
            self.sig = (data.field('ivar'))**(-0.5)
            #cont = data.field('model')
        
    def apply_mask(self):
        self.f[self.mask] = np.nan
        self.sig[self.mask] = np.nan
        
    def restframe(self):
        self.lr = self.l / (1 + self.z_qso)
                
    def normalize(self):
        mask = np.logical_and(self.lr > 1270, self.lr < 1290)
        self.norm = np.nanmean(self.f[mask])
        #print(np.arange(len(mask))[mask], self.f[mask], self.mask[mask])
        #print(self.name, self.norm)
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
        
class SDSS_fit():
    def __init__(self, parent, num=10, timer=False):
        self.parent = parent
        self.create_scale()
        self.data = []
        # set number of general values:
        self.num = num
        if timer:
            self.t = Timer()
        else:
            self.t = None

    def add_list(self, lst):
        """
        add list of SDSS spectra to model
        """
        for l in lst:
            self.add_spec(self.parent, l[0], l[1], l[2], l[3], l[4])
            
    def add_spec(self, name, z_qso, plate, MJD, fiber):
        self.data.append(sdss(self.parent, name, z_qso, plate, MJD, fiber))
    
    def preprocess(self):
        for s in self.data:
            s.read_data()
            s.apply_mask()
            s.restframe()
            s.normalize()
            #if np.isnan(s.norm):
            #    self.data.remove(s)
            
    def time(self, st=''):
        """
        print timer
        """
        if self.t is not None:
            self.t.time(st)

    def create_scale(self, Npix=None):
        if Npix is None:
            self.Npix = 153
            self.Npix = 1217
        self.l_min = 911.75
        self.l_max = 1215.75
        self.l = np.linspace(self.l_min, self.l_max, self.Npix)
        print('wavelength scale is created:', self.l)
        
    def plot(self, plot=['all']):
        if 'all' in plot:
            fig, ax = plt.subplots()
            for y in self.Y:
                ax.plot(self.l, y)
        if 'PCA' in plot:
            fig, ax = plt.subplots()
            for y in self.PCA:
                ax.plot(self.l, y)
        if 'covar' in plot:
            fig, ax = plt.subplots()
            if 0:
                U, s, Vt = np.linalg.svd(self.K)
                Z = np.dot(U, Vt)
            else:
                Z = self.K/np.max(self.K)
            print('covar array:', Z)
            X, Y = np.meshgrid(self.l, self.l)
            c = ax.contourf(X, Y, Z, 100)
            cbar = fig.colorbar(c, orientation='vertical')
        plt.show(block=False)

    def toGUI(self, plot=['all']):
        if 'all' in plot:
            for y, v, name in zip(self.Y, self.V, self.qso):
                print(name)
                s = Spectrum(self.parent, name=str(name.decode("utf-8")))
                s.set_data([self.l, y, v])
                self.parent.s.append(s)
        if 'mean' in plot:
            s = Spectrum(self.parent, name='mean')
            s.set_data([self.l, self.m])
            self.parent.s.append(s)
            self.parent.s.ind = len(self.parent.s)-1
            print(np.min(self.m), np.max(self.m))
            print(np.min(self.l), np.max(self.l))
            self.parent.vb.setRange(xRange=(np.min(self.l), np.max(self.l)), yRange=(np.min(self.m), np.max(self.m)))                                  
        if 'PCA' in plot:
            for i, y in enumerate(self.PCA):
                s = Spectrum(self.parent, name=str(i))
                s.set_data([self.l, y])
                self.parent.s.append(s)
        if 'w' in plot:
            s = Spectrum(self.parent, name='abs. vector')
            s.set_data([self.l, self.w])
            self.parent.s.append(s)
        self.parent.s.draw()  
                
    def SDSS_prepare(self):
        self.Y, self.V, self.qso = [], [], []
        for s in self.data:
            if not np.isnan(s.norm):
                f = interp1d(s.lr, s.f/s.norm, bounds_error=False)
                sig = interp1d(s.lr, s.sig/s.norm, bounds_error=False)
                self.qso.append(s.name)
                self.Y.append(f(self.l))
                self.V.append(sig(self.l))
        self.Y = np.array(self.Y)
        self.V = np.array(self.V)
        self.nspec = self.V.shape[1]
        print(self.nspec)
        self.qso = np.array(self.qso, dtype="|S20")

    def SDSS_remove_outliers(self):
        Y_in = np.subtract(self.Y, np.nanmean(self.Y, axis=0))
        w = np.nanstd(Y_in, axis=0)
        mask = np.abs(np.divide(Y_in, w)) > 3
        print(sum(mask))
        self.Y[mask] = np.NaN

    def savetofile(self, attrs, prefix=''):
        """
        save some arrays given in attrs:
            flux, variance, names, covariance matrix, abs vector, e.t.c.
        """
        if attrs == 'all':
            attrs = ['Y', 'V', 'qso', 'M', 'K', 'w']
        for a in attrs:
            fmt = {'Y': '%7.2f', 'V': '%7.2f', 'qso': '%22s', 'm': '%7.2f',  
                   'M': '%7.2f', 'K': '%7.2f', 'w': '%7.2f'}
            np.savetxt('D:/science/SDSS/saved_{0}{1}.dat'.format(a, prefix), getattr(self, a), fmt=fmt[a])
        
    def load(self, attrs, prefix=''):
        """
        load arrays given in attrs:
            flux, variance, names, covariance matrix, abs vector, e.t.c.
        
        """
        print('Read data:', attrs)
        print('prefix:', prefix)
        if attrs == 'all':
            attrs = ['Y', 'V', 'qso', 'K', 'w']

        # load data:
        for a in attrs:
            dtype = {'Y': 'f8', 'V': 'f8', 'qso': '|S22', 'm': 'f8',
                     'M': 'f8', 'K': 'f8', 'w': 'f8'}
            print('Data read form:', 'D:/science/SDSS/saved_{0}{1}.dat'.format(a, prefix), dtype[a])
            setattr(self, a, np.genfromtxt('D:/science/SDSS/saved_{0}{1}.dat'.format(a, prefix), dtype=dtype[a]))

        # post process data:
        if 'Y' in attrs:
            self.nspec = self.Y.shape[0]
            self.Npix = self.Y.shape[1]
            print('pixels number:', self.Npix)
            print('spectra number:', self.nspec)
            self.create_scale(self.Npix)

        if 'M' in attrs and 'K' not in attrs:
            self.K = np.dot(self.M, self.M.transpose())

    def calc_mean(self):
        """
        calculate the mean spectrum
        """
        self.m = np.nanmean(self.Y, axis=0)
    
    def pack(self, *args):
        """
        pack (unpack) variables M, w into array x
        """
        if len(args) == 2:
            return np.append(args[0], args[1])
            
        elif len(args) == 1:
            xs = np.split(args[0], [self.num*self.Npix])
            M = xs[0].reshape((self.Npix, self.num))
            w = xs[1]
            return M, w
            
    def lnL(self, x):
        """
        calculate likelihood and Jacobian for model prior
        """
        #self.time('start lnL')
        Ms, w = self.pack(x)
        w = np.exp(w)
        lnL = 0
        chi_s = 0
        free = 0
        dM = np.zeros_like(Ms)
        dw = np.zeros_like(w)
        for y, v in zip(self.Y_in, self.V):
            #self.time('pre')
            
            # mask nan values:
            mask = np.logical_and(~np.isnan(y), y < np.inf)
            mask2 = np.logical_and(~np.isnan(v), v < np.inf)
            mask = np.logical_and(mask, mask2)
            y = y[mask]
            W = w[mask]
            d = v[mask] + W
            #self.time('mask')
                
            if not np.isnan(d).any():
                num = len(y)
                free += num
                M = Ms[mask]
                MT = np.transpose(M)
                d_inv = np.reciprocal(d)
                D_inv = np.diag(d_inv)
                MT_D_inv = np.multiply(MT, d_inv)
                H = np.diag(np.ones(self.num)) + np.dot(MT_D_inv, M)
                S_inv = D_inv - np.dot(MT_D_inv.transpose(), np.dot(np.linalg.inv(H), MT_D_inv))
                #det_S = np.log(np.linalg.det(np.dot(M, MT) + D))
                det_S = np.sum(np.log(d)) + np.log(np.linalg.det(H))
                #print(np.dot(d.transpose(), np.dot(S_inv, d)), det_S, num*np.log(2*np.pi))
                alpha = np.dot(S_inv, y)
                chi = np.dot(y.transpose(), alpha)
                lnL += chi + det_S + num*np.log(2*np.pi)
                chi_s += chi 

                #calcuculate jac
                dM[mask] -= np.dot(np.outer(alpha,alpha.transpose()) - S_inv, M)
                dw[mask] -= np.multiply(W, np.multiply(alpha, alpha) - np.diag(S_inv))
                #input()
        lnL *= 0.5
        self.time('finish lnL')
        print(lnL)
        print(chi_s, free - len(x))
        return (lnL, self.pack(dM, dw))
    
    def calc_covar(self):
        
        # remove noise pixels and fill nan values
        snr = np.divide(self.Y, self.V)
        mask = np.nanmean(snr, axis=1) > 1
        self.Y_in = self.Y[mask]
        mask = np.divide(self.Y_in, self.V[mask]) > 1
        self.Y_in[~mask] = np.nan
        self.Y_in = np.subtract(self.Y, self.m)
        Y_hat, mu_hat, U, s, Vt = emsvd(self.Y_in)
        self.Y_in = Y_hat
        self.Y = self.Y_in
        self.savetofile(['Y'], prefix='_temp')
        fig, ax = plt.subplots()
        for Y in self.Y_in[:100]:
            ax.plot(self.l, Y)
            
        # define initial guess for M nd w
        w = 'PCA'
        if w == 'PCA':
            pca = PCA(n_components=self.num, svd_solver='full', whiten=True) #, svd_solver='full')
            self.PCA = pca.fit_transform(self.Y_in.transpose()).transpose()
        elif w == 'mPCA':
            pca = mPCA(self.Y, standardize=False)
            self.PCA = pca.Y[:self.num]
        elif w == 'fastICA':
            ica = FastICA(n_components=self.num)
            self.PCA = ica.fit_transform(self.Y_in.transpose()).transpose()

        #self.toGUI('PCA')
        if 1:
            self.M = self.PCA
            self.K = np.dot(self.M.transpose(), self.M)
            self.w = np.log(np.nanstd(self.Y, axis=0))
        else:
            self.M = np.ones((self.Npix, self.num))
            self.w = 0.5 * np.ones(self.Npix)
        
        x0 = self.pack(self.M, self.w)
        
        # maximize likelihood
        if 1:
            res = minimize(self.lnL, x0, method='L-BFGS-B', jac=True) #, options={'maxiter':500})
            self.M, self.w = self.pack(res.x)
            self.PCA = self.M.transpose()
            self.toGUI('PCA')
            self.K = np.dot(self.M, self.M.transpose())
            self.w = np.exp(self.w)
            print('res:', self.K, self.w)
        #self.lnL(x)
        #self.minimize(self.lnL, args=(), jac=self.jac)

def SDSS_to_asinh(flux):
    """
    convert nannomagies to asinh values.
    parameters:
            - flux    : fluxes in nannomagies, should be array of shape (n, 5) of floats.
    
    return:
            - m       : magnitudes in asinh 
    """
    b = [0.28, 0.18, 0.24, 0.36, 1.5]
    m0 = [24.63, 25.11, 24.80, 24.36, 22.83]
    return - 1.086 * (np.arcsinh(flux / b)) + m0

def getIDfromName(s):
    name = s.replace('.fits', '').replace('.dat', '')
    return [int(i) for i in name[-15:].split('-')]
