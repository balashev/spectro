#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.optimize import minimize
import sys
sys.path.append('C:/science/python/')
from .a_unc import a
from .stats import distr1d, distr2d

def column(matrix, attr):
    return [getattr(row,attr) for row in matrix]

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
    def __init__(self, species, debug=False):
        self.species = species
        self.debug = debug

    def calcTemp(self, n, calc='boot', plot=0, verbose=0):
        """
        estimate excitation temperature for given column density
        parameters:
            - n           : column densities on levels
            - calc        : type of calculation, can be from ['chi2', 'boot']
                                    - chi2   : based of simple chi2 likelihood
                                    - boot   : bootstrap resampling method
            - plot        : plot some results
             
        return:
            None
        """
        self.n = copy.deepcopy(n)
        self.num = len(self.n)
        self.plot = plot
        self.verbose = verbose
        
        if self.plot:
            if 1:
                fig = plt.figure(figsize=(14,7))
                self.dataplot = fig.add_subplot(121)
                self.regionplot = fig.add_subplot(122)
            else:
                fig = plt.figure()
                self.dataplot = fig.add_subplot(121)
                fig = plt.figure()
                self.regionplot = fig.add_subplot(111)

        self.set_data()
        if self.verbose:
            print('Excitation temperature estimate from column densitites')
            print('species:', self.species)
            print('number of levels:', self.num)
            print('excitation energies:', self.E)
            print('N, cm^-2:', self.n)

        self.set_ratio()
        self.plot_data()
        self.calc()

        if calc == 'boot':
            self.boot()



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
            data = [0, 16.42, 43.41]
            self.E = np.array(data[:self.num])
        
        if self.species == 'CO':
            self.g = np.array([2*i+1 for i in range(self.num)])
            data = [0, 1, 3, 6, 10]
            self.E = np.arange(self.num)*(np.arange(self.num)+1)*2.77
            print('E', self.E)

        if self.species == 'FeII':
            self.g = np.asarray([9/2, 7/2, 5/2, 3/2, 1/2, 9/2, 7/2, 5/2, 3/2, 7/2, 5/2, 3/2, 1/2])[:num]
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
        self.y = self.n[:]

        for i in range(self.num):
            self.y[i].val -= np.log10(self.g[i])

    def Z(self, temp=None):
        """
        return stat. sum
        """
        if temp is None:
            if isinstance(self.temp, a):
                return sum(self.g*np.exp(-self.E/self.temp.val))
            if isinstance(self.temp, (int, float)):
                return sum(self.g*np.exp(-self.E/self.temp))
        else:
            return sum(self.g * np.exp(-self.E/temp))
    
    def slope_to_temp(self, slope=None, zero=None):
        """
        transform slope and zero point to temp and total column density
        """
        if slope is None:
            self.temp = -np.log10(np.exp(1))/self.slope
            self.Ntot = self.zero - np.log10(self.g[0]) + np.log10(self.Z())
        else:
            temp = -np.log10(np.exp(1))/slope
            Ntot = zero - np.log10(self.g[0]) + np.log10(self.Z(temp=temp))
            return temp, Ntot
        
    def temp_to_slope(self, temp=None, Ntot=None):
        """
        transform slope and zero point to temp and total column density
        """
        if temp is None:
            self.slope = -np.log10(np.exp(1))/self.temp
            self.zero = self.Ntot + np.log10(self.g[0]) - np.log10(self.Z())
        else:
            slope = -np.log10(np.exp(1))/temp
            zero = Ntot + np.log10(self.g[0]) - np.log10(self.Z(temp))
            return slope, zero
    
    def lnL_temp(self, temp=None, Ntot=None):
        """
        return likelihood function
        """

        lnL = 0
        if temp is None:
            self.temp_to_slope()
            x = self.slope * self.E + self.zero
        else:
            slope, zero = self.temp_to_slope(temp=temp, Ntot=Ntot)
            x = slope * self.E + zero
       
        for i in range(self.num):
            #print(self.temp, i, self.y[i].lnL(x[i]))
            lnL += self.y[i].lnL(x[i])
        
        return lnL
        
    def linear(self, slope=None, zero=None):
        """
        return likelihood function for linear approximation
        parameters:
            - slope       :
            - zero        :
        """
        Ln = 0
        for i in range(len(self.n)):
            if slope is None:
                x = self.slope * self.E[i] + self.zero
            else:
                x = slope * self.E[i] + zero
                
            Ln = Ln - self.y[i].lnL(x)
        
        #print(al)
        return Ln
    
    def linear_fit(self, x, slope, zero):

        return slope * x + zero

    def calc_two_levels(self, y=None, E=None):    
        
        if y is None:
            self.slope = (self.y[-1].val - self.y[0].val)/self.E[-1]
            self.zero = self.y[0].val
            
            if len(self.n)>2:
                res = minimize(self.linear_fit, [self.slope, self.zero], method='Nelder-Mead')
                self.slope, self.zero = res.x
            
            self.slope_to_temp()
            print(self.temp)
            
            if self.verbose:
                print('minimal values (slope, zeropoint):', self.slope, self.zero)
                print('minimal values (T, Ntot):', self.temp, self.Ntot)
        else:
            if isinstance(y[0], a):
                slope = (y[-1].val - y[0].val)/E[-1]
                zero = y[0].val

            
                if len(y)>2:
                    res = minimize(self.linear_fit, a[slope, zero], method='Nelder-Mead')
                    slope, zero = res.x

            elif isinstance(y[0], float):
                slope = (y[-1] - y[0]) / E[-1]
                zero = y[0]

            return slope, zero
   
    def calc_two_levels_err(self):
        z_min = self.linear()
    
        num = 300
      
        n1 = [a(),a()]
        n1[0].val = self.y[0].val - 2*self.y[0].minus
        n1[1].val = self.y[1].val + 2*self.y[1].plus
        al_min = self.calc_two_levels(n1, self.E)
        n1 = [a(),a()]
        n1[0].val = self.y[0].val + 2*self.y[0].plus
        n1[1].val = self.y[1].val - 2*self.y[1].minus
        al_max = self.calc_two_levels(n1, self.E)
        
        print(al_min, al_max)
        
        if self.plot:
            self.dataplot.plot(self.E, al_min[0] * self.E + al_min[1], '--k')         
            self.dataplot.plot(self.E, al_max[0] * self.E + al_max[1], '--k')
                
        x1 = np.linspace(al_min[0], al_max[0], num)
        y1 = np.linspace(al_min[1], al_max[1], num)
        z = np.zeros(shape=(num, num))
        #z = np.zeros(num)
        a1 = np.zeros(2)
        slope, zero  = [], []
        for i in range(num):
            for k in range(num):
                a1[0] = x1[i]
                a1[1] = y1[k]
                z[k,i] = self.linear(a1[0], a1[1]) - z_min
                #print(x1[i], y1[k], z[k,i])
                if z[k,i] < 1:
                    slope.append(a1[0])
                    zero.append(a1[1])
        if self.verbose: 
            print('slope range: ', min(slope), self.slope, max(slope))
        if self.verbose: 
            print('zeropoint range: ', min(zero), self.zero, max(zero))

        print(max(slope), min(slope))
        if max(slope) < 0:
            self.slope = a(self.slope, max(slope)-self.slope, self.slope-min(slope), 'd')
        else:
            self.slope = a('<{:.4f}'.format(min(slope)), 'd')
        self.zero = a(self.zero, max(zero) - self.zero, self.zero - min(zero), 'd')
        try:
            self.slope_to_temp()
        except:
            pass
        print(self.temp)

        if self.plot:
            #plt.plot(y1,z)
            cs = self.regionplot.contour(x1, y1, z, levels=[1])
            p = cs.collections[0].get_paths()
            self.regionplot.plot(self.slope.val, self.zero.val, marker='+', color='k', ms=10, mew=3)
            self.regionplot.plot(min(slope),max(zero), marker='+', color='k', ms=10, mew=3)    
            self.regionplot.plot(max(slope),min(zero), marker='+', color='k', ms=10, mew=3)
            s = "$T_{kin} = $ " + self.temp.dec().latex(f=2, base=2)
            self.regionplot.text(0.1, 0.1, s, ha='left', va='top', transform = self.regionplot.transAxes)
            
            for l in p[0].vertices:
                self.dataplot.plot(self.E, l[0]*self.E+l[1], 'g', alpha=0.1, zorder=0)
    
        #self.temp = a(T_0, T_p, T_m, 'd')
            
    def calc(self):

        self.slope = (self.y[-1].val - self.y[0].val) / self.E[-1]
        self.zero = self.y[0].val

        if self.num == 2:
            if any([y.plus == 0 or y.minus ==0 for y in self.y]):
                self.calc_two_levels()
            else:
                self.calc_two_levels_err()

        elif self.num > 2:

            if 1:
                self.curve_fit()
                self.ongrid()
            else:
                self.curve_fit()
            #self.temp = -np.log10(np.exp(1))/self.slope

    def curve_fit(self):
        popt, pcov = opt.curve_fit(self.linear_fit, self.E, column(self.y, 'val'),
                                   p0=[self.slope, self.zero], sigma=column(self.y, 'plus'))
        # self.dataplot.plot(self.E, popt[0]*self.E+popt[1], '--b', lw=2)
        self.slope = a(popt[0], np.sqrt(pcov[0, 0]), np.sqrt(pcov[0, 0]), 'd')
        self.zero = a(popt[1], np.sqrt(pcov[1, 1]), np.sqrt(pcov[1, 1]), 'd')
        self.slope_to_temp()

    def ongrid(self):
        self.slope_to_temp()
        x = np.linspace(self.temp.val - self.temp.minus, self.temp.val + self.temp.plus, 200)
        y = np.linspace(self.Ntot.val - self.Ntot.minus, self.Ntot.val + self.Ntot.plus, 200)
        X, Y = np.meshgrid(x, y)
        z = np.zeros_like(X)
        for i in range(len(x)):
            for k in range(len(y)):
                z[k, i] = self.lnL_temp(temp=x[i], Ntot=y[k])

        d = distr2d(x, y, np.exp(z))
        d.dopoint()
        d.dointerval()
        self.temp, self.Ntot = d.point[0], d.point[1]

        if self.plot:
            d.plot_contour(xlabel=r'T$_{\rm ex}$, K', ylabel=r'log N$_{\rm tot}$')
        for axis, attr, label, form in zip(['y', 'x'], ['temp', 'Ntot'], [r'T$_{\rm ex}$, K', r'log N$_{\rm tot}$'], ['d', 'l']):
            d1 = d.marginalize(axis)
            d1.dopoint()
            d1.dointerval()
            setattr(self, attr, a(d1.point, d1.point-d1.interval[0], d1.interval[1]-d1.point, 'd'))
            print(d1.latex(f=2))
            if self.plot:
                d1.plot(xlabel=label)
        self.temp_to_slope()

        if self.plot:
            plt.show()

    def boot(self, iter=1000):
        y_rvs = np.empty([self.num, iter], dtype=np.float)
        for i, y in enumerate(self.y):
            y_rvs[i] = y.rvs(iter)

        temp = []
        for i in range(iter):
            y = y_rvs[:, i]
            slope, zero = self.calc_two_levels(y=y, E=self.E)
            popt, pcov = opt.curve_fit(self.linear_fit, self.E, y,
                                       p0=[slope, zero])
            temp.append(self.slope_to_temp(slope=popt[0], zero=popt[1])[0])

        fig, ax = plt.subplots()
        ax.hist(temp, 30)
        plt.show()
        temp = np.sort(temp)
        print(temp[int(0.157*iter)], temp[int((1-0.157)*iter)])

        self.temp.dec()
        self.temp.plus, self.temp.minus = temp[int((1-0.157)*iter)] - self.temp.val, self.temp.val - temp[int(0.157*iter)]

    def plot_data(self, ax=None):
        if self.plot or ax is not None:
            ax = self.dataplot if ax is None else ax
            ax.errorbar(self.E, column(self.n,'val'), yerr=[column(self.n,'plus'), column(self.n,'minus')],
                                   fmt='o', color='r' , ecolor='k', linewidth=0.5, markersize=8)
            #self.plot_temp()
            ax.set_xlim(self.E[0]-10, self.E[-1]+20)
    
    def plot_temp(self, temp=None, Ntot=None, color='r', ax=None):
        if self.plot or ax is not None:
            ax = self.dataplot if ax is None else ax
            if temp is None:
                if isinstance(self.slope, a):
                    ax.plot(self.E, self.slope.val * self.E + self.zero.val, '--', color=color, lw=2)
                else:
                    ax.plot(self.E, self.slope * self.E + self.zero, '--', color=color, lw=2)
            else:
                slope, zero = self.temp_to_slope(temp=temp, Ntot=Ntot)
                ax.plot(self.E, slope * self.E + zero, '--', color=color, lw=2)
    
    def latex(self, f=2, base=None):
        if isinstance(self.temp, a):
            return self.temp.dec().latex(f=f, base=base)
        else:
            return self.temp

    def col_dens(self, num=3, Temp=50, Ntot=None):
        """
        return column densities of <num> first levels with given <Temp> excitation and total column density <Ntot>
        column densities in log format
        """
        self.set_data(num)

        col = self.g / self.g[0] * np.exp(-self.E / Temp)

        if Ntot is not None:
            col *= 10**Ntot * self.g[0] / self.Z(Temp)

        return np.log10(col)

    def rescale_col_dens(self, N, Ntot):
        Nt = np.log10(np.sum(np.power(10, N)))
        return N + Ntot - Nt

if __name__ == '__main__':

    n = []

    species = 'FeII'
    Temp = ExcitationTemp(species, debug=True)

    if species == 'H2':
        n.append(a(17.57, 0.12, 0.12))
        n.append(a(17.53, 0.22, 0.22))
        Temp.calcTemp(n, calc='boot', plot=1, verbose=1)

    elif species == 'CO':
        species = 'CO'
        n.append(a(14.43, 0.12, 0.12))
        n.append(a(14.52, 0.08, 0.08))
        n.append(a(14.33, 0.06, 0.06))
        n.append(a(13.73, 0.05, 0.05))
        #n.append(a(13.14, 0.13, 0.13))
        Temp.calcTemp(n, calc='boot', plot=1, verbose=1)

    elif species == 'FeII':
        print(Temp.col_dens(num=13, T=15000))
    #Temp.plot_temp(temp=9.9, Ntot=14.94, color='b')
    #Temp.latex()
    