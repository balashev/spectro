#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import scipy.optimize as opt
sys.path.append('C:/science/python/spectro')
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.optimize import minimize

from .a_unc import a

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
    def __init__(self, species):
        self.species = species

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



    def set_data(self, num=None, verbose=0):
        """
        set atomic data
        """

        if num is not None:
            self.num = num

        if self.species == 'H2':
            
            self.g = np.array([(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(self.num)])

            f_in = open('C:\science\spectra_program\synthetic\synthetic\Data\energy.dat', 'r')
            data = np.loadtxt(f_in, skiprows=2)
            self.E = data[:self.num,1]
            # transform energy from cm^-1 to Kelvins
            self.E = self.E/0.695
            
        if self.species == 'CI':
            self.g = np.array([2*i+1 for i in range(self.num)])
            data = [0, 16.42, 43.41]
            self.E = np.array(data[:self.num])
        
        if self.species == 'CO':
            self.g = np.array([2*i+1 for i in range(self.num)])
            data = [0, 1, 3, 6, 10]
            self.E = np.arange(self.num)*(np.arange(self.num)+1)*2.77
            print('E', self.E)

        if verbose:
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
            return sum(self.g*np.exp(-self.E/temp))
    
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
        
        if max(slope) < 0:
            s = a(self.slope, max(slope)-self.slope, self.slope-min(slope), 'd')
        else:
            s = a('<{:.2f}'.format(min(slope)), 'd')
        #z = a(self.zero, max(zero)-self.zero, self.zero-min(zero))
        print(s.latex(f=4))
        self.temp = -np.log10(np.exp(1))/s
        print(self.temp)
#        T_0 = -1 / self.slope
#        T_m = -1 / self.slope + 1 / min(slope)
#        if max(slope) < 0:
#            T_p = -1 / max(slope) + 1 / self.slope
#        else:
#            T_p = np.Inf
#            
        if self.plot:
            #plt.plot(y1,z)
            cs = self.regionplot.contour(x1, y1, z, levels=[1])
            p = cs.collections[0].get_paths()
            self.regionplot.plot(self.slope, self.zero, marker='+', color='k', ms=10, mew=3)
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
            popt, pcov = opt.curve_fit(self.linear_fit, self.E, column(self.y, 'val'),
                                       p0=[self.slope, self.zero], sigma=column(self.y, 'plus'))
            #self.dataplot.plot(self.E, popt[0]*self.E+popt[1], '--b', lw=2)
            self.slope = a(popt[0], np.sqrt(pcov[0,0]), np.sqrt(pcov[0,0]), 'd')
            self.zero = a(popt[1], np.sqrt(pcov[1,1]), np.sqrt(pcov[1,1]), 'd')
            print(self.slope, self.zero)
            T_0 = -np.log10(np.exp(1))/self.slope
            print(T_0.dec())
            print(T_0.latex())

            self.plot_region()

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

    def plot_data(self):
        self.dataplot.errorbar(self.E, column(self.n,'val'), yerr=[column(self.n,'plus'), column(self.n,'minus')], 
                               fmt='o', color='r' , ecolor='k', linewidth=0.5, markersize=8) 
        #self.plot_temp()
        self.dataplot.set_xlim(self.E[0]-10, self.E[-1]+20)
    
    def plot_temp(self, temp=None, Ntot=None, color='r'):
        if temp is None:
            self.dataplot.plot(self.E, self.slope * self.E + self.zero, '--', color=color, lw=2) 
        else:
            slope, zero = self.temp_to_slope(temp=temp, Ntot=Ntot)           
            self.dataplot.plot(self.E, slope * self.E + zero, '--', color=color, lw=2) 
    
    def plot_region(self):
        self.slope_to_temp()
        x = np.linspace(self.temp.val - self.temp.minus*2, self.temp.val + self.temp.plus*2, 100)
        y = np.linspace(self.Ntot.val - self.Ntot.minus*2, self.Ntot.val + self.Ntot.plus*2, 100)
        X, Y = np.meshgrid(x,y)        
        z = np.zeros_like(X)
        for i in range(len(x)):
            for k in range(len(y)):
                z[k,i] = self.lnL_temp(temp=x[i], Ntot=y[k])
        print(np.amax(z))
        ind_max = np.unravel_index(z.argmax(), z.shape)
        z = z - np.amax(z)
        cont = self.regionplot.contourf(X, Y, z, 300, cmap='plasma')
        self.regionplot.contour(X, Y, z, levels=[-1.5, -1, -0.5], colors='k')
        self.regionplot.scatter(X[ind_max], Y[ind_max], 40, marker='+')
        plt.colorbar(cont, orientation='vertical', shrink=0.8)
        self.regionplot.set_xlabel(r'T$_{\rm ex}$, K')        
        self.regionplot.set_ylabel(r'log N$_{\rm tot}$')
        for i in range(len(x)):
            for k in range(len(y)):
                if z[k,i] > -.5:
                    slope, zero = self.temp_to_slope(temp=x[i], Ntot=y[k])           
                    self.dataplot.plot(self.E, slope * self.E + zero, '-y', lw=0.5, alpha=0.1, zorder=1)
        self.regionplot.xaxis.set_minor_locator(AutoMinorLocator(10))
        self.regionplot.xaxis.set_major_locator(MultipleLocator(1))
        self.regionplot.yaxis.set_minor_locator(AutoMinorLocator(10))
        self.regionplot.yaxis.set_major_locator(MultipleLocator(0.1))

    def latex(self):
        return self.temp.dec().latex()

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
    
    if 1:
        species = 'H2'
        n.append(a(17.57, 0.12, 0.12))
        n.append(a(17.53, 0.22, 0.22))
    else:
        species = 'CO'
        n.append(a(14.43, 0.12, 0.12))
        n.append(a(14.52, 0.08, 0.08))
        n.append(a(14.33, 0.06, 0.06))
        n.append(a(13.73, 0.05, 0.05))
        #n.append(a(13.14, 0.13, 0.13))
        
    Temp = ExcitationTemp(species)
    #print(Temp.col_dens(num=4, Temp=92, Ntot=21.3))
    Temp.calcTemp(n, calc='boot', plot=1, verbose=1)
    #Temp.plot_temp(temp=9.9, Ntot=14.94, color='b')
    #Temp.latex()
    