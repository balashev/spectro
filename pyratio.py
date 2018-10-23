#!/usr/bin/env python
# -*- coding: utf-8 -*-

import astropy.constants as ac
import astropy.units as au
from chainconsumer import ChainConsumer
import collections
import corner
import emcee
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pickle
from scipy import interpolate, optimize
from scipy.stats import chi2
import os, sys
sys.path.append('C:/Science/python')
from spectro import colors as col
from spectro.a_unc import a
from spectro.stats import distr1d, distr2d
from spectro.sviewer.utils import printProgressBar, Timer

def smooth_step(x, delta):
    x = np.clip(x/delta, 0, 1)
    return  x**3 * (x * (x*6 - 15) + 10)

class speci:
    """
    class of all necessary species data for calculations
    
        - name       : name of species
        - n          : list of column densities (each column density is <a> object values
                                                 i.e. can be with assymetric errors)
    """
    def __init__(self, parent, name, n, num=None):
        self.parent = parent
        self.name = name
        if num is None:
            self.num = len(n)  # number of levels with estimated column density
        else:
            self.num = num
        self.set_names()
        self.g = np.zeros(self.num)
        self.E = np.zeros(self.num)
        self.descr = [''] * self.num
        self.coll = {}
        self.A = np.zeros([self.num, self.num])
        self.B = np.zeros([self.num, self.num])
        self.n = n
        self.mask = [n is not None for n in self.n]
        self.const2 = (ac.h.cgs**2/(2*ac.m_e.cgs*np.pi)**(1.5)/(ac.k_B.cgs)**0.5).value
        
        # >> set population ratios
        self.y = [None] * (np.sum(self.mask) - 1)
        ind = np.where(self.mask)[0]
        if n[ind[0]].val != 0:
            for i, k in enumerate(ind[1:]):
                self.y[i] = (ind[0], k, self.n[k] / self.n[ind[0]])   #formatof y[i]: (level_1, level_2, ratio of column densities: level 2 / level 1)
        else:
            self.y = [(0, 0, a())]*(self.num-1)

        # >> calc total column density
        self.n_tot = a(0, 0, 0)
        for ni in n:
            if ni is not None:
                self.n_tot = ni + self.n_tot

        # make collisional data
        part = ['H', 'oH2', 'pH2', 'p', 'e', 'He4']
        for p in part:
            self.coll[p] = coll_list.make_zero(self, p, self.num)

        # read data from files
        if name == 'HD':
            self.readFlowerHD()
        elif name == 'CO':
            self.readCO()
        else:    
            self.read_popratio()

        self.fullnum = self.E.shape[0] # full number of levels

        self.setEij()
        self.setBij()
        self.setAij()

        #print(self.stats, self.energy, self.descr, self.Aij)
        #print(self.coll[2].rate)
    
    def set_stats(self):
        if self.name == 'SiII':
            self.g = [i*2+1 for i in range(self.num)]
        if self.name == 'CII':
            self.g = [i*2 for i in range(self.num)]

    def set_names(self):
        if self.name in ['CI', 'CII', 'SiII', 'OI']:
            self.names = [self.name + '*'*k for k in range(self.num)]
        elif self.name in ['HD', 'CO']:
            self.names = [self.name + 'j'+ str(k) for k in range(self.num)]

    def coll_rate(self, part, i, j, T):
        l = self.coll[part].rate(i, j)
        #print(l, part, i, j)
        if l!=0:
            return self.coll[abs(l)-1].rate(T, sign=np.sign(l))
        else:
            return 0
            
    def coll_ind(self, part, i, j):
        for l in range(len(self.coll)):
            if self.coll[l].part == part and self.coll[l].i == i and self.coll[l].j == j:
                return l+1
            if self.coll[l].part == part and self.coll[l].i == j and self.coll[l].j == i:
                return -(l+1)
        return 0
        
    def read_popratio(self):
        """
        read popratio data for species with given name and numbers of levels
        """
        with open(self.parent.folder+'/data/pyratio/'+self.name+'.dat') as f_in:
            f = 0
            n_coll = 0
            while True:
                line=f_in.readline()
                if not line: break
                if line.strip() == '#': 
                    f +=1
                    if f == 4:
                        l = 0
                        line = f_in.readline()
                        #print(line)
                        while line.strip() != '':
                            if l < self.num: 
                                self.g[l] = line.split()[1]
                                self.E[l] = line.split()[0]
                                self.descr[l] = line.split()[3]+line.split()[4]+line.split()[5]
                                l +=1
                            line = f_in.readline()
                    if f == 5:
                        n = int(f_in.readline())
                        for l in range(n):
                            line = f_in.readline()
                            i = int(line.split()[0])-1
                            j = int(line.split()[1])-1
                            #print(i,j)
                            if i < self.num and j < self.num:
                                self.A[j, i] = float(line.split()[2])
                            #    print(float(line.split()[2]))
                            #print(self.A)
                    if f == 6:
                        n = int(f_in.readline())
                        data = []
                        for l in range(n):
                            data.append(list(map(float, f_in.readline().split())))
                        data = np.array(data)
                        E, inds = np.unique(data.transpose()[0], return_index=True)
                        self.E = np.append(self.E, E)
                        self.g = np.append(self.g, data.transpose()[1][inds])
                        for E in self.E[self.num:]:
                            A = np.zeros(self.num)
                            for d in data:
                                if d[0] == E:
                                    #self.g = np.append(self.g, d[1])
                                    A[int(d[2])-1] = d[3]
                            self.A = np.append(self.A, [A], axis=0)
                        #print(self.E)
                        #print(self.g)
                        #print(self.A)

                    if f == 7:
                        n_coll = int(f_in.readline()) 
                        #print(n_coll)
                    if 7 < f < 8 + n_coll:
                        line = f_in.readline().strip()
                        f_in.readline()
                        n_1 = int(f_in.readline().strip())
                        d = {'electron': 'e', 'proton': 'p', 'H0': 'H', 'para-H2': 'pH2', 'ortho-H2': 'oH2', 'helium': 'He4'}
                        coll = d[line]
                        l = 0
                        c = coll_list()
                        while l < n_1:
                            line = f_in.readline()
                            l += int(line.split()[0])
                            n_2 = int(line.split()[1])
                            #print(l, n_2)
                            Temp = [float(x) for x in f_in.readline().strip().split()]
                            for m in range(int(line.split()[0])):
                                line = f_in.readline().split()
                                i = int(line[0])-1
                                j = int(line[1])-1
                                if i <= self.num and j <= self.num:
                                    #print(coll, i, j, np.log10(Temp), line[2:])
                                    c.append(collision(self, coll, i, j, np.array([np.log10(Temp), [np.log10(float(line[k+2])) for k in range(n_2)]])))
                        self.coll[coll] = c

    def readFlowerHD(self):
        """
        read Flower data for HD molecule
        """
        folder = self.parent.folder + '/data/pyratio/'
        # read collisional cross sections
        files = ['Flower2000_qh1hd.dat', 'Flower2000_qhehd.dat', 'Flower2000_qoh2hd.dat', 'Flower2000_qph2hd.dat']
        part = ['H', 'He4', 'oH2', 'pH2']
        dt = [2/3, 1/3, 1/2, 1/2]
        T = np.logspace(np.log10(30), np.log10(3000), 30) 
        for i, file in enumerate(files):
            with open(folder+'HD/'+file, 'r') as f_in:
                k = int(f_in.readline())
                c = coll_list()
                for l in range(k):
                    words = f_in.readline().split()
                    rate = np.zeros(len(T))
                    for k, t in enumerate(T):
                        rate[k] = float(words[2])+float(words[3])/(t*1e-3+dt[i]) + float(words[4])/(t*1e-3+dt[i])**2
                    c.append(collision(self, part[i], int(words[0])-1, int(words[1])-1, np.array([np.log10(T), rate])))
                self.coll[part[i]] = c
        with open(folder+'HD/HD.dat') as f_in:
            f = 0
            while True:
                line=f_in.readline()
                if not line: break
                if line.strip() == '#': 
                    f +=1
                    if f==4:
                        l = 0
                        line = f_in.readline()
                        while line.strip() != '':
                            if l < self.num: 
                                self.g[l] = line.split()[1]
                                self.E[l] = line.split()[0]
                                self.descr[l] = line.split()[3]
                                l +=1
                            line = f_in.readline()
                    if f == 5:
                        n = int(f_in.readline())
                        for l in range(n):
                            line = f_in.readline()
                            i = int(line.split()[0])-1
                            j = int(line.split()[1])-1
                            if i < self.num and j < self.num:
                                self.A[j, i] = float(line.split()[2])

    def readCO(self):
        """
        read data for CO: einstein coefficients, collisional excitation rates
        """
        #self.Lambda_read('data/pyratio/co_data_old.dat')
        self.Lambda_read(self.parent.folder + '/data/pyratio/co_data.dat')

    def Lambda_pars(self, line):
        remove = ['!', '\n', '(cm^-1)']
        for r in remove:
            line = line.replace(r, '')
        return line.split(' + ')

    def Lambda_read(self, file):
        """
        read data from <file> given in Lambda format

            - file    : filename
        """
        with open(file) as f_in:
            while True:
                line = f_in.readline()
                if  '!NUMBER OF ENERGY LEVELS' in line:
                    num = int(f_in.readline())
                    self.E = np.zeros(num)
                    self.g = np.zeros(num)
                    d = self.Lambda_pars(f_in.readline())
                    for i in range(num):
                        line = f_in.readline().split()
                        dic = dict((k, line[i]) for i,k in enumerate(d))
                        self.E[int(dic['J'])] = float(dic['ENERGIES'])
                        self.g[int(dic['J'])] = float(dic['WEIGHT'])

                if  '!NUMBER OF RADIATIVE TRANSITIONS' in line:
                    num = int(f_in.readline())
                    self.A = np.zeros([num+1, num+1])
                    self.B = np.zeros([num+1, num+1])
                    d = self.Lambda_pars(f_in.readline())
                    for i in range(num):
                        line = f_in.readline().split()
                        dic = dict((k, line[i]) for i,k in enumerate(d))
                        self.A[int(dic['UP'])-1, int(dic['LOW'])-1] = float(dic['EINSTEINA(s^-1)'])

                if '!COLLISIONS BETWEEN' in line:
                    c = coll_list()
                    coll = f_in.readline().split()[1].replace(self.name, '').replace('-', '')
                    f_in.readline()
                    num = int(f_in.readline())
                    for i in range(3):
                        f_in.readline()
                    temp = [float(x) for x in f_in.readline().split()]
                    d = self.Lambda_pars(f_in.readline())
                    for i in range(num):
                        line = f_in.readline().split()
                        dic = dict((k, line[i]) for i, k in enumerate(d))
                        c.append(collision(self, coll, int(dic['UP'])-1, int(dic['LOW'])-1, np.array(
                            [np.log10(temp), [np.log10(float(line[k + 3])) for k in range(len(temp))]])))
                    self.coll[coll] = c

                if not line:
                    break

    def setEij(self):
        """
        set energy difference matrix for calaulations
        """
        a = np.lib.stride_tricks.as_strided(self.E[:self.num], (self.E[:self.num].size, self.E[:self.num].size),
                                            (0, self.E[:self.num].itemsize))
        self.Eij = np.abs(a - a.transpose())

    def setBij(self):
        """
        set Einstein B_ij and B_ji coefficients from A_ij
        """
        self.B = np.zeros([self.fullnum, self.fullnum])
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                if self.A[i,j] != 0:
                    self.B[i,j] = self.A[i,j]*(self.E[i]-self.E[j])**(-3)/8/np.pi/ac.h.cgs.value
                    self.B[j,i] = self.B[i,j]*self.g[i]/self.g[j]

        self.Bij = self.B[:self.num, :self.num]

    def setAij(self):
        self.Aij = self.A[:self.num, :self.num]

    def plot_allCollCrossSect(self, ax=None):
        """
        plot collision cross section with given particles
        parameters:
            - part     : collision partner
            - i        : number of lower level of transition
            - j        : number of higher level of transition
            - ax       : Axis object where to plot, if None then create
        """
        partners = ['e', 'p', 'H', 'oH2', 'pH2', 'He4']
        for part in partners:
            axs = self.coll[part].plot_all(ax=ax)
            axs.set_title(self.name + ' collision with ' + part)
        return axs

class coll_list(list):
    """
    Class that contains lists of collisional data for specific collisional partner
    """
    def __init__(self):
        self = []

    def find(self, i, j):
        for s in self:
            if s.i == i and s.j == j:
                return s, 1
            elif s.i == j and s.j == i:
                return s, -1
        return None, 0

    def rate(self, i, j, T):
        s, l =  self.find(i, j)
        if l != 0:
            r = s.rate(T, sign=l)
            return r
        else:
            return 0

    def plot(self, i, j, ax=None, label=None):
        s, l = self.find(i, j)
        if l != 0:
            s.plot(ax=ax, sign=l, label=label)

    def plot_all(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        for s in self:
            self.plot(s.i, s.j, ax=ax)
        return ax

    @classmethod
    def make_zero(cls, parent, part, num):
        s = cls()
        for i in range(1,num):
            for k in range(i):
                s.append(collision(parent, part, i, k))
        return s

class collision():
    """
    Class for individual collisions data
    """
    def __init__(self, parent, part, i, j, rate=None):
        self.parent = parent
        self.part = part
        self.i = i
        self.j = j
        self.rates = rate
        try:
            self.rate_int = interpolate.InterpolatedUnivariateSpline(rate[0], rate[1], k=2)
        except:
            pass

    def __str__(self):
        return "{0} collision rate with {1} for {2} --> {3}".format(self.parent.name, self.part, self.i, self.j)

    def rate(self, T, sign=1):
        if self.rates is None:
            return 0

        rate = self.rate_int(T)
        if self.parent.name == 'e':
            c1 = self.const2 * T ** (-0.5) / self.g[min(self.i, self.j)]
        else:
            c1 = 1

        if sign == 1:
            Bf = 1
        else:
            l, u = max(self.i, self.j), min(self.i, self.j)
            if self.i > self.j:
                u, l = l, u
            Bf = self.parent.g[u] / self.parent.g[l] * np.exp(-(self.parent.E[u] - self.parent.E[l]) / 0.695 / 10**T)
        return 10 ** rate * c1 * Bf

    def plot(self, ax=None, label=None, sign=1):
        if self.rates is not None:
            if ax == None:
                fig, ax = plt.subplots(1)
            ax.scatter(self.rates[0], np.log10(self.rate(self.rates[0], sign=sign)), marker='+', label=label)
            if 0:
                T = np.linspace(np.min(self.rates[0]), np.max(self.rates[0]), 100)
                ax.plot(T, np.log10(self.rate(T, sign=sign)), '-k')
        #plt.show()

class par():
    """
    class for parameters handle:
        - name         : name of parameter, can be:
                            'n' - number density 
                            'T' - temperature 
                            'f' - molecular fraction
                            'UV' - UV flux in the units of Habing? field value
                            'H' - atomic hydrogen density
                            'e' - electron density
                            'H2' - molecular hydrogen density 
        - label        : label on plot
        - init         : init value
        - range        : range of parameter values, 2 element array
        - init_range   : range of initial parameter values for MCMC, 2 element array
        - prior        : prior values for the calculations, <a> object
    """
    def __init__(self, name, prior=None, parent=None):
        self.name = name
        self.prior = prior
        self.parent = parent

        # >>> particle number density
        if name == 'Ntot':
            self.label = r'$\log\,N_{\rm tot}$'
            self.init = 2
            self.range = [1, 25]
            self.init_range = 0.2

        # >>> particle number density
        if name == 'n':
            self.mol_fr = 0 # in log10 s !!!
            self.label = r'$\log\,$n'
            self.init = 2
            self.range = [0, 3]
            self.init_range = 0.3
        
        # >>> temperature
        if name == 'T':
            self.label = r'$\log\,$T'
            self.init = 2
            self.range = [1, 3]
            self.init_range = 0.3

        # >>> molecular fraction
        if name == 'f':
            name = label = r'$\log\,f$'
            self.init = -0.5
            self.range = [-3, 0]
            self.init_range = 0.3
        
        # >>> UV flux
        if name == 'UV':
            self.label = r'$\log\,\xi$'
            self.init = 0
            self.range = [-1, 2]
            self.init_range = 0.3
            
       # >>> electron number density
        if name == 'e':
            self.label = r'$\log\,\rm n_e$'
            self.init = -1.8
            self.range = [-2, -5]
            self.init_range = 0.3
            
        # >>> atomic hydrogen number density
        if name == 'H':
            self.label = r'$\log\,\rm n_H$'
            self.init = 2
            self.range = [1, 4]
            self.init_range = 0.3
            
        # >>> molecular hydrogen number density
        if name == 'H2':
            self.otop = 1
            self.label = r'$\log\,\rm n_{H2}$'
            self.init = 2
            self.range = [1, 4] 
            self.init_range = 0.3

        # >>> CMB temperature
        if name == 'CMB':
            self.label = r'T$_{\rm CMB}$'
            self.init = 2.72548 * (1 + self.parent.z)
            self.range = [1, 20]
            self.init_range = 2
       
        self.value = self.init
        
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
        
class pyratio():
    """
    class for population ratio calculations:
    
        - species     : list of the species data, including, e.g. [SiII, CI, OI, HD, CO], e.t.c.
                        with column densities of populated levels, len(n[i]) is 
                        automatically number of levels in analysis.
        - parlist     : list of parameters to control population ratios
                        e.g. ['n', 'T', 'f', 'H', 'e', 'H2']
                        'n'    - number density,
                        'T'    - temperature,
                        'f'    - molecular fraction,
                        'H'    - atomic hydrogen density,
                        'e'    - electron density,
                        'H2'   - molecular hydrogen density
                        'Ntot' - total column density of species

        - z           : redshift of the medium, need for CMB calculations
        - f_He        : fraction of Helium nuclei, respect to hydrogen nuclei
        - UVB         : include UV backgroun in calculation
        - logs        : if logs==1 compare log values of ratios when calculating likelihood.
        - calctype    : type of calculation to be performed:
                            allowed:
                            1. popratios
                            2. numbdens
                            3. numbdens_min

            
    """
    def __init__(self, z=0, f_He=0.085, UVB=True, pars=None, conf_levels=[0.68269], logs=1, calctype='popratios'):
        self.species = collections.OrderedDict()
        self.pars = collections.OrderedDict()
        self.Planck1 = 8*np.pi*ac.h.cgs.value
        self.Planck2 = (ac.h.cgs/ac.k_B.cgs*ac.c.cgs).value
        self.z = z
        self.logs = logs
        self.f_He = f_He
        self.theta_range = []
        self.calctype = calctype
        self.folder = os.path.dirname(os.path.realpath(__file__))
        self.UVB = UVB
        if self.UVB:
            self.set_UVB()
        if self.calctype == 'numbdens':
            self.set_pars(['Ntot'])
        elif pars is not None:
            self.set_pars(pars)
        self.conf_levels = conf_levels
        self.timer = Timer()

    def add_spec(self, name, n=None, num=None):
        d = {'OI': 3, 'CI': 3, 'CII': 2, 'SiII': 2, 'HD': 3, 'CO': 15}
        if n is not None:
            n += [None] * (d[name]- len(n))
        else:
            n = [a()] * d[name]

        print('add_spec:', n)
        self.species[name] = speci(self, name, n, num)

    def set_pars(self, parlist):
        for p in parlist:
            self.pars[p] = par(p, parent=self)

    def set_prior(self, par, prior):
        for p in self.pars:
            if p == par:
                if isinstance(prior, (int, float)):
                    self.pars[p].prior = a(prior, 0, 0)
                    self.pars[p].value = prior
                elif isinstance(prior, a):
                    self.pars[p].prior = prior
                    self.pars[p].value = prior.val
    
    def set_fixed(self, par, value):
        self.pars[par].prior = a(value, 0, 0)
        self.pars[par].value = value
        self.pars[par].init = value

    def get_vary(self):
        self.vary = []
        for p in self.pars.items():
            if p[1].prior is not None and p[1].prior.plus == 0 and p[1].prior.minus == 0:
                p[1].value = p[1].prior.val
            else:
                self.vary.append(p[0])
        return self.vary

    def print_species(self):
        for s in self.species.values():
            print('species: ', s.name)
            print('number of levels: ', s.num)
            print('column densities: ')
            for i in np.where(s.mask)[0]:
                print('{0} level: {1}'.format(i, s.n[i].log()))
            print('level population ratios: ')
            for i in s.y:
                print('{1}/{0} level: {2}'.format(i[0], i[1], i[2]))
            
        # some checking calculations
        if 0==1:
            T = 3
            print(sp.coll_rate('e',0,1,T), sp.coll_rate('e',1,0,T), np.exp(-(sp.E[1]-sp.E[0])/0.695/10**T)*sp.g[0]/sp.g[1])
            input()

        if 0==1:
            fig, ax0 = plt.subplots()
            plot_CollCrossSect(sp, 'oH2', 1, 0, ax0)
            plot_CollCrossSect(sp, 'pH2', 1, 0, ax0)
            plot_CollCrossSect(sp, 'H', 1, 0, ax0)
            plot_CollCrossSect(sp, 'He4', 1, 0, ax0)
      
    def print_pars(self):
        
        print('parameters in calculations: ', [p for p in self.pars])
        for p in self.pars.values():
            p.show()

    def set_UVB(self):
        files = glob.glob(self.folder+'/data/pyratio/KS18_Fiducial_Q18/*.txt')
        z = [float(f[-8:-4]) for f in files]
        try:
            ind = z.index(self.z)
            data = np.genfromtxt(files[ind], comments='#', unpack=True)
            self.uvb = interpolate.interp1d(data[0], data[1], fill_value='extrapolate')
        except:
            ind = np.searchsorted(z, self.z)
            data_min = np.genfromtxt(files[ind - 1], comments='#', unpack=True)
            data_max = np.genfromtxt(files[ind], comments='#', unpack=True)
            self.uvb = interpolate.interp1d(data_min[0], data_min[1] + (data_max[1] - data_max[1]) * (self.z - z[ind]) / (z[ind+1] - z[ind]), fill_value='extrapolate')

    def u_CMB(self, nu):
        """
        calculate CMB flux density at given wavelenght array frequencies
        parameters:
            - nu      : array of energies [in cm^-1] where u_CMB is calculated
        """
        if 'CMB' in self.pars.keys():
            temp = self.pars['CMB'].value
        else:
            temp = 2.72548 * (1 + self.z)

        R = np.zeros_like(nu)
        ind = np.where(nu != 0)
        R[ind] = self.Planck1 * nu[ind]**3 / (np.exp(self.Planck2 / temp * nu[ind])-1)

        return R
    
    def balance(self, name, debug=None):
        """
        calculate balance matrix for population of levels
        parameters:
            - name       :  name of the species
            - debug      :  if is not None, than return balance matrix for specified process
                            notations corresponds to:
                              'A'    -  spontangeous transitions
                              'CMB'  -  excitation by CMB
                              'C'    -  by collisions
                              'UV'   -  by uv pumping

        """
        for speci in self.species.values():
            if speci.name == name:
                break
        W = np.zeros([speci.num, speci.num])
        #self.timer.time('rates in')

        #self.timer.time('A')
        if debug in [None, 'A']:
            W += speci.Aij

        #self.timer.time('CMB')
        if debug in [None, 'CMB']:
            W += self.u_CMB(speci.Eij) * speci.Bij

        #self.timer.time('Coll')
        if debug in [None, 'C']:
            for u in range(speci.num):
                for l in range(speci.num):
                    if any(x in self.pars.keys() for x in ['n', 'e', 'H2', 'H']):
                        W[u, l] += self.collision_rate(speci, u, l)

        if debug in [None, 'UV']:
            for u in range(speci.num):
                for l in range(speci.num):
                    if 'UV' in self.pars:
                        W[u, l] += self.pumping_rate(speci, u, l, x=10**self.pars['UV'].value)
                        #print('Ratio:',  self.pumping_rate(speci, u, l, x=0) / self.pumping_rate(speci, u, l, x=0.15))

        #self.timer.time('solve')
        if debug is None:
            K = np.transpose(W).copy()
            for i in range(speci.num):
                for k in range(speci.num):
                    K[i, i] -= W[i, k]
            return np.insert(np.abs(np.linalg.solve(K[1:, 1:], -K[1:, 0])), 0, 1)

        elif debug in ['A', 'CMB', 'C', 'UV']:
            return W

        else:
            return None
    
    def collision_rate(self, speci, u, l, verbose=False):
        """
        calculates collisional excitation rates for l -> u levels of given species
        """
        coll = 0
        if verbose:
            print(speci.name, set(['e', 'H', 'H2', 'n']).intersection(self.pars))
        for p in self.pars:
            if p in ['e', 'H']:
                coll += 10 ** self.pars[p].value*speci.coll[p].rate(u, l, self.pars['T'].value)
            if p in ['H2']:
                otop = 9 * np.exp(-170. /10 ** (self.pars['T']))
                coll += 10 ** self.pars[p].value / (1+otop) * speci.coll['pH2'].rate(u, l, self.pars['T'].value)
                coll += 10 ** self.pars[p].value * otop/(1+otop) * speci.coll['oH2'].rate(u, l, self.pars['T'].value)
            if p in 'n':
                m_fr = 10 ** self.pars['f'].value if 'f' in self.pars else mol_fr
                f_HI, f_H2 = (1-m_fr) / (self.f_He + 1 - m_fr / 2), m_fr / 2 / (self.f_He + 1 - m_fr / 2)
                coll += 10 ** self.pars['n'].value * speci.coll['H'].rate(u, l, self.pars['T'].value) * f_HI
                #print(coll, 10 ** (self.pars['T'].value), speci.coll['H'].rate(u, l, self.pars['T'].value))
                otop = 9 * np.exp(-170.6 / 10 ** (self.pars['T'].value))
                coll += 10 ** self.pars['n'].value * speci.coll['pH2'].rate(u, l, self.pars['T'].value) * f_H2 / (1 + otop)
                #print(u, l, speci.coll['pH2'].rate(u, l, self.pars['T'].value), f_H2 / (1 + otop))
                coll += 10 ** self.pars['n'].value * speci.coll['oH2'].rate(u, l, self.pars['T'].value) * f_H2 * otop / (1 + otop)
                #print('fractions:', f_HI, self.f_He / (self.f_He + 1 - m_fr / 2), f_H2 / (1 + otop), f_H2 * otop / (1 + otop))
                #print(u, l, speci.coll['oH2'].rate(u, l, self.pars['T'].value), f_H2 * otop / (1 + otop))
                if self.f_He !=0:
                    coll += 10 ** self.pars['n'].value * speci.coll['He4'].rate(u, l, self.pars['T'].value) * self.f_He / (self.f_He + 1 - m_fr / 2)
                    #print(coll, 10 ** self.pars['n'].value, self.f_He, speci.coll['He4'].rate(u, l, self.pars['T'].value))

        return coll
    
    def pumping_rate(self, speci, u, l, x=1):
        """
        calculates fluorescence excitattion rates for l -> u transition of given species
        x is the
        """
        pump = 0
        for k in range(speci.num, speci.fullnum):
            if speci.A[k,l] != 0:
                s = 0
                for i in range(speci.num):
                    s += speci.A[k,i] + self.exc_rate(speci, k, i, galactic=x)
                pump += self.exc_rate(speci, u, k, galactic=x) * (speci.A[k,l] + self.exc_rate(speci, k, l, galactic=x)) / s
        return pump
    
    def exc_rate(self, speci, u, l, galactic=1):
        """
        calculates excitation rates (B_ul * u(\nu)) in specified field (by u), 
                            for l -> u transition of given species 
        """
        nu = abs(speci.E[u]-speci.E[l])*ac.c.cgs.value
        #print('exc rates', ac.c.cgs.value / nu * 1e8, galactic, self.uv(nu, kind='Draine') * galactic / self.uv(nu, kind='UVB'))
        return speci.B[u,l] * (self.uv(nu, kind='Draine') * galactic + self.uv(nu, kind='UVB')) / nu
        
    def uv(self, nu, kind='Draine'):
        """
        UV field density in [erg/cm^3]
        parameters:
            - nu      :      frequency in [Hz]
            - kind    :      type of interstellar field: 
                                    - Habing (Habing 1968)
                                    - Draine (Draine 1978)
                                    - Mathis (Mathis 1983)
        return:
            uv       : energy density of UV field in [erg/cm^3]
        """

        if kind=='UVB':
            if self.UVB:
                #print(ac.c.cgs.value / nu * 1e8, self.uvb(ac.c.cgs.value / nu * 1e8))
                return self.uvb(ac.c.cgs.value / nu * 1e8) * 4 * np.pi / ac.c.cgs.value * nu
            else:
                return 0

        if kind == 'Habing':
            return 4e-14 * np.ones_like(nu)
            
        if kind == 'Draine':
            l = ac.c.cgs.value / nu / 1e-5  # in 100 nm
            return 6.84e-14 * l**(-5) * (31.016 * l**2 - 49.913 * l + 19.897)
            
        if kind == 'Mathis':
            l = ac.c.cgs.value / nu / 1e-4 # in 1 mkm
            if isinstance(l, float):
                l = np.array(l)
            uvm = np.zeros_like(l)
            mask = np.logical_and(0.134 < l, l < 0.245)
            uvm[mask] = 2.373e-14 * l[mask]**(-0.6678)
            mask = np.logical_and(0.110 < l, l < 0.134)
            uvm[mask] = 6.825e-13 * l[mask]
            mask = np.logical_and(0.091 < l, l < 0.110)
            uvm[mask] = 1.287e-9 * l[mask]**4.4172
            return uvm

    def lnpops(self):
        """
        Calculates the likelihood function for the model using population ratios on the levels

        return: ln
            - ln        : a value of log likelihood
        """
        ln = 0
        for sp in self.species.values():
            f = self.balance(sp.name)
            for y in sp.y:
                z = f[y[1]] / f[y[0]]
                if self.logs:
                    y[2].log()
                    z = np.log10(z)
                if y[2].type == 'm':
                    ln += y[2].lnL(z, ind=2)
                elif y[2].type == 'u':
                    delta = 0.2
                    ln += -0.5*((1 - smooth_step(y[2].val - z + delta, delta))*2)**2
                elif y.type == 'l':
                    delta = 0.2
                    ln += -0.5*(smooth_step(y[2].val - z + delta, delta)*2)**2
        return ln


    def lndens(self):
        """
        Calculates the likelihood function for the model using total volume (number) density as nuisance parameter

        return: ln
            - ln        : a value of log likelihood
        """
        ln = 0
        for sp in self.species.values():
            f = self.balance(sp.name)
            f /= np.sum(f[sp.mask])
            if self.logs:
                y = np.log10(f[sp.mask]) + self.pars['Ntot'].value
            else:
                y = f[sp.mask] * 10 ** self.pars['Ntot'].value
            for y, n in zip(y, np.asarray(sp.n)[sp.mask]):
                if self.logs:
                    n.log()
                if n.type == 'm':
                    ln += n.lnL(y, ind=2)
                elif n.type == 'u':
                    delta = 0.2
                    ln += -0.5*((1 - smooth_step(n.val - y + delta, delta))*2)**2
                elif n.type == 'l':
                    delta = 0.2
                    ln += -0.5*(smooth_step(n.val - y + delta, delta)*2)**2

        return ln

    def lndensmin(self, x, f, n):
        ln = 0
        for fi, ni in zip(f, n):
            if ni is not None:
                ni.log()
                fi = np.log10(fi)
                if ni.type == 'm':
                    #print(ni, fi, x[0] + fi)
                    ln += ni.lnL(x[0] + fi, ind=2)
                elif ni.type == 'u':
                    delta = 0.2
                    ln += -0.5 * ((1 - smooth_step(10**ni.val - f[i] * 10**x + delta, delta)) * 2) ** 2
                elif ni.type == 'l':
                    delta = 0.2
                    ln += -0.5 * (smooth_step(10**ni.val - f[i] * 10**x + delta, delta) * 2) ** 2
        return -ln

    def lndens_min(self):
        ln = 0
        for sp in self.species.values():
            f = self.balance(sp.name)
            res = optimize.minimize(self.lndensmin, sp.n_tot.val, args=(f, sp.n), method='Nelder-Mead', tol=1e-3)
            ln += -self.lndensmin(res.x, f, sp.n)
        return ln

    def lnlike(self):
        """
        Calculates the likelihood function for the model.
        It supposts two types of calculation:
                1. using population ratios on the levels.
                2. using measurements of number density of each level (i.e. column density in one zone approximation),
                   minimizing over total number density at each step.
                3. The same as 2, but suggesting that total number density is a nuisance parameter.

        return: ln
            - ln        : a value of log likelihood
        """
        if self.calctype == 'popratios':

            return self.lnpops()

        elif self.calctype == 'numbdens':

            return self.lndens()

        elif self.calctype == 'numbdens_min':

            return self.lndens_min()


    # for MCMC calculation
    def lnprobMCMC(self, values):
        """
        Calculates likelihood for MCMC runs.
        """
        for p, v in zip(self.vary, values):
            self.pars[p].value = v

        return self.lnprob()

    def lnprob(self):
        """
        Calculates likelihood with priors.
        """
        lp = self.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike()
    
    def lnprior(self):
        """
        Adding prior on the parameters to the likelihood
        """
        pri = 0
        for p in self.pars.values():
            if p.value < p.range[0] or p.value > p.range[1]:
                pri = -np.inf
                return pri
            if p.prior is not None:
                if p.prior.plus != 0 and p.prior.minus != 0:
                    pri += p.prior.lnL(p.value)
        return pri
    
    def predict(self, name=None, level=0, logN=None):
        """
        predict column densities on levels
        parameters:
            - name       : name of the species
            - level      : level with known column density
                            if level == -1, use total column density
            - logN       : column density as <a> object
        """
        if name is None:
            name = list(self.species.keys())[0]

        x = self.balance(self.species[name].name)

        if logN is None:
            logN = 0.0

        if level == -1:
            ref = np.sum(x)
        else:
            ref = x[level]

        if isinstance(logN, (int, float)):
            return [np.log10(10**logN * x[i] / ref) for i in range(self.species[name].num)]

        if isinstance(logN, a):
            return [logN * x[i]/ref for i in range(self.species[name].num)]
        
    def calc_dep(self, par, grid_num=50, plot=1, verbose=1, title='', ax=None, alpha=1, stats=False, Texc=False):
        """
        calculate dependence of population of rotational levels on one parameter
        parameters:
            - par        : name of parameter
            - grid_num   : number of points in the grid for each parameter
            - plot       : specify graphs to plot
            - ax         : axes object where to plot data, of not present make it if plot==1.
            - alpha      : alpha value for the contour plots
            - stats      : if True, then include stat weights: i.e. N[k]/g[k] / N[0]/g[0]
            - Texc       : if True, then give results in excitation temperatures
        """
        
        # >> colors for plotting
        colors = ['#fb9a99', '#e31a1c', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fdbf6f', '#ff7f00']

        if ax is None:
             fig, ax = plt.subplots(len(self.species), figsize=(10, 10*len(self.species)))
       
        x = np.linspace(self.pars[par].range[0], self.pars[par].range[1], grid_num)
        for i, s in enumerate(self.species):
            if ax is not None:
                axi = ax
            else:
                if len(self.species) > 1:
                    axi = ax[i]
                else:
                    axi = ax

            # plot the data:
            for k, y in enumerate(s.y):
                if Texc:
                    yg = y / (s.g[k+1] / s.g[0])
                    yg.val = np.log10( - (s.E[k+1] - s.E[0]) / 0.685 / np.log(10 ** yg.val))
                    yg.minus, yg.plus = 0.1, 0.1
                    #yg.minus = yg.val - (s.E[k+1] - s.E[0]) / 0.685 / np.log(10**(yg.val + yg.plus))
                    #yg.plus = (s.E[k+1] - s.E[0]) / 0.685 / np.log(10 ** (yg.val - yg.minus)) - yg.val
                else:
                    yg = y/(s.g[k+1]/s.g[0]) if stats else y
                axi.axhline(yg.val , ls='--', c=colors[2*k])
                axi.axhspan(yg.val + yg.plus, yg.val - yg.minus, facecolor=colors[2*k], alpha=0.5, zorder=0)

            # calc theoretical curves:
            f = np.zeros([len(x), s.num-1])
            for k, z in enumerate(x):
                self.pars[par].value = z
                #print(self.sp.g[1]/sp.g[0], (sp.E[1]-sp.E[0])/0.695,10**t)
                #ax.axhline(np.log10(sp.g[1]/sp.g[0]*np.exp(-(sp.E[1]-sp.E[0])/0.695/10**t)), ls=':', c=colors[1])
                f[k] = self.balance(s.name)[1:]
            f = np.log10(f)

            # plot theoretical curves:
            for k in range(s.num-1):
                if Texc:
                    lab  = 'T_{0}0'
                    fg = np.log10(- (s.E[k+1] - s.E[0]) / 0.685 / np.log(10**(f[:, k] - np.log10(s.g[k+1] / s.g[0]))))
                else:
                    lab = 'log (n{0}/g{0})/(n0/g0)' if stats else 'log n{0}/n0'
                    fg = f[:, k] - int(stats)*np.log10(s.g[k+1]/s.g[0])
                axi.plot(x, fg, color=colors[2*k+1] , label=lab.format(k+1), lw=3)

            # plot boltzmann limit lines:
            if par == 'n':
                for k in range(0, s.num-1):
                    if Texc:
                        fg = self.pars['T'].value
                    else:
                        fg = np.log10(np.exp(-(s.E[k+1]-s.E[0]) / 0.695 / 10**self.pars['T'].value)) + (1-int(stats))*np.log10(s.g[k+1]/s.g[0])
                    axi.axhline(fg, color=colors[2*k+1], lw=2, ls=':')
            if Texc:
                ylabel = r'$\rm T_{exc}$'
            else:
                if stats:
                    ylabel = r'$\log (n_i/g_i) / (n_0/g_0)$'
                else:
                    ylabel = r'$\log (n_i/n_0)$'
            axi.set_title(s.name)
            axi.set_ylabel(ylabel)
            axi.set_xlabel(self.pars[par].label)
            axi.legend(loc=4)

    def calc_grid(self, grid_num=50, plot=1, verbose=1, output=None, marginalize=True, limits=0,
                  title='', ax=None, alpha=1, color=None, cmap='PuBu', color_point=None,
                  zorder=1):
        """
        calculate ranges for two parameters using given populations of levels
        parameters:
            - grid_num     :  number of points in the grid for each parameter
            - plot         :  specify graphs to plot: 1 is for contour, 2 is for filled contours
            - output       :  name of output file for probalbility
            - marginalize  :  provide 1d estimates of parameters by marginalization
            - limits       :  show upper of lower limits: 0 for not limits, >0 for lower, <0 for upper
            - title        :  title name for the plot
            - ax           :  axes object where to plot data, of not present make it if plot==1.
            - alpha        :  alpha value for the contour plots
            - color        :  color of the regions in the plot
            - cmap         :  color map for contour fill
            - color_point  :  show indicator for best fit
            - zorder       :  zorder of the graph object 
        """

        self.get_vary()

        print(self.vary)

        out = None

        if len(self.vary) == 1:
            out = self.calc_1d(self.vary, grid_num=grid_num, plot=plot, verbose=verbose, title=title, ax=ax, alpha=alpha,
                               color=color, zorder=zorder)

        if len(self.vary) == 2:
            out = self.calc_2d(self.vary, grid_num=grid_num, plot=plot, verbose=verbose, output=output, marginalize=marginalize, limits=limits,
                               title=title, ax=ax, alpha=alpha, color=color, cmap=cmap, color_point=color_point, zorder=zorder)

        return out

    def calc_1d(self, vary, grid_num=50, plot=1, verbose=1, title='', ax=None, alpha=1, color=None, zorder=1):
        """
        calculate ranges for two parameters using given populations of levels
        parameters:
            - vary         :  list of variables 
            - grid_num     :  number of points in the grid for each parameter
            - plot         :  specify graphs to plot: 1 is for contour, 2 is for filled contours
            - title        :  title name for the plot
            - ax           :  axes object where to plot data, of not present make it if plot==1.
            - alpha        :  alpha value for the contour plots
            - color        :  color of the regions in the plot
            - zorder       :  zorder of the graph object 
        """

        # >> colors for plotting
        if ax is None:
            fig, ax = plt.subplots()
        if color is None:
            color = (1, 0, 0)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #
        # ++++  Direct calculations  ++++++++++++++++++++
        if verbose:
            self.print_species()
            self.print_pars()

        out = []
        X1 = np.linspace(self.pars[vary[0]].range[0], self.pars[vary[0]].range[1], grid_num)
        Z = np.zeros_like(X1)
        printProgressBar(0, grid_num, prefix='Progress:')
        for i in range(len(X1)):
            self.pars[vary[0]].value = X1[i]
            Z[i] = self.lnprob()
            printProgressBar(i + 1, grid_num, prefix='Progress:')

        if verbose == 1:
            print(-max(Z))

        L = np.exp(Z - max(Z.flatten()))

        d = distr1d(X1, L)
        if plot > 0:
            print('plot pdf...')
            ax.plot(d.x, d.y, c=color, alpha=alpha, zorder=zorder)
            d.dopoint()
            for c in self.conf_levels:
                d.dointerval(c)
                #ax.axhline(d.inter(interval[0]))

            if title != '':
                if 0:
                    ax.set_title(title)
                else:
                    ax.text(0.9, 0.9, title, ha='right', va='top', transform=ax.transAxes)

            ax.set_xlabel(self.pars[vary[0]].label)

        if 1:
            lmax = max(L.flatten())
            out.append([vary[0], a(d.point, d.interval[1] - d.point, d.point - d.interval[0], 'd')])

        return out

    def calc_2d(self, vary, grid_num=50, plot=1, verbose=1, marginalize=True, limits=0, output=None,
                title='', ax=None, alpha=1, color=None, cmap='PuBu', color_point='gold', zorder=1):
        """
        calculate ranges for two parameters using given populations of levels
        parameters:
            - vary         :  list of variables
            - grid_num     :  number of points in the grid for each parameter
            - plot         :  specify graphs to plot: 1 is for contour, 2 is for filled contours
            - verbose      :  print details of calculations
            - marginalize  :  print and plot marginalized estimates
            - limits       :  show upper of lower limits: 0 for not limits, >0 for lower, <0 for upper
            - title        :  title name for the plot
            - ax           :  axes object where to plot data, of not present make it if plot==1.
            - alpha        :  alpha value for the contour plots
            - color        :  color of the regions in the plot
            - cmap         :  color map for the countour fill
            - color_point  :  color for indicator for best fit
            - zorder       :  zorder of the graph object 
        """

        # >> colors for plotting
        if ax is None:
             fig, ax = plt.subplots()
        if color is None:
            color = (1, 0, 0)
        print(self.pars)

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #
        #++++  Direct calculations  ++++++++++++++++++++
        if verbose:
            self.print_species()
            self.print_pars()

        out = []
        X1 = np.linspace(self.pars[vary[0]].range[0], self.pars[vary[0]].range[1], grid_num)
        X2 = np.linspace(self.pars[vary[1]].range[0], self.pars[vary[1]].range[1], grid_num)
        Z = np.zeros([len(X1), len(X2)])
        printProgressBar(0, grid_num, prefix='Progress:')
        for i in range(len(X1)):
            print(i)
            self.pars[vary[0]].value = X1[i]
            for k in range(len(X2)):
                self.pars[vary[1]].value = X2[k]
                Z[k, i] = self.lnprob()
            printProgressBar(i + 1, grid_num, prefix='Progress:')

        if verbose == 1:
            print(max(Z.flatten()))

        if output is not None:
            with open(output, "wb") as f:
                pickle.dump(X1, f)
                pickle.dump(X2, f)
                pickle.dump(np.exp(Z), f)

        d = distr2d(X1, X2, np.exp(Z))
        point = d.dopoint()

        if marginalize:
            if plot:
                d1 = d.marginalize('x')
                d1.dopoint()
                d1.dointerval()
                print('marg point:', d1.point)
                print('marg interval:', d1.interval)
                d1.plot()
                d2 = d.marginalize('y')
                d2.dopoint()
                d2.dointerval()
                print('marg point:', d2.point)
                print('marg interval:', d2.interval)
                d2.plot()

        if plot > 0:
            print('plot regions...')

            if 0: #self.species[self.species.keys()[0]].y[0][2].type in ['u', 'l']:
                typ = 'i'
            else:
                typ = 'l'
                #typ = 'chi2'

            if typ == 'chi2':
                conf_levels = np.asarray([chi2.isf(1-c, 2) for c in self.conf_levels])
                chi = - Z * 2
                m = np.min(chi.flatten())
                print('chi2_min', m)
                cs = ax.contour(d.X, d.Y, chi-m, levels=[1], lw=1, ls='--', origin='lower', zorder=zorder)

            if typ == 'l':
                if plot:
                    d.plot_contour(conf_levels=self.conf_levels, limits=limits, ax=ax, cmap=cmap, color=color, color_point=color_point, alpha=alpha, zorder=zorder)

            if typ == 'i':
                print('interpolate region...')
                if 1:
                    f = interpolate.RectBivariateSpline(X2[:,0], X1[0,:], np.exp(Z-max(Z.flatten())))
                else:
                    f = interpolate.interp2d(X1, X2, L, kind='cubic')
                x = np.linspace(self.pars[vary[0]].range[0], self.pars[vary[0]].range[1], grid_num)
                y = np.zeros_like(x)
                print('estimate line...')
                for i, xi in enumerate(x):
                    g = lambda z: f.ev(z, xi)-0.317
                    if g(self.pars[vary[1]].range[0]) * g(self.pars[vary[1]].range[1]) < 0:
                        y[i] = optimize.bisect(g, self.pars[vary[1]].range[0], self.pars[vary[1]].range[1])
                    else:
                        y[i] = self.pars[vary[1]].range[1]
                m = y != self.pars[vary[1]].range[1]
                ax.plot(x, y, c=color, zorder=zorder)
                x, y = x[m], y[m]
                if self.species[0].y[0].type == 'u':
                    uplims, lolims = True, False
                else:
                    uplims, lolims = False, True
                ax.errorbar(x[::3], y[::3], yerr=0.1, ls='none', uplims=uplims, lolims=lolims, color=color, zorder=zorder)
            if title !='':
                if 0:
                    ax.set_title(title)
                else:
                    ax.text(0.9, 0.9, title, ha='right', va='top', transform=ax.transAxes)
            ax.set_xlabel(self.pars[vary[0]].label)
            ax.set_ylabel(self.pars[vary[1]].label)
            # save contour to file
            if 0:
                #f_c = open(species+'_c.dat', 'w')
                p = cs.collections[0].get_paths()
                for l in p:
                    print(l.vertices)
                np.savez('res/'+species+'_'+str(comp)+'_c', *[l.vertices for l in p])

            if 1:
                print(vary)
                d.dointerval(0.6827)
                print(d.interval)
                out.append([vary[0], a(d.point[0], d.interval[0][1] - d.point[0], d.point[0] - d.interval[0][0])])
                out.append([vary[1], a(d.point[1], d.interval[1][1] - d.point[1], d.point[1] - d.interval[1][0])])

        return out
    
    def calc_MCMC(self, nwalkers=10, nsteps=100, plot='chainconsumer', verbose=0, diagnostic=True):
        """
        calculate regions using MCMC method by emcee package
        """
        self.get_vary()
        print(self.vary)
        ndim = len(self.vary)
        init = [self.pars[p].init for p in self.vary]
        init_range = [self.pars[p].init_range for p in self.vary]
        pos = [init + init_range*np.random.randn(ndim) for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprobMCMC)
        for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
            if i % int(nsteps / 100) == 0:
                print("{0:5.1%}".format(float(i) / nsteps))

        samples = sampler.chain[:, int(nsteps/2):, :].reshape((-1, ndim))

        if plot == 'corner':
            fig = corner.corner(samples, labels=self.vary,
                                quantiles=[0.16, 0.5, 0.84],
                                verbose=True,
                                no_fill_contours=False,
                                draw_datapoints=True)

            fig.savefig("triangle.png")

        if plot == 'chainconsumer':
            c = ChainConsumer().add_chain(samples, parameters=self.vary)
            # >>> diagnostic:
            if 0:
                c.diagnostic.gelman_rubin()
                c.diagnostic.geweke()
            # >>> plot:
            c.configure(statistics='max',  # 'max'
                        sigmas=[0, 1, 2],
                        cloud=True
                        )
            fig = c.plotter.plot(figsize=(12, 12), display=True, filename='mcmc.png')

    def critical_density(self, species=None, depend=None, verbose=1, ax=None):
        """
        function to calculate the critical density
        parameters:
            - species         :  species name, if None, than for all species in self.species 
            - depend          :  if not None, than specified calculate dependence of critical density on variable  
            - verbose         :  verbose
            - ax              :  axis object to plot the dependence 
        """
        if verbose:
            print('critical densities for:')
        
        for s in self.species:
            if species is None or species == s.name:
                if depend in ['T'] and ax is None:
                    fig, axi = plt.subplots(s.num-1, s.num-1, sharex=True, sharey=True)
                if verbose:
                    print('species: ', s.name)
                for i in range(s.num-1):
                    for j in range(i+1, s.num):
                        if s.num > 2 and ax is None:
                            axs = axi[i,j-1]
                        elif s.num == 1 and ax is None:
                            axs = axi
                        elif ax is not None:
                            axs = ax
                        if verbose:
                            coll = 0
                            for k in range(s.num):
                                coll += self.collision_rate(s, j, k) / 10 ** self.pars['n'].value
                        if depend in ['T']:
                            pars_range = np.linspace(self.pars[depend].range[0], self.pars[depend].range[1], 10)
                            y = []
                            for t in pars_range:
                                self.pars[depend].value = t
                                coll = 0
                                for k in range(s.num):
                                    coll += self.collision_rate(s, j, k)
                                y.append(coll/10**self.pars['n'].value)
                            axs.plot(pars_range, np.log10((y/s.A[j, i])**(-1)), '-b')
                            axs.text(0.9, 0.9, "{:0}-{:1}".format(j, i), transform=axs.transAxes, fontsize=20)
    
    def calc_cooling(self, verbose=0, depend=None, ax=None, species=None, color='k', factor=0):
        """
        Function for calculation cooling function by spontangeous transitions 
        in different species, like HD, H2, CII, OI, CI by 1 atom (molecule)
        
        Cooling function is derived using formula 
    
        L_ij = A_ij E_ij n_i
        
        applied to measured populations of fine structure levels, or rotational levels
        """    
        print(factor)
        out = []
        for s in self.species:
            if species is None or s.name == species: 
                if verbose == 1:        
                    print('species: ', s)
                    print('column densities: ', s.n)
                    print('data ratios: ', s.y)
                if depend in ['T', 'n']:
                    if ax is None:
                        fig, ax = plt.subplots()
                    pars_range = np.linspace(self.pars[depend].range[0], self.pars[depend].range[1], 100)
                    L = np.zeros_like(pars_range)
                else: 
                    L = 0
                for u in range(s.num-1,0,-1):
                    for l in range(u):
                        y = np.zeros_like(L)
                        if depend in ['T', 'n']:
                            for i, value in enumerate(pars_range):
                                self.pars[depend].value = value
                                y[i] = self.cooling_rate(s, u, l).value
                            if 0:
                                ax.plot(pars_range, np.log10(y)+factor, '--', color=color, label="{0} {1}-{2}".format(s.name, u, l))
                        else:
                            y = self.cooling_rate(s, u, l).value
                        L += y
                if depend in ['T', 'n']:            
                    ax.plot(pars_range, np.log10(L)+factor, '-', color=color, lw=1.5, label=s.name+" total")
                out.append([s.name, L])
        return out 
    
    def cooling_rate(self, s, u, l):
        """
        calculate the colling rate of the medium due to transition <u> -> <l> for <s> species
        parameters:
            - s      :  species
            - u      :  upper level
            - l      :  lower level
        return: L
            - L      :  coollinf rate in erg/s/cm^3 
        """
        hu = (s.E[u]-s.E[l])/au.cm*ac.c.cgs*ac.h.cgs  # in ergs
        if 0:
            stat = s.g[u]/s.g[l]*np.exp(-hu/(ac.k_B.cgs*10**self.pars['T'].value*au.K))
            #print(u, l, s.A[u,l], stat, s.A[u,l]/self.collision_rate(s, u, l)/10**self.pars['n'].value)
            L = s.A[u,l]/au.s * hu * \
                stat / (1 + stat + s.A[u,l]/self.collision_rate(s, u, l)) 
        else:
            x = self.balance(s.name)
            L = s.A[u,l]/au.s * hu * x[u]
        #L *= 10**self.pars['n'].value * au.cm**3
        return L
        
    def calc_cooling_Wolfe(self, logHI, plot=1, verbose=0):
        """
        Function for calculation emission per H atom in spontangeous transitions using Wolfe et al 2003 formalism
        different species can be treated: CII, OI, CI, HD
        
        Emission per H atom:
    
        paremeters:
            - logHI          : the column density of HI in gas (hydrogen in gas), 
                                <a> value, or float
        
        """    
        
        out = []
        for i,s in enumerate(self.species):
            if verbose == 1:        
                print('species: ', s)
                print('column densities: ', s.n)
            L = 0
            for i in range(s.num-1,0,-1):
                for k in range(i):
                    print(i, k, s.A, (s.E[i]-s.E[k])*1.3806e-16*1.4, s.n[i])
                    L = L + s.A[i,k]*(s.E[i]-s.E[k])*1.3806e-16*1.4*s.n[i].dec()
            out.append([s.name, L])
        return out

    def calc_emission_CO(self, par, col=None, grid_num=50, plot=1, verbose=1, title='', ax=None, alpha=1, stats=False):
        """
        calculate the dependence of CO emission in J->J-1 transition from parameters
        parameters:
            - par        : name of parameter
            - col        : column densities. If None -- read from species, if float or int -- the total, then for each par column densities will be derived
            - grid_num   : number of points in the grid for each parameter
            - plot       : specify graphs to plot
            - ax         : axes object where to plot data, of not present make it if plot==1.
            - alpha      : alpha value for the contour plots
            - stats      : if True, then include stat weights: i.e. N[k]/g[k] / N[0]/g[0]
        """

        # >> colors for plotting
        colors = ['#fb9a99', '#e31a1c', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fdbf6f', '#ff7f00', '#aaaaaa', '#666666']

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10 * len(self.species)))

        x = np.linspace(self.pars[par].range[0], self.pars[par].range[1], grid_num)
        for s in self.species:
            if s.name == 'CO':
                c = np.zeros(s.num-1)
                for k in range(s.num-1):
                    mu = 0.11e-18
                    # constant in (K km/s)
                    c[k] = 8 * np.pi**3 * (s.E[k+1]-s.E[0]) * ac.c.cgs.value / 3 / ac.k_B.cgs.value * mu**2 * (k+1) / s.g[k+1] / 1e5
                #print(c)
                f = np.zeros([len(x), s.num - 1])
                f_CMB = np.zeros([len(x), s.num - 1])
                for k, z in enumerate(x):
                    self.pars[par].value = z
                    f[k] = self.balance(s.name)[1:]
                    f_CMB[k] = 1 - (np.power(f[k]/s.g[1:s.num], -1) - 1)/(np.exp((s.E[1:s.num] - s.E[0]) / 0.695 / (2.725*(1+self.z)))-1)
                    if col is None:
                        f[k] = s.n[1:].val
                    else:
                        f[k] = np.log10(f[k] * 10**col / sum(f[k]))
                    print(f_CMB[k], f[k])
                for k in range(s.num-1):
                    ax.plot(x, f_CMB[:, k] * c[k] * np.power(10, f[:, k]), color=colors[2 * k + 1], label=r'{:}$\to${:}'.format(k + 1, k), lw=3)

                ax.set_title(s.name)
                ax.set_ylabel(r'W$_{\rm CO}$ [K km/s]')
                ax.set_xlabel(self.pars[par].label)
                ax.legend(loc=4)

if __name__ == '__main__':
    
    print('executing main program code')
    
    if 0:
        pr = pyratio()
        n_Si = [a(14.92, 0.14, 0.12), a(11.32, 0.14, 0.21)]
        n_CI = [a(13.16, 0.11, 0.09), a(13.25, 0.13, 0.09), a(13.01, 0.12, 0.11)]
        n_HD = [a(15.26, 0.37, 0.23), a(14.92, 0.06, 0.06)]
        n_CII = [a(16.93, 0.10, 0.10), a(15.46, 0.20, 0.21)]
        #pr.add_spec('SiII', n_Si)
        pr.add_spec('CI', n_CI)
        #pr.add_spec('HD', n_HD)
        #pr.add_spec('CII', n_CII)
        pr.set_pars(['T','n','f'])
        pr.set_fixed('f', -4)
        if 0:
            #pr.species[0].plot_allCollCrossSect()
            fig, ax = plt.subplots()
            #pr.species[0].coll['H'].plot(1, 0, ax=ax)
            #pr.species[0].coll['H'].plot(2, 1, ax=ax)
            pr.species[0].coll['H'].plot(2, 0, ax=ax)
        if 1:
            pr.pars['T'].value = 3.0
            pr.calc_dep('n')
            #pr.pars['n'].value = 2.5
            #pr.calc_dep('T')
        if 0:
            pr.calc_grid()
        plt.show()

    # check CI and OI fine structure cross sections:
    if 0:
        pr = pyratio()
        n_CI = [a(13.16, 0.11, 0.09), a(13.25, 0.13, 0.09), a(13.01, 0.12, 0.11)]
        spec = 'CI'
        pr.add_spec(spec, n_CI)
        #pr.set_pars(['T', 'n', 'f'])
        #pr.set_fixed('f', 0)
        lev = [(1, 0), (2, 0), (2, 1)]
        #lev = [(0, 1), (0, 2), (1, 2)]
        if spec == 'CI':
            Tmin, Tmax = 0.5, 3
            g = [1, 3, 5]
            J = [0, 1, 2]
            E = [0, 16.64, 43.40]
            # approxiamtion by Abrahamsson 2007
            a = [(3.6593, 56.6023, -802.9765, 5025.1882, -17874.4255, 38343.6655, -49249.4895, 34789.3941, -10390.9809),
                 (10.8377, -173.4153, 2024.0272, -13391.6549, 52198.5522, -124518.3586, 178182.5823, -140970.6106, 47504.5861),
                 (15.8996, -201.3030, 1533.6164, -6491.0083, 15921.9239, -22691.1632, 17334.7529, -5517.9360, 0)]
            b = [1 / 4, 1 / 3, 1 / 4]
            # approximation by Draine ISM 2011
            D = {}
            D['H'] = [(1.26e-10, 0.115, 0.057), (8.90e-11, 0.231, 0.046), (2.64e-10, 0.228, 0.046)]
            D['pH2'] = [(6.7e-11, -0.085, 0.102), (8.6e-11, -0.010, 0.048), (1.75e-10, 0.072, 0.064)]
            D['oH2'] = [(7.1e-11, -0.004, 0.049), (6.9e-11, 0.169, 0.038), (1.48e-10, 0.263, 0.031)]
        if spec == 'OI':
            Tmin, Tmax = 1.5, 3
            g = [5, 3, 1]
            J = [2, 1, 0]
            E = [0, 158.265, 226.977]
            a = [(4.581, -156.118, 2679.979, -78996.962, 1308323.468, -13011761.861, 71010784.971, -162826621.855),
                 (3.297, -168.382, 1844.099, -68362.889, 1376864.737, -17964610.169, 134374927.808, -430107587.886),
                 (3.437, 17.443, -618.761, 3757.156, -12736.468, 22785.266, -22759.228, 12668.261)]
            b = [3 / 4, 3 / 4, 1 / 2]
            # approximation by Draine ISM 2011
            D = {}
            D['H'] = [(3.57e-10, 0.419, -0.003), (3.19e-10, 0.369, -0.006), (4.34e-10, 0.755, -0.160)]
            D['pH2'] = [(6.7e-11, -0.085, 0.102), (8.6e-11, -0.010, 0.048), (1.75e-10, 0.072, 0.064)]
            D['oH2'] = [(7.1e-11, -0.004, 0.049), (6.9e-11, 0.169, 0.038), (1.48e-10, 0.263, 0.031)]
        T = np.logspace(Tmin, Tmax, 100)
        part = 'H'
        fig, axs = plt.subplots(3, figsize=(10,20), sharex=True)
        for i, l in enumerate(lev):
            # pr.species[0].plot_allCollCrossSect()
            B = g[l[0]] / g[l[1]] * np.exp(-(E[l[0]] - E[l[1]]) / 0.685 / T)
            #fig, ax = plt.subplots()
            ax = axs[i]
            pr.species[0].coll[part].plot(l[0], l[1], ax=ax, label='Abrahamsson 2007 data')
            ax.plot(np.log10(T), np.log10(D[part][i][0] * (T / 100) ** (D[part][i][1] + D[part][i][2] * np.log(T / 100))
                                          / B**((1-np.sign(l[0]-l[1]))/2)), '-r', label='Draine 2011 fit')
            if part == 'H':
                k = np.sum([a[i][k] * T ** (-k * b[i]) for k in range(len(a[i]))], axis=0)
                if l[0] > l[1]:
                    ax.plot(np.log10(T), np.log10(1e-11 * np.exp(k) / B), 'g', label='Abrahamsson 2007 fit')
                else:
                    ax.plot(np.log10(T), np.log10(1e-11 * np.exp(k)), 'g', label='Abrahamsson 2007 fit')
            ax.set_title(r"J={:} $\to$ J'={:}".format(J[l[0]], J[l[1]]))
            ax.set_xlabel('log T')
            ax.set_ylabel(r'$\log k_{J\to J^{\prime}}$')
            ax.legend(loc='best', fontsize=12)
        #plt.savefig('C:/Users/Serj/Desktop/{:}_comparison.pdf'.format(spec))
        plt.show()

    if 0:
        pr = pyratio(z=2.5)
        pr.add_spec('CO')
        d = {'T': 2, 'n': 3, 'f': 0}
        pr.set_pars(d.keys())
        for d in d.items():
            pr.pars[d[0]].value = d[1]
        print(pr.predict(0, 14.43))
    
    if 0:
        pr = pyratio()
        n_CII = [a(16.93, 0.10, 0.10), a(15.46, 0.20, 0.21)]
        #print('CII optical depth:', calctau0(1e8/63, 0, 15.46, 2, A=2.3e-6, gu=4, gl=2))
        pr.add_spec('CII', n_CII)
        pr.add_spec('CI', [a(), a(), a()])
        pr.add_spec('OI', [a(), a(), a()])
        pr.set_pars(['T','n','f'])
        pr.pars['f'].value = -3
        #pr.pars['n'].range = [-1, 4]
        pr.pars['n'].value = 2.5
        #pr.critical_density(depend='T')
        pr.pars['T'].value = 2.1
        fig, ax = plt.subplots()
        out = pr.calc_cooling(depend='T', ax=ax, species='CII', color=col.tableau10[0], factor=8.43-12) #factor=16.92-22.59)
        out = pr.calc_cooling(depend='T', ax=ax, species='CI', color=col.tableau10[4],  factor=8.43-12) #factor=14.01-21.59)
        out = pr.calc_cooling(depend='T', ax=ax, species='OI', color=col.tableau10[9], factor=8.69-12)
        H2 = np.genfromtxt('D:/science/python/spectro/data/pyratio/H2_cooling.csv', unpack=True)
        ax.plot(np.log10(H2[0]), H2[1], '-', color=col.tableau10[2], lw=1.5, label='H2')
        HD = np.genfromtxt('D:/science/python/spectro/data/pyratio/HD_cooling.csv', unpack=True)
        ax.plot(np.log10(HD[0]), HD[1], '--', color=col.tableau10[2], lw=1.5, label='HD')
        ax.legend(loc='best')
        #print(out)
    
    # check radiation fields
    if 0:
        pr = pyratio(z=2.0)
        print(r'\xi of Draine:', pr.uv(1e5 * ac.c.cgs.value, 'Draine') / pr.uv(1e5 * ac.c.cgs.value, 'Habing'))
        print(r'\xi of Mathis:', pr.uv(1e5 * ac.c.cgs.value, 'Mathis') / pr.uv(1e5 * ac.c.cgs.value, 'Habing'))
        fig, ax = plt.subplots()
        if 0:
            l = np.linspace(912, 3000, 1000)
        else:
            l = np.linspace(1e8 / (13.6*8065.54), 1e8 / (6*8065.54), 1000)
        nu = ac.c.cgs.value / l / 1e-8
        ax.plot(l, pr.uv(nu, kind='Habing'))
        ax.plot(l, pr.uv(nu, kind='Draine'))
        ax.plot(l, pr.uv(nu, kind='Mathis'))
        print('G_0 of Draine:', np.trapz(pr.uv(nu, kind='Draine')/l)*abs(l[1]-l[0])/5.29e-14)
        print('G_0 of Mathis:',  np.trapz(pr.uv(nu, kind='Mathis')/l)*abs(l[1]-l[0])/5.29e-14)
        
        
    if 0:
        fig, ax = plt.subplots()
        
        import H2_summary
        QSO = H2_summary.load_QSO()
        for q in QSO:
            if 1==1 and q.name.find('2140') > -1:
                pr = pyratio(z=q.z_dla)
                pr.set_pars(['T', 'n', 'f'])
                pr.set_prior('f', a(0,0,0))
                #pr.set_prior('T', a(2.04,0,0))
                pr.set_prior('T', q.e['T01'].col.log())
                print(q.e['T01'].col.log())
                pr.pars['n'].value = 0
                #pr.set_prior('T', q.e['T01'].col.log())
                pr.add_spec('CI', [q.e['CI'].col, q.e['CI*'].col, q.e['CI**'].col])
                if 0:
                    fig, ax = plt.subplots()
                #pr.calc_dep('UV', ax=ax)
                if 1:
                    out = pr.calc_grid(grid_num=100, plot=2, verbose=1, ax=ax, alpha=0.5, color=col.tableau10[4])
                    for o in out:
                        print("q.el(\'{:}\', {:.2f}, {:.2f}, {:.2f})".format(o[0], o[1].val, o[1].plus, o[1].minus))
                
        plt.savefig('pyratio.png', bbox_inches='tight', transparent=True)
        plt.show()

    # check calculation with Ntot
    if 0:
        pr = pyratio(z=2.525, calctype='numbdens')
        pr.set_pars(['T', 'n', 'f'])
        pr.set_prior('f', a(0, 0, 0))
        pr.set_prior('T', a(1.7, 0, 0))
        pr.pars['n'].value = 2
        pr.pars['n'].range = [1, 3]

        pr.add_spec('CO', [a(14.43, 0.12), a(14.52, 0.08), a(14.33, 0.06), a(13.73, 0.05), a(13.14, 0.13)])
        if 1:
            pr.pars['Ntot'].value = pr.species['CO'].n_tot.val
            pr.pars['Ntot'].init = pr.species['CO'].n_tot.val
            pr.pars['Ntot'].range = [pr.species['CO'].n_tot.val - pr.species['CO'].n_tot.minus * 5, pr.species['CO'].n_tot.val + pr.species['CO'].n_tot.plus * 5]
            pr.pars['Ntot'].init_range = np.max([pr.species['CO'].n_tot.minus, pr.species['CO'].n_tot.plus])
            print(pr.pars['Ntot'])
        #pr.add_spec('CO', [a(14.43, 0.12), a(14.52, 0.08), a(14.33, 0.06), a(13.73, 0.05), a(13.14, 0.13)])
        fig, ax = plt.subplots()
        if 0:
            pr.calc_MCMC()
        else:
            pr.calc_grid(grid_num=50, ax=ax)

        plt.savefig('C:/Users/Serj/Desktop/W_CO.pdf')
        plt.show()

    # check CO data
    if 0:
        pr = pyratio(z=2.525)
        pr.set_pars(['T', 'n', 'f'])
        pr.set_prior('f', a(0, 0, 0))
        pr.set_prior('T', a(1.7, 0, 0))
        pr.pars['n'].value = 2
        pr.pars['n'].range = [1, 6]

        pr.add_spec('CO', [a(14.43, 0.12), a(14.52, 0.08), a(14.33, 0.06), a(13.73, 0.05), a(13.14, 0.13)])
        # pr.add_spec('CO', [a(14.43, 0.12), a(14.52, 0.08), a(14.33, 0.06), a(13.73, 0.05), a(13.14, 0.13)])
        fig, ax = plt.subplots()
        # pr.species[0].plot_allCollCrossSect()
        # pr.critical_density(depend='T', ax=ax)
        # pr.calc_dep('n', ax=ax, stats=1, Texc=0)
        pr.calc_emission_CO('n')
        if 0:
            print(pr.species[0].coll['H'].rate(1, 0, 2), pr.species[0].coll['H'].rate(0, 1, 2))
        if 0:
            pr.species[0].coll['pH2'].plot(1, 0, ax=ax)
            pr.species[0].coll['pH2'].plot(0, 1, ax=ax)
        plt.savefig('C:/Users/Serj/Desktop/W_CO.pdf')
        plt.show()

    if 1:
        pr = pyratio(z=2.5)
        pr.set_pars(['T', 'n', 'f', 'UV'])
        pr.pars['T'].range = [1.5, 5]
        pr.set_prior('T', a(4, 0, 0))
        pr.pars['UV'].range = [2, 5]
        pr.pars['n'].range = [1, 5]
        pr.pars['f'].range = [-6, 0]
        pr.set_fixed('f', -3)
        pr.add_spec('SiII', [a(15, 0.1), a(14, 0.1)])
        if 0:
            pr.pars['n'].value = 7
            pr.pars['UV'].value = 3
            print(pr.predict())
        else:
            out = pr.calc_grid(grid_num=10, plot=2, verbose=1, alpha=0.5, color='dodgerblue')
            plt.show()

