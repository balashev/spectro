#!/usr/bin/env python
# -*- coding: utf-8 -*-

import astropy.constants as ac
from astropy.cosmology import FlatLambdaCDM
import astropy.units as au
from chainconsumer import ChainConsumer
import collections
import corner
import emcee
import glob
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import bisect
import pickle
from scipy import interpolate, optimize
from scipy.stats import chi2
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))[:-8])
from spectro.a_unc import a
from spectro.profiles import tau, voigt
from spectro.stats import distr1d, distr2d
from spectro.sviewer.utils import printProgressBar, Timer, flux_to_mag

def smooth_step(x, delta):
    x = np.clip(x/delta, 0, 1)
    return x**3 * (x * (x*6 - 15) + 10)

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
        self.const2 = (ac.hbar.cgs ** 2 * (2 * np.pi / ac.m_e.cgs ** 3 / ac.k_B.cgs) ** 0.5).value

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
            self.read_HD()
        elif name == 'CO':
            self.read_CO()
        elif name == 'FeII':
            self.read_FeII()
        elif name == 'H2':
            self.read_H2()
        else:
            self.read_popratio()
            # replace CII collisional rates from Barinovs+2005 paper
            if 1:
                if self.name == 'CII':
                    plot = 0
                    if plot:
                        fig, ax = plt.subplots(figsize=(8,5))
                        ax.plot(self.coll['H'].c[0].rates[0], self.coll['H'].c[0].rates[1], label=r'Launay \& Roueff 1977 + Keenan et al. 1986')
                    def Barinovs2005(t):
                        return 7.938e-11 * np.exp(-91.2 / t) * (16 + 0.344 * np.sqrt(t) - 47.7 / t)
                    x = np.logspace(np.log10(5), 5, 100)
                    self.coll['H'].c[0].rates = [np.log10(x), np.log10(Barinovs2005(x))]
                    self.coll['H'].c[0].rate_int = interpolate.InterpolatedUnivariateSpline(self.coll['H'].c[0].rates[0], self.coll['H'].c[0].rates[1], k=3)
                    if plot:
                        ax.plot(self.coll['H'].c[0].rates[0], self.coll['H'].c[0].rates[1], label='Barinovs et al. 2005')
                        ax.set_xlim([1.2, 4.5])
                        ax.set_ylim([-10.5, -7.9])
                        ax.set_xlabel("$\log T$\,[K]")
                        ax.set_ylabel("$\log$ Collision rate [cm$^{3}$ s$^{-1}$]")
                        fig.legend()
                        fig.savefig("CII_rates.pdf")
                        plt.show()

        self.fullnum = self.E.shape[0] # full number of levels

        self.setEij()
        self.setBij()
        self.setAij()

        self.rad_rate = np.zeros([self.num, self.num])
        self.pump_rate = np.zeros([self.num, self.num])
        #print(self.stats, self.energy, self.descr, self.Aij)
        #print(self.coll[2].rate)
    
    def set_stats(self):
        if self.name == 'SiII':
            self.g = [i * 2 + 1 for i in range(self.num)]
        if self.name == 'CII':
            self.g = [i * 2 for i in range(self.num)]

    def set_names(self):
        if self.name in ['CI', 'CII', 'SiII', 'OI', 'FeII']:
            self.names = [self.name] + [self.name + f'j{k}' for k in range(1, self.num)]
        elif self.name in ['HD', 'CO', 'H2']:
            self.names = [self.name + 'j' + str(k) for k in range(self.num)]

    def coll_rate(self, part, i, j, T):
        return self.coll[part].rate(i, j, T)
        #print(l, part, i, j, T)
        #if l != 0:
        #    return self.coll[self.coll_ind(part, i, j)].rate(T, sign=np.sign(l))
        #else:
        #    return 0
            
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
        with open(self.parent.folder+'/data/pyratio/'+self.name+'.dat', encoding="ISO-8859-1") as f_in:
            f = 0
            n_coll = 0
            while True:
                line = f_in.readline()
                if not line:
                    break
                if line.strip() == '#': 
                    f += 1
                    if f == 4:
                        l = 0
                        line = f_in.readline()
                        #print(line)
                        while line.strip() != '':
                            if l < self.num: 
                                self.g[l] = line.split()[1]
                                self.E[l] = line.split()[0]
                                self.descr[l] = line.split()[3]+line.split()[4]+line.split()[5]
                                l += 1
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
                                    c.append(collision(self, coll, i, j, np.array([np.log10(Temp), [np.log10(float(line[k+2])) for k in range(n_2)]])))
                        self.coll[coll] = c
                        #print(coll, c, self.coll[coll])

    def read_HD(self):
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
                line = f_in.readline()
                if not line:
                    break
                if line.strip() == '#': 
                    f += 1
                    if f == 4:
                        l = 0
                        line = f_in.readline()
                        while line.strip() != '':
                            if l < self.num: 
                                self.g[l] = line.split()[1]
                                self.E[l] = float(line.split()[0]) * 0.695
                                self.descr[l] = line.split()[3]
                                l += 1
                            line = f_in.readline()
                    if f == 5:
                        n = int(f_in.readline())
                        for l in range(n):
                            line = f_in.readline()
                            i = int(line.split()[0])-1
                            j = int(line.split()[1])-1
                            if i < self.num and j < self.num:
                                self.A[j, i] = float(line.split()[2])

    def read_FeII(self):
        """
        read atomic data for FeII: einstein coefficients, branching ratios, electron collisional excitation rates.
        """
        folder = self.parent.folder + '/data/pyratio/'
        source = 'nist' #'bautista'
        if source == 'nist':
            lines = np.genfromtxt(folder + 'FeII/nist_lines.dat', skip_header=1, delimiter='|', dtype=[('l', '<f8'), ('A_ki', '<f8'), ('acc', 'S5'), ('E_i', '<f8'), ('E_k', '<f8'), ('conf_i', 'S20'), ('term_i', 'S10'), ('J_i', 'S5'), ('conf_k', 'S20'), ('term_k', 'S10'), ('J_k', 'S5') , ('t', 'S')])
            self.E = np.sort(np.unique(np.append(lines['E_i'], lines['E_k'])))
            #print(self.E, len(self.E))
            self.g = []
            for e in self.E:
                if len(np.argwhere(e == lines['E_i'])) > 0:
                    name = lines['J_i'][np.argwhere(e == lines['E_i'])[0][0]]
                else:
                    name = lines['J_k'][np.argwhere(e == lines['E_k'])[0][0]]
                self.g.append((float(name.decode().split('/')[0]) / float(name.decode().split('/')[1]) ) * 2 + 1)
                #self.g = [2 * float(e.decode().split('_')[1].split('/')[0]) / float(e.decode().split('_')[1].split('/')[1]) + 1 for e in self.E]
            self.g = np.asarray(self.g)
            self.fullnum = len(self.E)
            self.A = np.zeros([self.fullnum, self.fullnum])
            for l in lines:
                #print(l['E_i'], np.argwhere(self.E==l['E_k']), np.argwhere(self.E==l['E_i']))
                self.A[np.argwhere(self.E==l['E_k'])[0], np.argwhere(self.E==l['E_i'])[0]] = l['A_ki']

            if 1:
                E = np.genfromtxt(folder + 'FeII/table3.dat', delimiter='|', dtype=[('J', 'i4'), ('term', '|S14'), ('name', '|S10'), ('energy', '<f8')])
                inds = [np.argmin(np.abs(self.E - e)) for e in E['energy'] * 109737.31568160]
                Ai = np.genfromtxt(folder + 'FeII/apj516563t4_ascii.txt', comments='#', unpack=True)
                Ai = np.insert(Ai[25] * 10 ** (- Ai[27]), 0, 0)
                br = np.genfromtxt(folder + 'FeII/table10.dat', dtype=[('Ju', 'i4'), ('Jl', 'i4'), ('l', '<f8'), ('br', '<f8'), ('br_e', '<f8')])
                for tr in br:
                    if tr['Ju'] < self.fullnum + 1 and tr['Jl'] < self.fullnum + 1:
                        if self.A[inds[tr['Ju'] - 1], inds[tr['Jl'] - 1]] == 0:
                            self.A[inds[tr['Ju'] - 1], inds[tr['Jl'] - 1]] = Ai[tr['Ju'] - 1] * tr['br']
                        else:
                            print(inds[tr['Ju'] - 1], tr['Jl'] - 1, self.A[inds[tr['Ju'] - 1], inds[tr['Jl'] - 1]], Ai[tr['Ju'] - 1] * tr['br'])

        elif source == 'bautista':
            E = np.genfromtxt(folder + 'FeII/table3.dat', delimiter='|', dtype=[('J', 'i4'), ('term', '|S14'), ('name', '|S10'), ('energy', '<f8')])
            self.E = E['energy'] * 109737.31568160
            for e1, e2 in zip(np.sort(np.unique(lines['E_i'])), np.sort(self.E)):
                print(e1/e2 - 1, e1, np.argmin(np.abs(np.sort(self.E) - e1)))
            self.g = [2 * float(e.decode().split('_')[1].split('/')[0]) / float(e.decode().split('_')[1].split('/')[1]) + 1 for e in E['name']]

            self.fullnum = len(self.E)
            Ai = np.genfromtxt(folder + 'FeII/apj516563t4_ascii.txt', comments='#', unpack=True)
            Ai = np.insert(Ai[25] * 10 ** (- Ai[27]), 0, 0)
            br = np.genfromtxt(folder + 'FeII/table10.dat', dtype=[('Ju', 'i4'), ('Jl', 'i4'), ('l', '<f8'), ('br', '<f8'), ('br_e', '<f8')])
            self.A = np.zeros([self.fullnum, self.fullnum])
            for tr in br:
                if tr['Ju'] < self.fullnum + 1 and tr['Jl'] < self.fullnum + 1:
                    self.A[tr['Ju']-1, tr['Jl']-1] = Ai[tr['Ju']-1] * tr['br']
        coll = coll_list()
        coll_source = 'Bautista'
        if coll_source == 'Bautista':
            c = np.genfromtxt(folder + 'FeII/table12.dat', unpack=False)
            temp = [5000, 7000, 10000, 15000, 20000]
            for l in c:
                if l[0] < self.num+1 and l[1] < self.num+1:
                    coll.append(collision(self, 'e', inds[int(l[0]) - 1], inds[int(l[1]) - 1], np.array(
                                [np.log10(temp), [np.log10(float(l[k])) for k in range(2, 7)]])))

        elif coll_source == 'Ramsbottom':
            c = np.genfromtxt(folder + 'FeII/Ramsbottom/table3.dat', unpack=False)
            temp = [30, 100, 300, 500, 750, 1000, 1300, 1500, 1800, 2000, 2300, 2500, 5000, 7500, 10000, 13000, 15000, 18000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
            for l in c:
                #print(int(l[0]), l[1], [np.log10(float(l[k])) for k in range(2, len(temp)+2)])
                if l[0] < self.num+1 and l[1] < self.num+1:
                    coll.append(collision(self, 'e', int(l[0]) - 1, int(l[1]) - 1, np.array(
                                [np.log10(temp), [np.log10(float(l[k])) for k in range(2, len(temp)+2)]])))

        elif coll_source == 'chianti':
            c = np.genfromtxt(folder + r'FeII/fe_2.splups', comments='%', delimiter=(3, 3, 3, 3, 3, 10, 10, 10, 10, 10, 10, 10, 10))
            temp = np.linspace(3.7, 4.7, 6)
            for l in c:
                #print(l[2]-1, l[3]-1, [np.log10(float(l[k])) for k in range(7, len(temp)+7)])
                if l[2] < self.num+1 and l[3] < self.num+1:
                    coll.append(collision(self, 'e', int(l[2]) - 1, int(l[3]) - 1, np.array(
                                [temp, [np.log10(float(l[k])) for k in range(7, len(temp)+7)]])))
        self.coll['e'] = coll
        self.coll['e'].make_table(self.num+1)

    def read_H2(self):
        """
        read data for H2 (for nu=0): einstein coefficients, collisional rates
        :return:
        """
        self.fullnum = self.num
        data = np.genfromtxt(self.parent.folder + r'/data/pyratio/H2/energy_X.dat', comments='#', unpack=True)
        m = np.logical_and(data[0] == 0, data[1] < self.num)
        self.E = data[2][m]

        self.g = (2 * (np.arange(self.num) % 2) + 1) * (2 * np.arange(self.num) + 1)
        data = np.genfromtxt(self.parent.folder + r'/data/pyratio/H2/transprob_X.dat', comments='#', unpack=True)
        self.A = np.zeros([self.fullnum, self.fullnum])
        for l in data:  
            for i in range(self.num):
                for k in range(self.num):
                    m = (data[1] == 0) * (data[4] == 0) * (data[2] == i) * (data[5] == k)
                    if np.sum(m) > 0:
                        self.A[i, k] = data[6][m]

        # make FAKE collisional rates
        for partner in ['H', 'oH2', 'pH2', 'He']:
            coll = coll_list()
            for levels in [[1, 0], [2, 1], [2, 0]]:
                coll.append(collision(self, partner, levels[0], levels[1], np.array([[10, 100, 1000], [1, 1, 1]])))
            self.coll[partner] = coll

    def read_CO(self):
        """
        read data for CO: einstein coefficients, collisional rates
        """
        # self.Lambda_read('data/pyratio/co_data_old.dat')
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
                if '!NUMBER OF ENERGY LEVELS' in line:
                    num = int(f_in.readline())
                    self.E = np.zeros(num)
                    self.g = np.zeros(num)
                    d = self.Lambda_pars(f_in.readline())
                    for i in range(num):
                        line = f_in.readline().split()
                        dic = dict((k, line[i]) for i,k in enumerate(d))
                        self.E[int(dic['J'])] = float(dic['ENERGIES'])
                        self.g[int(dic['J'])] = float(dic['WEIGHT'])

                if '!NUMBER OF RADIATIVE TRANSITIONS' in line:
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
                if self.A[i, j] != 0:
                    self.B[i ,j] = self.A[i, j] * (self.E[i] - self.E[j]) ** (-3) / 8 / np.pi / ac.h.cgs.value
                    self.B[j, i] = self.B[i, j] * self.g[i] / self.g[j]

        self.Bij = self.B[:self.num, :self.num]

    def setAij(self):
        self.Aij = self.A[:self.num, :self.num]

    def critical_density(self, part, i, j, T):
        return self.A[i, j] / self.coll_rate(part, i, j, T)

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

class coll_list():
    """
    Class that contains lists of collisional data for specific collisional partner
    """
    def __init__(self):
        self.c = []
        self.f = None

    def append(self, object):
        self.c.append(object)

    def make_table(self, num):
        self.f = np.empty([num, num])
        for i in range(num):
            self.f[i, i] = 0
        for k, s in enumerate(self.c):
            self.f[s.i, s.j] = k + 1
            self.f[s.j, s.i] = -(k + 1)
        #print(self.f)

    def find(self, i, j):
        if self.f is None:
            for s in self.c:
                if s.i == i and s.j == j:
                    return s, 1
                elif s.i == j and s.j == i:
                    return s, -1
        else:
            if self.f[i, j] != 0:
                return self.c[int(np.abs(self.f[i, j])-1)], np.sign(self.f[i, j])

        return None, 0

    def rate(self, i, j, T):
        s, l = self.find(i, j)
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
        for s in self.c:
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
        if rate is not None:
            self.rate_int = interpolate.InterpolatedUnivariateSpline(rate[0], rate[1], k=2)

    def __str__(self):
        return "{0} collision rate with {1} for {2} --> {3}".format(self.parent.name, self.part, self.i, self.j)

    def rate(self, T, sign=1):
        if self.rates is None:
            return 0

        #l, u = max(self.i, self.j), min(self.i, self.j)
        #if self.i > self.j:
        #    u, l = l, u

        if self.part == 'e':
            c1 = self.parent.const2 * 10 ** (- T * 0.5) / self.parent.g[self.i] * np.exp(-(self.parent.E[self.j] - self.parent.E[self.i]) / 0.695 / 10**T)
        else:
            c1 = 1

        if sign == 1:
            Bf = 1
        else:
            Bf = self.parent.g[self.i] / self.parent.g[self.j] * np.exp(-(self.parent.E[self.i] - self.parent.E[self.j]) / 0.695 / 10 ** T)
            #Bf = self.parent.g[u] / self.parent.g[l] * np.exp(-(self.parent.E[u] - self.parent.E[l]) / 0.695 / 10**T)

        return 10 ** self.rate_int(T) * c1 * Bf

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
                            'n'    - number density
                            'T'    - temperature
                            'f'    - molecular fraction
                            'rad'  - Scaling constant of radiation filed iether:
                                        flux in the units of standard field (in case of sed_type: 'Habing', 'Draine', 'Mathis')
                                        flux at 1kpc distance for source specified by agn keyword (in case of sed_type: 'AGN', 'QSO', 'GRB')
                            'H'    - atomic hydrogen density
                            'e'    - electron density
                            'H2'   - molecular hydrogen density
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
        
        # >>> Radiation field
        if name == 'rad':
            self.label = r'$\log\,\xi$'
            self.init = 0
            self.range = [-4, 6]
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
    def __init__(self,
                 z=0,
                 f_He=0.085,
                 pars=None,
                 EBL=True,
                 CMB=True,
                 pumping='full',
                 radiation='full',
                 conf_levels=[0.68269],
                 logs=1,
                 calctype='popratios',
                 sed_type='Draine',
                 agn={'filter': 'r', 'mag': 20}
                 ):
        self.folder = os.path.dirname(os.path.realpath(__file__))
        self.species = collections.OrderedDict()
        self.pars = collections.OrderedDict()
        self.pumping = pumping
        self.radiation = radiation
        self.Planck1 = 8 * np.pi * ac.h.cgs.value
        self.Planck2 = (ac.h.cgs / ac.k_B.cgs * ac.c.cgs).value
        self.z = z
        self.logs = logs
        self.f_He = f_He
        self.theta_range = []
        self.calctype = calctype
        self.EBL = EBL
        if self.EBL:
            self.set_EBL()
        self.CMB = CMB
        self.sed_type = sed_type
        self.agn_pars = agn
        self.load_sed()

        if self.calctype == 'numbdens':
            self.set_pars(['Ntot'])
        elif pars is not None:
            self.set_pars(pars)
        self.conf_levels = conf_levels
        self.timer = Timer()

    def add_spec(self, name, n=None, num=None):
        d = {'OI': 3, 'CI': 3, 'CII': 2, 'SiII': 2, 'H2': 7, 'HD': 3, 'CO': 15, 'FeII': 13}
        if n is not None:
            n += [None] * (d[name] - len(n))
        else:
            n = [a()] * d[name]

        #print('add_spec:', n)
        self.species[name] = speci(self, name, n, num)
        if self.pumping == 'simple':
            self.pars['rad'].value = 0
            self.pump_matrix(name)
        if self.radiation == 'simple':
            self.pars['rad'].value = 0
            self.rad_matrix(name)

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
        if 0:
            T = 3
            print(sp.coll_rate('e',0,1,T), sp.coll_rate('e',1,0,T), np.exp(-(sp.E[1]-sp.E[0])/0.695/10**T)*sp.g[0]/sp.g[1])
            input()

        if 0:
            fig, ax0 = plt.subplots()
            plot_CollCrossSect(sp, 'oH2', 1, 0, ax0)
            plot_CollCrossSect(sp, 'pH2', 1, 0, ax0)
            plot_CollCrossSect(sp, 'H', 1, 0, ax0)
            plot_CollCrossSect(sp, 'He4', 1, 0, ax0)
      
    def print_pars(self):
        
        print('parameters in calculations: ', [p for p in self.pars])
        for p in self.pars.values():
            p.show()

    def flux_to_mag_solve(self, c, flux, x, b, inter, mag):
        m = - 2.5 / np.log(10) * (np.arcsinh(flux * 10 ** c * x ** 2 / ac.c.to('Angstrom/s').value / 3.631e-20 / 2 / b) + np.log(b))
        return mag - np.trapz(m * inter(x), x=x) / np.trapz(inter(x), x=x)

    def load_sed(self):
        """
        """
        if self.sed_type != None:
            if 'Mathis' in self.sed_type:
                l = np.linspace(912, 2460, 100)
                uv = np.zeros_like(l)
                mask = np.logical_and(912 <= l, l <= 1100)
                uv[mask] = 1.287e-9 * (l[mask] / 1e4) ** 4.4172
                mask = np.logical_and(1110 < l, l <= 1340)
                uv[mask] = 6.825e-13 * (l[mask] / 1e4)
                mask = np.logical_and(1340 < l, l <= 2460)
                uv[mask] = 2.373e-14 * (l[mask] / 1e4) ** (-0.6678)
                self.mathis = interpolate.interp1d(1e8 / l, uv / (ac.c.cgs.value / l / 1e-8), bounds_error=0, fill_value=0)

            if 'Draine' in self.sed_type:
                l = np.linspace(0.912, 3.1, 100)
                self.draine = interpolate.interp1d(1e5 / l, 6.84e-14 * l ** (-5) * (31.016 * l ** 2 - 49.913 * l + 19.897) / (ac.c.cgs.value / l / 1e-5), bounds_error=0, fill_value=0)

            if self.sed_type in ['QSO', 'AGN', 'GRB', 'power']:
                self.DL = FlatLambdaCDM(70, 0.3, Tcmb0=2.725, Neff=0).luminosity_distance(self.z).to('cm').value

            if self.sed_type in ['QSO', 'AGN', 'power']:
                b = {'u': 1.4e-10, 'g': 0.9e-10, 'r': 1.2e-10, 'i': 1.8e-10, 'z': 7.4e-10}
                fil = np.genfromtxt(self.folder + r'/data/SDSS/' + self.agn_pars['filter'] + '.dat', skip_header=6, usecols=(0, 1), unpack=True)
                filt = interpolate.interp1d(fil[0], fil[1], bounds_error=False, fill_value=0, assume_sorted=True)

            if 'QSO' in self.sed_type:
                from astroquery.sdss import SDSS
                qso = SDSS.get_spectral_template('qso')
                x, flux = 10 ** (np.arange(len(qso[0][0].data[0])) * 0.0001 + qso[0][0].header['CRVAL1']), qso[0][0].data[0] * 1e-17
                mask = (x * (1 + self.z) > fil[0][0]) * (x * (1 + self.z) < fil[0][-1])
                scale = 10 ** bisect(self.flux_to_mag_solve, -5, 5, args=(flux[mask], x[mask] * (1 + self.z), b[self.agn_pars['filter']], filt, self.agn_pars['mag']))
                #print(scale, flux_to_mag(flux * scale, x * (1 + self.z), self.agn_pars['filter']), self.agn_pars['mag'])
                self.qso = interpolate.interp1d(1e8 / x, scale * flux * (self.DL / ac.kpc.cgs.value) ** 2 * x ** 2 / 1e8 / ac.c.cgs.value ** 2 * (1 + self.z), bounds_error=0, fill_value=0)

            if 'AGN' in self.sed_type:
                if 1:
                    data = np.genfromtxt(self.folder + '/data/pyratio/QSO1_template_norm.sed', unpack=True, comments='#')
                    mask = (data[0] * (1 + self.z) > fil[0][0]) * (data[0] * (1 + self.z) < fil[0][-1])
                    scale = 10 ** bisect(self.flux_to_mag_solve, -25, 25, args=(data[1][mask], data[0][mask] * (1 + self.z), b[self.agn_pars['filter']], filt, self.agn_pars['mag']))
                    #print(scale)
                    self.agn = interpolate.interp1d(1e8 / data[0], scale * data[1] * (self.DL / ac.kpc.cgs.value) ** 2 * data[0] ** 2 / 1e8 / ac.c.cgs.value ** 2 * (1 + self.z), bounds_error=0, fill_value=0)

                else:
                    data = np.genfromtxt(self.folder + '/data/pyratio/Richards2006.dat', unpack=True, comments='#')
                    x = 1e8 * ac.c.cgs.value / 10 ** data[0]
                    mask = (x * (1 + self.z) > fil[0][0]) * (x * (1 + self.z) < fil[0][-1])
                    flux = 10 ** data[1][mask] / 4 / np.pi / self.DL ** 2 / x[mask] * (1 + self.z)
                    scale = 10 ** bisect(self.flux_to_mag_solve, -5, 5, args=(flux, x[mask] * (1 + self.z), b[self.agn_pars['filter']], filt, self.agn_pars['mag']))
                    print(scale, flux_to_mag(flux * scale, x[mask] * (1 + self.z), self.agn_pars['filter']), self.agn_pars['mag'])
                    self.agn = interpolate.interp1d(10 ** data[0] / ac.c.cgs.value, scale * 10 ** data[1] / 4 / np.pi / (ac.kpc.cgs.value) ** 2 / 10 ** data[0] / ac.c.cgs.value, bounds_error=0, fill_value=0)

            if 'power' in self.sed_type:
                alpha = 1.2
                lmin, lmax = 0.1,  20000
                l = np.logspace(np.log10(lmin), np.log10(lmax), 1000)
                mask = (l * (1 + self.z) > fil[0][0]) * (l * (1 + self.z) < fil[0][-1])
                scale = 10 ** bisect(self.flux_to_mag_solve, -25, 25, args=((l[mask] / lmax) ** (alpha), l[mask] * (1 + self.z), b[self.agn_pars['filter']], filt, self.agn_pars['mag']))
                #scale = np.trapz(self.power(x) / (ac.h.cgs.value * x), x)
                self.power = interpolate.interp1d(1e8 / l, scale * (l / lmax) ** (alpha) * (self.DL / ac.kpc.cgs.value) ** 2 * l ** 2 / 1e8 / ac.c.cgs.value ** 2 * (1 + self.z), bounds_error=False, fill_value=0)

    def set_EBL(self):
        """
        set EBL model radiation specific intensity J_nu (erg/s/cm^2/Hz/Sr) as a function of energy in cm^{-1}
        """
        files = glob.glob(self.folder+'/data/pyratio/KS18_Fiducial_Q18/*.txt')
        z = [float(f[-8:-4]) for f in files]
        ind = np.searchsorted(z, self.z)
        data_min = np.genfromtxt(files[ind - 1], comments='#', unpack=True)
        data_max = np.genfromtxt(files[ind], comments='#', unpack=True)
        self.ebl = interpolate.interp1d(1e8 / data_min[0], data_min[1] + (data_max[1] - data_max[1]) * (self.z - z[ind]) / (z[ind+1] - z[ind]), fill_value='extrapolate')

    def u_CMB(self, nu):
        """
        calculate CMB flux density at given wavelength array frequencies
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

    def balance(self, name=None, debug=None):
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
        if name is None:
            name = next(iter(self.species))

        speci = self.species[name]

        W = np.zeros([speci.num, speci.num])
        #self.timer.time('rates in')

        #self.timer.time('A')
        if debug in [None, 'A']:
            W += speci.Aij

        if debug in [None, 'C']:
            for u in range(speci.num):
                for l in range(speci.num):
                    if any(x in self.pars.keys() for x in ['n', 'e', 'H2', 'H']):
                        W[u, l] += self.collision_rate(speci, u, l)

        if debug in [None, 'CMB']:
            if 'CMB' in self.pars:
                W += np.multiply(speci.Bij, self.rad_field(speci.Eij, sed_type='CMB'))
                print(np.multiply(speci.Bij, self.rad_field(speci.Eij, sed_type='CMB')))

        if 'rad' in self.pars:

            if debug in [None, 'UV']:
                if self.pumping == 'full':
                    for u in range(speci.num):
                        for l in range(speci.num):
                            if 'UV' in self.pars:
                                W[u, l] += self.pumping_rate(speci, u, l)
                elif self.pumping == 'simple':
                    W += self.species[name].pump_rate * 10 ** self.pars['rad'].value

            if debug in [None, 'IR']:
                print(self.radiation)
                if self.radiation == 'full':
                    for u in range(speci.num):
                        for l in range(speci.num):
                            W[u, l] += speci.Bij[u, l] * self.rad_field(speci.Eij[u, l])

                if self.radiation == 'simple':
                    W += self.species[name].rad_rate * 10 ** self.pars['rad'].value

        #self.timer.time('solve')
        if debug is None:
            K = np.transpose(W).copy()
            for i in range(speci.num):
                for k in range(speci.num):
                    K[i, i] -= W[i, k]
            return np.insert(np.abs(np.linalg.solve(K[1:, 1:], -K[1:, 0])), 0, 1)

        elif debug in ['A', 'CMB', 'C', 'UV', 'IR']:
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
                coll += 10 ** self.pars[p].value * speci.coll[p].rate(u, l, self.pars['T'].value)
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
                if self.f_He != 0:
                    coll += 10 ** self.pars['n'].value * speci.coll['He4'].rate(u, l, self.pars['T'].value) * self.f_He / (self.f_He + 1 - m_fr / 2)
                    #print(coll, 10 ** self.pars['n'].value, self.f_He, speci.coll['He4'].rate(u, l, self.pars['T'].value))

        return coll

    def pumping_rate(self, speci, u, l):
        """
        calculates fluorescence excitattion rates for l -> u transition of given species
        x is the
        """
        pump = 0
        for k in range(speci.num, speci.fullnum):
            if speci.A[k,l] != 0:
                s = 0
                for i in range(speci.num):
                    s += speci.A[k,i] + self.exc_rate(speci, k, i)
                pump += self.exc_rate(speci, u, k) * (speci.A[k,l] + self.exc_rate(speci, k, l)) / s
        return pump
    
    def exc_rate(self, speci, u, l, logN=0, b=5):
        """
        calculates excitation rates (B_ul * u(\nu)) in specified field (by u), 
                            for l -> u transition of given species 
        """

        if logN > 0:
            t = tau(logN=logN, b=b, l=1e8 / np.abs(speci.E[u] - speci.E[l]))
            #print(u, l, speci.A[l, u], speci.g[u], speci.g[l])
            t.calctau0(speci.A[l, u], speci.g[u], speci.g[l])
            if t.f != 0:
                t.voigt_range()
                x = np.linspace(-t.dx, t.dx, 100)
                S = np.trapz(1.0 - np.exp(-t.tau0 * voigt(t.a, x)), x=x) / t.tau0 / np.sqrt(np.pi)
                #print('S: ', 1e8 / np.abs(speci.E[u]-speci.E[l]), self.rad_field(np.abs(speci.E[u]-speci.E[l])), np.log10(speci.B[u, l] * S), S)
            else:
                S = 0
        else:
            S = 1
        return speci.B[u, l] * self.rad_field(np.abs(speci.E[u]-speci.E[l])) * S

    def pump_matrix(self, name, logN=0, b=5):
        """
        calculates the pumping matrix for <speci> in simple case:
        1. optically thin limit
        2. spontangeous transitions (for higher levels) dominate
        3. background radiation is negligible.
        """
        speci = self.species[name]
        pump = np.zeros([speci.num, speci.num])
        for u in range(speci.num):
            for l in range(speci.num):
                for k in range(speci.num, speci.fullnum):
                    if speci.A[k, l] != 0:
                        #print(u, l,  1e8 / np.abs(speci.E[u]-speci.E[k]), self.exc_rate(speci, u, k, logN=logN, b=b) * speci.A[k, l] / np.sum(speci.A[k, :speci.num]))
                        pump[u, l] += self.exc_rate(speci, u, k, logN=logN, b=b) * speci.A[k, l] / np.sum(speci.A[k, :speci.num])
        #print(pump)
        self.species[name].pump_rate = pump

    def rad_matrix(self, name):
        """
        calculates the radiation matrix for <speci> in simple case:
        1. optical thin limit
        """
        speci = self.species[name]
        CMB, EBL = self.CMB, self.EBL
        self.CMB, self.EBL = False, False
        self.species[name].rad_rate = speci.Bij * self.rad_field(speci.Eij)
        self.CMB, self.EBL = CMB, EBL
        #print(self.species[name].rad_rate, self.species[name].Aij)


    def rad_field(self, e, sed_type=None):
        """
        radiation field density in [erg/cm^3/Hz]
        parameters:
            - e       :      energy of the transition in [cm^-1]
            - sed     :      type of interstellar field:
                                    - Habing (Habing 1968)
                                    - Draine (Draine 1978)
                                    - Mathis (Mathis 1983)
                                    - AGN (based on Richards+2006)
                                    - GRB (some model)
        return:
            uv       : energy density of UV field in [erg/cm^3/Hz]
        """
        if sed_type is None:
            s = self.sed_type
        else:
            s = sed_type

        e = np.asarray(e)
        field = np.zeros_like(e)
        m = e != 0

        if sed_type is None and self.CMB or 'CMB' == s:
            temp = self.pars['CMB'].value if 'CMB' in self.pars.keys() else 2.72548 * (1 + self.z)
            field[m] += self.Planck1 * e[m] ** 3 / (np.exp(self.Planck2 / temp * e[m]) - 1)

        if sed_type is None and self.EBL or 'EBL' == s:
            field[m] += self.ebl(e[m]) * 4 * np.pi / ac.c.cgs.value

        if s == 'Habing':
            m = (e > 6 * 8065.54) * (e < 13.5947 * 8065.54)
            field[m] += 4e-14 / (ac.c.cgs.value * e[m]) * 10 ** self.pars['rad'].value

        if s == 'Draine':
            field[m] += self.draine(e[m]) * 10 ** self.pars['rad'].value

        if s == 'Mathis':
            field[m] += self.mathis(e[m]) * 10 ** self.pars['rad'].value

        if s == 'AGN':
            field[m] += self.agn(e[m]) * 10 ** self.pars['rad'].value

        if s == 'QSO':
            field[m] += self.qso(e[m]) * 10 ** self.pars['rad'].value

        if s == 'power':
            field[m] += self.power(e[m]) * 10 ** self.pars['rad'].value

        if s == 'GRB':
            t_obs, alpha, beta, z = 393, -1.1, -0.5, 1.5
            field[m] += 1.12e-25 * (t_obs / 393) ** alpha * (1e8 / e[m] / 5439) ** (-beta) * 1.083e+7 ** 2 / (1 + z) / ac.c.cgs.value * 10 ** self.pars['rad'].value

        #print(s, field[m])

        return field

    def ionization_parameter(self, ne=None):
        if self.sed_type in ['AGN', 'power']:
            if ne == None:
                ne = 10 ** self.pars['e'].value
            x = np.logspace(np.log10(1e8/912), 8, 100)
            return np.trapz(self.rad_field(x) / (ac.h.cgs.value * x), x) * 10**self.pars['rad'].value / ne

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
    
    def predict(self, name=None, level=0, logN=None, plot=None):
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
            out = [np.log10(10 ** logN * x[i] / ref) for i in range(self.species[name].num)]

        if isinstance(logN, a):
            out = [logN * x[i] / ref for i in range(self.species[name].num)]

        if plot is not None and plot.lower() in ['energy', 'levels']:
            fig, ax = plt.subplots()
            if plot.lower() == 'energy':
                 xi = self.species[name].E[:len(x)]
            if plot.lower() == 'levels':
                 xi = np.arange(len(x))

            print(xi, self.species[name].g[:len(x)], out - self.species[name].g[:len(x)])
            if isinstance(logN, (int, float)):
                ax.plot(xi, out - self.species[name].g[:len(x)], 'ok')
            if isinstance(logN, a):
                ax.errorbar(xi, [o.val for o in out] - self.species[name].g[:len(x)], yerr=[[o.plus for o in out], [o.minus for o in out]])

        return out

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
        for i, s in enumerate(self.species.values()):
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
                axi.axhline(yg.val, ls='--', c=colors[2*k])
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
                  title='', ax=None, fig=None, alpha=1, color=None, cmap='PuBu', color_point=None,
                  zorder=1):
        """
        calculate ranges for two parameters using given populations of levels
        parameters:
            - grid_num     :  number of points in the grid for each parameter
            - plot         :  specify graphs to plot: 1 is for contour, 2 is for filled contours
            - output       :  name of output file for probability
            - marginalize  :  provide 1d estimates of parameters by marginalization
            - limits       :  show upper of lower limits: 0 for not limits, >0 for lower, <0 for upper
            - title        :  title name for the plot
            - ax           :  axes object where to plot data, of not present make it if plot==1.
            - fig          :  fig object where to plot data, needed for 2dplot with marginalization
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
                               title=title, fig=fig, ax=ax, alpha=alpha, color=color, cmap=cmap, color_point=color_point, zorder=zorder)

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
                title='', fig=None, ax=None, alpha=1, color=None, cmap='PuBu', color_point='gold', zorder=1):
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
            - fig          :  fig object where to plot data, needed for plot with marginalization
            - ax           :  axes object where to plot data, of not present make it if plot==1.
            - alpha        :  alpha value for the contour plots
            - color        :  color of the regions in the plot
            - cmap         :  color map for the countour fill
            - color_point  :  color for indicator for best fit
            - zorder       :  zorder of the graph object 
        """

        # >> colors for plotting
        if ax is None and fig is None:
             fig, ax = plt.subplots()

        if color is None:
            color = 'orangered' #(1, 0, 0)
        print(self.pars)

        out = {}

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #
        #++++  Direct calculations  ++++++++++++++++++++
        if verbose:
            self.print_species()
            self.print_pars()

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

        print(output)
        if output is not None:
            with open(output, "wb") as f:
                pickle.dump([X1, X2, Z], f)

        d = distr2d(X1, X2, np.exp(Z))
        if marginalize:
            d.plot(fig=fig, frac=0.2, indent=0.1, limits=None, ls=None,
                   xlabel=self.pars[vary[0]].label, ylabel=self.pars[vary[1]].label,
                   color=color, color_point=color_point, color_marg='dodgerblue', cmap=cmap,
                   alpha=alpha, colorbar=False,
                   font=18, title=None, zorder=zorder)
        else:
            d.plot_contour(ax=ax, limits=None, ls=None,
                           xlabel=self.pars[vary[0]].label, ylabel=self.pars[vary[1]].label,
                           color=color, color_point=color_point, cmap=cmap, alpha=alpha, colorbar=False,
                           font=18, title=None, zorder=zorder)

        out[vary[0]], out[vary[1]], out['lnL'] = X1, X2, Z
        point = d.dopoint()

        d1 = d.marginalize('x')
        d1.dopoint()
        d1.dointerval()
        print('marg point:', d1.point)
        print('marg interval:', d1.interval)
        d2 = d.marginalize('y')
        d2.dopoint()
        d2.dointerval()
        print('marg point:', d2.point)
        print('marg interval:', d2.interval)

        out['res_'+vary[0]] = a(d1.point, d1.interval[1] - d1.point, d1.point - d1.interval[0])
        out['res_'+vary[1]] = a(d2.point, d2.interval[1] - d2.point, d2.point - d2.interval[0])

        if marginalize:
            d1.plot()
            d2.plot()

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

        for sp in self.species.values():
            if species is None or species == sp.name:
                if depend in ['T'] and ax is None:
                    fig, axi = plt.subplots(sp.num-1, sp.num-1, sharex=True, sharey=True)
                if verbose:
                    print('species: ', sp.name)
                for i in range(sp.num-1):
                    for j in range(i+1, sp.num):
                        if sp.num > 2 and ax is None:
                            axs = axi[i, j-1]
                        elif sp.num == 2 and ax is None:
                            axs = axi
                        elif ax is not None:
                            axs = ax
                        if verbose:
                            coll = 0
                            for k in range(sp.num):
                                coll += self.collision_rate(sp, j, k) / 10 ** self.pars['n'].value
                        if depend in ['T']:
                            pars_range = np.linspace(self.pars[depend].range[0], self.pars[depend].range[1], 10)
                            y = []
                            for t in pars_range:
                                self.pars[depend].value = t
                                coll = 0
                                for k in range(sp.num):
                                    coll += self.collision_rate(sp, j, k)
                                y.append(coll/10**self.pars['n'].value)
                            axs.plot(pars_range, np.log10((y/sp.A[j, i])**(-1)), '-b')
                            axs.text(0.9, 0.9, "{:0}-{:1}".format(j, i), transform=axs.transAxes, fontsize=20)
                            axs.set_xlabel(r'$\log T$')
                            axs.set_ylabel(r'$\log n_{\rm cr}$')


    def plot_cooling(self, verbose=0, depend=None, ax=None, species=None, color='k', factor=0):
        """
        Function for calculation cooling function by spontangeous transitions 
        in different species, like HD, H2, CII, OI, CI by 1 atom (molecule)
        
        Cooling function is derived using formula 
    
        L_ij = A_ij E_ij n_i
        
        applied to measured populations of fine structure levels, or rotational levels
        """    
        print(factor)
        out = []
        for sp in self.species.values():
            if species is None or sp.name == species:
                if verbose == 1:        
                    print('species: ', sp)
                    print('column densities: ', sp.n)
                    print('data ratios: ', sp.y)
                if depend in ['T', 'n']:
                    if ax is None:
                        fig, ax = plt.subplots()
                    pars_range = np.linspace(self.pars[depend].range[0], self.pars[depend].range[1], 100)
                    L = np.zeros_like(pars_range)
                else: 
                    L = 0
                for u in range(sp.num-1, 0, -1):
                    for l in range(u):
                        y = np.zeros_like(L)
                        if depend in ['T', 'n']:
                            for i, value in enumerate(pars_range):
                                self.pars[depend].value = value
                                y[i] = self.cooling_rate(sp, u, l).value
                            if 0:
                                ax.plot(pars_range, np.log10(y)+factor, '--', color=color, label="{0} {1}-{2}".format(sp.name, u, l))
                        else:
                            y = self.cooling_rate(sp, u, l).value
                        L += y
                if depend in ['T', 'n']:            
                    ax.plot(pars_range, np.log10(L)+factor, '-', color=color, lw=1.5, label=sp.name+" total")
                out.append([s.name, L])
        return out

    def calc_cooling(self, species=None, n=[], T=[], verbose=0):
        """
        Function for calculation cooling function by spontangeous transitions
        in different species, like HD, H2, CII, OI, CI by 1 atom (molecule)

        Cooling function is derived using formula

        L_ij = \sum_i A_ij E_ij n_i

        applied to measured populations of fine structure levels, or rotational levels
        """
        if species is None and len(self.species) > 0:
            species = list(self.species.keys())[0]
        if species is not None:
            sp = self.species[species]
            if verbose:
                print('species: ', sp.name)
                print('number of levels: ', sp.num)
            if isinstance(n, (int, float)):
                n = [n]
            if isinstance(T, (int, float)):
                T = [T]
            if len(n) == 0 and len(T) != 0:
                n = np.ones_like(T) * self.pars['n'].value
            if len(T) == 0 and len(n) != 0:
                T = np.ones_like(n) * self.pars['T'].value
            if len(n) == 0 and len(T) == 0:
                n, T = [self.pars['n'].value], [self.pars['T'].value]
            if len(n) == 1 or len(T) == 1:
                if len(n) > len(T):
                    T = np.ones_like(n) * T[0]
                else:
                    n = np.ones_like(T) * n[0]
            L = np.zeros(len(n))
            for ni, Ti, i in zip(n, T, range(len(n))):
                self.pars['n'].value, self.pars['T'].value = ni, Ti
                for u in range(sp.num - 1, 0, -1):
                    for l in range(u):
                        #print(ni, Ti, self.balance(sp.name), self.cooling_rate(sp, u, l).value)
                        L[i] += self.cooling_rate(sp, u, l).value

            return L


    def cooling_rate(self, sp, u, l):
        """
        calculate the colling rate of the medium due to transition <u> -> <l> for <s> species per one atom of species
        parameters:
            - sp     :  species
            - u      :  upper level
            - l      :  lower level
        return: L
            - L      :  coollinf rate in erg/s/cm^3 
        """
        hu = (sp.E[u] - sp.E[l]) / au.cm * ac.c.cgs * ac.h.cgs  # in ergs
        if 0:
            stat = s.g[u]/s.g[l]*np.exp(-hu/(ac.k_B.cgs*10**self.pars['T'].value*au.K))
            #print(u, l, s.A[u,l], stat, s.A[u,l]/self.collision_rate(s, u, l)/10**self.pars['n'].value)
            L = s.A[u,l] / au.s * hu * stat / (1 + stat + s.A[u,l]/self.collision_rate(s, u, l))
        else:
            x = self.balance(sp.name)
            L = sp.A[u, l] / au.s * hu * x[u]
        #print(sp.name, u, l, (sp.A[u, l] / au.s * hu).to('erg/s'))
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
        for i, s in enumerate(self.species):
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

    def calc_emission_CO(self, par, grid_num=50, ax=None, specific='co', coh2=4e-5, title='', alpha=1, stats=False):
        """
        calculate the dependence of CO emissivity in J->J-1 transition per CO atom from physical parameter

        parameters:
            - par        : name of parameter
            - col        : column densities. If None -- read from species, if float or int -- the total, then for each par column densities will be derived
            - grid_num   : number of points in the grid for each parameter
            - ax         : axes object where to plot data, of not present make it if plot==1.
            - specific   : if 'co' then per one co atom, if 'h2' then per one h2 atom, if <number>, then per <number> of solar masses
            - coh2       : coh2 ratio
            - title      : set title of the plot
            - alpha      : alpha value for the contour plots
            - stats      : if True, then include stat weights: i.e. N[k]/g[k] / N[0]/g[0]
        """

        # >> colors for plotting
        colors = ['#fb9a99', '#e31a1c', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fdbf6f', '#ff7f00', '#aaaaaa', '#666666']

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10 * len(self.species)))

        if specific == 'co':
            sp = 1
        elif specific == 'h2':
            sp = coh2
        elif isinstance(specific, (float, int)):
            sp = coh2 * ac.M_sun.cgs / 2.3 / ac.m_p.cgs / ac.L_sun.cgs.value

        x = np.linspace(self.pars[par].range[0], self.pars[par].range[1], grid_num)
        for s in self.species.values():
            if s.name == 'CO':
                c = np.zeros(s.num-1)
                for k in range(s.num-1):
                    mu = 0.11e-18
                    # constant in (K km/s)
                    #c[k] = 8 * np.pi**3 * (s.E[k+1]-s.E[k]) * ac.c.cgs.value / 3 / ac.k_B.cgs.value * mu**2 * (k+1) / s.g[k+1] / 1e5
                    # constant h*\nu_{ik}*A_{ik} in erg/s
                    c[k] = 64 * np.pi**4 / 3 * ac.c.cgs.value * (s.E[k+1]-s.E[k])**4 * mu**2 * (k+1) / s.g[k+1]
                print(c[0])

                f = np.zeros([len(x), s.num - 1])
                f_CMB = np.zeros([len(x), s.num - 1])
                for k, z in enumerate(x):
                    self.pars[par].value = z
                    f[k] = self.balance(s.name)[1:]
                    print(f[k])
                    f_CMB[k] = 1 - (np.power(f[k] / s.g[1:s.num], -1) - 1)/(np.exp((s.E[1:s.num] - s.E[0:s.num-1]) / 0.695 / (2.725 * (1 + self.z))) - 1)
                    #if col is None:
                    #    f[k] = s.n[1:].val
                    #else:
                    #    f[k] = np.log10(f[k] * 10**col / sum(f[k]))
                    #print(f_CMB[k], f[k])

                for k in range(s.num-1):
                    print(c[k] * f[:, k])
                    ax.plot(x, np.log10(sp * c[k] * f[:, k]), label=r'{:}$\to${:}'.format(k + 1, k), lw=3,
                            #color=colors[2 * k + 1],
                            )

                ax.set_title(s.name)
                ax.set_ylabel(r'W$_{\rm CO}$ [K km/s]')
                ax.set_xlabel(self.pars[par].label)
                #ax.set_ylim([-22, -15])
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
        pr.add_spec('CO', num=7)
        d = {'T': 1.5, 'n': 5, 'f': 0}
        pr.set_pars(d.keys())
        pr.pars['n'].range = [1, 5]
        for d in d.items():
            pr.pars[d[0]].value = d[1]
        print(pr.predict('CO', level=-1, logN=14., plot='levels'))
        pr.calc_emission_CO(par='n', specific=1, coh2=4e-5)
        plt.show()

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
        ax.plot(l, pr.uv(nu, sed_type='Habing'))
        ax.plot(l, pr.uv(nu, sed_type='Draine'))
        ax.plot(l, pr.uv(nu, sed_type='Mathis'))
        print('G_0 of Draine:', np.trapz(pr.uv(nu, sed_type='Draine')/l)*abs(l[1]-l[0])/5.29e-14)
        print('G_0 of Mathis:',  np.trapz(pr.uv(nu, sed_type='Mathis')/l)*abs(l[1]-l[0])/5.29e-14)
        

    # CI calculation
    if 1:
        fig, ax = plt.subplots()
        z_dla = 1.70465
        pr = pyratio(z=z_dla)
        pr.set_pars(['T', 'n', 'f', 'rad'])
        pr.set_prior('f', a(0, 0, 0))
        pr.set_prior('T', a(2.0, 0, 0))
        pr.pars['n'].range = [0, 5]
        pr.pars['n'].value = 2
        pr.pars['rad'].range = [0, 5]
        CIj0, CIj1, CIj2 = a(14.30, 0.1, 0.1), a(14.76, 0.1, 0.1), a(14.74, 0.1, 0.1)
        pr.add_spec('CI', [CIj0, CIj1, CIj2])
        #CI = pr.calc_grid(grid_num=20, plot=1, verbose=1, marginalize=True, alpha=1, color='greenyellow', cmap=None, color_point='gold', zorder=10)
        #print(pr.calc_cooling(n=[2], T=[2]))
        #plt.savefig('pyratio.png', bbox_inches='tight', transparent=True)
        plt.show()

    # >>> check CII collisions
    if 0:
        pr = pyratio(z=2.65)
        pr.set_pars(['T', 'n', 'f', 'e'])
        pr.pars['T'].range = [1, 6]
        pr.pars['n'].range = [-1, 4]
        pr.pars['e'].range = [-4, 0]
        pr.pars['f'].range = [-6, 0]
        pr.set_fixed('f', -5)
        pr.set_fixed('e', -4)
        species = 'CII'
        pr.add_spec(species, num=2)
        pr.critical_density(depend='T')
        print(pr.calc_cooling(n=np.linspace(-5, 6, 20), T=4, verbose=1))
        num = 20
        if 1:
            fig, ax = plt.subplots()
            for t, f, ls in zip([100, 100, 10000, 15000], [0, -4, -4, -4], ['--', '-', '-', '-']):
                pr.pars['T'].value = np.log10(t)
                pr.set_fixed('f', f)
                n = np.linspace(-2, 5, num)
                z = np.zeros_like(n)
                for i, ni in enumerate(n):
                    pr.pars['n'].value = ni
                    z[i] = pr.predict()[1]
                ax.plot(n, z, ls=ls, label='{0:d}'.format(t))
            for n in [10, 40]:
                ax.axvline(np.log10(n))
            ax.set_xlabel('log n')
            ax.set_ylabel('CII*/HI/Z')
            ax.legend()

        if 0:
            fig, ax = plt.subplots()
            pr.set_fixed('n', -4)
            for t in [1000, 6000, 10000]:
                pr.pars['T'].value = np.log10(t)
                e = np.linspace(-4, 0, num)
                z = np.zeros_like(e)
                for i, ei in enumerate(e):
                    pr.pars['e'].value = ei
                    z[i] = pr.predict()[1]
                ax.plot(e, z, '-', label='{0:d}'.format(t))
                #ax.plot(e, z - 3.57, '-', label='{0:d}'.format(t))
            ax.set_xlabel('log $n_e$')
            ax.set_ylabel('CII*/CII')
            ax.legend()

        if 0:
            fig, ax = plt.subplots()
            for t, f, ls in zip([100, 100, 8000, 15000], [0, -4, -4, -4], ['--', '-', '-', '-']):
                pr.pars['T'].value = np.log10(t)
                pr.set_fixed('f', f)
                p = np.linspace(3, 5, num)
                z = np.zeros_like(p)
                for i, pi in enumerate(p):
                    pr.pars['n'].value = pi - np.log10(t) + 0.3 * 10**pr.pars['f'].value
                    z[i] = pr.predict()[1]
                ax.plot(p, z - 3.57, ls=ls, label='{0:d}'.format(t))

            ax.set_xlabel('log p')
            ax.set_ylabel('CII*/CII')
            ax.legend()

        if 0:
            fig, ax = plt.subplots()
            for n in [0.1, 1, 100]:
                pr.pars['n'].value = np.log10(n)
                T = np.linspace(1, 4, num)
                z = np.zeros_like(T)
                for i, ti in enumerate(T):
                    pr.pars['T'].value = ti
                    z[i] = pr.predict()[1]
                ax.plot(T, z, '-')
                print(T, z)
            ax.set_xlabel('log T')
            ax.set_ylabel('CII*/CII')

        if 0:
            n = np.linspace(-1, 3, num)
            T = np.linspace(2, 5, num)
            X, Y = np.meshgrid(n, T)
            z = np.zeros_like(X)
            for i, ni in enumerate(n):
                pr.pars['n'].value = ni
                for k, tk in enumerate(T):
                    pr.pars['T'].value = tk
                    pop = pr.predict()
                    z[k, i] = pop[1]

            fig, ax = plt.subplots()
            cs = ax.contourf(X, Y, z, levels=100)
            cs1 = ax.contour(X, Y, z, levels=[-4, -3, -2, -1, 0.0], linestyles='--', colors='k')
            # cbar = plt.colorbar(cs)
            # cbar.ax.set_ylabel('log SiII*/SiII')
            ax.clabel(cs1, cs1.levels[::2], inline=True, fmt='%3.1f', fontsize=12)
            ax.set_xlabel('log n')
            ax.set_ylabel('log T')
            ax.set_title('log SiII*/SiII')
            plt.colorbar(cs)

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

    # emissivities of CO lines:
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

    # >>> check SiII
    if 0:
        pr = pyratio(z=2.65, pumping='simple', radiation='simple', sed_type='AGN', agn={'filter': 'r', 'mag': 20.22})
        pr.set_pars(['T', 'n', 'f', 'rad'])
        pr.pars['T'].range = [1.5, 5]
        pr.set_prior('T', a(2.2, 0, 0))
        pr.pars['rad'].range = [-2, 4]
        pr.pars['n'].range = [1, 4]
        pr.pars['f'].range = [-6, 0]
        pr.set_fixed('f', -3)
        species = 'SiII'
        pr.add_spec(species, num=2)
        pr.pump_matrix('SiII', logN=15, b=10)
        pr.pars['n'].value = 2.5
        pr.pars['rad'].value = np.log10(1)
        if 0:
            print('IR:', pr.balance(species, debug='IR'))
            print('UV:', pr.balance(species, debug='UV'))
            print('coll:', pr.balance(species, debug='C'))
            print(pr.predict())
        if 1:
            n = np.linspace(1, 8, 20)
            if pr.sed_type == 'Draine':
                rad = np.linspace(-4, 6, 20)
            elif pr.sed_type in ['QSO', 'AGN']:
                rad = np.linspace(-2, 2, 20)
            X, Y = np.meshgrid(n, rad)
            z = np.zeros_like(X)
            for i, ni in enumerate(n):
                pr.pars['n'].value = ni
                for k, rk in enumerate(rad):
                    if pr.sed_type == 'Draine':
                        pr.pars['rad'].value = rk
                    elif pr.sed_type in ['QSO', 'AGN']:
                        pr.pars['rad'].value = - 2 * rk
                    pop = pr.predict()
                    z[k, i] = pop[1]
                    #print(pop, z[k, i])

            fig, ax = plt.subplots()
            cs = ax.contourf(X, Y, z, levels=100)
            cs1 = ax.contour(X, Y, z, levels=[-3, -2, -1, 0.0], linestyles='--', colors='k')
            ax.clabel(cs1, cs1.levels[::2], inline=True, fmt='%3.1f', fontsize=12)
            ax.set_xlabel('log n')
            if pr.sed_type == 'Draine':
                ax.set_ylabel('log UV')
            elif pr.sed_type in ['QSO', 'AGN']:
                ax.set_ylabel('log (r / kpc)')
            plt.colorbar(cs)
            ax.set_title('logN(SiII) = 15')
            #ax.set_title('optically thin')
        else:
            pr.pars['n'] = 2
        plt.show()

    # >>> check SiII collisions
    if 0:
        pr = pyratio(z=2.65)
        pr.set_pars(['T', 'n', 'f'])
        pr.pars['T'].range = [1, 6]
        pr.pars['n'].range = [1, 4]
        pr.pars['f'].range = [-6, 0]
        pr.set_fixed('f', -3)
        species = 'SiII'
        pr.add_spec(species, num=2)
        num = 20
        if 1:
            n = np.linspace(0, 5, num)
            T = np.linspace(2, 5, num)
            X, Y = np.meshgrid(n, T)
            z = np.zeros_like(X)
            for i, ni in enumerate(n):
                pr.pars['n'].value = ni
                for k, tk in enumerate(T):
                    pr.pars['T'].value = tk
                    pop = pr.predict()
                    z[k, i] = pop[1]

            fig, ax = plt.subplots()
            cs = ax.contourf(X, Y, z, levels=100)
            cs1 = ax.contour(X, Y, z, levels=[-4, -3, -2, -1, 0.0], linestyles='--', colors='k')
            #cbar = plt.colorbar(cs)
            #cbar.ax.set_ylabel('log SiII*/SiII')
            ax.clabel(cs1, cs1.levels[::2], inline=True, fmt='%3.1f', fontsize=12)
            ax.set_xlabel('log n')
            ax.set_ylabel('log T')
            ax.set_title('log SiII*/SiII')
            plt.colorbar(cs)

        plt.show()

    # >>> check ionization parameter
    if 0:
        pr = pyratio(z=2.0, pumping='simple', radiation='simple', sed_type='AGN', agn={'filter': 'r', 'mag': 18})
        pr.set_pars(['T', 'rad', 'e'])
        pr.pars['T'].value = 4
        pr.pars['e'].value = 0
        pr.pars['rad'].value = 0
        q = np.log10(pr.ionization_parameter())
        print(q)

        e = np.logspace(2, 6, 10000)
        species = 'FeII'
        pr.add_spec(species)

        levels = [1, 2, 3, 4, 9]
        fig, ax = plt.subplots(nrows=len(levels), ncols=3, figsize=(17, 5 * len(levels)))
        n = np.linspace(2, 6, 15)
        rad = np.linspace(-2, 2, 15)
        d = {}
        for l in levels:
            d[l] = {'f': np.zeros([len(rad), len(n)]), 'r': np.zeros([len(rad), len(n)]), 'c': np.zeros([len(rad), len(n)])}
        for i, ri in enumerate(rad):
            for k, nk in enumerate(n):
                pr.pars['e'].value = nk
                pr.pars['rad'].value = ri
                for l in levels:
                    d[l]['f'][k, i] = pr.balance(species)[l]
                pr.pars['e'].value = -8
                for l in levels:
                    d[l]['r'][k, i] = pr.balance(species)[l]
                pr.pars['e'].value = nk
                pr.pars['rad'].value = -8
                for l in levels:
                    d[l]['c'][k, i] = pr.balance(species)[l]
                #print(ri, nk, d[l]['f'][k, i], d[l]['c'][k, i], d[l]['r'][k, i])

        R, N = np.meshgrid(rad, n)
        def plot_cont(axi, z, vmin=-2, vmax=0, xlabel=None, ylabel=None, cmap='viridis'):
            cs = axi.contourf(R, N, z, vmin=vmin, vmax=vmax, levels=np.linspace(vmin, vmax, 6), cmap=cmap, alpha=0.8)
            cbar = fig.colorbar(cs, ax=axi, shrink=0.9)
            if xlabel is not None:
                cbar.ax.set_xlabel(xlabel)
            if ylabel is not None:
                cbar.ax.set_ylabel(ylabel)
            Q = np.linspace(-3, 1, 3)
            xp, yp = [-1.7, 0, 1], [4.9, 4.6, 3.5]
            for qi, x, y in zip(Q, xp, yp):
                axi.plot(rad, q - qi + rad, '--k', lw=2, zorder=9)
                text = axi.text(x, y, r'Q = {0:d}'.format(int(qi)), va='bottom', ha='left')
                text.set_rotation(40)
            axi.set_xlim([rad[0], rad[-1]-0.1])
            axi.set_ylim([n[0], n[-1]])

        for i, l in enumerate(levels):
            plot_cont(ax[i, 0], np.log10(d[l]['f']), -2, 0, xlabel=f'$n_{l}/n_{0}$')
            plot_cont(ax[i, 1], np.log10(d[l]['c'] / d[l]['f']), -2, 0, ylabel='collisions', cmap='RdBu_r')
            plot_cont(ax[i, 2], np.log10(d[l]['r'] / d[l]['f']), -2, 0, ylabel='radiative', cmap='RdBu_r')
        plt.show()

    # >>> check radiation fields
    if 0:
        z, r, d = 2.65, 20.2, 5
        pr = pyratio(z=z, pumping='simple', radiation='simple', sed_type='power', agn={'filter': 'r', 'mag': r})
        pr.set_pars(['T', 'rad'])
        print(pr.ionization_parameter())


    # >>> check radiation fields
    if 0:
        z, r, d = 2.8, 17.7, 150
        z, r, d = 2.65, 20.2, 5
        pr = pyratio(z=z, pumping='simple', radiation='simple', sed_type='Draine', agn={'filter': 'r', 'mag': r})
        pr.set_pars(['T', 'rad'])
        e = np.logspace(-1, 6, 10000)
        fig = plt.figure(figsize=(10, 6))
        add_lines = 1
        if add_lines:
            ax = plt.axes([0.10, 0.10, 0.90, 0.61])
            ax2 = plt.axes([0.10, 0.71, 0.90, 0.25])
        else:
            ax = plt.axes([0.10, 0.10, 0.90, 0.90])

        ax.plot(np.log10(e), np.log10(pr.rad_field(e, sed_type='CMB')), label='CMB, z={0:3.1f}'.format(z))
        ax.plot(np.log10(e), np.log10(pr.rad_field(e, sed_type='EBL')), label='EBL, z={0:3.1f}'.format(z))
        ax.plot(np.log10(e), np.log10(pr.rad_field(e, sed_type='Draine') * 1000), label='Draine x1000')
        for sed, label in zip(['AGN', 'QSO', 'GRB', 'power'], ['AGN, r={0:4.1f}, z={1:3.1f}, d={2:d}kpc'.format(r, z, d), 'QSO, r={0:4.1f}, z={1:3.1f}, d={2:d}kpc'.format(r, z, d), 'GRB, d={0:d}kpc'.format(d), 'power, d={0:d}kpc'.format(d)]):
            if sed not in ['GRB']:
                pr.sed_type = sed
                pr.load_sed()
                if sed == 'power':
                    print(np.sum(pr.rad_field(e, sed_type=sed) != 0))
                print(sed, pr.ionization_parameter(ne=1e4))
                ax.plot(np.log10(e), np.log10(pr.rad_field(e, sed_type=sed)) - np.log10(d)*2, label=label)

        ax.set_ylim([-39, -18])
        ax.set_xlim(ax.get_xlim())
        ax.set_ylabel(r'log u$_{\nu}$ [$\rm erg\,cm^{-3}\,Hz^{-1}$]')
        ax.set_xlabel(r'log E [$\rm cm^{-1}$]')
        ax.legend()

        if 0:
            #with open("C:\Users\Serj\Downloads\Slack\J0015BCJ0015BC.con", 'r') as f:
            #    f.readlines()
            data = np.genfromtxt("C:/Users/Serj/Downloads/Slack/J0015BCJ0015BC.con", comments='#', usecols=(0, 1), unpack=True)
            inds = np.where(np.diff(data[0]) > 0)[0]
            inds = np.insert(inds, 0 , 0)
            for s, e in zip(inds[:-1], inds[1:]):
                ax.plot(np.log10(1e8 / data[0][s+1:e-1]), np.log10(data[1][s+1:e-1] * data[0][s+1:e-1] * 1e-8 / (ac.c.cgs.value ** 2)), '--k')

        if 0:
            qso = np.genfromtxt('C:/science/data/swire_library/QSO1_template_norm.sed', unpack=True)
            ax.plot(np.log10(1e8 / qso[0]), np.log10(qso[1] * qso[0]**2 / ac.c.cgs.value / 1e8 / 1e10), )
            print(qso[1] * qso[0]**2 / ac.c.cgs.value / 1e8 / 1e10)
        mpl.rcParams['hatch.linewidth'] = 5

        if add_lines:
            species = 'SiII'
            pr.add_spec(species)
            inds = np.where(pr.species[species].A != 0)
            E = np.abs(pr.species[species].E[inds[0]] - pr.species[species].E[inds[1]])
            if 0:
                ax2.scatter(np.log10(E), np.log10(pr.species[species].A[inds[0], inds[1]]))
                ax2.set_ylabel(r'$\rm \log\,A_{ik}, s^{-1}$')
            else:
                ax2.scatter(np.log10(E), np.log10(pr.species[species].B[inds[0], inds[1]] * (pr.rad_field(E) * 10**d)))
                ax2.set_ylabel(r'$\rm \log\,B_{ik} \rho(E_{ik}), s^{-1}$')

            ax2.set_xlim(ax.get_xlim())
            ax2.axvspan(np.log10(1e8 / 912), ax2.get_xlim()[-1], facecolor='lightseagreen', edgecolor='k', alpha=0.1, hatch='/', zorder=0)
            ax2.set_title(species, fontsize=16)

        ax.axvspan(np.log10(1e8 / 912), ax.get_xlim()[-1], facecolor='lightseagreen', edgecolor='k', alpha=0.1, hatch='/', zorder=0)
        ax.text(np.log10(1e8 / 912), np.mean(ax.get_ylim()), 'Ly cutoff', rotation=90, ha='left', va='center')
        fig.savefig(r'C:/users/serj/desktop/' + species + '.png')
        plt.show()

    # >>> FeII calculations
    if 0:
        pr = pyratio(z=0.34, pumping='simple', radiation='simple', sed_type='AGN', agn={'filter': 'r', 'mag': 18})
        pr.set_pars(['T', 'rad', 'e'])
        pr.pars['T'].range = [3, 5]
        pr.add_spec('FeII', num=13)
        pr.pars['T'].value = np.log10(10000)
        pr.pars['e'].value = np.log10(5)
        pr.pars['rad'].value = np.log10(1)
        print('IR:', pr.balance('FeII', debug='IR')[:3, :3])
        print('UV:', pr.balance('FeII', debug='UV')[:3, :3])
        print('coll:', pr.balance('FeII', debug='C')[:3, :3])
        #input()
        rad_range = np.linspace(-4, 8, 50)

        if 1:
            fig, ax = plt.subplots()
            n = []
            for rad in rad_range:
                pr.pars['rad'].value = rad
                n.append(pr.predict())
            for i, ni in enumerate(np.transpose(n)[1:13]):
                ax.plot(rad_range/-2, 10**ni, ls='-', label=str(i+1))

            ax.set_xlim([rad_range[0]/-2, rad_range[-1]/-2])
            ax.set_ylim([0, 1.2])
            ax.set_xlabel('log d, [1 kpc]')
            ax.legend(frameon=False, fontsize=14)

            ax.set_ylabel('relative population to the ground level')

            plt.tight_layout()
            plt.savefig('C:/users/serj/desktop/AGN.png')
            plt.show()

    # >>> OI calculations
    if 0:
        pr = pyratio(z=0, pumping='simple', radiation='simple')
        pr.set_pars(['T', 'n', 'f', 'rad'])
        pr.pars['T'].range = [1, 4]
        pr.pars['n'].range = [1, 4]
        pr.pars['rad'].range = [0, 4]
        pr.pars['n'].value = 2.3
        pr.pars['rad'].value = -4
        pr.pars['T'].value = np.log10(150)
        pr.set_prior('f', a(-3, 0, 0))
        pr.add_spec('OI', num=3)
        pr.critical_density(depend='T')
        #print(pr.calc_cooling(n=np.linspace(-5, 6, 20), T=4, verbose=1))
        print(pr.balance('OI', debug='A'))
        print(pr.balance('OI', debug='C'))
        print(pr.balance('OI', debug='UV'))
        print(pr.predict())
        n_range = np.linspace(1, 10, 30)
        UV_range = np.linspace(0, 4, 30)
        fig, ax = plt.subplots(ncols=3, figsize=(20, 8))

        if 0:
            for i, T in enumerate([100, 1000, 5000]):
                ax[i].set_prop_cycle(None)
                pr.pars['T'].value = np.log10(T)
                n = []
                ax[i].set_prop_cycle(None)
                for ni in n_range:
                    pr.pars['n'].value = ni
                    n.append(pr.predict())
                for k, ni in enumerate(np.transpose(n)[1:3]):
                    ax[i].plot(n_range, ni, ls='-', label=f'{k+1} level')

                ax[i].set_title(f"T={T}K")

                ax[i].set_xlim([1, 10])
                ax[i].set_ylim([-5, 0])
                ax[i].set_xlabel('log n')
                ax[i].legend(frameon=False, fontsize=14)
                if i == 0:
                    ax[i].set_ylabel(r'log(n$_i$/n$_0$)')
        if 0:
            for i, T in enumerate([150, 1000, 5000]):
                ax[i].set_prop_cycle(None)
                pr.pars['T'].value = np.log10(T)
                n = []
                ax[i].set_prop_cycle(None)
                for uv in UV_range:
                    pr.pars['rad'].value = uv
                    n.append(pr.predict())
                for k, ni in enumerate(np.transpose(n)[1:3]):
                    ax[i].plot(UV_range, ni, ls='-', label=f'{k+1} level')

                ax[i].set_title(f"T={T}K")

                ax[i].set_xlim([0, 4])
                ax[i].set_ylim([-5, 0])
                ax[i].set_xlabel('log UV')
                ax[i].legend(frameon=False, fontsize=14)
                if i == 0:
                    ax[i].set_ylabel(r'log(n$_i$/n$_0$)')
        plt.tight_layout()
        #plt.savefig('C:/users/serj/desktop/OI.png')
        plt.show()

    # >>> H2 calculations:
    if 0:
        pr = pyratio()
        pr.add_spec('H2', num=7)
        pr.set_pars(['T', 'n', 'f'])
        pr.pars['n'].value = 2
        pr.pars['T'].value = np.log10(100)
        pr.pars['f'].value = 0
        print(pr.predict(), np.log10(9), np.log10(5))

    # >>> HD calculations:
    if 0:
        pr = pyratio()
        pr.add_spec('HD', num=6)
        pr.set_pars(['T', 'n', 'f'])
        pr.pars['n'].value = 4
        pr.pars['T'].value = np.log10(100)
        pr.pars['f'].value = 0
        print(pr.predict())
        if 1:
            n, T = np.linspace(1, 4, 10), np.linspace(1, 2.5, 10)
            X, Y = np.meshgrid(n, T)
            z = np.zeros_like(X)
            for i, ni in enumerate(n):
                for k, Tk in enumerate(T):
                    pr.pars['n'].value, pr.pars['T'].value = ni, Tk
                    pop = pr.predict()
                    z[i, k] = np.log10(np.sum(np.power(10, pop[:2]))) - np.log10(np.sum(np.power(10, pop)))
                    #z[i, k] = np.sum(np.power(10, pop[:2])) / np.sum(np.power(10, pop))
                    print(pop, z[i, k])
            cs = plt.contourf(X, Y, z, levels=100)
            plt.contour(X, Y, z, levels=[-1.5, -1.0, -0.5, 0.0], linestyles='--', colors='k')
            plt.colorbar(cs)
            plt.show()

    # >>> CI calculations:
    if 0:
        pr = pyratio(z=2.6)
        pr.add_spec('CI', num=3)
        pr.set_pars(['T', 'n', 'f', 'rad'])
        pr.pars['n'].value = 4
        pr.pars['T'].value = np.log10(100)
        pr.pars['f'].value = 0
        print(pr.predict())
        if 1:
            n, T = np.linspace(1, 4, 10), np.linspace(1, 2.5, 10)
            X, Y = np.meshgrid(n, T)
            z = np.zeros_like(X)
            for i, ni in enumerate(n):
                for k, Tk in enumerate(T):
                    pr.pars['n'].value, pr.pars['T'].value = ni, Tk
                    pop = pr.predict()
                    z[i, k] = pop[1] - np.log10(np.sum(np.power(10, pop)))
                    print(pop, z[i, k])
            cs = plt.contourf(X, Y, z, levels=100)
            plt.contour(X, Y, z, levels=[-1.5, -1.0, -0.5, 0.0], linestyles='--', colors='k')
            plt.colorbar(cs)
            plt.show()