# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 19:20:44 2016

@author: Serj
"""
import astropy.convolution as conv
from astropy import constants as const
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.special import wofz

#import sys
#sys.path.append('D:/science/python/')
#from spectro.sviewer.utils import timer


#==============================================================================
# Voigt Real and Imaginery fucntion and partial derivatives
#==============================================================================

class Voigt():
    def __init__(self, n):
        self.h = np.zeros((n,2))
        self.k = np.zeros((n,2))
        
    def set(self, a, x, n):
        z = wofz(x + 1j*a)
        if n > -1:
            self.H = z.real
            self.K = z.imag
        if n > 0:
            self.H1a = 2*(z.real*a + z.imag*x - 1/np.sqrt(np.pi))
            self.K1a = 2*(z.imag*a - z.real*x)
            self.H1x = self.K1a
            self.K1x = -self.H1a
        if n > 1:
            self.H2a = 4*(z.real*(a**2-x**2+.5) + 2*z.imag*a*x - a/np.sqrt(np.pi))
            self.K2a = 4*(z.imag*(a**2-x**2+.5) - 2*z.real*a*x + x/np.sqrt(np.pi))
            self.H2x = -self.H2a
            self.K2x = -self.K2a
        if n > 2:
            self.H3a = 8*(z.real*(a**3-3*a*x**2+1.5*a) - z.imag*(x**3-3*a**2*x-1.5*x) + (x**2-a**2-1)/np.sqrt(np.pi))
            self.K3a = 8*(z.imag*(a**3-3*a*x**2+1.5*a) + z.real*(x**3-3*a**2*x-1.5*x) + 2*x*a/np.sqrt(np.pi))
            self.H3x = -self.K3a
            self.K3x = self.H3a
 
def voigt(a, x, calc='spec'):
    """
    Returns voigt function
    
    parameters:
        - a       : a parameter
        - x       : x parameter
        - calc    : type of calculation
    
    return:
        voigt     : voigt function at x positions
    """
    if calc == 'spec':
        v = Voigt(0)
        v.set(a, x, 0)
        return v.H
        
#==============================================================================
# 
#==============================================================================

class tau:
    """
    class for optical depth Voigt profile calculation
    Line is specified by:
        - l         :  transition rest wavelength
        - f         :  oscillator strength
        - g         :  natural linewidth
        - b         :  doppler parameter (thermal and turbulent)
        - logN      :  log10 (column density)

    Notes:
        default settings - lyman alpha line of LLS system

    """
    def __init__(self, line=None, logN=19, b=5.0, l=1215.6701, f=0.4164, g=6.265e8, z=0.0, resolution=50000):
        items = ['logN', 'b', 'l', 'f', 'g', 'z']
        if line is None:
            d = locals()
            for k in items:
                setattr(self, k, d[k])
        else:
            for k in items:
                setattr(self, k, getattr(line, k))

        self.a = self.g / 4 / np.pi / self.b / 1e5 * self.l * 1e-8  # dimensionless
        self.resolution = resolution
        self.calctau0()

    def calctau0(self, A=None, gu=None, gl=None):
        """ Returns the optical depth of transition at the center of line.
        parameters:
            - A        : float
                            Einstein parameter, if given, then f is calculated
            - gu, gl   : int
                            statistical weights of upper and lower levels
        """
        e2_me_c = const.e.gauss.value ** 2 / const.m_e.cgs.value / const.c.cgs.value
        if A is not None:
            self.f = (self.l * 1e-8) ** 2 / 8 / np.pi ** 2 / e2_me_c * A * gu / gl

        self.tau0 = np.sqrt(np.pi) * e2_me_c * (self.l * 1e-8) * self.f * 10 ** self.logN / (self.b * 1e5)
        return self.tau0


    def calctau(self, x=None, vel=False, debug=False, verbose=False, convolve=None, tlim=0.01):
        """ Returns the optical depth (Voigt profile) for a transition.

        parameters:
            - x        : array shape (N)
                            velocity or wavelenght grid, specified by vel
            - vel      : boolean
                            if True - x is the velocity grid, else wavelenght
            - convolve : float
                            if present, specify the width of instrument function

        returns:
            - tau    : The optical depth as a function of vel.

        """

        # note units are cgs
        c = const.c.cgs.value
        self.calctau0()

        if verbose:
            print('calculate optical depth for line:')
            print('lambda = ', self.l)
            print('f = ', self.f)
            print('g = ', self.g)
            print('z = ', self.z)
            print('a = ', self.a)
            print('tau_0 = ', self.tau0)

        if x is None:
            x = self.getgrid(vel=vel, tlim=tlim)

        if vel:
            u = x / self.b  # dimensionless
        else:
            u = (x / (1 + self.z) / self.l - 1) * c / (self.b * 1e5)

        xlim = self.xrange(tlim=tlim) * const.c.to('km/s').value

        mask = np.logical_and(u > -xlim, u < xlim)

        tau = np.zeros_like(x)
        tau[mask] = self.tau0 * voigt(self.a, u[mask])  # dimensionless

        self.x = x
        self.tau = tau

        return tau

    def x_instr(self):
        """
        return additional characteristic offset for calculation convolution with instrument function in dimensionless
        """
        if self.resolution not in [None, 0]:
            return 1.0 / self.resolution / 1.66511
        else:
            return 0

    def voigt_range(self, tlim=0.001, debug=False):
        """
        Returns an estimate of the offset from line center at specified optical depth level for Voigt profile.

        parameters:
            - tlim      : optical depth level

        returns:
            velocity offset in km/s
        """
        # print(logN, f, g, l, tlim)
        if self.tau0 < tlim:
            self.dx = 0
        else:
            a = [[-2, -3, -4], [2.67, 3.12, 3.51]]
            inter = interp1d(a[0], a[1], bounds_error=False, fill_value='extrapolate')
            x_0 = inter(np.log10(self.a))
            if debug:
                print('x0:', x_0)
                print(np.sqrt(-np.log(tlim / self.tau0)), np.sqrt(self.tau0 / tlim * self.a / np.sqrt(np.pi)))

            self.dx = np.max([np.sqrt(-np.log(tlim / self.tau0)), np.sqrt(self.tau0 / tlim * self.a / np.sqrt(np.pi))])
            if self.dx > x_0 / 1.2 and self.dx < x_0 * 1.2:
                self.dx *= 1.2

        return self.dx

    def xrange(self, tlim=0.001, instr=3):
        """
        calculate range of absorption line in dimiensionless units
        """
        return self.voigt_range(tlim=tlim) * self.b / const.c.to('km/s').value + instr * self.x_instr()

    def getrange(self, instr=3, tlim=0.001, vel=False):
        """
        calculate range of absorption line in wavelengths
        """
        dx = self.xrange(tlim=tlim, instr=instr)

        if vel:
            self.range =  [ -dx * const.c.to('km/s').value, dx * const.c.to('km/s').value]
        else:
            self.range = [ self.l * (1 - dx) * (1 + self.z), self.l * (1 + dx) * (1 + self.z)]

        return self.range

    def delta(self, vel=False, num=5):
        if vel:
            delt = self.b / num
        else:
            delt = self.l * self.b / const.c.to('km/s').value / num

        if self.resolution not in [None, 0]:
            delt *= np.min([self.x_instr() / self.b * const.c.to('km/s').value, 1])

        return delt

    def getgrid(self, x=None, num=None, ran=None, vel=False, tlim=0.001):
        """
        create grid for calculation of line profiles in wavelengths space
        """
        if x is not None and len(x) > 1:
            x1 = x[:]
            if num is not None:
                d = np.diff(x) / (num + 1)
                for k in range(num):
                    x1 = np.insert(l, (np.arange(len(x)-1)+1)*(k+1), x[:-1]+d*(k+1))
                self.grid = x1
        else:
            if ran is None:
                ran = self.getrange(vel=vel, tlim=tlim)
            self.grid = np.linspace(ran[0], ran[1], 2 * int((ran[1] - ran[0]) / 2 / self.delta(vel=vel)) + 1)
            return self.grid

    def grid_spec(self, x=None, nb=None):
        """
        another version of grid for line profiles
        """
        r = self.getrange()
        mask = np.logical_and(x > r[0], x < r[1])
        inds = np.where(mask == True)
        mask[inds[0]-1], mask[inds[-1]+1] = True, True
        if sum(mask) < 2:
            ind = np.argmin(np.abs(x-(r[0]+r[1])/2))
            mask[ind], mask[ind+1], mask[ind-1] = True, True, True
        if nb is None:
            num = np.round(np.diff(x[mask])/self.delta()) + 1
        else:
            num = np.ones_like(x[mask][:-1], dtype=int) * nb
        n = np.zeros_like(x)
        n[mask] = np.append(num, num[-1])
        return n

def convolveflux(l, f, res, vel=False, kind='astropy', verbose=False, debug=False):
    """
    Convolve flux with instrument function. 
    Data can be unevenly spaced. 
    There are several types of convolution
    
    parameters:
        - l         : float array, shape(N)
                        wavelength array (or velocity in km/s)
        - f         : float array, shape(N)
                        flux
        - res       : float
                        resolution of instrument function
        - vel       : boolean
                        if true velocity format for l, otherwise wavelength
        - kind      : str
                        specified type of convolution:
                           'astropy'     : convolution from astropy package (for evenly spaced data)
                           'gauss'       : convolution with fixed gaussian function
                           'resolution'  : convolution with fixed resolution

    returns:
        - fc        : float array, shape(N)
                        convolved flux
    """
    
    if kind == 'astropy':     
        if vel:
            pixels = const.c.cgs.value / res / 1e5 / ((l[-1]-l[0])/(len(l)-1))
        else:
            pixels = (l[-1]+l[0]) / 2 / res / ((l[-1]-l[0])/(len(l)-1))
            
        #print(pixels)
        
        gauss_kernel = conv.Gaussian1DKernel(pixels/3)
        
        fc = conv.convolve(f, gauss_kernel, boundary='extend') 
    
    if kind == 'gauss':

        # >>> renormalize res to satisfy dispersion of Gauss set to be <l/R>
        R = res * 2 * np.sqrt(2*np.log(2))

        fc = np.zeros_like(f)

        # expand the regions of wavelength array and flux
        delta = 4
        addl = np.linspace(delta, 0, 21)
        lt = np.concatenate((l[0] * (1 - addl/R), l, l[-1] * (1 + addl/R)), axis=0)
        ft = np.concatenate((f[0]*np.ones_like(addl), f, np.ones_like(addl)*f[-1]), axis=0)
        #print(len(lt), len(ft), lt[0], lt[-1], lt[0]*(4/R))

        def gauss(x, s):
            return 1/np.sqrt(2*np.pi)/s * np.exp(-.5*(x/s)**2)

        for i in range(len(l)):
            mask = (lt < l[i]*(1 + delta/R)) & (lt > l[i]* (1 - delta/R))
            fl = ft[mask]
            #print(np.sum(f1), sum(mask))
            if (np.sum(fl) < 0.998 * sum(mask)):
                x = lt[mask]/l[i]
                #print(1-l1/l[i], gauss(1-l1/l[i], 1.0/R))
                #input()
                fc[i] = simps(fl*gauss(1-x, 1.0/R), x)
            else:
                fc[i] = 1
        #print(fc)

    if kind == 'direct':

        return convolve_res2(l, f, res)

    return fc


# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
@jit
def gauss(x, s):
    return 1 / np.sqrt(2 * np.pi) / s * np.exp(-.5 * (x / s) ** 2)

@jit
def errf(x):
    a = [0.3480242, -0.0958798, 0.7478556]
    t = 1 / (1 + 0.47047 * np.abs(x))
    return np.sign(x) * (1 - t * (a[0] + t * (a[1] + t * a[2])) * np.exp(-x**2))

@jit
def errf_v2(x):
    a = [-1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277]
    t = 1 / (1 + 0.5 * np.abs(x))
    tau = t * np.exp(-x ** 2 + a[0] + t * (a[1] + t * (a[2] + t * (a[3] + t * (a[4] + t * (a[5] + t * (a[6] + t * (a[7] + t * (a[8] + t * a[9])))))))))
    if x >= 0:
        return 1 - tau
    else:
        return tau - 1

@jit
def convolve_res(l, f, R):
    """
    Convolve flux with instrument function specified by resolution R
    Data can be unevenly spaced. 

    parameters:
        - l         : float array, shape(N)
                        wavelength array (or velocity in km/s)
        - f         : float array, shape(N)
                        flux
        - R         : float
                        resolution of the instrument function. Assumed to be constant with wavelength. 
                        i.e. the width of the instrument function is linearly dependent on wavelenth.  

    returns:
        - fc        : float array, shape(N)
                        convolved flux
    """
    #sig = 127301 / R
    delta = 3.0

    n = len(l)
    fc = np.zeros_like(f)

    d = [l[1] - l[0]]
    for i in range(1, n-1):
        d.append((l[i + 1] - l[i - 1]) / 2)
    d.append(l[-1]-l[-2])

    il = 0
    for i, x in enumerate(l):
        sig = x / R / 2.355
        k = il
        while l[k] < x - delta * sig:
            k += 1
        il = k
        s = f[k] * (1 - errf(np.abs(l[k] - x - d[0]/2) / np.sqrt(2) / sig)) / 2
        while k < n and l[k] < x + delta * sig:
            #s += f[k] * 1 / np.sqrt(2 * np.pi) / sig * np.exp(-.5 * ((l[k] - x) / sig) ** 2) * d[k]
            s += f[k] * gauss(l[k] - x, sig) * d[k]
            k += 1

        k -= 1
        s += f[k] * (1 - errf(np.abs(l[k] - x + d[k]/2) / np.sqrt(2) / sig)) / 2
        fc[i] = s

    return fc

@jit
def convolve_res2(l, f, R):
    """
    Convolve flux with instrument function specified by resolution R
    Data can be unevenly spaced. 

    parameters:
        - l         : float array, shape(N)
                        wavelength array (or velocity in km/s)
        - f         : float array, shape(N)
                        flux
        - R         : float
                        resolution of the instrument function. Assumed to be constant with wavelength. 
                        i.e. the width of the instrument function is linearly dependent on wavelenth.  

    returns:
        - fc        : float array, shape(N)
                        convolved flux
    """
    #sig = 127301 / R
    delta = 3.0

    n = len(l)
    fc = np.zeros_like(f)

    il = 0
    for i, x in enumerate(l):
        sig = x / R / 2.355
        k = il
        while l[k] < x - delta * sig:
            k += 1
        il = k
        s = f[il] * (1 - errf_v2((x - l[il]) / np.sqrt(2) / sig)) / 2
        #ie = il + 30
        while k < n-1 and l[k+1] < x + delta * sig:
            #s += f[k] * 1 / np.sqrt(2 * np.pi) / sig * np.exp(-.5 * ((l[k] - x) / sig) ** 2) * d[k]
            s += (f[k+1] * gauss(l[k+1] - x, sig) + f[k] * gauss(l[k] - x, sig)) / 2 * (l[k+1] - l[k])
            #print(i, k , gauss(l[k] - x, sig))
            k += 1
        #input()
        s += f[k] * (1 - errf_v2(np.abs(l[k] - x) / np.sqrt(2) / sig)) / 2
        fc[i] = s

    return fc

#@jit
def makegrid(x, n):
    ind = np.argwhere(n)
    l = x[ind]
    for k, i in zip(range(len(ind)-2, -1, -1), reversed(ind[:-1])):
        d = (x[i+1] - x[i]) / (int(n[i]) + 1)
        l = np.insert(l, np.ones(int(n[i][0]), dtype=int) * (k+1), x[i] + d * (np.arange(int(n[i])) + 1))

    return l

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
        mask = x < (z_em-0.05+1)*1215.67
        corr[mask] = inter(x[mask])

    if kind == 'lines':
        # not implemented yet!!!
        pass

    return corr

def add_LyaCutoff(x, z=0, factor=1, kind='trans'):
    """
    add lyman cutoff 
    parameters: 
        - x         : the wavelength grid
        - z         : redshift of the cutoff
         
    return:
        - corr      : correction array at each x.
    """
    corr = np.ones_like(x)
    corr[x < (1+z-0.2)*912] = 0
    return corr

def add_ext(x, z_ext=0, Av=0, kind='SMC'):
    """
    calculate extinction at given redshift
    parameters: 
        - x         : the wavelength grid
        - z_ext     : redshift of extinction applied
        - Av        : Av 
        - kind      : type of extinction curve, can be either 'SMC', 'LMC'
         
    return:
        - corr      : correction array at each x.
    """
    if Av > 0:
        et = {'SMC': 2, 'LMC': 6}
        data = np.genfromtxt('data/extinction.dat', skip_header=3, usecols=[0, et[kind]], unpack=True)
        inter = interp1d(data[0]*1e4, data[1], fill_value='extrapolate')

        return np.exp(- 0.92 * Av * inter(x / (1+z_ext)))
    else:
        return np.ones_like(x)

def add_ext_bump(x, z_ext=0, Av=0, Av_bump=0):
    print(Av, Av_bump)
    if Av > 0:
        x_0 = 4.593
        g = 0.85
        Rv = 4.1
        c1 = -2.62
        c2 = 2.24
        c3 = Av_bump/Av * (2 * Rv * g) / np.pi
        y = np.power(x / (1+z_ext), -2) * 1e8
        print(c3, y)
        return np.exp(- 0.92 * Av * ((c1 + c2 * np.power(x, -1) * 1e4 + c3 * y / (np.power((y - x_0**2), 2) + y * g**2)) / Rv + 1))

    else:
        return np.ones_like(x)


def fisherbN(N, b, lines, ston=1, cgs=0, convolve=1, resolution=50000, z=2.67, tlim=0.99,
             verbose=False, plots=False):
    """
    calculate the Fisher matrix for a given b and logN, the parameters of line profile

    input:
        - N           :  column density in log10[, cm^-2] units
        - b           :  b parameter in km/s
        - lines       :  list of lines
        - ston        :  Signal to Noise ratio, inverse of dispersion

    options:
        - cgs         :  if 0 then derivative for N in cm^-2 and b in cm/s
        - convolve    :  if 1 convolve data else not convolve
        - resolution  :  resolution of the spectrograph (assuming 3 pixels in FWHM)
        - z           :  redshift of line
        - tlim        :  limiting flux for caclulation (specify line range)

    return:
        - db          :  uncertainty for b parameter in km/s
        - dN          :  uncertainty for column density in log10[, cm^-2]
        - F           :  fisher matrix
    """

    V = Voigt(3)

    F_con_extr = [0] * 6
    F = np.zeros((2, 2))

    if plots:
        fig, ax = plt.subplots(6, 1, figsize=(8, 20))

    for line in lines:
        line.logN, line.b = N, b
        l = tau(line=line, resolution=resolution)

        x = l.getgrid(vel=True) / b

        l.calctau0()

        if verbose:
            print('tau_0=', l.tau0)

        V.set(l.a, x, 3)
        l.calctau(l.grid, vel=True)
        F_unc, F_con = np.zeros((6, len(x))), np.zeros((6, len(x)))
        F_unc[0] = np.ones_like(x)
        if not cgs:
            F_unc[1] = l.tau * (l.tau - 1) * np.log(10) ** 2
            F_unc[2] = l.tau0 * V.H2x / 2 / b * (l.tau - 1) * np.log(10)
            F_unc[3] = l.tau0 / 2 / b / b * (l.tau0 / 2 * V.H2x ** 2 - l.a * V.H3a - x * V.H3x + 2 * V.H2x)
            F_unc[4] = l.tau * np.log(10)
            F_unc[5] = l.tau0 / 2 / b * V.H2x
        else:
            F_unc[1] = (l.tau / np.power(10.0, N)) ** 2
            F_unc[2] = l.tau0 * V.H2x / 2 / np.power(10.0, N) / b / 1e5 * (l.tau - 1)
            F_unc[3] = l.tau0 / 2 / b / b / 1e10 * (l.tau0 / 2 * V.H2x ** 2 - l.a * V.H3a - x * V.H3x + 2 * V.H2x)
            F_unc[4] = l.tau / 10 ** N
            F_unc[5] = l.tau0 / 2 / b / 1e5 * V.H2x

        F_unc *= np.exp(-l.tau0 * V.H)

        if convolve:
            for i in range(6):
                F_con[i] = convolveflux(l.grid, F_unc[i], resolution, vel=True)

        colors = ['k', 'r', 'b', 'g', 'r', 'r']
        if plots:
            for i in range(6):
                ax[i].plot(x, F_unc[i, :], '--', color=colors[i])
                ax[i].plot(x, F_con[i, :], '-', color=colors[i])

        for i in range(6):
            F_con_extr[i] = interp1d(x, F_con[i, :])

        mask = F_con[0] < 0.99
        x_lim = x[np.where(np.diff(mask) > 0)[0]][1]
        x1 = np.linspace(-x_lim, x_lim, 2 * int( x_lim * resolution * 3 * (b / const.c.to('km/s').value)) + 1)
        if verbose:
            print('number of points = ', len(x1))

        if plots:
            for i in range(6):
                ax[i].plot(x1, F_con_extr[i](x1), 'o')

        F[0, 0] = np.sum(F_con_extr[4](x1) ** 2)
        F[1, 0] = np.sum(F_con_extr[4](x1) * F_con_extr[5](x1))
        F[0, 1] = F[1, 0]
        F[1, 1] = np.sum(F_con_extr[5](x1) ** 2)
        F *= 2

        cov = np.abs(np.linalg.inv(F))

        if verbose:
            print('Fisher matrix:', F)
            print('Covariance matrix', cov)

        if not cgs:
            dN = np.sqrt(np.abs(cov))[0, 0] / ston
            db = np.sqrt(np.abs(cov))[1, 1] / ston
        else:
            print(np.sqrt(np.abs(cov))[0, 0] / ston)
            dN = N - np.log10(np.power(10, N) - np.sqrt(np.abs(cov))[0, 0] / ston)
            db = np.sqrt(np.abs(cov))[1, 1] / 1e5 / ston

        return dN, db, F, min(F_con[0, :])

if __name__ == '__main__':

    import sys
    sys.path.append('D:/science/python')
    from spectro.atomic import H2list
    import matplotlib.pyplot as plt
    
    H2 = H2list.Malec(0)
    
    l = np.linspace(1000, 1120, 40000)
    tau = np.zeros_like(l)    
    
    for line in H2:
        print(line)
        tau += calctau(l, line.l, line.f, line.g, 19, 3, z=0, vel=False)
    
    I = convolveflux(l, np.exp(-tau), res=1800)
    
    fig, ax = plt.subplots()
    ax.plot(l, I)
    ax.set_ylim([-0.1, 1.2])
    
    
    
        