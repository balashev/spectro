# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:38:59 2016

@author: Serj
"""
import astropy.constants as ac
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
from astropy.io import fits
from astropy.modeling.models import Moffat1D
#from ccdproc import cosmicray_lacosmic
import itertools
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
import re
from scipy.interpolate import interp1d, interp2d, splrep, splev
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit, least_squares
from scipy.signal import savgol_filter, lombscargle, medfilt
from scipy.stats import gaussian_kde

from ..profiles import tau, convolveflux, makegrid, add_ext
from .external import sg_smooth as sg
from .utils import Timer, MaskableList, moffat_func, smooth

class Speclist(list):
    def __init__(self, parent):
        self.parent = parent
        self.ind = None
            
    def draw(self):
        self[self.ind].view = self.parent.specview
        for i, s in enumerate(self):
            if i != self.ind:
                s.initGUI()
        self[self.ind].initGUI()
        self.parent.vb.enableAutoRange()
        
    def redraw(self, ind=None):
        """
        procedure for redraw spectra: 
            if <ind> is provided then redraw only <current> and <ind> spectra.
        """
        if len(self) > 0:
            if ind == None:
                for i, s in enumerate(self):
                    if i != self.ind:
                        s.redraw()
                self[self.ind].redraw()
            else:
                saved_ind = self.ind
                self.ind = ind
                self[saved_ind].redraw()
                self[self.ind].redraw()
            self.parent.plot.specname.setText(self[self.ind].filename)

    def remove(self, ind=None):
        if ind is None:
            ind = self.ind
        if ind > -1:
            if len(self) > 0:
                self[ind].remove()
                del self[ind]
                if ind == len(self):
                    ind -= 1
                if len(self) == 0:
                    self.ind = None
                self.ind = ind
                self.redraw()
                try:
                    self.parent.exp.update()
                except:
                    pass
            else:
                self.parent.sendMessage('There is nothing to remove')

    def rearrange(self, inds):
        if len(inds) == len(self):
            self.ind = inds[self.ind]
            self[:] = [self[i] for i in inds]
            #print([s.filename for s in self])

    def find(self, name):
        """
        Find exposure index by name:
        parameters:
            - name            :  exact name of the exposure

        return: ind
             - ind            :  index of the exposure
        """
        try:
            ind = [s.filename for s in self].index(name)
            return ind
        except:
            NameError('exposure not found')


    def setSpec(self, ind=0, new=False):
        """
        set Spectrum as active (and disactivate other spectra)
        NOTE!!!!: sviewer.s.ind is changed in specClicked() function
        parameter:
            - ind       : index of spectrum to activate 
        """
        ind = ind
        if ind > len(self)-1:
            ind = 0
        if ind < 0:
            ind = len(self) - 1
        #debug(ind)
        #self[ind].specClicked()
        self.parent.s.redraw(ind)
        self.parent.plot.e_status = 2
        try:
            self.parent.exp.selectRow(self.ind)
        except:
            pass
        for name, status in self.parent.filters_status.items():
            if status:
                m = max([max(self[self.ind].spec.y())])
                for f in self.parent.filters[name]:
                    f.update(m)

    def normalize(self, action='normalize'):
        for i, s in enumerate(self):
            s.normalize(action=action)
        self.redraw()

    def prepareFit(self, ind=-1, exp_ind=-1, all=True):
        self.parent.fit.update()

        if self.parent.fitType == 'julia':
            #self.parent.reload_julia()
            self.parent.julia_pars = self.parent.julia.make_pars(self.parent.fit.list(), tieds=self.parent.fit.tieds)
            self.parent.julia_add = self.parent.julia.prepare_add(self.parent.fit, self.parent.julia_pars)

        for i, s in enumerate(self):
            if exp_ind in [-1, i]:
                s.findFitLines(ind, all=all, debug=False)

        if self.parent.fitType == 'julia':
            self.parent.julia_spec = self.parent.julia.prepare(self, self.parent.julia_pars, self.parent.julia_add)

    def calcFit(self, ind=-1, exp_ind=-1, recalc=False, redraw=True, timer=False):

        if timer:
            t = Timer()

        for i, s in enumerate(self):
            if exp_ind in [-1, i]:
                if hasattr(s, 'fit_lines') and len(s.fit_lines) > 0:
                    if self.parent.fitType == 'regular':
                        s.calcFit(ind=ind, recalc=recalc, redraw=redraw, tau_limit=self.parent.tau_limit, timer=timer)

                    elif self.parent.fitType == 'fft':
                        s.calcFit_fft(ind=ind, recalc=recalc, redraw=redraw, tau_limit=self.parent.tau_limit, timer=timer)

                    elif self.parent.fitType == 'fast':
                        s.calcFit_fast(ind=ind, recalc=recalc, redraw=redraw, num_between=self.parent.num_between, tau_limit=self.parent.tau_limit, timer=timer)

                    elif self.parent.fitType == 'julia':
                        s.calcFit_julia(comp=ind, recalc=recalc, redraw=redraw, tau_limit=self.parent.tau_limit, timer=timer)

                else:
                    s.set_fit(x=self[self.ind].spec.raw.x[self[self.ind].cont_mask], y=np.ones_like(self[self.ind].spec.raw.x[self[self.ind].cont_mask]))
                    s.set_gfit()
        if timer:
            t.time('fit ' + self.parent.fitType)

    def calcFitComps(self, ind=-1, exp_ind=-1, recalc=False):
        self.refreshFitComps()
        for i, s in enumerate(self):
            if exp_ind in [i, -1] and len(s.fit_lines) > 0:
                for sys in self.parent.fit.sys:
                    #print('calcFitComps', sys.ind)
                    if self.parent.fitType == 'regular':
                        s.calcFit(ind=sys.ind, x=s.fit.x(), recalc=recalc, tau_limit=self.parent.tau_limit)

                    elif self.parent.fitType == 'fft':
                        s.calcFit_fft(ind=sys.ind, recalc=recalc, tau_limit=self.parent.tau_limit)

                    elif self.parent.fitType == 'fast':
                        s.calcFit_fast(ind=sys.ind, recalc=recalc, num_between=self.parent.num_between, tau_limit=self.parent.tau_limit)

                    elif self.parent.fitType == 'julia':
                        s.calcFit_julia(comp=sys.ind, recalc=recalc, tau_limit=self.parent.tau_limit, timer=False)

    def reCalcFit(self, ind=-1, exp_ind=-1):
        #self.prepareFit(ind=-1)
        #self.calcFit(ind=-1)
        self.prepareFit(ind=ind, exp_ind=exp_ind)
        self.calcFit(ind=ind, exp_ind=exp_ind)
        self.chi2(exp_ind=exp_ind)

    def refreshFitComps(self):
        for s in self:
            s.construct_fit_comps()

    def redrawFitComps(self):
        if self.ind is not None:
            self[self.ind].redrawFitComps()

    def chi2(self, exp_ind=-1):
        chi2 = np.sum(np.power(self.chi(exp_ind=exp_ind), 2))
        n = 0
        for i, s in enumerate(self):
            if exp_ind in [-1, i] and hasattr(s, 'fit_mask') and s.fit_mask is not None:
                n += np.sum(s.fit_mask.x())

        self.parent.chiSquare.setText('  chi2 / dof = {0:.2f} / {1:d}'.format(chi2, int(n - len(self.parent.fit.list_fit()))))
        return chi2

    def chi(self, exp_ind=-1):
        chi = np.asarray([])
        for i, s in enumerate(self):
            if exp_ind in [-1, i]:
                chi = np.append(chi, s.chi())
        return chi

    def selectCosmics(self):
        for i, s in enumerate(self):
            if i != self.ind:
                s.selectCosmics()
        self[self.ind].selectCosmics()
        
    def calcSmooth(self):
        for i, s in enumerate(self):
            if i != self.ind:
                s.smooth()
        self[self.ind].smooth()
        
    def coscaleExposures_old(self):
        for i, s in enumerate(self):
            if s.sm.n() == 0:
                s.smooth()
        if 1:
            coef = np.nansum(self[self.ind].sm.inter(s.spec.y()))
            for i, s in enumerate(self):
                k = self[self.ind].sm.inter(s.spec.x()) / s.sm.inter(s.spec.x()) #/ coef
                s.spec.raw.y *= k
                s.spec.raw.err *= k
                s.spec.raw.interpolate()
                s.sm.raw.y *= k
                s.sm.raw.interpolate()

        self.redraw()

    def coscaleExposures(self):
        for i, s in enumerate(self):
            if s.sm.n() == 0:
                s.smooth()

        self.coscaleExps(full=True, ind=self.ind)
        self.coscaleExps(full=False)

    def coscaleExps(self, full=True, ind=None):
        for i, si in enumerate(self):
            if ind is None or i == ind:
                mi = np.logical_and(si.spec.y() != 0, si.spec.err() != 0)
                xmin, xmax = np.min(si.spec.x()[mi]), np.max(si.spec.x()[mi])
                for k, sk in enumerate(self):
                    if k != i:
                        c = np.ones_like(sk.spec.y(), dtype=float)
                        mk = np.logical_and(sk.spec.y() != 0, sk.spec.err() != 0)
                        # >>> intersection
                        mask = np.logical_and(mk, np.logical_and(sk.spec.x() > xmin, sk.spec.x() < xmax))
                        if np.sum(mask) > 0:
                            c[mask] = si.sm.inter(sk.spec.x()[mask]) / sk.sm.inter(sk.spec.x()[mask])
                        if full:
                            # >>> left
                            mask = np.logical_and(mk, sk.spec.x() < xmin)
                            if np.sum(mask) > 0:
                                c[mask] = si.sm.inter(xmin) / sk.sm.inter(xmin)
                            # >>> right
                            mask = np.logical_and(mk, sk.spec.x() > xmax)
                            if np.sum(mask) > 0:
                                c[mask] = si.sm.inter(xmax) / sk.sm.inter(xmax)
                        if 1:
                            sk.spec.raw.y *= c
                            sk.spec.raw.err *= c
                            sk.spec.raw.interpolate()
                            sk.sm.raw.y *= c[mk]
                            sk.sm.raw.interpolate(fill_value=(sk.sm.raw.y[0], sk.sm.raw.y[-1]))

        self.redraw()

    def minmax(self):
        minv, maxv = self[0].spec.x()[0], self[0].spec.x()[0]
        for s in self:
            minv = np.min([minv, s.spec.x()[0]])
            maxv = np.max([maxv, s.spec.x()[-1]])
        return minv, maxv

    def apply_regions(self):
        for s in self:
            s.apply_regions()


class gline():
    """
    class for working with lines inside Spectrum plotting
    """

    def __init__(self, x=[], y=[], err=[], mask=[]):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.err = np.asarray(err)
        self.mask = np.asarray(mask)
        # self.x_s, self.y_s, self.err_s = self.x, self.y, self.err
        self.n = self.x.shape[0]
        if self.x.shape != self.y.shape:
            raise IndexError("Dimensions of x and y data are not the same")

    def initial(self, save=True):
        if save:
            self.saved = [self.x[:], self.y[:], self.err[:]]
            print('saved', self.saved)
        else:
            print(self.saved)
            self.x, self.y, self.err = self.saved[0][:], self.saved[1][:], self.saved[2][:]

    def set_data(self, *args, **kwargs):
        self.delete()
        self.add(**kwargs)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, np.append(getattr(self, k), v))
        # self.apply_region()
        self.sort()
        self.n = len(self.x)

    def delete(self, arg=None, x=None, y=None, err=None):
        if arg is None and x is None and y is None and err is None:
            self.x, self.y, self.err = np.array([]), np.array([]), np.array([])
        else:
            if arg is None:
                if x is not None:
                    arg = np.where(self.x == x)
                if y is not None:
                    arg = np.where(self.y == y)
            self.x = np.delete(self.x, arg)
            self.y = np.delete(self.y, arg)
            if len(self.err) > 0:
                self.err = np.delete(self.err, arg)
        self.n = len(self.x)

    def interpolate(self, err=False, fill_value=np.NaN):
        m = np.logical_and(np.isfinite(self.y), np.isfinite(self.x))
        #print(np.sum(np.logical_not(m)))
        if not err:
            self.interpol = interp1d(self.x[m], self.y[m], bounds_error=False, fill_value=fill_value, assume_sorted=True)
        else:
            self.err_interpol = interp1d(self.x[m], self.err[m], bounds_error=False, fill_value=fill_value, assume_sorted=True)

    def index(self, x):
        return np.searchsorted(self.x, x)

    def f(self, x):
        return self.y[self.index(x)]

    def inter(self, x):
        if not hasattr(self, 'interpol'):
            self.interpolate()
        return self.interpol(x)

    def err_inter(self, x):
        if not hasattr(self, 'err_interpol'):
            self.interpolate(err=True)
        return self.err_interpol(x)

    def sort(self, axis=0):
        try:
            if axis == 0 or axis == 'x':
                args = np.argsort(self.x)
            elif axis == 1 or axis == 'y':
                args = np.argsort(self.y)
        except:
            raise ValueError("Illegal axis argument")
        if len(self.x) > 0:
            self.x = self.x[args]
        if len(self.y) > 0:
            self.y = self.y[args]
        if len(self.err) > 0:
            self.err = self.err[args]
        if len(self.mask) > 0:
            self.mask = self.mask[args]

    def find_nearest(self, x=None, y=None):
        if self.n > 0:
            dist = np.zeros_like(self.x)
            if x is not None:
                dist += np.power(self.x - x, 2)
            if y is not None:
                dist += np.power(self.y - y, 2)
            return np.argmin(dist)

    def __str__(self):
        if self.x is not None:
            return 'gline object: ' + '[{0}..{1}]'.format(self.x[0], self.x[-1])
        else:
            return 'empty gline object'

    def __repr__(self):
        st = 'gline: '
        if self.x is not None:
            st += "[{0}..{1}]".format(self.x[0], self.x[-1])
        else:
            return 'empty gline object'
        if self.x is not None:
            st += ", [{0}..{1}]".format(self.y[0], self.y[-1])
        if len(self.err) > 0:
            st += "[{0}..{1}], ".format(self.err[0], self.err[-1])
        return st

    def clean(self, min=None, max=None):
        mask = np.ones_like(self.x, dtype=bool)
        if min is not None:
            mask = np.logical_and(mask, self.y > min)
        if max is not None:
            mask = np.logical_and(mask, self.y < max)
        self.x, self.y, self.err = self.x[mask], self.y[mask], self.err[mask]
        try:
            self.mask = self.mask[mask]
        except:
            pass
        self.n = len(self.x)

    def apply_region(self, regions=[]):
        mask = np.ones_like(self.x_s, dtype=bool)
        if len(regions) > 0:
            regions = np.sort(regions, axis=0)
            for r in regions:
                mask = np.logical_and(mask, np.logical_or(self.x_s < np.min(r), self.x_s > np.max(r)))
        self.x, self.y, self.err, self.mask = self.x_s[mask], self.y_s[mask], self.err_s[mask], self.mask_s[mask]
        self.n = len(self.x)

        if len(regions) > 0:
            regions = np.sort(regions, axis=0)
            for r in reversed(regions):
                d = abs(r[1] - r[0])
                self.x[self.x > r[1]] -= d

    def convolve(self, resolution=None):
        """
        convolve the line using given resolution
        """
        if resolution is not None:
            m = np.tile(self.x, (len(self.x), 1))
            d = m - np.transpose(m)
            #print(np.exp(-np.power(np.multiply(d, np.transpose(self.x / resolution)), 2) / 2))

    def copy(self):
        return gline(x=self.x[:], y=self.y[:], err=self.err[:])

class plotLineSpectrum(pg.PlotCurveItem):
    """
    class for plotting step spectrum centered at pixels
    slightly modified from PlotCurveItem
    """
    def __init__(self, *args, **kwargs):
        self.parent = kwargs['parent']
        self.view = kwargs['view']
        #self.parent.spec_save = self.parent.spec.raw.copy()
        #print({k: v for k, v in kwargs.items() if k not in ['parent', 'view']})
        super().__init__(*args, **{k: v for k, v in kwargs.items() if k not in ['parent', 'view']})

    def initial(self):
        self.parent.spec.raw.initial(False)
        self.parent.redraw()

    def generatePath(self, xi, yi, path=True):
        if 'step' in self.view:
            ## each value in the x/y arrays generates 2 points.
            x = xi[:, np.newaxis] + [0, 0]
            dx = np.diff(xi) / 2
            dx = np.append(dx, dx[-1])
            x[:, 0] -= dx
            x[:, 1] += dx
            x = x.flatten()
            y = as_strided(yi, shape=[len(yi), 2], strides=[yi.strides[0], 0]).flatten()
        if 'line' in self.view:
            x, y = xi[:], yi[:]

        if path:
            return pg.functions.arrayToQPath(x, y, connect=self.opts['connect'])
        else:
            return x, y

    def returnPathData(self):
        return self.generatePath(self.xData, self.yData, path=False)

    def mouseDragEvent(self, ev):
        if QApplication.keyboardModifiers() in [Qt.ShiftModifier, Qt.ControlModifier]:
            if ev.button() != Qt.RightButton:
                ev.ignore()
                return

            if ev.isStart():
                self.start = self.parent.parent.vb.mapSceneToView(ev.buttonDownPos())
            elif ev.isFinish():
                self.start = None
                return
            else:
                if QApplication.keyboardModifiers() == Qt.ShiftModifier:
                    self.shift(start=self.start, finish=self.parent.parent.vb.mapSceneToView(ev.pos()))
                if QApplication.keyboardModifiers() == Qt.ControlModifier:
                    self.rescale(start=self.start, finish=self.parent.parent.vb.mapSceneToView(ev.pos()))
                self.start = self.parent.parent.vb.mapSceneToView(ev.pos())
                if self.start is None:
                    ev.ignore()
                    return

            ev.accept()

    def shift(self, start, finish):
        if not self.parent.parent.normview:
            self.parent.spec.raw.x += finish.x() - start.x()
            self.parent.spec.raw.y += finish.y() - start.y()
            self.setData(x=self.parent.spec.raw.x, y=self.parent.spec.raw.y)
        self.parent.redraw()

    def rescale(self, start, finish):
        if not self.parent.parent.normview:
            self.parent.spec.raw.x += finish.x() - start.x()
            self.parent.spec.raw.y *= finish.y() / start.y()
            self.parent.spec.raw.err *= finish.y() / start.y()
            self.setData(x=self.parent.spec.raw.x, y=self.parent.spec.raw.y)
        self.parent.redraw()

    #def mouseClickEvent(self, ev):
    #    if QApplication.keyboardModifiers() == Qt.Key_I and ev.button() == Qt.LeftButton:
    #        self.parent.remove()

class fitLineProfile(pg.PlotCurveItem):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.setClickable(True)

    def mouseDragEvent(self, ev):

        if QApplication.keyboardModifiers() == Qt.ShiftModifier:
            if ev.button() != Qt.LeftButton:
                ev.ignore()
                return

            if ev.isStart():
                # We are already one step into the drag.
                # Find the point(s) at the mouse cursor when the button was first
                # pressed:
                pos = self.parent.parent.vb.mapSceneToView(ev.pos())
                self.st_pos = pos.x()

            pos = self.parent.parent.vb.mapSceneToView(ev.pos())
            ev.accept()

class specline():
    def __init__(self, parent):
        self.parent = parent
        self.raw = gline()
        self.norm = gline()

    def current(self):
        attr = 'norm' if self.parent.parent.normview else 'raw'
        return getattr(self, attr)

    def x(self):
        return self.current().x

    def y(self):
        return self.current().y

    def err(self):
        return self.current().err

    def mask(self):
        return self.current().mask

    def f(self, x):
        return self.current().inter(x)

    def n(self):
        return len(self.current().x)

    def index(self, x):
        return self.current().index(x)

    def add(self, x, y=[], err=[]):
        self.raw.add(x=x, y=y, err=err)

    def set(self, x=None, y=None, err=None, mask=None):
        if x is not None:
            self.current().x = x
            self.current().n = len(x)
        if y is not None:
            self.current().y = y
        if err is not None:
            self.current().err = err
        if mask is not None:
            self.current().mask = mask
        #self.current().sort()

    def interpolate(self):
        self.current().interpolate()

    def normalize(self, norm=True, cont_mask=True, inter=False, action='normalize'):
        if cont_mask:
            if norm:
                self.norm.x = self.raw.x[self.parent.cont_mask]
                if len(self.raw.y) > 0:
                    if action == 'normalize':
                        self.norm.y = self.raw.y[self.parent.cont_mask] / self.parent.cont.y
                    elif action == 'subtract':
                        self.norm.y = self.raw.y[self.parent.cont_mask] - self.parent.cont.y
                    elif action == 'aod':
                        self.norm.y = - np.log(self.raw.y[self.parent.cont_mask] / self.parent.cont.y)
                if len(self.raw.err) > 0:
                    if action == 'normalize':
                        self.norm.err = self.raw.err[self.parent.cont_mask] / self.parent.cont.y
                    elif action == 'subtract':
                        self.norm.err = self.raw.err[self.parent.cont_mask]
                    elif action == 'aod':
                        self.norm.err = - np.log(1 + self.raw.err[self.parent.cont_mask] / self.raw.y[self.parent.cont_mask])
                self.norm.n = len(self.norm.x)
            else:
                self.raw.x[self.parent.cont_mask] = self.norm.x
                if len(self.raw.y) > 0:
                    if action == 'normalize':
                        self.raw.y[self.parent.cont_mask] = self.norm.y * self.parent.cont.y
                    elif action == 'subtract':
                        self.raw.y[self.parent.cont_mask] = self.norm.y + self.parent.cont.y
                    elif action ==  'aod':
                        self.raw.y[self.parent.cont_mask] = np.exp(-self.norm.y) * self.parent.cont.y
                if len(self.raw.err) > 0:
                    if action == 'normalize':
                        self.raw.err[self.parent.cont_mask] = self.norm.err * self.parent.cont.y
                    elif action == 'subtract':
                        self.raw.err[self.parent.cont_mask] = self.norm.err
                    elif action == 'aod':
                        self.raw.err[self.parent.cont_mask] = (np.exp(-self.norm.err) - 1) * self.norm.y
        else:
            if (norm and len(self.raw.y) > 0) or (not norm and len(self.norm.y) > 0):
                if not inter:
                    cont = self.parent.cont.y
                else:
                    self.parent.cont.interpolate()
                    if norm:
                        cont = self.parent.cont.inter(self.raw.x)
                    else:
                        cont = self.parent.cont.inter(self.norm.x)
            if norm:
                self.norm.x = self.raw.x[:]
                if len(self.raw.y) > 0:
                    if action == 'normalize':
                        self.norm.y = self.raw.y / cont
                    elif action == 'subtract':
                        self.norm.y = self.raw.y - cont
                    elif action == 'aod':
                        self.norm.y = - np.log(self.raw.y / cont)
                self.norm.n = len(self.norm.x)
            else:
                self.raw.x = self.norm.x[:]
                if len(self.norm.y) > 0:
                    if action == 'normalize':
                        self.raw.y = self.norm.y * cont
                    elif action == 'subtract':
                        self.raw.y = self.norm.y + cont
                    elif action == 'aod':
                        self.raw.y = np.exp(-self.norm.y) * cont

    def inter(self, x):
        return self.current().inter(x)

class fitline():
    def __init__(self, parent):
        self.parent = parent
        self.line = specline(parent)
        self.disp = [specline(parent), specline(parent)]
        self.disp_corr = [specline(parent), specline(parent)]
        self.g_disp = [None]*3

    def interpolate(self):
        self.line.current().interpolate()
        self.disp[0].current().interpolate()
        self.disp[1].current().interpolate()
        self.disp_corr[0].current().interpolate()
        self.disp_corr[1].current().interpolate()

    def normalize(self, *args, **kwargs):
        self.line.normalize(*args, **kwargs)
        self.disp[0].normalize(*args, **kwargs)
        self.disp[1].normalize(*args, **kwargs)
        self.disp_corr[0].normalize(*args, **kwargs)
        self.disp_corr[1].normalize(*args, **kwargs)

    def n(self):
        return self.line.n()

    def x(self):
        return self.line.x()

    def y(self):
        return self.line.y()

    def err(self):
        return self.line.err()

    def f(self):
        return self.line.f()

class image():
    """
    class for working with images (2d spectra) inside Spectrum plotting
    """
    def __init__(self, x=None, y=None, z=None, err=None, mask=None):
        if any([v is not None for v in [x, y, z, err, mask]]):
            self.set_data(x=x, y=y, z=z, err=err, mask=mask)
        else:
            self.z = None

    def set_data(self, x=None, y=None, z=None, err=None, mask=None):
        for attr, val in zip(['z', 'err', 'mask'], [z, err, mask]):
            if val is not None:
                setattr(self, attr, np.asarray(val))
            else:
                setattr(self, attr, val)
        if x is not None:
            self.x = np.asarray(x)
        else:
            self.x = np.arange(z.shape[0])
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = np.arange(z.shape[1])

        self.pos = [self.x[0] - (self.x[1] - self.x[0]) / 2, self.y[0] - (self.y[1] - self.y[0]) / 2]
        self.scale = [(self.x[-1] - self.x[0]) / (self.x.shape[0]-1), (self.y[-1] - self.y[0]) / (self.y.shape[0]-1)]
        for attr in ['z', 'err']:
            self.getQuantile(attr=attr)
            self.setLevels(attr=attr)

    def getQuantile(self, quantile=0.95, attr='z'):
        if getattr(self, attr) is not None:
            x = np.sort(getattr(self, attr).flatten())
            x = x[~np.isnan(x)]
            setattr(self, attr+'_quantile', [x[int(len(x)*(1-quantile)/2)], x[int(len(x)*(1+quantile)/2)]])
        else:
            setattr(self, attr + '_quantile', [0, 1])

    def setLevels(self, bottom=None, top=None, attr='z'):
        quantile = getattr(self, attr+'_quantile')
        if bottom is None:
            bottom = quantile[0]
        if top is None:
            top = quantile[1]
        top, bottom = np.max([top, bottom]), np.min([top, bottom])
        if top - bottom < (quantile[1] - quantile[0]) / 100:
            top += ((quantile[1] - quantile[0]) / 100 - (top - bottom)) / 2
            bottom -= ((quantile[1] - quantile[0]) / 100 - (top - bottom)) / 2
        setattr(self, attr + '_levels', [bottom, top])

    def find_nearest(self, x, y, attr='z'):
        z = getattr(self, attr)
        if len(z.shape) == 2:
            return z[np.min([z.shape[0] - 1, (np.abs(self.y - y)).argmin()]), np.min([z.shape[1]-1, (np.abs(self.x - x)).argmin()])]
        else:
            return None

    def collapse(self, axis='x', rect=None):
        if rect is None:
            rect = [[self.x[0], self.x[-1]], [self.y[0], self.y[-1]]]
        self.y_imin, self.y_imax = np.searchsorted(self.y, rect[1][0]), np.searchsorted(self.y, rect[1][1])
        self.x_imin, self.x_imax = np.searchsorted(self.x, rect[0][0]), np.searchsorted(self.x, rect[0][1])
        s_axis = 'y' if axis == 'x' else 'x'
        ind = 1 if axis == 'x' else 0
        return getattr(self, s_axis)[getattr(self, s_axis + '_imin'):getattr(self, s_axis + '_imax')], \
               np.sum(self.z[self.y_imin:self.y_imax, self.x_imin:self.x_imax], axis=ind)

    def add_mask(self, rect=None, add=True):
        if self.mask is None:
            self.mask = np.zeros_like(self.z)
        if rect is not None:
            x1, x2 = (np.abs(self.x - rect[0][0])).argmin(), (np.abs(self.x - rect[0][1])).argmin()
            y1, y2 = (np.abs(self.y - rect[1][0])).argmin(), (np.abs(self.y - rect[1][1])).argmin()
            self.mask[y1:y2+1, x1:x2+1] = int(add)

class spec2d():
    def __init__(self, parent):
        self.parent = parent
        self.filename = ''
        self.raw = image()
        self.cr = None
        self.sky = None
        self.slits = []
        self.gslits = []
        self.trace = None
        self.trace_width = [None, None]
        self.moffat = moffat_func()
        self.moffat_inter = None

    def set(self, x=None, y=None, z=None, err=None, mask=None):
        self.raw.set_data(x=x, y=y, z=z, err=err, mask=mask)

    def set_image(self, name, colormap):
        image = pg.ImageItem(colorMap=colormap)
        if name == 'raw':
            image.setImage(self.raw.z.T)
        elif name == 'err':
            image.setImage(self.raw.err.T)
        elif name == 'mask':
            image.setImage(self.raw.mask.T)
        elif name == 'cr':
            image.setImage(self.cr.mask.T)
        elif name == 'sky':
            image.setImage(self.sky.z.T)
        image.translate(self.raw.pos[0], self.raw.pos[1])
        image.scale(self.raw.scale[0], self.raw.scale[1])
        #image.setLookupTable(colormap)
        #image.setLevels(self.raw.levels)
        return image

    def cr_remove(self, update, **kwargs):
        if self.cr is None:
            self.cr = image(x=self.raw.x, y=self.raw.y, mask=np.zeros_like(self.raw.z))
        z = self.raw.z
        z = np.insert(z, 0, z[:4], axis=0)
        z = np.insert(z, z.shape[0], z[-4:], axis=0)
        z, mask = cosmicray_lacosmic(z, **kwargs)
        if update == 'new':
            self.cr.mask = mask[4:-4]
        if update == 'add':
            self.cr.mask = np.logical_or(self.cr.mask, mask[4:-4])
        print(update, np.sum(self.cr.mask.flatten()))

    def expand_mask(self, exp_pixel=1):
        m = np.copy(self.cr.mask)
        for p in itertools.product(np.linspace(-exp_pixel, exp_pixel, 2*exp_pixel+1).astype(int), repeat=2):
            m1 = np.copy(self.cr.mask)
            if p[0] < 0:
                m1 = np.insert(m1[:p[0],:], [0]*np.abs(p[0]), 0, axis=0)
            if p[0] > 0:
                m1 = np.insert(m1[p[0]:, :], [m1.shape[0]-p[0]]*p[0], 0, axis=0)
            if p[1] < 0:
                m1 = np.insert(m1[:,:p[1]], [0]*np.abs(p[1]), 0, axis=1)
            if p[1] > 0:
                m1 = np.insert(m1[:, p[1]:], [m1.shape[1]-p[1]]*p[1], 0, axis=1)
            m = np.logical_or(m, m1)
        self.cr.mask = m

    def neighbour_count(self):
        m = np.copy(self.cr.mask)
        c = np.zeros_like(self.cr.mask, dtype=int)
        for p in itertools.product(np.linspace(-1, 1, 3).astype(int), repeat=2):
            m1 = np.copy(self.cr.mask)
            if p[0] < 0:
                m1 = np.insert(m1[:p[0], :], [0] * np.abs(p[0]), 0, axis=0)
            if p[0] > 0:
                m1 = np.insert(m1[p[0]:, :], [m1.shape[0] - p[0]] * p[0], 0, axis=0)
            if p[1] < 0:
                m1 = np.insert(m1[:, :p[1]], [0] * np.abs(p[1]), 0, axis=1)
            if p[1] > 0:
                m1 = np.insert(m1[:, p[1]:], [m1.shape[1] - p[1]] * p[1], 0, axis=1)
            c = np.add(c, m1)
        return c

    def intelExpand(self, exp_factor=3, exp_pixel=1):
        z_saved, mask_saved = np.copy(self.raw.z), np.copy(self.cr.mask)
        self.expand_mask(exp_pixel=exp_pixel)
        self.extrapolate(inplace=True)
        self.cr.mask = np.logical_or(mask_saved, np.logical_and(np.abs((self.raw.z - z_saved) / self.raw.err) > exp_factor, self.cr.mask))
        neighbour = self.neighbour_count()
        self.cr.mask = np.logical_or(np.logical_and(self.cr.mask, neighbour>2), neighbour>6)
        if 1:
            self.parent.parent.s.append(Spectrum(self.parent.parent, 'delta'))
            self.parent.parent.s[-1].spec2d.set(x=self.raw.x, y=self.raw.y, z=(self.raw.z - z_saved) / self.raw.err)
            self.parent.parent.s[-1].spec2d.raw.setLevels(-exp_factor, exp_factor)
        if 1:
            self.parent.parent.s.append(Spectrum(self.parent.parent, 'neighbours'))
            self.parent.parent.s[-1].spec2d.set(x=self.raw.x, y=self.raw.y, z=neighbour)
            self.parent.parent.s[-1].spec2d.raw.setLevels(0, np.max(neighbour.flatten()))
        self.raw.z = z_saved

    def clean(self):
        mask_saved = np.copy(self.cr.mask)
        self.expand_mask(exp_pixel=2)
        z = (self.raw.z - self.sky.z)
        self.moffat_kind = 'pdf'
        inds = np.searchsorted(self.raw.x, self.trace[0][(self.trace[0] >= self.raw.x[0]) * (self.trace[0] <= self.raw.x[-1])])
        y, err = np.zeros(len(inds)), np.zeros(len(inds))
        for k, i in enumerate(inds):
            print(k, self.raw.x[i])
            ind = np.searchsorted(self.trace[0], self.raw.x[i])
            x_0 = self.trace[1][ind]
            gamma = self.trace[2][ind] / 2 / np.sqrt(2 ** (1 / 4.765) - 1)
            profile = self.moffat_fit_integ(self.raw.y, a=1, x_0=x_0, gamma=gamma, c=0)

            #v = np.sum((1 - self.cr.mask[:, i]) * profile ** 2)  # / self.raw.err[:, i]**2)
            flux = np.sum(z[:, i] * profile * (1 - self.cr.mask[:, i])) / np.sum((1 - self.cr.mask[:, i]) * profile ** 2)  # / self.raw.err[:, i]**2)
            z[:, i] = z[:, i] - profile * flux

        self.cr.mask = mask_saved
        self.parent.parent.s.append(Spectrum(self.parent.parent, 'clean'))
        self.parent.parent.s[-1].spec2d.set(x=self.raw.x, y=self.raw.y, z=z, err=self.raw.err, mask=self.cr.mask)

    def extrapolate(self, inplace=False, extr_width=1, extr_height=1, sky=1):
        z = np.copy(self.raw.z)
        if sky and self.sky is not None:
            z -= self.sky.z
        self.cr.mask = self.cr.mask.astype(bool)
        z[self.cr.mask] = np.nan
        kernel = Gaussian2DKernel(x_stddev=extr_width, y_stddev=extr_height)
        z = convolve(z, kernel)
        z1 = np.copy(self.raw.z)
        z1[self.cr.mask] = z[self.cr.mask]
        if sky and self.sky is not None:
            z1[self.cr.mask] += self.sky.z[self.cr.mask]
        if not inplace:
            self.parent.parent.s.append(Spectrum(self.parent.parent, 'CR_removed'))
            self.parent.parent.s[-1].spec2d.set(x=self.raw.x, y=self.raw.y, z=z1, err=self.raw.err, mask=self.raw.mask)
            if sky and self.sky is not None:
                self.parent.parent.s[-1].spec2d.sky = image(x=self.raw.x[:], y=self.raw.y[:], z=np.copy(self.sky.z), mask=np.copy(self.sky.mask))
            if self.trace is not None:
                self.parent.parent.s[-1].spec2d.trace = np.copy(self.trace)
        else:
            self.raw.z = z1

    def moffat_grid(self, gamma=1.0):
        g = np.linspace(gamma*0.5, gamma*2, 100)
        x = np.linspace(-7, 7, 100)
        grid = np.zeros([g.shape[0], x.shape[0]])
        for i, gi in enumerate(g):
            grid[i,:] = self.moffat.cdf(x, loc=0, scale=gi)
        self.moffat_inter = interp2d(x, g, grid)
        if 0:
            from mpl_toolkits.mplot3d import Axes3D

            X, Y = np.meshgrid(x, g)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X, Y, grid, linewidth=0, antialiased=False)
            plt.show()

    def moffat_fit_integ(self, x, a=1, x_0=0, gamma=1.0, c=0.0):
        dx = np.median(np.diff(x)) / 2
        x = np.append(x - dx, x[-1] + dx)
        if self.moffat_kind == 'cdf':
            y = self.moffat.cdf(x, loc=x_0, scale=gamma)
            return a * np.diff(y) + c
        elif self.moffat_kind == 'pdf':
            y = self.moffat.pdf(x, loc=x_0, scale=gamma)
            return a * (y[:-1] + np.diff(y)/2) + c
        elif self.moffat_kind == 'inter' and self.moffat_inter is not None:
            if gamma > self.moffat_inter.y[0] and gamma < self.moffat_inter.y[-1]:
                y = self.moffat_inter(x - x_0, np.ones_like(x)*gamma)
            else:
                y = self.moffat.cdf(x, loc=x_0, scale=gamma)
            return a * np.diff(y[0,:]) + c

    def profile(self, xmin, xmax, ymin, ymax, x_0=None, slit=None, plot=False, kind='pdf'):
        self.moffat_kind = kind
        x, y = self.raw.collapse(rect=[[xmin, xmax], [ymin, ymax]])
        if x_0 is None:
            x_0 = x[np.argmax(y)]
        c = np.median(np.append(y[:int(len(y) / 4)], y[int(3 * len(y) / 4)]))
        if slit is None:
            gamma = 2.35482 * np.std((y - c) * x) / np.std(y - c) / 2 / np.sqrt(2 ** (1 / 4.765) - 1)
        else:
            gamma = 2.35482 * slit / 2 / np.sqrt(2 ** (1 / 4.765) - 1)
        a = np.max(y) - c

        if plot:
            try:
                plt.close()
            except:
                pass
            fig, ax = plt.subplots()
            ax.plot(x, y, '-r')

        popt, pcov = curve_fit(self.moffat_fit_integ, x, y, p0=[a, x_0, gamma, c])

        print('popt', popt)
        print(Moffat1D(popt[0], popt[1], popt[2], 4.765).fwhm, popt[2] * 2 * np.sqrt(2 ** (1 / 4.765) - 1))
        pos, fwhm = self.raw.x[np.searchsorted(self.raw.x, (xmin+xmax)/2)], Moffat1D(popt[0], popt[1], popt[2], 4.765).fwhm
        if slit is None or (np.abs(popt[1] - x_0) < slit / 3 and np.abs(fwhm / slit - 1) < 0.3):
            self.slits.append([pos, np.min([np.max([popt[1], x[0]]), x[-1]]), fwhm])

        if plot:
            ax.plot(x, self.moffat_fit_integ(x, popt[0], popt[1], popt[2], popt[3]), '-b')
            ax.set_title('FWHM = {0:.2f}, center={1:.2f}'.format(fwhm, popt[1]))
            plt.show()
            self.parent.parent.s.redraw()

    def addSlits(self):
        for s in self.slits:
            self.gslits.append([pg.ErrorBarItem(x=np.asarray([s[0]]), y=np.asarray([s[1]]),
                                                top=np.asarray([s[2]]) / 2, bottom=np.asarray([s[2]]) / 2,
                                                pen=pg.mkPen('c', width=2), beam=4),
                                pg.ScatterPlotItem(x=np.asarray([s[0]]), y=np.asarray([s[1]]),
                                                   pen=pg.mkPen('k', width=0.5), brush=pg.mkBrush('c'))])
            self.parent.parent.spec2dPanel.vb.addItem(self.gslits[-1][0])
            self.parent.parent.spec2dPanel.vb.addItem(self.gslits[-1][1])

    def fit_trace(self, shape='poly'):
        pos, trace, width = np.transpose(np.asarray(self.slits))
        if shape == 'poly':
            p = np.polyfit(pos, trace, 3)
            trace_pos = np.polyval(p, self.parent.cont2d.x)
            p = np.polyfit(pos, width, 3)
            trace_width = np.polyval(p, self.parent.cont2d.x)
        elif shape == 'power':
            x1, x2, y1, y2 = pos[0], pos[-1], trace[0], trace[-1]
            powerlaw = lambda x, amp, index, c: amp * (x ** index) + c
            popt, pcov = curve_fit(powerlaw, p0=(x2 * (y2 - y1) / (x1 - x2), -1, (y1*x1 - y2*x2)/(x1-x2)))
            self.trace_pos = powerlaw(self.parent.cont2d.x, popt[0], popt[1], popt[2])
        if 0:
            self.parent.s.redraw()
        else:
            self.trace = [self.parent.cont2d.x[:], trace_pos, trace_width]

    def set_trace(self):
        self.trace_pos = pg.PlotCurveItem(x=self.trace[0], y=self.trace[1], pen=pg.mkPen(255, 255, 255, width=3))
        self.parent.parent.spec2dPanel.vb.addItem(self.trace_pos)
        self.trace_width = pg.PlotCurveItem(x=np.concatenate((self.trace[0], np.array([np.inf]), self.trace[0])),
                                            y=np.concatenate((self.trace[1] + self.trace[2] / 2, np.array([np.inf]), self.trace[1] - self.trace[2] / 2)),
                                            connect="finite", pen=pg.mkPen(255, 255, 255, width=3, style=Qt.DashLine))
        self.parent.parent.spec2dPanel.vb.addItem(self.trace_width)

    def sky_model(self, xmin, xmax, border=0, slit=None, mask_type='moffat', model='median', window=0, poly=3, conf=0.03, smooth=0, smooth_coef=0.3, inplace=True, plot=0):
        if self.cr is None:
            self.cr = image(x=self.raw.x, y=self.raw.y, mask=np.zeros_like(self.raw.z))

        if self.sky is None or not inplace:
            self.sky = image(x=self.raw.x, y=self.raw.y, z=np.zeros_like(self.raw.z), mask=np.zeros_like(self.raw.z))

        if self.trace is not None:
            if xmin == xmax:
                inds = np.searchsorted(self.raw.x, [xmin])
            else:
                inds = np.searchsorted(self.raw.x, self.trace[0][(self.trace[0] >= xmin) * (self.trace[0] <= xmax)])
        elif slit is not None:
            inds = np.where(np.logical_and(self.parent.cont_mask2d, np.logical_and(self.raw.x >= xmin, self.raw.x <= xmax)))[0]
        else:
            inds = []

        def fun(p, x, y):
            return np.polyval(p, x) - y

        def fun_wavy(p, x, y):
            return np.polyval(p[:-3], x) + (p[-1] * np.sin(p[-3] * x + p[-2])) - y

        for k, i in enumerate(inds):
            print(self.raw.x[i])
            if mask_type == 'moffat':
                if self.trace is None and slit is not None:
                    x_0, gamma = self.parent.cont2d.y[k], self.extr_slit / 2 / np.sqrt(2 ** (1 / 4.765) - 1)
                else:
                    ind = np.searchsorted(self.trace[0], self.raw.x[i])
                    x_0 = self.trace[1][ind]
                    gamma = self.trace[2][ind] * 2 / 2 / np.sqrt(2 ** (1 / 4.765) - 1)
                m = self.moffat.ppf([conf, 1-conf], loc=x_0, scale=gamma)
                mask_sky = np.logical_or(self.raw.y < m[0], self.raw.y > m[1])
            elif slit is not None:
                mask_sky = 1 / (np.exp(-40 * (np.abs(self.raw.y - self.parent.cont2d.y[k]) - slit * 2)) + 1)
            mask_sky[:border] = 0
            mask_sky[-border:] = 0
            mask_reg = np.zeros_like(self.raw.mask, dtype=bool)
            mask_reg[:, i - window:i + window + 1] = True
            #print(self.cr.mask,  mask_reg * mask_sky[:, np.newaxis])
            mask = np.logical_and(np.logical_not(self.cr.mask), mask_reg * mask_sky[:, np.newaxis])
            self.sky.mask[:, i] = mask[:, i]
            if window > 0:
                y = np.mean(self.raw.z[mask], axis=0)
            else:
                y = self.raw.z[mask]

            if len(y) > poly + 2:
                if model == 'median':
                    sky = np.median(y) * np.ones_like(self.raw.y)
                elif model == 'mean':
                    sky = np.mean(y) * np.ones_like(self.raw.y)
                elif model == 'polynomial':
                    p = np.polyfit(self.raw.y[mask[:,i]], y, poly)
                    sky = np.polyval(p, self.raw.y)
                elif model == 'robust':
                    p = np.polyfit(self.raw.y[mask[:, i]], y, poly)
                    res_robust = least_squares(fun, p, loss='soft_l1', f_scale=0.02, args=(self.raw.y[mask[:,i]], y))
                    sky = np.polyval(res_robust.x, self.raw.y)
                    if plot:
                        fig, ax = plt.subplots()
                        ax.plot(self.raw.y[mask[:, i]], y, 'ok')
                        ax.plot(self.raw.y, np.polyval(p, self.raw.y), '--r')
                        ax.plot(self.raw.y, sky, '-r')
                        plt.show()
                elif model == 'wavy':
                    p = np.polyfit(self.raw.y[mask[:, i]], y, poly)
                    periods = np.linspace(0.5, 8, 100)
                    angular_freqs = 2 * np.pi / periods
                    pgram = lombscargle(self.raw.y[mask[:, i]], y - np.polyval(p, self.raw.y[mask[:, i]]), angular_freqs)
                    guess_period = periods[np.argmax(pgram)]
                    guess_amp = np.std(y - np.polyval(p, self.raw.y[mask[:, i]])) * 2. ** 0.5
                    #print('guess:', guess_period, guess_amp)
                    p = np.append(p, [2 *np.pi / guess_period, 0, guess_amp])
                    res_robust = least_squares(fun_wavy, p, loss='linear', f_scale=0.003, args=(self.raw.y[mask[:, i]], y))
                    #print('fit:', 2 * np.pi / res_robust.x[-3], res_robust.x[-1])
                    sky = np.polyval(res_robust.x[:-3], self.raw.y) + (res_robust.x[-1] * np.sin(res_robust.x[-3] * self.raw.y + res_robust.x[-2]))
                    if plot:
                        fig, ax = plt.subplots()
                        ax.plot(self.raw.y[mask[:, i]], y, 'ok')
                        ax.plot(self.raw.y, np.polyval(res_robust.x[:-3], self.raw.y), '--r')
                        ax.plot(self.raw.y, sky, '-r')
                        plt.show()

            else:
                sky = 0

            self.sky.z[:,i] = sky

        if smooth > 0 and len(inds) > 30:
            smooth = smooth + 1 if smooth % 2 == 0 else smooth
            for k in range(self.sky.z.shape[0]):
                print(k)
                sk = np.asarray(self.sky.z[k, inds])

                if k == 30:
                    fig, ax = plt.subplots()
                    ax.plot(self.raw.x[inds], sk)
                mask = np.ones_like(sk, dtype=bool)
                for i in range(4):
                    y = savgol_filter(sk[mask], smooth*5, 5)
                    if k == 30:
                        ax.plot(self.raw.x[inds][mask], y, '--r')

                    mask[mask] = np.logical_and(np.abs((sk[mask] / y - 1)) < smooth_coef, mask[mask])

                    # y, lmbd = smooth_data(data[0], data[1], d=4, stdev=1e-4)
                sk[mask] = savgol_filter(sk[mask], 21, 5)
                self.sky.z[k, inds] = sk[:]
                #self.sky.z[k, inds][mask] = savgol_filter(sk[mask], 21, 5)
                if k == 30:
                    #print(self.sky.z[k, inds][mask], savgol_filter(sk[mask], 21, 5))
                    ax.plot(self.raw.x[inds][mask], savgol_filter(sk[mask], smooth, 5), '-g')
                    ax.plot(self.raw.x[inds], self.sky.z[k, inds], '-k')
                    plt.show()

        self.sky.getQuantile(quantile=0.9995)
        self.sky.setLevels()

    def sky_model_simple(self, xmin, xmax, border=0, conf=0.03, inplace=True):
        if self.cr is None:
            self.cr = image(x=self.raw.x, y=self.raw.y, mask=np.zeros_like(self.raw.z))

        if self.sky is None or not inplace:
            self.sky = image(x=self.raw.x, y=self.raw.y, z=np.zeros_like(self.raw.z), mask=np.zeros_like(self.raw.z))

        if self.trace is not None:
            if xmin == xmax:
                inds = np.searchsorted(self.raw.x, [xmin])
            else:
                inds = np.searchsorted(self.raw.x, self.trace[0][(self.trace[0] >= xmin) * (self.trace[0] <= xmax)])

        imin, imax = np.argmin(self.trace[0]), np.argmax(self.trace[0])
        tmin, tmax = self.trace[0][imin] - self.trace[1][imin], self.trace[0][imax] + self.trace[1][imax]

        mask_sky = np.logical_or(self.raw.y < tmin, self.raw.y > tmax)
        mask_sky[:border] = 0
        mask_sky[-border:] = 0
        mask = np.zeros_like(self.cr.mask, dtype=bool)
        mask[:, :] = mask_sky[:,np.newaxis] #np.repeat(mask_sky, self.cr.mask.shape[1], axis=1)

        self.sky.z[:, :] = np.ma.median(np.ma.array(self.raw.z, mask=np.logical_and(mask, self.cr.mask)), axis=0)[np.newaxis, :]

    def extract(self, xmin, xmax, slit=None, mask_type='moffat', helio=None, airvac=True, inplace=False, kind='pdf'):
        self.moffat_kind = 'pdf'
        if self.cr is None:
            self.cr = image(x=self.raw.x, y=self.raw.y, mask=np.zeros_like(self.raw.z))

        if self.trace is not None:
            inds = np.searchsorted(self.raw.x, self.trace[0][(self.trace[0] >= xmin) * (self.trace[0] <= xmax)])
        elif slit is not None:
            inds = np.where(np.logical_and(self.parent.cont_mask2d, np.logical_and(self.raw.x >= xmin, self.raw.x <= xmax)))[0]
        else:
            inds = []
        if self.sky is not None:
            sky = self.sky.raw.z
        else:
            sky = np.zeros_like(self.raw.z)
        y, err = np.zeros(len(inds)), np.zeros(len(inds))
        for k, i in enumerate(inds):
            print(k, self.raw.x[i])
            if mask_type == 'moffat':
                if self.trace is None and slit is not None:
                    x_0, gamma = self.parent.cont2d.y[k], self.extr_slit / 2 / np.sqrt(2 ** (1 / 4.765) - 1)
                else:
                    ind = np.searchsorted(self.trace[0], self.raw.x[i])
                    x_0 = self.trace[1][ind]
                    gamma = self.trace[2][ind] / 2 / np.sqrt(2 ** (1 / 4.765) - 1)
                profile = self.moffat_fit_integ(self.raw.y, a=1, x_0=x_0, gamma=gamma, c=0)
            elif mask_type == 'rectangular':
                profile = 1 / (np.exp(-40 * (np.abs(self.raw.y - self.parent.cont2d.y[k]) - self.extr_slit)) + 1)
            elif mask_type == 'gaussian':
                profile = np.exp(-(np.abs(self.raw.y - self.parent.cont2d.y[k]) / self.extr_slit / 2.35482) ** 2)

            #self.raw.err[:, i] = 1
            v = np.sum((1 - self.cr.mask[:, i]) * profile) #/ self.raw.err[:, i]**2)
            flux = np.sum((self.raw.z[:, i] - sky[:, i]) * profile * (1 - self.cr.mask[:, i])) # / self.raw.err[:, i]**2)

            if v is not np.nan:
                y[k] = flux / v
                if self.raw.err is not None:
                    err[k] = np.sqrt(np.sum((1 - self.cr.mask[:, i]) * profile * self.raw.err[:, i] ** 2) / v)

        if inplace:
            pass
        else:
            if self.raw.err is not None:
                data = self.raw.x[inds], np.asarray(y), np.asarray(err)
            else:
                data = self.raw.x[inds], np.asarray(y)
            self.parent.parent.s.append(Spectrum(self.parent.parent, 'extracted', data=data))
            if helio is not None:
                self.parent.parent.s[-1].helio_vel = helio
                self.parent.parent.s[-1].apply_shift(helio)

            if airvac:
                self.parent.parent.s[-1].airvac()
            self.parent.parent.s[-1].spec2d.set(x=self.parent.parent.s[-1].spec.x(), y=self.raw.y,
                                         z=self.raw.z[:, inds] - sky[:, inds])

class Spectrum():
    """
    class for plotting Spectrum with interactive functions
    """
    def __init__(self, parent, name=None, data=None, resolution=0):
        self.parent = parent
        self.filename = name
        self.resolution = resolution
        self.scaling_factor = 1
        self.date = ''
        self.wavelmin = None
        self.wavelmax = None
        self.init_pen()
        self.spec = specline(self)
        self.spec_factor = 1
        self.mask = specline(self)
        self.bad_mask = specline(self)
        self.fit_mask = specline(self)
        self.cont = gline()
        self.cont2d = gline()
        self.spline = gline()
        self.spline2d = gline()
        self.cont_mask = None
        self.cont_mask2d = None
        #self.norm = gline()
        self.sm = specline(self)
        self.rebin = None
        self.fit = fitline(self)
        self.fit_comp = []
        self.cheb = fitline(self)
        self.res = gline()
        self.kde = gline()
        self.parent.plot.specname.setText(self.filename)
        self.view = 'step'
        self.parent.s.ind = len(self.parent.s)
        if data is not None:
            self.set_data(data)
            self.parent.s.ind = len(self.parent.s)-1
            self.initGUI()
        self.spec2d = spec2d(self)
        self.mask2d = None
        self.cr_mask2d = None
        self.err2d = None
        self.sky2d = None

    def init_pen(self):
        self.err_pen = pg.mkPen(70, 130, 180)
        self.cont_pen = pg.mkPen(168, 66, 195, width=3)
        self.fit_pen = pg.mkPen(255, 69, 0, width=4)
        self.fit_disp_pen = pg.mkPen(255, 204, 35, width=2)
        self.fit_comp_pen = pg.mkPen(255, 215, 63, width=1.0)
        self.spline_brush = pg.mkBrush(0, 191, 255, 255) # pg.mkBrush(117, 218, 50, 255)

    def ind(self):
        return self.parent.s.index(self)

    def active(self):
        return self == self.parent.s[self.parent.s.ind]

    def initGUI(self):
        #print('caller name:', inspect.stack()[1][3], inspect.stack()[2][3], inspect.stack()[3][3], inspect.stack()[4][3])
        if self.active():
            self.view = self.parent.specview
            self.pen = pg.mkPen(255, 255, 255)
            self.brush = pg.mkBrush(52, 152, 219, 255)
            self.points_brush = pg.mkBrush(145, 224, 29)
            self.points_size = 15
            self.sm_pen = pg.mkPen(245, 0, 80)
            self.bad_brush = pg.mkBrush(252, 58, 38)
            self.region_brush = pg.mkBrush(147, 185, 69, 60)
            self.mask_region_brush = pg.mkBrush(0, 0, 0, 255)
            self.fit_pixels_pen = pg.mkPen(145, 180, 29, width=4)
            self.colormap = pg.colormap.getFromMatplotlib('viridis')
            self.maskcolormap = pg.ColorMap(np.linspace(0, 1, 2), [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], mode='rgb')
            self.cr_maskcolormap = pg.ColorMap(np.linspace(0, 1, 2), [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], mode='rgb')
        else:
            if self.parent.showinactive:
                self.view = self.parent.specview.replace('err', '')
                self.pen = pg.mkPen(100, 100, 100)
                self.brush = pg.mkBrush(66, 90, 113, 255)
                self.points_brush = pg.mkBrush(81, 122, 136)
                self.points_size = 8 if self.parent.normview else 3
                self.sm_pen = pg.mkPen(105, 30, 30, style=Qt.DashLine)
                self.bad_brush = pg.mkBrush(252, 58, 38, 10)
                self.region_brush = pg.mkBrush(72, 112, 202, 40)
                self.fit_pixels_pen = pg.mkPen(70, 100, 20, width=0)
            else:
                return None

        # >>> plot spectrum
        if len(self.spec.x()) > 0:
            x, y, err = self.spec.x(), self.spec.y(), self.spec.err()

            if 'err' in self.view and len(err) == len(y):
                self.g_err = pg.ErrorBarItem(x=x, y=y, top=err, pen=self.err_pen, bottom=err, beam=(x[1]-x[0])/2)
                self.g_err.setZValue(2)
                self.parent.vb.addItem(self.g_err)
            if 'point' in self.view:
                self.g_point = pg.ScatterPlotItem(x=x, y=y, size=10, brush=self.brush, pen=self.pen)
                self.g_point.setZValue(2)
                self.parent.vb.addItem(self.g_point)
            if 'step' in self.view or 'line' in self.view:
                self.g_line = plotLineSpectrum(parent=self, view=self.view, x=x, y=y, clickable=True, connect='finite')
                self.g_line.setPen(self.pen)
                self.g_line.sigClicked.connect(self.specClicked)
                self.g_line.setZValue(2)
                self.parent.vb.addItem(self.g_line)

            if len(self.spec.mask()) > 0 and np.sum(np.logical_not(self.spec.mask())) > 0:
                self.updateMask()

        # >>> plot fit point:
        self.set_fit_mask()
        if len(self.fit_mask.x()) > 0:
            if self.parent.selectview == 'points':
                self.points = pg.ScatterPlotItem(x=self.spec.x()[self.fit_mask.x()], y=self.spec.y()[self.fit_mask.x()], size=self.points_size, brush=self.points_brush)
                self.points.setZValue(3)
                self.parent.vb.addItem(self.points)

            elif self.parent.selectview == 'color':
                x, y = np.copy(self.spec.x()), np.copy(self.spec.y())
                if len(x) > 0:
                    y[np.logical_not(self.fit_mask.x())] = np.NaN
                self.points = plotLineSpectrum(parent=self, view=self.view, connect='finite', pen=self.fit_pixels_pen)
                if self.active:
                    self.points.setZValue(3)
                else:
                    self.points.setZValue(0)
                self.points.setData(x=x, y=y)
                self.parent.vb.addItem(self.points)

            elif self.parent.selectview == 'regions':
                self.updateRegions()


        # >>> plot bad point:
        if len(self.bad_mask.x()) > 0 and len(self.spec.x()) > 0:
            self.bad_pixels = pg.ScatterPlotItem(x=self.spec.x()[self.bad_mask.x()], y=self.spec.y()[self.bad_mask.x()],
                                                 size=30, symbol='d', brush=self.bad_brush)
            self.parent.vb.addItem(self.bad_pixels)

        # >>> plot fit:
        if self.parent.fitPoints:
            self.g_fit = pg.ScatterPlotItem(x=[], y=[], size=5, symbol='o',
                                            pen=self.fit_pen, brush=pg.mkBrush(self.fit_pen.color()))
        else:
            self.g_fit = pg.PlotCurveItem(x=[], y=[], pen=self.fit_pen)
        self.g_fit.setZValue(7)
        self.parent.vb.addItem(self.g_fit)
        self.set_gfit()
        if len(self.parent.fit.sys) > 0:
            self.construct_g_fit_comps()

        if self.parent.normview:
            self.normline = pg.InfiniteLine(pos=1, angle=0, pen=pg.mkPen(color=self.cont_pen.color(), style=Qt.DashLine))
            self.parent.vb.addItem(self.normline)
        else:
            if len(self.parent.s) == 0 or self.active():
                self.g_cont = pg.PlotCurveItem(x=self.cont.x, y=self.cont.y, pen=self.cont_pen)
                self.g_cont.setZValue(4)
                self.parent.vb.addItem(self.g_cont)
                self.g_spline = pg.ScatterPlotItem(x=self.spline.x, y=self.spline.y, size=12, symbol='s',
                                                   pen=pg.mkPen(0, 0, 0, 255), brush=self.spline_brush)
                self.g_spline.setZValue(5)
                self.parent.vb.addItem(self.g_spline)

        # >>> plot chebyshev continuum:
        self.g_cheb = pg.PlotCurveItem(x=[], y=[], pen=pg.mkPen(color=self.cont_pen.color(), style=Qt.DashLine, width=3))
        self.g_cheb.setZValue(4)
        self.parent.vb.addItem(self.g_cheb)

        if self.parent.fit.cont_fit:
            self.set_cheb()

        # >>> plot smooth of spectrum:
        if self.sm.n() > 0:
            self.sm_line = pg.PlotCurveItem(x=self.sm.x(), y=self.sm.y(), pen=self.sm_pen)
            self.parent.vb.addItem(self.sm_line)

        # >>> plot residuals:
        if self.parent.show_residuals and (len(self.parent.s) == 0 or self.active()):
            self.residuals = pg.ScatterPlotItem(x=self.res.x, y=self.res.y, size=10,
                                                brush=pg.mkBrush(52, 152, 219, 255))
            self.parent.residualsPanel.vb.addItem(self.residuals)
            self.kde_line = pg.PlotCurveItem(x=-self.kde.x, y=self.kde.y, pen=pg.mkPen(52, 152, 219, 255), fillLevel=0,
                                             brush=pg.mkBrush(52, 152, 219, 100))
            self.kde_line.setRotation(270)
            self.parent.residualsPanel.kde.addItem(self.kde_line)
            self.kde_local = pg.PlotCurveItem(x=-self.kde.x, y=self.kde.y, pen=pg.mkPen(46, 204, 113, 255))
            self.kde_local.setRotation(270)
            self.parent.residualsPanel.kde.addItem(self.kde_local)
            x = np.linspace(-3, 3, 100)
            y = 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)
            self.kde_gauss = pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(252, 52, 19, 255))
            self.kde_gauss.setRotation(270)
            self.parent.residualsPanel.kde.addItem(self.kde_gauss)

        # >>> plot 2d spectrum:
        if self.parent.show_2d and (len(self.parent.s) == 0 or self.active()):
            if self.spec2d.raw.z is not None and self.spec2d.raw.z.shape[0] > 0 and self.spec2d.raw.z.shape[1] > 0:
                self.image2d = self.spec2d.set_image('raw', self.colormap)
                print("levels:", self.spec2d.raw.z_levels)
                self.image2d.setLevels(self.spec2d.raw.z_levels)
                self.parent.spec2dPanel.vb.addItem(self.image2d)
                self.parent.spec2dPanel.vb.removeItem(self.parent.spec2dPanel.cursorpos)
                self.parent.spec2dPanel.vb.addItem(self.parent.spec2dPanel.cursorpos, ignoreBounds=True)
                if self.spec2d.raw.err is not None:
                    self.err2d = self.spec2d.set_image('err', self.colormap)
                if self.spec2d.raw.mask is not None:
                    self.mask2d = self.spec2d.set_image('mask', self.maskcolormap)
                if self.spec2d.cr is not None and self.spec2d.cr.mask is not None:
                    self.cr_mask2d = self.spec2d.set_image('cr', self.cr_maskcolormap)
                    self.parent.spec2dPanel.vb.addItem(self.cr_mask2d)
                if self.spec2d.trace is not None:
                    self.spec2d.set_trace()
                if len(self.spec2d.slits) > 0:
                    self.spec2d.addSlits()
                if self.spec2d.sky is not None:
                    self.sky2d = self.spec2d.set_image('sky', self.colormap)
            if len(self.parent.s) == 0 or self.active():
                self.g_cont2d = pg.PlotCurveItem(x=self.cont2d.x, y=self.cont2d.y, pen=self.cont_pen)
                self.parent.spec2dPanel.vb.addItem(self.g_cont2d)
                self.g_spline2d = pg.ScatterPlotItem(x=self.spline2d.x, y=self.spline2d.y, size=12, symbol='s',
                                                   pen=pg.mkPen(0, 0, 0, 255), brush=self.spline_brush)
                self.parent.spec2dPanel.vb.addItem(self.g_spline2d)

    def remove(self):
        try:
            if 'err' in self.view:
                try:
                    self.parent.vb.removeItem(self.g_err)
                except:
                    pass
            if 'point' in self.view:
                self.parent.vb.removeItem(self.g_point)
            if 'step' in self.view:
                self.parent.vb.removeItem(self.g_line)
            if 'line' in self.view:
                self.parent.vb.removeItem(self.g_line)
        except:
            pass
        try:
            self.remove_g_fit_comps()
        except:
            pass
        attrs = ['g_fit', 'g_fit_comp', 'points', 'bad_pixels', 'g_cont', 'g_spline',
                 'normline', 'sm_line', 'g_cheb', 'rebin']
        for attr in attrs:
            try:
                self.parent.vb.removeItem(getattr(self, attr))
            except:
                pass
        try:
            if self.parent.selectview == 'regions':
                for r in self.regions:
                    self.parent.vb.removeItem(r)
        except:
            pass

        try:
            for r in self.mask_regions:
                self.parent.vb.removeItem(r)
        except:
            pass

        try:
            self.parent.residualsPanel.vb.removeItem(self.residuals)
        except:
            pass
        try:
            self.parent.residualsPanel.kde.removeItem(self.kde_line)
            self.parent.residualsPanel.kde.removeItem(self.kde_gauss)
            self.parent.residualsPanel.kde.removeItem(self.kde_local)
            self.parent.residualsPanel.struct(clear=True)
        except:
            pass

        attrs = ['image2d', 'mask2d', 'g_cont2d', 'g_spline2d', 'trace_pos', 'trace_width']
        for attr in attrs:
            try:
                self.parent.spec2dPanel.vb.removeItem(getattr(self, attr))
            except:
                pass

        for g in self.spec2d.gslits:
            try:
                self.parent.spec2dPanel.vb.removeItem(g[0])
                self.parent.spec2dPanel.vb.removeItem(g[1])
            except:
                pass

    def redraw(self):
        self.remove()
        self.initGUI()
        try:
            self.parent.abs.redraw()
        except:
            pass
        self.set_res()

    def set_data(self, data=None, mask=None):
        if data is not None:
            if len(data) >= 3:
                if mask is not None:
                    mask = np.logical_and(mask, np.logical_and(data[1] != 0, data[2] != 0))
                self.spec.raw.set_data(x=data[0], y=data[1], err=data[2], mask=mask)
                if len(data) == 4:
                    self.cont.set_data(x=data[0], y=data[3])
                    self.cont_mask = np.ones_like(self.spec.x(), dtype=bool)
            elif len(data) == 2:
                if mask is not None:
                    mask = np.logical_and(mask, (data[1] != 0))
                self.spec.raw.set_data(x=data[0], y=data[1], mask=mask)
            else:
                mask = data[1] != np.NaN
        self.spec.raw.interpolate()
        self.wavelmin, self.wavelmax = self.spec.raw.x[0], self.spec.raw.x[-1]
        self.mask.set(x=np.zeros_like(self.spec.raw.x, dtype=bool))
        self.bad_mask.set(x=np.isnan(self.spec.raw.y))
        self.set_res()

    def rescale(self, scaling_factor):
        rescale = scaling_factor / self.scaling_factor
        self.scaling_factor = scaling_factor
        self.spec.raw.y *= rescale
        self.spec.raw.err *= rescale
        self.spec.raw.interpolate()
        #self.set_res()

    def update_fit(self):
        if len(self.fit.line.norm.x) > 0 and self.cont.n > 0 and self.active():
            self.set_gfit()
            self.set_res()
            if self.parent.fit.cont_fit:
                self.set_cheb()
            self.parent.s.chi2()

    def set_fit(self, x, y):
        if self.cont.n > 0: # and self.active():
            self.fit.line.norm.set_data(x=x, y=y)
            self.fit.line.norm.interpolate(fill_value=1)
            if not self.parent.normview:
                self.fit.line.normalize(norm=False, cont_mask=False, inter=True)
                self.fit.line.raw.interpolate()

    def set_gfit(self):
        if len(self.fit.line.norm.x) > 0 and self.cont.n > 0 and self.active():
            self.g_fit.setData(self.fit.line.x(), self.fit.line.y())

    def set_fit_disp(self, show=True):
        if show:
            for i in [0, 1]:
                self.fit.g_disp[i] = pg.PlotCurveItem(x=self.fit.disp[i].norm.x, y=self.fit.disp[i].norm.y, pen=self.fit_disp_pen)
                self.parent.vb.addItem(self.fit.g_disp[i])
            self.fit.g_disp[2] = pg.FillBetweenItem(self.fit.g_disp[0], self.fit.g_disp[1], brush=pg.mkBrush(tuple(list(self.fit_disp_pen.color().getRgb()[:3]) + [200])))
            self.parent.vb.addItem(self.fit.g_disp[2])
            if self.parent.fit.cont_fit:
                for i in [0, 1]:
                    self.cheb.g_disp[i] = pg.PlotCurveItem(x=self.cheb.disp[i].norm.x, y=self.cheb.disp[i].norm.y, pen=self.cont_pen)
                    self.parent.vb.addItem(self.cheb.g_disp[i])
                self.cheb.g_disp[2] = pg.FillBetweenItem(self.cheb.g_disp[0], self.cheb.g_disp[1], brush=pg.mkBrush(tuple(list(self.cont_pen.color().getRgb()[:3]) + [200])))
                self.parent.vb.addItem(self.cheb.g_disp[2])
            for k in range(len(self.fit_comp)):
                for i in [0, 1]:
                    self.fit_comp[k].g_disp[i] = pg.PlotCurveItem(x=self.fit_comp[k].disp[i].norm.x, y=self.fit_comp[k].disp[i].norm.y, pen=pg.mkPen(tuple(list(self.g_fit_comp[k].opts['pen'].color().getRgb()[:3]) + [100])))
                    self.parent.vb.addItem(self.fit_comp[k].g_disp[i])
                self.fit_comp[k].g_disp[2] = pg.FillBetweenItem(self.fit_comp[k].g_disp[0], self.fit_comp[k].g_disp[1], brush=pg.mkBrush(tuple(list(self.g_fit_comp[k].opts['pen'].color().getRgb()[:3]) + [50])))
                self.parent.vb.addItem(self.fit_comp[k].g_disp[2])

        else:
            try:
                for i in [0, 1, 2]:
                    self.parent.vb.removeItem(self.fit.g_disp[i])
                    if self.parent.fit.cont_fit:
                        self.parent.vb.removeItem(self.cheb.g_disp[i])
                    for k in range(len(self.fit_comp)):
                        self.parent.vb.removeItem(self.fit_comp[k].g_disp[i])
            except:
                pass

    def construct_fit_comps(self):
        self.set_fit_disp(show=False)
        self.fit_comp = []
        for sys in self.parent.fit.sys:
            self.fit_comp.append(fitline(self))
        if self.active():
            self.remove_g_fit_comps()
            self.construct_g_fit_comps()

    def set_fit_comp(self, x, y, ind=-1):
        for i in range(len(self.fit_comp)):
            if ind == i or ind == -1:
                self.fit_comp[i].line.norm.set_data(x=x, y=y)
                if not self.parent.normview:
                    self.fit_comp[i].normalize(norm=False, cont_mask=False, inter=True)
                if self.active():
                    if self.parent.comp_view == 'one' and self.parent.comp == i or self.parent.comp_view == 'all':
                        ind = i if self.parent.comp_view == 'all' else 0
                        self.g_fit_comp[ind].setData(x=self.fit_comp[i].line.x(), y=self.fit_comp[i].line.y())

    def construct_g_fit_comps(self):
        if self.active():
            self.g_fit_comp = []
            for i, c in enumerate(self.fit_comp):
                if self.parent.comp_view == 'one' and self.parent.comp == i or self.parent.comp_view == 'all':
                    style = Qt.DashLine if self.parent.comp_view == 'all' and self.parent.comp != i else Qt.SolidLine
                    color = pg.mkPen(50, 115, 235, width=1.0) if self.parent.comp_view == 'all' and self.parent.comp != i else self.fit_comp_pen.color()
                    pen = color = pg.mkPen(50, 115, 235, width=1.0) if self.parent.comp_view == 'all' and self.parent.comp != i else self.fit_comp_pen
                    self.g_fit_comp.append(pg.PlotCurveItem(x=c.line.x(), y=c.line.y(), pen=pen)) #pg.mkPen(color=color, style=style)))
                    self.g_fit_comp[-1].setZValue(6)
                    self.parent.vb.addItem(self.g_fit_comp[-1])

    def remove_g_fit_comps(self):
        try:
            for g in self.g_fit_comp:
                try:
                    self.parent.vb.removeItem(g)
                except:
                    pass
        except:
            pass

    def redrawFitComps(self):
        if self.active():
            self.remove_g_fit_comps()
            self.construct_g_fit_comps()

    def set_cheb(self, x=None, y=None):
        if self.active() and self.cont_mask is not None:
            if x is None and y is None:
                self.cheb.line.norm.set_data(x=self.spec.raw.x[self.cont_mask], y=self.correctContinuum(self.spec.raw.x[self.cont_mask]))
                self.cheb.line.normalize(norm=False, cont_mask=False)
            else:
                self.cheb.line.set(x=x, y=y)
            self.g_cheb.setData(x=self.cheb.x(), y=self.cheb.y())

    def set_res(self):
        if 1 and hasattr(self.parent, 'residualsPanel') and self.parent.s.ind < len(self.parent.s) and self.active() and len(self.fit.line.x()) > 0:
            self.res.x = self.spec.x()[self.fit_mask.x()]
            self.res.y = (self.spec.y()[self.fit_mask.x()] - self.fit.line.f(self.spec.x()[self.fit_mask.x()])) / self.spec.err()[self.fit_mask.x()]
            self.residuals.setData(x=self.res.x, y=self.res.y)
            if len(self.res.y) > 1 and np.sum(~np.isfinite(self.res.y)) == 0:
                kde = gaussian_kde(self.res.y)
                self.kde.x = np.linspace(np.min(self.res.y)-1, np.max(self.res.y)+1, len(self.res.x))
                self.kde.y = kde.evaluate(self.kde.x)
                self.kde_line.setData(x=-self.kde.x, y=self.kde.y)
                #self.fit_kde()

    def fit_kde(self):
        def gauss(x, *p):
            A, mu, sigma = p
            return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        p0 = [1., 0., 1.]

        coeff, var_matrix = curve_fit(gauss, self.kde.x, self.kde.y, p0=p0)

        print(coeff)
        # Get the fitted curve
        self.kde_fit.setData(x=-self.kde.x, y=gauss(self.kde.x, *coeff))

    def add_exact_points(self, points, tollerance=1e-6, remove=False, bad=False, redraw=True):
        for p in points:
            self.add_points(p*(1-tollerance), -np.inf, p*(1+tollerance), np.inf, remove=remove, bad=bad, redraw=redraw)

        self.set_fit_mask()
        try:
            self.set_res()
        except:
            pass

    def add_points(self, x1, y1, x2, y2, remove=False, bad=False, redraw=True):
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if not bad:
            mask = 'mask'
            points = 'points'
        else:
            mask = 'bad_mask'
            points = 'bad_pixels'

        local_mask = np.logical_and(np.logical_and((self.spec.x() > x1), (self.spec.x() < x2)), np.logical_and((self.spec.y() > y1), (self.spec.y() < y2)))
        if len(local_mask) > 0:
            if not remove:
                getattr(self, mask).set(x=np.logical_or(getattr(self, mask).x(), local_mask))
            else:
                getattr(self, mask).set(x=np.logical_and(getattr(self, mask).x(), np.logical_not(local_mask)))

            if redraw:
                if not bad:
                    self.update_points()
                else:
                    getattr(self, points).setData(x=self.spec.x()[getattr(self, mask).x()], y=self.spec.y()[getattr(self, mask).x()])
        #getattr(self, mask).normalize(not self.parent.normview)

    def update_points(self):
        x, y = np.copy(self.spec.x()), np.copy(self.spec.y())
        if len(x) > 0:
            y[np.logical_not(self.fit_mask.x())] = np.NaN
        if self.parent.selectview == 'points' or 'point' in self.parent.specview:
            self.points.setData(x=x, y=y)
        elif self.parent.selectview == 'color':
            self.points.setData(x=x, y=y, connect='finite')
        elif self.parent.selectview == 'regions':
            self.updateRegions()

    def updateRegions(self):
        try:
            for r in self.regions:
                self.parent.vb.removeItem(r)
        except:
            pass

        i = np.where(self.fit_mask.x())
        if i[0].shape[0] > 0:
            ind = np.where(np.diff(i[0]) > 1)[0]
            ind = np.sort(np.append(np.append(i[0][ind], i[0][ind + 1]), [i[0][-1], i[0][0]]))

            self.regions = []
            for i, k in enumerate(range(0, len(ind), 2)):
                x_r = self.spec.x()[-1] if ind[k+1] > len(self.spec.x())-2 else (self.spec.x()[ind[k+1]+1] + self.spec.x()[ind[k+1]]) / 2
                self.regions.append(VerticalRegionItem([(self.spec.x()[max(0, ind[k]-1)] + self.spec.x()[ind[k]]) / 2, x_r],
                                                       brush=self.region_brush))
                # self.regions.append(pg.LinearRegionItem([self.spec.x()[ind[k]], self.spec.x()[ind[k+1]]], movable=False, brush=pg.mkBrush(100, 100, 100, 30)))
                self.parent.vb.addItem(self.regions[-1])

    def updateMask(self):
        try:
            for r in self.mask_regions:
                self.parent.vb.removeItem(r)
        except:
            pass

        i = np.where(self.spec.mask())
        if i[0].shape[0] > 0:
            ind = np.where(np.diff(i[0]) > 1)[0]
            ind = np.sort(np.append(np.append(i[0][ind], i[0][ind + 1]), [i[0][-1], i[0][0]]))

            self.mask_regions = []
            for i, k in enumerate(range(0, len(ind), 2)):
                x_r = self.spec.x()[-1] if ind[k+1] > len(self.spec.x())-2 else (self.spec.x()[ind[k+1]+1] + self.spec.x()[ind[k+1]]) / 2
                self.mask_regions.append(VerticalRegionItem([(self.spec.x()[max(0, ind[k]-1)] + self.spec.x()[ind[k]]) / 2, x_r],
                                                       brush=self.mask_region_brush))
                #self.regions.append(pg.LinearRegionItem([self.spec.x()[ind[k]], self.spec.x()[ind[k+1]]], movable=False, brush=pg.mkBrush(100, 100, 100, 30)))
                self.parent.vb.addItem(self.mask_regions[-1])

    def add_spline(self, x, y, name=''):
        getattr(self, 'spline'+name).add(x=x, y=y)
        #getattr(self, 'spline'+name).sort()
        getattr(self, 'g_spline'+name).setData(x=getattr(self, 'spline'+name).x, y=getattr(self, 'spline'+name).y)
        self.calc_spline(name=name)
        self.update_fit()

    def del_spline(self, x1=None, y1=None, x2=None, y2=None, arg=None, name=''):
        if arg is None:
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            for i in reversed(range(getattr(self, 'spline'+name).n)):
                if x1 < getattr(self, 'spline'+name).x[i] < x2 and y1 < getattr(self, 'spline'+name).y[i] < y2:
                    getattr(self, 'spline'+name).delete(i)
        else:
            getattr(self, 'spline'+name).delete(arg)
        getattr(self, 'g_spline'+name).setData(x=getattr(self, 'spline'+name).x, y=getattr(self, 'spline'+name).y)
        if getattr(self, 'spline'+name).n > 1:
            self.calc_spline(name=name)
        self.update_fit()

    def set_spline(self, x, y, name=''):
        getattr(self, 'spline'+name).delete()
        self.add_spline(x, y, name=name)

    def calc_spline(self, name=''):
        if getattr(self, 'spline'+name).n > 1:
            k = 3 if getattr(self, 'spline'+name).n > 3 else getattr(self, 'spline'+name).n-1
            tck = splrep(getattr(self, 'spline'+name).x, getattr(self, 'spline'+name).y, k=k)

        if getattr(self, 'spline'+name).n > 1:
            setattr(self, 'cont_mask'+name, (getattr(self, 'spec'+name).raw.x > getattr(self, 'spline'+name).x[0]) & (getattr(self, 'spec'+name).raw.x < getattr(self, 'spline'+name).x[-1]))
            getattr(self, 'cont'+name).set_data(x=getattr(self, 'spec'+name).raw.x[getattr(self, 'cont_mask'+name)],
                                                y=splev(getattr(self, 'spec'+name).raw.x[getattr(self, 'cont_mask'+name)], tck))
        else:
            setattr(self, 'cont_mask' + name, None)
            setattr(self, 'cont'+name, gline())
        try:
            getattr(self, 'g_cont' + name).setData(x=getattr(self, 'cont'+name).x, y=getattr(self, 'cont'+name).y)
        except:
            pass

    def set_fit_mask(self):
        self.fit_mask.set(x=np.logical_and(self.mask.x(), np.logical_not(self.bad_mask.x())))

    def normalize(self, action='normalize'):
        if self.parent.normview and self.cont_mask is not None:
            self.spec.normalize(True, action=action)
            self.spec.norm.interpolate()

        if len(self.fit_comp) > 0:
            for comp in self.fit_comp:
                comp.normalize(self.parent.normview + self.parent.aodview, cont_mask=False, inter=True, action=action)
        if self.fit.line.norm.n > 0:
            self.fit.normalize(self.parent.normview + self.parent.aodview, cont_mask=False, inter=True, action=action)
            if not self.parent.normview:
                self.fit.line.raw.interpolate()

        self.mask.normalize(self.parent.normview, cont_mask=self.cont_mask is not None, action=action)
        self.bad_mask.normalize(self.parent.normview, cont_mask=self.cont_mask is not None, action=action)
        self.set_fit_mask()

    def calcCont(self, method='SG', xl=None, xr=None, iter=5, window=201, clip=2.5, sg_order=5, filter='flat', new=True, cont=False, sign=1):
        if self.spec.raw.n > 0:
            if xl is None:
                xl = self.spec.raw.x[0]
            if xr is None:
                xr = self.spec.raw.x[-1]

            y = np.copy(self.spec.raw.y)

            mask = (xl < self.spec.raw.x) * (self.spec.raw.x < xr)
            if cont:
                mask *= self.cont_mask
                self.cont.interpolate()
                y[mask] = self.cont.inter(self.spec.raw.x[mask])
                iter = 1
            ys = y[mask]

            if method == 'Bspline':
                inds = np.r_[np.arange(len(ys) // window + 1) * window, len(ys)] + np.nonzero(mask)[0][0]

            for i in range(iter):
                if i > 0:
                    if sign == 0:
                        mask[mask] *= np.abs(ys - y[mask]) / self.spec.raw.err[mask] < clip
                    else:
                        mask[mask] *= sign * (ys - y[mask]) / self.spec.raw.err[mask] < clip

                if method == 'SG':
                    ys = sg.savitzky_golay(y[mask], window_size=window, order=sg_order)
                if method == 'Smooth':
                    ys = smooth(y[mask], window_len=window, window=filter, mode='same')
                if method == 'Bspline':
                    ys = smooth(y[mask], window_len=window, window=filter, mode='same')
                    inter = interp1d(self.spec.raw.x[mask], ys, fill_value=(ys[0], ys[-1]), bounds_error=False)
                    if i == iter - 1:
                        self.spline.set_data(x=self.spec.raw.x[inds], y=inter(self.spec.raw.x[inds]))
                        self.g_spline.setData(x=self.spline.x, y=self.spline.y)
                    tck = splrep(self.spec.raw.x[inds], inter(self.spec.raw.x[inds]), k=3)
                    ys = splev(self.spec.raw.x[mask], tck)

            inter = interp1d(self.spec.raw.x[mask], ys, fill_value=(ys[0], ys[-1]), bounds_error=False)
            if new:
                self.cont_mask = (xl < self.spec.raw.x) & (self.spec.raw.x < xr)
                self.cont.set_data(x=self.spec.raw.x[self.cont_mask], y=inter(self.spec.raw.x[self.cont_mask]))
            else:
                y = np.copy(self.spec.raw.y)
                y[self.cont_mask] = self.cont.y
                mask = (xl < self.spec.raw.x) * (self.spec.raw.x < xr)
                y[mask] = inter(self.spec.raw.x[mask])
                self.cont_mask = np.logical_or(self.cont_mask, mask)
                self.cont.set_data(x=self.spec.raw.x[self.cont_mask], y=y[self.cont_mask])
            self.redraw()

    def findFitLines(self, ind=-1, tlim=0.01, all=True, debug=True):
        """
        Function to prepare lines to fit.
          - ind         : specify component to fit
          - all         : if all then look for all the lines located in the  normalized spectrum

        Prepared lines where fit will be calculated are stored in self.fit_lines list.
        """
        #debug = True

        if ind == -1:
            self.fit_lines = MaskableList([])
        elif hasattr(self, 'fit_lines') and len(self.fit_lines) > 0:
            mask = [line.sys != ind for line in self.fit_lines]
            self.fit_lines = self.fit_lines[mask]
        else:
            self.fit_lines = MaskableList([])

        if self.spec.norm.n > 0 and self.cont.n > 0:
            if all:
                x = self.spec.norm.x
            else:
                x = self.spec.norm.x[self.fit_mask.norm.x]
            for sys in self.parent.fit.sys:
                if ind == -1 or sys.ind == ind:
                    for sp in sys.sp.keys():
                        lin = self.parent.atomic.list(sp)
                        for l in lin:
                            if str(l) not in sys.exclude:
                                l.b = sys.sp[sp].b.val
                                l.logN = sys.sp[sp].N.val
                                l.z = sys.z.val
                                l.recalc = True
                                l.sys = sys.ind
                                l.range = tau(l, resolution=self.resolution).getrange(tlim=tlim)
                                l.cf = -1
                                if self.parent.fit.cf_fit:
                                    for i in range(self.parent.fit.cf_num):
                                        cf = getattr(self.parent.fit, 'cf_'+str(i))
                                        cf_sys = cf.addinfo.split('_')[0]
                                        if cf_sys == 'all':
                                            cf_sys = np.arange(len(self.parent.fit.sys))
                                        else:
                                            cf_sys = [int(s) for s in cf_sys.split('sys')[1:]]
                                        cf_exp = cf.addinfo.split('_')[1] if len(cf.addinfo.split('_')) > 1 else 'all'
                                        if (sys.ind in cf_sys) and (cf_exp == 'all' or self.ind() == int(cf_exp[3:])) and l.l()*(1+l.z) > cf.left and l.l()*(1+l.z) < cf.right:
                                            l.cf = i
                                l.stack = -1
                                if self.parent.fit.stack_num > 0:
                                    for i in range(self.parent.fit.stack_num):
                                        st = self.parent.fit.getValue('sts_'+str(i), 'addinfo')
                                        if st.strip() == 'N_{0:d}_{1:s}'.format(l.sys, sp.strip()):
                                            l.stack = i

                                if all:
                                    #if l.range[0] < x[-1] and l.range[1] > x[0]:
                                    if np.sum(l.range) / 2 < x[-1] and np.sum(l.range) / 2 > x[0]:
                                        self.fit_lines += [l]
                                else:
                                    if np.sum(np.logical_and(x >= l.range[0], x <= l.range[1])) > 0:
                                        self.fit_lines += [l]
        if debug:
            print('findFitLines', self.fit_lines, [l.cf for l in self.fit_lines])

    def calcFit(self, ind=-1, x=None, recalc=False, redraw=True, timer=False, tau_limit=0.01):
        """
           calculate the absorption profile
           - ind             : specify the exposure for which fit is calculated
           - recalc          : if True recalculate profile in all lines (given in self.fit_lines)
           - redraw          : if True redraw the fit
        :return:
        """
        if timer:
            t = Timer(str(ind))

        if self.spec.norm.n > 0 and self.cont.n > 0:
            # >>> update line parameters:
            for line in self.fit_lines:
                if ind == -1 or ind == line.sys:
                    if line.recalc or recalc:
                        sys = self.parent.fit.sys[line.sys]
                        line.b = sys.sp[line.name.split()[0]].b.val
                        line.logN = sys.sp[line.name.split()[0]].N.val
                        line.z = sys.z.val
                        line.tau = tau(line, resolution=self.resolution)
            if timer:
                t.time('update')

            # >>> create lambda grid:
            if x is None:
                x_spec = self.spec.norm.x
                mask_glob = self.mask.norm.x
                #mask_glob = np.zeros_like(x_spec)
                for line in self.fit_lines:
                    mask_glob = np.maximum(mask_glob, line.tau.grid_spec(x_spec, tlim=tau_limit))
                x = makegrid(x_spec, mask_glob)
                if timer:
                    t.time('create x')

            # >>> calculate the intrinsic absorption line spectrum
            flux = np.zeros_like(x)
            for line in self.fit_lines:
                if ind == -1 or ind == line.sys:
                    if line.recalc or recalc:
                        line.profile = line.tau.calctau(x, vel=False, verbose=False, convolve=None, tlim=tau_limit)
                        line.recalc = False
                    if not self.parent.fit.cf_fit:
                        flux += line.profile
                    else:
                        cf = np.min([1, np.sum([getattr(self.parent.fit, 'cf_' + str(i)).val for i in line.cf])])
                        flux += - np.log(np.exp(-line.profile) * (1 - cf) + cf)

            flux = np.exp(-flux)

            if timer:
                t.time('calc profiles')

            # >>> convolve the spectrum with instrument function
            #self.resolution = None
            if self.resolution not in [None, 0]:
                flux = convolveflux(x, flux, self.resolution, kind='direct')
            if timer:
                t.time('convolve')

            # >>> correct for artificial continuum:
            if self.parent.fit.cont_fit and self.parent.fit.cont_num > 0:
                flux = flux * self.correctContinuum(x)

            # >>> correct for dispersion:
            if self.parent.fit.disp_num > 0:
                for i in range(self.parent.fit.disp_num):
                    if getattr(self.parent.fit, 'dispz_' + str(i)).addinfo == 'exp_' + str(self.ind()):
                        f = interp1d(x + (x - getattr(self.parent.fit, 'dispz_' + str(i)).val) * getattr(self.parent.fit,'disps_' + str(i)).val, flux, bounds_error=False, fill_value=1)
                        flux = f(x)

            # >>> set fit graphics
            if ind == -1:
                self.set_fit(x=x, y=flux)
                if redraw:
                    self.set_gfit()
                    self.set_res()
            else:
                self.set_fit_comp(x=x, y=flux, ind=ind)
            if timer:
                t.time('set_fit')

    def calcFit_julia(self, comp=-1, x=None, recalc=False, redraw=True, timer=True, tau_limit=0.01):
        """
            calculate the absorption profile using Julia interface
           - comp            : specify the component for which fit is calculated
           - recalc          : if True recalculate profile in all lines (given in self.fit_lines)
           - redraw          : if True redraw the fit
        :return:
        """
        if timer:
            t = Timer(str(ind))
        if self.spec.norm.n > 0 and self.cont.n > 0:

            # >>> calculate the intrinsic absorption line spectrum
            if 1:
                x, flux = self.parent.julia.calc_spectrum(self.parent.julia_spec[self.ind()], self.parent.julia_pars, comp=comp + 1)
            else:
                flux = self.parent.julia.calc_spectrum(self.parent.julia_spec[self.ind()], self.parent.julia_pars, ind=comp + 1, out="init")
                x = self.spec.norm.x[self.mask.norm.x]

            if timer:
                t.time('calc fit')

            # >>> set fit graphics
            if comp == -1:
                self.set_fit(x=x, y=flux)
                if redraw:
                    self.set_gfit()
                    self.set_res()
            else:
                self.set_fit_comp(x=x, y=flux, ind=comp)
            if timer:
                t.time('set_fit')

    def calcFit_fft(self, ind=-1, recalc=True, redraw=True, debug=False, tau_limit=0.01):
        """
            calculate the absorption profile using convolution by fft (on the regular grid)
           - ind             : specify the exposure for which fit is calculated
           - recalc          : if True recalculate profile in all lines (given in self.fit_lines)
           - redraw          : if True redraw the fit
        :return:
        """
        #print(ind, self.spec.norm.n, self.cont.n)
        if self.spec.norm.n > 0 and self.cont.n > 0:
            # >>> update line parameters:
            for line in self.fit_lines:
                if ind == -1 or ind == line.sys:
                    if line.recalc or recalc:
                        sys = self.parent.fit.sys[line.sys]
                        line.b = sys.sp[line.name.split()[0]].b.val
                        line.logN = sys.sp[line.name.split()[0]].N.val
                        line.z = sys.z.val
                        line.tau = tau(line, resolution=self.resolution)

            # >>> create lambda grid:
            for line in self.fit_lines:
                line.tau.getrange(tlim=tau_limit)
            x_min = np.min([line.tau.range[0] for line in self.fit_lines])
            x_max = np.max([line.tau.range[1] for line in self.fit_lines])
            if len(self.spec.x()[self.mask.x()]) > 0:
                x_min = np.min([x_min, np.min(self.spec.x()[self.mask.x()])])
                x_max = np.max([x_max, np.max(self.spec.x()[self.mask.x()])])
            if self.resolution not in [None, 0]:
                x_mid = np.mean([x_min, x_max])
                delta = x_mid / self.resolution / 10
                x_min, x_max = x_min * (1 - 3/self.resolution), x_max * (1 + 3/self.resolution)
            else:
                delta = (self.spec.raw.x[1] - self.spec.raw.x[0])
                # x_min, x_max = x_min - 3 * 2.5 * delta, x_max + 3 * 2.5 * delta
            for l in self.fit_lines:
                delta = min(delta, l.tau.delta())
            num = int((x_max-x_min) / delta)
            #print(x_max, x_min, delta, num)
            x = np.logspace(np.log10(x_min), np.log10(x_max), num)

            mask = np.zeros_like(x, dtype=bool)
            for line in self.fit_lines:
                mask = np.logical_or(mask, np.logical_and(x > line.tau.range[0], x < line.tau.range[1]))
            sigma_n = (x_max + x_min) / 2 / delta / self.resolution / 2 / np.sqrt(2*np.log(2))
            #mask = np.ones(len(x), dtype=bool)
            if np.sum(mask) % 2 == 1:
                mask[np.nonzero(mask == True)[0][0]] = False
            num = np.sum(mask)
            #print(len(x), num)

            # >>> calculate the intrinsic absorption line spectrum
            flux = np.zeros_like(x)
            for line in self.fit_lines:
                if ind == -1 or ind == line.sys:
                    if line.recalc or recalc:
                        line.profile = line.tau.calctau(x, vel=False, verbose=False, convolve=None, tlim=tau_limit)
                        line.recalc = False
                    if not self.parent.fit.cf_fit:
                        flux += line.profile
                    else:
                        cf = np.min([1, np.sum([getattr(self.parent.fit, 'cf_' + str(i)).val for i in line.cf])])
                        flux += - np.log(np.exp(-line.profile) * cf + (1 - cf))

            flux = np.exp(-flux)

            # >>> convolve the spectrum with instrument function
            if self.resolution not in [None, 0]:
                f = np.fft.rfft(flux[mask])
                if 0:
                    freq = np.fft.rfftfreq(num, d=(x_max - x_min)/2/np.pi/num)
                    f *= np.exp(- np.power(x_mid/self.resolution/2/np.sqrt(2*np.log(2)) * freq, 2) / 2)
                else:
                    freq = np.fft.rfftfreq(num, d=(num - 0) / 2 / np.pi / num)
                    f *= np.exp(- 0.5 * sigma_n**2 * freq**2)
                flux[mask] = np.fft.irfft(f)

            if debug:
                print('calcFit_fft', x, flux)

            # >>> correct for artificial continuum:
            if self.parent.fit.cont_fit and self.parent.fit.cont_num > 0:
                flux = flux * self.correctContinuum(x)


            # >>> set fit graphics
            if ind == -1:
                self.set_fit(x=x, y=flux)
                if redraw:
                    self.update_fit()
            else:
                self.set_fit_comp(x=x, y=flux, ind=ind)

    def calcFit_fast(self, ind=-1, recalc=False, redraw=True, timer=False, num_between=3, tau_limit=0.01):
        """

           - ind             : specify the exposure for which fit is calculated
           - recalc          : if True recalculate profile in all lines (given in self.fit_lines)
           - redraw          : if True redraw the fit
           - num_between     : number of points to add between spectral pixels
           - tau_limit       : limit of optical depth to cutoff the line (set the range of calculations)
        :return:
        """
        if timer:
            t = Timer(str(ind))
        if self.spec.norm.n > 0 and self.cont.n > 0:
            # >>> update line parameters:
            for line in self.fit_lines:
                if ind == -1 or ind == line.sys:
                    if line.recalc or recalc:
                        sys = self.parent.fit.sys[line.sys]
                        line.b = sys.sp[line.name.split()[0]].b.val
                        line.logN = sys.sp[line.name.split()[0]].N.val
                        line.z = sys.z.val
                        line.tau = tau(line, resolution=self.resolution)
            if timer:
                t.time('update')

            # >>> create lambda grid:
            if self.resolution not in [None, 0]:
                if ind == -1:
                    x_spec = self.spec.norm.x
                    mask_glob = np.zeros_like(x_spec)
                    #mask_glob = self.mask.norm.x
                    for line in self.fit_lines:
                        if ind == -1 or ind == line.sys:
                            line.tau.getrange(tlim=tau_limit)
                            mask_glob = np.logical_or(mask_glob, ((x_spec > line.tau.range[0]) * (x_spec < line.tau.range[-1])))
                    x = makegrid(x_spec, mask_glob.astype(int) * num_between)
                else:
                    x = self.fit.line.norm.x
            else:
                x = self.spec.norm.x
            if timer:
                t.time('create x')

            # >>> calculate the intrinsic absorption line spectrum
            flux = np.zeros_like(x)
            for line in self.fit_lines:
                if ind == -1 or ind == line.sys:
                    if line.recalc or recalc:
                        line.profile = line.tau.calctau(x, vel=False, verbose=False, convolve=None, tlim=tau_limit)
                        line.recalc = False
                    if not self.parent.fit.cf_fit:
                        flux += line.profile

            # >>> include partial covering:
            if self.parent.fit.cf_fit:
                cfs, inds = [], []
                for i, line in enumerate(self.fit_lines):
                    if ind == -1 or ind == line.sys:
                        cfs.append(line.cf)
                        inds.append(i)
                cfs = np.array(cfs)
                for l in np.unique(cfs):
                    if l > -1:
                        cf = self.parent.fit.getValue('cf_' + str(l))
                    else:
                        cf = 1
                    profile = np.zeros_like(x)
                    for i, c in zip(inds, cfs):
                        if c == l:
                            profile += self.fit_lines[i].profile

                    flux += - np.log(np.exp(-profile) * cf + (1 - cf))

                #cf = np.min([1, np.sum([getattr(self.parent.fit, 'cf_' + str(i)).val for i in line.cf])])
                #flux += - np.log(np.exp(-line.profile) * (1 - cf) + cf)

            flux = np.exp(-flux)

            if timer:
                t.time('calc profiles')

            # >>> convolve the spectrum with instrument function
            #self.resolution = None
            if self.resolution not in [None, 0]:
                flux = convolveflux(x, flux, self.resolution, kind='direct')
            if timer:
                t.time('convolve')

            # >>> correct for artificial continuum:
            if self.parent.fit.cont_fit and self.parent.fit.cont_num > 0:
                flux = flux * self.correctContinuum(x)

            # >>> correct for dispersion:
            if self.parent.fit.disp_num > 0:
                for i in range(self.parent.fit.disp_num):
                    if getattr(self.parent.fit, 'dispz_' + str(i)).addinfo == 'exp_' + str(self.ind()):
                        f = interp1d(x + (x - getattr(self.parent.fit, 'dispz_' + str(i)).val) * getattr(self.parent.fit, 'disps_' + str(i)).val, flux, bounds_error=False, fill_value=1)
                        flux = f(x)

            # >>> set fit graphics
            if ind == -1:
                self.set_fit(x=x, y=flux)
                if redraw:
                    self.set_gfit()
                    self.set_res()
            else:
                self.set_fit_comp(x=x, y=flux, ind=ind)

            if timer:
                t.time('set_fit')

    def correctContinuum(self, x):
        """
        Calculate the correction to the continuum given chebyshev polinomial coefficients in self.fit
        """
        corr = np.ones_like(x)
        for k, c in enumerate(self.parent.fit.cont):
            if c.exp == self.ind():
                mask = (x > c.left) * (x < c.right)
                #print(mask)
                if len(x[mask]) > 0:
                    cheb = np.array([getattr(self.parent.fit, 'cont_' + str(k) + '_' + str(i)).val for i in range(c.num)])
                    #base = (x[mask] - x[mask][0]) * 2 / (x[mask][-1] - x[mask][0]) - 1
                    base = (x[mask] - c.left) * 2 / (c.right - c.left) - 1
                    corr[mask] = np.polynomial.chebyshev.chebval(base, cheb)

        return corr

    def chi(self):
        mask = self.fit_mask.x()
        if len(self.spec.x()) > 0 and np.sum(mask) > 0 and self.fit.line.n() > 0:
            return ((self.spec.y()[mask] - self.fit.line.f(self.spec.x()[mask])) / self.spec.err()[mask])
        else:
            return np.asarray([])

    def chi2(self):
        mask = self.fit_mask.norm.x
        spec = self.spec.norm
        chi2 = np.sum(np.power(((spec.y[mask] - self.fit.line.norm.f(spec.x[mask])) / spec.err[mask]), 2))
        return chi2

    def selectCosmics(self):
        y_sm = medfilt(self.spec.y(), 5)
        sigma = medfilt(self.spec.err(), 101)
        bad = np.abs(self.spec.y() - y_sm) / sigma > 3.5
        n = 2
        print(np.sum(bad))
        for i in range(1, n+1):
            bad[:len(bad)-i] = np.logical_or(bad[:len(bad)-i], bad[i:len(bad)])
            bad[i:len(bad)] = np.logical_or(bad[i:len(bad)], bad[:len(bad)-i])
            print(np.sum(bad))
        self.bad_mask.set(x=np.logical_or(self.bad_mask.x(), bad))
        self.remove()
        self.initGUI()

    def smooth(self, kind='astropy'):
        print('smoothing: ', self.filename)
        mask = np.logical_and(self.spec.y() != 0, self.spec.err() != 0)
        m = np.logical_and(np.logical_not(self.bad_mask.x()), mask)

        if kind == 'astropy':
            stddev = 1000.0 / 299794.25 * np.median(self.spec.x()) / np.median(np.diff(self.spec.x()))
            print(stddev)
            y = convolve(self.spec.y()[m], Gaussian1DKernel(stddev=stddev), boundary='extend')

        elif kind == 'convolveflux':
            y = convolveflux(self.spec.x()[m], self.spec.y()[m], 200.0, kind='gauss')

        elif kind == 'regular':
            d = 5
            y, lmbd = ds.smooth_data(self.spec.x()[m], self.spec.y()[m], d, xhat=np.linspace(self.spec.x()[m][0], self.spec.x()[m][-1], 1000))

        inter = interp1d(self.spec.x()[m], y, bounds_error=False,
                         fill_value=(y[0], y[-1]), assume_sorted=True)
        self.sm.set(x=self.spec.x()[mask], y=inter(self.spec.x()[mask]))
        self.sm.raw.interpolate(fill_value=(self.sm.raw.y[0], self.sm.raw.y[-1]))
        self.cont.x, self.cont.y = self.sm.x(), self.sm.y()
        self.redraw()
        #self.g_cont.setData(x=self.cont.x, y=self.cont.y)

    def rebinning(self, factor):
        if self.spec_factor == 1:
            self.spec_save = self.spec.raw.copy()
        self.spec_factor *= factor

        if self.spec_factor < 1:
            self.spec_factor = 1
        self.spec_factor = int(self.spec_factor)
        if self.spec_factor == 1:
            y, err = self.spec_save.y, self.spec_save.err
        else:
            # y = np.convolve(self.spec.raw.y, np.ones((factor,)) / factor, mode='same')
            cumsum = np.cumsum(np.r_[np.zeros((int(self.spec_factor / 2),)), self.spec_save.y, np.zeros(int(self.spec_factor / 2))])
            y = (cumsum[self.spec_factor:] - cumsum[:-self.spec_factor]) / float(self.spec_factor)
            cumsum = np.cumsum(np.r_[np.zeros((int(self.spec_factor / 2),)), self.spec_save.err, np.zeros(int(self.spec_factor / 2))])
            err = (cumsum[self.spec_factor:] - cumsum[:-self.spec_factor]) / float(self.spec_factor) / np.sqrt(float(self.spec_factor))
        self.set_data([self.spec.raw.x, y, err])
        self.redraw()

        if 0:
            if factor == 0:
                self.parent.vb.removeItem(self.rebin)
                self.rebin = None
            else:
                if self.rebin is None:
                    #self.rebin = plotStepSpectrum(x=[], y=[], stepMode=True)
                    self.rebin = pg.PlotCurveItem(x=[], y=[], pen=pg.mkPen(250, 100, 0, width=4))
                    self.parent.vb.addItem(self.rebin)

                if 0:
                    n = self.spec.raw.x.shape[0] // factor
                    x = self.spec.raw.x[:n * factor].reshape(self.spec.raw.x.shape[0] // factor, factor).sum(1) / factor
                    y = self.spec.raw.y[:n * factor].reshape(self.spec.raw.x.shape[0] // factor, factor).sum(1) / factor
                else:
                    if factor == 1:
                        y = self.spec_save.y
                    else:
                    #y = np.convolve(self.spec.raw.y, np.ones((factor,)) / factor, mode='same')
                        cumsum = np.cumsum(np.r_[np.zeros((int(factor/2),)), self.spec_save.y, np.zeros(int(factor/2))])
                        y = (cumsum[factor:] - cumsum[:-factor]) / float(factor)
                #y, err = spectres(self.spec.raw.x, self.spec.raw.y, x1, spec_errs=self.spec.raw.err)
                self.rebin.setData(x=self.spec.raw.x, y=y)

    def crosscorrExposures(self, ind, dv=50):
        fig, ax = plt.subplots()
        vgrid = np.linspace(-dv, dv, 100)
        c = np.zeros_like(vgrid)
        if len(self.parent.plot.regions) > 0:
            mask = np.zeros_like(self.spec.y(), dtype=bool)
            for r in self.parent.plot.regions:
                print(r.getRegion())
                mask = np.logical_or(mask, np.logical_and(self.spec.x() > r.getRegion()[0], self.spec.x() < r.getRegion()[1]))
        else:
            mask = np.ones_like(self.spec.y(), dtype=bool)
        for i, v in enumerate(vgrid):
            inter = interp1d(self.parent.s[ind].spec.x() * (1 + v/299792.458), self.parent.s[ind].spec.y(), bounds_error=False, fill_value=1)
            c[i] = np.sum(self.spec.y()[mask] * inter(self.spec.x()[mask]))
        ax.plot(vgrid, c)
        plt.show()

    def auto_select(self, x):
        ind = self.spec.index(x)
        i = ind
        while (not self.parent.normview and self.cont.f(self.spec.x()[i]) - self.spec.y()[i] > self.spec.err()[i]) or (self.parent.normview and 1 - self.spec.y()[i] > self.spec.err()[i]):
            self.add_exact_points([self.spec.x()[i]])
            i += 1
        self.add_exact_points([self.spec.x()[i]])
        i = ind
        while (not self.parent.normview and self.cont.f(self.spec.x()[i]) - self.spec.y()[i] > self.spec.err()[i]) or (self.parent.normview and 1 - self.spec.y()[i] > self.spec.err()[i]):
            self.add_exact_points([self.spec.x()[i]])
            i -= 1
        self.add_exact_points([self.spec.x()[i]])

    def apply_regions(self):
        regions = []
        for r in self.parent.plot.regions:
            if not r.active:
                    regions.append(r.size_full)
        self.spec.apply_region(regions)
        self.redraw()

    def apply_shift(self, vel):
        """
        apply shift of wavelenght in velocity space, specified by vel
        """
        factor = (1 + vel/ac.c.to('km/s').value)
        self.spec.raw.x *= factor
        if len(self.cont.x) > 0:
            print('shift cont')
            self.cont.x *= factor
        if len(self.fit.line.raw.x) > 0:
            print('shift fit')
            self.fit.line.raw.x *= factor
        print('Converted to Heliocentric velocities, helio_vel:', vel)

    def airvac(self):
        """
        correct from air to vacuum wavelenghts
        """
        n = 1.0
        for i in range(5):
            n_it = n
            sig2 = 1.0e8 / (self.spec.raw.x * self.spec.raw.x * n_it * n_it)
            n = 1.0e-8 * (15997.0 / (38.90 - sig2) + 2406030.0 / (130.0 - sig2) + 8342.13) + 1.0
        self.spec.raw.x *= n
        print('Converted to air-vacuum wavelenghts')

    def specClicked(self):
        if self.parent.plot.e_status:
            for i, s in enumerate(self.parent.s):
                if self == s:
                    self.parent.s.setSpec(i)
                    return

            #self.g_line.setPen(pg.mkPen(255, 255, 255))


    def mouseDragEvent(self, ev):
        
        if self.parent.plot.e_status:
            if ev.button() != Qt.LeftButton:
                ev.ignore()
                return
            
            if ev.isStart():
                # We are already one step into the drag.
                # Find the point(s) at the mouse cursor when the button was first 
                # pressed:
                pos = self.parent.parent.vb.mapSceneToView(ev.pos())
                self.st_pos = pos.x()
            
            pos = self.parent.parent.vb.mapSceneToView(ev.pos())
            self.parent.parent.delta += (pos.x() - self.st_pos)/self.line.l()
            self.parent.s[self.parent.s.s_ind].spec_x -= self.st_pos - pos.x()
            self.st_pos = pos.x()
            self.parent.s[self.parent.s.s_ind].redraw()
            ev.accept() 

class regionList(list):
    def __init__(self, parent):
        super(regionList).__init__()
        self.parent = parent

    def check(self, reg):
        if isinstance(reg, str):
            if reg in [str(r) for r in self]:
                return [str(r) for r in self].index(reg)
        elif isinstance(reg, regionItem):
            if reg in self:
                return self.index(reg)

    def add(self, reg=None, sort=True):
        if reg is None or (self.check(reg) is None and len(re.findall('[\d\.]+\.\.[\d\.]+', reg))>0):
            if reg is None:
                self.append(regionItem(self))
            else:
                self.append(regionItem(self, xmin=float(reg.split()[0].split('..')[0]), xmax=float(reg.split()[0].split('..')[1])))
                if len(reg.split()) > 1:
                    self[-1].addinfo = ' '.join(reg.split()[1:])

            self.parent.vb.addItem(self[-1])
        if sort:
            self.sortit()
        self.update()

    def remove(self, reg):
        i = self.check(reg)
        if i is not None:
            self.parent.vb.removeItem(self[i])
            del self[i]

    def sortit(self):
        self[:] = [self[i] for i in np.argsort([r.getRegion()[0] for r in self])]

    def update(self):
        for i, r in enumerate(self):
            color = cm.terrain(i / len(self), bytes=True)[:3] + (150,)
            r.setBrush(pg.mkBrush(color=color))

    def fromText(self, text, sort=True):
        for i in reversed(range(len(self))):
            self.remove(str(self[i]))
        for reg in text.splitlines():
            self.add(reg, sort=sort)

    def __str__(self):
        return '\n'.join([str(r) for r in self])

class regionItem(pg.LinearRegionItem):
    def __init__(self, parent, brush=None, xmin=None, xmax=None, span=(0.75, 1.0), addinfo=''):
        self.parent = parent
        if 1:
            brush = pg.mkBrush(173, 173, 173, 100)
        else:
            brush = pg.mkBrush(100+np.random.randint(0,100), 100+np.random.randint(0,100), 100+np.random.randint(0,100), 120)

        if xmin is None:
            xmin = self.parent.parent.mousePoint_saved.x()
        if xmax is None:
            xmax = self.parent.parent.mousePoint_saved.x()
        super().__init__(values=[xmin, xmax],
                         orientation=pg.LinearRegionItem.Vertical,
                         brush=brush,
                         span=span)
        self.active = False

        self.activeBrush = brush
        self.activeBrush.setStyle(Qt.SolidPattern)
        self.activePen = pg.mkPen(brush.color())

        self.inactivePen = pg.mkPen(150, 150, 150, 255, style=Qt.DashLine)
        self.inactiveBrush = pg.mkBrush(100, 100, 100, 255)
        self.inactiveBrush.setStyle(Qt.Dense5Pattern)

        self.updateLines()
        self.addinfo = addinfo

    def updateLines(self):
        if self.active:
            for l in self.lines:
                l.setPen(self.activePen)
                c = self.brush.color()
                c.setAlpha(255)
                l.setHoverPen(pg.mkPen(c))
                #l.setHoverPen(QPen(c))
            self.setSpan(0, 1)

        else:
            for l in self.lines:
                l.setPen(self.inactivePen)
                l.setHoverPen(self.inactivePen)
            self.setSpan(0.75, 1)

    def hoverEvent(self, ev):
        self.lines[0].setMovable((QApplication.keyboardModifiers() == Qt.ShiftModifier))
        self.lines[1].setMovable((QApplication.keyboardModifiers() == Qt.ShiftModifier))
        #if (QApplication.keyboardModifiers() == Qt.ShiftModifier):
        #    super(regionItem).hoverEvent(ev)

    def setMouseHover(self, hover):
        ## Inform the item that the mouse is(not) hovering over it
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        if hover:
            c = self.brush.color()
            #c.setAlpha(c.alpha() / 2)
            self.currentBrush = pg.mkBrush(c)
        else:
            self.currentBrush = self.brush
        self.update()

    def mouseDragEvent(self, ev):
        if (QApplication.keyboardModifiers() == Qt.ShiftModifier):
            super().mouseDragEvent(ev)

    def mouseClickEvent(self, ev):

        if ev.double(): # and (QApplication.keyboardModifiers() == Qt.ShiftModifier):
            self.active = not self.active
            if 0:
                self.setMovable(self.active) # and (QApplication.keyboardModifiers() == Qt.ShiftModifier))
                if self.active:
                    self.setBrush(self.activeBrush)
                    self.setRegion(self.size_full)
                else:
                    self.size_full = self.getRegion()
                    self.setRegion([self.size_full[0], self.size_full[0]+1])
                    self.setBrush(self.inactiveBrush)
            self.updateLines()
            #self.parent.parent.updateRegions()

        if ev.button() == Qt.LeftButton:
            if (QApplication.keyboardModifiers() == Qt.ControlModifier):
                self.parent.remove(self)

    #def __eq__(self, other):
    #    return (self.xmin == other.xmin) * (self.xmax == other.xmax)

    def __str__(self):
        return "{0:.1f}..{1:.1f} ".format(self.getRegion()[0], self.getRegion()[1]) + self.addinfo

class SpectrumFilter():
    def __init__(self, parent, name=None, system='AB'):
        self.parent = parent
        self.name = name
        self.system = system
        self.correct_name()
        self.init_data()
        self.gobject = None
        self.get_value()

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
            data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/data/filters/' + self.name + '.dat',
                                 skip_header=6, usecols=(0, 1), unpack=True)
            self.data = gline(x=data[0], y=data[1])
        if self.name in ['G', 'G_BP', 'G_RP']:
            data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/data/filters/GaiaDR2_Passbands.dat',
                                 skip_header=0, usecols=(0, {'G': 1, 'G_BP': 3, 'G_RP': 5}[self.name]), unpack=True)
            self.data = gline(x=data[0][data[1] < 1] * 10, y=data[1][data[1] < 1])
        if self.name in ['J_2MASS', 'H_2MASS', 'Ks_2MASS']:
            data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + f'/data/filters/2MASS_2MASS.{self.name}.dat'.replace('_2MASS.dat', '.dat'),
                skip_header=0, usecols=(0, 1), unpack=True)
            self.data = gline(x=data[0], y=data[1])
        if self.name in ['Y_VISTA', 'J_VISTA', 'H_VISTA', 'Ks_VISTA', 'K_VISTA']:
            data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + f'/data/filters/Paranal_VISTA.{self.name}.dat'.replace('_VISTA.dat', '.dat'),
                skip_header=0, usecols=(0, 1), unpack=True)
            self.data = gline(x=data[0], y=data[1])
        if self.name in ['Z_UKIDSS', 'Y_UKIDSS', 'J_UKIDSS', 'H_UKIDSS', 'K_UKIDSS']:
            data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + f'/data/filters/UKIRT_UKIDSS.{self.name}.dat'.replace('_UKIDSS.dat', '.dat'),
                skip_header=0, usecols=(0, 1), unpack=True)
            self.data = gline(x=data[0], y=data[1])
        if self.name in ['W1', 'W2', 'W3', 'W4']:
            data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + f'/data/filters/WISE_WISE.{self.name}.dat',
                                 skip_header=0, usecols=(0, 1), unpack=True)
            self.data = gline(x=data[0], y=data[1])
        if self.name in ['NUV', 'FUV']:
            data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + f'/data/filters/GALEX_GALEX.{self.name}.dat',
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

    def update(self, level):
        self.gobject.setData(x=self.data.x, y=level * self.data.y)
        self.get_value()
        self.label.setText(self.name + ':' + "{:0.2f}".format(self.value))
        self.label.setPos(self.data.x[self.ymax_pos], level * self.data.y[self.ymax_pos])

    def set_gobject(self, level):
        self.gobject = pg.PlotCurveItem(x=self.data.x, y=level * self.data.y, pen=pg.mkPen(color=self.color, width=0.5),
                                        fillLevel=0, brush=pg.mkBrush(self.color + (3,)))
        self.label = pg.TextItem(text=self.name + ':' + "{:0.2f}".format(self.value), anchor=(0, 1.2), color=self.color)
        self.label.setFont(QFont("SansSerif", 16))
        self.label.setPos(self.data.x[self.ymax_pos], level * self.data.y[self.ymax_pos])

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
                    flux = np.trapz(y * 1e-17 * x * self.inter(x), x=x) / np.trapz(x * self.inter(x), x=x) * self.l_eff ** 2 / ac.c.to('Angstrom/s').value
                    if self.name in ['u', 'g', 'r', 'i', 'z']:
                        self.value = - 2.5 / np.log(10) * np.arcsinh(flux / self.flux_0 / 2 / self.b) + self.m0
                    elif self.name in ['NUV', 'FUV']:
                        self.value = - 2.5 * np.log10(np.exp(1)) * np.arcsinh(flux / self.flux_0 / 1e-9 / 0.01) + self.m0
                else:
                    self.value = np.nan

            elif self.mag_type == 'Pogson':
                if flux is None:
                    if x is None or y is None:
                        x, y = self.parent.s[self.parent.s.ind].spec.x(), self.parent.s[self.parent.s.ind].spec.y()
                    mask = np.logical_and(x > self.data.x[0], x < self.data.x[-1])
                    x, y = x[mask], y[mask]
                    #print(np.trapz(y * 1e-17 * x ** 2 / ac.c.to('Angstrom/s').value * self.inter(x), x=x) / np.trapz(self.inter(x), x=x))
                    # y in 1e-17 erg/s/cm^2/AA a typical units in SDSS data
                    flux = np.trapz(y * 1e-17 * x * self.inter(x), x=x) / np.trapz(x * self.inter(x), x=x)
                self.value = - 2.5 * np.log10(flux / self.zp[system])
                #print(self.value)
        except:
            self.value = np.nan
        return self.value

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

class VerticalRegionItem(pg.UIGraphicsItem):

    def __init__(self, range=[0, 1],  brush=None):
        """Create a new LinearRegionItem.

        ==============  =====================================================================
        **Arguments:**
        range           A list of the positions of the lines in the region. These are not
                        limits; limits can be set by specifying bounds.
        brush           Defines the brush that fills the region. Can be any arguments that
                        are valid for :func:`mkBrush <pyqtgraph.mkBrush>`. Default is
                        transparent blue.
        ==============  =====================================================================
        """

        pg.UIGraphicsItem.__init__(self)
        self.bounds = QRectF()
        self.range = range

        if brush is None:
            brush = pg.mkBrush()
        self.setBrush(brush)


    def setBrush(self, *br, **kargs):
        """Set the brush that fills the region. Can have any arguments that are valid
        for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.brush = pg.mkBrush(*br, **kargs)
        self.currentBrush = self.brush

    def boundingRect(self):
        br = pg.UIGraphicsItem.boundingRect(self)
        br.setLeft(self.range[0])
        br.setRight(self.range[1])
        return br.normalized()

    def paint(self, p, *args):
        pg.UIGraphicsItem.paint(self, p, *args)
        p.setBrush(self.currentBrush)
        p.setPen(pg.mkPen(None))
        p.drawRect(self.boundingRect())


class CompositeSpectrum():
    def __init__(self, parent, kind='QSO', z=0.0):
        self.parent = parent
        self.z = z
        self.f, self.av = 1, 0
        self.type = kind
        if self.type == 'QSO':
            self.parent.compositeQSO_status += 1
            self.pen = pg.mkPen(color=pg.mkColor(243, 193, 58), width=5, alpha=0.7)
        elif self.type == 'Galaxy':
            self.parent.compositeGal_status += 1
            self.pen = pg.mkPen(color=pg.mkColor(103, 120, 245), width=3, alpha=0.5)
        self.setData()
        self.calc_scale()
        self.draw()

    def setData(self):
        if self.type == 'QSO':
            self.qso_names = ['X-shooter', 'SDSS', 'HST', 'power', 'power+FeII']
            if self.parent.compositeQSO_status == 1:
                self.spec = np.genfromtxt('data/SDSS/Slesing2016.dat', skip_header=0, unpack=True)
                self.spec = self.spec[:, np.logical_or(self.spec[1] != 0, self.spec[1] != 0)]
                self.spec[1] = smooth(self.spec[1], mode='same')
                if 0:
                    #x = self.spec[0][-1] + np.arange(1, int((25000 - self.spec[0][-1]) / 0.4)) * 0.4
                    x = self.spec[0][-1] * np.linspace(1, 30, 100)
                    y = np.power(x / 11350, -0.5) * 0.43
                    self.spec = np.append(self.spec, [x, y, y / 10], axis=1)
                else:
                    data = np.genfromtxt('data/SDSS/QSO1_template_norm.sed', skip_header=0, unpack=True)
                    m = data[0] > self.spec[0][-1]
                    self.spec = np.append(self.spec, [data[0][m], data[1][m] * self.spec[1][-1] / data[1][m][0], data[1][m] / 30], axis=1)
            elif self.parent.compositeQSO_status == 3:
                self.spec = np.genfromtxt('data/SDSS/medianQSO.dat', skip_header=2, unpack=True)
            elif self.parent.compositeQSO_status == 5:
                self.spec = np.genfromtxt('data/SDSS/hst_composite.dat', skip_header=2, unpack=True)
            elif self.parent.compositeQSO_status == 7:
                self.spec = np.ones((2, 1000))
                self.spec[0] = np.linspace(500, 25000, self.spec.shape[1])
                self.spec[1] = np.power(self.spec[0] / 2500, -1.9)
            elif self.parent.compositeQSO_status == 9:
                self.spec = np.ones((2, 25000))
                self.spec[0] = np.linspace(500, 25000, self.spec.shape[1])
                self.spec[1] = np.power(self.spec[0] / 2500, -1.7)
                if 0:
                    fe_opt = np.genfromtxt("data/Models/fe_optical.txt", comments='#', unpack=True)
                    fe_opt = interp1d(10 ** fe_opt[0], convolveflux(10 ** fe_opt[0], fe_opt[1] * 5e13, res=100, vel=False, kind='gauss'), bounds_error=False, fill_value=0)
                    fe_uv = np.genfromtxt("data/Models/fe_uv.txt", comments='#', unpack=True)
                    fe_uv = interp1d(10 ** fe_uv[0], convolveflux(10 ** fe_uv[0], fe_uv[1] * 5e13, res=100, vel=False, kind='gauss'), bounds_error=False, fill_value=0)
                    print(self.spec[0][2000], self.spec[1][2000], fe_uv(self.spec[0][2000]), fe_opt(self.spec[0][2000]))
                    self.spec[1] += fe_uv(self.spec[0]) + fe_opt(self.spec[0])
            self.parent.statusBar.setText(f'QSO template {self.qso_names[self.parent.compositeQSO_status // 2]} is loaded' )
        elif self.type == 'Galaxy':
            #print('gal_status: ', self.parent.compositeGal_status)
            if self.parent.compositeGal_status % 2:
                if 0:
                    f = fits.open(f"data/SDSS/spDR2-0{23 + self.parent.compositeGal_status // 2}.fit")
                    self.spec = 10 ** (f[0].header['COEFF0'] + f[0].header['COEFF1'] * np.arange(f[0].header['NAXIS1'])), f[0].data[0]
                else:
                    self.gal_names = ['S0', 'Sa', 'Sb', 'Sc', 'Sd', 'Sdm', 'Ell2', 'Ell5', 'Ell13', 'Sey2', 'Sey18']
                    self.spec = np.genfromtxt(f'data/SDSS/{self.gal_names[self.parent.compositeGal_status // 2]}_template_norm.sed', unpack=True)
                    self.parent.statusBar.setText(f'Galaxy composite loaded {self.gal_names[self.parent.compositeGal_status // 2]}_template_norm.sed')

    def draw(self):
        #self.gline = pg.PlotCurveItem(x=self.spec[0] * (1 + self.z), y=self.spec[1] * (1 + self.f), pen=pg.mkPen(color=self.color, width=2), clickable=True)
        self.gline = CompositeGraph(self, x=self.spec[0] * (1 + self.z), y=self.spec[1] * self.f, pen=self.pen, clickable=True)
        self.gline.sigClicked.connect(self.lineClicked)
        self.parent.vb.addItem(self.gline)

    def redraw(self):
        self.gline.setData(x=self.spec[0] * (1 + self.z), y=self.spec[1] * self.f * add_ext(self.spec[0], z_ext=0, Av=self.av, kind='SMC'))
        #self.label.redraw()

    def calc_scale(self):
        s = self.parent.s[self.parent.s.ind]
        mc = (self.spec[0] * (1 + self.z) > s.spec.x()[0]) * (self.spec[0] * (1 + self.z) < s.spec.x()[-1])
        ms = (s.spec.x() > self.spec[0][0] * (1 + self.z)) * (s.spec.x() < self.spec[0][-1] * (1 + self.z))
        #print(np.sum(mc), np.sum(ms))
        #print(np.nanmean(s.spec.y()[ms]), np.nanmean(self.spec[1][mc]))
        if np.sum(mc) > 10 and np.sum(ms) > 10:
            self.f = np.nanmean(s.spec.y()[ms]) / np.nanmean(self.spec[1][mc])
        #print(self.f)

    def remove(self):
        if self.type == 'QSO':
            self.parent.compositeQSO_status += 1
            if self.parent.compositeQSO_status == 10:
                self.parent.compositeQSO_status = 0
        if self.type == 'Galaxy':
            self.parent.compositeGal_status += 1
            if self.parent.compositeGal_status == len(self.gal_names) * 2: #12:
                self.parent.compositeGal_status = 0

        self.parent.vb.removeItem(self.gline)
        del self

    def lineClicked(self):
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.remove()

class CompositeGraph(pg.PlotCurveItem):
    def __init__(self, parent, **kwargs):
        super().__init__(**kwargs)
        self.parent = parent
        self.setZValue(5)

    def mouseDragEvent(self, ev):
        if QApplication.keyboardModifiers() in [Qt.ShiftModifier, Qt.AltModifier]:
            if ev.button() != Qt.LeftButton:
                ev.ignore()
                return

            if ev.isStart():
                self.start = ev.buttonDownPos()
            elif ev.isFinish():
                self.start = None
                return
            else:
                self.redraw_pos(start=self.start, finish=ev.pos(), modifier=QApplication.keyboardModifiers())
                self.start = ev.pos()
                if self.start is None:
                    ev.ignore()
                    return

            ev.accept()


    def redraw_pos(self, start, finish, modifier=Qt.ShiftModifier):
        if modifier == Qt.ShiftModifier:
            self.parent.z = finish.x() / start.x() * (1 + self.parent.z) - 1
            self.parent.f = self.parent.f * finish.y() / start.y()
        elif modifier == Qt.AltModifier:
            self.parent.av = self.parent.av + (1 - finish.y() / start.y())
            self.parent.parent.statusBar.setText('Av={0:5.2}'.format(self.parent.av))
        self.parent.redraw()

    def mouseClickEvent(self, ev):

        if QApplication.keyboardModifiers() == Qt.ControlModifier and ev.button() == Qt.LeftButton:
            self.parent.remove()


