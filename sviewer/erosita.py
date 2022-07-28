import astropy.constants as ac
from astropy.cosmology import Planck15, FlatLambdaCDM, LambdaCDM
from astropy.io import fits
import astropy.units as u
from collections import OrderedDict
import corner
from chainconsumer import ChainConsumer
from dust_extinction.averages import G03_SMCBar
import emcee
import itertools
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
import lmfit
import pyqtgraph as pg
from pyqtgraph.dockarea import *
from functools import partial
import os
import pandas as pd
import pickle
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QMenu, QToolButton,
                             QLabel, QCheckBox, QFrame, QTextEdit, QSplitter, QComboBox, QAction, QSizePolicy)
from scipy.interpolate import interp1d
from scipy.special import gamma, gammainc
from scipy.stats import linregress, gaussian_kde

from sklearn import linear_model
import sfdmap
from .graphics import SpectrumFilter
from .tables import *
from .utils import smooth, Timer
from ..a_unc import a
from ..profiles import add_ext, add_LyaForest
from ..stats import distr1d, distr2d

class dataPlot(pg.PlotWidget):
    def __init__(self, parent, axis):
        super(dataPlot, self).__init__()
        self.parent = parent
        self.vb = self.getViewBox()
        self.name = axis[0] + axis[1]
        self.x_axis = axis[0]
        self.y_axis = axis[1]
        self.setLabel('bottom', self.x_axis)
        self.setLabel('left', self.y_axis)
        self.reg = {'all': None, 'all_r': None, 'selected': None}
        self.vb = self.getViewBox()
        self.cursorpos = pg.TextItem(anchor=(0, 1), fill=pg.mkBrush(0, 0, 0, 0.5))
        self.vb.addItem(self.cursorpos, ignoreBounds=True)
        self.selectedstat = {'shown': pg.TextItem(anchor=(1, 1), fill=pg.mkBrush(0, 0, 0, 0.5)),
                             'selected': pg.TextItem(anchor=(1, 1), fill=pg.mkBrush(0, 0, 0, 0.5)),
                             'shown_sel': pg.TextItem(anchor=(1, 1), fill=pg.mkBrush(0, 0, 0, 0.5))}
        for k in self.selectedstat.keys():
            self.vb.addItem(self.selectedstat[k], ignoreBounds=True)
        self.corr = {'all': pg.TextItem(anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 0.5)),
                     'all_r': pg.TextItem(anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 0.5)),
                     'selected': pg.TextItem(anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 0.5))}
        self.s_status = False
        self.d_status = False
        self.show()

    def plotData(self, x=None, y=None, c=None, name='sample', color=(255, 255, 255), size=7, rescale=True):
        if hasattr(self, name) and getattr(self, name) is not None:
            self.removeItem(getattr(self, name))

        if c is not None and len(c) == len(x):
            col = np.ones((len(c), 4))
            col[np.isfinite(c)] = cm.get_cmap(self.parent.cmap.currentText())((np.nanmax(c) - c[np.isfinite(c)]) / (np.nanmax(c) - np.nanmin(c)))
            brush = [pg.mkBrush([int(i * 255) for i in cl]) for cl in col]
        else:
            brush = pg.mkBrush(*color, 255)

        if x is not None and y is not None and len(x) > 0 and len(y) > 0:
            if size < 3:
                setattr(self, name, pg.ScatterPlotItem(x, y, brush=brush, pen=pg.mkPen(None), size=size, pxMode=True))
            else:
                setattr(self, name, pg.ScatterPlotItem(x, y, brush=brush, size=size, pxMode=True))
            self.addItem(getattr(self, name))
            if rescale:
                self.enableAutoRange()
            else:
                self.disableAutoRange()

    def plotRegression(self, x=None, y=None, name='all', remove=True):
        if self.reg[name] is not None and remove:
            self.removeItem(self.reg[name])
            self.vb.removeItem(self.corr[name])

        if x is not None and y is not None:
            pens = {'all': pg.mkPen(0, 127, 255, width=6), 'selected': pg.mkPen(30, 250, 127, width=6), 'all_r': pg.mkPen(0, 127, 255, width=4, style=Qt.DashLine)}
            self.reg[name] = pg.PlotCurveItem(x, y, pen=pens[name])
            self.addItem(self.reg[name])
            self.vb.addItem(self.corr[name], ignoreBounds=True)
            self.plotRegLabels(self.vb.sceneBoundingRect())

    def keyPressEvent(self, event):
        super(dataPlot, self).keyPressEvent(event)

        if not event.isAutoRepeat():

            if event.key() == Qt.Key_S:
                self.s_status = True

            if event.key() == Qt.Key_D:
                self.d_status = True

        if any([event.key() == getattr(Qt, 'Key_' + s) for s in 'SD']):
            self.vb.setMouseMode(self.vb.RectMode)

    def keyReleaseEvent(self, event):
        super(dataPlot, self).keyPressEvent(event)

        if not event.isAutoRepeat():

            if event.key() == Qt.Key_S:
                self.s_status = False

            if event.key() == Qt.Key_D:
                self.d_status = False

            if (event.key() == Qt.Key_S or event.key() == Qt.Key_D) and hasattr(self.parent, 'd'):
                if self.parent.showKDE.isChecked():
                    self.parent.d.plot(frac=0.15, stats=True, xlabel=self.parent.axis_info[self.parent.ero_x_axis][1],
                                       ylabel=self.parent.axis_info[self.parent.ero_y_axis][1])
                    plt.show()

        if any([event.key() == getattr(Qt, 'Key_' + s) for s in 'SD']):
            self.vb.setMouseMode(self.vb.PanMode)

    def mousePressEvent(self, event):
        super(dataPlot, self).mousePressEvent(event)

        self.mousePoint_saved = self.vb.mapSceneToView(event.pos())

    def mouseReleaseEvent(self, event):
        if any([getattr(self, s + '_status') for s in 'sd']):
            self.vb.setMouseMode(self.vb.PanMode)
            self.vb.rbScaleBox.hide()

        if self.s_status or self.d_status:
            self.parent.select_points(self.mousePoint_saved.x(), self.mousePoint_saved.y(), self.mousePoint.x(),
                                      self.mousePoint.y(), remove=self.d_status, add=QApplication.keyboardModifiers() == Qt.ShiftModifier)

        if event.isAccepted():
            super(dataPlot, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        super(dataPlot, self).mouseMoveEvent(event)
        self.mousePoint = self.vb.mapSceneToView(event.pos())
        self.mouse_moved = True

        pos = self.vb.sceneBoundingRect()
        self.cursorpos.setText('x={0:.3f}, y={1:.2f}'.format(self.mousePoint.x(), self.mousePoint.y()))
        self.cursorpos.setPos(self.vb.mapSceneToView(QPoint(pos.left() + 10, pos.bottom() - 10)))
        for ind, name in enumerate(['shown_sel', 'selected', 'shown']):
            if name == 'shown_sel':
                s = np.sum(np.logical_and(np.isfinite(self.parent.x[self.parent.mask]),
                                          np.isfinite(self.parent.y[self.parent.mask])))
            if name == 'selected':
                s = np.sum(self.parent.mask)
            if name == 'shown':
                s = np.sum(np.logical_and(np.isfinite(self.parent.x),
                                          np.isfinite(self.parent.y)))
            self.selectedstat[name].setText(name + '={0:d}'.format(s))
            self.selectedstat[name].setPos(self.vb.mapSceneToView(QPoint(pos.right() - 10, pos.bottom() - 10 - ind * 20)))
        self.plotRegLabels(pos)

    def plotRegLabels(self, pos):
        for name, ind in zip(['all', 'all_r', 'selected'], [10, 30, 50]):
            if self.reg[name] is not None:
                self.corr[name].setPos(self.vb.mapSceneToView(QPoint(pos.left() + 10, pos.top() + ind)))
                self.corr[name].setText(name + ': ' + self.parent.reg[name])

    def mouseDoubleClickEvent(self, ev):
        super(dataPlot, self).mouseDoubleClickEvent(ev)
        if ev.button() == Qt.LeftButton:
            self.mousePos = self.vb.mapSceneToView(ev.pos())
            self.parent.index(x=self.mousePos.x(), y=self.mousePos.y())
            ev.accept()

class ComboMultipleBox(QToolButton):
    def __init__(self, parent, name=None):
        super(ComboMultipleBox, self).__init__()
        self.parent = parent
        self.name = name
        self.setFixedSize(200, 30)
        self.toolmenu = QMenu(self)
        self.list = []
        self.setMenu(self.toolmenu)
        self.setPopupMode(QToolButton.InstantPopup)

    def addItems(self, items):
        for item in items:
            if item not in self.list:
                self.list.append(item)
                setattr(self, item, QAction(item, self.toolmenu))
                getattr(self, item).setCheckable(True)
                getattr(self, item).triggered.connect(partial(self.set, item))
                self.toolmenu.addAction(getattr(self, item))

    def update(self):
        for item in self.list:
            if self.name == 'table':
                if item in self.parent.shown_cols:
                    getattr(self, item).setChecked(True)
            if self.name == 'hosts':
                if item in self.parent.host_templates:
                    getattr(self, item).setChecked(True)
            if self.name == 'filters':
                if item in self.parent.filter_names:
                    getattr(self, item).setChecked(True)

    def currentText(self):
        return ' '.join([s for s in self.list if s.isChecked()])

    def set(self, item):
        #print(item, self.name)
        if self.name == 'table':
            if getattr(self, item).isChecked():
                self.parent.shown_cols.append(item)
            else:
                self.parent.shown_cols.remove(item)
            l = [self.list.index(o) for o in self.parent.shown_cols]
            self.parent.shown_cols = [self.list[l[o]] for o in np.argsort(l)]
            self.parent.parent.options('ero_colnames', ' '.join(self.parent.shown_cols))
            self.parent.ErositaTable.setdata(self.parent.df[self.parent.shown_cols].to_records(index=False))

        if self.name == 'extcat':
            self.parent.addExternalCatalog(item, show=getattr(self, item).isChecked())

        if self.name == 'hosts':
            if getattr(self, item).isChecked():
                self.parent.host_templates.append(item)
            else:
                self.parent.host_templates.remove(item)
            l = [self.list.index(o) for o in self.parent.host_templates]
            self.parent.host_templates = [self.list[l[o]] for o in np.argsort(l)]
            self.parent.parent.options('ero_hosts', ' '.join(self.parent.host_templates))

        if self.name == 'filters':
            if getattr(self, item).isChecked():
                self.parent.filter_names.append(item)
            else:
                self.parent.filter_names.remove(item)
            l = [self.list.index(o) for o in self.parent.filter_names]
            self.parent.filter_names = [self.list[l[o]] for o in np.argsort(l)]
            self.parent.parent.options('ero_filters', ' '.join(self.parent.filter_names))

class Filter():
    def __init__(self, parent, name, value=None, flux=None, err=None, system='AB'):
        self.parent = parent
        self.name = name
        self.filter = SpectrumFilter(self.parent.parent, name=self.name, system=system)
        self.system = system

        if value is not None:
            self.value, self.err = value, err
        elif flux is not None:
            self.value, self.err = self.filter.get_value(flux=flux), self.filter.get_value(flux+err) - self.filter.get_value(flux)

        #print(name, self.value, self.err)
        if value is not None and err is not None:
            self.flux = self.filter.get_flux(value) * 1e17
            self.err_flux = [(self.filter.get_flux(value + err) - self.filter.get_flux(value)) * 1e17, (self.filter.get_flux(value) - self.filter.get_flux(value - err)) * 1e17]
            self.scatter = pg.ErrorBarItem(x=self.filter.l_eff, y=self.flux, top=self.err_flux[0], bottom=self.err_flux[1],
                                          beam=2, pen=pg.mkPen(width=2, color=self.filter.color))
            self.errorbar = pg.ScatterPlotItem(x=[self.filter.l_eff], y=[self.flux],
                                                size=10, pen=pg.mkPen(width=2, color='w'), brush=pg.mkBrush(*self.filter.color))
        else:
            self.scatter, self.errorbar = None, None
        self.x = self.filter.data.x[:]
        self.calc_weight()
        self.range = self.filter.range

    def calc_weight(self, z=0):
        num = int((np.log(self.filter.range[1]) - np.log(self.filter.range[0])) / 0.001)
        x = np.exp(np.linspace(np.log(self.x[0]), np.log(self.x[-1]), num))
        if 'W' in self.name:
            self.weight = 1
        else:
            self.weight = 1 #np.sqrt(np.sum(self.filter.inter(x)) / np.max(self.filter.data.y))
        #self.weight = 1
        #print('weight:', self.name, self.weight)

class sed_template():
    def __init__(self, name, smooth_window=None, xmin=None, xmax=None, z=0, x=None, y=None):
        self.name = name
        self.load_data(smooth_window=smooth_window, xmin=xmin, xmax=xmax, z=z, x=x, y=y)

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
                    self.x, self.y = np.genfromtxt('data/SDSS/medianQSO.dat', skip_header=2, unpack=True)
                elif self.name == 'HST':
                    self.x, self.y = np.genfromtxt('data/SDSS/hst_composite.dat', skip_header=2, unpack=True)
                elif self.name == 'Slesing':
                    self.x, self.y = np.genfromtxt('data/SDSS/Slesing2016.dat', skip_header=0, unpack=True, usecols=(0, 1))
                elif self.name == 'power':
                    self.x = np.linspace(500, 25000, 1000)
                    self.y = np.power(self.x / 2500, -1.9)
                    smooth_window = None
                elif self.name == 'composite':
                    if 1:
                        self.x, self.y = np.genfromtxt('data/SDSS/QSO_composite.dat', skip_header=0, unpack=True)
                        self.x, self.y = self.x[self.x > 0], self.y[self.x > 0]
                    else:
                        self.template_qso = np.genfromtxt('data/SDSS/Slesing2016.dat', skip_header=0, unpack=True)
                        self.template_qso = self.template_qso[:,
                                            np.logical_or(self.template_qso[1] != 0, self.template_qso[2] != 0)]
                        if 0:
                            x = self.template_qso[0][-1] + np.arange(1, int((25000 - self.template_qso[0][
                                -1]) / 0.4)) * 0.4
                            y = np.power(x / 2500, -1.9) * 6.542031
                            self.template_qso = np.append(self.template_qso, [x, y, y / 10], axis=1)
                        else:
                            if 1:
                                data = np.genfromtxt('data/SDSS/QSO1_template_norm.sed', skip_header=0, unpack=True)
                                data[0] = ac.c.cgs.value() / 10 ** data[0] * 1e8
                                print(data[0])
                                m = (data[0] > self.template_qso[0][-1]) * (data[0] < self.ero_tempmax)
                                self.template_qso = np.append(self.template_qso, [data[0][m],
                                                                                  data[1][m] * self.template_qso[1][
                                                                                      -1] / data[1][m][0],
                                                                                  data[1][m] / 30], axis=1)
                            else:
                                data = np.genfromtxt('data/SDSS/Richards2006.dat', skip_header=0, unpack=True)
                                m = (data[0] > self.template_qso[0][-1]) * (data[0] < self.ero_tempmax)
                                self.template_qso = np.append(self.template_qso,
                                                              [data[0][m], data[1][m] * self.template_qso[1][-1] /
                                                               data[1][m][0], data[1][m] / 30], axis=1)
                            x = self.template_qso[0][0] + np.linspace(
                                int((self.ero_tempmin - self.template_qso[0][0]) / 0.4), -1) * 0.4
                            y = np.power(x / self.template_qso[0][0], -1.0) * self.template_qso[1][0]
                            self.template_qso = np.append([x, y, y / 10], self.template_qso, axis=1)

            elif self.name in ['S0', 'Sa', 'Sb', 'Sc', 'Sd', 'Sdm', 'Ell2', 'Ell5', 'Ell13', 'Sey2', 'Sey18']:
                if 0:
                    if isinstance(self.template_name_gal, int):
                        f = fits.open(f"data/SDSS/spDR2-0{23 + self.template_name_gal}.fit")
                        self.template_gal = [
                            10 ** (f[0].header['COEFF0'] + f[0].header['COEFF1'] * np.arange(f[0].header['NAXIS1'])),
                            f[0].data[0]]

                        if smooth_window is not None:
                            self.template_gal[1] = smooth(self.template_gal[1], window_len=smooth_window,
                                                          window='hanning', mode='same')
                self.x, self.y = np.genfromtxt(f'data/SDSS/{self.name}_template_norm.sed', unpack=True)

                        # mask = (self.template_gal[0] > self.ero_tempmin) * (self.template_gal[0] < self.ero_tempmax)
                        # self.template_gal = [self.template_gal[0][mask], self.template_gal[1][mask]]

            elif self.name in ['torus']:
                self.x, self.y = np.genfromtxt(f'data/SDSS/torus.dat', unpack=True)
                self.x, self.y = ac.c.cgs.value / 10 ** self.x[::-1] * 1e8, self.y[::-1]
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
        self.name = name
        self.smooth_window, self.xmin, self.xmax, self.z = smooth_window, xmin, xmax, z
        self.load_data()
        self.data = {}

    def load_data(self):
        self.models = []
        if self.name in ['bbb']:
            self.models.append(sed_template('composite', smooth_window=self.smooth_window, xmin=self.xmin, xmax=self.xmax, z=self.z))
            self.values = [0]

        if self.name in ['tor']:
            torus = np.load('C:/science/programs/AGNfitter/models/TORUS/silva_v1_files.npz')
            for i in range(len(torus['arr_0'])):
                self.models.append(sed_template(self.name, smooth_window=self.smooth_window, xmin=self.xmin, xmax=self.xmax, z=self.z,
                                                x=ac.c.cgs.value / 10 ** torus['arr_1'][i][::-1] * 1e8,
                                                y=(torus['arr_2'][i] * 10 ** (2 * torus['arr_1'][i]) * 1e41)[::-1]
                                                ))
            self.values = torus['arr_0']

        if self.name in ['host']:
            self.values = ['S0', 'Sa', 'Sb', 'Sc', 'Ell2', 'Ell5', 'Ell13'] #['S0', 'Sa', 'Sb', 'Sc', 'Sd', 'Sdm', 'Ell2', 'Ell5', 'Ell13', 'Sey2', 'Sey18']
            for n in self.values:
                self.models.append(sed_template(n, smooth_window=self.smooth_window, xmin=self.xmin, xmax=self.xmax, z=self.z))

        if self.name in ['gal']:
            self.values = []
            with open('C:/science/erosita/UV_Xray/AGNfitter/bc03_275templates.pickle', 'rb') as f:
                l = pickle.load(f)
                self.tau = pickle.load(f)
                self.tg = pickle.load(f)
                SED = pickle.load(f)
                #print(tau.shape, tg.shape, SED.shape)
                for i in range(len(self.tau)):
                    for k in range(len(self.tg)):
                        self.values.append([self.tau[i], self.tg[k]])
                        self.models.append(sed_template('gal', smooth_window=self.smooth_window, xmin=self.xmin, xmax=self.xmax, z=self.z, x=l.value, y=SED[k, i, :].value * 1e5))
            self.n_tau = self.tau.shape[0]
            self.n_tg = self.tg.shape[0]

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
            #print('get_model_ind:', params['host_tau'].value, np.log10(self.tau.value), np.argmin(np.abs((params['host_tau'].value - np.log10(self.tau.value)))))
            #ind = np.argmin(np.abs((params['host_tau'].value - self.tau.value))) * self.n_tg + np.argmin(np.abs((params['host_tg'].value - np.log10(self.tg.value))))
            #print(params['host_tau'].value, self.values[ind][0], np.log10(self.tau.value))
            #print(params['host_tg'].value, self.values[ind][1], self.tg.value)
            return np.argmin(np.abs((params['host_tau'].value - self.tau.value))) * self.n_tg + np.argmin(np.abs((params['host_tg'].value - np.log10(self.tg.value))))

        if self.name == 'tor':
            return np.argmin(np.abs((params['tor_type'].value - np.arange(len(self.values)))))

class ErositaWidget(QWidget):
    def __init__(self, parent):
        super(ErositaWidget, self).__init__()
        self.parent = parent
        self.setStyleSheet(open('config/styles.ini').read())
        self.setGeometry(0, 0, 1920, 1080)

        self.initData()
        self.initGUI()
        self.loadTable(recalc=True)
        self.updateData()
        self.show()
        #self.addSDSSQSO()
        #self.addCustom()

    def initData(self):
        self.opts = {'ero_x_axis': str, 'ero_y_axis': str, 'ero_c_axis': str, 'ero_lUV': float, 'ero_tempmin': float, 'ero_tempmax': float,
                     }
        for opt, func in self.opts.items():
            print(opt, self.parent.options(opt), func(self.parent.options(opt)))
            setattr(self, opt, func(self.parent.options(opt)))

        self.axis_list = ['z', 'F_X_int', 'F_X', 'DET_LIKE_0', 'F_UV', 'u-b', 'r-i', 'FIRST_FLUX', 'R', 'Av_gal',
                          'L_X', 'L_UV', 'L_UV_corr', 'L_UV_corr_host_photo',
                          'Av_int', 'Av_int_host', 'Av_int_photo', 'Av_int_host_photo',
                          'chi2_av', 'chi2_av_host', 'chi2_av_photo', 'chi2_av_host_photo',
                          'Av_host', 'Av_host_photo', 'f_host', 'f_host_photo', 'L_host', 'L_host_photo', 'r_host',
                          'SDSS_photo_scale', 'SDSS_photo_slope', 'SDSS_var']
        self.axis_info = {'z': [lambda x: x, 'z'],
                          'F_X_int': [lambda x: np.log10(x), 'log (F_X_int, erg/s/cm2)'],
                          'F_X': [lambda x: np.log10(x), 'log (F_X, erg/s/cm2/Hz)'],
                          'DEL_LIKE_0': [lambda x: np.log10(x), 'log (Xray detection lnL)'],
                          'F_UV': [lambda x: np.log10(x), 'log (F_UV, erg/s/cm2/Hz)'],
                          'L_X': [lambda x: np.log10(x), 'log (L_X, erg/s/Hz)'],
                          'L_UV': [lambda x: np.log10(x), 'log (L_UV, erg/s/Hz)'],
                          'L_UV_corr': [lambda x: np.log10(x), 'log (L_UV corrected, erg/s/Hz)'],
                          'L_UV_corr_host_photo': [lambda x: np.log10(x), 'log (L_UV corrected, erg/s/Hz) with host and photometry'],
                          'u-b': [lambda x: x, 'u - b'],
                          'r-i': [lambda x: x, 'r - i'],
                          'FIRST_FLUX': [lambda x: np.log10(np.abs(x)), 'log (FIRST flux, mJy)'],
                          'R': [lambda x: np.log10(x), 'log R'],
                          'Av_gal': [lambda x: x, 'Av (galactic)'],
                          'Av_int': [lambda x: x, 'Av (intrinsic)'],
                          'Av_int_host': [lambda x: x, 'Av (intrinsic) with host'],
                          'Av_int_photo': [lambda x: x, 'Av (intrinsic) with photometry'],
                          'Av_int_host_photo': [lambda x: x, 'Av (intrinsic) with host and photometry'],
                          'chi2_av': [lambda x: np.log10(x), 'log chi^2 extinction fit'],
                          'chi2_av_host': [lambda x: np.log10(x), 'log chi^2 extinction fit with host'],
                          'chi2_av_photo': [lambda x: np.log10(x), 'log chi^2 extinction fit with photometry'],
                          'chi2_av_host_photo': [lambda x: np.log10(x), 'log chi^2 extinction fit with host and photometry'],
                          'Av_host': [lambda x: x, 'Av (host)'],
                          'Av_host_photo': [lambda x: x, 'Av (host) with photometry'],
                          'f_host': [lambda x: x, 'Galaxy fraction (from my fit)'],
                          'f_host_photo': [lambda x: x, 'Galaxy fraction (from my fit) with photometry'],
                          'L_host': [lambda x: np.log10(x), 'log (L_host, L_sun)'],
                          'L_host_photo': [lambda x: np.log10(x), 'log (L_host, L_sun) with photometry'],
                          'r_host': [lambda x: x, 'Galaxy fraction (from Rakshit)'],
                          'SDSS_photo_scale': [lambda x: x, 'Difference between spectrum and photometry in SDSS'],
                          'SDSS_photo_slope': [lambda x: x, 'Difference in slope between spectrum and photometry in SDSS'],
                          'SDSS_var': [lambda x: np.log10(x), 'log Variability at 2500A from SDSS'],
                          }
        self.df = None
        self.ind = None
        self.corr_status = 0
        self.ext = {}
        self.filters = {}

        self.template_name_qso = 'composite'
        self.template_name_gal = 0
        self.host_templates = self.parent.options('ero_hosts').split()
        self.filter_names = self.parent.options('ero_filters').split()
        # self.template_name_gal = "Sb_template_norm" #"Sey2_template_norm"
        # self.host_templates = ['S0', 'Sa', 'Sb', 'Sc', 'Ell2', 'Ell5', 'Ell13']

    def initGUI(self):
        area = DockArea()
        layout = QVBoxLayout()
        layout.addWidget(area)

        d1 = Dock("Plot", size=(900, 900))
        self.dataPlot = dataPlot(self, [self.ero_x_axis, self.ero_y_axis])
        d1.addWidget(self.dataPlot)
        d2 = Dock("Panel", size=(900, 100))

        d2.addWidget(self.initPanel())
        d2.hideTitleBar()
        d3 = Dock("Table", size=(1000, 1000))
        self.ErositaTable = QSOlistTable(self.parent, 'Erosita', folder=os.path.dirname(self.parent.ErositaFile))
        d3.addWidget(self.ErositaTable)
        area.addDock(d1, 'top')
        area.addDock(d2, 'bottom', d1)
        area.addDock(d3, 'right')

        self.setLayout(layout)

    def initPanel(self):
        widget = QWidget()
        layout = QVBoxLayout()

        l = QHBoxLayout()

        xaxis = QLabel('x axis:')
        xaxis.setFixedSize(40, 30)
        self.x_axis = QComboBox()
        self.x_axis.setFixedSize(140, 30)
        self.x_axis.addItems(self.axis_list)
        self.x_axis.setCurrentText(self.ero_x_axis)
        self.x_axis.currentIndexChanged.connect(partial(self.axisChanged, 'x_axis'))

        yaxis = QLabel('y axis:')
        yaxis.setFixedSize(40, 30)
        self.y_axis = QComboBox()
        self.y_axis.setFixedSize(140, 30)
        self.y_axis.addItems(self.axis_list)
        self.y_axis.setCurrentText(self.ero_y_axis)
        self.y_axis.currentIndexChanged.connect(partial(self.axisChanged, 'y_axis'))

        caxis = QLabel('color code:')
        caxis.setFixedSize(60, 30)
        self.c_axis = QComboBox()
        self.c_axis.setFixedSize(140, 30)
        self.c_axis.addItems([''] + self.axis_list)
        self.c_axis.setCurrentText(self.ero_c_axis)
        self.c_axis.currentIndexChanged.connect(partial(self.axisChanged, 'c_axis'))

        cmap = QLabel('cmap:')
        cmap.setFixedSize(40, 30)
        self.cmap = QComboBox()
        self.cmap.setFixedSize(70, 30)
        self.cmap.addItems(plt.colormaps())
        self.cmap.setCurrentText('Oranges_r')
        self.cmap.currentIndexChanged.connect(partial(self.updateData, ind=None))

        self.showKDE = QCheckBox("show KDE")
        self.showKDE.setChecked(False)
        self.showKDE.setFixedSize(110, 30)

        cols = QLabel('Cols:')
        cols.setFixedSize(40, 30)

        self.cols = ComboMultipleBox(self, name='table')

        l.addWidget(xaxis)
        l.addWidget(self.x_axis)
        l.addWidget(QLabel('   '))
        l.addWidget(yaxis)
        l.addWidget(self.y_axis)
        l.addWidget(caxis)
        l.addWidget(self.c_axis)
        l.addWidget(cmap)
        l.addWidget(self.cmap)
        l.addWidget(self.showKDE)
        l.addStretch()
        l.addWidget(cols)
        l.addWidget(self.cols)
        layout.addLayout(l)

        l = QHBoxLayout()

        lambdaUV = QLabel('lambda UV:')
        lambdaUV.setFixedSize(80, 30)

        self.lambdaUV = QLineEdit()
        self.lambdaUV.setFixedSize(60, 30)
        self.lambdaUV.setText('{0:6.1f}'.format(self.ero_lUV))
        self.lambdaUV.textEdited.connect(self.lambdaUVChanged)

        calcFUV = QPushButton('Calc FUV')
        calcFUV.setFixedSize(100, 30)
        calcFUV.clicked.connect(self.calc_FUV)

        calcLum = QPushButton('Calc Luminosities')
        calcLum.setFixedSize(130, 30)
        calcLum.clicked.connect(self.calc_Lum)

        correlate = QPushButton('Correlate')
        correlate.setFixedSize(120, 30)
        correlate.clicked.connect(self.correlate)

        robust = QLabel('+ robust:')
        robust.setFixedSize(60, 30)
        self.robust = QComboBox(self)
        self.robust.addItems(['None', 'RANSAC', 'Huber'])
        self.robust.setCurrentText('RANSAC')
        self.robust.setFixedSize(100, 30)

        cats = QLabel('+ cat.:')
        cats.setFixedSize(40, 30)
        self.extcat = ComboMultipleBox(self, name='extcat')
        self.extcat.addItems(['Risaliti2015'])
        self.extcat.setFixedSize(120, 30)

        l.addWidget(lambdaUV)
        l.addWidget(self.lambdaUV)
        l.addWidget(QLabel(' '))
        l.addWidget(calcFUV)
        l.addWidget(calcLum)
        l.addWidget(correlate)
        l.addWidget(robust)
        l.addWidget(self.robust)
        l.addWidget(cats)
        l.addWidget(self.extcat)
        l.addStretch()

        layout.addLayout(l)

        l = QHBoxLayout()

        lambdaUV = QLabel('   ')
        lambdaUV.setFixedSize(130, 30)

        calcExtGal = QPushButton('Av galactic')
        calcExtGal.setFixedSize(100, 30)
        calcExtGal.clicked.connect(self.calc_Av_gal)

        calcExt = QPushButton('Calc Ext:')
        calcExt.setFixedSize(80, 30)
        calcExt.clicked.connect(partial(self.calc_ext, ind=None, gal=None))

        self.method = QComboBox(self)
        self.method.addItems(['leastsq', 'least_squares', 'nelder', 'annealing', 'emcee'])
        self.method.setFixedSize(120, 30)
        self.method.setCurrentText('emcee')

        self.numSteps = QLineEdit()
        self.numSteps.setFixedSize(80, 30)
        self.numSteps.setText("100")

        self.plotExt = QCheckBox("plot")
        self.plotExt.setChecked(True)
        self.plotExt.setFixedSize(80, 30)

        self.saveFig = QCheckBox("savefig")
        self.saveFig.setChecked(False)
        self.saveFig.setFixedSize(100, 30)

        l.addWidget(QLabel('                   '))
        l.addWidget(calcExtGal)
        l.addWidget(calcExt)
        l.addWidget(self.method)
        l.addWidget(QLabel('steps:'))
        l.addWidget(self.numSteps)
        l.addWidget(self.plotExt)
        l.addWidget(self.saveFig)
        l.addStretch()

        layout.addLayout(l)

        l = QHBoxLayout()

        self.QSOtemplate = QComboBox(self)
        self.QSOtemplate.addItems(['Slesing', 'VanDen Berk', 'HST', 'power', 'composite'])
        self.QSOtemplate.setFixedSize(120, 30)
        self.QSOtemplate.setCurrentText(self.template_name_qso)

        self.tempmin = QLineEdit()
        self.tempmin.setFixedSize(80, 30)
        self.tempmin.setText('{0:6.1f}'.format(self.ero_tempmin))
        self.tempmin.textEdited.connect(self.tempRangeChanged)

        self.tempmax = QLineEdit()
        self.tempmax.setFixedSize(80, 30)
        self.tempmax.setText('{0:6.1f}'.format(self.ero_tempmax))
        self.tempmax.textEdited.connect(self.tempRangeChanged)

        self.hostExt = QCheckBox("add host:")
        self.hostExt.setChecked(True)
        self.hostExt.setFixedSize(120, 30)

        self.hosts = ComboMultipleBox(self, name='hosts')
        self.hosts.addItems(['S0', 'Sa', 'Sb', 'Sc', 'Sd', 'Sdm', 'Ell2', 'Ell5', 'Ell13', 'Sey2', 'Sey18'])
        self.hosts.setFixedSize(120, 30)
        self.hosts.update()

        l.addWidget(QLabel('                   '))
        l.addWidget(self.QSOtemplate)
        l.addWidget(QLabel('range:'))
        l.addWidget(self.tempmin)
        l.addWidget(QLabel('..'))
        l.addWidget(self.tempmax)
        l.addWidget(self.hostExt)
        l.addWidget(self.hosts)
        l.addStretch()

        layout.addLayout(l)

        l = QHBoxLayout()

        self.addPhoto = QCheckBox("add photometry:")
        self.addPhoto.setChecked(True)
        self.addPhoto.setFixedSize(160, 30)

        self.filters_used = ComboMultipleBox(self, name='filters')
        self.filters_used.addItems(['FUV', 'NUV', 'u', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'K', 'W1', 'W2', 'W3', 'W4'])
        self.filters_used.setFixedSize(120, 30)
        self.filters_used.update()

        compPhoto = QPushButton('Compare photo')
        compPhoto.setFixedSize(120, 30)
        compPhoto.clicked.connect(partial(self.compare_photo, ind=None))

        l.addWidget(QLabel('                   '))
        l.addWidget(self.addPhoto)
        l.addWidget(self.filters_used)
        l.addWidget(compPhoto)
        l.addStretch()

        layout.addLayout(l)

        layout.addStretch()
        widget.setLayout(layout)

        return widget

    def loadTable(self, recalc=False):
        self.df = pd.read_csv(self.parent.ErositaFile)
        try:
            self.df.rename(columns={'Z': 'z', 'ML_FLUX_0_ero': 'F_X_int'}, inplace=True)
        except:
            pass
        if recalc:
            if 'F_X' not in self.df.columns:
                self.df.insert(len(self.df.columns), 'F_X', np.nan)
            gamma = 1.9
            scale = ((2.2 / 2) ** (2 - gamma) - (0.3 / 2) ** (2 - gamma)) / (2 - gamma)
            #print(scale, ((2 * u.eV).to(u.Hz, equivalencies=u.spectral()).value))
            #print((1 + self.df['z']) ** gamma)
            self.df['F_X'] = self.df['F_X_int'] / ((2e3 * u.eV).to(u.Hz, equivalencies=u.spectral()).value) / scale / (1 + self.df['z']) ** (2 - gamma)
            # self.df['ML_FLUX_0'] in erg/s/cm^2
            # self.df['F_X'] in erg/s/cm^2/Hz
        self.cols.addItems(self.df.columns)
        #self.cols.addItems(list(self.df.columns)[40:59] + list(self.df.columns)[:40] + list(self.df.columns)[:59])
        self.shown_cols = self.parent.options('ero_colnames').split()
        self.cols.update()
        self.ErositaTable.setdata(self.df[self.shown_cols].to_records(index=False))
        self.mask = np.zeros(len(self.df['z']), dtype=bool)

    def getData(self):
        self.x_lambda, self.y_lambda = self.axis_info[self.ero_x_axis][0], self.axis_info[self.ero_y_axis][0]
        self.x, self.y, self.SDSScolors = None, None, False
        if self.df is not None:
            if self.ero_x_axis in self.df.columns:
                self.x = self.x_lambda(self.df[self.ero_x_axis])
            elif self.ero_x_axis in ['u-b', 'r-i']:
                self.x = self.x_lambda(self.df['PSFMAG' + str('_ubriz'.index(self.ero_x_axis.split('-')[0]))] - self.df['PSFMAG' + str('_ubriz'.index(self.ero_x_axis.split('-')[1]))])
                self.SDSScolors = True

            if self.ero_y_axis in self.df.columns:
                self.y = self.y_lambda(self.df[self.ero_y_axis])
            elif self.ero_y_axis in ['u-b', 'r-i']:
                self.y = self.y_lambda(self.df['PSFMAG' + str('_ubriz'.index(self.ero_y_axis.split('-')[0]))] - self.df['PSFMAG' + str('_ubriz'.index(self.ero_y_axis.split('-')[1]))])
                self.SDSScolors = True

            if self.ero_c_axis in self.df.columns:
                self.c_lambda = self.axis_info[self.ero_c_axis][0]
                if self.ero_y_axis in self.df.columns:
                    self.c = self.c_lambda(self.df[self.ero_c_axis])
            else:
                self.c = None

    def updateData(self, ind=None):
        print(self.ero_x_axis, self.ero_y_axis, ind)
        self.getData()

        #print(self.x, self.y)
        if self.x is not None and self.y is not None:
            self.dataPlot.setLabel('bottom', self.axis_info[self.ero_x_axis][1])
            self.dataPlot.setLabel('left', self.axis_info[self.ero_y_axis][1])

            if self.ind is not None:
                self.dataPlot.plotData(x=[self.x[self.ind]], y=[self.y[self.ind]],
                                       name='clicked', color=(255, 3, 62), size=20, rescale=False)

            if ind is None:
                self.dataPlot.plotData(x=self.x, y=self.y, c=self.c)

                if not self.SDSScolors:
                    for name in self.ext.keys():
                        self.dataPlot.plotData(x=self.x_lambda(self.ext[name][self.ero_x_axis]),
                                               y=self.y_lambda(self.ext[name][self.ero_y_axis]),
                                               name=name, color=(200, 150, 50))

            #if np.sum(self.mask) > 0:
            self.dataPlot.plotData(x=self.x[self.mask], y=self.y[self.mask],
                                   name='selected', color=(100, 255, 30), rescale=False)

    def addExternalCatalog(self, name, show=True):
        if name == 'Risaliti2015':
            if show:
                self.ext[name] = np.genfromtxt(os.path.dirname(self.parent.ErositaFile) + '/Risaliti2015/table2.dat',
                                     names=['name', 'ra', 'dec', 'z', 'L_UV', 'L_X', 'F_UV', 'eF_UV', 'F_X', 'eF_X', 'group'])
                for attr in ['L_UV', 'L_X', 'F_UV', 'F_X']:
                    self.ext[name][attr] = 10 ** self.ext[name][attr]

                self.ext[name] = rfn.append_fields(self.ext[name], 'F_X_int', np.empty(self.ext[name].shape[0], dtype='<f4'), dtypes='<f4')
                gamma = 1.9
                self.ext[name]['F_X_int'] = self.ext[name]['F_X'] * (1 + self.ext[name]['z']) ** (2 - gamma) * \
                                            ((2.2 / 2) ** (2 - gamma) - (0.3 / 2) ** (2 - gamma)) / (2 - gamma) * \
                                            ((2e3 * u.eV).to(u.Hz, equivalencies=u.spectral()).value)

        if show:
            self.dataPlot.plotData(x=self.x_lambda(self.ext[name][self.ero_x_axis]),
                                   y=self.y_lambda(self.ext[name][self.ero_y_axis]),
                                   name=name, color=(200, 150, 50))

        else:
            self.dataPlot.removeItem(getattr(self.dataPlot, name))
            del self.ext[name]

    def index(self, x=None, y=None, name=None, ind=None, ext=True):

        if ind is None:
            if x is not None and y is not None:
                ind = np.argmin((x - self.x) ** 2 +
                                (y - self.y) ** 2)
            elif name is not None:
                ind = np.where(self.df['SDSS_NAME'] == name)[0][0]

        print(x, y, name, ind, self.df['SDSS_NAME'][ind])

        if ind is not None:

            self.ind = ind
            if name is None and self.ErositaTable.columnIndex('SDSS_NAME') is not None:
                row = self.ErositaTable.getRowIndex(column='SDSS_NAME', value=self.df['SDSS_NAME'][ind])
                self.ErositaTable.setCurrentCell(row, 0)
                self.ErositaTable.selectRow(row)
                self.ErositaTable.row_clicked(row=row)

            if x is None and self.plotExt.isChecked():
                self.calc_ext(self.ind, plot=self.plotExt.isChecked(), gal=self.hostExt.isChecked())
            else:
                self.set_filters(ind, clear=True)

            for k, f in self.filters.items():
                self.parent.plot.vb.addItem(f.scatter)
                self.parent.plot.vb.addItem(f.errorbar)

            self.updateData(ind=self.ind)

    def set_filters(self, ind, clear=False, names=None):
        self.photo = {}
        if clear:
            for k, f in self.filters.items():
                try:
                    self.parent.plot.vb.removeItem(f.scatter)
                    self.parent.plot.vb.removeItem(f.errorbar)
                except:
                    pass
        self.filters = {}
        if self.addPhoto.isChecked():
            for k in ['FUV', 'NUV']:
                if (names is None and k in self.filter_names) or (names is not None and k in names):
                    if 0:
                        if not np.isnan(self.df[f'{k}mag'][ind]) and not np.isnan(self.df[f'e_{k}mag'][ind]):
                            self.filters[k] = Filter(self, k, value=self.df[f'{k}mag'][ind], err=self.df[f'e_{k}mag'][ind])
                    else:
                        if self.df['GALEX_MATCHED'][ind] and not np.isnan(self.df[f'{k}mag'][ind]) and not np.isnan(self.df[f'e_{k}mag'][ind]):
                            self.filters[k] = Filter(self, k, value=-2.5 * np.log10(np.exp(1.0)) * np.arcsinh(self.df[f'{k}'][ind] / 0.01) + 28.3,
                                                     err=-2.5 * np.log10(np.exp(1.0)) * (np.arcsinh(self.df[f'{k}'][ind] / 0.01) - np.arcsinh((self.df[f'{k}'][ind] + 1 / np.sqrt(self.df[f'{k}_IVAR'][ind])) / 0.01)))
                            #print(-2.5 * np.log10(np.exp(1.0)) * np.arcsinh(self.df[f'{k}'][ind] / 0.01) + 28.3)

                    #self.photo['GALEX'] = self.photo['GALEX'] + [k] if 'GALEX' in self.photo.keys() else [k]
                            self.photo[k] = 'GALEX'

            for i, k in enumerate(['u', 'g', 'r', 'i', 'z']):
                if (names is None and k in self.filter_names) or (names is not None and k in names):
                    self.filters[k] = Filter(self, k, value=self.df[f'PSFMAG{i}'][ind], err=self.df[f'ERR_PSFMAG{i}'][ind])
                    #self.photo['SDSS'] = self.photo['SDSS'] + [k] if 'SDSS' in self.photo.keys() else [k]
                    self.photo[k] = 'SDSS'


            for k in ['J', 'H', 'K']:
                if (names is None and k in self.filter_names) or (names is not None and k in names):
                    if self.df[k + 'RDFLAG'][ind] == 2:
                        self.filters[k + '_2MASS'] = Filter(self, k + '_2MASS', value=self.df[k + 'MAG'][ind], err=self.df['ERR_' + k + 'MAG'][ind], system='Vega')
                        #self.photo['2MASS'] = self.photo['2MASS'] + [k] if '2MASS' in self.photo.keys() else [k]
                        self.photo[k+'_2MASS'] = '2MASS'

            for k in ['Y', 'J', 'H', 'K']:
                if (names is None and k in self.filter_names) or (names is not None and k in names):
                    if self.df['UKIDSS_MATCHED'][ind]:
                        self.filters[k + '_UKIDSS'] = Filter(self, k + '_UKIDSS', system='AB',
                                                             value=22.5 - 2.5 * np.log10(self.df[k + 'FLUX'][ind] / 3.631e-32),
                                                             err=2.5 * np.log10(1 + self.df[k + 'FLUX_ERR'][ind] / self.df[k + 'FLUX'][ind]))
                        #self.photo['UKIDSS'] = self.photo['UKIDSS'] + [k] if 'UKIDSS' in self.photo.keys() else [k]
                        self.photo[k+'_UKIDSS'] = 'UKIDSS'

            for k in ['W1', 'W2', 'W3', 'W4']:
                if (names is None and k in self.filter_names) or k in names:
                    if not np.isnan(self.df[k + 'MAG'][ind]) and not np.isnan(self.df['ERR_' + k + 'MAG'][ind]):
                        self.filters[k] = Filter(self, k, system='Vega',
                                                 value=self.df[k + 'MAG'][ind], err=self.df['ERR_' + k + 'MAG'][ind])
                        #self.photo['WISE'] = self.photo['WISE'] + [k] if 'WISE' in self.photo.keys() else [k]
                        self.photo[k] = 'WISE'

        for k, f in self.filters.items():
            self.filters[k].fit = False
            if f.filter.range[0] > self.ero_tempmin * (1 + self.d['z']) and f.filter.range[1] < self.ero_tempmax * (1 + self.d['z']) and np.isfinite(f.value) and np.isfinite(f.err):
                self.filters[k].fit = True
            #print(k, self.filters[k].fit)

    def select_points(self, x1, y1, x2, y2, remove=False, add=False):
        x1, x2, y1, y2 = np.min([x1, x2]), np.max([x1, x2]), np.min([y1, y2]), np.max([y1, y2])
        mask = (self.x > x1) * (self.x < x2) * (self.y > y1) * (self.y < y2)
        if not add and not remove:
            self.mask = mask
        elif add and not remove:
            self.mask = np.logical_or(self.mask, mask)
        elif not add and remove:
            self.mask = np.logical_and(self.mask, ~mask)

        if np.sum(self.mask) > 20:
            if self.showKDE.isChecked():
                self.d = distr2d(self.x[self.mask], self.y[self.mask])
                for axis in ['x', 'y']:
                    d = self.d.marginalize(axis)
                    print(axis, d.stats(latex=3))

        self.updateData()

    def add_mask(self, name=None):
        if name is not None:
            self.mask = np.logical_or(self.mask, self.df['SDSS_NAME'] == name)
            #print(np.sum(self.mask))

    def extinction(self, x, Av=None):
        """
        Return extinction for provided wavelengths
        Args:
            Av: visual extinction
            x: wavelenghts in Angstrem

        Returns: extinction
        """
        return add_ext(x, z_ext=0, Av=Av, kind='SMC')

        #if Av in [None, 0] or np.sort(x)[2] > 1e5 / 3 or np.sort(x)[-2] < 1e3:
        #    return np.ones_like(x)
        #else:
        #    x1 = x[(x < 1e5 / 3) * (x > 1e3)]
        #    ext = interp1d(x1, G03_SMCBar().extinguish(1e4 / np.asarray(x1, dtype=np.float64), Av=Av), fill_value='extrapolate')
        #    ext = ext(x)
        #    ext[(ext > 1) * (x > 1e5 / 3)] = 1
        #    return ext

    def axisChanged(self, axis):
        self.parent.options('ero_' + axis, getattr(self, axis).currentText())
        setattr(self, 'ero_' + axis, getattr(self, axis).currentText())
        self.updateData()

    def lambdaUVChanged(self):
        self.ero_lUV = float(self.lambdaUV.text())
        self.parent.options('ero_lUV', self.ero_lUV)

    def tempRangeChanged(self):
        for attr in ['tempmin', 'tempmax']:
            setattr(self, "ero_" + attr, float(getattr(self, attr).text()))
            self.parent.options('ero_' + attr, getattr(self, "ero_" + attr))

    def getSDSSind(self, name):
        ind = np.where(self.df['SDSS_NAME'] == name)[0][0]
        return self.df['PLATE'][ind], self.df['MJD'][ind], self.df['FIBERID'][ind]

    def loadSDSS(self, plate, fiber, mjd, Av_gal=np.nan):
        filename = os.path.dirname(self.parent.ErositaFile) + '/spectra/spec-{0:04d}-{2:05d}-{1:04d}.fits'.format(int(plate), int(fiber), int(mjd))
        if os.path.exists(filename):
            qso = fits.open(filename)
            ext = self.extinction(10 ** qso[1].data['loglam'][:], Av_gal) #G03_SMCBar().extinguish(1e4 / np.asarray(10 ** qso[1].data['loglam'][:], dtype=np.float64), Av=Av_gal) if ~np.isnan(Av_gal) else np.ones_like(qso[1].data['loglam'])
            return [10 ** qso[1].data['loglam'][:], qso[1].data['flux'][:] / ext, np.sqrt(1.0 / qso[1].data['ivar'][:]) / ext, qso[1].data['and_mask']]

    def calc_FUV(self):
        k = 0
        if 'F_UV' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'F_UV', np.nan)
        self.df['F_UV'] = np.nan

        name = 'F_UV_{0:4d}'.format(int(float(self.lambdaUV.text())))
        if name not in self.df.columns:
            self.df.insert(len(self.df.columns), name, np.nan)
        self.df[name] = np.nan

        for i, d in self.df.iterrows():
            #print(i)
            if np.isfinite(d['z']) and d['z'] > 3500 / self.ero_lUV - 1:
                spec = self.loadSDSS(d['PLATE'], d['FIBERID'], d['MJD'], Av_gal=d['Av_gal'])
                if spec is not None:
                    mask = (spec[0] > (self.ero_lUV - 10) * (1 + d['z'])) * (
                                spec[0] < (self.ero_lUV + 10) * (1 + d['z'])) * (spec[3] == 0)
                    if np.sum(mask) > 0 and not np.isnan(np.mean(spec[1][mask])) and not np.mean(spec[2][mask]) == 0:
                        k += 1
                        nm = np.nanmean(spec[1][mask]) * 1e-17 * u.erg / u.cm ** 2 / u.AA / u.s
                        wm = np.average(spec[1][mask], weights=spec[2][mask]) * 1e-17 * u.erg / u.cm ** 2 / u.AA / u.s
                        # print(nm, wm)
                        self.df.loc[i, 'F_UV'] = nm.to(u.erg / u.cm ** 2 / u.s / u.Hz, equivalencies=u.spectral_density(
                            self.ero_lUV * u.AA * (1 + d['z']))).value / (1 + d['z'])
                        self.df.loc[i, name] = nm.to(u.erg / u.cm ** 2 / u.s / u.Hz, equivalencies=u.spectral_density(
                            self.ero_lUV * u.AA * (1 + d['z']))).value
                        # self.df['F_UV'] in erg/s/cm^2/Hz

        self.updateData()
        self.save_data()

    def calc_radioLoudness(self):

        if 'R' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'R', np.nan)
        self.df['R'] = np.nan

        mask = self.df['FIRST_MATCHED'].to_numpy() == 0
        self.df.loc[mask, 'R'] = 1

        mask = self.df['FIRST_MATCHED'].to_numpy() == 1
        self.df.loc[mask, 'R'] = self.df['FIRST_FLUX'] * 1e-3 / self.df['F_UV'] * 1e-23

    def calc_Lum(self):

        self.calc_radioLoudness()

        mask = np.isfinite(self.df['z'].to_numpy())
        dl = Planck15.luminosity_distance(self.df['z'].to_numpy()).to('cm')

        if 'L_X' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'L_X', np.nan)
        self.df['L_X'] = np.nan

        if 'L_UV' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'L_UV', np.nan)
        self.df['L_UV'] = np.nan

        if 'L_UV_corr' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'L_UV_corr', np.nan)
        self.df['L_UV_corr'] = np.nan

        if 'L_UV_corr_host_photo' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'L_UV_corr_host_photo', np.nan)
        self.df['L_UV_corr'] = np.nan

        name = 'L_UV_{0:4d}'.format(int(float(self.lambdaUV.text())))
        if name not in self.df.columns:
            self.df.insert(len(self.df.columns), name, np.nan)
        self.df[name] = np.nan

        if 'F_X' in self.df.columns:
            self.df.loc[mask, 'L_X'] = 4 * np.pi * dl[mask] ** 2 * self.df['F_X'][mask] #/ (1 + self.df['Z'])

        if 'F_UV' in self.df.columns:
            self.df.loc[mask, 'L_UV'] = 4 * np.pi * dl[mask] ** 2 * self.df['F_UV'][mask] #/ (1 + self.df['Z'])
            self.df.loc[mask, name] = 4 * np.pi * dl[mask] ** 2 * self.df[name.replace('L', 'F')][mask] / (1 + self.df['z'][mask])

        for attr in ['', '_host_photo']:
            if 'Av_int' + attr in self.df.columns:
                for i, d in self.df.iterrows():
                    if mask[i] and np.isfinite(d['Av_int' + attr]):
                        #print(i, dl[i].value)
                        #print(self.df.loc[i, 'L_UV_corr' + attr])
                        self.df.loc[i, 'L_UV_corr' + attr] = 4 * np.pi * dl[i].value ** 2 * d[name.replace('L', 'F')] / (1 + d['z']) / self.extinction(float(self.lambdaUV.text()), Av=d['Av_int' + attr])
                    else:
                        print(i)

        self.save_data()
        self.updateData()

    def compare_photo(self, ind=None):

        for attr in ['SDSS_photo_scale', 'SDSS_photo_slope', 'SDSS_var']:
            if attr not in self.df.columns:
                self.df.insert(len(self.df.columns), attr, np.nan)

        if np.sum(self.mask) == 0:
            self.mask = np.ones(len(self.df['SDSS_NAME']), dtype=bool)

        print('compare photo:', ind)

        if ind is not None:
            fig, ax = plt.subplots()

        for i, d in self.df.iterrows():
            if ((ind is None and self.mask[i]) or (ind is not None and i == int(ind))):
                print(i)
                spec = self.loadSDSS(d['PLATE'], d['FIBERID'], d['MJD'], Av_gal=d['Av_gal'])

                self.set_filters(i, clear=True, names=['g', 'r', 'i', 'z'])

                x, data, err = [], [], []
                for f in self.filters.values():
                    if f.value != 0:
                        x.append(np.log10(f.filter.l_eff))
                        data.append(f.value - f.filter.get_value(x=spec[0], y=spec[1], err=spec[2], mask=np.logical_not(spec[3])))
                        err.append(f.err)
                        if ind is not None:
                            ax.errorbar(np.log10(f.filter.l_eff), f.value, yerr=f.err)
                            ax.scatter(np.log10(f.filter.l_eff), f.filter.get_value(x=spec[0], y=spec[1], err=spec[2],
                                                                                    mask=np.logical_not(spec[3])))

                if len(x) > 0:
                    x, data, err = np.asarray(x), np.asarray(data), np.asarray(err)
                    err[err == 0] = 0.5
                    print(x, data, err)
                    ma = np.ma.MaskedArray(data, mask=np.isnan(data))
                    if ma.shape[0] - np.sum(ma.mask) > 1:
                        self.df.loc[i, 'SDSS_photo_scale'] = np.ma.average(ma, weights=err)
                        print(self.df.loc[i, 'SDSS_photo_scale'])
                        model = linear_model.LinearRegression().fit(x[~ma.mask].reshape((-1, 1)), ma[~ma.mask], sample_weight=err[~ma.mask])
                        self.df.loc[i, 'SDSS_photo_slope'] = model.coef_[0] / 2.5
                        if not np.isnan(d['z']):
                            print(model.predict(np.asarray(np.log10(2500 * (1 + d['z']))).reshape((-1, 1))))
                            self.df.loc[i, 'SDSS_var'] = 10 ** (model.predict(np.asarray(np.log10(2500 * (1 + d['z']))).reshape((-1, 1))) / 2.5)[0]
                            print(self.df.loc[i, 'SDSS_var'])
                            #print(model.intercept_, model.coef_[0], model.score(x.reshape((-1, 1)), data, sample_weight=err))

        if np.sum(self.mask) == len(self.df['SDSS_NAME']):
            self.mask = np.zeros(len(self.df['SDSS_NAME']), dtype=bool)

        self.save_data()
        self.updateData()

    def calc_Av_gal(self):
        if 'Av_gal' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'Av_gal', np.nan)
        self.df['Av_gal'] = np.nan

        m = sfdmap.SFDMap(self.parent.SFDMapPath)

        for i, d in self.df.iterrows():
            self.df.loc[i, 'Av_gal'] = 3.1 * m.ebv(d['RA'], d['DEC'])

        self.save_data()
        self.updateData()

    def calc_host_luminosity(self, ind=None, norm=None):
        """
        Calculate the lumonicity of the host galaxy
        Args:
            ind: index of the QSO, if None, than run for all

        Returns:

        """
        self.df.loc[i, 'Av_host' + '_photo' * self.addPhoto.isChecked()]

        for i, d in self.df.iterrows():
            if ((ind is None) or (ind is not None and i == int(ind))) and np.isfinite(d['z']):
                self.df.loc[i, 'Av_host' + '_photo' * self.addPhoto.isChecked()]

    def spec_model(self, params, x):
        return None

    def model(self, params, x, dtype, mtype='total'):
        model = np.zeros_like(self.models['bbb'].data[dtype][0])
        if mtype in ['total', 'bbb']:
            model = self.models['bbb'].data[dtype][0] * params.valuesdict()['bbb_norm'] * self.extinction(x / (1 + self.d['z']), Av=params.valuesdict()['Av'])
        if mtype in ['total', 'tor'] and params.valuesdict()['tor_type'] > -1:
            model += self.models['tor'].data[dtype][self.models['tor'].get_model_ind(params)] * params.valuesdict()['tor_norm']
        if mtype in ['total', 'gal'] and params.valuesdict()['host_tau'] > -1 and params.valuesdict()['host_tg'] > -1:
            model += self.models['gal'].data[dtype][self.models['gal'].get_model_ind(params)] * params.valuesdict()['host_norm'] * self.extinction(x / (1 + self.d['z']), Av=params.valuesdict()['host_Av'])
        #if params.valuesdict()['host_type'] > -1:
        #    model += self.models['host'].data[kind][params.valuesdict()['host_type']] * params.valuesdict()['host_norm'] * self.extinction(x / (1 + d['z']), Av=params.valuesdict()['host_Av'])
        return model

    def model_emcee(self, params, x, dtype, mtype='total'):
        self.photo['spec'] = 'spec'
        alpha = 10 ** params['alpha_' + self.photo[dtype]].value if self.photo[dtype] not in ['WISE', 'spec'] else 1
        #print(dtype, self.photo[dtype], alpha)

        model = np.zeros_like(self.models['bbb'].data[dtype][0])
        if mtype in ['total', 'bbb']:
            model = self.models['bbb'].data[dtype][0] * alpha * params.valuesdict()['bbb_norm'] * self.extinction(x / (1 + self.d['z']), Av=params.valuesdict()['Av'])
        if mtype in ['total', 'tor'] and params.valuesdict()['tor_type'] > -1:
            model += self.models['tor'].data[dtype][self.models['tor'].get_model_ind(params)] * params.valuesdict()['tor_norm']
        if mtype in ['total', 'gal'] and params.valuesdict()['host_tau'] > -1 and params.valuesdict()['host_tg'] > -1:
            model += self.models['gal'].data[dtype][self.models['gal'].get_model_ind(params)] * params.valuesdict()['host_norm'] * self.extinction(x / (1 + self.d['z']), Av=params.valuesdict()['host_Av'])
        #if params.valuesdict()['host_type'] > -1:
        #    model += self.models['host'].data[kind][params.valuesdict()['host_type']] * params.valuesdict()['host_norm'] * self.extinction(x / (1 + d['z']), Av=params.valuesdict()['host_Av'])
        return model

    def calc_lum(self, params, wave=1700):
        if isinstance(wave, (int, float)):
            f = np.log10(self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(wave) * \
                         params.valuesdict()['host_norm'] * self.extinction(wave, Av=params.valuesdict()['host_Av']) * \
                         1e-17 * 4 * np.pi * Planck15.luminosity_distance(self.d['z']).to('cm').value ** 2)
            #print(f)
        elif wave in ['bol', 'total']:
            wave = self.models['gal'].models[self.models['gal'].get_model_ind(params)].x
            f = np.trapz(self.models['gal'].models[self.models['gal'].get_model_ind(params)].y * params.valuesdict()['host_norm'] * self.extinction(wave, Av=params.valuesdict()['host_Av']),
                         x=self.models['gal'].models[self.models['gal'].get_model_ind(params)].x)
            f *= 1e-17 * 4 * np.pi * Planck15.luminosity_distance(self.d['z']).to('cm').value ** 2
        return f

    def fcn2min(self, params):
        # print(params)
        # t = Timer()
        chi = (self.model(params, self.sm[0], 'spec') - self.sm[1]) / self.sm[2]
        # t.time('spec')
        for k, f in self.filters.items():
            if self.filters[k].fit:
                chi = np.append(chi, [f.weight / f.err * (f.value - f.filter.get_value(x=f.x, y=self.model(params, f.x, k)))])
                # t.time(f.name)
        return chi

    def fcn2min_emcee(self, params):
        #params['host_L'].value = self.calc_lum(params)[0]
        ret = self.ln_priors(params)
        if ret != -np.inf:
            # print(.5 * np.sum(np.power(fcn2min_gal(params), 2)))
            ret -= .5 * np.sum(np.power(self.ln_like(params), 2))
        return ret

    def anneal_priors(self, params):
        return 10 * self.lum_prior(params) + self.agn_host_prior(params)

    def agn_host_prior(self, params):
        host = self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(1400) * params.valuesdict()['host_norm'] * self.extinction(1400, Av=params.valuesdict()['host_Av'])
        agn = self.models['bbb'].models[0].flux(1400) * params.valuesdict()['bbb_norm'] * self.extinction(1400, Av=params.valuesdict()['Av'])
        #print(host, agn, - host / agn)
        return - host / agn

    def lum_prior(self, params):
        alpha, tmin = -1.2, 0.0001
        C = (alpha + 1) / (gamma(alpha + 2) * (1 - gammainc(alpha + 2, tmin)) - (tmin) ** (alpha + 1) * np.exp(-tmin))
        M_UVs = -20.9 - 1.1 * (self.d['z'] - 1)
        f = self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(1700) * params.valuesdict()['host_norm'] * self.extinction(1700, Av=params.valuesdict()['host_Av'])
        #print('f:', f)
        M_UV = -2.5 * np.log10(f * 1e-17 / 3.77e-8) - 5 * np.log10(Planck15.luminosity_distance(self.d['z']).to('pc').value) + 5
        if f == 0 or M_UV > M_UVs + 10:
            M_UV = M_UVs + 10
        #print('M_UV:', M_UV)
        LLs = 10 ** (-0.4 * (M_UV - M_UVs))
        #print(LLs ** (alpha) * np.exp(-LLs))
        #print(C)
        #print(np.log(C * LLs ** (-1.2) * np.exp(-LLs)))
        return np.log(C * LLs ** (-1.2) * np.exp(-LLs))

    def ln_priors(self, params):
        prior = self.lum_prior(params)
        #print(prior)
        for p in params.values():
            if p.value < p.min or p.value > p.max:
                return -np.inf
            if p.name == 'host_tg':
                prior -= 10 ** (p.max - p.value) - 1
            if p.name == 'host_Av':
                prior -= (p.value - p.min) * 2
            if 'alpha' in p.name:
                prior -= .5 * (p.value / params['sigma'].value) ** 2 + np.log(params['sigma'].value)
            if p.name == 'sigma':
                prior -= .5 * ((p.value - 0.2) / 0.03) ** 2
            #print(p.name, prior)
        return prior

    def ln_like(self, params):
        # print(params)
        # t = Timer()
        chi = (self.model_emcee(params, self.sm[0], 'spec') - self.sm[1]) / self.sm[2]
        # t.time('spec')
        for k, f in self.filters.items():
            if self.filters[k].fit:
                chi = np.append(chi, [f.weight / f.err * (f.value - f.filter.get_value(x=f.x, y=self.model_emcee(params, f.x, k)))])
                # t.time(f.name)
        return chi

    def emcee_fit(self, params=None, tvary=True, plot=True):
        # import os
        # os.environ["OMP_NUM_THREADS"] = "1"
        # from multiprocessing import Pool

        #print(self.photo)

        new = True if params is None else False

        if params is None:
            norm_bbb = np.nanmean(self.sm[1]) / np.nanmean(self.models['bbb'].data['spec'])
            params = lmfit.Parameters()
            params.add('bbb_norm', value=norm_bbb, min=0, max=1e10)
            params.add('Av', value=0.0, min=-10, max=10)
            params.add('tor_type', value=10, vary=True, min=0, max=self.models['tor'].n - 1)
            params.add('tor_norm', value=norm_bbb / np.max(self.models['bbb'].data['spec'][0]) * np.max(self.models['tor'].data['spec'][params.valuesdict()['tor_type']]), min=0, max=1e10)
            params.add('host_tau', value=0.1, vary=True, min=self.models['gal'].tau[0].value, max=0.5) # self.models['gal'].tau[-1].value)
            params.add('host_tg', value=np.log10(Planck15.age(self.d['z']).to('yr').value), vary=True, min=np.log10(self.models['gal'].tg[0].value), max=np.log10(Planck15.age(self.d['z']).to('yr').value))
            params.add('host_norm', value=np.nanmean(self.sm[1]) / np.nanmean(self.models['gal'].data['spec'][self.models['gal'].get_model_ind(params)]), min=0, max=1e10)
            params.add('host_Av', value=0.1, min=0, max=5.0)

        cov = {'bbb_norm': params['bbb_norm'].value / 3, 'Av': 0.1, 'tor_type': 5, 'tor_norm': params['tor_norm'].value / 3,
               'host_tau': 0.3, 'host_tg': 2, 'host_norm': params['host_norm'].value / 3, 'host_Av': 0.1}

        if not new:
            params['host_Av'].max = 5
            params['host_tau'].value = 0.1
            for k in cov.keys():
                if params[k].stderr is None:
                    params[k].stderr = cov[k]
                if params[k].vary == True:
                    cov[k] = min([cov[k], params[k].stderr])

        #params.add('host_L', value=0, min=0, max=100, vary=False)
        #cov['host_L'] = 0.1

        if tvary:
            params.add('sigma', value=0.2, min=0.0, max=3)
            cov['sigma'] = 0.02
            #params.add('alpha_spec', value=0.0, min=-3, max=3)
            #cov['alpha_spec'] = params['sigma'].value / 10
            #print(self.filters)
            for p in set([v for k, v in self.photo.items() if self.filters[k].fit]):
                if p != 'WISE':
                    params.add('alpha_' + p, value=0, min=-3, max=3)
                    cov['alpha_' + p] = params['sigma'].value

            for p in set(self.photo.values()):
                if p != 'WISE':
                    chi = []
                    for k, f in self.filters.items():
                        if self.photo[k] == p and self.filters[k].fit:
                            chi.append([f.value, f.filter.get_value(x=f.x, y=self.model_emcee(params, f.x, k))])
                    if len(chi) > 0:
                        params['alpha_' + p].value = (np.sum(np.asarray(chi), axis=0)[0] - np.sum(np.asarray(chi), axis=0)[1]) / len(np.sum(np.asarray(chi), axis=0)) / 2.5
        print(params)
        print(cov)

        nwalkers, steps, ndims = 30, int(self.numSteps.text()), len(params)

        pos = np.asarray([params[p].value + np.random.randn(nwalkers) * cov[p] for p in params])
        for k, p in enumerate(params.values()):
            params[p.name].vary = True
            pos[k][pos[k] < p.min] = p.min
            pos[k][pos[k] > p.max] = p.max

        #print(pos)
        #print(pos.shape, nwalkers, )
        result = lmfit.minimize(self.fcn2min_emcee, method='emcee', params=params, progress=True,
                                nwalkers=nwalkers, steps=steps, burn=int(steps / 2), thin=2, pos=pos.transpose(),
                                nan_policy='omit')

        #self.calc_lum(result.flatchain)

        if plot:
            if 0:
                emcee_plot = corner.corner(result.flatchain, labels=[r.replace('_', ' ') for r in result.var_names], truths=list(result.params.valuesdict().values()))
            else:
                c = ChainConsumer()
                #print(np.asarray(result.flatchain))
                #print([r.replace('_', ' ') for r in result.var_names])
                c.add_chain(np.asarray(result.flatchain), parameters=[r.replace('_', ' ') for r in result.var_names])
                c.configure(summary=True, bins=1.4, cloud=True, sigmas=np.linspace(0, 2, 3), smooth=1,
                            colors="#673AB7", shade_alpha=1)
                fig = c.plotter.plot(figsize=(20, 15))

        # >>> statistical determination:
        pars, samples = [r.replace('_', ' ') for r in result.var_names], np.asarray(result.flatchain)
        print(pars)
        print(samples.shape)

        k = int(len(pars)) #samples.shape[1]
        n_hor = int(k ** 0.5)
        n_hor = np.max([n_hor, 2])
        n_vert = k // n_hor + 1 if k % n_hor > 0 else k // n_hor
        n_vert = np.max([n_vert, 2])

        fig, ax = plt.subplots(nrows=n_vert, ncols=n_hor, figsize=(6 * n_vert, 4 * n_hor))
        k = 0
        res = {}
        for i, p in enumerate(pars):
            s = samples[:, i].flatten()
            x = np.linspace(np.min(s), np.max(s), 100)
            kde = gaussian_kde(s)
            d = distr1d(x, kde(x))
            d.dopoint()
            d.dointerval()
            res[p] = a(d.point, d.interval[1] - d.point, d.point - d.interval[0])
            f = np.asarray([res[p].plus, res[p].minus])
            f = int(np.round(np.abs(np.log10(np.min(f[np.nonzero(f)])))) + 1)
            print(d.interval[0], x[1], d.interval[1], x[-2])
            if d.interval[0] <= x[1] and d.interval[1] < x[-2]:
                d.dointerval(conf=0.95)
                print(d.dointerval(conf=0.95), 'u')
                res[p] = a(d.point, d.interval[1] - d.point, np.inf, t='u')
            elif d.interval[1] >= x[-2] and d.interval[0] > x[1]:
                d.dointerval(conf=0.95)
                print(d.dointerval(conf=0.95), 'l')
                res[p] = a(d.point, np.inf, d.point - d.interval[0], t='l')

            print(p, res[p].latex(f=f))
            vert, hor = k // n_hor, k % n_hor
            k += 1
            d.plot(conf=0.683, ax=ax[vert, hor], ylabel='')
            ax[vert, hor].yaxis.set_ticklabels([])
            ax[vert, hor].yaxis.set_ticks([])
            ax[vert, hor].text(.05, .9, str(p).replace('_', ' '), ha='left', va='top', transform=ax[vert, hor].transAxes)
            ax[vert, hor].text(.95, .9, res[p].latex(f=f), ha='right', va='top', transform=ax[vert, hor].transAxes)
            #ax[vert, hor].set_title(pars[i].replace('_', ' '))

        return result, -np.max(result.lnprob), res

    def anneal_fit(self, params=None):

        #print(self.models['gal'].tg)
        #print(self.models['gal'].tau)
        if params is None:
            norm_bbb = np.nanmean(self.sm[1]) / np.nanmean(self.models['bbb'].data['spec'])
            params = lmfit.Parameters()
            params.add('bbb_norm', value=norm_bbb, min=0, max=1e10)
            params.add('Av', value=0.0, min=-10, max=10)
            params.add('tor_type', value=10, vary=False, min=0, max=self.models['tor'].n - 1)
            params.add('tor_norm', value=norm_bbb / np.max(self.models['bbb'].data['spec'][0]) * np.max(self.models['tor'].data['spec'][params.valuesdict()['tor_type']]), min=0, max=1e10)
            if 0:
                params.add('host_type', value=0, vary=False, min=0, max=self.models['host'].n - 1)
            else:
                print('age:', np.log10(Planck15.age(self.d['z']).to('yr').value))
                params.add('host_tau', value=0.2, vary=False, min=self.models['gal'].tau[0].value, max=self.models['gal'].tau[-1].value)
                params.add('host_tg', value=np.log10(Planck15.age(self.d['z']).to('yr').value), vary=False, min=np.log10(self.models['gal'].tg[0].value), max=np.log10(Planck15.age(self.d['z']).to('yr').value))
            norm_gal = np.nanmean(self.sm[1]) / np.nanmean(self.models['gal'].data['spec'][233])
            params.add('host_norm', value=norm_gal, min=0, max=1e10)
            params.add('host_Av', value=0.1, min=0, max=1.0)

        anneal_pars = OrderedDict([('tor_type', int), ('host_tau', float), ('host_tg', float)])

        def objective(best, anneal_pars, params):
            for i, (p, f) in enumerate(anneal_pars.items()):
                params[p].value = best[i]

            #print(params)
            minner = lmfit.Minimizer(self.fcn2min, params, nan_policy='propagate', calc_covar=True)
            result = minner.minimize(method='leastsq')
            #lmfit.report_fit(result)
            chi = self.fcn2min(result.params)
            #print(np.sum(chi ** 2) / (len(chi) - len(result.params)))
            #print(np.sum(chi ** 2), -10 * self.lum_prior(result.params), -self.anneal_priors(result.params))
            return (np.sum(chi ** 2) - self.anneal_priors(result.params)) / (len(chi) - len(result.params)), result

        def simulated_annealing(objective, params, anneal_pars, n_iterations=100, temp=10000):
            # generate an initial point
            best = [f(params[p].value) for p, f in anneal_pars.items()]
            # evaluate the initial point
            best_eval, res = objective(best, anneal_pars, params)
            print(best, best_eval)
            # current working solution
            curr, curr_eval = best, best_eval
            # run the algorithm
            for i in range(n_iterations):
                # take a step
                #candidate = curr + randn(len(bounds)) * step_size
                candidate = [f(params[p].value + np.random.randn() / 6 * (params[p].max - params[p].min)) for p, f in anneal_pars.items()]
                candidate = [p.min * (c < p.min) + p.max * (c > p.max) + c * ((c >= p.min) * (c <= p.max)) for c, p in zip(candidate, [params[p] for p in anneal_pars.keys()])]
                # evaluate candidate point
                candidate_eval, res = objective(candidate, anneal_pars, params)
                # check for new best solution
                if candidate_eval < best_eval:
                    # store new best point
                    best, best_eval = candidate, candidate_eval
                    # report progress
                    print('>%d f(%s) = %.5f' % (i, best, best_eval))
                # difference between candidate and current point evaluation
                diff = candidate_eval - curr_eval
                # calculate temperature for current epoch
                t = temp / float((i + 1) / 3)
                # calculate metropolis acceptance criterion
                metropolis = np.exp(-diff / t)
                # check if we should keep the new point
                if diff < 0 or np.random.rand() < metropolis:
                    # store the new current point
                    curr, curr_eval = candidate, candidate_eval
            #print('best:', best)
            return objective(best, anneal_pars, params)

        chi2_min, result = simulated_annealing(objective, params, anneal_pars)
        #print(result.params)
        print(chi2_min, lmfit.report_fit(result))
        return result, chi2_min

    def calc_ext(self, ind=None, plot=False, gal=True):

        for attr in ['Av_int', 'Av_int_photo', 'Av_int_host', 'Av_int_host_photo', 'f_host', 'f_host_photo',
                     'chi2_av', 'chi2_av_photo', 'chi2_av_host', 'chi2_av_host_photo',
                     'Av_host', 'Av_host_photo', 'host_type', 'host_type_photo', 'L_host', 'L_host_photo']:
            if attr not in self.df.columns:
                self.df.insert(len(self.df.columns), attr, np.nan)

        if np.sum(self.mask) == 0:
            self.mask = np.ones(len(self.df['SDSS_NAME']), dtype=bool)

        if self.saveFig.isChecked():
            plot = True

        if ind is None:
            fmiss = open("temp/av_missed.dat", "w")

        print('calc_ext:', ind)

        if 0 and ind is not None:
            self.compare_photo(ind)

        for i, d in self.df.iterrows():
            if ((ind is None and self.mask[i]) or (ind is not None and i == int(ind))) and np.isfinite(d['z']):
                print(i)
                self.d = d

                spec = self.loadSDSS(d['PLATE'], d['FIBERID'], d['MJD'], Av_gal=d['Av_gal'])

                #print(spec, d['PLATE'], d['FIBERID'], d['MJD'])
                mask = self.calc_mask(spec, z_em=d['z'], iter=3, window=201, clip=2.5)
                if np.sum(mask) > 20:
                    self.set_filters(i, clear=True)
                    self.sm = [np.asarray(spec[0][mask], dtype=np.float64), spec[1][mask], spec[2][mask]]

                    self.models = {}
                    for name in ['bbb', 'tor', 'host', 'gal']:
                        self.models[name] = sed(name=name, xmin=self.ero_tempmin, xmax=self.ero_tempmax, z=self.d['z'])
                        self.models[name].set_data('spec', self.sm[0])
                        for k, f in self.filters.items():
                            if self.filters[k].fit:
                                self.models[name].set_data(k, f.x)

                    fig = self.plot_spec(spec=spec, mask=mask)

                    #print(self.models)

                    norm_bbb = np.nanmean(self.sm[1]) / np.nanmean(self.models['bbb'].data['spec'])
                    #print(norm_bbb,)

                    # brute force grid over host types
                    if 0:
                        results, chi2s = [], []
                        if self.hostExt.isChecked():
                            hosts = [self.models['host'].values.index(host) for host in self.host_templates]
                        else:
                            hosts = [-1]

                        for h in hosts:
                            params = lmfit.Parameters()
                            params.add('bbb_norm', value=norm_bbb, min=0, max=1e10)
                            params.add('Av', value=0.0, min=-10, max=10)
                            params.add('tor_type', value=10, vary=False, min=-1, max=self.models['tor'].n - 1)
                            if any([f in self.filters.keys() for f in ['W3', 'W4']]):
                                params.add('tor_norm', value=norm_bbb / np.max(self.models['bbb'].data['spec'][0]) * np.max(self.models['tor'].data['spec'][params.valuesdict()['tor_type']]), min=0, max=1e10)
                            else:
                                params.add('tor_norm', value=0, vary=False, min=0, max=1e10)

                            params.add('host_type', value=h, vary=False, min=-1, max=self.models['host'].n-1)
                            if h > -1 and any([f in self.filters.keys() for f in ['J', 'H', 'K', 'W1', 'W2']]):
                                params.add('host_norm', value=params.valuesdict()['bbb_norm'] / 100, min=0, max=1e10)
                                params.add('host_Av', value=0.0, min=0, max=5)
                            else:
                                params.add('host_norm', value=0, vary=False, min=0, max=1e10)
                                params.add('host_Av', value=0, vary=False, min=0, max=0.5)
                            minner = lmfit.Minimizer(fcn2min_gal, params, nan_policy='propagate', calc_covar=True)
                            results.append(minner.minimize(method=self.method.currentText()))
                            # lmfit.report_fit(results[-1])
                            chi = fcn2min_gal(results[-1].params)
                            chi2s.append(np.sum(chi ** 2) / (len(chi) - len(results[-1].params)))
                            print(i, "{0:s} Av={1:.3f} b={2:.2f} t={3:.2f} h={4:.5f} Avh={5:.2f} {6:.5f}".format(self.models['host'].values[h], results[-1].params['Av'].value, results[-1].params['bbb_norm'].value, results[-1].params['tor_norm'].value, results[-1].params['host_norm'].value, results[-1].params['host_Av'].value, chi2s[-1]))

                        host_min = np.argmin(chi2s)
                        result, chi2_min = results[host_min], chi2s[host_min]
                        self.df.loc[i, 'Av_int' + '_host' * self.hostExt.isChecked() + '_photo' * self.addPhoto.isChecked()] = result.params['Av'].value
                        self.df.loc[i, 'chi2_av' + '_host' * self.hostExt.isChecked() + '_photo' * self.addPhoto.isChecked()] = chi2_min
                        #self.df.loc[i, 'f_host' + '_photo' * self.addPhoto.isChecked()] = 1 / (1 + (result.params['norm_bbb'].value * np.trapz(bbb.flux(gal.x) * self.extinction(gal.x, Av=result.params['Av'].value), x=gal.x) / (result.params['norm_host'].value * np.trapz(gal.y, x=gal.x))))
                        if self.hostExt.isChecked() and any([f in self.filters.keys() for f in ['J', 'H', 'K', 'W1', 'W2']]):
                            self.template_name_gal = host_min
                            self.df.loc[i, 'Av_host' + '_photo' * self.addPhoto.isChecked()] = result.params['host_Av'].value
                            self.df.loc[i, 'f_host' + '_photo' * self.addPhoto.isChecked()] = 1 / (1 + (result.params['bbb_norm'].value * self.models['bbb'].models[0].flux(5100) * self.extinction(5100, Av=result.params['Av'].value) + result.params['tor_norm'].value * self.models['tor'].models[result.params['tor_type'].value].flux(5100)) / (result.params['host_norm'].value * self.models['host'].models[result.params['host_type'].value].flux(5100) * self.extinction(5100, Av=result.params['host_Av'].value)))
                            self.df.loc[i, 'host_type' + '_photo' * self.addPhoto.isChecked()] = self.models['host'].values[host_min]
                            self.df.loc[i, 'L_host' + '_photo' * self.addPhoto.isChecked()] = result.params['host_norm'].value * np.trapz(self.models['host'].models[result.params['host_type'].value].y * self.extinction(self.models['host'].models[result.params['host_type'].value].x, Av=result.params['host_Av'].value), x=self.models['host'].models[result.params['host_type'].value].x * (1 + d['z'])) * 4 * np.pi * Planck15.luminosity_distance(self.df.loc[i, 'z']).to('cm').value ** 2 * 1e-17 / 3.846e33

                    elif self.method.currentText() == 'annealing' and self.hostExt.isChecked() and any([f in self.filters.keys() for f in ['J', 'H', 'K', 'W1', 'W2']]) and any([f in self.filters.keys() for f in ['W3', 'W4']]):

                        result, chi2_min = self.anneal_fit()
                        host_min = self.models['gal'].get_model_ind(result.params)
                        #print(result.params)
                        print(chi2_min, lmfit.report_fit(result))

                        self.df.loc[i, 'Av_int' + '_host' * self.hostExt.isChecked() + '_photo' * self.addPhoto.isChecked()] = result.params['Av'].value
                        self.df.loc[i, 'chi2_av' + '_host' * self.hostExt.isChecked() + '_photo' * self.addPhoto.isChecked()] = chi2_min
                        # self.df.loc[i, 'f_host' + '_photo' * self.addPhoto.isChecked()] = 1 / (1 + (result.params['norm_bbb'].value * np.trapz(bbb.flux(gal.x) * self.extinction(gal.x, Av=result.params['Av'].value), x=gal.x) / (result.params['norm_host'].value * np.trapz(gal.y, x=gal.x))))
                        if self.hostExt.isChecked() and any([f in self.filters.keys() for f in ['J', 'H', 'K', 'W1', 'W2']]):
                            self.template_name_gal = host_min
                            self.df.loc[i, 'Av_host' + '_photo' * self.addPhoto.isChecked()] = result.params['host_Av'].value
                            self.df.loc[i, 'f_host' + '_photo' * self.addPhoto.isChecked()] = 1 / (1 + (result.params['bbb_norm'].value * self.models['bbb'].models[0].flux(5100) * self.extinction(5100, Av=result.params['Av'].value) + result.params['tor_norm'].value * self.models['tor'].models[result.params['tor_type'].value].flux(5100)) / (result.params['host_norm'].value * self.models['gal'].models[host_min].flux(5100) * self.extinction(5100, Av=result.params['host_Av'].value)))
                            self.df.loc[i, 'host_type' + '_photo' * self.addPhoto.isChecked()] = host_min
                            self.df.loc[i, 'L_host' + '_photo' * self.addPhoto.isChecked()] = result.params['host_norm'].value * np.trapz(self.models['gal'].models[host_min].y * self.extinction(self.models['gal'].models[host_min].x, Av=result.params['host_Av'].value), x=self.models['gal'].models[host_min].x * (1 + d['z'])) * 4 * np.pi * Planck15.luminosity_distance(self.df.loc[i, 'z']).to('cm').value ** 2 * 1e-17 / 3.846e33

                        fig = self.plot_spec(spec=spec, params=result.params, fig=fig)

                    elif self.method.currentText() == 'emcee' and self.hostExt.isChecked() and any([f in self.filters.keys() for f in ['J', 'H', 'K', 'W1', 'W2']]) and any([f in self.filters.keys() for f in ['W3', 'W4']]):

                        if 1:
                            result, chi2_min = self.anneal_fit()
                            #fig = self.plot_spec(spec=spec, params=result.params, fig=fig)

                            result, chi2_min, res = self.emcee_fit(params=result.params)
                        else:
                            result, chi2_min, res = self.emcee_fit()

                        print(res)
                        # >>> calc host luminosity:
                        if 1:
                            inds = np.random.randint(result.flatchain.shape[1], size=10000)

                            fig2, ax2 = plt.subplots()
                            s = []
                            for ind in inds:
                                params = result.params
                                for k, p in enumerate(result.params.keys()):
                                    params[p].value = result.flatchain.iloc[ind, k]
                                s.append(np.log10(self.calc_lum(params, 'bol')))
                            x = np.linspace(np.min(s), np.max(s), 100)
                            kde = gaussian_kde(s)
                            d1 = distr1d(x, kde(x))
                            d1.dopoint()
                            d1.dointerval()
                            res = a(d1.point, d1.interval[1] - d1.point, d1.point - d1.interval[0])
                            f = np.asarray([res.plus, res.minus])
                            f = int(np.round(np.abs(np.log10(np.min(f[np.nonzero(f)])))) + 1)
                            if d1.interval[0] <= x[1] and d1.interval[1] < x[-2]:
                                d1.dointerval(conf=0.95)
                                res = a(d1.interval[1], t='u')
                            elif d1.interval[1] >= x[-2] and d1.interval[0] > x[1]:
                                d1.dointerval(conf=0.95)
                                res = a(d1.interval[0], t='l')
                            print(p, res.latex(f=f))
                            d1.plot(conf=0.683, ax=ax2, ylabel='')
                            ax2.yaxis.set_ticklabels([])
                            ax2.yaxis.set_ticks([])
                            ax2.text(.05, .9, 'log Lhost', ha='left', va='top', transform=ax2.transAxes)
                            ax2.text(.95, .9, res.latex(f=f), ha='right', va='top', transform=ax2.transAxes)

                        # >>> plot results:
                        inds = np.random.randint(result.flatchain.shape[1], size=100)

                        total, bbb, tor, host = [], [], [], []
                        s = {}
                        for k in ['total', 'bbb', 'tor', 'host'] + list(self.filters.keys()):
                            s[k] = []
                        for ind in inds:
                            params = result.params
                            for k, p in enumerate(result.params.keys()):
                                params[p].value = result.flatchain.iloc[ind, k]
                                #if p in ['tor_type', 'host_tau', 'host_tg']:
                                #    params[p].value = np.max([params[p].min, np.min([params[p].max, round(params[p].value)])])
                            s['total'].append(self.models['bbb'].models[0].flux(spec[0] / (1 + self.d['z'])) * params['bbb_norm'].value * self.extinction(spec[0] / (1 + self.d['z']), Av=params['Av'].value) + self.models['tor'].models[self.models['tor'].get_model_ind(params)].flux(spec[0] / (1 + self.d['z'])) * params['tor_norm'].value)
                            s['bbb'].append(self.models['bbb'].models[0].y * params['bbb_norm'].value * self.extinction(self.models['bbb'].models[0].x, Av=params['Av'].value))
                            s['tor'].append(self.models['tor'].models[self.models['tor'].get_model_ind(params)].y * params['tor_norm'].value)
                            if self.hostExt.isChecked():
                                host.append(self.models['gal'].models[self.models['gal'].get_model_ind(params)].y * params['host_norm'].value * self.extinction(self.models['gal'].models[self.models['gal'].get_model_ind(params)].x, Av=params['host_Av'].value))
                                s['total'][-1] += self.models['gal'].models[self.models['gal'].get_model_ind(params)].flux(spec[0] / (1 + self.d['z'])) * params['host_norm'].value * self.extinction(spec[0] / (1 + self.d['z']), Av=params['host_Av'].value)

                            # >>> filters fluxes:
                            for k, f in self.filters.items():
                                if self.filters[k].fit:
                                    s[k].append(self.model_emcee(params, f.x, k, mtype='total'))

                        fig.axes[0].fill_between(spec[0] / (1 + self.d['z']), *np.quantile(np.asarray(s['total']), [0.05, 0.95], axis=0), lw=1, color='tab:red', zorder=3, label='total', alpha=0.5)
                        fig.axes[0].fill_between(self.models['bbb'].models[0].x, *np.quantile(np.asarray(s['bbb']), [0.05, 0.95], axis=0), lw=1, color='tab:blue', zorder=3, label='bbb', alpha=0.5)
                        fig.axes[0].fill_between(self.models['tor'].models[self.models['tor'].get_model_ind(params)].x, *np.quantile(np.asarray(s['tor']), [0.05, 0.95], axis=0), lw=1, color='tab:green', zorder=3, label='tor', alpha=0.5)
                        #ax.plot(tor.x, tor.y * params['tor_norm'].value, '--', color='tab:orange', zorder=2, label='composite', alpha=alpha)
                        if self.hostExt.isChecked():
                            fig.axes[0].fill_between(self.models['gal'].models[self.models['gal'].get_model_ind(params)].x, *np.quantile(np.asarray(host), [0.05, 0.95], axis=0), lw=1, color='tab:purple', zorder=2, label='host galaxy', alpha=0.5)

                        for k, f in self.filters.items():
                            if self.filters[k].fit:
                                #print(k, f.x / (1 + self.d['z']), *np.quantile(np.asarray(s[k]), [0.05, 0.95], axis=0))
                                fig.axes[0].fill_between(f.x / (1 + self.d['z']), *np.quantile(np.asarray(s[k]), [0.05, 0.95], axis=0), color='tomato', lw=1, zorder=3, alpha=0.5)

                        if 0:
                            self.df.loc[i, 'Av_int' + '_host' * self.hostExt.isChecked() + '_photo' * self.addPhoto.isChecked()] = result.params['Av'].value
                            self.df.loc[i, 'chi2_av' + '_host' * self.hostExt.isChecked() + '_photo' * self.addPhoto.isChecked()] = chi2_min
                            # self.df.loc[i, 'f_host' + '_photo' * self.addPhoto.isChecked()] = 1 / (1 + (result.params['norm_bbb'].value * np.trapz(bbb.flux(gal.x) * self.extinction(gal.x, Av=result.params['Av'].value), x=gal.x) / (result.params['norm_host'].value * np.trapz(gal.y, x=gal.x))))
                            if self.hostExt.isChecked() and any([f in self.filters.keys() for f in ['J', 'H', 'K', 'W1', 'W2']]):
                                self.template_name_gal = host_min
                                self.df.loc[i, 'Av_host' + '_photo' * self.addPhoto.isChecked()] = result.params['host_Av'].value
                                self.df.loc[i, 'f_host' + '_photo' * self.addPhoto.isChecked()] = 1 / (1 + (result.params['bbb_norm'].value * self.models['bbb'].models[0].flux(5100) * self.extinction(5100, Av=result.params['Av'].value) + result.params['tor_norm'].value * self.models['tor'].models[result.params['tor_type'].value].flux(5100)) / (result.params['host_norm'].value * self.models['gal'].models[host_min].flux(5100) * self.extinction(5100, Av=result.params['host_Av'].value)))
                                self.df.loc[i, 'host_type' + '_photo' * self.addPhoto.isChecked()] = host_min
                                self.df.loc[i, 'L_host' + '_photo' * self.addPhoto.isChecked()] = result.params['host_norm'].value * np.trapz(self.models['gal'].models[host_min].y * self.extinction(self.models['gal'].models[host_min].x, Av=result.params['host_Av'].value), x=self.models['gal'].models[host_min].x * (1 + d['z'])) * 4 * np.pi * Planck15.luminosity_distance(self.df.loc[i, 'z']).to('cm').value ** 2 * 1e-17 / 3.846e33


                    title = "id={0:4d} {1:19s} ({2:5d} {3:5d} {4:4d}) z={5:5.3f} Av={6:4.2f} chi2={7:4.2f}".format(i, d['SDSS_NAME'], d['PLATE'], d['MJD'], d['FIBERID'], d['z'], result.params['Av'].value, chi2_min)
                    if 0 and self.hostExt.isChecked():
                        # title += " fgal={1:4.2f} {0:s}".format(self.models['host'].values[host_min], self.df['f_host' + '_photo' * self.addPhoto.isChecked()][i])
                        title += " fgal={2:4.2f} tau={0:4.2f} tg={1:4.2f}".format(self.models['gal'].values[host_min][0], self.models['gal'].values[host_min][1], self.df['f_host' + '_photo' * self.addPhoto.isChecked()][i])
                    fig.axes[0].set_title(title)

                    if self.saveFig.isChecked():
                        fig.savefig(os.path.dirname(self.parent.ErositaFile) + '/QC/plots/' + self.df['SDSS_NAME'][i] + '.png', bbox_inches='tight', pad_inches=0.1)
                        plt.close()

                else:
                    if not self.addPhoto.isChecked():
                        self.df.loc[i, 'Av_int'] = np.nan
                        self.df.loc[i, 'chi2_av'] = np.nan
                        if self.hostExt.isChecked():
                            self.df.loc[i, 'Av_int_host'] = np.nan
                            self.df.loc[i, 'chi2_av_host'] = np.nan
                            self.df.loc[i, 'Av_host'] = np.nan
                            self.df.loc[i, 'f_host'] = np.nan
                            self.df.loc[i, 'f_host_type'] = np.nan
                    else:
                        self.df.loc[i, 'Av_int_photo'] = np.nan
                        self.df.loc[i, 'chi2_av_photo'] = np.nan
                        if self.hostExt.isChecked():
                            self.df.loc[i, 'Av_int_host_photo'] = np.nan
                            self.df.loc[i, 'chi2_av_host_photo'] = np.nan
                            self.df.loc[i, 'Av_host_photo'] = np.nan
                            self.df.loc[i, 'f_host_photo'] = np.nan
                            self.df.loc[i, 'f_host_photo_type'] = np.nan

                    if ind is None:
                        fmiss.write("{0:4d} {1:19s} {2:5d} {3:5d} {4:4d} \n".format(i+1, self.df['SDSS_NAME'][i], self.df['PLATE'][i], self.df['MJD'][i], self.df['FIBERID'][i]))

        if plot and not self.saveFig.isChecked():
            plt.show()

        if np.sum(self.mask) == len(self.df['SDSS_NAME']):
            self.mask = np.zeros(len(self.df['SDSS_NAME']), dtype=bool)

        if ind is None:
            fmiss.close()
        self.updateData()
        self.save_data()

    def plot_spec(self, spec=None, mask=None, params=None, fig=None, alpha=1):
        if fig is None:
            fig, ax = plt.subplots(figsize=(20, 12))
        else:
            ax = fig.axes[0]

        if spec is not None and params is None:
            ax.plot(spec[0] / (1 + self.d['z']), spec[1], '-k', lw=.5, zorder=2, label='spectrum')
            for k, f in self.filters.items():
                ax.errorbar([f.filter.l_eff / (1 + self.d['z'])], [f.flux], yerr=[[f.err_flux[0]], [f.err_flux[1]]], marker='s', color=[c / 255 for c in f.filter.color])

            if self.addPhoto.isChecked():
                ax.set_xlim([8e2, 3e5])
            else:
                ax.set_xlim([np.min(spec[0] / (1 + self.d['z'])), np.max(spec[0] / (1 + self.d['z']))])
            ax.set_ylim([0.001, ax.get_ylim()[1]])

            if mask is not None:
                ymin, ymax = ax.get_ylim()[0] * np.ones_like(spec[0] / (1 + self.d['z'])), ax.get_ylim()[1] * np.ones_like(spec[0] / (1 + self.d['z']))
                ax.fill_between(spec[0] / (1 + self.d['z']), ymin, ymax, where=mask, color='tab:green', alpha=0.3, zorder=0)
            #fig.legend(loc=1, fontsize=16, borderaxespad=0)

            if self.addPhoto.isChecked():
                ax.set_ylim(0.2, ax.get_ylim()[1])
                ax.set_xscale('log')
                ax.set_yscale('log')

        if params is not None:
            host_min = params['host_tau'].value * (params['host_tg'].max + 1) + params['host_tg'].value

            bbb, tor, host = self.models['bbb'].models[0], self.models['tor'].models[params['tor_type'].value], self.models['gal'].models[host_min]
            # >>> plot templates:
            if alpha == 1:
                ax.plot(bbb.x, bbb.y * params['bbb_norm'].value, '--', color='tab:blue', zorder=2, label='composite', alpha=alpha)
            ax.plot(bbb.x, bbb.y * params['bbb_norm'].value * self.extinction(bbb.x, Av=params['Av'].value),
                    '-', color='tab:blue', zorder=3, label='comp with ext', alpha=alpha)
            ax.plot(tor.x, tor.y * params['tor_norm'].value, '--', color='tab:orange', zorder=2, label='composite', alpha=alpha)
            if self.hostExt.isChecked():
                if alpha == 1:
                    ax.plot(host.x, host.y * params['host_norm'].value, '--', color='tab:purple', zorder=2, label='host galaxy', alpha=alpha)

                ax.plot(host.x, host.y * params['host_norm'].value * self.extinction(host.x, Av=params['host_Av'].value), '-', color='tab:purple', zorder=2, label='host galaxy', alpha=alpha)

            # >>> plot filters fluxes:
            for k, f in self.filters.items():
                temp = bbb.flux(f.x / (1 + self.d['z'])) * self.extinction(f.x / (1 + self.d['z']), Av=params['Av'].value) * params['bbb_norm'].value + tor.flux(f.x / (1 + self.d['z'])) * params['tor_norm'].value
                if self.hostExt.isChecked():
                    temp += host.flux(f.x / (1 + self.d['z'])) * params['host_norm'].value * self.extinction(f.x / (1 + self.d['z']), Av=params['host_Av'].value)
                ax.plot(f.x / (1 + self.d['z']), temp, '-', color='tomato', lw=2, zorder=3, alpha=alpha)
                # ax.scatter(f.filter.l_eff, f.filter.get_value(x=f.x, y=temp * self.extinction(f.x * (1 + self.d['z']), Av=params['Av'].value) * params['norm'].value),
                #           s=20, marker='o', c=[c/255 for c in f.filter.color])

            # >>> total profile:
            temp = bbb.flux(spec[0] / (1 + self.d['z'])) * params['bbb_norm'].value * self.extinction(spec[0] / (1 + self.d['z']), Av=params['Av'].value) + tor.flux(spec[0] / (1 + self.d['z'])) * params['tor_norm'].value
            if self.hostExt.isChecked():
                temp += host.flux(spec[0] / (1 + self.d['z'])) * params['host_norm'].value * self.extinction(spec[0] / (1 + self.d['z']), Av=params['host_Av'].value)

            ax.plot(spec[0] / (1 + self.d['z']), temp, '-', lw=2, color='tab:red', zorder=3, label='total profile', alpha=alpha)
            # print(np.sum(((temp - spec[1]) / spec[2])[mask] ** 2) / np.sum(mask))

        return fig


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
        print(np.sum(mask))

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
        print(np.sum(mask))

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
                       [2760, 2860], #[2940, 2990], [3280, 3330],
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

    def correlate(self):
        self.corr_status = 1 - self.corr_status
        if self.corr_status:
            self.reg = {'all': 'checked', 'all_r': 'checked', 'selected': 'checked'}
            x, y, = self.x_lambda(self.df[self.ero_x_axis]), self.y_lambda(self.df[self.ero_y_axis])
            m = x.notna() * y.notna()  # * (x > 0) * (y > 0)
            x, y = x[m].to_numpy(), y[m].to_numpy()
            print(x, y)

            slope, intercept, r, p, stderr = linregress(x, y)
            print(slope, intercept, r, p, stderr)
            reg = lambda x: intercept + slope * x
            xreg = np.asarray([np.min(x), np.max(x)])
            self.reg['all'] = "{0:.3f} x + {1:.3f}, disp={2:.2f}".format(slope, intercept, np.std(y - reg(x)))
            self.dataPlot.plotRegression(x=xreg, y=reg(xreg), name='all')

            if self.robust.currentText() in ['Huber', 'RANSAC']:
                if self.robust.currentText() == 'Huber':
                    regres = linear_model.HuberRegressor()
                elif self.robust.currentText() == 'RANSAC':
                    regres = linear_model.RANSACRegressor()
                regres.fit(x[:, np.newaxis], y)
                if self.robust.currentText() == 'RANSAC':
                    inlier_mask = regres.inlier_mask_
                    outlier_mask = np.logical_not(inlier_mask)
                    print(np.sum(outlier_mask))
                    slope, intercept = regres.estimator_.coef_[0], regres.estimator_.intercept_
                elif self.robust.currentText() == 'Huber':
                    inlier_mask = np.ones_like(x, dtype=bool)
                    slope, intercept = regres.coef_[0], regres.intercept_
                self.dataPlot.plotRegression(x=xreg, y=regres.predict(xreg[:, np.newaxis]), name='all_r')
                std = np.std(y[inlier_mask] - regres.predict(x[:, np.newaxis])[inlier_mask])
                print(slope, intercept, std)
                self.reg['all_r'] = "{0:.3f} x + {1:.3f}, disp={2:.2f}".format(slope, intercept, std)

            print(np.sum(np.logical_and(np.isfinite(self.x[self.mask]), np.isfinite(self.y[self.mask]))))
            if np.sum(np.logical_and(np.isfinite(self.x[self.mask]), np.isfinite(self.y[self.mask]))):
                x, y, = self.x_lambda(self.df[self.ero_x_axis]), self.y_lambda(self.df[self.ero_y_axis])
                m = self.mask * x.notna() * y.notna() #* (x > 0) * (y > 0)
                x, y = x[m].to_numpy(), y[m].to_numpy()
                slope, intercept, r, p, stderr = linregress(x, y)
                print(slope, intercept, r, p, stderr)
                reg = lambda x: intercept + slope * x
                xreg = np.asarray([np.min(x), np.max(x)])
                self.reg['selected'] = "{0:.3f} x + {1:.3f}, disp={2:.2f}".format(slope, intercept, np.std(y - reg(x)))
                self.dataPlot.plotRegression(x=xreg, y=reg(xreg), name='selected')
            print(self.reg)

        else:
            self.dataPlot.plotRegression(name='all', remove=True)
            self.dataPlot.plotRegression(name='all_r', remove=True)
            self.dataPlot.plotRegression(name='selected', remove=True)

    def save_data(self):
        self.df.to_csv(self.parent.ErositaFile, index=False)

    def closeEvent(self, event):
        super(ErositaWidget, self).closeEvent(event)
        self.parent.ErositaWidget = None

