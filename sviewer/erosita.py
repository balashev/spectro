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
from PyQt5.QtCore import Qt, QPoint, QUrl
from PyQt5.QtGui import QDesktopServices
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
from .QSOSEDfit import QSOSEDfit
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
        self.QSOSEDfit = QSOSEDfit(catalog=self.parent.ErositaFile, plot=self.plotExt.isChecked(),
                                   save=self.saveFig.isChecked(), mcmc_steps=1000, anneal_steps=100, corr=30, verbose=1)

        self.show()
        #self.addSDSSQSO()
        #self.addCustom()

    def initData(self):
        self.opts = {'ero_x_axis': str, 'ero_y_axis': str, 'ero_c_axis': str, 'ero_lUV': float, 'ero_tempmin': float, 'ero_tempmax': float,
                     }
        for opt, func in self.opts.items():
            print(opt, self.parent.options(opt), func(self.parent.options(opt)))
            setattr(self, opt, func(self.parent.options(opt)))

        self.axis_list = ['z', 'DEC', 'RA', 'F_X_int', 'F_X', 'DET_LIKE_0', 'F_UV', 'u-b', 'r-i', 'FIRST_FLUX', 'R', 'Av_gal',
                          'L_X', 'L_UV', 'L_UV_corr', #'L_UV_corr_host_photo',
                          'bbb_slope', 'Av_int', 'Rv', 'Abump', 'EBV', 'FeII',
                          # 'Av_int_host', 'Av_int_photo', 'Av_int_host_photo',
                          #'chi2_av', 'chi2_av_host', 'chi2_av_photo', 'chi2_av_host_photo',
                          'Av_host', #'Av_host_photo',
                          'r_host', 'host_tg', 'host_tau', #'f_host', 'f_host_photo',
                          'L_host', #'L_host_photo',
                          'SDSS_photo_scale', 'SDSS_photo_slope', 'SDSS_var', 'alpha_SDSS', 'slope_SDSS', 'lnL',
                          'F_OIII', 'FWHM_OIII',
                          ]
        self.axis_info = {'z': [lambda x: x, 'z'],
                          'DEC': [lambda x: x, 'DEC'],
                          'RA': [lambda x: x, 'RA'],
                          'F_X_int': [lambda x: np.log10(x), 'log (F X int, erg/s/cm2)'],
                          'F_X': [lambda x: np.log10(x), 'log (F X, erg/s/cm2/Hz)'],
                          'DEL_LIKE_0': [lambda x: np.log10(x), 'log (X-ray detection lnL)'],
                          'F_UV': [lambda x: np.log10(x), 'log (F UV, erg/s/cm2/Hz)'],
                          'L_X': [lambda x: np.log10(x), 'log (L X, erg/s/Hz)'],
                          'L_UV': [lambda x: np.log10(x), 'log (L UV, erg/s/Hz)'],
                          'L_UV_corr': [lambda x: x, 'log (L UV corrected, erg/s/Hz)'],
                          'L_UV_corr_host_photo': [lambda x: np.log10(x), 'log (L UV corrected, erg/s/Hz) with host and photometry'],
                          'u-b': [lambda x: x, 'u - b'],
                          'r-i': [lambda x: x, 'r - i'],
                          'FIRST_FLUX': [lambda x: np.log10(np.abs(x)), 'log (FIRST flux, mJy)'],
                          'R': [lambda x: np.log10(x), 'log R'],
                          'Av_gal': [lambda x: x, 'Av (galactic)'],
                          'bbb_slope': [lambda x: x, 'Slope correction (for big blue bump)'],
                          'FeII': [lambda x: x, 'Strength of the FeII template, a.u.'],
                          'Av_int': [lambda x: x, 'Av (intrinsic)'],
                          'Rv': [lambda x: x, 'Rv (intrinsic)'],
                          'Abump': [lambda x: np.log10(x), '2175 bump strength'],
                          'EBV': [lambda x: x, 'E(B-V) (intrinsic)'],
                          'Av_int_host': [lambda x: x, 'Av (intrinsic) with host'],
                          'Av_int_photo': [lambda x: x, 'Av (intrinsic) with photometry'],
                          'Av_int_host_photo': [lambda x: x, 'Av (intrinsic) with host and photometry'],
                          'lnL': [lambda x: x, 'lnL ~ fit quality'],
                          'chi2_av': [lambda x: np.log10(x), 'log chi^2 extinction fit'],
                          'chi2_av_host': [lambda x: np.log10(x), 'log chi^2 extinction fit with host'],
                          'chi2_av_photo': [lambda x: np.log10(x), 'log chi^2 extinction fit with photometry'],
                          'chi2_av_host_photo': [lambda x: np.log10(x), 'log chi^2 extinction fit with host and photometry'],
                          'Av_host': [lambda x: x, 'Av (host)'],
                          'Av_host_photo': [lambda x: x, 'Av (host) with photometry'],
                          'host_tg': [lambda x: x, 'log (age of host host, Gyr) '],
                          'host_tau': [lambda x: x, 'exp factor of SF decay, Gyr'],
                          'r_host': [lambda x: x, 'Galaxy fraction (from Rakshit)'],
                          'f_host': [lambda x: x, 'Galaxy fraction (from my fit)'],
                          'f_host_photo': [lambda x: x, 'Galaxy fraction (from my fit) with photometry'],
                          'L_host': [lambda x: x, 'log (L host, [erg/s])'],
                          'L_host_photo': [lambda x: np.log10(x), 'log (L_host, L_sun) with photometry'],
                          'SDSS_photo_scale': [lambda x: x, 'Difference between spectrum and photometry in SDSS'],
                          'SDSS_photo_slope': [lambda x: x, 'Difference in slope between spectrum and photometry in SDSS'],
                          'SDSS_var': [lambda x: np.log10(x), 'log Variability at 2500A from SDSS'],
                          'alpha_SDSS': [lambda x: x, 'Scale for SDSS photometry'],
                          'slope_SDSS': [lambda x: x, 'Slope for SDSS photometry'],
                          'F_OIII': [lambda x: np.log10(x), 'log F(OIII), in 1e14 erg/s/cm^2'],
                          'FWHM_OIII': [lambda x: x, 'FWHM OIII, km/s'],
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
        calcExt.clicked.connect(self.calc_ext)

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

        lineEmission = QPushButton('Line emission')
        lineEmission.setFixedSize(120, 30)
        lineEmission.clicked.connect(partial(self.calc_line_emission, ind=None))

        l.addWidget(QLabel('                   '))
        l.addWidget(self.addPhoto)
        l.addWidget(self.filters_used)
        l.addWidget(compPhoto)
        l.addWidget(lineEmission)
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
                self.df.insert(len(self.df.columns), 'F_X_err', np.nan)
            gamma = 1.9
            scale = ((2.2 / 2) ** (2 - gamma) - (0.3 / 2) ** (2 - gamma)) / (2 - gamma)
            #print(scale, ((2 * u.eV).to(u.Hz, equivalencies=u.spectral()).value))
            #print((1 + self.df['z']) ** gamma)
            self.df['F_X'] = self.df['F_X_int'] / ((2e3 * u.eV).to(u.Hz, equivalencies=u.spectral()).value) / scale / (1 + self.df['z']) ** (2 - gamma)
            self.df['F_X_err'] = self.df['ML_FLUX_ERR_0_ero'] / ((2e3 * u.eV).to(u.Hz, equivalencies=u.spectral()).value) / scale / (
                        1 + self.df['z']) ** (2 - gamma)
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

        print(x, y)
        if ind is None:
            if x is not None and y is not None:
                ind = np.argmin((x - self.x) ** 2 + (y - self.y) ** 2)
            elif name is not None:
                ind = np.where(self.df['SDSS_NAME'] == name)[0][0]

        print(x, y, ind, self.df['SDSS_NAME'][ind])

        if ind is not None:

            self.ind = ind
            if name is None and self.ErositaTable.columnIndex('SDSS_NAME') is not None:
                row = self.ErositaTable.getRowIndex(column='SDSS_NAME', value=self.df['SDSS_NAME'][ind])
                self.ErositaTable.setCurrentCell(row, 0)
                self.ErositaTable.selectRow(row)
                self.ErositaTable.row_clicked(row=row)

            self.set_filters(ind, clear=True)
            if x is None and self.plotExt.isChecked():
                self.plot_sed(self.ind)
                #self.calc_line_emission(ind=self.ind, line='OIII', plot=1)
                #self.calc_ext(self.ind)

            #for k, f in self.filters.items():
            #    self.parent.plot.vb.addItem(f.scatter)
            #    self.parent.plot.vb.addItem(f.errorbar)

            self.updateData(ind=self.ind)

    def set_filters(self, ind, clear=False, names=None):
        for f in self.filters:
            try:
                self.parent.plot.vb.removeItem(f[0])
                self.parent.plot.vb.removeItem(f[1])
            except:
                pass
        self.QSOSEDfit.set_filters(ind=ind)
        self.filters = []
        for k, f in self.QSOSEDfit.filters.items():
            self.filters.append([0, 0])
            flux = f.get_flux(f.value) * 1e17
            err_flux = [(f.get_flux(f.value + f.err) - f.get_flux(f.value)) * 1e17, (f.get_flux(f.value) - f.get_flux(f.value - f.err)) * 1e17]
            self.filters[-1][0] = pg.ErrorBarItem(x=f.l_eff, y=flux, top=err_flux[0], bottom=err_flux[1], beam=2, pen=pg.mkPen(width=2, color=f.color))
            self.filters[-1][1] = pg.ScatterPlotItem(x=[f.l_eff], y=[f.flux], size=10, pen=pg.mkPen(width=2, color='w'), brush=pg.mkBrush(*f.color))
            self.parent.plot.vb.addItem(self.filters[-1][0])
            self.parent.plot.vb.addItem(self.filters[-1][1])

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

        name = 'F_UV_{0:4d}'.format(int(float(self.lambdaUV.text()))).replace('_2500', '')
        if name not in self.df.columns:
            self.df.insert(len(self.df.columns), name, np.nan)
            self.df.insert(len(self.df.columns), name + '_err', np.nan)
        self.df[name], self.df[name+'_err'] = np.nan, np.nan

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
                        em = np.average(spec[2][mask], weights=spec[2][mask]) * 1e-17 * u.erg / u.cm ** 2 / u.AA / u.s
                        # print(nm, wm)
                        #self.df.loc[i, 'F_UV'] = nm.to(u.erg / u.cm ** 2 / u.s / u.Hz, equivalencies=u.spectral_density(
                        #    self.ero_lUV * u.AA * (1 + d['z']))).value / (1 + d['z'])
                        self.df.loc[i, name] = nm.to(u.erg / u.cm ** 2 / u.s / u.Hz, equivalencies=u.spectral_density(
                            self.ero_lUV * u.AA * (1 + d['z']))).value
                        self.df.loc[i, name+'_err'] = em.to(u.erg / u.cm ** 2 / u.s / u.Hz, equivalencies=u.spectral_density(
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
            self.df.insert(len(self.df.columns), 'L_X_err', np.nan)
        self.df['L_X'] = np.nan

        name = 'L_UV_{0:4d}'.format(int(float(self.lambdaUV.text()))).replace('_2500', '')
        if 'L_UV' not in self.df.columns:
            self.df.insert(len(self.df.columns), name, np.nan)
            self.df.insert(len(self.df.columns), name + '_err', np.nan)
        self.df[name], self.df[name + '_err'] = np.nan, np.nan

        if 'F_X' in self.df.columns:
            self.df.loc[mask, 'L_X'] = 4 * np.pi * dl[mask] ** 2 * self.df['F_X'][mask] #/ (1 + self.df['Z'])
            self.df.loc[mask, 'L_X_err'] = 4 * np.pi * dl[mask] ** 2 * self.df['F_X_err'][mask]

        if 'F_UV' in self.df.columns:
            #self.df.loc[mask, 'L_UV'] = 4 * np.pi * dl[mask] ** 2 * self.df['F_UV'][mask] #/ (1 + self.df['Z'])
            self.df.loc[mask, name] = 4 * np.pi * dl[mask] ** 2 * self.df[name.replace('L', 'F')][mask] #/ (1 + self.df['z'][mask])
            self.df.loc[mask, name + '_err'] = 4 * np.pi * dl[mask] ** 2 * self.df[name.replace('L', 'F') + '_err'][mask]

        if 0:
            for attr in ['']: # ['', '_host_photo']:
                if 'Av_int' + attr in self.df.columns:
                    if 'L_UV_corr' not in self.df.columns:
                        self.df.insert(len(self.df.columns), 'L_UV_corr', np.nan)
                        self.df.insert(len(self.df.columns), 'L_UV_corr_err', np.nan)
                    self.df['L_UV_corr'] = np.nan

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
                self.d = d
                self.set_filters(i, clear=True, names=['g', 'r', 'i', 'z'])

                x, data, err = [], [], []
                for f in self.filters.values():
                    if f.value != 0:
                        x.append(np.log10(f.filter.l_eff))
                        data.append(f.filter.get_value(x=spec[0], y=spec[1], err=spec[2], mask=np.logical_not(spec[3])) - f.value)
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
                        self.df.loc[i, 'SDSS_photo_scale'] = round(np.ma.average(ma, weights=err), 3)
                        print(self.df.loc[i, 'SDSS_photo_scale'])
                        model = linear_model.LinearRegression().fit(x[~ma.mask].reshape((-1, 1)), ma[~ma.mask], sample_weight=err[~ma.mask])
                        self.df.loc[i, 'SDSS_photo_slope'] = round(model.coef_[0] / 2.5, 3)
                        if not np.isnan(d['z']):
                            print(model.predict(np.asarray(np.log10(2500 * (1 + d['z']))).reshape((-1, 1))))
                            ### SDSS_var ~ spectrum - photometry
                            self.df.loc[i, 'SDSS_var'] = round(10 ** (model.predict(np.asarray(np.log10(2500 * (1 + d['z']))).reshape((-1, 1))) / 2.5)[0], 3)
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
            self.df.loc[i, 'Av_gal'] = round(3.1 * m.ebv(d['RA'], d['DEC']), 3)

        self.save_data()
        self.updateData()

    def calc_host_luminosity(self, ind=None, norm=None):
        """
        Calculate the luminosity of the host galaxy
        Args:
            ind: index of the QSO, if None, than run for all

        Returns:

        """
        self.df.loc[i, 'Av_host' + '_photo' * self.addPhoto.isChecked()]

        for i, d in self.df.iterrows():
            if ((ind is None) or (ind is not None and i == int(ind))) and np.isfinite(d['z']):
                self.df.loc[i, 'Av_host' + '_photo' * self.addPhoto.isChecked()]

    def calc_line_emission(self, ind=None, line='OIII', plot=0):
        """
        Estimate the line flux emission in specified emission line
        Args:
            ind: index of the QSO, if None, than run for all
            line: name of the line
        Returns:
        """
        k = 0

        name = 'F_{0:s}'.format(line)
        if name not in self.df.columns:
            self.df.insert(len(self.df.columns), name, np.nan)
            self.df.insert(len(self.df.columns), name + '_err', np.nan)
            self.df.insert(len(self.df.columns), name.replace('F', 'FWHM'), np.nan)
            self.df.insert(len(self.df.columns), name.replace('F', 'FWHM') + '_err', np.nan)

        if ind == None:
            self.df[name], self.df[name + '_err'], self.df[name.replace('F', 'FWHM')], self.df[name.replace('F', 'FWHM') + '_err'] = np.nan, np.nan, np.nan, np.nan

        l = 5008.23
        for i, d in self.df.iterrows():
            if (ind == None or i == ind) and np.isfinite(d['z']) and d['z'] < 10000 / l - 1:
                print(i)
                spec = self.loadSDSS(d['PLATE'], d['FIBERID'], d['MJD'], Av_gal=d['Av_gal'])
                if spec is not None and d['z'] < (spec[0][-1] / l) * (1 - 2/300) - 1:
                    mask = (spec[0] > l * (1 + d['z']) * (1 - 2.5/300)) * (spec[0] < l * (1 + d['z']) * (1 + 6/300)) * (spec[3] == 0)
                    maskline = (spec[0] > l * (1 + d['z']) * (1 - 0.2/300)) * (spec[0] < l * (1 + d['z']) * (1 + 0.2/300)) * (spec[3] == 0)
                    if np.sum(mask) > 10 and np.sum(maskline) > 4 and not np.isnan(np.mean(spec[1][mask])) and not np.mean(spec[2][mask]) == 0:
                        x, y, err = spec[0][mask] / l / (1 + d['z']) - 1, spec[1][mask], spec[2][mask]
                        def gaussian(x, amp, cen, wid, zero, slope):
                            """1-d gaussian: gaussian(x, amp, cen, wid)"""
                            d = wid / 3e5
                            if slope > 3:
                                slope = 3
                            return amp / (np.sqrt(2 * np.pi) * d) * np.exp(-(x - cen) ** 2 / (2 * d ** 2)) + zero + slope * (x - np.mean(x)) / (np.max(x) - np.min(x))

                        gmodel = lmfit.Model(gaussian)
                        result = gmodel.fit(y, x=x, amp=(np.max(y) - np.min(y)) * (np.sqrt(2 * np.pi) * 200 / 3e5), cen=0, wid=300, zero=np.min(y), slope=0)
                        print(result.params['amp'], result.params['wid'], result.params['slope'])
                        if result.params['amp'].stderr is not None and result.params['wid'].stderr is not None:
                            self.df.loc[i, name] = round(result.params['amp'].value * l * (1 + d['z']), 3)
                            self.df.loc[i, name + '_err'] = round(result.params['amp'].stderr * l * (1 + d['z']), 3)
                            self.df.loc[i, name.replace('F', 'FWHM')] = round(result.params['wid'].value * 2.355, 3)
                            self.df.loc[i, name.replace('F', 'FWHM') + '_err'] = round(result.params['wid'].stderr * 2.355, 3)

                        if plot:
                            fig, ax = plt.subplots()
                            x = x * 3e5
                            x = spec[0][mask]
                            ax.plot(x, y, 'o')
                            ax.plot(x, result.init_fit, '--', label='initial fit')
                            ax.plot(x, result.best_fit, '-', label='best fit')
                            ax.plot(x, result.params['zero'] + result.params['slope'] * (x - np.mean(x)) / (np.max(x) - np.min(x)), '-', label='continuum')
                            ax.legend()

        if ind != None:
            print(self.df.loc[ind, 'RA'], self.df.loc[ind, 'DEC'])
            url = QUrl("https://www.legacysurvey.org/viewer?ra={0:f}&dec={1:f}&layer=ls-dr9&zoom=16".format(self.df.loc[ind, 'RA'], self.df.loc[ind, 'DEC']))
            QDesktopServices.openUrl(url)

        if plot:
            plt.show()
        else:
            self.updateData()
        self.save_data()

    def plot_sed(self, ind=None):
        for attr in ['_mcmc', '_post', '_spec']:
            if os.name == 'nt':
                print(os.path.dirname(self.parent.ErositaFile) + '/QC/plots/' + self.df['SDSS_NAME'][ind] + attr + '.png')
                os.startfile(os.path.dirname(self.parent.ErositaFile) + '/QC/plots/' + self.df['SDSS_NAME'][ind] + attr + '.png')

    def calc_ext(self):
        self.QSOSEDfit.plot, self.QSOSEDfit.save = self.plotExt.isChecked(), self.saveFig.isChecked()
        self.QSOSEDfit.mcmc_steps, self.QSOSEDfit.anneal_steps = int(self.numSteps.text()), int(int(self.numSteps.text()) / 10)
        if self.QSOSEDfit.prepare(self.ind):
            res = self.QSOSEDfit.fit(self.ind, method='emcee')
            print(res)
        plt.show()

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

