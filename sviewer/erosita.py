from astropy.cosmology import Planck15, FlatLambdaCDM, LambdaCDM
from astropy.io import fits
import astropy.units as u
from dust_extinction.averages import G03_SMCBar
import itertools
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
from lmfit import Minimizer, Parameters, report_fit, fit_report
import pyqtgraph as pg
from pyqtgraph.dockarea import *
from functools import partial
import os
import pandas as pd
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QMenu, QToolButton,
                             QLabel, QCheckBox, QFrame, QTextEdit, QSplitter, QComboBox, QAction, QSizePolicy)
from scipy.interpolate import interp1d
from scipy.stats import linregress
import sfdmap
from .graphics import SpectrumFilter
from .tables import *
from .utils import smooth
from ..profiles import add_ext, add_LyaForest

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
        self.reg = {'all': None, 'selected': None}
        self.vb = self.getViewBox()
        self.cursorpos = pg.TextItem(anchor=(0, 1), fill=pg.mkBrush(0, 0, 0, 0.5))
        self.vb.addItem(self.cursorpos, ignoreBounds=True)
        self.selectedstat = {'shown': pg.TextItem(anchor=(1, 1), fill=pg.mkBrush(0, 0, 0, 0.5)),
                             'selected': pg.TextItem(anchor=(1, 1), fill=pg.mkBrush(0, 0, 0, 0.5)),
                             'shown_sel': pg.TextItem(anchor=(1, 1), fill=pg.mkBrush(0, 0, 0, 0.5))}
        for k in self.selectedstat.keys():
            self.vb.addItem(self.selectedstat[k], ignoreBounds=True)
        self.corr = {'all': pg.TextItem(anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 0.5)),
                     'selected': pg.TextItem(anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 0.5))}
        self.s_status = False
        self.d_status = False
        self.show()

    def plotData(self, x=None, y=None, name='sample', color=(255, 255, 255), size=7, rescale=True):
        if hasattr(self, name) and getattr(self, name) is not None:
            self.removeItem(getattr(self, name))

        if x is not None and y is not None and len(x) > 0 and len(y) > 0:
            brush = pg.mkBrush(*color, 255)
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
            pens = {'all': pg.mkPen(0, 127, 255, width=6), 'selected': pg.mkPen(30, 250, 127, width=6)}
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
        for name, ind in zip(['all', 'selected'], [10, 30]):
            if self.reg[name] is not None:
                self.corr[name].setPos(self.vb.mapSceneToView(QPoint(pos.left() + 10, pos.top() + ind)))
                self.corr[name].setText(name + ': ' + self.parent.reg[name])

    def mouseDoubleClickEvent(self, ev):
        super(dataPlot, self).mouseDoubleClickEvent(ev)
        if ev.button() == Qt.LeftButton:
            self.mousePos = self.vb.mapSceneToView(ev.pos())
            print(self.mousePos.x(), self.mousePos.y())
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

    def currentText(self):
        return ' '.join([s for s in self.list if s.isChecked()])

    def set(self, item):
        print(item, self.name)
        if self.name == 'table':
            if getattr(self, item).isChecked():
                self.parent.shown_cols.append(item)
            else:
                self.parent.shown_cols.remove(item)
            l = [self.list.index(o) for o in self.parent.shown_cols]
            print(self.parent.shown_cols)
            self.parent.shown_cols = [self.list[l[o]] for o in np.argsort(l)]
            self.parent.parent.options('ero_colnames', ' '.join(self.parent.shown_cols))
            print(self.parent.shown_cols)
            self.parent.ErositaTable.setdata(self.parent.df[self.parent.shown_cols].to_records(index=False))

        if self.name == 'extcat':
            self.parent.addExternalCatalog(item, show=getattr(self, item).isChecked())

        if self.name == 'hosts':
            if getattr(self, item).isChecked():
                self.parent.host_templates.append(item)
            else:
                self.parent.host_templates.remove(item)
            l = [self.list.index(o) for o in self.parent.host_templates]
            print(self.parent.host_templates)
            self.parent.host_templates = [self.list[l[o]] for o in np.argsort(l)]
            self.parent.parent.options('ero_hosts', ' '.join(self.parent.host_templates))
            print(self.parent.host_templates)


class Filter():
    def __init__(self, parent, name, value=None, flux=None, err=None):
        self.parent = parent
        self.name = name
        self.filter = SpectrumFilter(self.parent.parent, self.name)

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

    def calc_weight(self, z=0):
        num = int((np.log(self.x[-1]) - np.log(self.x[0])) / 0.001)
        x = np.exp(np.linspace(np.log(self.x[0]), np.log(self.x[-1]), num))
        if 'W' in self.name:
            self.weight = 1
        else:
            self.weight = 1 #np.sqrt(np.sum(self.filter.inter(x)) / np.max(self.filter.data.y))
        #self.weight = 1
        #print('weight:', self.name, self.weight)

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
        self.opts = {'ero_x_axis': str, 'ero_y_axis': str, 'ero_lUV': float, 'ero_tempmin': float, 'ero_tempmax': float,
                     }
        for opt, func in self.opts.items():
            #print(opt, self.parent.options(opt), func(self.parent.options(opt)))
            setattr(self, opt, func(self.parent.options(opt)))

        self.axis_list = ['z', 'F_X_int', 'F_X', 'DET_LIKE_0', 'F_UV', 'u-b', 'r-i', 'Av_gal',
                          'L_X', 'L_UV', 'L_UV_corr', 'L_UV_corr_host_photo',
                          'Av_int', 'Av_int_host', 'Av_int_photo', 'Av_int_host_photo',
                          'chi2_av', 'chi2_av_host', 'chi2_av_photo', 'chi2_av_host_photo',
                          'f_host', 'f_host_photo', 'r_host']
        self.axis_info = {'z': [lambda x: x, 'z'],
                          'F_X_int': [lambda x: np.log10(x), 'log (F_X_int, erg/s/cm2)'],
                          'F_X': [lambda x: np.log10(x), 'log (F_X, erg/s/cm2/Hz)'],
                          'DEL_LIKE_0': [lambda x: np.log10(x), 'log (Xray detection lnL)'],
                          'F_UV': [lambda x: np.log10(x), 'log (F_UV, erg/s/cm2/Hz)'],
                          'L_X': [lambda x: np.log10(x), 'log (L_X, erg/s/Hz)'],
                          'L_UV': [lambda x: np.log10(x), 'log (L_UV, erg/s/Hz)'],
                          'u-b': [lambda x: x, 'u - b'],
                          'r-i': [lambda x: x, 'r - i'],
                          'Av_gal': [lambda x: x, 'Av (galactic)'],
                          'Av_int': [lambda x: x, 'Av (intrinsic)'],
                          'Av_int_host': [lambda x: x, 'Av (intrinsic) with host'],
                          'Av_int_photo': [lambda x: x, 'Av (intrinsic) with photometry'],
                          'Av_int_host_photo': [lambda x: x, 'Av (intrinsic) with host and photometry'],
                          'chi2_av': [lambda x: x, 'chi^2 extinction fit'],
                          'chi2_av_host': [lambda x: x, 'chi^2 extinction fit with host'],
                          'chi2_av_photo': [lambda x: x, 'chi^2 extinction fit with photometry'],
                          'chi2_av_host_photo': [lambda x: x, 'chi^2 extinction fit with host and photometry'],
                          'f_host': [lambda x: x, 'Galaxy fraction (from my fit)'],
                          'f_host_photo': [lambda x: x, 'Galaxy fraction (from my fit) with photometry'],
                          'r_host': [lambda x: x, 'Galaxy fraction (from Rakshit)'],
                          'L_UV_corr': [lambda x: np.log10(x), 'log (L_UV corrected, erg/s/Hz)'],
                          'L_UV_corr_host_photo': [lambda x: np.log10(x), 'log (L_UV corrected, erg/s/Hz)']
                          }
        self.ind = None
        self.corr_status = 0
        self.ext = {}
        self.filters = {}

        self.template_name_qso = 'composite'
        self.template_name_gal = 0
        self.host_templates = self.parent.options('ero_hosts').split()
        # self.template_name_gal = "Sb_template_norm" #"Sey2_template_norm"
        # self.host_templates = ['S0', 'Sa', 'Sb', 'Sc', 'Ell2', 'Ell5', 'Ell13']

    def initGUI(self):
        area = DockArea()
        layout = QVBoxLayout()
        layout.addWidget(area)

        d1 = Dock("Plot", size=(900, 850))
        self.dataPlot = dataPlot(self, [self.ero_x_axis, self.ero_y_axis])
        d1.addWidget(self.dataPlot)
        d2 = Dock("Panel", size=(900, 150))

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

        cols = QLabel('Cols:')
        cols.setFixedSize(40, 30)

        self.cols = ComboMultipleBox(self, name='table')

        l.addWidget(xaxis)
        l.addWidget(self.x_axis)
        l.addWidget(QLabel('   '))
        l.addWidget(yaxis)
        l.addWidget(self.y_axis)
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

        l.addWidget(QLabel('                   '))
        l.addWidget(calcExtGal)
        l.addWidget(calcExt)
        l.addWidget(self.QSOtemplate)
        l.addWidget(QLabel('range:'))
        l.addWidget(self.tempmin)
        l.addWidget(QLabel('..'))
        l.addWidget(self.tempmax)
        l.addStretch()

        layout.addLayout(l)

        l = QHBoxLayout()

        self.plotExt = QCheckBox("plot")
        self.plotExt.setChecked(True)
        self.plotExt.setFixedSize(80, 30)

        self.addPhoto = QCheckBox("add photometry")
        self.addPhoto.setChecked(True)
        self.addPhoto.setFixedSize(140, 30)

        self.hostExt = QCheckBox("add host:")
        self.hostExt.setChecked(True)
        self.hostExt.setFixedSize(120, 30)

        self.hosts = ComboMultipleBox(self, name='hosts')
        self.hosts.addItems(['S0', 'Sa', 'Sb', 'Sc', 'Sd', 'Sdm', 'Ell2', 'Ell5', 'Ell13', 'Sey2', 'Sey18'])
        self.hosts.setFixedSize(120, 30)
        self.hosts.update()

        l.addWidget(QLabel('                   '))
        l.addWidget(self.plotExt)
        l.addWidget(self.addPhoto)
        l.addWidget(self.hostExt)
        l.addWidget(self.hosts)
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
        print(self.df.columns)
        print('F_X_int' in self.df.columns)
        self.cols.addItems(self.df.columns)
        #self.cols.addItems(list(self.df.columns)[40:59] + list(self.df.columns)[:40] + list(self.df.columns)[:59])
        self.shown_cols = self.parent.options('ero_colnames').split()
        print(self.shown_cols)
        self.cols.update()
        self.ErositaTable.setdata(self.df[self.shown_cols].to_records(index=False))
        self.mask = np.zeros(len(self.df['z']), dtype=bool)

    def getData(self):
        self.x_lambda, self.y_lambda = self.axis_info[self.ero_x_axis][0], self.axis_info[self.ero_y_axis][0]
        self.x, self.y, self.SDSScolors = None, None, False
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

    def updateData(self, ind=None):
        #print(self.ero_x_axis, self.ero_y_axis)
        self.getData()

        #print(self.x, self.y)
        if self.x is not None and self.y is not None:
            self.dataPlot.setLabel('bottom', self.axis_info[self.ero_x_axis][1])
            self.dataPlot.setLabel('left', self.axis_info[self.ero_y_axis][1])

            if self.ind is not None:
                self.dataPlot.plotData(x=[self.x[self.ind]], y=[self.y[self.ind]],
                                       name='clicked', color=(255, 3, 62), size=20, rescale=False)

            if ind is None:
                self.dataPlot.plotData(x=self.x, y=self.y)

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
            self.dataPlot.plotData(x=self.x_lambda(self.ext[name][self.self.ero_x_axis]),
                                   y=self.y_lambda(self.ext[name][self.ero_y_axis]),
                                   name=name, color=(200, 150, 50))

        else:
            self.dataPlot.removeItem(getattr(self.dataPlot, name))
            del self.ext[name]

    def index(self, x=None, y=None, name=None, ext=True):
        ind = None
        if x is not None and y is not None:
            ind = np.argmin((x - self.x) ** 2 +
                            (y - self.y) ** 2)
        elif name is not None:
            ind = np.where(self.df['SDSS_NAME'] == name)[0][0]

        #print(x, y, name, ind, self.df['SDSS_NAME_fl'][ind])

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

    def set_filters(self, ind, clear=False):
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
                if not np.isnan(self.df[f'{k}mag'][ind]) and not np.isnan(self.df[f'e_{k}mag'][ind]):
                    self.filters[k] = Filter(self, k, value=self.df[f'{k}mag'][ind], err=self.df[f'e_{k}mag'][ind])

            for k in ['J', 'H', 'K', 'W1', 'W2', 'W3', 'W4']:
                if not np.isnan(self.df[k + 'MAG'][ind]) and not np.isnan(self.df['ERR_' + k + 'MAG'][ind]):
                    self.filters[k] = Filter(self, k, value=self.df[k + 'MAG'][ind], err=self.df['ERR_' + k + 'MAG'][ind])

    def select_points(self, x1, y1, x2, y2, remove=False, add=False):
        x1, x2, y1, y2 = np.min([x1, x2]), np.max([x1, x2]), np.min([y1, y2]), np.max([y1, y2])
        mask = (self.x > x1) * (self.x < x2) * (self.y > y1) * (self.y < y2)
        if not add and not remove:
            self.mask = mask
        elif add and not remove:
            self.mask = np.logical_or(self.mask, mask)
        elif not add and remove:
            self.mask = np.logical_and(self.mask, ~mask)
        self.updateData()

    def extinction(self, x, Av=None):
        """
        Return extinction for provided wavelengths
        Args:
            Av: visual extinction
            x: wavelenghts in Ansgstrem

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
            print(i)
            if d['z'] > 3500 / self.ero_lUV - 1:
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

    def calc_Lum(self):

        dl = Planck15.luminosity_distance(self.df['z'].to_numpy()).to('cm')

        if 'L_X' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'L_X', np.nan)
        self.df['L_X'] = np.nan

        if 'F_X' in self.df.columns:
            self.df['L_X'] = 4 * np.pi * dl ** 2 * self.df['F_X'] #/ (1 + self.df['Z'])

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

        if 'F_UV' in self.df.columns:
            self.df['L_UV'] = 4 * np.pi * dl ** 2 * self.df['F_UV'] #/ (1 + self.df['Z'])
            self.df[name] = 4 * np.pi * dl ** 2 * self.df[name.replace('L', 'F')] / (1 + self.df['z'])

        for attr in ['', '_host_photo']:
            if 'Av_int' + attr in self.df.columns:
                for i, d in self.df.iterrows():
                    if np.isfinite(d['Av_int' + attr]):
                        self.df.loc[i, 'L_UV_corr' + attr] = 4 * np.pi * dl[i].value ** 2 * d[name.replace('L', 'F')] / (1 + d['z']) / self.extinction(float(self.lambdaUV.text()), Av=d['Av_int' + attr])
                    else:
                        print(i)


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

    def calc_ext(self, ind=None, plot=False, gal=True):

        for attr in ['Av_int', 'Av_int_photo', 'Av_int_host', 'Av_int_host_photo', 'f_host', 'f_host_photo', 'chi2_av', 'chi2_av_photo', 'chi2_av_host', 'chi2_av_host_photo', 'host_type', 'host_type_photo']:
            if attr not in self.df.columns:
                self.df.insert(len(self.df.columns), attr, np.nan)

        if np.sum(self.mask) == 0:
            self.mask = np.ones(len(self.df['z']), dtype=bool)

        if ind is None and np.sum(self.mask) == len(self.df['z']):
            if not self.addPhoto.isChecked():
                self.df['Av_int'] = np.nan
                self.df['chi2_av'] = np.nan
                if self.hostExt.isChecked():
                    self.df['Av_int_host'] = np.nan
                    self.df['chi2_av_host'] = np.nan
                    self.df['f_host'] = np.nan
                    self.df['f_host_type'] = np.nan
            else:
                self.df['Av_int_photo'] = np.nan
                self.df['chi2_av_photo'] = np.nan
                if self.hostExt.isChecked():
                    self.df['Av_int_host_photo'] = np.nan
                    self.df['chi2_av_host_photo'] = np.nan
                    self.df['f_host_photo'] = np.nan
                    self.df['f_host_photo_type'] = np.nan

        self.load_template_qso(smooth_window=7, init=True)
        self.load_template_gal(smooth_window=3, init=True)

        if ind is None:
            fmiss = open("temp/av_missed.dat", "w")

        print('calc_ext:', ind)
        for i, d in self.df.iterrows():
            if (ind is None and self.mask[i]) or (ind is not None and i == int(ind)):
                spec = self.loadSDSS(d['PLATE'], d['FIBERID'], d['MJD'], Av_gal=d['Av_gal'])

                self.set_filters(i, clear=True)

                if plot:
                    fig, ax = plt.subplots()
                    ax.plot(spec[0] / (1 + d['z']), spec[1], '-k', lw=.5, zorder=2, label='spectrum')
                    for k, f in self.filters.items():
                        ax.errorbar([f.filter.l_eff / (1 + d['z'])], [f.flux], yerr=[[f.err_flux[0]], [f.err_flux[1]]], marker='s', color=[c/255 for c in f.filter.color])

                if spec is not None:
                    mask = self.calc_mask(spec, z_em=d['z'], iter=3, window=201, clip=2.5)
                    if np.sum(mask) > 50:
                        sm = [np.asarray(spec[0][mask], dtype=np.float64), spec[1][mask], spec[2][mask]]
                        temp = self.load_template_qso(x=sm[0], z_em=d['z'])
                        temp_gal = self.load_template_gal(x=sm[0], z_em=d['z'])

                        def fcn2min(params):
                            chi = (temp * self.extinction(sm[0] / (1 + d['z']), Av=params.valuesdict()['Av']) * params.valuesdict()['norm'] - sm[1]) / sm[2]
                            for f in self.filters.values():
                                if f.x[0] > self.ero_tempmin * (1 + d['z']) and f.x[-1] < self.ero_tempmax * (1 + d['z']) and np.isfinite(f.value) and np.isfinite(f.err):
                                    chi = np.append(chi, [f.weight / f.err * (f.value - f.filter.get_value(x=f.x, y=self.load_template_qso(x=f.x, z_em=d['z']) * self.extinction(f.x / (1 + d['z']), Av=params.valuesdict()['Av']) * params.valuesdict()['norm']))])
                            return chi

                        def fcn2min_gal(params):
                            chi = (temp * self.extinction(sm[0] / (1 + d['z']), Av=params.valuesdict()['Av']) * params.valuesdict()['norm'] + temp_gal * params.valuesdict()['norm_host'] - sm[1]) / sm[2]
                            for f in self.filters.values():
                                if f.x[0] > self.ero_tempmin * (1 + d['z']) and f.x[-1] < self.ero_tempmax * (1 + d['z']) and np.isfinite(f.value) and np.isfinite(f.err):
                                    #print(f.name, f.weight, f.err, f.value, f.filter.get_value(x=f.x, y=self.load_template_qso(x=f.x, z_em=d['z']) * self.extinction(f.x / (1 + d['z']), Av=params.valuesdict()['Av']) * params.valuesdict()['norm'] + self.load_template_gal(x=f.x, z_em=d['z']) * params.valuesdict()['norm_gal']), self.load_template_qso(x=f.x, z_em=d['z']) * self.extinction(f.x / (1 + d['z']), Av=params.valuesdict()['Av']) * params.valuesdict()['norm'], self.load_template_gal(x=f.x, z_em=d['z']) * params.valuesdict()['norm_gal'])
                                    #print(f.name, f.weight, f.err, f.value, f.filter.get_value(x=f.x, y=self.load_template_qso(x=f.x, z_em=d['z']) * self.extinction(f.x / (1 + d['z']), Av=params.valuesdict()['Av']) * params.valuesdict()['norm'] + self.load_template_gal(x=f.x, z_em=d['z']) * params.valuesdict()['norm_gal']))
                                    chi = np.append(chi, [f.weight / f.err * (f.value - f.filter.get_value(x=f.x, y=self.load_template_qso(x=f.x, z_em=d['z']) * self.extinction(f.x / (1 + d['z']), Av=params.valuesdict()['Av']) * params.valuesdict()['norm'] + self.load_template_gal(x=f.x, z_em=d['z']) * params.valuesdict()['norm_host']))])
                            return chi

                        #norm = np.average(sm[1], weights=sm[2]) / np.average(temp)
                        norm = np.nanmean(sm[1]) / np.nanmean(temp)
                        # create a set of Parameters
                        params = Parameters()
                        params.add('Av', value=0.0, min=-10, max=10)
                        params.add('norm', value=norm, min=0, max=1e10)
                        minner = Minimizer(fcn2min, params, nan_policy='propagate', calc_covar=True)
                        result = minner.minimize()
                        chi = fcn2min(result.params)
                        chi2_min = np.sum(chi ** 2) / (len(chi) - len(params))
                        print(i, result.params['Av'].value, result.params['norm'].value, chi2_min)
                        self.df.loc[i, 'Av_int' + '_photo' * self.addPhoto.isChecked()] = result.params['Av'].value
                        self.df.loc[i, 'chi2_av' + '_photo' * self.addPhoto.isChecked()] = chi2_min

                        results = []
                        if self.hostExt.isChecked():
                            for host in range(len(self.host_templates)):
                                self.template_name_gal = host
                                self.load_template_gal(smooth_window=3, init=True)
                                temp_gal = self.load_template_gal(x=sm[0], z_em=d['z'])
                                params = Parameters()
                                params.add('Av', value=result.params['Av'].value, min=-10, max=10)
                                params.add('norm', value=result.params['norm'].value, min=0, max=1e10)
                                params.add('norm_host', value=result.params['norm'].value / 10, min=0, max=1e10)
                                minner = Minimizer(fcn2min_gal, params, nan_policy='propagate', calc_covar=True)
                                results.append(minner.minimize())
                                chi = fcn2min_gal(results[-1].params)
                                chi2 = np.sum(chi ** 2) / (len(chi) - len(results[-1].params))
                                print(i, "{0:s} {1:.3f} {2:.2f} {3:.2f} {4:.5f}".format(self.host_templates[self.template_name_gal], results[-1].params['Av'].value, results[-1].params['norm'].value, results[-1].params['norm_host'].value, chi2))
                                if host == 0 or chi2 < chi2_min:
                                    chi2_min, host_min = chi2, host
                            #print(i, result.params['Av'].value, result.params['norm'].value, result.params['norm_host'].value, np.sum(chi ** 2) / (len(chi) - len(params)))

                            result = results[host_min]
                            self.template_name_gal = host_min
                            self.load_template_gal(smooth_window=3, init=True)
                            temp_gal = self.load_template_gal(x=sm[0], z_em=d['z'])
                            self.df.loc[i, 'Av_int_host' + '_photo' * self.addPhoto.isChecked()] = result.params['Av'].value
                            self.df.loc[i, 'chi2_av_host' + '_photo' * self.addPhoto.isChecked()] = chi2_min
                            self.df.loc[i, 'f_host' + '_photo' * self.addPhoto.isChecked()] = 1 / (1 + (result.params['norm'].value * np.trapz(self.load_template_qso(x=self.template_gal[0]) * self.extinction(self.template_gal[0], Av=result.params['Av'].value), x=self.template_gal[0])) / (result.params['norm_host'].value * np.trapz(self.template_gal[1], x=self.template_gal[0])))
                            self.df.loc[i, 'host_type' + '_photo' * self.addPhoto.isChecked()] = self.host_templates[host_min]
                            print(self.df['f_host' + '_photo' * self.addPhoto.isChecked()][i], self.df['host_type' + '_photo' * self.addPhoto.isChecked()][i])

                        if plot:
                            # >>> plot templates:
                            ax.plot(self.template_qso[0], self.template_qso[1] * result.params['norm'].value,
                                    '--', color='tab:blue', zorder=2, label='composite')
                            ax.plot(self.template_qso[0], self.template_qso[1] * result.params['norm'].value * self.extinction(self.template_qso[0], Av=result.params['Av'].value),
                                    '-', color='tab:blue', zorder=3, label='comp with ext')
                            if self.hostExt.isChecked():
                                ax.plot(self.template_gal[0], self.template_gal[1] * result.params['norm_host'].value,
                                        '-', color='tab:purple', zorder=2, label='galaxy')

                            # >>> plot filters fluxes:
                            for k, f in self.filters.items():
                                temp = self.load_template_qso(x=f.x, z_em=d['z']) * self.extinction(f.x / (1 + d['z']), Av=result.params['Av'].value) * result.params['norm'].value
                                if self.hostExt.isChecked():
                                    temp += self.load_template_gal(x=f.x, z_em=d['z']) * result.params['norm_host'].value
                                ax.plot(f.x / (1 + d['z']), temp, '-', color='tomato', zorder=3)
                                #ax.scatter(f.filter.l_eff, f.filter.get_value(x=f.x, y=temp * self.extinction(f.x * (1 + d['z']), Av=result.params['Av'].value) * result.params['norm'].value),
                                #           s=20, marker='o', c=[c/255 for c in f.filter.color])

                            # >>> total profile:
                            temp = self.load_template_qso(x=spec[0], z_em=d['z']) * result.params['norm'].value * self.extinction(spec[0] / (1 + d['z']), Av=result.params['Av'].value)
                            if self.hostExt.isChecked():
                                temp += self.load_template_gal(x=spec[0], z_em=d['z']) * result.params['norm_host'].value

                            ax.plot(spec[0] / (1 + d['z']), temp, '-', color='tab:red', zorder=3, label='total profile')
                            #print(np.sum(((temp - spec[1]) / spec[2])[mask] ** 2) / np.sum(mask))

                            if self.addPhoto.isChecked():
                                ax.set_xlim([8e2, 3e5])
                            else:
                                ax.set_xlim([np.min(spec[0] / (1 + d['z'])), np.max(spec[0] / (1 + d['z']))])
                            title = "id={0:4d} {1:19s} ({2:5d} {3:5d} {4:4d}) z={5:5.3f} Av={6:4.2f} chi2={7:4.2f}".format(i, d['SDSS_NAME'], d['PLATE'], d['MJD'], d['FIBERID'], d['z'], result.params['Av'].value, chi2_min)
                            if self.hostExt.isChecked():
                                title += " f_gal={1:4.2f} {0:s}".format(self.host_templates[host_min], self.df['f_host' + '_photo' * self.addPhoto.isChecked()][i])
                            ax.set_title(title)

                            if 0:
                                inds = np.where(np.diff(mask))[0]
                                for s, f in zip(range(0, len(inds), 2), range(1, len(inds), 2)):
                                    ax.axvspan(spec[0][inds[s]], spec[0][inds[f]], color='tab:green', alpha=0.3, zorder=1)
                            else:
                                ymin, ymax = ax.get_ylim()[0] * np.ones_like(spec[0] / (1 + d['z'])), ax.get_ylim()[1] * np.ones_like(spec[0] / (1 + d['z']))
                                ax.fill_between(spec[0] / (1 + d['z']), ymin, ymax, where=mask, color='tab:green', alpha=0.3, zorder=0)
                            fig.legend(loc=1, fontsize=16, borderaxespad=2)

                    else:
                        if ind is None:
                            fmiss.write("{0:4d} {1:19s} {2:5d} {3:5d} {4:4d} \n".format(i, self.df['SDSS_NAME'][i], self.df['PLATE'][i], self.df['MJD'][i], self.df['FIBERID'][i]))
        if plot:
            if self.addPhoto.isChecked():
                ax.set_xscale('log')
                ax.set_yscale('log')

            plt.show()

        if np.sum(self.mask) == len(self.df['z']):
            self.mask = np.zeros(len(self.df['z']), dtype=bool)

        if ind is None:
            fmiss.close()
        self.updateData()
        self.save_data()

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

    def calc_mask(self, spec, z_em=0, iter=3, window=301, clip=2.0):
        mask = np.logical_not(self.sdss_mask(spec[3]))
        #mask = np.asarray(spec[3][:] == 0, dtype=bool)
        #print(np.sum(mask))
        mask *= spec[0] > 1280 * (1 + z_em)
        #print(np.sum(mask))
        for i in range(iter):
            m = np.zeros_like(spec[0])
            if window > 0 and np.sum(mask) > window:
                if i > 0:
                    m[mask] = np.abs(sm - spec[1][mask]) / spec[2][mask] > clip
                    mask *= np.logical_not(self.expand_mask(m, exp_pixel=3))
                    #mask[mask] *= np.abs(sm - spec[1][mask]) / spec[2][mask] < clip
                sm = smooth(spec[1][mask], window_len=window, window='hanning', mode='same')

        mask = np.logical_not(self.expand_mask(np.logical_not(mask), exp_pixel=3))
        #print(np.sum(mask))
        # remove prominent emission lines regions
        windows = [[1295, 1320], [1330, 1360], [1375, 1430], [1500, 1600], [1625, 1700], [1740, 1760],
                   [1840, 1960], [2050, 2120], [2250, 2650], [2710, 2890], [2940, 2990], [3280, 3330],
                   [3820, 3920], [4200, 4680], [4780, 5080], [5130, 5400], [5500, 5620], [5780, 6020],
                   [6300, 6850], [7600, 8050], [8250, 8300], [8400, 8600], [9000, 9400], [9500, 9700],
                   [9950, 10200]]
        for w in windows:
            mask *= (spec[0] < w[0] * (1 + z_em)) + (spec[0] > w[1] * (1 + z_em))

        #print(np.sum(mask))
        # remove atmospheric absorption region
        windows = [[5560, 5600], [6865, 6930], [7580, 7690], [9300, 9600], [10150, 10400], [13200, 14600]]
        for w in windows:
            mask *= (spec[0] < w[0]) + (spec[0] > w[1])

        #print(np.sum(mask))
        return mask

    def load_template_qso(self, x=None, z_em=0, smooth_window=None, init=False):
        if init:
            if self.template_name_qso in ['VandenBerk', 'HST', 'Slesing', 'power', 'composite']:
                if self.template_name_qso == 'VandenBerk':
                    self.template_qso = np.genfromtxt('data/SDSS/medianQSO.dat', skip_header=2, unpack=True)
                elif self.template_name_qso == 'HST':
                    self.template_qso = np.genfromtxt('data/SDSS/hst_composite.dat', skip_header=2, unpack=True)
                elif self.template_name_qso == 'Slesing':
                    self.template_qso = np.genfromtxt('data/SDSS/Slesing2016.dat', skip_header=0, unpack=True)
                elif self.template_name_qso == 'power':
                    self.template_qso = np.ones((2, 1000))
                    self.template_qso[0] = np.linspace(500, 25000, self.template_qso.shape[1])
                    self.template_qso[1] = np.power(self.template_qso[0] / 2500, -1.9)
                    smooth_window = None
                elif self.template_name_qso == 'composite':
                    self.template_qso = np.genfromtxt('data/SDSS/Slesing2016.dat', skip_header=0, unpack=True)
                    self.template_qso = self.template_qso[:, np.logical_or(self.template_qso[1] != 0, self.template_qso[2] != 0)]
                    if 0:
                        x = self.template_qso[0][-1] + np.arange(1, int((25000 - self.template_qso[0][-1]) / 0.4)) * 0.4
                        y = np.power(x / 2500, -1.9) * 6.542031
                        self.template_qso = np.append(self.template_qso, [x, y, y / 10], axis=1)
                    else:
                        if 1:
                            data = np.genfromtxt('data/SDSS/QSO1_template_norm.sed', skip_header=0, unpack=True)
                            m = (data[0] > self.template_qso[0][-1]) * (data[0] < self.ero_tempmax)
                            self.template_qso = np.append(self.template_qso, [data[0][m], data[1][m] * self.template_qso[1][-1] / data[1][m][0], data[1][m] / 30], axis=1)
                        x = self.template_qso[0][0] + np.linspace(int((self.ero_tempmin - self.template_qso[0][0]) / 0.4), -1) * 0.4
                        y = np.power(x / self.template_qso[0][0], -1.0) * self.template_qso[1][0]
                        self.template_qso = np.append([x, y, y / 10], self.template_qso, axis=1)
            if smooth_window is not None:
                self.template_qso[1] = smooth(self.template_qso[1], window_len=smooth_window, window='hanning', mode='same')
        else:
            if x is not None:
                inter = interp1d(self.template_qso[0] * (1 + z_em), self.template_qso[1] * add_LyaForest(self.template_qso[0] * (1 + z_em), z_em=z_em), bounds_error=False, fill_value=0, assume_sorted=True)
                return inter(x)
            else:
                return self.template_qso[1]

    def load_template_gal(self, x=None, z_em=0, smooth_window=None, init=False):
        if init:
            if 0:
                if isinstance(self.template_name_gal, int):
                    f = fits.open(f"data/SDSS/spDR2-0{23 + self.template_name_gal}.fit")
                    self.template_gal = [10 ** (f[0].header['COEFF0'] + f[0].header['COEFF1'] * np.arange(f[0].header['NAXIS1'])), f[0].data[0]]

                    if smooth_window is not None:
                        self.template_gal[1] = smooth(self.template_gal[1], window_len=smooth_window, window='hanning', mode='same')
            if isinstance(self.template_name_gal, int):
                self.template_gal = np.genfromtxt(f'data/SDSS/{self.host_templates[self.template_name_gal]}_template_norm.sed', unpack=True)
                if 1:
                    mask = (self.template_gal[0] > self.ero_tempmin) * (self.template_gal[0] < self.ero_tempmax)
                    self.template_gal = [self.template_gal[0][mask], self.template_gal[1][mask]]
        else:
            if x is not None:
                inter = interp1d(self.template_gal[0] * (1 + z_em), self.template_gal[1], bounds_error=False, fill_value=0, assume_sorted=True)
                return inter(x)
            else:
                return self.template_gal[1]

    def correlate(self):
        self.corr_status = 1 - self.corr_status
        if self.corr_status:
            self.reg = {'all': 'checked', 'selected': 'checked'}
            x, y, = self.x_lambda(self.df[self.ero_x_axis]), self.y_lambda(self.df[self.ero_y_axis])
            m = x.notna() * y.notna() # * (x > 0) * (y > 0)
            x, y = x[m].to_numpy(), y[m].to_numpy()
            print(x, y)
            slope, intercept, r, p, stderr = linregress(x, y)
            print(slope, intercept, r, p, stderr)
            reg = lambda x: intercept + slope * x
            xreg = np.asarray([np.min(x), np.max(x)])
            self.reg['all'] = "{0:.3f} x + {1:.3f}, disp={2:.2f}".format(slope, intercept, np.std(y - reg(x)))
            self.dataPlot.plotRegression(x=xreg, y=reg(xreg), name='all')
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
            self.dataPlot.plotRegression(name='selected', remove=True)

    def save_data(self):
        self.df.to_csv(self.parent.ErositaFile, index=False)

    def closeEvent(self, event):
        super(ErositaWidget, self).closeEvent(event)
        self.parent.ErositaWidget = None
