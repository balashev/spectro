from astropy.cosmology import Planck15, FlatLambdaCDM, LambdaCDM
from astropy.io import fits
import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as rfn
import pyqtgraph as pg
from pyqtgraph.dockarea import *
from functools import partial
import os
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QMenu, QToolButton,
                             QLabel, QCheckBox, QFrame, QTextEdit, QSplitter, QComboBox, QAction)
from scipy.stats import linregress
from .tables import *

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
        self.reg = None
        self.show()

    def plotData(self, x=None, y=None, name='sample', color=(255, 255, 255), size=7, rescale=True):
        if hasattr(self, name) and getattr(self, name) is not None:
            self.removeItem(getattr(self, name))

        if x is not None and y is not None:
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

    def plotRegression(self, x=None, y=None):
        if self.reg is not None:
            self.removeItem(self.reg)

        if x is not None and y is not None:
            self.reg = pg.PlotCurveItem(x, y, pen=pg.mkPen(0, 127, 255, width=6))
            self.addItem(self.reg)

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
            if item in self.parent.shown_cols:
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
        self.opts = {'ero_x_axis': str, 'ero_y_axis': str, 'ero_lUV': float
                     }
        for opt, func in self.opts.items():
            #print(opt, self.parent.options(opt), func(self.parent.options(opt)))
            setattr(self, opt, func(self.parent.options(opt)))

        self.axis_list = ['z', 'Fx_int', 'Fx', 'Fuv', 'Lx', 'Luv']
        self.axis_info = {'z': ['z', lambda x: x, 'z'], 'Fx_int': ['F_X_int', lambda x: np.log10(x), 'log (F_X_int, erg/s/cm2)'],
                          'Fx': ['F_X', lambda x: np.log10(x), 'log (F_X, erg/s/cm2/Hz)'], 'Fuv': ['F_UV', lambda x: np.log10(x), 'log (F_UV, erg/s/cm2/Hz)'],
                          'Lx': ['L_X', lambda x: np.log10(x), 'log (L_X, erg/s/Hz)'], 'Luv': ['L_UV', lambda x: np.log10(x), 'log (L_UV, erg/s/Hz)']}
        self.ind = None
        self.ext = {}

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
        self.x_axis.setFixedSize(50, 30)
        self.x_axis.addItems(self.axis_list)
        self.x_axis.setCurrentText(self.ero_x_axis)
        self.x_axis.currentIndexChanged.connect(partial(self.axisChanged, 'x_axis'))

        yaxis = QLabel('y axis:')
        yaxis.setFixedSize(40, 30)

        self.y_axis = QComboBox()
        self.y_axis.setFixedSize(50, 30)
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
        lambdaUV.setFixedSize(70, 30)

        self.lambdaUV = QLineEdit()
        self.lambdaUV.setFixedSize(60, 30)
        self.lambdaUV.setText('{0:6.1f}'.format(self.ero_lUV))
        self.lambdaUV.textEdited.connect(self.lambdaUVChanged)

        calcFUV = QPushButton('Calc FUV')
        calcFUV.setFixedSize(80, 30)
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
        l.addWidget(QLabel('   '))
        l.addWidget(calcFUV)
        l.addWidget(calcLum)
        l.addWidget(correlate)
        l.addWidget(cats)
        l.addWidget(self.extcat)
        l.addStretch()

        layout.addLayout(l)

        layout.addStretch()
        widget.setLayout(layout)

        return widget

    def loadTable(self, recalc=False):
        self.df = pd.read_csv(self.parent.ErositaFile)
        try:
            self.df.rename(columns={'Z_fl': 'z', 'ML_FLUX_0': 'F_X_int'}, inplace=True)
        except:
            pass
        if recalc:
            if 'F_X' not in self.df.columns:
                self.df.insert(len(self.df.columns), 'F_X', np.nan)
            gamma = 0.9
            scale = ((2.2 / 2) ** (1 - gamma) - (0.3 / 2) ** (1 - gamma)) / (1 - gamma)
            #print(scale, ((2 * u.eV).to(u.Hz, equivalencies=u.spectral()).value))
            #print((1 + self.df['Z_fl']) ** gamma)
            self.df['F_X'] = self.df['F_X_int'] / ((2e3 * u.eV).to(u.Hz, equivalencies=u.spectral()).value) / scale / (1 + self.df['z']) ** (2 - gamma)
            # self.df['ML_FLUX_0'] in erg/s/cm^2
            # self.df['F_X'] in erg/s/cm^2/Hz

        self.cols.addItems(list(self.df.columns)[40:59] + list(self.df.columns)[:40] + list(self.df.columns)[:59])
        self.shown_cols = self.parent.options('ero_colnames').split()
        self.cols.update()
        self.ErositaTable.setdata(self.df[self.shown_cols].to_records(index=False))

    def updateData(self, ind=None):
        self.x_lambda, self.y_lambda = self.axis_info[self.ero_x_axis][1], self.axis_info[self.ero_y_axis][1]
        if self.axis_info[self.ero_x_axis][0] in self.df.columns and self.axis_info[self.ero_y_axis][0] in self.df.columns:
            self.dataPlot.setLabel('bottom', self.axis_info[self.ero_x_axis][2])
            self.dataPlot.setLabel('left', self.axis_info[self.ero_y_axis][2])

            if ind is None:
                self.dataPlot.plotData(x=self.x_lambda(self.df[self.axis_info[self.ero_x_axis][0]]),
                                       y=self.y_lambda(self.df[self.axis_info[self.ero_y_axis][0]])
                                       )
                for name in self.ext.keys():
                    self.dataPlot.plotData(x=self.x_lambda(self.ext[name][self.axis_info[self.ero_x_axis][0]]),
                                           y=self.y_lambda(self.ext[name][self.axis_info[self.ero_y_axis][0]]),
                                           name=name, color=(200, 150, 50))
            if self.ind is not None:
                self.dataPlot.plotData(x=[self.x_lambda(self.df[self.axis_info[self.ero_x_axis][0]][self.ind])],
                                       y=[self.y_lambda(self.df[self.axis_info[self.ero_y_axis][0]][self.ind])],
                                       name='selected', color=(255, 3, 62), size=20, rescale=False)

    def addExternalCatalog(self, name, show=True):
        if name == 'Risaliti2015':
            if show:
                self.ext[name] = np.genfromtxt(os.path.dirname(self.parent.ErositaFile) + '/Risaliti2015/table2.dat',
                                     names=['name', 'ra', 'dec', 'z', 'L_UV', 'L_X', 'F_UV', 'eF_UV', 'F_X', 'eF_X', 'group'])
                for attr in ['L_UV', 'L_X', 'F_UV', 'F_X']:
                    self.ext[name][attr] = 10 ** self.ext[name][attr]

                self.ext[name] = rfn.append_fields(self.ext[name], 'F_X_int', np.empty(self.ext[name].shape[0], dtype='<f4'), dtypes='<f4')
                gamma = 0.9
                self.ext[name]['F_X_int'] = self.ext[name]['F_X'] * (1 + self.ext[name]['z']) ** (2 - gamma) * \
                                            ((2.2 / 2) ** (1-gamma) - (0.3 / 2) ** (1-gamma)) / (1 - gamma) * \
                                            ((2e3 * u.eV).to(u.Hz, equivalencies=u.spectral()).value)

        if show:
            self.dataPlot.plotData(x=self.x_lambda(self.ext[name][self.axis_info[self.ero_x_axis][0]]),
                                   y=self.y_lambda(self.ext[name][self.axis_info[self.ero_y_axis][0]]),
                                   name=name, color=(200, 150, 50))

        else:
            self.dataPlot.removeItem(getattr(self.dataPlot, name))
            del self.ext[name]

    def index(self, x=None, y=None, name=None):
        ind = None
        if x is not None and y is not None:
            ind = np.argmin((x - self.x_lambda(self.df[self.axis_info[self.ero_x_axis][0]])) ** 2 +
                            (y - self.y_lambda(self.df[self.axis_info[self.ero_y_axis][0]])) ** 2)
        elif name is not None:
            ind = np.where(self.df['SDSS_NAME_fl'] == name)[0][0]

        if ind is not None:
            self.ind = ind
            if name is None and self.ErositaTable.columnIndex('SDSS_NAME_fl') is not None:
                row = self.ErositaTable.getRowIndex(column='SDSS_NAME_fl', value=self.df['SDSS_NAME_fl'][ind])
                self.ErositaTable.setCurrentCell(row, 0)
                self.ErositaTable.selectRow(row)
                self.ErositaTable.row_clicked(row=row)

            self.updateData(ind=self.ind)

            #self.parent.loadSDSS(plate=self.df['PLATE_fl'][ind], fiber)

    def axisChanged(self, axis):
        self.parent.options('ero_' + axis, getattr(self, axis).currentText())
        setattr(self, 'ero_' + axis, self.parent.options('ero_' + axis))
        self.updateData()

    def lambdaUVChanged(self):
        self.ero_lUV = float(self.lambdaUV.text())
        self.parent.options('ero_lUV', self.ero_lUV)

    def loadSDSS(self, plate, fiber, mjd):

        filename = os.path.dirname(self.parent.ErositaFile) + '/spectra/spec-{0:04d}-{2:05d}-{1:04d}.fits'.format(int(plate), int(fiber), int(mjd))
        if os.path.exists(filename):
            qso = fits.open(filename)
            return [10 ** qso[1].data['loglam'][:], qso[1].data['flux'][:], np.sqrt(1.0 / qso[1].data['ivar'][:]), qso[1].data['and_mask']]

    def calc_Lum(self):

        dl = Planck15.luminosity_distance(self.df['z']).to('cm')

        if 'L_X' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'L_X', np.nan)
        self.df['L_X'] =np.nan

        if 'F_X' in self.df.columns:
            self.df['L_X'] = 4 * np.pi * dl ** 2 * self.df['F_X'] #/ (1 + self.df['Z_fl'])

        if 'L_UV' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'L_UV', np.nan)
        self.df['L_UV'] = np.nan

        if 'F_UV' in self.df.columns:
            self.df['L_UV'] = 4 * np.pi * dl ** 2 * self.df['F_UV'] #/ (1 + self.df['Z_fl'])

        self.save_data()
        self.updateData()

    def calc_FUV(self):
        k = 0
        if 'F_UV' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'F_UV', np.nan)
        self.df['F_UV'] = np.nan
        for i, d in self.df.iterrows():
            if d['z'] > 3600 / self.ero_lUV - 1:
                if i < len(self.df['z']):
                    k += 1
                    spec = self.loadSDSS(d['PLATE_fl'], d['FIBERID_fl'], d['MJD_fl'])
                    #print(spec)
                    if spec is not None:
                        mask = (spec[0] > (self.ero_lUV - 10) * (1 + d['z'])) * (spec[0] < (self.ero_lUV + 10) * (1 + d['z']))
                        if not np.isnan(np.mean(spec[1][mask])) and not np.mean(spec[2][mask]) == 0:
                            nm = np.mean(spec[1][mask])
                            wm = np.average(spec[1][mask], weights=spec[2][mask]) * 1e-17 * u.erg / u.cm ** 2 / u.AA / u.s
                            #print(nm, wm)
                            self.df['F_UV'][i] = wm.to(u.erg / u.cm ** 2 / u.s / u.Hz, equivalencies=u.spectral_density(self.ero_lUV * u.AA * (1 + d['z']))).value * (1 + d['z'])
                            # self.df['F_UV'] in erg/s/cm^2/Hz

        self.updateData()
        self.save_data()

    def correlate(self):
        x, y, = self.df[self.axis_info[self.ero_x_axis][0]], self.df[self.axis_info[self.ero_y_axis][0]]
        m = x.notna() * y.notna() * (x > 0) * (y > 0)
        x, y = self.x_lambda(x[m].to_numpy()), self.y_lambda(y[m].to_numpy())
        slope, intercept, r, p, stderr = linregress(x, y)
        print(slope, intercept, r, p, stderr)
        line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
        reg = lambda x: intercept + slope * x
        xreg = np.asarray([np.min(x), np.max(x)])
        self.dataPlot.plotRegression(x=xreg, y=reg(xreg))

        print(np.std(y - reg(x)))

    def save_data(self):
        self.df.to_csv(self.parent.ErositaFile, index=False)

    def closeEvent(self):
        self.parent.ErositaWidget = None
        super(ErositaWidget, self).closeEvent()