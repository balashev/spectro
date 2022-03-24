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
from scipy.stats import linregress
import sfdmap
from .graphics import SpectrumFilter
from .tables import *
from .utils import smooth

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

class Filter():
    def __init__(self, parent, name, value=None, err=None):
        self.parent = parent
        self.name, self.value, self.err = name, value, err
        self.filter = SpectrumFilter(self.parent.parent, self.name + '_VISTA')
        if value is not None and err is not None:
            print(self.filter.get_flux(value) * 1e17)
            self.scatter = pg.ErrorBarItem(x=self.filter.l_eff, y=self.filter.get_flux(value) * 1e17,
                                          top=(self.filter.get_flux(value + err) - self.filter.get_flux(value)) * 1e17,
                                          bottom=(self.filter.get_flux(value) - self.filter.get_flux(value - err)) * 1e17,
                                          beam=2, pen=pg.mkPen(width=2, color=self.filter.color))
            self.errorbar = pg.ScatterPlotItem(x=[self.filter.l_eff], y=[self.filter.get_flux(value) * 1e17],
                                                size=10, pen=pg.mkPen(width=2, color='w'), brush=pg.mkBrush(*self.filter.color))
        else:
            self.scatter, self.errorbar = None, None
        self.x = self.filter.data.x[:]
        self.calc_weight()

    def calc_weight(self, z=0):
        num = int((np.log(self.x[-1]) - np.log(self.x[0])) / 0.001)
        x = np.exp(np.linspace(np.log(self.x[0]), np.log(self.x[-1]), num))
        self.weight = np.sqrt(np.sum(self.filter.inter(x)) / np.max(self.filter.data.y))
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
        self.opts = {'ero_x_axis': str, 'ero_y_axis': str, 'ero_lUV': float
                     }
        for opt, func in self.opts.items():
            #print(opt, self.parent.options(opt), func(self.parent.options(opt)))
            setattr(self, opt, func(self.parent.options(opt)))

        self.axis_list = ['z', 'Fx_int', 'Fx', 'det_like', 'Fuv', 'Lx', 'Luv', 'u-b', 'r-i', 'Av_gal', 'Av_int', 'r_gal', 'Luv_corr']
        self.axis_info = {'z': ['z', lambda x: x, 'z'],
                          'Fx_int': ['F_X_int', lambda x: np.log10(x), 'log (F_X_int, erg/s/cm2)'],
                          'Fx': ['F_X', lambda x: np.log10(x), 'log (F_X, erg/s/cm2/Hz)'],
                          'det_like': ['DET_LIKE_0', lambda x: np.log10(x), 'log (Xray detection lnL)'],
                          'Fuv': ['F_UV', lambda x: np.log10(x), 'log (F_UV, erg/s/cm2/Hz)'],
                          'Lx': ['L_X', lambda x: np.log10(x), 'log (L_X, erg/s/Hz)'],
                          'Luv': ['L_UV', lambda x: np.log10(x), 'log (L_UV, erg/s/Hz)'],
                          'u-b': ['u-b', lambda x: x, 'u - b'],
                          'r-i': ['r-i', lambda x: x, 'r - i'],
                          'Av_gal': ['Av_gal', lambda x: x, 'Av (galactic)'],
                          'Av_int': ['Av_int', lambda x: x, 'Av (intrinsic)'],
                          'r_gal': ['r_gal', lambda x: x, 'Galaxy fraction (from Rakshit)'],
                          'Luv_corr': ['L_UV_corr', lambda x: np.log10(x), 'log (L_UV corrected, erg/s/Hz)']}
        self.ind = None
        self.corr_status = 0
        self.ext = {}
        self.filters = {}

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
        self.x_axis.setFixedSize(70, 30)
        self.x_axis.addItems(self.axis_list)
        self.x_axis.setCurrentText(self.ero_x_axis)
        self.x_axis.currentIndexChanged.connect(partial(self.axisChanged, 'x_axis'))

        yaxis = QLabel('y axis:')
        yaxis.setFixedSize(40, 30)

        self.y_axis = QComboBox()
        self.y_axis.setFixedSize(70, 30)
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
        l.addWidget(QLabel('   '))
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
        self.QSOtemplate.setFixedSize(100, 30)
        self.template_name_qso = 'composite'
        self.QSOtemplate.setCurrentText(self.template_name_qso)
        self.template_name_gal = 0

        self.plotExt = QCheckBox("plot")
        self.plotExt.setChecked(True)
        self.plotExt.setFixedSize(80, 30)

        self.galExt = QCheckBox("add galaxy")
        self.galExt.setChecked(True)
        self.galExt.setFixedSize(120, 30)

        l.addWidget(lambdaUV)
        l.addWidget(QLabel('    '))
        l.addWidget(calcExtGal)
        l.addWidget(calcExt)
        l.addWidget(self.QSOtemplate)
        l.addWidget(self.plotExt)
        l.addWidget(self.galExt)
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
            gamma = 1.9
            scale = ((2.2 / 2) ** (2 - gamma) - (0.3 / 2) ** (2 - gamma)) / (2 - gamma)
            #print(scale, ((2 * u.eV).to(u.Hz, equivalencies=u.spectral()).value))
            #print((1 + self.df['Z_fl']) ** gamma)
            self.df['F_X'] = self.df['F_X_int'] / ((2e3 * u.eV).to(u.Hz, equivalencies=u.spectral()).value) / scale / (1 + self.df['z']) ** (2 - gamma)
            # self.df['ML_FLUX_0'] in erg/s/cm^2
            # self.df['F_X'] in erg/s/cm^2/Hz

        self.cols.addItems(list(self.df.columns)[40:59] + list(self.df.columns)[:40] + list(self.df.columns)[:59])
        self.shown_cols = self.parent.options('ero_colnames').split()
        self.cols.update()
        self.ErositaTable.setdata(self.df[self.shown_cols].to_records(index=False))
        self.mask = np.zeros(len(self.df['z']), dtype=bool)

    def getData(self):
        print()
        self.x_lambda, self.y_lambda = self.axis_info[self.ero_x_axis][1], self.axis_info[self.ero_y_axis][1]
        self.x, self.y, self.SDSScolors = None, None, False
        if self.axis_info[self.ero_x_axis][0] in self.df.columns:
            self.x = self.x_lambda(self.df[self.axis_info[self.ero_x_axis][0]])
        elif self.axis_info[self.ero_x_axis][0] in ['u-b', 'r-i']:
            self.x = self.x_lambda(self.df['PSFMAG' + str('_ubriz'.index(self.axis_info[self.ero_x_axis][0].split('-')[0])) + '_fl'] - self.df['PSFMAG' + str('_ubriz'.index(self.axis_info[self.ero_x_axis][0].split('-')[1])) + '_fl'])
            self.SDSScolors = True

        if self.axis_info[self.ero_y_axis][0] in self.df.columns:
            self.y = self.y_lambda(self.df[self.axis_info[self.ero_y_axis][0]])
        elif self.axis_info[self.ero_y_axis][0] in ['u-b', 'r-i']:
            self.y = self.y_lambda(self.df['PSFMAG' + str('_ubriz'.index(self.axis_info[self.ero_y_axis][0].split('-')[0])) + '_fl'] - self.df['PSFMAG' + str('_ubriz'.index(self.axis_info[self.ero_y_axis][0].split('-')[1])) + '_fl'])
            self.SDSScolors = True

    def updateData(self, ind=None):
        self.getData()

        if self.x is not None and self.y is not None:
            self.dataPlot.setLabel('bottom', self.axis_info[self.ero_x_axis][2])
            self.dataPlot.setLabel('left', self.axis_info[self.ero_y_axis][2])

            if self.ind is not None:
                self.dataPlot.plotData(x=[self.x[self.ind]], y=[self.y[self.ind]],
                                       name='clicked', color=(255, 3, 62), size=20, rescale=False)

            if ind is None:
                self.dataPlot.plotData(x=self.x, y=self.y)

                if not self.SDSScolors:
                    for name in self.ext.keys():
                        self.dataPlot.plotData(x=self.x_lambda(self.ext[name][self.axis_info[self.ero_x_axis][0]]),
                                               y=self.y_lambda(self.ext[name][self.axis_info[self.ero_y_axis][0]]),
                                               name=name, color=(200, 150, 50))

            if np.sum(self.mask) > 0:
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
            self.dataPlot.plotData(x=self.x_lambda(self.ext[name][self.axis_info[self.ero_x_axis][0]]),
                                   y=self.y_lambda(self.ext[name][self.axis_info[self.ero_y_axis][0]]),
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
            ind = np.where(self.df['SDSS_NAME_fl'] == name)[0][0]

        print(x, y, name, ind, self.df['SDSS_NAME_fl'][ind])

        if ind is not None:

            self.ind = ind
            if name is None and self.ErositaTable.columnIndex('SDSS_NAME_fl') is not None:
                row = self.ErositaTable.getRowIndex(column='SDSS_NAME_fl', value=self.df['SDSS_NAME_fl'][ind])
                self.ErositaTable.setCurrentCell(row, 0)
                self.ErositaTable.selectRow(row)
                self.ErositaTable.row_clicked(row=row)

            if x is None and self.plotExt.isChecked():
                self.calc_ext(self.ind, plot=self.plotExt.isChecked(), gal=self.galExt.isChecked())
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
        for k in ['J', 'H', 'K']:
            #print(k, self.df[k + 'MAG_fl'][ind], self.df['ERR_' + k + 'MAG_fl'][ind])
            if not np.isnan(self.df[k + 'MAG_fl'][ind]) and not np.isnan(self.df['ERR_' + k + 'MAG_fl'][ind]):
                self.filters[k] = Filter(self, k, value=self.df[k + 'MAG_fl'][ind], err=self.df['ERR_' + k + 'MAG_fl'][ind])

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

    def axisChanged(self, axis):
        self.parent.options('ero_' + axis, getattr(self, axis).currentText())
        setattr(self, 'ero_' + axis, self.parent.options('ero_' + axis))
        self.updateData()

    def lambdaUVChanged(self):
        self.ero_lUV = float(self.lambdaUV.text())
        self.parent.options('ero_lUV', self.ero_lUV)

    def getSDSSind(self, name):
        ind = np.where(self.df['SDSS_NAME_fl'] == name)[0][0]
        return self.df['PLATE_fl'][ind], self.df['MJD_fl'][ind], self.df['FIBERID_fl'][ind]

    def loadSDSS(self, plate, fiber, mjd, Av_gal=np.nan):
        filename = os.path.dirname(self.parent.ErositaFile) + '/spectra/spec-{0:04d}-{2:05d}-{1:04d}.fits'.format(int(plate), int(fiber), int(mjd))
        if os.path.exists(filename):
            qso = fits.open(filename)
            ext = G03_SMCBar().extinguish(1e4 / np.asarray(10 ** qso[1].data['loglam'][:], dtype=np.float64), Av=Av_gal) if ~np.isnan(Av_gal) else np.ones_like(qso[1].data['loglam'])
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
                spec = self.loadSDSS(d['PLATE_fl'], d['FIBERID_fl'], d['MJD_fl'], Av_gal=d['Av_gal'])
                if spec is not None:
                    mask = (spec[0] > (self.ero_lUV - 10) * (1 + d['z'])) * (
                                spec[0] < (self.ero_lUV + 10) * (1 + d['z'])) * (spec[3] == 0)
                    if np.sum(mask) > 0 and not np.isnan(np.mean(spec[1][mask])) and not np.mean(spec[2][mask]) == 0:
                        k += 1
                        nm = np.nanmean(spec[1][mask]) * 1e-17 * u.erg / u.cm ** 2 / u.AA / u.s
                        wm = np.average(spec[1][mask], weights=spec[2][mask]) * 1e-17 * u.erg / u.cm ** 2 / u.AA / u.s
                        # print(nm, wm)
                        self.df['F_UV'][i] = nm.to(u.erg / u.cm ** 2 / u.s / u.Hz, equivalencies=u.spectral_density(
                            self.ero_lUV * u.AA * (1 + d['z']))).value / (1 + d['z'])
                        self.df[name][i] = nm.to(u.erg / u.cm ** 2 / u.s / u.Hz, equivalencies=u.spectral_density(
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
            self.df['L_X'] = 4 * np.pi * dl ** 2 * self.df['F_X'] #/ (1 + self.df['Z_fl'])

        if 'L_UV' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'L_UV', np.nan)
        self.df['L_UV'] = np.nan

        if 'L_UV_corr' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'L_UV_corr', np.nan)
        self.df['L_UV_corr'] = np.nan

        name = 'L_UV_{0:4d}'.format(int(float(self.lambdaUV.text())))
        if name not in self.df.columns:
            self.df.insert(len(self.df.columns), name, np.nan)
        self.df[name] = np.nan

        if 'F_UV' in self.df.columns:
            self.df['L_UV'] = 4 * np.pi * dl ** 2 * self.df['F_UV'] #/ (1 + self.df['Z_fl'])
            self.df[name] = 4 * np.pi * dl ** 2 * self.df[name.replace('L', 'F')] / (1 + self.df['z'])

        if 'Av_int' in self.df.columns:
            for i, d in self.df.iterrows():
                if np.isfinite(d['Av_int']):
                    self.df['L_UV_corr'][i] = 4 * np.pi * dl[i].value ** 2 * d[name.replace('L', 'F')] / (1 + d['z']) / G03_SMCBar().extinguish(1e4 / np.asarray([float(self.lambdaUV.text())]), Av=d['Av_int'])
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
            self.df['Av_gal'][i] = 3.1 * m.ebv(d['RA_fl'], d['DEC_fl'])

        self.save_data()
        self.updateData()

    def calc_ext(self, ind=None, plot=False, gal=True):

        if gal is None:
            gal = self.galExt.isChecked()

        if 'Av_int' not in self.df.columns:
            self.df.insert(len(self.df.columns), 'Av_int', np.nan)
        if ind is None:
            self.df['Av_int'] = np.nan

        self.load_template_qso(smooth_window=7)
        self.load_template_gal(smooth_window=3)

        if ind is None:
            fmiss = open("temp/av_missed.dat", "w")

        print('calc_ext:', ind)
        for i, d in self.df.iterrows():
            if ind is None or i == int(ind):
                spec = self.loadSDSS(d['PLATE_fl'], d['FIBERID_fl'], d['MJD_fl'], Av_gal=d['Av_gal'])

                self.set_filters(i, clear=True)

                if plot:
                    fig, ax = plt.subplots()
                    ax.plot(spec[0], spec[1], '-k', lw=.5, zorder=2, label='spectrum')
                    for k, f in self.filters.items():
                        print(f.value, f.err, f.filter.get_flux(d[k + 'MAG_fl']))
                        ax.errorbar(f.filter.l_eff, f.filter.get_flux(d[k + 'MAG_fl']) * 1e17,
                                    yerr=(f.filter.get_flux(f.value + f.err) - f.filter.get_flux(f.value)) * 1e17,
                                    marker='s', color=[c/255 for c in f.filter.color])


                if spec is not None:
                    mask = self.calc_mask(spec, z_em=d['z'], iter=3, window=201, clip=2.5)
                    if np.sum(mask) > 50:
                        sm = [np.asarray(spec[0][mask], dtype=np.float64), spec[1][mask], spec[2][mask]]
                        temp = self.load_template_qso(x=sm[0], z_em=d['z'], init=False)
                        temp_gal = self.load_template_gal(x=sm[0], z_em=d['z'], init=False)

                        def fcn2min(params):
                            y = temp * G03_SMCBar().extinguish(1e4 / sm[0] * (1 + d['z']), Av=params.valuesdict()['Av']) * params.valuesdict()['norm']
                            return np.append((y - sm[1]) / sm[2], [f.weight / f.err * (f.value - f.filter.get_value(x=f.x, y=self.load_template_qso(x=f.x, z_em=d['z'], init=False) * G03_SMCBar().extinguish(1e4 / f.x * (1 + d['z']), Av=params.valuesdict()['Av']) * params.valuesdict()['norm'])) for f in self.filters.values()])

                        def fcn2min_gal(params):
                            y = temp * G03_SMCBar().extinguish(1e4 / sm[0] * (1 + d['z']), Av=params.valuesdict()['Av']) * params.valuesdict()['norm'] + temp_gal * params.valuesdict()['norm_gal']
                            return np.append((y - sm[1]) / sm[2], [f.weight / f.err * (f.value - f.filter.get_value(x=f.x, y=self.load_template_qso(x=f.x, z_em=d['z'], init=False) * G03_SMCBar().extinguish(1e4 / f.x * (1 + d['z']), Av=params.valuesdict()['Av']) * params.valuesdict()['norm'])) for f in self.filters.values()])

                        #norm = np.average(sm[1], weights=sm[2]) / np.average(temp)
                        norm = np.nanmean(sm[1]) / np.nanmean(temp)
                        # create a set of Parameters
                        params = Parameters()
                        params.add('Av', value=0.0, min=-5, max=5)
                        params.add('norm', value=norm, min=0, max=1e10)
                        if gal:
                            params.add('norm_gal', value=np.nanmean(sm[1]) / np.nanmean(temp_gal), min=0, max=1e10)
                            fcn2min = fcn2min_gal
                        # do fit, here with leastsq model
                        minner = Minimizer(fcn2min, params, nan_policy='propagate', calc_covar=True)
                        result = minner.minimize()

                        if plot:
                            temp = self.load_template_qso(x=spec[0], z_em=d['z'], init=False)
                            ax.plot(spec[0], temp * result.params['norm'].value, '-', color='tab:blue', zorder=2, label='composite')
                            m = spec[0] / (1 + d['z']) > 1000
                            ax.plot(spec[0][m], temp[m] * G03_SMCBar().extinguish(1e4 / np.asarray(spec[0][m], dtype=np.float64) * (1 + d['z']), Av=result.params['Av'].value) * result.params['norm'].value,
                                    '-', color='tomato', zorder=3, label='comp with ext')
                            for k, f in self.filters.items():
                                temp = self.load_template_qso(x=f.x, z_em=d['z'], init=False)
                                print(k, d[k + 'MAG_fl'], f.filter.get_value(x=f.x, y=temp * G03_SMCBar().extinguish(1e4 / f.x * (1 + d['z']), Av=result.params['Av'].value) * result.params['norm'].value))
                                ax.plot(f.x, temp * G03_SMCBar().extinguish(1e4 / f.x * (1 + d['z']), Av=result.params['Av'].value) * result.params['norm'].value,
                                        '-', color='tomato', zorder=3)
                                    #ax.scatter(f[0].l_eff, f[0].get_value(x=f[0].data.x, y=temp * G03_SMCBar().extinguish(1e4 / f[0].data.x * (1 + d['z']), Av=result.params['Av'].value) * result.params['norm'].value),
                                    #           s=20, marker='o', c=[c/255 for c in f[0].color])

                            if gal:
                                ax.plot(spec[0], self.load_template_gal(x=spec[0], z_em=d['z'], init=False) * result.params['norm_gal'].value, '-', color='tab:purple', zorder=2, label='galaxy')
                            ax.set_title("{0:4d} {1:19s} {2:5d} {3:5d} {4:4d} Av={5:4.2f}".format(i, d['SDSS_NAME_fl'], d['PLATE_fl'], d['MJD_fl'], d['FIBERID_fl'], result.params['Av'].value))
                            #ax.set_title("{0:4d} {1:19s} Av={2:4.2f}".format(i, d['SDSS_NAME_fl'], result.params['Av'].value))
                            if 0:
                                inds = np.where(np.diff(mask))[0]
                                for s, f in zip(range(0, len(inds), 2), range(1, len(inds), 2)):
                                    ax.axvspan(spec[0][inds[s]], spec[0][inds[f]], color='tab:green', alpha=0.3, zorder=1)
                            else:
                                ymin, ymax = ax.get_ylim()[0] * np.ones_like(spec[0]), ax.get_ylim()[1] * np.ones_like(spec[0])
                                ax.fill_between(spec[0], ymin, ymax, where=mask, color='tab:green', alpha=0.3, zorder=0)
                            fig.legend(loc=1, fontsize=16, borderaxespad=2)
                        print(i, result.params['Av'].value)
                        self.df['Av_int'][i] = result.params['Av'].value
                    else:
                        if ind is None:
                            fmiss.write("{0:4d} {1:19s} {2:5d} {3:5d} {4:4d} \n".format(i, self.df['SDSS_NAME_fl'][i], self.df['PLATE_fl'][i], self.df['MJD_fl'][i], self.df['FIBERID_fl'][i]))
        if plot:
            plt.show()

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

    def load_template_qso(self, x=None, z_em=0, smooth_window=None, init=True):
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
                    data = np.genfromtxt('data/SDSS/Slesing2016.dat', skip_header=0, unpack=True)
                    data = data[:, np.logical_or(data[1] != 0, data[2] != 0)]
                    x = data[0][-1] + np.arange(1, int((25000 - data[0][-1]) / 0.4)) * 0.4
                    y = np.power(x / 2500, -1.9) * 6.542031
                    data = np.append(data, [x, y, y / 10], axis=1)
                    self.template_qso = data
            if smooth_window is not None:
                self.template_qso[1] = smooth(self.template_qso[1], window_len=smooth_window, window='hanning', mode='same')

        if x is not None:
            inter = interp1d(self.template_qso[0] * (1 + z_em), self.template_qso[1], bounds_error=False, fill_value=0, assume_sorted=True)
            return inter(x)

    def load_template_gal(self, x=None, z_em=0, smooth_window=None, init=True):
        if init:
            f = fits.open(f"data/SDSS/spDR2-0{23 + self.template_name_gal}.fit")
            self.template_gal = [10 ** (f[0].header['COEFF0'] + f[0].header['COEFF1'] * np.arange(f[0].header['NAXIS1'])), f[0].data[0]]

            if smooth_window is not None:
                self.template_gal[1] = smooth(self.template_gal[1], window_len=smooth_window, window='hanning', mode='same')

        if x is not None:
            inter = interp1d(self.template_gal[0] * (1 + z_em), self.template_gal[1], bounds_error=False, fill_value=0, assume_sorted=True)
            return inter(x)

    def correlate(self):
        self.corr_status = 1 - self.corr_status
        if self.corr_status:
            self.reg = {'all': 'checked', 'selected': 'checked'}
            x, y, = self.df[self.axis_info[self.ero_x_axis][0]], self.df[self.axis_info[self.ero_y_axis][0]]
            m = x.notna() * y.notna() * (x > 0) * (y > 0)
            x, y = self.x_lambda(x[m].to_numpy()), self.y_lambda(y[m].to_numpy())
            print(x, y)
            slope, intercept, r, p, stderr = linregress(x, y)
            print(slope, intercept, r, p, stderr)
            reg = lambda x: intercept + slope * x
            xreg = np.asarray([np.min(x), np.max(x)])
            self.reg['all'] = "{0:.3f} x + {1:.3f}, disp={2:.2f}".format(slope, intercept, np.std(y - reg(x)))
            self.dataPlot.plotRegression(x=xreg, y=reg(xreg), name='all')
            print(np.sum(np.logical_and(np.isfinite(self.x[self.mask]), np.isfinite(self.y[self.mask]))))
            if np.sum(np.logical_and(np.isfinite(self.x[self.mask]), np.isfinite(self.y[self.mask]))):
                x, y, = self.df[self.axis_info[self.ero_x_axis][0]], self.df[self.axis_info[self.ero_y_axis][0]]
                m = x.notna() * y.notna() * (x > 0) * (y > 0) * self.mask
                x, y = self.x_lambda(x[m].to_numpy()), self.y_lambda(y[m].to_numpy())
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
