import astroplan
import astropy.coordinates
from astropy.cosmology import Planck15
from astropy.io import ascii, fits
from astropy.table import Table
import astropy.time
from astroquery import sdss as aqsdss
from chainconsumer import Chain, ChainConfig, ChainConsumer, PlotConfig, Truth
from collections import OrderedDict
from copy import deepcopy, copy
import corner
import ctypes
import emcee
import h5py
#from importlib import reload
from lmfit import Minimizer, Parameters, report_fit, fit_report, conf_interval, printfuncs, Model
from matplotlib.colors import to_hex
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from multiprocessing import Process
import numpy as np
import pandas as pd
import pickle
import pyGPs
import os
import platform
from PyQt6.QtWidgets import (QApplication, QMessageBox, QMainWindow, QWidget,
                             QFileDialog, QTextEdit, QVBoxLayout, QFontComboBox,
                             QSplitter, QFrame, QLineEdit, QLabel, QPushButton, QCheckBox,
                             QGridLayout, QTabWidget, QFormLayout, QHBoxLayout, QRadioButton,
                             QTreeWidget, QComboBox, QTreeWidgetItem, QAbstractItemView,
                             QStatusBar, QMenu, QButtonGroup, QMessageBox, QToolButton, QColorDialog)
from PyQt6.QtCore import Qt, QPointF, QRectF, QEvent, QUrl, QTimer, pyqtSignal, QObject, QPropertyAnimation, QDir
from PyQt6.QtGui import QDesktopServices, QPainter, QFont, QColor, QIcon
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import argrelextrema
from scipy.special import erf
from scipy.stats import gaussian_kde
#import sfdmap
from shutil import copyfile
import subprocess
import tarfile
#from threading import Thread
from ..a_unc import a
from ..absorption_systems import vel_offset
from ..atomic import *
from ..plot_spec import *
from ..profiles import add_LyaForest, add_ext, add_ext_bump, add_LyaCutoff, convolveflux, tau, dv90
from ..stats import distr1d, distr2d
from ..XQ100 import load_QSO
from .console import *
from .external import spectres
from .erosita import *
from .fit_model import *
from .fit import *
from .graphics import *
from .lines import *
from .sdss_fit import *
from .tables import *
from .obs_tool import *
from .colorcolor import *
from .utils import *

def lnprob(x, pars, prior, self):
    return lnprior(x, pars, prior) + lnlike(x, pars, self)

def lnprior(x, pars, prior):
    lp = 0
    for k, v in prior.items():
        if k in pars:
            lp += v.lnL(x[pars.index(k)])
    return lp

def lnlike(x, pars, self):
    res = True
    for xi, p in zip(x, pars):
        res *= self.parent.fit.setValue(p, xi)
    self.parent.fit.update()
    self.parent.s.calcFit(recalc=True, redraw=False, timer=False)
    chi = self.parent.s.chi2()
    if res and not np.isnan(chi):
        return -chi
    else:
        return -np.inf

class plotSpectrum(pg.PlotWidget):
    """
    class for plotting main spectrum widget
    class for plotting main spectrum widget
    based on pg.PlotWidget
    """
    def __init__(self, parent):
        bottomaxis = pg.AxisItem(orientation='bottom')
        #stringaxis.setTickSpacing(minor=[(10, 0)])
        bottomaxis.setStyle(tickLength=-15, tickTextOffset=2)
        topaxis = pg.AxisItem(orientation='top')
        #topaxis.setStyle(tickLength=-15, tickTextOffset=2, stopAxisAtTick=(True, True))
        pg.PlotWidget.__init__(self, axisItems={'bottom': bottomaxis, 'top': topaxis}, background=(29,29,29))
        #self.vb = pg.PlotWidget.getPlotItem(self).getViewBox()
        self.parent = parent
        self.initstatus()
        self.vb = self.getViewBox()
        self.customMenu = True
        self.vb.setMenuEnabled(not self.customMenu)
        self.vb.disableAutoRange()
        self.regions = regionList(self)
        self.cursorpos = pg.TextItem(anchor=(0, 1))
        self.vb.addItem(self.cursorpos, ignoreBounds=True)
        self.specname = pg.TextItem(anchor=(1, 1))
        self.vb.addItem(self.specname, ignoreBounds=True)
        self.w_region = None
        self.menu = None  # Override pyqtgraph ViewBoxMenu
        self.menu = self.getMenu()  # Create the menu

        self.v_axis = pg.ViewBox(enableMenu=False)
        self.v_axis.setYLink(self)  #this will synchronize zooming along the y axis
        self.showAxis('top')
        self.scene().addItem(self.v_axis)
        self.v_axis.setGeometry(self.getPlotItem().sceneBoundingRect())
        self.getAxis('top').setStyle(tickLength=-15, tickTextOffset=2, stopAxisAtTick=(False, False))
        self.getAxis('top').linkToView(self.v_axis)
        self.getPlotItem().sigRangeChanged.connect(self.updateVelocityAxis)

    def initstatus(self):
        for l in 'abcdehiklmprsuwxyz':
            setattr(self, l+"_status", False)
        self.mouse_moved = False
        self.saveState = None
        self.addline = None
        self.doublet = [None, None]
        self.doublets = doubletList(self)
        self.pcRegions = []
        self.instr_file = None
        self.instr_plot = None
        self.showfullfit = False
        self.restframe = True
    
    def set_range(self, x1, x2):
        self.vb.disableAutoRange()
        s = self.parent.s[self.parent.s.ind].spec
        mask = np.logical_and(s.x() > x1, s.x() < x2)
        self.vb.setRange(xRange=(x1, x2), yRange=(np.min(s.y()[mask]), np.max(s.y()[mask])))
    
    def updateVelocityAxis(self):
        self.v_axis.setGeometry(self.getPlotItem().sceneBoundingRect())
        self.v_axis.linkedViewChanged(self.getViewBox(), self.v_axis.YAxis)
        MainPlotXMin, MainPlotXMax = self.viewRange()[0]
        if self.restframe:
            AuxPlotXMin, AuxPlotXMax = MainPlotXMin / (self.parent.z_abs + 1), MainPlotXMax / (self.parent.z_abs + 1)
        else:
            AuxPlotXMin = (MainPlotXMin / (self.parent.z_abs + 1) / self.parent.abs.reference.line.l() - 1) * ac.c.to('km/s').value
            AuxPlotXMax = (MainPlotXMax / (self.parent.z_abs + 1) / self.parent.abs.reference.line.l() - 1) * ac.c.to('km/s').value
        self.v_axis.setXRange(AuxPlotXMin, AuxPlotXMax, padding=0)

    def raiseContextMenu(self, ev):
        """
        Raise the context menu
        """
        menu = self.getMenu()
        menu.popup(ev.globalPosition().toPoint())

    def getMenu(self):
        """
        Create the menu
        """
        if self.menu is None:
            self.menu = QMenu()
            self.menu.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
            self.viewAll = QAction("View all", self.menu)
            self.viewAll.triggered.connect(self.autoRange)
            self.menu.addAction(self.viewAll)
            self.menu.addSeparator()

            self.export = QMenu("Export...", self.menu)
            self.menu.addMenu(self.export)

            self.exportLine = QAction("Line tau", self.export)
            self.export.addAction(self.exportLine)
            self.exportLine.triggered.connect(self.showExportLine)
            self.exportDialog = None

        return self.menu

    def showExportLine(self):
        if self.exportDialog is None:
            folder = QFileDialog.getExistingDirectory(self, 'Select export folder...', self.parent.work_folder)
            #fname = QFileDialog.getSaveFileName(self, 'Export line tau', self.parent.work_folder)
            self.parent.exportLine(folder)

        #self.exportDialog.show() #self.contextMenuItem)

    def keyPressEvent(self, event):
        super(plotSpectrum, self).keyPressEvent(event)
        key = event.key()

        if not event.isAutoRepeat():

            if event.key() == Qt.Key.Key_Down or event.key() == Qt.Key.Key_Right:
                if self.e_status:
                    self.parent.s.setSpec(self.parent.s.ind + 1)

                if self.p_status:
                    self.parent.fitPoly(np.max([0, self.parent.polyDeg-1]))

            if event.key() == Qt.Key.Key_Up or event.key() == Qt.Key.Key_Left:
                if self.e_status:
                    self.parent.s.setSpec(self.parent.s.ind - 1)

                if self.p_status:
                    self.parent.fitPoly(self.parent.polyDeg + 1)

            if event.key() == Qt.Key.Key_A:
                if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier):
                    self.parent.fit.delSys(self.parent.comp)
                    try:
                        self.parent.fitModel.tab.removeTab(self.parent.comp)
                        for i in range(self.parent.fitModel.tabNum):
                            self.parent.fitModel.tab.setTabText(i, "sys {:}".format(i + 1))
                            self.parent.fitModel.tab.widget(i).ind = i
                    except:
                        pass
                    self.parent.comp -= 1
                    self.parent.s.refreshFitComps()
                    self.parent.showFit(all=self.showfullfit)
                    try:
                        self.parent.fitModel.onTabChanged()
                    except:
                        pass
                else:
                    self.vb.setMouseMode(self.vb.RectMode)
                    self.vb.rbScaleBox.hide()
                    self.a_status = True

            if event.key() == Qt.Key.Key_B:
                if not self.parent.normview:
                    self.vb.setMouseMode(self.vb.RectMode)
                    self.b_status = True
                    self.mouse_moved = False
                    self.parent.statusBar.setText('B-spline mode' )
            
            if event.key() == Qt.Key.Key_C:

                if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier):
                    l = ['all', 'one', 'none']
                    ind = l.index(self.parent.comp_view)
                    ind += 1
                    if ind > 2:
                        ind = 0
                    print(ind, self.parent.comp_view)
                    self.parent.comp_view = l[ind]
                    d = {0: "Don't show components", 1: "Show only selected component", 2: "Show all components"}
                    self.parent.statusBar.setText(d[ind])
                    self.parent.s.redraw()
                else:
                    self.c_status = 1
                    self.vb.setMouseMode(self.vb.RectMode)
                    self.vb.rbScaleBox.hide()

            if event.key() == Qt.Key.Key_D:
                self.vb.setMouseMode(self.vb.RectMode)
                self.d_status = True
                self.parent.statusBar.setText('Points selection mode')

            if event.key() == Qt.Key.Key_E:
                if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier):
                    self.parent.s.remove(self.parent.s.ind)
                    self.e_status = False

                if self.w_region is not None and not event.isAutoRepeat():
                    for attr in ['w_region', 'w_label']:
                        if hasattr(self, attr):
                            self.vb.removeItem(getattr(self, attr))
                    self.w_region = None
                else:
                    self.vb.setMouseMode(self.vb.RectMode)
                    self.e_status = True

            if event.key() == Qt.Key.Key_F:
                if (QApplication.keyboardModifiers() != Qt.KeyboardModifier.ControlModifier):
                    if QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier:
                        self.showfullfit = True
                    else:
                        self.showfullfit = False
                    self.parent.showFit(all=self.showfullfit)

            if event.key() == Qt.Key.Key_G:
                self.g_status = True
                self.parent.fitGauss(kind='integrate')

            if event.key() == Qt.Key.Key_H:
                self.h_status = True

            if event.key() == Qt.Key.Key_J:
                self.j_status = True
                self.parent.s[self.parent.s.ind].set_fit_disp(show=True)

            if event.key() == Qt.Key.Key_K:
                self.k_status = True

            if event.key() == Qt.Key.Key_I:
                if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier):
                    print('initial', self.parent.s[self.parent.s.ind].g_line)
                    self.parent.s[self.parent.s.ind].g_line.initial()
                else:
                    self.i_status = True
                    self.parent.statusBar.setText('Estimate the width of Instrument function')

            if event.key() == Qt.Key.Key_L:
                self.l_status = True
                self.parent.statusBar.setText('Lya select')

            if event.key() == Qt.Key.Key_M:
                if (QApplication.keyboardModifiers() != Qt.KeyboardModifier.ControlModifier):
                    self.m_status = True
                    self.parent.statusBar.setText('Spectrum redin mode')

            if event.key() == Qt.Key.Key_N:
                self.parent.normalize(not self.parent.panel.normalize.isChecked())

            if event.key() == Qt.Key.Key_P:
                self.p_status = True
                self.parent.statusBar.setText('Add partial coverage region')

            if event.key() == Qt.Key.Key_R:
                if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier):
                    pass
                    #self.parent.showResiduals.toggle()
                    #self.parent.showResidualsPanel()
                elif (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier):
                    self.restframe = 1 - self.restframe
                    if self.restframe:
                        self.parent.abs.set_reference()
                    else:
                        self.parent.abs.set_reference(self.parent.abs.reference)

                    self.updateVelocityAxis()
                else:
                    self.vb.setMouseMode(self.vb.RectMode)
                    self.r_status = True
                    self.parent.statusBar.setText('Set region mode')
                    #self.vb.removeItem(self.w_label)
               
            if event.key() == Qt.Key.Key_S:
                self.vb.setMouseMode(self.vb.RectMode)
                self.s_status = True
                self.parent.statusBar.setText('Points selection mode')

            if event.key() == Qt.Key.Key_T:
                if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier):
                    if self.parent.blindMode:
                        if self.parent.fitResults is None:
                            self.parent.showFitResults()
                        else:
                            self.parent.fitResults.close()
                    else:
                        self.sendMessage("Blind mode is on. Disable it in Preference menu (F11)")

            if event.key() == Qt.Key.Key_Q:
                if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier):
                    pass
                else:
                    self.parent.calc_cont()

            if event.key() == Qt.Key.Key_U:
                self.u_status += 1
                self.parent.statusBar.setText('Find doublet mode')

            if event.key() == Qt.Key.Key_V:
                self.parent.s[self.parent.s.ind].remove()
                sl = ['step', 'steperr', 'line', 'lineerr', 'point', 'pointerr']
                self.parent.specview = sl[(sl.index(self.parent.specview)+1)*int((sl.index(self.parent.specview)+1) < len(sl))]
                self.parent.options('specview', self.parent.specview)
                self.parent.s[self.parent.s.ind].initGUI()
                
            if event.key() == Qt.Key.Key_W:
                if self.w_region is not None and not event.isAutoRepeat():
                    for attr in ['w_region', 'w_label']:
                        if hasattr(self, attr):
                            self.vb.removeItem(getattr(self, attr))
                    self.w_region = None
                else:
                    self.vb.setMouseMode(self.vb.RectMode)
                    self.w_status = True

            if event.key() == Qt.Key.Key_X:
                self.vb.setMouseMode(self.vb.RectMode)
                self.x_status = True
                self.parent.statusBar.setText('Select bad pixels mode')

            if event.key() == Qt.Key.Key_Y:
                self.y_status = True

            if event.key() == Qt.Key.Key_Z:
                if (QApplication.keyboardModifiers() != Qt.KeyboardModifier.ControlModifier):
                    self.vb.setMouseMode(self.vb.RectMode)
                    self.z_status = True
                    self.parent.statusBar.setText('Zooming mode')
                    if not event.isAutoRepeat():
                        self.saveState = self.vb.getState()
                    self.parent.s[self.parent.s.ind].g_line.initial()
                else:
                    if self.saveState is not None:
                        if 1:
                            a = np.array(self.saveState['targetRange']).flatten()
                            self.vb.setRange(QRectF(a[0], a[2], a[1]-a[0], a[3]-a[2]))
                        else:
                            self.vb.setState(self.saveState)
                #vb = self.plot.getPlotItem(self).getViewBox()
                #vb.setMouseMode(ViewBox.RectMode)
        else:
            if event.key() == Qt.Key.Key_C:
                if self.c_status == 2:
                    self.vb.setMouseMode(self.vb.RectMode)
                    self.vb.rbScaleBox.hide()

        if event.key() in [Qt.Key.Key_Right, Qt.Key.Key_Left]:
            if not self.e_status and not self.p_status:
                self.parent.setz_abs(self.parent.z_abs + (-1 + 2 * (event.key() == Qt.Key.Key_Right))
                                     * (self.viewRange()[0][-1] - self.viewRange()[0][0]) / (np.sum(self.viewRange()[0]) / 2) / 3000 * (1 + 9 * (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier))
                                     * (self.parent.z_abs + 1))


    def keyReleaseEvent(self, event):

        if not event.isAutoRepeat():

            if event.key() == Qt.Key.Key_Left:
                if self.c_status:
                    self.switch_component(-1)
                    self.c_status = 2

            if event.key() == Qt.Key.Key_Right:
                if self.c_status:
                    self.switch_component(1)
                    self.c_status = 2

            if event.key() == Qt.Key.Key_A:
                self.a_status = False

            if event.key() == Qt.Key.Key_B:
                self.b_status = False
                if not self.mouse_moved:
                    if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.AltModifier):
                        spec = self.parent.s[self.parent.s.ind].spec
                        imin, imax = np.argmin(np.abs(spec.x() - (self.mousePoint.x() - (self.viewRange()[0][-1] - self.viewRange()[0][0]) / 40))), np.argmin(np.abs(spec.x() - (self.mousePoint.x() + (self.viewRange()[0][-1] - self.viewRange()[0][0]) / 40)))
                        med = np.median(spec.raw.y[imin:imax]) if imax > imin + 3 else np.median(spec.raw.y[(imin + imax) % 2 - 1: (imin + imax) % 2 + 1])
                        self.parent.s[self.parent.s.ind].add_spline(self.mousePoint.x(), med)
                    else:
                        self.parent.s[self.parent.s.ind].add_spline(self.mousePoint.x(), self.mousePoint.y())

            if event.key() == Qt.Key.Key_C:
                if (QApplication.keyboardModifiers() != Qt.KeyboardModifier.ShiftModifier):
                    if self.c_status == 1:
                        self.switch_component(1)
                    self.c_status = False

            if event.key() == Qt.Key.Key_D:
                self.d_status = False

            if event.key() == Qt.Key.Key_E:
                self.e_status = False

            if event.key() == Qt.Key.Key_H:
                self.h_status = False

            if event.key() == Qt.Key.Key_I:
                self.i_status = False

            if event.key() == Qt.Key.Key_K:
                self.k_status = False

            if event.key() == Qt.Key.Key_J:
                self.parent.s[self.parent.s.ind].set_fit_disp(show=False)
                self.j_status = False

            if event.key() == Qt.Key.Key_L:
                self.l_status = False

            if event.key() == Qt.Key.Key_M:
                self.m_status = False

            if event.key() == Qt.Key.Key_O:
                self.parent.UVESSetup_status += 1
                if self.parent.UVESSetup_status > len(self.parent.UVESSetups):
                    self.parent.UVESSetup_status = 0
                self.parent.chooseUVESSetup()

            if event.key() == Qt.Key.Key_R:
                self.r_status = False
                self.regions.sortit()

            if event.key() == Qt.Key.Key_S:
                self.s_status = False

            if event.key() == Qt.Key.Key_P:
                self.p_status = False

            if event.key() == Qt.Key.Key_U:
                if self.u_status:
                    if len(self.doublets) == 0 or self.doublets[-1].temp is None:
                        self.doublets.append(Doublet(self))
                        self.doublets[-1].draw_temp(self.mousePoint.x())
                    else:
                        self.doublets[-1].find(self.doublets[-1].line_temp.value(), self.mousePoint.x())
                        self.doublets.update()
                self.u_status = False

            if event.key() == Qt.Key.Key_W:
                self.w_status = False

            if event.key() == Qt.Key.Key_X:
                self.x_status = False

            if event.key() == Qt.Key.Key_Y:
                self.y_status = False

            if event.key() == Qt.Key.Key_Z:
                self.z_status = False
        
            if any([event.key() == getattr(Qt.Key, 'Key_'+s) for s in 'ABCDERSWXZ']):
                #if self.vb.rbScaleBox in self.vb.addedItems:
                #    self.vb.removeItem(self.vb.rbScaleBox())
                self.vb.rbScaleBox.hide()
                self.vb.setMouseMode(self.vb.PanMode)
                self.parent.statusBar.setText('')

        if event.isAccepted():
            super(plotSpectrum, self).keyReleaseEvent(event)

    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self.menuEnabled():
            event.accept()
            self.raiseContextMenu(event)

    def mouseDoubleClickEvent(self, event):
        super(plotSpectrum, self).mouseDoubleClickEvent(event)
        if self.l_status:
            self.doublets.append(Doublet(self, name='Ly', z=self.mousePoint.x() / 1215.6701 - 1))
            self.doublets.update()

    def mousePressEvent(self, event):
        super(plotSpectrum, self).mousePressEvent(event)

        self.mousePoint_saved = self.vb.mapSceneToView(event.position())

        if self.r_status and event.button() == Qt.MouseButton.LeftButton:
            self.r_status = 2
            self.regions.add(sort=False)

    def mouseReleaseEvent(self, event):
        if any([getattr(self, s+'_status') for s in 'abcdersuwx']):
            self.vb.rbScaleBox.hide()
            self.vb.setMouseMode(self.vb.PanMode)
            event.accept()
        else:
            if event.button() == Qt.MouseButton.RightButton and self.menuEnabled() and self.customMenu:
                if self.mousePoint == self.mousePoint_saved:
                    self.raiseContextMenu(event)
                    event.accept()

        if self.a_status:
            if self.mousePoint == self.mousePoint_saved:
                if self.parent.abs.reference.line.name in self.parent.fit.sys[self.parent.comp].sp:
                    self.parent.fit.addSys(self.parent.comp)
                    self.parent.fit.sys[-1].z.val = self.mousePoint.x() / self.parent.abs.reference.line.l() - 1
                    self.parent.fit.sys[-1].zrange(200)
                    self.parent.comp = len(self.parent.fit.sys) - 1
                    if self.parent.fitModel is not None:
                        sys = fitModelSysWidget(self.parent.fitModel, len(self.parent.fit.sys) - 1)
                        self.parent.fitModel.tab.addTab(sys, "sys {:}".format(self.parent.fitModel.tabNum))
                        self.parent.fitModel.tab.setCurrentIndex(len(self.parent.fit.sys) - 1)
                    self.parent.s.refreshFitComps()
                    self.parent.showFit(all=self.showfullfit)
                else:
                    self.a_status = True
                    self.parent.sendMessage("Select a reference absorption line")
        if self.b_status:
            if event.button() == Qt.MouseButton.LeftButton:
                if self.mousePoint == self.mousePoint_saved:
                    if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.AltModifier):
                        spec = self.parent.s[self.parent.s.ind].spec
                        imin, imax = np.argmin(np.abs(spec.x() - (self.mousePoint.x() - (self.viewRange()[0][-1] - self.viewRange()[0][0]) / 40))), np.argmin(np.abs(spec.x() - (self.mousePoint.x() + (self.viewRange()[0][-1] - self.viewRange()[0][0]) / 40)))
                        med = np.median(spec.raw.y[imin:imax]) if imax > imin + 3 else np.median(spec.raw.y[(imin+imax) % 2-1: (imin+imax) % 2+1])
                        self.parent.s[self.parent.s.ind].add_spline(self.mousePoint.x(), med)
                    else:
                        self.parent.s[self.parent.s.ind].add_spline(self.mousePoint.x(), self.mousePoint.y())
                else:
                    self.parent.s[self.parent.s.ind].del_spline(self.mousePoint_saved.x(), self.mousePoint_saved.y(), self.mousePoint.x(), self.mousePoint.y())

            if event.button() == Qt.MouseButton.RightButton:
                ind = self.parent.s[self.parent.s.ind].spline.find_nearest(self.mousePoint.x(), None)
                self.parent.s[self.parent.s.ind].del_spline(arg=ind)
                event.accept()

        if self.c_status:
            #try:
            if hasattr(self.parent.abs, 'reference') and self.parent.abs.reference.line.name in self.parent.fit.sys[self.parent.comp].sp:
                if self.check_line():
                    self.parent.fit.sys[self.parent.comp].z.set(self.mousePoint.x() / self.parent.abs.reference.line.l() - 1)
                    if self.mousePoint.y() != self.mousePoint_saved.y():
                        sp = self.parent.fit.sys[self.parent.comp].sp[self.parent.abs.reference.line.name]
                        # sp.b.set(sp.b.val + (self.mousePoint_saved.x() / self.mousePoint.x() - 1) * 299794.26)
                        sp.N.set(sp.N.val + np.sign(self.mousePoint_saved.y() - self.mousePoint.y()) * 2 * np.abs((self.mousePoint_saved.y() - self.mousePoint.y()) / (self.viewRange()[1][-1] - self.viewRange()[1][0])))
                    try:
                        self.parent.fitModel.refresh()
                    except:
                        pass
                    self.c_status = 2
                else:
                    self.parent.sendMessage('select line within window wavelength range', 3500)
                    self.vb.setMouseMode(self.vb.PanMode)
                    self.c_status = 0

                self.parent.s.prepareFit(self.parent.comp, all=self.showfullfit)
                self.parent.s.calcFit(self.parent.comp, recalc=True, redraw=True)
                self.parent.s.calcFit(recalc=True, redraw=True)

            else:
                self.c_status = 0
            #    pass

        if self.i_status:
            self.parent.console.exec_command('show HI')
            self.parent.abs.redraw(z=self.mousePoint.x() / 1215.6701 - 1)
            self.parent.s[self.parent.s.ind].mask.raw.x = np.zeros_like(self.parent.s[self.parent.s.ind].mask.raw.x)
            self.parent.s[self.parent.s.ind].mask.normalize(norm=True)
            self.parent.s[self.parent.s.ind].auto_select(self.mousePoint.x())
            self.parent.fit = fitPars(self.parent)
            self.parent.fit.addSys(z=self.mousePoint.x() / 1215.6701 - 1)
            self.parent.fit.sys[0].addSpecies('HI')
            self.parent.fit.sys[0].sp['HI'].b.set(1)
            self.parent.fit.sys[0].sp['HI'].b.vary = False
            self.parent.fit.sys[0].sp['HI'].N.set(14.5)
            self.parent.fit.add('res')
            self.parent.fit.res.set(7000)
            self.parent.fitLM()

        if self.l_status:
            self.parent.console.exec_command('show HI')
            self.parent.abs.redraw(z=self.mousePoint.x() / 1215.6701 - 1)

        if self.p_status:
            self.doublet[self.p_status-1] = self.mousePoint
            if self.p_status == 2:
                self.add_pcRegion(self.doublet[0], self.doublet[1])
            self.p_status = 1 if self.p_status == 2 else 2

        if self.r_status:
            if event.button() == Qt.MouseButton.LeftButton:
                self.r_status = False
                if np.abs(self.mousePoint_saved.x() - self.mousePoint.x()) > (self.viewRange()[0][-1] - self.viewRange()[0][0]) / 100:
                    self.regions.sortit()
                else:
                    self.regions.remove(self.regions[-1])

        if self.s_status or self.d_status:
            for s in self.parent.s:
                #if QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier or i == self.parent.s.ind:
                if QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier or s.active():
                    s.add_points(self.mousePoint_saved.x(), self.mousePoint_saved.y(), self.mousePoint.x(), self.mousePoint.y(), remove=self.d_status, redraw=False)
                    #self.parent.s[i].add_points(self.mousePoint_saved.x(), self.mousePoint_saved.y(), self.mousePoint.x(), self.mousePoint.y(), remove=False)
                    s.set_fit_mask()
                    s.update_points()
                    s.set_res()
            self.parent.s.chi2()
            #self.vb.removeItem(self.vb._rbScaleBox)
            self.vb.setMouseMode(self.vb.PanMode)

        if self.u_status:
            if self.u_status == 1 and self.mousePoint.x() == self.mousePoint_saved.x() and self.mousePoint.y() == self.mousePoint_saved.y():
                if len(self.doublets) == 0 or self.doublets[-1].temp is None:
                    self.doublets.append(Doublet(self))
                    self.doublets[-1].draw_temp(self.mousePoint.x())
                else:
                    self.doublets[-1].find(self.doublets[-1].line_temp.value(), self.mousePoint.x())
                    self.doublets.update()
                    self.u_status = False
                #self.u_status += 1

        if self.w_status or self.e_status:
            for attr in ['w_region', 'w_label']:
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    self.vb.removeItem(getattr(self, attr))

            s = self.parent.s[self.parent.s.ind]
            mask = np.logical_and(s.spec.x() > min(self.mousePoint.x(), self.mousePoint_saved.x()),
                                  s.spec.x() < max(self.mousePoint.x(), self.mousePoint_saved.x()))
            if np.sum(mask) > 0:
                curve1 = plotLineSpectrum(parent=s, view='step', name='EW', x=s.spec.x()[mask], y=s.spec.y()[mask], pen=pg.mkPen())
                x, y = curve1.returnPathData()
                if len(s.cont.x) > 0 and s.cont.x[0] < x[0] and s.cont.x[-1] > x[-1]:
                    if QApplication.keyboardModifiers() != Qt.KeyboardModifier.ShiftModifier:
                        if self.parent.normview:
                            cont = interp1d(x, np.ones_like(x), fill_value=1)
                        else:
                             s.cont.interpolate()
                             cont = s.cont.inter
                    else:
                        s.fit.interpolate()
                        cont = s.fit.inter
                    curve2 = pg.PlotCurveItem(x=x, y=cont(x), pen=pg.mkPen())
                    if self.w_status:
                        w = np.trapz(1.0 - y / cont(x), x=x)
                        err_w = np.sqrt(np.sum((s.spec.err()[mask] / cont(x)[:-1:2] * np.diff(x)[::2])**2))
                        text = 'w = {0:0.5f}+/-{1:0.5f}'.format(w, err_w)
                    else:
                        w = np.trapz(y - cont(x), x=x)
                        err_w = np.sqrt(np.sum(((s.spec.err()[mask] - cont(x)[:-1:2]) * np.diff(x)[::2])**2))
                        text = 'flux = {0:0.5f}+/-{1:0.5f}'.format(w, err_w)

                    self.w_region = pg.FillBetweenItem(curve1, curve2, brush=pg.mkBrush(44, 160, 44, 150))
                    self.vb.addItem(self.w_region)
                    if hasattr(self.parent.abs, 'reference'):
                        if self.w_status:
                            text += ', log(w/l)={0:0.2f}, w_r = {1:0.5f}+/-{2:0.5f}'.format(np.log10(2 * np.abs(w) / (x[0] + x[-1])), w / (1 + self.parent.z_abs), err_w / (1 + self.parent.z_abs))
                            text += dv90((s.spec.x()[mask] / self.parent.abs.reference.line.l() / (1 + self.parent.z_abs) - 1) * ac.c.to('km/s').value,
                                         s.spec.y()[mask] / cont(s.spec.x()[mask]),
                                         s.spec.err()[mask] / cont(s.spec.x()[mask]),
                                         resolution=self.parent.s[self.parent.s.ind].resolution(), plot=1)
                        else:
                            vx = (s.spec.x()[mask] / self.parent.abs.reference.line.l() / (1 + self.parent.z_abs) - 1) * ac.c.to('km/s')
                            y = s.spec.y()[mask] - cont(s.spec.x()[mask])
                            spline = UnivariateSpline(vx.value, y - np.max(y) / 2, s=0)
                            if len(spline.roots()) == 2:
                                r1, r2 = spline.roots()
                                text += ', FWHM={:0.1f}'.format(np.abs(r1 - r2))
                            Sv = np.trapz(y, x=vx) * (1e-17 * u.erg / u.cm ** 2 / u.s / u.AA).to(u.Jy, equivalencies=u.spectral_density(self.parent.abs.reference.line.l() * (1 + self.parent.z_abs) * u.AA))
                            Lnu = 1.04e-3 * Sv.value * (self.parent.abs.reference.line.l() * u.AA).to(u.GHz, equivalencies=u.spectral()).value / (1 + self.parent.z_abs) * Planck15.luminosity_distance(self.parent.z_abs).to('Mpc').value ** 2
                            text += r', log L={:0.1f}'.format(np.log10(Lnu) + 33.58)

                    self.w_label = pg.TextItem(text, anchor=(0, 1), color=(44, 160, 44))
                    self.w_label.setFont(QFont("SansSerif", 14))
                    self.parent.console.set(text)
                    self.w_label.setPos((x[0] + x[-1]) / 2, cont((x[0] + x[-1]) / 2))
                    self.vb.addItem(self.w_label)
                else:
                    w = np.trapz(y, x=x)
                    err_w = np.sqrt(np.sum((s.spec.err()[mask] * np.diff(x)[::2]) ** 2))
                    self.w_region = pg.FillBetweenItem(curve1, pg.PlotCurveItem(x=x, y=np.zeros_like(x), pen=pg.mkPen()), brush=pg.mkBrush(44, 160, 44, 150))
                    self.vb.addItem(self.w_region)
                    text = 'flux = {0:0.5f}+/-{1:0.5f}'.format(w, err_w)
                    self.w_label = pg.TextItem(text, anchor=(0, 1), color=(44, 160, 44))
                    self.w_label.setFont(QFont("SansSerif", 14))
                    self.parent.console.set(text)
                    self.w_label.setPos((x[0] + x[-1]) / 2, np.max(y))
                    self.vb.addItem(self.w_label)

        if self.x_status:
            for s in self.parent.s:
                #if QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier or i == self.parent.s.ind:
                if (Qt.KeyboardModifier.ControlModifier in QApplication.keyboardModifiers()) or s.active():
                    s.add_points(self.mousePoint_saved.x(), self.mousePoint_saved.y(), self.mousePoint.x(), self.mousePoint.y(), remove=(Qt.KeyboardModifier.ShiftModifier in QApplication.keyboardModifiers()), bad=True)

        if event.isAccepted():
            super(plotSpectrum, self).mouseReleaseEvent(event)
            
    def mouseMoveEvent(self, event):
        super(plotSpectrum, self).mouseMoveEvent(event)
        self.mousePoint = self.vb.mapSceneToView(event.position())
        self.mouse_moved = True
        self.cursorpos.setText('x={0:.3f}, y={1:.2f}, rest={2:.3f}'.format(self.mousePoint.x(), self.mousePoint.y(), self.mousePoint.x()/(1+self.parent.z_abs)))
        #self.cursorpos.setText("<span style='font-size: 12pt'>x={0:.3f}, <span style='color: red'>y={1:.2f}</span>".format(mousePoint.x(),mousePoint.y()))
        pos = self.vb.sceneBoundingRect()
        self.cursorpos.setPos(self.vb.mapSceneToView(QPointF(int(pos.left()+10), int(pos.bottom()-10))))
        self.specname.setPos(self.vb.mapSceneToView(QPointF(int(pos.right()-10), int(pos.bottom()-10))))
        if self.r_status == 2 and event.type() == QEvent.Type.MouseMove:
            self.regions[-1].setRegion([self.mousePoint_saved.x(), self.mousePoint.x()])
        if any([getattr(self, s + '_status') for s in 'acr']) and event.type() == QEvent.Type.MouseMove:
            self.vb.rbScaleBox.hide()


    def wheelEvent(self, event):
        if self.c_status:
            if self.check_line():
                sp = self.parent.fit.sys[self.parent.comp].sp[self.parent.abs.reference.line.name]
                if sp.b.addinfo == '':
                    sp.b.set(sp.b.val * np.power(1.2, np.sign(event.angleDelta().y())))
                else:
                    self.parent.fit.sys[self.parent.comp].sp[sp.b.addinfo].b.set(sp.b.val * np.power(1.2, np.sign(event.angleDelta().y())))
                self.parent.s.prepareFit(self.parent.comp, all=self.showfullfit)
                self.parent.s.calcFit(self.parent.comp, recalc=True, redraw=True)
                self.parent.s.calcFit(recalc=True, redraw=True)
                try:
                    self.parent.fitModel.refresh()
                except:
                    pass
                event.accept()
                self.c_status = 2
            else:
                self.parent.sendMessage('select line within window wavelength range', 3500)
                self.vb.setMouseMode(self.vb.PanMode)
                self.c_status = 0

        elif self.m_status:
            self.parent.s[self.parent.s.ind].rebinning(np.power(2.0, np.sign(event.angleDelta().y())))
            self.parent.statusBar.setText(f'Spectrum rebinned by factor {self.parent.s[self.parent.s.ind].spec_factor}')
        else:
            super(plotSpectrum, self).wheelEvent(event)
            pos = self.vb.sceneBoundingRect()
            self.cursorpos.setPos(self.vb.mapSceneToView(QPointF(int(pos.left()+10), int(pos.bottom()-10))))
            self.specname.setPos(self.vb.mapSceneToView(QPointF(int(pos.right()-10), int(pos.bottom()-10))))

    def mouseDragEvent(self, ev):
        
        if ev.button() == Qt.MouseButton.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev)
         
        ev.accept() 
        pos = ev.pos()
        
        if ev.button() == Qt.MouseButton.RightButton:
            self.updateScaleBox(ev.buttonDownPos(), ev.pos())
            
            if ev.isFinish():  
                self.rbScaleBox.hide()
                ax = QRectF(QPointF(ev.buttonDownPos(ev.button())), QPointF(pos))
                ax = self.childGroup.mapRectFromParent(ax) 
                MouseRectCoords =  ax.getCoords()  
                self.dataSelection(MouseRectCoords)      
            else:
                self.updateScaleBox(ev.buttonDownPos(), ev.pos())

    def switch_component(self, n):
        if len(self.parent.fit.sys) > 0:
            self.parent.comp += n
            if self.parent.comp > len(self.parent.fit.sys) - 1:
                self.parent.comp = 0
            if self.parent.comp < 0:
                self.parent.comp = len(self.parent.fit.sys) - 1
            self.parent.componentBar.setText("{:d} component".format(self.parent.comp))
            try:
                self.parent.fitModel.tab.setCurrentIndex(self.parent.comp)
            except:
                pass
            # self.parent.s.redraw()
            self.parent.s.redrawFitComps()
            self.parent.abs.redraw(z=self.parent.fit.sys[self.parent.comp].z.val)

    def updateRegions(self):
        if len(self.regions) > 0:
            for r in self.regions:
                if not r.active:
                    pass
                    #print(r.size_full)
            self.parent.s.apply_regions()

    def check_line(self):
        return (self.vb.getState()['viewRange'][0][0] < self.parent.abs.reference.line.l() * (1 + self.parent.z_abs)) * (self.parent.abs.reference.line.l() * (1 + self.parent.z_abs) < self.vb.getState()['viewRange'][0][1])

    def add_line(self, x, y):
        if self.addline is not None and self.addline in self.vb.addedItems:
            self.vb.removeItem(self.addline)
        self.addline = pg.PlotCurveItem(x=x, y=y, clickable=True)
        #self.add_line.sigClicked.connect(self.specClicked)
        self.addline.setPen(pg.mkPen(255, 69, 0, width=3))
        self.vb.addItem(self.addline)

    def add_doublet(self, x1, x2):
        self.doublets.append(Doublet(self))
        self.doublets[-1].find(x1, x2)

    def add_pcRegion(self, x1=None, x2=None):
        self.pcRegions.append(pcRegion(self, len(self.pcRegions), x1, x2))

    def remove_pcRegion(self, ind=None):
        if len(self.pcRegions) > 0:
            if ind is None:
                for p in reversed(self.pcRegions):
                    p.remove()
            elif isinstance(ind, int):
                self.pcRegions[ind].remove()

    def dataSelection(self,MouseRectCoords):
        print(MouseRectCoords)       

class residualsWidget(pg.PlotWidget):
    """
    class for plotting residual panel tighten with 1d spectrum panel
    """
    def __init__(self, parent):
        bottomaxis = pg.AxisItem(orientation='bottom')
        bottomaxis.setStyle(tickLength=-15, tickTextOffset=2)
        pg.PlotWidget.__init__(self, axisItems={'bottom': bottomaxis}, background=(29, 29, 29))

        self.scene().removeItem(bottomaxis)
        self.parent = parent
        self.vb = self.getViewBox()
        self.vb.enableAutoRange(y=self.vb.YAxis)
        self.setXLink(self.parent.plot)
        self.addLines()
        self.customMenu = True
        self.menu = None
        self.st = []
        self.vb.setMenuEnabled(not self.customMenu)

        # create new plot for kde and link its y axis
        if 0:
            self.kde = pg.PlotItem(axisItems={})
            self.kde.hideAxis('bottom')
            self.kde.setFixedHeight(300)
            self.kde.setFixedWidth(300)
        else:
            self.kde = pg.ViewBox()
            self.kde.setGeometry(self.vb.sceneBoundingRect())
            self.kde.setGeometry(QRectF(25.0, 1.0, 150.0, 458.0))
        self.scene().addItem(self.kde)
        self.kde.setYLink(self)
        #self.getAxis('right').setLabel('axis2', color='#0000ff')

    def addLines(self):
        #self.addItem(pg.InfiniteLine(0.0, 0, pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.PenStyle.DashLine)))
        self.region = pg.LinearRegionItem([-1, 1], orientation=pg.LinearRegionItem.Horizontal, brush=pg.mkBrush(182, 232, 182, 20))
        self.region.setMovable(False)
        for l in self.region.lines:
            l.setPen(pg.mkPen(None))
            l.setHoverPen(pg.mkPen(None))
        self.addItem(self.region)
        levels = [1, 2, 3]
        colors = [(100, 100, 100), (100, 100, 100), (100, 100, 100)]
        widths = [1.5, 1.0, 0.5]
        for l, color, width in zip(levels, colors, widths):
            self.addItem(pg.InfiniteLine(l, 0, pen=pg.mkPen(color=color, width=width, style=Qt.PenStyle.DashLine)))
            self.addItem(pg.InfiniteLine(-l, 0, pen=pg.mkPen(color=color, width=width, style=Qt.PenStyle.DashLine)))

    def struct(self, x=None, y=None, clear=False):
        if clear:
            if hasattr(self, 'st'):
                for st in self.st:
                    for si in st:
                        if si in self.vb.addedItems:
                            self.vb.removeItem(si)
        else:
            if x is not None and y is not None:
                self.st.append([pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(190, 30, 70, 100, width=10)),
                                pg.LinearRegionItem([x[0], x[-1]], orientation=pg.LinearRegionItem.Vertical, movable=False, brush=pg.mkBrush(190, 30, 70, 20), pen=pg.mkPen(190, 30, 70, 0, width=0))])
                self.addItem(self.st[-1][0])
                self.addItem(self.st[-1][-1])

    def viewRangeChanged(self, view, range):
        self.sigRangeChanged.emit(self, range)
        if len(self.parent.s) > 0:
            res = self.parent.s[self.parent.s.ind].res
            mask = np.logical_and(res.x > range[0][0], res.x < range[0][1])
            y = res.y[mask]
            if np.sum(mask) > 3 and not np.isnan(np.sum(y)) and not np.isinf(np.sum(y)):
                kde = gaussian_kde(y)
                kde_x = np.linspace(np.min(y) - 1, np.max(y) + 1, int((np.max(y) - np.min(y))/0.1))
                self.parent.s[self.parent.s.ind].kde_local.setData(x=-kde_x, y=kde.evaluate(kde_x))

    def mouseMoveEvent(self, event):
        super(residualsWidget, self).mouseMoveEvent(event)
        self.mousePoint = self.vb.mapSceneToView(event.position())

    def mouseReleaseEvent(self, event):
        super(residualsWidget, self).mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.RightButton and self.menuEnabled() and self.customMenu:
            if self.mousePoint == self.vb.mapSceneToView(event.position()):
                self.raiseContextMenu(event)
        #event.accept()
        #pass

    def raiseContextMenu(self, ev):
        """
        Raise the context menu
        """
        menu = self.getMenu()
        menu.popup(ev.globalPosition().toPoint())

    def getMenu(self):
        """
        Create the menu
        """
        if self.menu is None:
            self.menu = QMenu()
            self.menu.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
            self.export = QAction("Export...", self.menu)
            self.export.triggered.connect(self.showExportDialog)
            self.exportDialog = None
            self.structure = QAction("Check structure in residuals", self.menu)
            self.structure.triggered.connect(self.parent.resAnal)
            self.menu.addAction(self.structure)
            self.menu.addSeparator()
            self.menu.addAction(self.export)

        return self.menu

    def showExportDialog(self):
        if self.exportDialog is None:
            fname = QFileDialog.getSaveFileName(self, 'Export residuals', self.parent.work_folder)

            s = self.parent.s[self.parent.s.ind]
            print(np.c_[s.res.x, s.res.y])
            if fname[0]:
                np.savetxt(fname[0], np.c_[s.res.x, s.res.y])
                self.parent.statusBar.setText('Residuals is written to ' + fname[0])


class spec2dWidget(pg.PlotWidget):
    """
    class for plotting 2d spectrum panel tight with 1d spectrum panel
    """
    def __init__(self, parent):
        bottomaxis = pg.AxisItem(orientation='bottom')
        #stringaxis.setTickSpacing(minor=[(10, 0)])
        bottomaxis.setStyle(tickLength=-15, tickTextOffset=2)
        topaxis = pg.AxisItem(orientation='top')
        #topaxis.setStyle(tickLength=-15, tickTextOffset=2, stopAxisAtTick=(True, True))
        pg.PlotWidget.__init__(self, axisItems={'bottom': bottomaxis, 'top': topaxis}, background=(29, 29, 29))

        self.initstatus()

        self.scene().removeItem(bottomaxis)
        self.parent = parent
        self.vb = self.getViewBox()
        self.vb.enableAutoRange(y=self.vb.YAxis)
        self.setXLink(self.parent.plot)
        self.vb.setMenuEnabled(False)
        #self.addLines()

        self.slits = []
        self.cursorpos = pg.TextItem(anchor=(0, 1), fill=pg.mkBrush(0, 0, 0, 0.5))
        self.vb.addItem(self.cursorpos, ignoreBounds=True)
        self.levels = pg.TextItem(anchor=(1, 1), fill=pg.mkBrush(0, 0, 0, 0.5))
        self.vb.addItem(self.levels, ignoreBounds=True)

        #self.getAxis('right').setLabel('axis2', color='#0000ff')

    def initstatus(self):
        self.b_status = False
        self.e_status = False
        self.r_status = False
        self.s_status = False
        self.t_status = False
        self.q_status = False
        self.x_status = False
        self.mouse_moved = False

    def addLines(self):
        # self.addItem(pg.InfiniteLine(0.0, 0, pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.PenStyle.DashLine)))
        self.region = pg.LinearRegionItem([-1, 1], orientation=pg.LinearRegionItem.Horizontal,
                                          brush=pg.mkBrush(182, 232, 182, 20))
        self.region.setMovable(False)
        for l in self.region.lines:
            l.setPen(pg.mkPen(None))
            l.setHoverPen(pg.mkPen(None))
        self.addItem(self.region)
        levels = [1, 2, 3]
        colors = [(100, 100, 100), (100, 100, 100), (100, 100, 100)]
        widths = [1.5, 1.0, 0.5]
        for l, color, width in zip(levels, colors, widths):
            self.addItem(pg.InfiniteLine(l, 0, pen=pg.mkPen(color=color, width=width, style=Qt.PenStyle.DashLine)))
            self.addItem(pg.InfiniteLine(-l, 0, pen=pg.mkPen(color=color, width=width, style=Qt.PenStyle.DashLine)))

    def viewRangeChanged(self, view, range):
        self.sigRangeChanged.emit(self, range)

    def keyPressEvent(self, event):
        super(spec2dWidget, self).keyPressEvent(event)
        key = event.key()

        if not event.isAutoRepeat():

            if event.key() == Qt.Key.Key_Down or event.key() == Qt.Key.Key_Right:
                if self.e_status:
                    self.parent.s.setSpec(self.parent.s.ind + 1)

            if event.key() == Qt.Key.Key_Up or event.key() == Qt.Key.Key_Left:
                if self.e_status:
                    self.parent.s.setSpec(self.parent.s.ind - 1)

            if event.key() == Qt.Key.Key_B:
                self.vb.setMouseMode(self.vb.RectMode)
                self.b_status = True
                self.mouse_moved = False

            if event.key() == Qt.Key.Key_E:
                if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier):
                    self.parent.s.remove(self.parent.s.ind)
                    self.e_status = False
                else:
                    self.e_status = True
                    if self.parent.s[self.parent.s.ind].err2d is not None:
                        self.parent.s[self.parent.s.ind].err2d.setLevels(self.parent.s[self.parent.s.ind].spec2d.raw.err_levels)
                        self.vb.addItem(self.parent.s[self.parent.s.ind].err2d)

            if event.key() == Qt.Key.Key_M:
                if self.parent.s[self.parent.s.ind].mask2d is not None:
                    self.vb.addItem(self.parent.s[self.parent.s.ind].mask2d)

            if event.key() == Qt.Key.Key_R:
                self.r_status = True
                self.vb.setMouseMode(self.vb.RectMode)

            if event.key() == Qt.Key.Key_S:
                self.s_status = True
                self.vb.setMouseEnabled(x=False, y=False)

            if event.key() == Qt.Key.Key_T:
                self.t_status = True
                self.vb.setMouseMode(self.vb.RectMode)
                if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier):
                    for s in self.slits:
                        self.vb.removeItem(s[0])
                        self.vb.removeItem(s[1])
                    self.slits = []

            if event.key() == Qt.Key.Key_Q:
                self.q_status = True
                if self.parent.s[self.parent.s.ind].sky2d is not None:
                    self.parent.s[self.parent.s.ind].sky2d.setLevels(self.parent.s[self.parent.s.ind].spec2d.raw.z_levels)
                    self.vb.addItem(self.parent.s[self.parent.s.ind].sky2d)

            if event.key() == Qt.Key.Key_X:
                self.x_status = True
                self.vb.setMouseMode(self.vb.RectMode)
                s = self.parent.s[self.parent.s.ind].spec2d
                if s.cr is None:
                    self.parent.s[self.parent.s.ind].spec2d.cr = image(x=s.raw.x, y=s.raw.y, mask=np.zeros_like(s.raw.z))
                if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.AltModifier):
                    self.vb.removeItem(self.parent.s[self.parent.s.ind].cr_mask2d)


    def keyReleaseEvent(self, event):
        #super(spec2dWidget, self).keyReleaseEvent(event)

        if not event.isAutoRepeat():

            if event.key() == Qt.Key.Key_B:
                self.b_status = False
                if not self.mouse_moved:
                    self.parent.s[self.parent.s.ind].add_spline(self.mousePoint.x(), self.mousePoint.y(), name='2d')
                print('keyRelease', self.b_status)

            if event.key() == Qt.Key.Key_E:
                self.e_status = False
                if self.parent.s[self.parent.s.ind].err2d is not None and self.parent.s[self.parent.s.ind].err2d in self.vb.addedItems:
                    self.vb.removeItem(self.parent.s[self.parent.s.ind].err2d)

            if event.key() == Qt.Key.Key_M:
                if self.parent.s[self.parent.s.ind].mask2d is not None and self.parent.s[self.parent.s.ind].mask2d in self.vb.addedItems:
                    self.vb.removeItem(self.parent.s[self.parent.s.ind].mask2d)

            if event.key() == Qt.Key.Key_R:
                self.r_status = False
                self.parent.regions.sortit()

            if event.key() == Qt.Key.Key_S:
                self.s_status = False

            if event.key() == Qt.Key.Key_T:
                self.t_status = False

            if event.key() == Qt.Key.Key_Q:
                self.q_status = False
                if self.parent.s[self.parent.s.ind].sky2d is not None and self.parent.s[self.parent.s.ind].sky2d in self.vb.addedItems:
                    self.vb.removeItem(self.parent.s[self.parent.s.ind].sky2d)

            if event.key() == Qt.Key.Key_X:
                self.x_status = False
                #self.vb.addItem(self.parent.s[self.parent.s.ind].cr_mask2d)
                self.parent.s.redraw()

            if any([event.key() == getattr(Qt.Key, 'Key_'+s) for s in ['S']]):
                self.vb.setMouseEnabled(x=True, y=True)

            if any([event.key() == getattr(Qt.Key, 'Key_' + s) for s in 'BRSTX']):
                self.vb.setMouseMode(self.vb.PanMode)
                self.parent.statusBar.setText('')

        if event.isAccepted():
            super(spec2dWidget, self).keyReleaseEvent(event)


    def mousePressEvent(self, event):
        super(spec2dWidget, self).mousePressEvent(event)
        if self.s_status:
            self.s_status = 2

        if any([getattr(self, s + '_status') for s in 'brtsx']):
            self.mousePoint_saved = self.vb.mapSceneToView(event.position())

        if self.t_status:
            self.t_status = 1

        if self.q_status:
            self.q_status = False
            self.mousePoint = self.vb.mapSceneToView(event.position())
            s = self.parent.s[self.parent.s.ind].spec2d
            x = s.raw.x[np.argmin(np.abs(self.mousePoint.x() - s.raw.x))]
            if self.parent.extract2dwindow is not None:
                border = self.parent.extract2dwindow.extr_border
                poly = self.parent.extract2dwindow.sky_poly
                model = self.parent.extract2dwindow.skymodeltype
                conf = self.parent.extract2dwindow.extr_conf
            else:
                border, poly, model, conf = 5, 3, 'median', 0.03
            s.sky_model(x, x, border=border, poly=poly, model=model, conf=conf, plot=1, smooth=0)

            if s.parent.sky2d in self.parent.spec2dPanel.vb.addedItems:
                self.parent.spec2dPanel.vb.removeItem(s.parent.sky2d)
            s.parent.sky2d = s.set_image('sky', s.parent.colormap)
            self.parent.spec2dPanel.vb.addItem(s.parent.sky2d)

        if self.x_status:
            self.mousePoint_saved = self.vb.mapSceneToView(event.position())
            #if (self.mousePoint_saved.x() == self.mousePoint.x()) and (self.mousePoint_saved.y() == self.mousePoint.y())
            if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier) and (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier):
                self.parent.s[self.parent.s.ind].spec2d.raw.add_mask(
                    rect=[[np.min([self.mousePoint_saved.x(), self.mousePoint_saved.x()]),
                           np.max([self.mousePoint_saved.x(), self.mousePoint_saved.x()])],
                          [np.min([self.mousePoint_saved.y(), self.mousePoint_saved.y()]),
                           np.max([self.mousePoint_saved.y(), self.mousePoint_saved.y()])]
                          ], add=(QApplication.keyboardModifiers() != Qt.KeyboardModifier.ControlModifier))
            if self.parent.s[self.parent.s.ind].cr_mask2d in self.parent.spec2dPanel.vb.addedItems:
                self.parent.spec2dPanel.vb.removeItem(self.parent.s[self.parent.s.ind].cr_mask2d)
            self.parent.s[self.parent.s.ind].cr_mask2d = self.parent.s[self.parent.s.ind].spec2d.set_image('cr', self.parent.s[self.parent.s.ind].cr_maskcolormap)
            self.parent.spec2dPanel.vb.addItem(self.parent.s[self.parent.s.ind].cr_mask2d)

    def mouseReleaseEvent(self, event):
        if any([getattr(self, s+'_status') for s in 'brtx']):
            self.vb.rbScaleBox.hide()
            #if self.vb.rbScaleBox in self.vb.addedItems:
            #    self.vb.removeItem(self.vb.rbScaleBox)
            self.vb.setMouseMode(self.vb.PanMode)

        if any([getattr(self, s + '_status') for s in 's']):
            self.mousePoint_saved = self.vb.mapSceneToView(event.position())

        if self.b_status:
            if event.button() == Qt.MouseButton.LeftButton:
                if self.mousePoint == self.mousePoint_saved:
                    self.parent.s[self.parent.s.ind].add_spline(self.mousePoint.x(), self.mousePoint.y(), name='2d')

                else:
                    self.parent.s[self.parent.s.ind].del_spline(self.mousePoint_saved.x(), self.mousePoint_saved.y(),
                                                                self.mousePoint.x(), self.mousePoint.y(), name='2d')

            if event.button() == Qt.MouseButton.RightButton:
                ind = self.parent.s[self.parent.s.ind].spline2d.find_nearest(self.mousePoint.x(), None)
                self.parent.s[self.parent.s.ind].del_spline(arg=ind, name='2d')
                event.accept()

        if self.t_status:
            self.mousePoint = self.vb.mapSceneToView(event.position())

            if self.t_status == 1:
                spec2d = self.parent.s[self.parent.s.ind].spec2d
                if len(spec2d.slits) > 0:
                    data = np.asarray([[s[0], s[1]] for s in spec2d.slits])
                    print(data)
                    ind = np.argmin(np.sum((data - np.array([self.mousePoint.x(), self.mousePoint.y()]))**2, axis=1))
                    spec2d.slits.remove(spec2d.slits[ind])
                self.parent.s[self.parent.s.ind].redraw()

            if self.t_status == 2:
                self.t_status = False
                self.parent.s[self.parent.s.ind].spec2d.profile(np.min([self.mousePoint_saved.x(), self.mousePoint.x()]),
                                                                np.max([self.mousePoint_saved.x(), self.mousePoint.x()]),
                                                                np.min([self.mousePoint_saved.y(), self.mousePoint.y()]),
                                                                np.max([self.mousePoint_saved.y(), self.mousePoint.y()]),
                                                                plot=True)

        if self.x_status:
            if (QApplication.keyboardModifiers() != Qt.KeyboardModifier.ShiftModifier) and (QApplication.keyboardModifiers() != Qt.KeyboardModifier.ControlModifier):
                self.parent.s[self.parent.s.ind].spec2d.cr.add_mask(
                    rect=[[np.min([self.mousePoint_saved.x(), self.mousePoint.x()]),
                           np.max([self.mousePoint_saved.x(), self.mousePoint.x()])],
                          [np.min([self.mousePoint_saved.y(), self.mousePoint.y()]),
                           np.max([self.mousePoint_saved.y(), self.mousePoint.y()])]
                          ], add=True)
                self.parent.s[self.parent.s.ind].spec2d.intelExpand(exp_factor=2, exp_pixel=1, pixel=(self.mousePoint_saved.x(), self.mousePoint_saved.y()))
            else:
                self.parent.s[self.parent.s.ind].spec2d.cr.add_mask(
                    rect=[[np.min([self.mousePoint_saved.x(), self.mousePoint.x()]),
                           np.max([self.mousePoint_saved.x(), self.mousePoint.x()])],
                          [np.min([self.mousePoint_saved.y(), self.mousePoint.y()]),
                           np.max([self.mousePoint_saved.y(), self.mousePoint.y()])]
                          ], add=(QApplication.keyboardModifiers() != Qt.KeyboardModifier.ControlModifier))
            #self.parent.s.redraw()

        if event.isAccepted():
            super(spec2dWidget, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        super(spec2dWidget, self).mouseMoveEvent(event)
        self.mousePoint = self.vb.mapSceneToView(event.position())
        self.mouse_moved = True
        s = 'x={0:.3f}, y={1:.2f}'.format(self.mousePoint.x(), self.mousePoint.y())
        if len(self.parent.s) > 0 and self.parent.s[self.parent.s.ind].spec2d.raw.z is not None and len(self.parent.s[self.parent.s.ind].spec2d.raw.z.shape) == 2:
            s += ', z={:.2e}'.format(self.parent.s[self.parent.s.ind].spec2d.raw.find_nearest(self.mousePoint.x(), self.mousePoint.y()))
            if self.parent.s[self.parent.s.ind].spec2d.raw.err is not None:
                s += ', err={:.2e}'.format(self.parent.s[self.parent.s.ind].spec2d.raw.find_nearest(self.mousePoint.x(), self.mousePoint.y(), attr='err'))
        self.cursorpos.setText(s)

        pos = self.vb.sceneBoundingRect()
        self.cursorpos.setPos(self.vb.mapSceneToView(QPointF(int(pos.left() + 10), int(pos.bottom() - 10))))
        if self.s_status == 2 and event.type() == QEvent.Type.MouseMove:
            range = self.vb.getState()['viewRange']
            delta = ((self.mousePoint.x() - self.mousePoint_saved.x()) / (range[0][1] - range[0][0]),
                     (self.mousePoint.y() - self.mousePoint_saved.y()) / (range[1][1] - range[1][0]))
            #print(delta)
            self.mousePoint_saved = self.mousePoint
            im = self.parent.s[self.parent.s.ind].image2d
            levels = im.getLevels()
            d = self.parent.s[self.parent.s.ind].spec2d.raw.z_quantile[1] - self.parent.s[self.parent.s.ind].spec2d.raw.z_quantile[0]
            self.parent.s[self.parent.s.ind].spec2d.raw.setLevels(levels[0] + d * delta[0] * 2, levels[1] + d*delta[1] / 4)
            im.setLevels(self.parent.s[self.parent.s.ind].spec2d.raw.z_levels)
            self.levels.setText("levels: {:.4f} {:.4f}".format(levels[0], levels[1]))

        self.levels.setPos(self.vb.mapSceneToView(QPointF(int(pos.right() - 10), int(pos.bottom() - 10))))
        if self.t_status:
            self.t_status = 2

        if self.s_status and event.type() == QEvent.Type.MouseMove:
            self.vb.rbScaleBox.hide()


    def wheelEvent(self, event):
        super(spec2dWidget, self).wheelEvent(event)
        pos = self.vb.sceneBoundingRect()
        self.cursorpos.setPos(self.vb.mapSceneToView(QPointF(int(pos.left() + 10), int(pos.bottom() - 10))))

    def mouseDragEvent(self, ev):

        if ev.button() == Qt.MouseButton.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev)

        ev.accept()
        pos = ev.pos()

        if ev.button() == Qt.MouseButton.RightButton:
            self.updateScaleBox(ev.buttonDownPos(), ev.pos())

            if ev.isFinish():
                self.rbScaleBox.hide()
                ax = QRectF(QPointF(ev.buttonDownPos(ev.button())), QPointF(pos))
                ax = self.childGroup.mapRectFromParent(ax)
                MouseRectCoords = ax.getCoords()
                self.dataSelection(MouseRectCoords)
            else:
                self.updateScaleBox(ev.buttonDownPos(), ev.pos())

class preferencesWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.move(200, 100)
        self.setWindowTitle('Preferences')
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())

        self.initGUI()
        self.setGeometry(300, 200, 100, 100)
        self.show()

    def initGUI(self):
        layout = QVBoxLayout()

        self.tab = QTabWidget()
        self.tab.setGeometry(0, 0, 1050, 900)
        #self.tab.setMinimumSize(1050, 300)
        
        for t in ['Appearance', 'Fit', 'Colors']:
            self.tab.addTab(self.initTabGUI(t), t)

        self.tab.currentChanged.connect(self.setTabIndex)
        self.tab.setCurrentIndex(self.parent.preferencesTabIndex)

        layout.addWidget(self.tab)
        h = QHBoxLayout()
        h.addStretch(1)
        ok = QPushButton("Ok")
        ok.setFixedSize(110, 30)
        ok.clicked.connect(self.close)
        h.addWidget(ok)
        layout.addStretch(1)
        layout.addLayout(h)
        self.setLayout(layout)


    def initTabGUI(self, window=None):

        frame = QFrame()
        validator = QDoubleValidator()
        locale = QLocale('C')
        validator.setLocale(locale)

        layout = QHBoxLayout()
        self.grid = QGridLayout()

        if window == 'Fit':

            ind = 0
            self.fitGroup = QButtonGroup(self)
            self.fittype = ['julia', 'regular', 'uniform']
            for i, f in enumerate(self.fittype):
                setattr(self, f, QRadioButton(f if f == 'julia' else 'py:' + f + ':' * (f == 'uniform')))
                if self.parent.fitType == f:
                    getattr(self, f).toggle()
                getattr(self, f).clicked.connect(self.setFitType)
                self.fitGroup.addButton(getattr(self, f))
                self.grid.addWidget(getattr(self, f), ind, i)

            self.num_between = QLineEdit(str(self.parent.num_between))
            self.num_between.setValidator(validator)
            self.num_between.textChanged[str].connect(self.setNumBetween)
            self.grid.addWidget(self.num_between, ind, 4)

            ind += 1
            self.grid.addWidget(QLabel('Tau limit:'), ind, 0)

            self.tau_limit = QLineEdit(str(self.parent.tau_limit))
            self.tau_limit.setValidator(validator)
            self.tau_limit.textChanged[str].connect(self.setTauLimit)
            self.grid.addWidget(self.tau_limit, ind, 1)

            ind += 1
            self.grid.addWidget(QLabel('Accuracy:'), ind, 0)

            self.accuracy = QLineEdit(str(self.parent.accuracy))
            self.accuracy.setValidator(validator)
            self.accuracy.textChanged[str].connect(self.setAccuracy)
            self.grid.addWidget(self.accuracy, ind, 1)

            ind += 1
            self.grid.addWidget(QLabel("Julia grid method:"), ind, 0)

            self.julia_grid = QComboBox()
            self.julia_grid.addItems(["minimized", "adaptive", "uniform"])
            #self.julia_grid.setCurrentIndex(int(self.parent.options("julia_grid")) + 2 if int(self.parent.options("julia_grid")) < 0 else 2)
            self.julia_grid.setCurrentText(self.parent.options("julia_grid"))
            self.julia_grid.currentIndexChanged.connect(self.setJuliaGrid)
            self.julia_grid.setFixedSize(120, 30)
            self.grid.addWidget(self.julia_grid, ind, 1)
            self.julia_grid_num = QLineEdit(self.parent.options("julia_grid_num"))
            #self.julia_grid_num.setEnabled(int(self.parent.options("julia_grid")) > -1)
            self.julia_grid_num.setFixedSize(80, 30)
            self.julia_grid_num.setValidator(validator)
            self.julia_grid_num.textChanged[str].connect(self.setJuliaGridNum)
            self.grid.addWidget(self.julia_grid_num, ind, 2)
            self.julia_binned = QCheckBox('binned')
            self.julia_binned.setChecked(self.parent.options("julia_binned"))
            self.julia_binned.stateChanged.connect(partial(self.setChecked, 'julia_binned'))
            self.julia_binned.setFixedSize(80, 30)
            self.grid.addWidget(self.julia_binned, ind, 3)

            ind += 1
            self.grid.addWidget(QLabel('Minimization method:'), ind, 0)
            self.fitmethod = QComboBox()
            self.fitmethod_py = ['leastsq', 'least_squares', 'differential_evolution', 'brute', 'basinhopping',
                                     'ampgo', 'nelder', 'lbfgsb', 'powell', 'cg', 'newton', 'cobyla', 'bfgs',
                                     'tnc', 'trust-ncg', 'trust-exact', 'trust-krylov', 'trust-constr', 'dogleg',
                                     'slsqp', 'shgo', 'dual_annealing']
            self.fitmethod_jl = ["LsqFit.lmfit", "lmfit"]
            self.fitmethod.addItems(self.fitmethod_jl if self.parent.fitType == 'julia' else self.fitmethod_py)
            self.fitmethod.setCurrentText(self.parent.fit_method)
            self.fitmethod.currentIndexChanged.connect(self.setMethod)
            self.fitmethod.setFixedSize(120, 30)
            self.grid.addWidget(self.fitmethod, ind, 1)

            self.grid.addWidget(QLabel("tolerance:"), ind, 2)
            self.fit_tol = QLineEdit(self.parent.options("fit_tolerance"))
            # self.julia_grid_num.setEnabled(int(self.parent.options("julia_grid")) > -1)
            self.fit_tol.setFixedSize(80, 30)
            self.fit_tol.setValidator(validator)
            self.fit_tol.textChanged[str].connect(self.setFitTolerance)
            self.grid.addWidget(self.fit_tol, ind, 3)

            ind += 1
            self.grid.addWidget(QLabel('Fit components:'), ind, 0)
            self.compGroup = QButtonGroup(self)
            self.compview = ['all', 'one', 'none']
            for i, f in enumerate(self.compview):
                setattr(self, f, QRadioButton(f))
                getattr(self, f).toggled.connect(self.setCompView)
                self.compGroup.addButton(getattr(self, f))
                self.grid.addWidget(getattr(self, f), ind, i+1)
            getattr(self, self.parent.comp_view).toggle()

            ind += 1
            self.grid.addWidget(QLabel('Fit view:'), ind, 0)
            self.viewGroup = QButtonGroup(self)
            self.fitView = ['line', 'points', 'bins']
            for i, f in enumerate(self.fitView):
                setattr(self, f, QRadioButton(f))
                getattr(self, f).toggled.connect(self.setFitView)
                self.viewGroup.addButton(getattr(self, f))
                self.grid.addWidget(getattr(self, f), ind, i + 1)
            getattr(self, self.parent.fitview).toggle()

            ind += 1
            self.telluric = QCheckBox('add telluric/accompanying asborption')
            self.telluric.setChecked(self.parent.options("telluric"))
            self.telluric.stateChanged.connect(partial(self.setChecked, 'telluric'))
            self.grid.addWidget(self.telluric, ind, 0)

            ind +=1
            self.animateFit = QCheckBox('animate fit')
            self.animateFit.setChecked(self.parent.animateFit)
            self.animateFit.stateChanged.connect(partial(self.setChecked, 'animateFit'))
            self.animateFit.setEnabled(False)
            self.grid.addWidget(self.animateFit, ind, 0)

        if window == 'Appearance':
            ind = 0
            self.grid.addWidget(QLabel('Working display:'), 0, 0)
            self.display = QComboBox()
            self.dispdict = ["Main", "Secondary", "Additional"]
            self.display.addItems([self.dispdict[i] for i in range(len(QApplication.screens()))])
            self.display.setCurrentText(self.dispdict[min([int(self.parent.options("display")), len(QApplication.screens())])])
            self.display.currentIndexChanged.connect(self.setDisplay)
            self.display.setFixedSize(120, 30)
            self.grid.addWidget(self.display, ind, 1)

            ind += 1
            self.grid.addWidget(QLabel('Spectrum view:'), ind, 0)
            self.specview = QComboBox()
            self.viewdict = OrderedDict([('step', 'step'), ('steperr', 'step + uncert.'), ('line', 'lines'),
                                         ('lineerr', 'lines + uncert.'), ('point', 'points'), ('pointerr', 'points + uncert.')])
            self.specview.addItems(list(self.viewdict.values()))
            self.specview.setCurrentText(self.viewdict[self.parent.specview])
            self.specview.currentIndexChanged.connect(self.setSpecview)
            self.specview.setFixedSize(120, 30)
            self.grid.addWidget(self.specview, ind, 1)

            ind += 1
            self.grid.addWidget(QLabel('Fitting pixels view:'), ind, 0)
            self.selectview = QComboBox()
            self.selectview.addItems(['points', 'color', 'regions'])
            self.selectview.setCurrentText(self.parent.selectview)
            self.selectview.currentIndexChanged.connect(self.setSelect)
            self.selectview.setFixedSize(120, 30)
            self.grid.addWidget(self.selectview, ind, 1)

            ind += 1
            self.grid.addWidget(QLabel('Line labels view:'), ind, 0)
            self.selectlines = QComboBox()
            self.selectlines.addItems(['short', 'infinite'])
            self.selectlines.setCurrentText(self.parent.linelabels)
            self.selectlines.currentIndexChanged.connect(self.setLabels)
            self.selectlines.setFixedSize(120, 30)
            self.grid.addWidget(self.selectlines, ind, 1)

            ind += 1
            self.showinactive = QCheckBox('show inactive exps')
            self.showinactive.setChecked(self.parent.showinactive)
            self.showinactive.stateChanged.connect(partial(self.setChecked, 'showinactive'))
            self.grid.addWidget(self.showinactive, ind, 0)

            ind += 1
            self.show_osc = QCheckBox('f in line labels')
            self.show_osc.setChecked(self.parent.show_osc)
            self.show_osc.stateChanged.connect(partial(self.setChecked, 'show_osc'))
            self.grid.addWidget(self.show_osc, ind, 0)

            ind += 1
            self.blindMode = QCheckBox('blind mode')
            self.blindMode.setChecked(self.parent.blindMode)
            self.blindMode.stateChanged.connect(partial(self.setChecked, 'blindMode'))
            self.grid.addWidget(self.blindMode, ind, 0)

        layout.addLayout(self.grid)
        layout.addStretch()
        frame.setLayout(layout)
        return frame

    def setDisplay(self):
        self.parent.options("display", self.display.currentIndex())
        self.parent.setScreen()

    def setSpecview(self):
        self.parent.specview = list(self.viewdict.keys())[list(self.viewdict.values()).index(self.specview.currentText())]
        self.parent.options('specview', self.parent.specview)
        if self.parent.s.ind is not None:
            self.parent.s[self.parent.s.ind].remove()
            self.parent.s[self.parent.s.ind].initGUI()

    def setSelect(self):
        if self.parent.s.ind is not None:
            self.parent.s[self.parent.s.ind].remove()
        self.parent.selectview = self.selectview.currentText()
        self.parent.options('selectview', self.parent.selectview)
        if self.parent.s.ind is not None:
            self.parent.s[self.parent.s.ind].initGUI()

    def setLabels(self):
        self.parent.linelabels = self.selectlines.currentText()
        self.parent.options('linelabels', self.parent.linelabels)
        self.parent.abs.changeStyle()

    def setFitType(self):
        for f in self.fittype:
            if getattr(self, f).isChecked():
                self.parent.fitType = f
                self.parent.options('fitType', self.parent.fitType)
                if self.parent.fitType == 'julia' and not self.parent.reload_julia():
                    self.parent.julia = None
                    self.parent.options('uniform', self.parent.fitType)
                    self.uniform.toggle()
        self.fitmethod.clear()
        self.fitmethod.addItems(self.fitmethod_jl if self.parent.fitType == 'julia' else self.fitmethod_py)
        self.parent.options("fit_method", self.fitmethod.itemText(0))

    def setTabIndex(self):
        self.parent.preferencesTabIndex = self.tab.currentIndex()

    def setCompView(self):
        for f in self.compview:
            if getattr(self, f).isChecked():
                self.parent.comp_view = f
                self.parent.options('comp_view', self.parent.comp_view)
                self.parent.s.redraw()
                return

    def setFitView(self):
        for f in self.fitView:
            if getattr(self, f).isChecked():
                self.parent.fitview = f
                self.parent.options('fitview', self.parent.fitview)
                self.parent.s.redraw()
                return

    def setChecked(self, attr):
        self.parent.options(attr, getattr(self, attr).isChecked())
        if attr == 'show_osc':
            self.parent.abs.redraw()
        if attr == 'blindMode' and self.parent.blindMode:
            self.parent.sendMessage("Switch blind mode on. The fit results will be hidden.")
        self.parent.s.redraw()

    def setNumBetween(self):
        self.parent.num_between = int(self.num_between.text())
        self.parent.options("num_between", self.parent.num_between)

    def setTauLimit(self):
        try:
            t = float(self.tau_limit.text())
            if t < 1 and t > 0:
                self.parent.tau_limit = t
                self.parent.options('tau_limit', self.parent.tau_limit)
        except:
            pass

    def setAccuracy(self):
        try:
            t = float(self.accuracy.text())
            if t > 0 and t < 0.5:
                self.parent.accuracy = t
                self.parent.options('accuracy', self.parent.accuracy)
        except:
            pass

    def setJuliaGrid(self):
        if 1:
            self.parent.options("julia_grid", self.julia_grid.currentText())
        else:
            if self.julia_grid.currentText() == 'uniform':
                self.julia_grid_num.setEnabled(True)
                jd = int(self.julia_grid_num.text())
            else:
                self.julia_grid_num.setEnabled(False)
                jd = self.julia_grid.currentIndex() - 2
            self.parent.options("julia_grid", jd)

    def setJuliaGridNum(self):
        #if self.julia_grid.currentText() == 'uniform':
        self.parent.options("julia_grid_num", int(self.julia_grid_num.text()))

    def setFitTolerance(self):
        # if self.julia_grid.currentText() == 'uniform':
        self.parent.options("fit_tolerance", float(self.fit_tol.text()))

    def setMethod(self):
        self.parent.fit_method = self.fitmethod.currentText()
        self.parent.options("fit_method", self.parent.fit_method)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11:
            self.close()

    def closeEvent(self, event):
        self.parent.preferences = None
        event.accept()

class choosePC(QToolButton):
    def __init__(self, parent):
        super(choosePC, self).__init__()
        self.parent = parent
        self.setFixedSize(90, 30)
        self.toolmenu = QMenu(self)
        self.cf = []
        self.update()
        self.setText('choose:')

        self.setMenu(self.toolmenu)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

    def update(self):
        try:
            self.toolmenu.removeAction(self.setall)
        except:
            pass
        try:
            for i in reversed(range(len(self.cf))):
                self.toolmenu.removeAction(self.cf[i])
                self.cf.remove(self.cf[i])
        except:
            pass
        if self.parent.parent.fit.cf_num > 0:
            self.setall = QAction("all", self.toolmenu)
            self.setall.setCheckable(True)
            self.setall.triggered.connect(partial(self.set, cf=None))
            self.toolmenu.addAction(self.setall)

            for i in range(self.parent.parent.fit.cf_num):
                self.cf.append(QAction("cf_" + str(i), self.toolmenu))
                self.cf[i].triggered.connect(partial(self.set, cf=i))
                self.toolmenu.addAction(self.cf[i])
                self.cf[i].setCheckable(True)

    def currentText(self):
        if hasattr(self, 'setall') and self.setall.isChecked():
            return 'all'
        else:
            return '_'.join([f'cf{i}' for i, s in enumerate(self.cf) if s.isChecked()])

    def fromtext(self, text):
        print(text)
        if 'all' in text or text.strip() == '':
            if hasattr(self, 'setall'):
                self.setall.setChecked(True)
                self.set(cf=None)
        else:
            for c in text.split('_'):
                if self.parent.parent.fit.cf_num > int(c.replace('cf', '')):
                    self.cf[int(c.replace('cf', ''))].setChecked(True)
                    self.set(int(c.replace('cf', '')))

    def set(self, cf=None):
        if cf == None:
            if self.setall.isChecked():
                for cf in self.cf:
                    cf.setChecked(False)
            else:
                self.setall.setChecked(True)
        else:
            if hasattr(self, 'setall'):
                self.setall.setChecked(False)
        self.parent.cfs = self.currentText()

class showLinesWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.resize(800, 1100)
        self.move(200, 100)
        #self.setWindowFlags(Qt.FramelessWindowHint)

        self.initData()
        self.initGUI()
        self.setWindowTitle('Plot line profiles using Matploplib')
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())

    def initData(self):
        self.savedText = None
        self.opts = OrderedDict([
                    ('width', float), ('height', float),
                    ('rows', int), ('cols', int), ('order', str),
                    ('v_indent', float), ('h_indent', float),
                    ('col_offset', float), ('row_offset', float),
                    ('units', str), ('regions', int),
                    ('xmin', float), ('xmax', float), ('ymin', float), ('ymax', float),
                    ('spec_lw', float), ('show_err', int), ('error_cap', float),
                    ('residuals', int), ('res_sigma', int), ('gray_out', int),
                    ('fit_color', int), ('fit_lw', float), ('fit_ls', str),
                    ('show_telluric', int), ('tell_color', int), ('telluric_fill', int), ('telluric_lw', float),
                    ('show_comps', int), ('comp_lw', float), ('comp_ls', str),
                    ('z_ref', float), ('sys_ind', int),
                    ('show_disp', int), ('disp_alpha', float), ('res_style', str),
                    ('res_color', int), ('comp_colors', str),
                    ('font', str), ('font_size', int),
                    ('labels_corr', int), ('xlabel', str), ('ylabel', str),
                    ('x_ticks', float), ('xnum', int), ('y_ticks', float), ('ynum', int),
                    ('title', str), ('show_title', int), ('font_title', int), ('title_x_pos', float), ('title_y_pos', float),
                    ('show_labels', int), ('font_labels', int), ('name_x_pos', float), ('name_y_pos', float),
                    ('indlines_ls', str), ('indlines_lw', float), ('add_lines', str), ('addlines_ls', str),
                    ('add_short', int), ('add_full', int),
                    ('show_H2', str), ('only_marks', int), ('all_comps_marks', int), ('pos_H2', float),
                    ('plotfile', str), ('show_cont', int), ('corr_cheb', int),
                    ('show_cf', int), ('cfs', str), ('show_cf_value', int),
                    ('cf_color', int),
                    ('over_file', str), ('over_color', int),
                    ])

        for opt, func in self.opts.items():
            setattr(self, opt, func(self.parent.options(opt)))

        self.y_formatter = None

    def initGUI(self):
        hlayout = QHBoxLayout()
        self.setLayout(hlayout)
        layout = QVBoxLayout()
        hlayout.addLayout(layout)
        hlayout.addStretch(1)
        l = QVBoxLayout()
        l1 = QHBoxLayout()
        self.numLines = QPushButton('Lines: '+str(len(self.parent.lines)))
        self.numLines.setCheckable(True)
        self.numLines.setFixedSize(110, 25)
        self.numLines.clicked.connect(partial(self.changeState, 'lines'))
        self.numRegions = QPushButton('Region: '+str(len(self.parent.plot.regions)))
        self.numRegions.setCheckable(True)
        self.numRegions.setFixedSize(110, 25)
        self.numRegions.clicked.connect(partial(self.changeState, 'regions'))
        l1.addWidget(self.numLines)
        l1.addStretch(1)
        l1.addWidget(self.numRegions)
        l.addLayout(l1)
        self.lines = QTextEdit()
        self.setLines(init=True)
        self.lines.setFixedSize(240, self.frameGeometry().height())
        self.lines.textChanged.connect(self.readLines)
        l.addWidget(self.lines)
        self.chooseLine = QComboBox()
        self.chooseLine.setFixedSize(130, 30)
        self.chooseLine.addItems(['choose...'] + [str(l.line) for l in self.parent.abs.lines])
        self.chooseLine.activated.connect(self.selectLine)
        l.addWidget(self.chooseLine)
        l.addStretch()
        hlayout.addLayout(l)

        grid = QGridLayout()
        layout.addLayout(grid)
        validator = QDoubleValidator()
        locale = QLocale('C')
        validator.setLocale(locale)
        #validator.ScientificNotation
        names = ['Size:', 'width:', '', 'height:', '',
                 'Panels:', 'cols:', '', 'rows:', '',
                 'Indents:', 'hor.:', '', 'vert.:', '',
                 'Order', '', '', '', '',
                 '0ffets between:', 'col:', '', 'row:', '',
                 'X-units:', '', '', '', '',
                 'X-scale:', 'min:', '', 'max:', '',
                 'Y-scale:', 'min:', '', 'max:', '',
                 'Spectrum:', '', '', 'error caps:', '',
                 'Telluric:', '', '', '', '',
                 'Fit:', '', '', 'linestyle:', '',
                 'Comps:', '', '', 'linestyle:', '',
                 'Reference z:', '', '', '', '',
                 'Residuals:', '', '', 'sig:', '',
                 'Disp:' , '', '', 'style:', '',
                 'Fonts:', '', '', 'size:', '',
                 'Labels:', 'x:', '', 'y:', '',
                 'X-ticks:', 'scale:', '', 'num', '',
                 'Y-ticks:', 'scale:', '', 'num', '',
                 'Title:', '', '', 'font:', '',
                 '', 'hor.:', '', 'vert.:', '',
                 'Line labels:', '', '', 'font', '',
                 '', 'hor.:', '', 'vert.:', '',
                 'Position lines:', 'widths:', '', 'linestyle:', '',
                 '', 'add. lines:', '', 'linestyle:', '',
                 'H2/CO labels:', '', '', '', '',
                 'Continuum', '', '', '', '',
                 'Covering factor:', '', '', '', '',]

        positions = [(i, j) for i in range(28) for j in range(5)]

        for position, name in zip(positions, names):
            if name == '':
                continue
            grid.addWidget(QLabel(name), *position)

        self.opt_but = OrderedDict([('width', [0, 2]), ('height', [0, 4]), ('cols', [1, 2]), ('rows', [1, 4]),
                                    ('v_indent', [2, 2]), ('h_indent', [2, 4]), ('col_offset', [4, 2]), ('row_offset', [4, 4]),
                                    ('xmin', [6, 2]), ('xmax', [6, 4]), ('ymin', [7, 2]), ('ymax', [7, 4]),
                                    ('spec_lw', [8, 1]), ('error_cap', [8, 4]),
                                    ('telluric_lw', [9, 4]),
                                    ('fit_lw', [10, 1]),
                                    ('comp_lw', [11, 2]),
                                    ('z_ref', [12, 1]),
                                    ('res_sigma', [13, 4]),
                                    ('disp_alpha', [14, 2]),
                                    ('font_size', [15, 4]),
                                    ('xlabel', [16, 2]), ('ylabel', [16, 4]),
                                    ('x_ticks', [17, 2]), ('xnum', [17, 4]),
                                    ('y_ticks', [18, 2]), ('ynum', [18, 4]),
                                    ('title', [19, 2]), ('font_title', [19, 4]),
                                    ('title_x_pos', [20, 2]), ('title_y_pos', [20, 4]),
                                    ('font_labels', [21, 4]),
                                    ('name_x_pos', [22, 2]), ('name_y_pos', [22, 4]),
                                    ('indlines_lw', [23, 2]),
                                    ('add_lines', [24, 2]),
                                    ('show_H2', [25, 1]), ('pos_H2', [25, 4])])
        self.buttons = {}
        for opt, v in self.opt_but.items():
            self.buttons[opt] = QLineEdit(str(getattr(self, opt)))
            self.buttons[opt].setFixedSize(80, 30)
            if opt not in ['xlabel', 'ylabel', 'show_H2', 'title', 'add_lines']:
                self.buttons[opt].setValidator(validator)
            self.buttons[opt].textChanged[str].connect(partial(self.onChanged, attr=opt))
            grid.addWidget(self.buttons[opt], v[0], v[1])

        self.buttons['z_ref'].setMaxLength(10)

        self.orderh = QCheckBox("hor.", self)
        self.orderh.clicked.connect(partial(self.setOrder, 'h'))
        self.orderv = QCheckBox("vert.", self)
        self.orderv.clicked.connect(partial(self.setOrder, 'v'))
        self.setOrder(self.order)
        grid.addWidget(self.orderh, 3, 2)
        grid.addWidget(self.orderv, 3, 4)

        self.unitsv = QCheckBox("vel.", self)
        self.unitsv.clicked.connect(partial(self.setUnits, 'v'))
        self.unitsl = QCheckBox("lambda", self)
        self.unitsl.clicked.connect(partial(self.setUnits, 'l'))
        self.setUnits(self.units)
        grid.addWidget(self.unitsv, 5, 2)
        grid.addWidget(self.unitsl, 5, 4)

        self.showerr = QCheckBox('show err')
        self.showerr.setChecked(self.show_err)
        self.showerr.clicked[bool].connect(self.setErr)
        grid.addWidget(self.showerr, 8, 2)

        self.telluric = QCheckBox('show')
        self.telluric.setChecked(self.show_telluric)
        self.telluric.clicked[bool].connect(self.setTelluric)
        grid.addWidget(self.telluric, 9, 1)

        self.tellcolor = pg.ColorButton()
        # self.fitcolor = QColorDialog()
        self.tellcolor.setFixedSize(30, 30)
        self.tellcolor.setColor(color=self.tell_color.to_bytes(4, byteorder='big'))
        self.tellcolor.sigColorChanged.connect(partial(self.setColor, comp="tell"))
        self.tellcolor.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        grid.addWidget(self.tellcolor, 9, 2)

        self.telluricFill = QCheckBox('fill')
        self.telluricFill.setChecked(self.telluric_fill)
        self.telluricFill.clicked[bool].connect(self.setFillTelluric)
        grid.addWidget(self.telluricFill, 9, 3)

        self.fitcolor = pg.ColorButton()
        #self.fitcolor = QColorDialog()
        self.fitcolor.setFixedSize(30, 30)
        self.fitcolor.setColor(color=self.fit_color.to_bytes(4, byteorder='big'))
        self.fitcolor.sigColorChanged.connect(partial(self.setColor, comp=-1))
        self.fitcolor.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        grid.addWidget(self.fitcolor, 10, 2)

        self.lsfit = QComboBox(self)
        self.lsfit.addItems(['solid', 'dashed', 'dotted', 'dashdot'])
        self.lsfit.setCurrentText(self.fit_ls)
        self.lsfit.currentIndexChanged.connect(self.onFitLsChoose)
        grid.addWidget(self.lsfit, 10, 4)

        self.plotcomps = QCheckBox('show')
        self.plotcomps.setChecked(self.show_comps)
        self.plotcomps.clicked[bool].connect(self.setPlotComps)
        grid.addWidget(self.plotcomps, 11, 1)

        #self.colorcomps = colorComboBox(self, len(self.parent.fit.sys))
        #grid.addWidget(self.colorcomps, 11, 2)

        self.lscomp = QComboBox(self)
        self.lscomp.addItems(['solid', 'dashed', 'dotted', 'dashdot'])
        self.lscomp.setCurrentText(self.comp_ls)
        self.lscomp.currentIndexChanged.connect(self.onCompLsChoose)
        grid.addWidget(self.lscomp, 11, 4)

        self.refcomp = QComboBox(self)
        self.refcomp.addItems([str(i) for i in range(len(self.parent.fit.sys))])
        self.sys_ind = min(self.sys_ind, len(self.parent.fit.sys)-1)
        self.refcomp.setCurrentIndex(self.sys_ind)
        self.refcomp.activated.connect(self.onIndChoose)
        grid.addWidget(self.refcomp, 12, 2)

        self.gray = QCheckBox('gray')
        self.gray.setChecked(self.gray_out)
        self.gray.clicked[bool].connect(self.setGray)
        grid.addWidget(self.gray, 12, 4)

        self.resid = QCheckBox('')
        self.resid.setChecked(self.residuals)
        self.resid.clicked[bool].connect(self.setResidual)
        grid.addWidget(self.resid, 13, 1)

        self.rescolor = pg.ColorButton()
        # self.fitcolor = QColorDialog()
        self.rescolor.setFixedSize(30, 30)
        self.rescolor.setColor(color=self.res_color.to_bytes(4, byteorder='big'))
        self.rescolor.sigColorChanged.connect(partial(self.setColor, comp='res'))
        self.rescolor.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        grid.addWidget(self.rescolor, 13, 2)

        self.showdisp = QCheckBox('show disp')
        self.showdisp.setChecked(self.show_disp)
        self.showdisp.clicked[bool].connect(self.setDisp)
        grid.addWidget(self.showdisp, 14, 1)

        self.resstyle = QComboBox(self)
        self.resstyle.addItems(['scatter', 'step'])
        self.resstyle.setCurrentText(self.res_style)
        self.resstyle.currentIndexChanged.connect(self.onResStyleChoose)
        grid.addWidget(self.resstyle, 14, 4)

        self.fontname = QComboBox(self) #QFontComboBox(self)
        self.fontname.addItems([f.name for f in matplotlib.font_manager.fontManager.ttflist])
        self.fontname.setCurrentText(self.font)
        self.fontname.currentIndexChanged.connect(self.onFontChoose)
        grid.addWidget(self.fontname, 15, 1)

        self.showtitle = QCheckBox('show')
        self.showtitle.setChecked(self.show_title)
        self.showtitle.clicked[bool].connect(self.setTitle)
        grid.addWidget(self.showtitle, 19, 1)

        self.showlabels = QCheckBox('show')
        self.showlabels.setChecked(self.show_labels)
        self.showlabels.clicked[bool].connect(self.setLabels)
        grid.addWidget(self.showlabels, 21, 1)

        self.labelscorr = QCheckBox('j1 -> *')
        self.labelscorr.setChecked(self.labels_corr)
        self.labelscorr.clicked[bool].connect(self.setLabelsCorr)
        grid.addWidget(self.labelscorr, 21, 2)

        self.lsindlines = QComboBox(self)
        self.lsindlines.addItems(['solid', 'dashed', 'dotted', 'dashdot'])
        self.lsindlines.setCurrentText(self.indlines_ls)
        self.lsindlines.currentIndexChanged.connect(self.onIndLinesLsChoose)
        grid.addWidget(self.lsindlines, 23, 4)

        self.lsaddlines = QComboBox(self)
        self.lsaddlines.addItems(['solid', 'dashed', 'dotted', 'dashdot'])
        self.lsaddlines.setCurrentText(self.addlines_ls)
        self.lsaddlines.currentIndexChanged.connect(self.onAddLinesLsChoose)
        grid.addWidget(self.lsaddlines, 24, 4)

        self.onlyLineMarks = QCheckBox('only marks')
        self.onlyLineMarks.setChecked(self.only_marks)
        self.onlyLineMarks.clicked[bool].connect(self.onlyMarks)
        grid.addWidget(self.onlyLineMarks, 25, 2)

        self.allCompsMarks = QCheckBox('all comps')
        self.allCompsMarks.setChecked(self.all_comps_marks)
        self.allCompsMarks.clicked[bool].connect(self.allComps)
        grid.addWidget(self.allCompsMarks, 26, 3)

        self.showcont = QCheckBox('show')
        self.showcont.setChecked(self.show_cont)
        self.showcont.clicked[bool].connect(self.setCont)
        grid.addWidget(self.showcont, 26, 1)

        self.corrcheb = QCheckBox('cheb. applied')
        self.corrcheb.setChecked(self.corr_cheb)
        self.corrcheb.clicked[bool].connect(self.setCheb)
        grid.addWidget(self.corrcheb, 26, 2)

        self.showcf = QCheckBox('show')
        self.showcf.setChecked(self.show_cf)
        self.showcf.clicked[bool].connect(self.setCf)
        grid.addWidget(self.showcf, 27, 1)

        self.cf = choosePC(self)
        self.cf.fromtext(self.cfs)
        self.cf.triggered.connect(self.setcfs)
        grid.addWidget(self.cf, 27, 2)

        self.showcfvalue = QCheckBox('value')
        self.showcfvalue.setChecked(self.show_cf_value)
        self.showcfvalue.clicked[bool].connect(self.setCfValue)
        grid.addWidget(self.showcfvalue, 27, 3)

        self.cfcolor = pg.ColorButton()
        # self.fitcolor = QColorDialog()
        self.cfcolor.setFixedSize(30, 30)
        self.cfcolor.setColor(color=self.cf_color.to_bytes(4, byteorder='big'))
        self.cfcolor.sigColorChanged.connect(partial(self.setColor, comp='cf'))
        self.cfcolor.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        grid.addWidget(self.cfcolor, 27, 4)

        self.colorComps = colorCompBox(self, num=len(self.parent.fit.sys))
        layout.addLayout(self.colorComps)

        layout.addStretch(1)
        l = QHBoxLayout()
        self.overButton = QPushButton("Overplot from:")
        self.overButton.setFixedSize(120, 30)
        self.overButton.clicked.connect(self.chooseOverFile)
        self.overFile = QLineEdit(self.over_file)
        self.overFile.setFixedSize(450, 30)
        self.overFile.textChanged[str].connect(self.setOverFilename)
        self.overColor = pg.ColorButton()
        # self.fitcolor = QColorDialog()
        self.overColor.setFixedSize(30, 30)
        self.overColor.setColor(color=self.over_color.to_bytes(4, byteorder='big'))
        self.overColor.sigColorChanged.connect(partial(self.setColor, comp='over'))
        self.overColor.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())

        l.addWidget(self.overButton)
        l.addWidget(self.overFile)
        l.addWidget(self.overColor)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout()
        self.showButton = QPushButton("Show")
        self.showButton.setFixedSize(110, 30)
        self.showButton.clicked.connect(partial(self.showPlot, False))
        expButton = QPushButton("Export to :")
        expButton.setFixedSize(110, 30)
        expButton.clicked.connect(partial(self.showPlot, True))

        l.addWidget(self.showButton)
        l.addWidget(expButton)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout()
        self.file = QLineEdit(self.plotfile)
        self.file.setFixedSize(450, 30)
        self.file.textChanged[str].connect(self.setFilename)
        self.chooseFileButton = QPushButton("Choose")
        self.chooseFileButton.setFixedSize(80, 30)
        self.chooseFileButton.clicked.connect(self.chooseFile)

        l.addWidget(self.file)
        l.addWidget(self.chooseFileButton)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout()
        save = QPushButton("Save Settings")
        save.setFixedSize(150, 30)
        save.clicked.connect(self.saveSettings)
        load = QPushButton("Load Settings")
        load.setFixedSize(150, 30)
        load.clicked.connect(partial(self.loadSettings, None))
        l.addWidget(save)
        l.addWidget(load)
        l.addStretch(1)
        layout.addLayout(l)

        self.changeState()

    def onChanged(self, text, attr=None):
        if attr is not None:
            setattr(self, attr, self.opts[attr](text))

    def setLines(self, init=False):
        if self.regions:
            self.lines.setText(str(self.parent.plot.regions))
        else:
            self.lines.setText(str(self.parent.lines))
        self.readLines()

    def changeState(self, s=None):
        if s == 'lines' and self.regions or s == 'regions' and not self.regions:
            if s == 'lines':
                self.regions = False
            if s == 'regions':
                self.regions = True
            text = self.lines.toPlainText()
            if self.savedText is None:
                self.setLines(init=True)
            else:
                self.lines.setText(self.savedText)
            self.savedText = text
            self.readLines()
        self.chooseLine.clear()
        if self.regions:
            self.chooseLine.addItems(['choose...'] + [str(p) for p in self.parent.plot.regions])
        else:
            self.chooseLine.addItems(['choose...'] + [str(l.line) for l in self.parent.abs.lines])
        self.numLines.setChecked(not self.regions)
        self.numRegions.setChecked(self.regions)

    def setOrder(self, s):
        for o in ['v', 'h']:
            getattr(self, 'order'+o).setChecked(s == o)
        self.order = s

    def setUnits(self, s):
        for u in ['v', 'l']:
            getattr(self, 'units'+u).setChecked(s == u)
        self.units = s

    def setColor(self, color, comp=-1):
        if comp == -1:
            self.fit_color = int.from_bytes(color.color(mode="byte"), byteorder='big')
        if comp == 'cf':
            self.cf_color = int.from_bytes(color.color(mode="byte"), byteorder='big')
        if comp == 'res':
            self.res_color = int.from_bytes(color.color(mode="byte"), byteorder='big')
        if comp == 'tell':
            self.tell_color = int.from_bytes(color.color(mode="byte"), byteorder='big')
        if comp == 'over':
            self.over_color = int.from_bytes(color.color(mode="byte"), byteorder='big')

    def setErr(self):
        self.show_err = int(self.showerr.isChecked())

    def setDisp(self):
        self.show_disp = int(self.showdisp.isChecked())

    def setTitle(self):
        self.show_title = int(self.showtitle.isChecked())

    def setLabels(self):
        self.show_labels = int(self.showlabels.isChecked())

    def setResidual(self):
        self.residuals = int(self.resid.isChecked())

    def setGray(self):
        self.gray_out = int(self.gray.isChecked())

    def setTelluric(self):
        self.show_telluric = int(self.telluric.isChecked())

    def setFillTelluric(self):
        self.telluric_fill = int(self.telluricFill.isChecked())

    def setPlotComps(self):
        self.show_comps = int(self.plotcomps.isChecked())

    def setLabelsCorr(self):
        self.labels_corr = int(self.labelscorr.isChecked())

    def setCont(self):
        self.show_cont = int(self.showcont.isChecked())

    def setCheb(self):
        self.corr_cheb = int(self.corrcheb.isChecked())

    def onlyMarks(self):
        self.only_marks = int(self.onlyLineMarks.isChecked())

    def allComps(self):
        self.all_comps_marks = int(self.allCompsMarks.isChecked())

    def setCf(self):
        self.show_cf = int(self.showcf.isChecked())

    def setcfs(self):
        self.cfs = self.cf.currentText()

    def setCfValue(self):
        self.show_cf_value = int(self.showcfvalue.isChecked())

    def setFilename(self):
        self.plotfile = self.file.text()

    def setOverFilename(self):
        self.over_file = self.overFile.text()

    def readLines(self):
        if self.regions:
            self.parent.plot.regions.fromText(self.lines.toPlainText(), sort=False)
            self.numRegions.setText('Regions: ' + str(len(self.parent.plot.regions)))
        else:
            self.parent.lines.fromText(self.lines.toPlainText())
            self.numLines.setText('Lines: '+str(len(self.parent.lines)))

    def selectLine(self, line):
        if self.regions:
            if line not in self.parent.regions:
                self.parent.regions.append(line)
                self.lines.setText(self.lines.toPlainText() + '\n' + line)
        else:
            if line not in self.parent.lines:
                self.parent.lines.append(line)
                self.lines.setText(self.lines.toPlainText() + '\n' + line)
        self.chooseLine.setCurrentIndex(0)

    def onIndChoose(self):
        self.sys_ind = self.refcomp.currentIndex()
        self.buttons['z_ref'].setText(str(self.parent.fit.sys[self.sys_ind].z.val))

    def onResStyleChoose(self):
        self.res_style = self.resstyle.currentText()

    def onFontChoose(self):
        self.font = self.fontname.currentText()

    def onFitLsChoose(self):
        self.fit_ls = self.lsfit.currentText()

    def onCompLsChoose(self):
        self.comp_ls = self.lscomp.currentText()

    def onIndLinesLsChoose(self):
        self.indlines_ls = self.lsindlines.currentText()

    def onAddLinesLsChoose(self):
        self.addlines_ls = self.lsaddlines.currentText()

    def chooseFile(self):
        fname = QFileDialog.getExistingDirectory(self, 'Select export folder...', self.parent.plot_set_folder)
        self.file.setText(fname)

    def chooseOverFile(self):
        fname = QFileDialog.getSaveFileName(self, 'Select overplot file...', self.parent.plot_set_folder)
        self.over_file.setText(fname)

    def showPlot(self, savefig=True):
        fig = plt.figure(figsize=(self.width, self.height), dpi=300)
        #self.subplot = self.mw.getFigure().add_subplot(self.rows, self.cols, 1)

        matplotlib.rcParams["axes.formatter.useoffset"] = False
        matplotlib.rcParams["font.family"] = 'sans-serif'
        matplotlib.rcParams["font.sans-serif"] = self.font
        
        try:
            over = np.genfromtxt(self.over_file, unpack=True)
        except:
            over = None
        if not self.regions:
            if not self.parent.normview:
                self.parent.normalize()

            if len(self.parent.lines) > int(self.rows) * int(self.cols):
                self.parent.sendMessage("The number of lines is larger than number of panels")
                return
            self.ps = plot_spec(len(self.parent.lines), font=self.font, font_size=self.font_size, font_labels=self.font_labels,
                                vel_scale=(self.units == 'v'), gray_out=self.gray_out, show_telluric=self.show_telluric,
                                show_err=self.show_err, error_cap=self.error_cap, figure=fig)
            rects = rect_param(n_rows=int(self.rows), n_cols=int(self.cols), order=self.order,
                               v_indent=self.v_indent, h_indent=self.h_indent,
                               col_offset=self.col_offset, row_offset=self.row_offset)
            self.ps.specify_rects(rects)
            self.ps.set_ticklabels(xlabel=self.xlabel,  ylabel=self.ylabel)
            self.ps.set_limits(x_min=self.xmin, x_max=self.xmax, y_min=self.ymin, y_max=self.ymax)
            self.ps.set_ticks(x_tick=self.x_ticks, x_num=self.xnum, y_tick=self.y_ticks, y_num=self.ynum)
            self.ps.specify_comps(*(sys.z.val for sys in self.parent.fit.sys))
            self.ps.specify_styles(lw=self.comp_lw, lw_total=self.fit_lw, lw_spec=self.spec_lw, lw_tell=self.telluric_lw,
                                   ls=self.comp_ls, ls_total=self.fit_ls, tell_fill=self.telluric_fill,
                                   ind_ls=self.indlines_ls, ind_lw=self.indlines_lw,
                                   add_lines=self.add_lines, add_ls=self.addlines_ls,
                                   color_total=self.fit_color.to_bytes(4, byteorder='big'),
                                   color=[tuple(int(c).to_bytes(4, byteorder='big')) for c in self.comp_colors.split(', ')],
                                   color_tell=self.tell_color.to_bytes(4, byteorder='big'),
                                   disp_alpha=self.disp_alpha, res_style=self.res_style, res_color=self.res_color.to_bytes(4, byteorder='big')
                                   )
            if len(self.parent.fit.sys) > 0:
                if self.buttons['z_ref'].text().strip() != '':
                    self.ps.z_ref = float(self.buttons['z_ref'].text())
                else:
                    self.ps.z_ref = self.parent.fit.sys[self.sys_ind].z.val
                    self.buttons['z_ref'].setText(str(self.ps.z_ref))
            else:
                self.ps.z_ref = self.parent.z_abs

            for i, p in enumerate(self.ps):
                p.name = ' '.join(self.parent.lines[self.ps.index(p)].split()[:2])
                ind = self.parent.s.ind
                print(self.parent.lines[self.ps.index(p)].split())
                if len(self.parent.lines[self.ps.index(p)].split()) > 2:
                    for s in self.parent.lines[self.ps.index(p)].split()[2:]:
                        print(s)
                        if 'exp' in s:
                            ind = int(s[4:])
                        if 'note' in s:
                            p.label = str(s[5:])
                        if 'nofit' in s:
                            p.show_fit = False
                print(p.name, ind)
                s = self.parent.s[ind]

                if self.corr_cheb and self.parent.fit.cont_fit:
                    cheb = interp1d(s.spec.raw.x[s.cont_mask], s.correctContinuum(s.spec.raw.x[s.cont_mask]), fill_value='extrapolate')
                else:
                    cheb = interp1d(s.spec.raw.x[s.cont_mask], np.ones_like(s.spec.raw.x[s.cont_mask]), fill_value=1)

                mask = (s.sky.raw.x > s.spec.raw.x[s.cont_mask][0]) * (s.sky.raw.x < s.spec.raw.x[s.cont_mask][-1])
                sky = [s.sky.raw.x[mask], s.sky.raw.y[mask] / cheb(s.sky.raw.x[mask])]

                if s.fit.n() > 0:
                    fit = np.array([s.fit.x(), s.fit.y() / cheb(s.fit.x())])
                    if self.show_comps:
                        fit_comp = []
                        for c in s.fit_comp:
                            fit_comp.append(np.array([c.x(), c.y() / cheb(c.x())]))
                    else:
                        fit_comp = None
                else:
                    fit = None
                    fit_comp = None

                if self.show_disp and len(s.fit.disp[0].norm.x) > 0:
                    fit_disp = [s.fit.disp[0].norm.x, s.fit.disp[0].norm.y / cheb(s.fit.disp[0].norm.x), s.fit.disp[1].norm.y / cheb(s.fit.disp[0].norm.x)]
                    fit_comp_disp = []
                    for comp in s.fit_comp:
                        fit_comp_disp.append([comp.disp[0].norm.x, comp.disp[0].norm.y / cheb(s.fit.disp[0].norm.x), comp.disp[1].norm.y / cheb(s.fit.disp[0].norm.x)])
                else:
                    fit_disp, fit_comp_disp = None, None

                p.loaddata(d=np.array([s.spec.x(), s.spec.y() / cheb(s.spec.x()), s.spec.err() / cheb(s.spec.x()), s.mask.x()]),
                           f=fit, fit_comp=fit_comp, fit_disp=fit_disp, fit_comp_disp=fit_comp_disp, sky=sky, z=[sys.z.val for sys in self.parent.fit.sys])
                if len(self.parent.lines[self.ps.index(p)].split()) > 3:
                    for s in self.parent.lines[self.ps.index(p)].split()[2:]:
                        if 'ymin' in s:
                            p.y_min = float(s[5:])
                        if 'ymax' in s:
                            p.y_max = float(s[5:])
                for l in self.parent.abs.lines:
                    if p.name == str(l.line):
                        p.wavelength = l.line.l()
                print(p.wavelength)
                p.show_comps = self.show_comps
                if self.show_labels:
                    p.name_pos = [self.name_x_pos, self.name_y_pos]
                else:
                    p.name_pos = None

                if any([s in p.name for s in ['H2', 'HD', 'CO']]):
                    p.name = ' '.join([p.name.split()[0][:-2], p.name.split()[1]])
                if self.labels_corr and all([not s in p.name for s in ['H2', 'HD', 'CO']]):
                    if 'j' in p.name:
                        m = re.findall(r'(j\d+)', p.name)[0]
                        if int(m[1:]) < 3:
                            p.name = p.name.replace(m, '*'*int(m[1:]))
                        else:
                            p.name = p.name.replace(m, r"$^{" + str(m[1:]) + "}$*")

                p.add_residual, p.sig = self.residuals, self.res_sigma
                p.y_formatter = self.y_formatter

                p.plot_line()

                def conv(x):
                    return (x / p.wavelength / (1 + self.ps.z_ref) - 1) * 299794.26

                if over is not None:
                    if p.vel_scale:
                        x = conv(over[0])
                    else:
                        x = over[0]
                    p.ax.plot(x, over[1], ls='-', color=to_hex(tuple(c / 255 for c in self.over_color.to_bytes(4, byteorder='big'))), lw=self.fit_lw, zorder=12)


                if self.show_cont:

                    if not self.show_disp or len(s.fit.disp[0].norm.x) == 0:
                        if p.vel_scale:
                            x = conv(s.cheb.x())
                        else:
                            x = s.cheb.x()
                        #print(x, s.cheb.y(), self.corr_cheb)
                        p.ax.plot(x, np.power(s.cheb.y(), 1 - self.corr_cheb), '--k', lw=1)
                    else:
                        if p.vel_scale:
                            x = conv(s.cheb.disp[0].norm.x)
                        else:
                            x = s.cheb.disp[0].norm.x
                        p.ax.fill_between(x, s.cheb.disp[0].norm.y / cheb(s.cheb.disp[0].norm.x), s.cheb.disp[1].norm.y / cheb(s.cheb.disp[0].norm.x), fc='k', alpha=p.parent.disp_alpha / 2, zorder=11)
                        p.ax.plot(x, s.cheb.disp[0].norm.y / cheb(s.cheb.disp[0].norm.x), '--k', lw=1, zorder=10)
                        p.ax.plot(x, s.cheb.disp[1].norm.y / cheb(s.cheb.disp[0].norm.x), '--k', lw=1, zorder=10)
                    if 0:
                        self.showContCorr(ax=ax)

                if self.buttons['add_lines'].text().strip() != '':
                    print(self.buttons['add_lines'].text().strip())

                if self.show_cf and self.parent.fit.cf_fit and p.show_fit:
                    for k in range(self.parent.fit.cf_num):
                        if self.cfs == 'all' or 'cf' + str(k) in self.cfs:
                            attr = 'cf_' + str(k)
                            if hasattr(self.parent.fit, attr):
                                cf = getattr(self.parent.fit, attr)
                                if (len(cf.addinfo.split('_')) > 1 and cf.addinfo.split('_')[1] == 'all') or (cf.addinfo.find('exp') > -1 and int(cf.addinfo[cf.addinfo.find('exp')+3:]) == ind):
                                    color = to_hex(tuple(c / 255 for c in self.cf_color.to_bytes(4, byteorder='big')))
                                    if p.fit_disp is None:
                                        print([np.max([conv(cf.left), p.x_min]), np.min([conv(cf.right), p.x_max])], [1 - cf.val, 1 - cf.val])
                                        p.ax.plot([np.max([conv(cf.left), p.x_min]), np.min([conv(cf.right), p.x_max])], [1 - cf.val, 1 - cf.val], '--', lw=1.0, color=color)
                                    else:
                                        p.ax.plot([np.max([conv(cf.left), p.x_min]), np.min([conv(cf.right), p.x_max])], [1 - cf.unc.val, 1 - cf.unc.val], '--', lw=0.5, color=color)
                                        p.ax.fill_between([np.max([conv(cf.left), p.x_min]), np.min([conv(cf.right), p.x_max])], 1 - cf.unc.val - cf.unc.plus, 1 - cf.unc.val + cf.unc.minus, ls=':', color=color, alpha=0.1)
                                    if self.show_cf_value and (cf.left < p.x_max) and (cf.right > p.x_min):
                                        p.ax.text(p.x_max - (p.x_max - p.x_min) / 30, 1 - cf.unc.val, cf.fitres(latex=True), ha='right', va='bottom', fontsize=p.font_labels, fontname=p.font, color=color)

                if i == 0 and self.show_title:
                    print('Title:', self.title)
                    p.ax.text(self.title_x_pos, self.title_y_pos, str(self.title).strip(), ha='left', va='top', fontsize=self.font_title, fontname=self.font, transform=p.ax.transAxes)

        else:
            if len(self.parent.plot.regions) > int(self.rows) * int(self.cols):
                self.parent.sendMessage("The number of regions is larger than number of panels")
                return

            self.ps = plot_spec(len(self.parent.plot.regions), font=self.font, font_size=self.font_size, font_labels=self.font_labels,
                                vel_scale=False, gray_out=self.gray_out, show_telluric=self.show_telluric,
                                show_err=self.show_err, error_cap=self.error_cap, figure=fig)
            rects = rect_param(n_rows=int(self.rows), n_cols=int(self.cols), order=self.order,
                               v_indent=self.v_indent, h_indent=self.h_indent,
                               col_offset=self.col_offset, row_offset=self.row_offset)
            self.ps.specify_rects(rects)
            self.ps.set_ticklabels(xlabel=self.xlabel,  ylabel=self.ylabel)
            self.ps.set_limits(x_min=self.xmin, x_max=self.xmax, y_min=self.ymin, y_max=self.ymax)
            self.ps.set_ticks(x_tick=self.x_ticks, x_num=self.xnum, y_tick=self.y_ticks, y_num=self.ynum)
            self.ps.specify_comps(*(sys.z.val for sys in self.parent.fit.sys))
            self.ps.specify_styles(lw=self.comp_lw, lw_total=self.fit_lw, lw_spec=self.spec_lw,
                                   lw_tell=self.telluric_lw, tell_fill=self.telluric_fill,
                                   ls=self.comp_ls, ls_total=self.fit_ls, color_total=self.fit_color.to_bytes(4, byteorder='big'),
                                   color=[tuple(int(c).to_bytes(4, byteorder='big')) for c in self.comp_colors.split(', ')],
                                   color_tell=self.tell_color.to_bytes(4, byteorder='big'),
                                   disp_alpha=self.disp_alpha, res_style=self.res_style, res_color=self.res_color.to_bytes(4, byteorder='big')
                                   )
            if len(self.parent.fit.sys) > 0:
                if self.buttons['z_ref'].text().strip() != '':
                    self.ps.z_ref = float(self.buttons['z_ref'].text())
                else:
                    self.ps.z_ref = self.parent.fit.sys[self.sys_ind].z.val
                    self.buttons['z_ref'].setText(str(self.ps.z_ref))
            else:
                self.ps.z_ref = self.parent.z_abs

            for i, p in enumerate(self.ps):
                regions = self.lines.toPlainText().splitlines()
                #regions = self.parent.plot.regions
                st = str(regions[i]).split()
                p.x_min, p.x_max = (float(s) for s in st[0].split('..'))
                #p.y_formater = '%.1f'
                ind = self.parent.s.ind
                for s in st[1:]:
                    if 'name' in s:
                        p.name = s[5:]
                    if 'exp' in s:
                        ind = int(s[4:])
                    if 'note' in s:
                        p.label = str(s[5:])
                    if 'nofit' in s:
                        p.show_fit = False
                s = self.parent.s[ind]

                if self.corr_cheb and self.parent.fit.cont_fit:
                    if not self.show_disp:
                        cheb = interp1d(s.spec.raw.x[s.cont_mask], s.correctContinuum(s.spec.raw.x[s.cont_mask]), fill_value='extrapolate')
                    else:
                        cheb = interp1d(s.cheb.disp[0].norm.x, (s.cheb.disp[0].norm.y + s.cheb.disp[1].norm.y) / 2, fill_value='extrapolate')
                else:
                    cheb = interp1d(s.spec.raw.x[s.cont_mask], np.ones_like(s.spec.raw.x[s.cont_mask]), fill_value=1, bounds_error=False)

                if s.fit.n() > 0:
                    fit = np.array([s.fit.x(), s.fit.y() / cheb(s.fit.x())])
                    if self.show_comps:
                        fit_comp = []
                        for c in s.fit_comp:
                            fit_comp.append(np.array([c.x(), c.y() / cheb(c.x())]))
                    else:
                        fit_comp = None
                else:
                    fit = None
                    fit_comp = None

                if self.show_disp and len(s.fit.disp[0].norm.x) > 0:
                    fit_comp_disp = []
                    fit_disp = [s.fit.disp[0].norm.x, s.fit.disp[0].norm.y / cheb(s.fit.disp[0].norm.x), s.fit.disp[1].norm.y / cheb(s.fit.disp[1].norm.x)]
                    for comp in s.fit_comp:
                        fit_comp_disp.append([comp.disp[0].norm.x, comp.disp[0].norm.y / cheb(comp.disp[0].norm.x), comp.disp[1].norm.y / cheb(comp.disp[1].norm.x)])
                    #else:
                    #    fit_disp = [[s.fit.disp_corr[0].norm.x, s.fit.disp_corr[0].norm.y], [s.fit.disp_corr[1].norm.x, s.fit.disp_corr[1].norm.y]]
                    #    for comp in s.fit_comp:
                    #        fit_comp_disp.append([[comp.disp_corr[0].norm.x, comp.disp_corr[0].norm.y], [comp.disp_corr[1].norm.x, comp.disp_corr[1].norm.y]])


                else:
                    fit_disp, fit_comp_disp = None, None

                p.loaddata(d=np.array([s.spec.x(), s.spec.y() / cheb(s.spec.x()), s.spec.err() / cheb(s.spec.x()), s.mask.x()]),
                           f=fit, fit_comp=fit_comp, fit_disp=fit_disp, fit_comp_disp=fit_comp_disp, z=[sys.z.val for sys in self.parent.fit.sys])
                p.show_comps = self.show_comps
                if self.show_labels:
                    p.name_pos = [self.name_x_pos, self.name_y_pos]
                else:
                    p.name_pos = None
                p.add_residual, p.sig = self.residuals, self.res_sigma
                p.y_formatter = self.y_formatter
                p.plot_line()

                if over is not None:
                    p.ax.plot(over[0], over[1], ls='-', color=to_hex(tuple(c / 255 for c in self.over_color.to_bytes(4, byteorder='big'))), lw=self.fit_lw, zorder=12)

                if self.show_cf and p.show_fit:
                    for k in range(self.parent.fit.cf_num):
                        if self.cfs == 'all' or 'cf' + str(k) in self.cfs:
                            attr = 'cf_' + str(k)
                            if hasattr(self.parent.fit, attr):
                                cf = getattr(self.parent.fit, attr)
                                if (len(cf.addinfo.split('_')) > 1 and cf.addinfo.split('_')[1] == 'all') or (cf.addinfo.find('exp') > -1 and int(cf.addinfo[cf.addinfo.find('exp')+3:]) == ind):
                                    color = to_hex(tuple(c / 255 for c in self.cf_color.to_bytes(4, byteorder='big')))
                                    if p.fit_disp is None:
                                        p.ax.plot([np.max([cf.left, p.x_min]), np.min([cf.right, p.x_max])], [1 - cf.val, 1 - cf.val], '--', lw=0.5, color=color)
                                    else:
                                        p.ax.plot([np.max([cf.left, p.x_min]), np.min([cf.right, p.x_max])], [1 - cf.unc.val, 1 - cf.unc.val], '--', lw=0.5, color=color)
                                        p.ax.fill_between([np.max([cf.left, p.x_min]), np.min([cf.right, p.x_max])], 1 - cf.unc.val - cf.unc.plus, 1 - cf.unc.val + cf.unc.minus, ls=':', color=color, alpha=0.1)
                                    if self.show_cf_value and (cf.left < p.x_max) and (cf.right > p.x_min):
                                        #p.ax.text(p.x_max - (p.x_max - p.x_min) / 30, 1 - cf.unc.val, cf.fitres(latex=True), ha='right', va='bottom', fontsize=p.font_labels, color=color)
                                        p.ax.text(max(cf.left, p.x_min + (p.x_max - p.x_min) / 30), 1 - cf.unc.val, cf.fitres(latex=True), ha='left', va='bottom', fontsize=p.font_labels, fontname=p.font, color=color)

                if self.show_H2.strip() != '':
                    for speci in ['H2', 'CO', '13CO']:
                        if any([sp.startswith(speci) for sp in self.parent.fit.list_species()]):
                            if self.show_H2 == 'all':
                                levels = [sp for sp in self.parent.fit.list_species() if sp.startswith(speci)]
                            else:
                                levels = [(speci + "j") * l.isdigit() + l for l in self.show_H2.split()]
                            levels = [l for l in levels if l.startswith(speci)]
                            #print("line marks:", speci, levels)
                            if len(levels) > 0:
                                p.showLineLabels(levels=levels, pos=self.pos_H2, kind='full', only_marks=self.only_marks, show_comps=self.all_comps_marks)

                if self.show_cont:
                    if not self.show_disp or len(s.fit.disp[0].norm.x) == 0:
                        p.ax.plot(s.cheb.x(), np.power(s.cheb.y(), 1 - self.corr_cheb), '--k', lw=1)
                    else:
                        if p.vel_scale:
                            x = (s.cheb.disp[0].norm.x / p.wavelength / (1 + p.parent.z_ref) - 1) * 299794.26
                        else:
                            x = s.cheb.disp[0].norm.x
                        p.ax.fill_between(x, s.cheb.disp[0].norm.y / cheb(x), s.cheb.disp[1].norm.y / cheb(x), fc='k', alpha=p.parent.disp_alpha / 2, zorder=11)
                        p.ax.plot(x, s.cheb.disp[0].norm.y / cheb(x), '--k', lw=1, zorder=10)
                        p.ax.plot(x, s.cheb.disp[1].norm.y / cheb(x), '--k', lw=1, zorder=10)
                    if 0:
                        self.showContCorr(ax=ax)

        if 0:
            #for fit, color in zip(['C:/science/Noterdaeme/HE0001/FeI_ESPRESSO_model.spv', 'C:/science/Noterdaeme/HE0001/FeI_UVES_model.spv'], ['tab:blue', 'tab:green']):
            for fit, color in zip(['C:/science/Noterdaeme/HE0001/FeI_ESPRESSO_model.spv'], ['dodgerblue']):
                self.parent.openFile(fit)
                self.parent.showFit()
                self.ps.color_total = color
                for p in self.ps:
                    if len(self.parent.lines[self.ps.index(p)].split()) > 2:
                        for s in self.parent.lines[self.ps.index(p)].split()[2:]:
                            if 'exp' in s:
                                ind = int(s[4:])
                            if 'note' in s:
                                p.label = str(s[5:])
                    print(p.name, ind)
                    s = self.parent.s[ind]

                    if self.corr_cheb and self.parent.fit.cont_fit:
                        cheb = interp1d(s.spec.raw.x[s.cont_mask], s.correctContinuum(s.spec.raw.x[s.cont_mask]), fill_value='extrapolate')
                    else:
                        cheb = interp1d(s.spec.raw.x[s.cont_mask], np.ones_like(s.spec.raw.x[s.cont_mask]), fill_value=1)
                    self.ps.lw_total = 2
                    p.loaddata(f=np.array([s.fit.x(), s.fit.y()/cheb(s.fit.x())]))
                    p.fit_disp = None
                    p.plot_fit()

        if savefig:
            plotfile = self.plotfile
        else:
            plotfile = os.path.dirname(os.path.realpath(__file__)) + '/output/lines.pdf'

        fig.savefig(plotfile, dpi=fig.dpi)
        plt.close(fig)

        if sys.platform.startswith('darwin'):
            subprocess.call(('open', plotfile))
        elif os.name == 'nt':
            os.startfile(plotfile)
        elif os.name == 'posix':
            subprocess.call(('xdg-open', plotfile))

    def saveSettings(self):
        fname = QFileDialog.getSaveFileName(self, 'Save settings...', self.parent.plot_set_folder)[0]
        self.parent.options('plot_set_folder', os.path.dirname(fname))

        if fname:
            f = open(fname, "wb")
            o = deepcopy(self.opts)
            for opt, func in self.opts.items():
                o[opt] = func(getattr(self, opt))
            pickle.dump(o, f)
            pickle.dump(str(self.parent.lines), f)
            #if self.regions:
            #    pickle.dump(self.lines.toPlainText(), f)
            #else:
            pickle.dump(str(self.parent.plot.regions), f)
            f.close()

    def showContCorr(self, ax):
        for i in range(5,15):
            print(i)
            self.parent.fitPoly(i)
            ax.plot(self.parent.s[self.parent.s.ind].cheb.x(), self.parent.s[self.parent.s.ind].cheb.y(), '-', lw=0.5, color='mediumseagreen')

    def loadSettings(self, fname=None):
        if fname is None:
            fname = QFileDialog.getOpenFileName(self, 'Load settings...', self.parent.plot_set_folder)[0]
            self.parent.options('plot_set_folder', os.path.dirname(fname))
        if fname:
            f = open(fname, "rb")
            o = pickle.load(f)
            for opt, item in o.items():
                setattr(self, opt, item)
            self.parent.lines.fromText(str(pickle.load(f)))
            self.parent.plot.regions.fromText(str(pickle.load(f)), sort=False)
            f.close()
        self.close()
        self.parent.showLines()

    def keyPressEvent(self, event):
        super(showLinesWidget, self).keyPressEvent(event)
        key = event.key()

        if not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_F7:
                #if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier):
                self.parent.showlines.close()

    def closeEvent(self, ev):
        for opt, func in self.opts.items():
            self.parent.options(opt, func(getattr(self, opt)))
        self.parent.showlines = None
        ev.accept()

class snapShotWidget(QWidget):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.resize(800, 250)
        self.move(200,100)

        self.initData()

        layout = QVBoxLayout()
        l = QHBoxLayout()

        self.filename = QLineEdit(self.parent.work_folder + '/figure.pdf')
        self.filename.setFixedSize(650, 30)
        self.getfilename = QPushButton('Choose')
        self.getfilename.setFixedSize(70, 30)
        self.getfilename.clicked[bool].connect(self.loadfile)
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        l.addWidget(self.filename)
        l.addWidget(self.getfilename)
        l.addStretch(0)
        layout.addLayout(l)

        grid = QGridLayout()
        l = QHBoxLayout()
        l.addLayout(grid)
        l.addStretch(1)
        layout.addLayout(l)
        validator = QDoubleValidator()
        locale = QLocale('C')
        validator.setLocale(locale)
        # validator.ScientificNotation
        names = ['Size:', 'width:', '', 'height:', '',
                 'Fonts:', 'axis:', '', '', '',
                 'Label:', 'x:', '', 'y:', '',
                 'X-ticks:', 'scale:', '', 'num', '',
                 'Y-ticks:', 'scale:', '', 'num', '',
                 ]
        positions = [(i, j) for i in range(3) for j in range(5)]

        for position, name in zip(positions, names):
            if name == '':
                continue
            grid.addWidget(QLabel(name), *position)

        self.opt_but = OrderedDict([('snap_width', [0, 2]), ('snap_height', [0, 4]), ('snap_font', [1, 2]),
                                    ('snap_xlabel', [2, 2]), ('snap_ylabel', [2, 4]),
                                    ('snap_x_ticks', [3, 2]), ('snap_xnum', [3, 4]),
                                    ('snap_y_ticks', [4, 2]), ('snap_ynum', [4, 4])
                                     ])
        for opt, v in self.opt_but.items():
            b = QLineEdit(str(getattr(self, opt)))
            b.setFixedSize(100, 30)
            b.setValidator(validator)
            b.textChanged[str].connect(partial(self.onChanged, attr=opt))
            grid.addWidget(b, v[0], v[1])

        l = QHBoxLayout()
        self.ok = QPushButton('Plot')
        self.ok.setFixedSize(60, 30)
        self.ok.clicked[bool].connect(self.plot)

        self.cancel = QPushButton('Cancel')
        self.cancel.setFixedSize(60, 30)
        self.cancel.clicked[bool].connect(self.close)

        l.addStretch(0)
        l.addWidget(self.ok)
        l.addWidget(self.cancel)
        layout.addStretch(0)
        layout.addLayout(l)

        self.setLayout(layout)
        self.show()

    def initData(self):
        self.opts = {'snap_width': float, 'snap_height': float, 'snap_font': int,
                     'snap_xlabel': str, 'snap_ylabel': str,
                     'snap_x_ticks': float, 'snap_xnum': int, 'snap_y_ticks': float, 'snap_ynum': int
                     }
        for opt, func in self.opts.items():
            # print(opt, self.parent.options(opt), func(self.parent.options(opt)))
            setattr(self, opt, func(self.parent.options(opt)))

    def onChanged(self, text, attr=None):
        if attr is not None:
            setattr(self, attr, self.opts[attr](text))

    def loadfile(self):
        fname = QFileDialog.getSaveFileName(self, 'Export graph', self.parent.work_folder)
        self.filename.setText(fname[0])

    def plot(self):

        x_range = self.parent.plot.vb.viewRange()[0]
        y_range = self.parent.plot.vb.viewRange()[1]
        s = self.parent.s[self.parent.s.ind].spec
        fit = self.parent.s[self.parent.s.ind].fit
        mask = np.logical_and(s.x() > x_range[0], s.x() < x_range[1])

        fig, ax = plt.subplots(figsize=(self.snap_width,self.snap_height))
        font = 16

        if 0:
            ax.errorbar(s.x()[mask], s.y()[mask], s.err()[mask], lw=1, elinewidth=0.5, drawstyle='steps-mid',
                        color='k', ecolor='0.3', capsize=1.5)
        else:
            ax.errorbar(s.x()[mask], s.y()[mask], lw=1, elinewidth=0.5, drawstyle='steps-mid',
                        color='k', ecolor='0.3', capsize=1.5)

        if fit.n() > 0:
            ax.plot(fit.x()[mask], fit.y()[mask], lw=1, color='#e74c3c')

        ax.axis([x_range[0], x_range[1], y_range[0], y_range[1]])

        # >>> specify ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator(self.snap_xnum))
        ax.xaxis.set_major_locator(MultipleLocator(self.snap_x_ticks))
        ax.yaxis.set_minor_locator(AutoMinorLocator(self.snap_ynum))
        ax.yaxis.set_major_locator(MultipleLocator(self.snap_y_ticks))

        ax.tick_params(which='both', width=1)
        ax.tick_params(which='major', length=5)
        ax.tick_params(which='minor', length=3)
        ax.tick_params(axis='both', which='major', labelsize=self.snap_font-2)

        # >>> set axis ticks formater:
        #y_formater = "%.1f"
        #if y_formater is not None:
        #    ax.yaxis.set_major_formatter(FormatStrFormatter(y_formater))
        #x_formater = None
        #if x_formater is not None:
        #    ax.xaxis.set_major_formatter(FormatStrFormatter(.x_formater))

        # >>> set axis labels:
        ax.set_ylabel(self.snap_ylabel, fontsize=self.snap_font)
        ax.set_xlabel(self.snap_xlabel, fontsize=self.snap_font, labelpad=-4)

        print(self.filename.text())
        plt.savefig(self.filename.text())

    def closeEvent(self, ev):
        for opt, func in self.opts.items():
            self.parent.options(opt, func(getattr(self, opt)))
        ev.accept()

class fitMCMCWidget(QWidget):
    def __init__(self, parent):
        super(fitMCMCWidget, self).__init__()
        self.parent = parent

        self.initData()
        self.initGUI()
        self.setWindowTitle('Fit with MCMC')
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())

    def initData(self):
        self.savedText = ''
        self.opts = OrderedDict([
            ('MCMC_walkers', int), ('MCMC_iters', int), ('MCMC_threads', int),
            ('MCMC_burnin', int), ('MCMC_smooth', bool), ('MCMC_truth', bool),
            ('MCMC_thinning', int),
        ])
        self.thread = None

    def initGUI(self):
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)

        h = QHBoxLayout()
        grid = QGridLayout()
        validator = QDoubleValidator()
        locale = QLocale('C')
        validator.setLocale(locale)
        # validator.ScientificNotation
        names = ['Sampler:     ', '',
                 'Walkers:     ', '',
                 'Iterations:   ', '',
                 'Thinning:   ', '',
                 'Threads:', '',
                 'Priors:    ', '',
                 'Constraints:', '',
                 '', '',
                 ]
        positions = [(i, j) for i in range(8) for j in range(2)]

        for position, name in zip(positions, names):
            if name == '':
                continue
            grid.addWidget(QLabel(name), *position)

        self.sampler_opts = {'emcee': ['MCMC_threads'], 'Affine': [], 'ESS': [], 'UltraNest': ['MCMC_walkers', 'MCMC_iters'],
             'Hamiltonian': ['MCMC_walkers', 'MCMC_threads']}
        self.sampler = QComboBox()
        self.sampler.addItems(['emcee', 'Affine', 'ESS', 'UltraNest', 'Hamiltonian'])
        self.sampler.setFixedSize(120, 30)
        self.sampler.currentTextChanged.connect(self.selectSampler)
        self.sampler.setCurrentText(self.parent.options('MCMC_sampler'))
        grid.addWidget(self.sampler, 0, 1)

        self.opt_but = OrderedDict([('MCMC_walkers', [1, 1]),
                                    ('MCMC_iters', [2, 1]),
                                    ('MCMC_thinning', [3, 1]),
                                    ('MCMC_threads', [4, 1]),
                                    ])
        for opt, v in self.opt_but.items():
            setattr(self, opt, QLineEdit(str(self.parent.options(opt))))
            getattr(self, opt).setFixedSize(80, 30)
            getattr(self, opt).setValidator(validator)
            getattr(self, opt).textChanged[str].connect(partial(self.onChanged, attr=opt))
            getattr(self, opt).setEnabled(opt not in self.sampler_opts[self.parent.options('MCMC_sampler')])
            grid.addWidget(getattr(self, opt), v[0], v[1])

        self.priorField = QTextEdit('')
        self.priorField.setFixedSize(300, 400)
        self.priorField.textChanged.connect(self.priorsChanged)
        self.priorField.setText('# you can specify priors here, e.g. \n# N_0_HI 19 0.2 0.3 \n # otherwise assume to be flat \n # for comments use #')
        grid.addWidget(self.priorField, 5, 1)

        self.b_increase = QCheckBox('b increase')
        self.b_increase.setChecked(False)
        #self.b_increase.clicked[bool].connect(partial(self.setOpts, 'smooth'))
        grid.addWidget(self.b_increase, 6, 1)

        self.H2_excitation = QCheckBox('H2 excitation')
        self.H2_excitation.setChecked(False)
        # self.b_increase.clicked[bool].connect(partial(self.setOpts, 'smooth'))
        grid.addWidget(self.H2_excitation, 7, 1)

        self.hier_continuum = QCheckBox('continuum (hierarchical)')
        if hasattr(self.parent.fit, 'hcont'):
            self.hier_continuum.setChecked(self.parent.fit.getValue('hcont', 'vary'))
        else:
            self.hier_continuum.setEnabled(False)
        # self.b_increase.clicked[bool].connect(partial(self.setOpts, 'smooth'))
        grid.addWidget(self.hier_continuum, 8, 1)

        self.telluric = QCheckBox('take into account telluric')
        self.telluric.setChecked(False)
        self.telluric.setChecked(self.parent.options("telluric"))
        grid.addWidget(self.telluric, 9, 1)

        self.chooseFit = chooseFitParsWidget(self.parent, closebutton=False)
        self.chooseFit.setFixedSize(200, 700)
        v = QVBoxLayout()
        v.addLayout(grid)
        v.addStretch(1)
        h.addLayout(v)
        h.addStretch(1)
        h.addWidget(self.chooseFit)

        self.lmfit_button = QPushButton("LM fit")
        self.lmfit_button.setCheckable(True)
        self.lmfit_button.setFixedSize(60, 30)
        self.lmfit_button.clicked[bool].connect(partial(self.LMfit, init=True, filename=None))
        self.start_button = QPushButton("Start MCMC")
        self.start_button.setCheckable(True)
        self.start_button.setFixedSize(100, 30)
        self.start_button.clicked[bool].connect(partial(self.MCMC, init=True, filename=None))
        self.continue_button = QPushButton("Continue MCMC")
        self.continue_button.setFixedSize(100, 30)
        self.continue_button.clicked[bool].connect(self.continueMC)
        self.init_cluster_button = QPushButton("Init")
        self.init_cluster_button.setFixedSize(60, 30)
        self.init_cluster_button.clicked[bool].connect(partial(self.initCluster, init=True))
        self.continue_cluster_button = QPushButton("Continue")
        self.continue_cluster_button.setFixedSize(90, 30)
        self.continue_cluster_button.clicked[bool].connect(partial(self.initCluster, init=False))
        self.import_cluster_button = QPushButton("Import")
        self.import_cluster_button.setFixedSize(90, 30)
        self.import_cluster_button.clicked[bool].connect(partial(self.importCluster))
        #self.cont_fit = QPushButton("Fit cont")
        #self.cont_fit.setFixedSize(70, 30)
        #self.cont_fit.clicked[bool].connect(self.fitCont)
        hbox = QHBoxLayout()
        hbox.addWidget(self.lmfit_button)
        hbox.addWidget(self.start_button)
        hbox.addWidget(self.continue_button)
        hbox.addWidget(QLabel('  Cluster:'))
        hbox.addWidget(self.init_cluster_button)
        hbox.addWidget(self.continue_cluster_button)
        hbox.addWidget(self.import_cluster_button)
        hbox.addStretch(1)
        #hbox.addWidget(self.cont_fit)

        fitlayout = QVBoxLayout()
        #fitlayout.addWidget(QLabel('Fit MCMC:'))
        fitlayout.addLayout(h)
        fitlayout.addStretch(1)
        fitlayout.addLayout(hbox)
        widget = QWidget()
        widget.setLayout(fitlayout)
        splitter.addWidget(widget)

        h = QHBoxLayout()
        grid = QGridLayout()
        names = ['', '',
                 'Burn-in: ', '',
                 '', '',
                 '', '',
                 '', '',
                 'Results:', '',
                 ]
        positions = [(i, j) for i in range(6) for j in range(2)]

        for position, name in zip(positions, names):
            if name == '':
                continue
            grid.addWidget(QLabel(name), *position)

        self.opt_but = OrderedDict([('MCMC_burnin', [1, 1]),
                                    ])
        for opt, v in self.opt_but.items():
            b = QLineEdit(str(self.parent.options(opt)))
            b.setFixedSize(80, 30)
            b.setValidator(validator)
            b.textChanged[str].connect(partial(self.onChanged, attr=opt))
            grid.addWidget(b, v[0], v[1])

        grid.addWidget(QLabel('Plot in:'), 0, 0)
        self.graph = QComboBox()
        self.graph.addItems(['chainConsumer', 'corner'])
        self.graph.setFixedSize(120, 30)
        self.graph.setCurrentIndex(['chainConsumer', 'corner'].index(self.parent.options('MCMC_graph')))
        self.graph.activated.connect(self.selectGraph)
        grid.addWidget(self.graph, 0, 1)
        grid.addWidget(QLabel('Truths:'), 2, 0)
        self.truths = QComboBox()
        self.truths.addItems(['None', 'Max L', 'Model', 'MAP'])
        self.truths.setFixedSize(120, 30)
        self.truths.setCurrentText(self.parent.options('MCMC_truths'))
        self.truths.currentTextChanged.connect(self.selectTruths)
        grid.addWidget(self.truths, 2, 1)
        self.smooth = QCheckBox('smooth')
        self.smooth.setChecked(bool(self.parent.options('MCMC_smooth')))
        self.smooth.clicked[bool].connect(partial(self.setOpts, 'smooth'))
        grid.addWidget(self.smooth, 3, 0)
        self.likelihood = QCheckBox('show likelihood')
        self.likelihood.setChecked(bool(self.parent.options('MCMC_likelihood')))
        self.likelihood.clicked[bool].connect(partial(self.setOpts, 'likelihood'))
        grid.addWidget(self.likelihood, 4, 0)
        self.results = QTextEdit('')
        self.results.setFixedSize(500, 400)
        self.results.setText('# fit results are here')
        self.results.textChanged.connect(self.fitresChanged)
        grid.addWidget(self.results, 5, 1)
        self.chooseShow = chooseShowParsWidget(self.parent)
        self.chooseShow.setFixedSize(200, 700)
        v = QVBoxLayout()
        v.addLayout(grid)
        v.addStretch(1)
        h.addLayout(v)
        h.addStretch(1)
        h.addWidget(self.chooseShow)

        self.show_button = QPushButton("Show")
        self.show_button.setFixedSize(100, 30)
        self.show_button.clicked[bool].connect(partial(self.showMC, mask=None, pars=None, samples=None, lnprobs=None))
        self.show_comp_button = QPushButton("Show comps")
        self.show_comp_button.setFixedSize(100, 30)
        self.show_comp_button.clicked[bool].connect(self.showCompsMC)
        self.check_button = QPushButton("Check")
        self.check_button.setFixedSize(100, 30)
        self.check_button.clicked[bool].connect(self.check)
        self.bestfit_button = QPushButton("Best fit")
        self.bestfit_button.setFixedSize(100, 30)
        self.bestfit_button.clicked[bool].connect(self.show_bestfit)
        self.stats_button = QPushButton("Stats")
        self.stats_button.setFixedSize(100, 30)
        self.stats_button.clicked[bool].connect(partial(self.stats, t='fit'))
        self.stats_all_button = QPushButton("Stats all")
        self.stats_all_button.setFixedSize(100, 30)
        self.stats_all_button.clicked[bool].connect(partial(self.stats, t='all'))
        self.stats_cols_button = QPushButton("Stats cols")
        self.stats_cols_button.setFixedSize(100, 30)
        self.stats_cols_button.clicked[bool].connect(partial(self.stats, t='cols'))
        self.stats_ratios_button = QPushButton("Stats ratios:")
        self.stats_ratios_button.setFixedSize(100, 30)
        self.stats_ratios_button.clicked[bool].connect(partial(self.stats, t='ratios'))
        self.ratios_species = QLineEdit()
        self.ratios_species.setFixedSize(100, 30)

        self.fit_disp_button = QPushButton("Fit disp:")
        self.fit_disp_button.setFixedSize(100, 30)
        self.fit_disp_button.clicked[bool].connect(partial(self.fit_disp, calc=True))
        self.fit_disp_num = QLineEdit(str(self.parent.options('MCMC_disp_num')))
        self.fit_disp_num.setFixedSize(80, 30)
        self.fit_disp_num.textChanged.connect(self.set_fit_disp_num)

        self.loadres_button = QPushButton("Import")
        self.loadres_button.setFixedSize(120, 30)
        self.loadres_button.clicked[bool].connect(self.loadres)

        self.loaddisp_button = QPushButton("Load disp")
        self.loaddisp_button.setFixedSize(100, 30)
        self.loaddisp_button.clicked[bool].connect(self.loaddisp)

        self.export_button = QPushButton("Export")
        self.export_button.setFixedSize(120, 30)
        self.export_button.clicked[bool].connect(self.export)

        showlayout = QVBoxLayout()
        showlayout.addLayout(h)
        showlayout.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addWidget(self.show_button)
        hbox.addWidget(self.show_comp_button)
        hbox.addWidget(self.check_button)
        hbox.addWidget(self.bestfit_button)
        hbox.addStretch(1)
        hbox.addWidget(self.loadres_button)
        showlayout.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(self.stats_button)
        hbox.addWidget(self.stats_all_button)
        hbox.addWidget(self.stats_cols_button)
        hbox.addWidget(self.stats_ratios_button)
        hbox.addWidget(self.ratios_species)
        hbox.addStretch(1)
        hbox.addWidget(self.export_button)
        showlayout.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(self.fit_disp_button)
        hbox.addWidget(self.fit_disp_num)
        hbox.addWidget(self.loaddisp_button)
        hbox.addStretch(1)
        showlayout.addLayout(hbox)

        widget = QWidget()
        widget.setLayout(showlayout)
        splitter.addWidget(widget)
        splitter.setSizes([1000, 1200])
        layout.addWidget(splitter)

        self.setLayout(layout)

        self.setGeometry(200, 200, 1450, 800)
        self.setWindowTitle('Fit model')
        self.show()

    def onChanged(self, text, attr=None):
        if attr is not None:
            setattr(self, attr, self.opts[attr](text))
            self.parent.options(attr, self.opts[attr](text))

    def selectSampler(self):
        self.parent.options('MCMC_sampler', self.sampler.currentText())
        for attr in ['MCMC_walkers', 'MCMC_iters', 'MCMC_thinning', 'MCMC_threads']:
            if hasattr(self, attr):
                getattr(self, attr).setEnabled(attr not in self.sampler_opts[self.sampler.currentText()])

    def priorsChanged(self):
        self.priors = {}
        for line in self.priorField.toPlainText().splitlines():
            if not line.startswith('#'):
                line = line.replace('=', '')
                words = line.split()
                if words[0] in self.parent.fit.pars():
                    if len(words) == 2:
                        self.priors[words[0]] = a(words[1], 'd')
                    elif len(words) == 3:
                        self.priors[words[0]] = a(float(words[1]), float(words[2]), 'd')
                    elif len(words) == 4:
                        self.priors[words[0]] = a(float(words[1]), float(words[2]), float(words[3]), 'd')

    def fitresChanged(self):
        for line in self.results.toPlainText().splitlines():
            if not line.startswith('#'):
                words = line.split()
                if words[0] in self.parent.fit.pars():
                    if len(words) == 3:
                        self.parent.fit.setValue(words[0], words[2], attr='unc')

    def setOpts(self, arg=None):
        setattr(self, 'MCMC_'+arg, getattr(self, arg).isChecked())
        #print(arg, getattr(self, arg).isChecked(), getattr(self, 'MCMC_'+arg))
        self.parent.options('MCMC_'+arg, getattr(self, 'MCMC_'+arg))

    def selectGraph(self, text):
        self.parent.options('MCMC_graph', ['chainConsumer', 'corner'][text])
        self.graph.setCurrentIndex(['chainConsumer', 'corner'].index(self.parent.options('MCMC_graph')))

    def selectTruths(self, text):
        self.parent.options('MCMC_truths', text)
        self.truths.setCurrentText(self.parent.options('MCMC_truths'))

    def start(self, init=True, filename=None):
        self.MCMC(init=init, filename=filename)
        if 0 and self.thread is None:
            self.start_button.setChecked(True)
            if 0:
                from multiprocessing import Process
                self.thread = Process(target=dosome, args=(1,))
                #self.thread = Process(target=self.MCMC, args=(self,), kwargs={'init': init})
            else:
                self.thread = StoppableThread(target=self.MCMC, args=(), kwargs={'init': init}, daemon=True)
                self.thread.daemon = True
                #self.thread = threading.Thread(target=self.MCMC, args=(), kwargs={'init': init}, daemon=True)
            self.thread.start()

    def initCluster(self, init=True):

        #fname = QFileDialog.getSaveFileName(self, 'Select/set file', self.parent.work_folder + , ".spj")
        fname = QFileDialog.getSaveFileName(self, 'Select/set file', self.parent.options('filename_saved').replace('.spv', '.spj'), ".spj")
        if fname[0]:
            self.MCMC(init=init, filename=fname[0])

    def loadJulia(self, filename):
        self.parent.julia.include("MCMC.jl")

        chain, lns = self.parent.julia.readMCMC(filename, convert=True)
        chain, lns = np.asarray(chain), np.asarray(lns)

        if os.path.exists("output/mcmc.hdf5"):
            os.remove("output/mcmc.hdf5")

        nwalkers, npars, nsteps = chain.shape[0], chain.shape[1], chain.shape[2]
        backend = emcee.backends.HDFBackend("output/mcmc.hdf5")
        backend.reset(nwalkers, npars)

        with backend.open("w") as f:
            backend.reset(nwalkers, npars) #np.sum([p.vary for p in self.parent.julia_pars.values()]))
            backend.grow(nsteps, None)
            g = f[backend.name]
            g.attrs["iteration"] = nsteps
            if lns is not None:
                g["log_prob"][...] = lns.transpose()
            print(chain.shape, chain.transpose(2, 0, 1).shape)
            g["chain"][...] = chain.transpose(2, 0, 1)
            g.attrs["pars"] = [p.encode() for p in [str(p) for p in self.parent.fit.list_fit()]]

    def importCluster(self):
        fname = QFileDialog.getOpenFileName(self, 'Import MCMC model', self.parent.work_folder)

        if fname[0]:
            if fname[0].endswith('.spj'):
                self.importJulia(fname[0])
            else:
                self.parent.options('work_folder', os.path.dirname(fname[0]))
                self.parent.MCMC_output = fname[0]

    def importJulia(self, filename):
        print(filename)
        self.parent.julia.include("MCMC.jl")
        pars = self.parent.julia.readJulia(filename)
        print(pars)

    def stop(self):
        self.start_button.setChecked(False)
        if 1:
            self.thread.terminate()
        else:
            self.thread.stop()
        self.thread = None

    def continueMC(self):
        self.start(init=False, filename=None)

    def LMfit(self, init=True, filename=None):
        opts = {'b_increase': self.b_increase.isChecked(), 'H2_excitation': self.H2_excitation.isChecked(), 'hier_continuum': self.hier_continuum.isChecked()}

        self.parent.fitJulia(opts=opts)

    def MCMC(self, init=True, filename=None):
        print(init, filename)
        #self.parent.setFit(comp=-1)
        nwalkers, nsteps, nthreads, thinning = int(self.parent.options('MCMC_walkers')), int(self.parent.options('MCMC_iters')), int(self.parent.options('MCMC_threads')), int(self.parent.options('MCMC_thinning'))

        opts = {"b_increase": self.b_increase.isChecked(), "H2_excitation": self.H2_excitation.isChecked(),
                "hier_continuum": self.hier_continuum.isChecked(), "telluric": self.telluric.isChecked()}

        if init:
            init = []
            for par in self.parent.fit.list_fit():
                val = par.val * np.ones(nwalkers) + np.random.randn(nwalkers) * par.step
                val[val > par.max] = par.max
                val[val < par.min] = par.min
                init.append(val)
            init = np.array(init).transpose()
        else:
            pars, samples, lnprobs = self.readChain()
            init = samples[-1, :, :]

        print(self.priors)

        if not self.parent.normview:
            self.parent.normalize(True)

        self.parent.s.prepareFit(ind=-1, all=False)
        self.parent.s.calcFit(ind=-1, redraw=False)

        pars = [str(p) for p in self.parent.fit.list_fit()]
        print(len(pars), pars)

        backend = emcee.backends.HDFBackend("output/mcmc.hdf5")
        backend.reset(nwalkers, np.sum([p.vary for p in self.parent.julia_pars.values()]))

        t = Timer("MCMC run")

        if self.sampler.currentText() in ['Affine', 'ESS', 'UltraNest', 'Hamiltonian']:

            self.parent.julia.include("MCMC.jl")

            if filename is not None:

                if 0:
                    self.parent.julia.initJulia(filename, self.parent.julia_spec, self.parent.julia_pars, self.parent.julia_add, self.parent.fit.list_names(),
                                                sampler=self.sampler.currentText(), prior=self.priors, nwalkers=nwalkers, nsteps=nsteps,
                                                nthreads=nthreads, thinning=thinning, init=init, opts=opts)
                else:
                    self.parent.julia.initJulia2(filename, self.parent.s, fit=self.parent.fit, fit_list=self.parent.fit.list(),
                                                 parnames=self.parent.fit.list_names(), tieds=self.parent.fit.tieds,
                                                 sampler=self.sampler.currentText(), prior=self.priors,
                                                 nwalkers=nwalkers, nsteps=nsteps,
                                                 nthreads=nthreads, thinning=thinning, init=init, opts=opts)

            else:
                chain, lns = self.parent.julia.fitMCMC(self.parent.julia_spec, self.parent.julia_pars, self.parent.julia_add, self.parent.fit.list_names(),
                                                       sampler=self.sampler.currentText(), prior=self.priors, nwalkers=nwalkers, nsteps=nsteps,
                                                       nthreads=nthreads, thinning=thinning, init=init, opts=opts)
                chain, lns = np.asarray(chain), np.asarray(lns)
                if self.sampler.currentText() == 'UltraNest':
                    from ultranest.plot import cornerplot
                    cornerplot(chain)
                    plt.show()

                #backend.grow(nsteps, None)

                with backend.open("w") as f:
                    backend.reset(nwalkers, np.sum([p.vary for p in self.parent.julia_pars.values()]))
                    backend.grow(nsteps // thinning, None)
                    g = f[backend.name]
                    g.attrs["iteration"] = nsteps // thinning
                    if lns is not None:
                        #print(lns)
                        g["log_prob"][...] = np.transpose(lns)
                    print(chain.shape, chain.transpose(2, 0, 1).shape)
                    g["chain"][...] = chain.transpose(2, 0, 1)
                    g.attrs["pars"] = [p.encode() for p in pars]

        elif self.sampler.currentText() in ['emcee']:

            def lnprob(params, pars, priors, self):
                #print(params)
                for v, p in zip(params, pars):
                    if not self.parent.fit.setValue(p, v):
                        return -np.inf
                self.parent.fit.update(redraw=False)
                self.parent.julia_pars = self.parent.julia.make_pars(self.parent.fit.list(), tieds=self.parent.fit.tieds)
                #self.parent.s.prepareFit(all=False)
                self.parent.s.calcFit(recalc=True)
                lp = 0
                for k, v in priors.items():
                    print(k, v)
                    lp += v.lnL(p)
                return lp - 0.5 * self.parent.s.chi2()

            ndims = len(pars)
            sampler = emcee.EnsembleSampler(nwalkers, ndims, lnprob, args=[pars, self.priors, self], backend=backend)

            for i, result in enumerate(sampler.sample(init, iterations=nsteps)):
                print(i)
                self.parent.MCMCprogress.setText('     MCMC is running: {0:d} / {1:d}'.format(i, nsteps))

            with backend.open("a") as f:
                g = f[backend.name]
                f[backend.name].attrs["pars"] = [p.encode() for p in pars]

        t.time("finished")

        self.parent.MCMC_output = "output/mcmc.hdf5"
        self.start_button.setChecked(False)

    def readChain(self, ):
        #with open("output/MCMC_pars.pkl", "rb") as f:
        #    pars = pickle.load(f)

        if self.parent.MCMC_output.endswith('hdf5'):
            backend = emcee.backends.HDFBackend(self.parent.MCMC_output)

            try:
                with backend.open('r') as f:
                    g = f[backend.name]
                    print(list(g.keys()))
                    print(list(g.attrs.keys()))
                    #print([g.attrs[k] for k in list(g.attrs.keys())])
                    g.attrs['iteration'] = g['chain'][:].shape[0]
                    print(g.attrs['iteration'])
                    pars = [p.decode() for p in g.attrs['pars']]
            except:
                pars = [str(p) for p in self.parent.fit.list_fit()]
            lnprobs = backend.get_log_prob()
            samples = backend.get_chain()
            return pars, samples, lnprobs

        elif self.parent.MCMC_output.endswith('pickle'):
            with open(self.parent.MCMC_output, 'rb') as f:
                pars = pickle.load(f)
                samples = np.asarray(pickle.load(f))
                lnprobs = np.asarray(pickle.load(f))
            samples = samples.reshape((samples.shape[0], 1, samples.shape[1]))
            lnprobs = lnprobs.reshape((lnprobs.shape[0], 1))
            return pars, samples, lnprobs

    def showCompsMC(self):
        pars, samples, lnprobs = self.readChain()

        for ind in range(-1, len(self.parent.fit.sys)):
            loc_pars = [str(i) for i in self.parent.fit.list(ind)]
            mask = np.array([self.parent.fit.list()[[str(i) for i in self.parent.fit.list()].index(p)].fit and p in loc_pars for p in pars])
            if np.sum(mask) > 0:
                self.showMC(mask=mask, pars=pars, samples=samples, lnprobs=lnprobs)

    def showMC(self, mask=None, pars=None, samples=None, lnprobs=None):

        if any([pars is None, samples is None, lnprobs is None]):
            pars, samples, lnprobs = self.readChain()
        nsteps, nwalkers, = lnprobs.shape
        burnin = int(self.parent.options('MCMC_burnin'))
        if burnin < samples.shape[0]:
            samples = samples[burnin:, :, :]
            lnprobs = lnprobs[burnin:, :]
        else:
            self.parent.sendMessage("Burn-in length is larger than the sample size")
            return

        if mask is None:
            mask = np.array([self.parent.fit.list()[[str(i) for i in self.parent.fit.list()].index(p)].show for p in pars])

        #print(mask)
        names = [str(p).replace('_', ' ') for i, p in enumerate(self.parent.fit.list_fit()) if mask[i]]
        truth = None
        print(self.parent.options('MCMC_truths'))
        print(samples.shape)
        if self.parent.options('MCMC_truths') == 'Max L':
            inds = np.where(lnprobs == np.max(lnprobs))
            truth = samples[inds[0][0], inds[1][0], :][np.where(mask)[0]]
        elif self.parent.options('MCMC_truths') == 'Model':
            truth = np.asarray([self.parent.fit.getValue(par) for par in pars])[np.where(mask)[0]]
        elif self.parent.options('MCMC_truths') == 'MAP':
            truth = []
            for i, p in enumerate(pars):
                print(p, names)
                if p.replace('_', ' ') in names:
                    s = samples[:, :, i].flatten()
                    # print(np.min(s), np.max(s))
                    x = np.linspace(np.min(s), np.max(s), 100)
                    kde = gaussian_kde(s)
                    d = distr1d(x, kde(x))
                    d.dopoint()
                    truth.append(d.point)
            #print('MAP estimate is not currently available')
            print("MAP:", truth)
        print('truths:', truth)
        
        if self.parent.options('MCMC_likelihood'):
            names = [r'$\chi^2$'] + names
            samples = np.insert(samples, 0, lnprobs, axis=len(samples.shape)-1)
            if truth is not None:
                truth = np.insert(truth, 0, np.max(lnprobs))
            mask = np.insert(mask, 0, True)

        if self.parent.blindMode:
            truth = None


        if self.parent.options('MCMC_graph') == 'chainConsumer':
            c = ChainConsumer()
            #print(pd.DataFrame(data=samples.reshape(-1, samples.shape[-1])[:, np.where(mask)[0]], columns=names))
            c.add_chain(Chain(samples=pd.DataFrame(data=samples.reshape(-1, samples.shape[-1])[:, np.where(mask)[0]], columns=names),# walkers=nwalkers,
                        name="posteriors",
                        #parameters=names,
                        smooth=self.parent.options('MCMC_smooth'),
                        #colors='tab:red',
                        #cmap='Reds',
                        #marker_size=2,
                        plot_cloud=True,
                        shade=True,
                        sigmas=[0, 1, 2, 3],
                        ))
            #c.configure_truth(ls='--', lw=1., c='lightblue')  # c='darkorange')
            print(truth)
            if truth is not None:
                c.add_truth(Truth(location={n:t for n, t in zip(names, truth)}))

            c.set_plot_config(PlotConfig(blind=self.parent.blindMode,
                                         # flip=True,
                                         # labels={"A": "$A$", "B": "$B$", "C": r"$\alpha^2$"},
                                         # contour_label_font_size=12,
                                         ))
            figure = c.plotter.plot(figsize=(20, 20),
                                    #filename="output/fit.png",
                                    #display=not self.parent.blindMode,
                                    )
            if self.parent.blindMode:
                for ax in figure.axes:
                    ax.set_title('')

            if self.parent.options('MCMC_graph') == 'corner':
                figure = corner.corner(samples.reshape(-1, samples.shape[-1])[:, np.where(mask)[0]],
                                       labels=names,
                                       show_titles= not self.parent.blindMode,
                                       plot_contours=self.parent.options('MCMC_smooth'),
                                       truths=truth,
                                       )
            if self.parent.blindMode:
                for ax in figure.axes:
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                self.parent.sendMessage("You are currently in blind mode. The actual values are not shown. Disable it in Preference menu (F11)")

            plt.show()

    def stats(self, t='fit'):
        pars, samples, lnprobs = self.readChain()
        nsteps, nwalkers, = lnprobs.shape
        burnin = int(self.parent.options('MCMC_burnin'))

        inds = np.where(lnprobs == np.max(lnprobs))
        truth = samples[inds[0][0], inds[1][0], :] if bool(self.parent.options('MCMC_bestfit')) else None
        print(truth)
        self.results.setText('')

        print(t)
        if t == 'fit':
            mask = np.array([p.show for p in self.parent.fit.list_fit()])
            names = [str(p) for p in self.parent.fit.list_fit() if p.show]

            k = int(np.sum(mask)) #samples.shape[1]
            n_hor = int(k ** 0.5)
            n_hor = np.max([n_hor, 2])
            n_vert = k // n_hor + 1 if k % n_hor > 0 else k // n_hor
            n_vert = np.max([n_vert, 2])

            fig, ax = plt.subplots(nrows=n_vert, ncols=n_hor, figsize=(6 * n_vert, 4 * n_hor))
            k = 0

            mask = lnprobs[burnin:, :] > -30000
            print(lnprobs)
            print(lnprobs.shape, lnprobs[burnin:, :].shape, np.sum(mask))

            for i, p in enumerate(pars):
                print(i, p)
                if p in names:
                    s = samples[burnin:, :, i][mask].flatten()
                    #print(np.min(s), np.max(s))
                    x = np.linspace(np.min(s), np.max(s), 100)
                    kde = gaussian_kde(s)
                    d = distr1d(x, kde(x))
                    d.dopoint()
                    d.dointerval()
                    res = a(d.point, d.interval[1] - d.point, d.point - d.interval[0], self.parent.fit.getPar(p).form)

                    self.parent.fit.setValue(p, res, 'unc')
                    self.parent.fit.setValue(p, res.val)
                    self.parent.fit.setValue(p, (res.plus + res.minus) / 2, 'step')
                    if not self.parent.blindMode:
                        f = np.asarray([res.plus, res.minus])
                        f = int(np.round(np.abs(np.log10(np.min(f[np.nonzero(f)])))) + 1)
                        print(p, res.latex(f=f))
                        self.results.setText(self.results.toPlainText() + p + ': ' + res.latex(f=f) + '\n')
                        vert, hor = k // n_hor, k % n_hor
                        k += 1
                        d.plot(conf=0.683, ax=ax[vert, hor], ylabel='')
                        if truth is not None:
                            ax[vert, hor].axvline(truth[i], c='navy', ls='--', lw=1)
                        ax[vert, hor].yaxis.set_ticklabels([])
                        ax[vert, hor].yaxis.set_ticks([])
                        ax[vert, hor].text(.05, .9, str(p).replace('_', ' '), ha='left', va='top', transform=ax[vert, hor].transAxes)
                        ax[vert, hor].text(.95, .9, self.parent.fit.getPar(p).fitres(latex=True, showname=False), ha='right', va='top', transform=ax[vert, hor].transAxes)
                    else:
                        self.parent.sendMessage("You are currently in blind mode. The actual values are not shown. Disable it in Preference menu (F11)")
                    #ax[vert, hor].set_title(pars[i].replace('_', ' '))

        elif t == 'ratios':
            ratios = [s.split('/') for s in self.ratios_species.text().split()]
            if len(ratios) > 0:
                print(ratios)
                k = len(self.parent.fit.sys) * len(ratios)  # samples.shape[1]
                n_hor = int(k ** 0.5)
                n_hor = np.max([n_hor, 2])
                n_vert = k // n_hor + 1 if k % n_hor > 0 else k // n_hor
                n_vert = np.max([n_vert, 2])

                fig, ax = plt.subplots(nrows=n_vert, ncols=n_hor, figsize=(6 * n_vert, 4 * n_hor))
                k = 0
                for i in range(len(self.parent.fit.sys)):
                    for ratio in ratios:
                        d = {}
                        for r in ratio:
                            if 'total' in r:
                                inds = np.where(['N_{0:d}_{1:s}'.format(i, r.split('total')[0]) in str(s) for s in pars])[0]
                                print(r, inds)
                                d[r] = np.log10(np.sum(10 ** samples[burnin:, :, inds], axis=2)).flatten()
                            else:
                                print(r, np.where([str(s) == 'N_{0:d}_{1:s}'.format(i, r) for s in pars])[0])
                                d[r] = samples[burnin:, :, np.where([str(s) == 'N_{0:d}_{1:s}'.format(i, r) for s in pars])[0]].flatten()
                        #inds = [np.where([str(s) == 'N_{0:d}_{1:s}'.format(i, r) for s in pars])[0] for r in ratio]
                        #print(inds)
                        if len(d.keys()) > 1:
                            d = distr1d(d[ratio[0]] - d[ratio[1]])
                            d.dopoint()
                            d.dointerval()
                            res = a(d.point, d.interval[1] - d.point, d.point - d.interval[0], 'log')
                            f = int(np.round(np.abs(np.log10(np.min([res.plus, res.minus])))) + 1)
                            self.results.setText(self.results.toPlainText() + str('{0:s}_{1:s}_{2:d}'.format(ratio[0], ratio[1], i)) + ': ' + res.latex(f=f) + '\n')
                            print('{0:s}_{1:s}_{2:d}'.format(ratio[0], ratio[1], i) + ': ' + res.latex(f=f) + '\n')
                            # vert, hor = int((i) / n_hor), i - n_hor * int((i) / n_hor)
                            vert, hor = k // n_hor, k % n_hor
                            print(vert, hor)
                            k += 1
                            d.plot(conf=0.683, ax=ax[vert, hor], ylabel='')
                            ax[vert, hor].yaxis.set_ticklabels([])
                            ax[vert, hor].yaxis.set_ticks([])
                            ax[vert, hor].text(.1, .9, str('{0:s}/{1:s} {2:d}'.format(ratio[0], ratio[1], i)), ha='left', va='top', transform=ax[vert, hor].transAxes)
                            ax[vert, hor].text(.1, .8, res.latex(f=f), ha='left', va='top', transform=ax[vert, hor].transAxes)
        else:
            values = []
            if 0:
                for k in range(burnin, samples.shape[0]):
                    for i in range(samples.shape[1]):
                        for xi, p in zip(samples[k, i], pars):
                            self.parent.fit.setValue(p, xi)
                        self.parent.fit.update()
                        values.append([p.val for p in self.parent.fit.list()])
            else:
                kis, iis = burnin + np.random.randint(samples.shape[0] - burnin, size=5000), np.random.randint(samples.shape[1], size=5000)
                for k, i in zip(kis, iis):
                    #print(k, i)
                    for xi, p in zip(samples[k, i], pars):
                        self.parent.fit.setValue(p, xi)
                    self.parent.fit.update()
                    values.append([p.val for p in self.parent.fit.list()])
            values = np.asarray(values)
            print("values done")

            if t == 'all':
                k = len(self.parent.fit.list())  # samples.shape[1]
                n_hor = int(k ** 0.5)
                if n_hor <= 1:
                    n_hor = 2
                n_vert = k // n_hor + 1 if k % n_hor > 0 else k // n_hor

                fig, ax = plt.subplots(nrows=n_vert, ncols=n_hor, figsize=(6 * n_vert, 4 * n_hor))

                k = 0
                for i, p in enumerate(self.parent.fit.list()):
                    print(p, np.std(values[:, i]))
                    if np.std(values[:, i]) > 0:
                        d = distr1d(values[:, i])
                        try:
                            d.dopoint()
                            d.dointerval()
                            res = a(d.point, d.interval[1] - d.point, d.point - d.interval[0], p.form)
                            f = int(np.round(np.abs(np.log10(np.min([res.plus, res.minus])))) + 1)
                            self.results.setText(self.results.toPlainText() + str(p) + ': ' + res.latex(f=f) + '\n')
                            #vert, hor = int((i) / n_hor), i - n_hor * int((i) / n_hor)
                            vert, hor = k // n_hor, k % n_hor
                            k += 1
                            d.plot(conf=0.683, ax=ax[vert, hor], ylabel='')
                            ax[vert, hor].yaxis.set_ticklabels([])
                            ax[vert, hor].yaxis.set_ticks([])
                            ax[vert, hor].text(.1, .9, str(p).replace('_', ' '), ha='left', va='top', transform=ax[vert, hor].transAxes)
                            #ax[vert, hor].set_title(str(p).replace('_', ' '))
                        except:
                            k += 1

            elif t == 'cols':
                species = set()
                for i, sys in enumerate(self.parent.fit.sys):
                    for s in sys.sp.keys():
                        if s not in species and sys.sp[s].N.fit:
                            species.add(s)
                    for el in ['H2', 'HD', 'CO', 'CI', 'CII', 'FeII']:
                        if any([el+'j' in s for s in sys.sp.keys()]):
                            self.parent.fit.sys[i].addSpecies(el, 'total')
                            self.parent.fit.total.addSpecies(el + '_total')
                print(species)
                for s in species:
                    self.parent.fit.total.addSpecies(s)

                species = self.parent.fit.list_total()
                print(species)
                n_hor = int(len(species) ** 0.5)
                if n_hor <= 1:
                    n_hor = 2
                n_vert = len(species) // n_hor + 1 if len(species) % n_hor > 0 else len(species) // n_hor
                if n_vert <= 1:
                    n_vert = 2
                print(self.parent.fit.list())
                fig, ax = plt.subplots(nrows=n_vert, ncols=n_hor, figsize=(6 * n_vert, 4 * n_hor))
                k = 0
                for sp, v in species.items():
                    print(sp, v)
                    if 'total' in sp:
                        if sp.split('_')[-1] == 'total':
                            inds = np.where([sp.split('_')[2] in str(s) and str(s)[0] == 'N' for s in self.parent.fit.list()])[0]
                        else:
                            inds = np.where([str(s)[0] == 'N' and sp.split('_')[2] == str(s).split('_')[2] for s in self.parent.fit.list()])[0]
                    else:
                        inds = np.where([sp[2:] in str(s) and str(s)[0] == 'N' for s in self.parent.fit.list()])[0]
                    print(inds)
                    if 1:
                        d = distr1d(np.log10(np.sum(10 ** values[:, inds], axis=1)))
                        d.dopoint()
                        d.dointerval()
                        res = a(d.point, d.interval[1] - d.point, d.point - d.interval[0], v.form)
                        v.set(res, attr='unc')
                        if 'total' in sp:
                            self.parent.fit.total.sp['_'.join(sp.split('_')[2:])].N.set(res, attr='unc')
                            print(self.parent.fit.total.sp['_'.join(sp.split('_')[2:])].N)
                        else:
                            self.parent.fit.sys[int(sp.split('_')[1])].total[sp.split('_')[2]].N.set(res, attr='unc')
                            print(self.parent.fit.sys[int(sp.split('_')[1])].total[sp.split('_')[2]].N)
                        v.set(d.point)
                        f = int(np.round(np.abs(np.log10(np.min([res.plus, res.minus])))) + 1)
                        self.results.setText(self.results.toPlainText() + sp + ': ' + v.fitres(latex=True, dec=f, showname=False) + '\n')
                        vert, hor = k // n_hor, k % n_hor
                        k += 1
                        d.plot(conf=0.683, ax=ax[vert, hor], ylabel='')
                        ax[vert, hor].yaxis.set_ticklabels([])
                        ax[vert, hor].yaxis.set_ticks([])
                        ax[vert, hor].text(.1, .9, sp.replace('_', ' '), ha='left', va='top', transform=ax[vert, hor].transAxes)

        if k < n_hor * n_vert - 1:
            for i in range(i+1, n_hor * n_vert):
                vert, hor = i // n_hor, i % n_hor
                fig.delaxes(ax[vert, hor])

        plt.tight_layout()
        plt.subplots_adjust(wspace=0)
        plt.show()

    def check(self):
        self.MCMCqc()

    def MCMCqc(self, qc='all'):
        pars, samples, lnprobs = self.readChain()
        nsteps, nwalkers, = lnprobs.shape
        print(lnprobs.shape)
        ndims = len(pars)
        print(ndims, pars)
        print(samples.shape)

        if any([s in qc for s in ['current', 'moments', 'all']]):
            n_hor = int(ndims ** 0.5)
            if n_hor <= 1:
                n_hor = 2
            n_vert = int(ndims / n_hor + 1)

        if any([s in qc for s in ['current', 'all']]):
            for i, p in enumerate(pars):
                if p.startswith('z_'):
                    samples[:, :, i] = (samples[:, :, i] / np.mean(samples[:, :, i], axis=0) - 1) * 300000
                #print(i, p)
            fig, ax0 = plt.subplots(nrows=n_vert, ncols=n_hor, figsize=(6 * n_vert, 4 * n_hor))
            ax0[0, 0].hist(-lnprobs[nsteps-1, :][np.isfinite(lnprobs[nsteps-1, :])], 20, density=1, histtype='bar', color='crimson', label=r'$\chi^2$')
            ax0[0, 0].legend()
            ax0[0, 0].set_title(r'$\chi^2$ distribution')
            for i in range(ndims):
                vert, hor = int((i + 1) / n_hor), i + 1 - n_hor * int((i + 1) / n_hor)
                ax0[vert, hor].scatter(samples[nsteps-1, :, i], -lnprobs[nsteps-1, :], c='r')
                ax0[vert, hor].text(.1, .9, str(pars[i]).replace('_', ' '), ha='left', va='top',
                                   transform=ax0[vert, hor].transAxes)

        if self.parent.blindMode:
            for ax in fig.axes:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

        plt.subplots_adjust(wspace=0)
        plt.tight_layout()

        if any([s in qc for s in ['moments', 'all']]):
            ind = np.random.randint(0, nwalkers)
            SomeChain = samples[:, 1+ind, :]
            mean, std, chimin = np.empty([ndims, nsteps]), np.empty([ndims, nsteps]), np.empty([3, nsteps])
            for i in range(nsteps):
                mean[:, i] = np.mean(samples[i, :, :], axis=0)
                std[:, i] = np.std(samples[i, :, :], axis=0)
                chimin[0, i] = np.min(-lnprobs[i, :])
                chimin[1, i] = np.mean(-lnprobs[i, :])
                chimin[2, i] = np.std(-lnprobs[i, :])
            fig, ax = plt.subplots(nrows=n_vert, ncols=n_hor, figsize=(6 * n_vert, 4 * n_hor), sharex=True)
            ax[0, 0].plot(np.arange(nsteps), np.log10(chimin[0]), label=r'$\chi^2_{min}$')
            ax[0, 0].plot(np.arange(nsteps), np.log10(-lnprobs[:, ind+1]), label=r'$\chi^2$ at chain')
            ax[0, 0].plot(np.arange(nsteps), np.log10(chimin[1]), label=r'$\chi^2$ mean')
            ax[0, 0].plot(np.arange(nsteps), np.log10(chimin[2]), label=r'$\chi^2$ disp')
            ax[0, 0].legend(loc=1)
            for i in range(ndims):
                vert, hor = int((i + 1) / n_hor), i + 1 - n_hor * int((i + 1) / n_hor)
                # print(vert, hor)
                ax[vert, hor].plot(np.arange(nsteps), mean[i], color='r')
                ax[vert, hor].fill_between(np.arange(nsteps), mean[i]-std[i], mean[i]+std[i],
                                           facecolor='green', interpolate=True, alpha=0.5)
                ax[vert, hor].plot(np.arange(nsteps), SomeChain[:, i], color='b')
                ax[vert, hor].text(.1, .9, str(pars[i]).replace('_', ' '), ha='left', va='top', transform=ax[vert, hor].transAxes)
                #ax[vert, hor].set_title(pars[i].replace('_', ' '))

        if self.parent.blindMode:
            for ax in fig.axes:
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.show()

    def show_bestfit(self):
        pars, samples, lnprobs = self.readChain()
        ind = np.where(lnprobs == np.nanmax(lnprobs))
        truth = samples[ind[0][0], ind[1][0], :]
        for p, t in zip(pars, truth):
            self.parent.fit.setValue(p, t)

    def fit_disp(self, calc=True, filename=''):

        #self.show_bestfit()
        self.parent.s.prepareFit(-1, all=all)
        self.parent.s.calcFit(recalc=True)
        self.parent.s.calcFitComps(recalc=True)

        opts = {"b_increase": self.b_increase.isChecked(), "H2_excitation": self.H2_excitation.isChecked(),
                "hier_continuum": self.hier_continuum.isChecked(), "telluric": self.telluric.isChecked()}

        if calc:
            fit, fit_disp, fit_comp, fit_comp_disp = [], [], [], []
            for i, s in enumerate(self.parent.s):
                if s.fit.line.norm.n > 0:
                    fit.append(deepcopy(s.fit.line.norm))
                    fit_disp.append([s.fit.line.norm.y])
                    fit_comp_disp.append([])
                    for k, sys in enumerate(self.parent.fit.sys):
                        fit_comp_disp[i].append([s.fit_comp[k].line.norm.y])
                else:
                    fit_disp.append([])
                    fit_comp_disp.append([])


            burnin = int(self.parent.options('MCMC_burnin'))
            pars, samples, lnprobs = self.readChain()
            samples[burnin:, :, :]
            num = int(self.parent.options('MCMC_disp_num'))

            if self.parent.fitType == 'julia':
                self.parent.julia.include("MCMC.jl")
                filename = self.parent.options("filename_saved").replace(".spv", ".spd")
                self.parent.julia.fit_disp([s.fit.line.norm.x if s.fit.line.norm.n > 0 else [] for s in self.parent.s], samples[burnin:, :, :], self.parent.julia_spec, self.parent.fit.list(),
                                           self.parent.julia_add, sys=len(self.parent.fit.sys), tieds=self.parent.fit.tieds, opts=opts,
                                           nthreads=int(self.parent.options('MCMC_threads')), nums=int(self.parent.options('MCMC_disp_num')),
                                           savename=filename)
            else:
                for i1, i2, k in zip(np.random.randint(burnin, high=samples.shape[0], size=num),
                                     np.random.randint(0, high=samples.shape[1], size=num), range(num)):
                    for p, t in zip(pars, samples[i1, i2, :]):
                        self.parent.fit.setValue(p, t)
                    self.parent.s.prepareFit()
                    self.parent.s.calcFit(recalc=True, redraw=False)
                    self.parent.s.calcFitComps(recalc=True)

                    for i, s in enumerate(self.parent.s):
                        if s.fit.line.norm.n > 0:
                            fit_disp[i] = np.r_[fit_disp[i], [s.fit.line.norm.inter(fit[i].x)]]
                            for k, sys in enumerate(self.parent.fit.sys):
                                if s.fit_comp[k].line.n() > 2 and len(fit_comp_disp[i][k]) > 0 and len(fit[i].x) > 0:
                                    fit_comp_disp[i][k] = np.r_[
                                        fit_comp_disp[i][k], [s.fit_comp[k].line.norm.inter(fit[i].x)]]

                for i, s in enumerate(self.parent.s):
                    if s.fit.line.norm.n > 0:
                        fit_disp[i] = np.sort(fit_disp[i], axis=0)
                        self.parent.s[i].fit.disp[0].set(x=fit[i].x, y=fit_disp[i][int((1 - 0.683) / 2 * num), :])
                        self.parent.s[i].fit.disp[1].set(x=fit[i].x, y=fit_disp[i][num - int((1 - 0.683) / 2 * num), :])
                        if self.parent.fit.cont_fit:
                            cheb_disp[i] = np.sort(cheb_disp[i], axis=0)
                            self.parent.s[i].cheb.disp[0].set(x=fit[i].x, y=cheb_disp[i][int((1 - 0.683) / 2 * num), :])
                            self.parent.s[i].cheb.disp[1].set(x=fit[i].x,
                                                              y=cheb_disp[i][num - int((1 - 0.683) / 2 * num), :])

                        for k, sys in enumerate(self.parent.fit.sys):
                            if len(fit_comp_disp[i][k][0]) > 0:
                                fit_comp_disp[i][k] = np.sort(np.asarray(fit_comp_disp[i][k]), axis=0)
                                self.parent.s[i].fit_comp[k].disp[0].set(x=fit[i].x, y=fit_comp_disp[i][k][
                                                                                       int((1 - 0.683) / 2 * num), :])
                                self.parent.s[i].fit_comp[k].disp[1].set(x=fit[i].x, y=fit_comp_disp[i][k][
                                                                                       num - int((1 - 0.683) / 2 * num),
                                                                                       :])
                            else:
                                self.parent.s[i].fit_comp[k].disp[0].set(x=self.parent.s[i].fit.disp[0].norm.x,
                                                                         y=self.parent.s[i].fit.disp[0].norm.y)
                                self.parent.s[i].fit_comp[k].disp[1].set(x=self.parent.s[i].fit.disp[1].norm.x,
                                                                         y=self.parent.s[i].fit.disp[1].norm.y)

        if self.parent.fitType == 'julia':
            x, fit_disp, fit_comp_disp, cheb_disp = self.parent.julia.load_disp(filename)
            for i, s in enumerate(self.parent.s):
                if s.fit.line.norm.n > 0:
                    x[i] = np.asarray(x[i])
                    #if s.fit.line.norm.n > 0:
                    self.parent.s[i].fit.disp[0].set(x=x[i], y=np.asarray(fit_disp[i][:, 0]))
                    self.parent.s[i].fit.disp[1].set(x=x[i], y=np.asarray(fit_disp[i][:, 1]))
                    if self.parent.fit.cont_fit:
                        self.parent.s[i].cheb.disp[0].set(x=x[i], y=np.asarray(cheb_disp[i][:, 0]))
                        self.parent.s[i].cheb.disp[1].set(x=x[i], y=np.asarray(cheb_disp[i][:, 1]))
                    for k, sys in enumerate(self.parent.fit.sys):
                        if len(fit_comp_disp[i][k]) > 0:
                            self.parent.s[i].fit_comp[k].disp[0].set(x=x[i], y=np.asarray(fit_comp_disp[i][k][:, 0]))
                            self.parent.s[i].fit_comp[k].disp[1].set(x=x[i], y=np.asarray(fit_comp_disp[i][k][:, 1]))
                        else:
                            self.parent.s[i].fit_comp[k].disp[0].set(x=self.parent.s[i].fit.disp[0].norm.x, y=self.parent.s[i].fit.disp[0].norm.y)
                            self.parent.s[i].fit_comp[k].disp[1].set(x=self.parent.s[i].fit.disp[1].norm.x, y=self.parent.s[i].fit.disp[1].norm.y)

        print("disp done")

    def set_fit_disp_num(self):
        self.parent.options('MCMC_disp_num', int(self.fit_disp_num.text()))

    def loadres(self):
        fname = QFileDialog.getOpenFileName(self, 'Load MCMC results', self.parent.work_folder)

        if fname[0]:
            if fname[0].endswith('.spr'):
                self.loadJulia(fname[0])
            else:
                self.parent.options('work_folder', os.path.dirname(fname[0]))
                self.parent.MCMC_output = fname[0]


    def load_disp(self, filename):
        self.parent.julia.include("MCMC.jl")
        self.fit_disp(calc=False, filename=filename)

    def loaddisp(self):
        fname = QFileDialog.getOpenFileName(self, 'Load dispersion of the fit model', self.parent.options("filename_saved").replace(".spv", ".spd"))[0]
        #fname = "output/disp.spd"
        if fname:
            if fname.endswith('.spd'):
                self.load_disp(fname)
            else:
                raise Exception("You can load disp only from a file with .spd extension")
                self.parent.options('work_folder', os.path.dirname(fname))
                self.parent.MCMC_output = fname


    def export(self):
        fname = QFileDialog.getSaveFileName(self, 'Export MCMC results', self.parent.work_folder)

        if fname[0]:
            if fname[0].endswith('.dat'):
                pars, samples, lnprobs = self.readChain()
                nsteps, nwalkers, = lnprobs.shape
                burnin = int(self.parent.options('MCMC_burnin'))

                mask = np.array([p.show for p in self.parent.fit.list_fit()])
                #print(mask)
                names = [str(p) for p in self.parent.fit.list_fit() if p.show]
                print(names)

                s = samples[burnin:, :, mask]
                #print(s.shape)
                s = s.reshape(-1, s.shape[-1])
                #print(s.shape)
                #np.save(fname[0], s)
                np.savetxt(fname[0], s, header=','.join(names), fmt=','.join(['%s'] * s.shape[1]))

    def keyPressEvent(self, event):
        super(fitMCMCWidget, self).keyPressEvent(event)
        key = event.key()
        if not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_F5:
                #if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier):
                self.parent.MCMC.close()

    def closeEvent(self, event):
        #for opt, func in self.opts.items():
        #    print(opt, func(getattr(self, opt)))
        #    self.parent.options(opt, func(getattr(self, opt)))
        self.parent.MCMC = None

class fitExtWidget(QWidget):
    def __init__(self, parent):
        super(fitExtWidget, self).__init__()
        self.parent = parent
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        l = QHBoxLayout()
        l.addWidget(QLabel('Template:'))
        temp_group = QButtonGroup(self)
        self.smooth_template = {'SDSS': 3, 'VandenBerk': 7, 'HST': 7, 'Selsing': 7, 'power': None, 'composite': 7}
        for template in ['VandenBerk', 'HST', 'Selsing', 'composite', 'power']:
            setattr(self, template, QRadioButton(template))
            temp_group.addButton(getattr(self, template))
            l.addWidget(getattr(self, template))
            getattr(self, template).toggled.connect(partial(self.set, 'template', template))
        self.Selsing.setChecked(True)
        self.slope = -1.7
        self.slopeField = QLineEdit()
        self.slopeField.setFixedSize(70, 30)
        self.slopeField.setText('{0:.2f}'.format(self.slope))
        l.addWidget(self.slopeField)
        l.addStretch(1)
        layout.addLayout(l)

        l = QGridLayout()

        self.z_em = QCheckBox('z_em:', self)
        self.z_em.setChecked(False)
        l.addWidget(self.z_em, 0, 0)
        self.z_em_value = QLineEdit(self)
        self.z_em_value.setText('2.87')
        self.z_em_value.returnPressed.connect(self.showExt)
        l.addWidget(self.z_em_value, 0, 1)

        self.z_abs = QCheckBox('z_abs:', self)
        self.z_abs.setChecked(False)
        l.addWidget(self.z_abs, 1, 0)
        self.z_abs_value = QLineEdit(self)
        self.z_abs_value.setText(str(self.parent.z_abs))
        self.z_abs_value.returnPressed.connect(self.showExt)
        l.addWidget(self.z_abs_value, 1, 1)

        self.Av = QCheckBox('Av:', self)
        self.Av.setChecked(False)
        l.addWidget(self.Av, 2, 0)
        self.Av_value = QLineEdit(self)
        self.Av_value.setText('0.0')
        self.Av_value.returnPressed.connect(self.showExt)
        l.addWidget(self.Av_value, 2, 1)

        layout.addLayout(l)
        self.tab = QTabWidget()
        self.tab.setGeometry(0, 0, 1050, 900)
        # self.tab.setMinimumSize(1050, 300)

        for t in ['Emperical', 'Analytical']:
            self.tab.addTab(self.initTabGUI(t), t)

        layout.addWidget(self.tab)

        l = QHBoxLayout()
        select = QPushButton('Auto select', self)
        select.setFixedSize(100, 30)
        select.clicked.connect(self.autoSelect)
        l.addWidget(select)

        l.addWidget(QLabel('window:'))
        self.smooth_window = QLineEdit(self)
        self.smooth_window.setText('100')
        self.smooth_window.setFixedSize(60, 30)
        l.addWidget(self.smooth_window)
        l.addStretch(1)

        layout.addLayout(l)

        l = QHBoxLayout()
        show = QPushButton('Show', self)
        show.setFixedSize(100, 30)
        show.clicked.connect(self.showExt)
        l.addWidget(show)

        fit = QPushButton('Fit', self)
        fit.setFixedSize(100, 30)
        fit.clicked.connect(self.fitExt)
        l.addWidget(fit)
        l.addStretch(1)

        layout.addLayout(l)

        self.setLayout(layout)
        self.setGeometry(300, 300, 280, 330)
        self.setWindowTitle('fit Extinction curve')
        self.show()

    def set(self, name, value):
        setattr(self, name, value)

    def initTabGUI(self, window=None):

        frame = QFrame()
        validator = QDoubleValidator()
        locale = QLocale('C')
        validator.setLocale(locale)

        layout = QHBoxLayout()
        self.grid = QGridLayout()

        if window == 'Emperical':
            l = QHBoxLayout()
            type_group = QButtonGroup(self)
            for ec in ['MW', 'SMC', 'LMC']:
                setattr(self, ec, QRadioButton(ec))
                type_group.addButton(getattr(self, ec))
                l.addWidget(getattr(self, ec))
                getattr(self, ec).toggled.connect(partial(self.set, 'ec', ec))
            self.SMC.setChecked(True)
            l.addStretch(1)
            layout.addLayout(l)

        if window == 'Analytical':

            l = QHBoxLayout()
            self.Av_bump = QCheckBox('Av_bump:', self)
            self.Av_bump.setChecked(False)
            l.addWidget(self.Av_bump)
            self.Av_bump_value = QLineEdit(self)
            self.Av_bump_value.setText('0.0')
            l.addWidget(self.Av_bump_value)
            l.addStretch(1)
            layout.addLayout(l)

        layout.addLayout(self.grid)
        layout.addStretch()
        frame.setLayout(layout)
        return frame

    def load_template(self, x=None, z_em=0, smooth_window=None):
        if self.template in ['SDSS', 'VandenBerk', 'HST', 'Selsing', 'power']:
            fill_value = 'extrapolate'
            if self.template == 'SDSS':
                data = np.genfromtxt(self.parent.folder + "/data/SDSS/medianQSO.dat", skip_header=2, unpack=True)
                data = data[:, np.logical_or(data[1] != 0, data[2] != 0)]
                fill_value = (1.3, 0.5)
            if self.template == 'VandenBerk':
                data = np.genfromtxt(self.parent.folder + "/data/SDSS/QSO_composite.dat", unpack=True)
                data = data[:, np.logical_or(data[1] != 0, data[2] != 0)]
            elif self.template == 'HST':
                data = np.genfromtxt(self.parent.folder + "/data/SDSS/hst_composite.dat", skip_header=2, unpack=True)
            elif self.template == 'Selsing':
                data = np.genfromtxt(self.parent.folder + "/data/SDSS/Selsing2016.dat", skip_header=0, unpack=True)
            elif self.template == 'power':
                self.slope = float(self.slopeField.text())
                data = np.ones((2, 10000))
                data[0] = np.linspace(500, 25000, data.shape[1])
                data[1] = np.power(data[0] / 2500, self.slope)
            elif self.template == 'composite':
                data = np.genfromtxt(self.parent.folder + "/data/SDSS/Selsing2016.dat", skip_header=0, unpack=True)
                data = data[:, np.logical_or(data[1] != 0, data[2] != 0)]
                if 0:
                    x = data[0][-1] + np.arange(1, int((25000 - data[0][-1]) / 0.4)) * 0.4
                    y = np.power(x / 2500, -1.9) * 6.542031
                    data = np.append(data, [x, y, y / 10], axis=1)
                else:
                    d2 = np.genfromtxt(self.parent.folder + "/data/SDSS/QSO1_template_norm.sed", skip_header=0, unpack=True)
                    m = d2[0] > data[0][-1]
                    data = np.append(data, [d2[0][m], d2[1][m] * data[1][-1] / d2[1][m][0], d2[1][m] / 30], axis=1)

            data[0] *= (1 + z_em)
            if smooth_window is not None:
                data[1] = smooth(data[1], window_len=smooth_window, window='hanning', mode='same')

            inter = interp1d(data[0], data[1], bounds_error=False, fill_value=fill_value, assume_sorted=True)

        if x is None:
            s = self.parent.s[self.parent.s.ind]
            x = s.spec.raw.x

        return inter(x)

    def autoSelect(self):
        s = self.parent.s[self.parent.s.ind]
        z_em = float(self.z_em_value.text())
        mask = s.spec.x() > 1280 * (1 + z_em)
        if 0:
            y_0 = s.spec.y()[:]
            for factor in range(6):
                y = np.convolve(y_0, np.ones(1 + 2*2**factor) / (1 + 2*2**factor),  mode='same')
                print(y)
                mask *= np.abs((y_0 - y) / s.spec.err()) < 1.5
                print(np.sum(mask))
                y_0 = y[:]
        else:
            window = int(float(self.smooth_window.text()) / (s.spec.x()[-1] - s.spec.x()[0]) * s.spec.n())
            s.calcCont(method='Smooth', xl=s.spec.x()[0]-1, xr=s.spec.x()[-1]+1, iter=3, clip=3, filter='hanning', window=window)
            mask *= np.abs((s.spec.y() - s.cont.y) / s.spec.err()) < 2

        if 1:
            # remove prominent emission lines regions
            windows = [[1295, 1320], [1330, 1360], [1375, 1430], [1500, 1600], [1625, 1700], [1740, 1760],
                       [1840, 1960], [2050, 2120], [2250, 2650], [2710, 2890], [2940, 2990], [3280, 3330],
                       [3820, 3920], [4200, 4680], [4780, 5080], [5130, 5400], [5500, 5620], [5780, 6020],
                       [6300, 6850], [7600, 8050], [8250, 8300], [8400, 8600], [9000, 9400], [9500, 9700],
                       [9950, 10200]]
            for w in windows:
                mask *= (s.spec.x() < w[0] * (1 + z_em)) + (s.spec.x() > w[1] * (1 + z_em))

        else:
            windows = [[1318, 1325], [1348, 1360], [1446, 1494], [1682, 1696], [1765, 1771],
                       [1875, 1884], [2008, 2036], [2124, 2153], [2238, 2257], [2448, 2458], [2482, 2595],
                       [3021, 3100], [3224, 3248], [3297, 3329], [3356, 3394], [3537, 3554], [3613, 3714], [3832, 3850],
                       [3898, 3950], [3978, 4050], [4202, 4227], [4260, 4285], [4412, 4469]]
            m = np.zeros_like(s.spec.x(), dtype=bool)
            for w in windows:
                m += (s.spec.x() > w[0] * (1 + z_em)) * (s.spec.x() < w[1] * (1 + z_em))
            mask *= m

        # remove atmospheric absorption region
        windows = [[5560, 5600], [6865, 6930], [7580, 7690], [9300, 9600], [10150, 10400], [13200, 14600]]
        for w in windows:
            mask *= (s.spec.x() < w[0]) + (s.spec.x() > w[1])

        s.mask.set(mask)
        s.set_fit_mask()
        s.redraw()

    def showExt(self, signal, norm=None):

        print(self.tab.tabText(self.tab.currentIndex()), self.ec)
        print(self.template)
        z_em = float(self.z_em_value.text())
        z_abs = float(self.z_abs_value.text())
        Av = float(self.Av_value.text())
        Av_bump = float(self.Av_bump_value.text())

        y = self.load_template(z_em=z_em, smooth_window=self.smooth_template[self.template])

        s = self.parent.s[self.parent.s.ind]
        if self.tab.tabText(self.tab.currentIndex()) == 'Emperical':
            y *= add_ext(x=s.spec.raw.x, z_ext=z_abs, Av=Av, kind=self.ec)
        elif self.tab.tabText(self.tab.currentIndex()) == 'Analytical':
            if Av > 0 or Av_bump > 0:
                y *= add_ext_bump(x=s.spec.raw.x, z_ext=z_abs, Av=Av, Av_bump=Av_bump)

        if norm is None:
            norm = np.sum(s.spec.raw.y[s.mask.x()]) / np.sum(y[s.mask.x()])

        if ~np.isnan(norm):
            y *= norm
            s.cont.x, s.cont.y = s.spec.raw.x[:], y
            s.cont.n = len(s.cont.y)
            s.cont_mask = np.logical_not(np.isnan(s.spec.raw.x))
            s.redraw()

    def fitExt(self):
        s = self.parent.s[self.parent.s.ind]
        z_em = float(self.z_em_value.text())
        z_abs = float(self.z_abs_value.text())
        Av = float(self.Av_value.text())
        Av_bump = float(self.Av_bump_value.text())
        temp = self.load_template(z_em=z_em, smooth_window=self.smooth_template[self.template])

        def fcn2min(params):
            y = temp * add_ext(x=s.spec.raw.x, z_ext=z_abs, Av=params.valuesdict()['Av'], kind=self.ec) * params.valuesdict()['norm']
            return (y[s.fit_mask.x()] - s.spec.raw.y[s.fit_mask.x()]) / s.spec.raw.err[s.fit_mask.x()]

        norm = np.sum(s.spec.raw.y[s.mask.x()]) / np.sum(temp[s.mask.x()])
        print(norm)
        # create a set of Parameters
        params = Parameters()
        params.add('Av', value=Av, min=-5, max=5)
        params.add('norm', value=norm, min=0, max=1e10)

        # do fit, here with leastsq model
        minner = Minimizer(fcn2min, params, nan_policy='propagate', calc_covar=True)
        result = minner.minimize()
        report_fit(result)

        self.Av_value.setText("{0:6.3f}".format(result.params['Av'].value))
        self.showExt(True, norm=result.params['norm'].value)

    def keyPressEvent(self, qKeyEvent):
        if qKeyEvent.key() == Qt.Key.Key_Return:
            self.showExt(True)

class extract2dWidget(QWidget):
    def __init__(self, parent):
        super(extract2dWidget, self).__init__()
        self.parent = parent
        self.mask_type = 'moffat'
        self.trace_pos = None
        self.trace_width = [None, None]
        self.init_GUI()
        self.init_Parameters()
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        self.setGeometry(200, 200, 550, 800)
        self.setWindowTitle('Spectrum Extraction')

    def init_Parameters(self):
        self.opts = OrderedDict([
            ('trace_step', ['traceStep', int, 200]),
            ('exp_pixel', ['expPixel', int, 1]),
            ('exp_factor', ['expFactor', float, 3]),
            ('extr_height', ['extrHeight', float, 1]),
            ('extr_width', ['extrWidth', float, 3]),
            ('extr_slit', ['extrSlit', float, 0.9]),
            ('extr_window', ['extrWindow', int, 0]),
            ('extr_border', ['extrBorder', int, 1]),
            ('extr_conf', ['extrConf', float, 0.03]),
            ('sky_poly', ['skyPoly', int, 3]),
            ('sky_smooth', ['skySmooth', int, 0]),
            ('sky_smooth_coef', ['skySmoothCoef', float, 0.3]),
            ('bary_corr', ['baryCorr', float, -20.894]),
            ('rescale_window', ['rescaleWindow', int, 30]),
        ])
        for opt in self.opts.keys():
            setattr(self, opt, self.opts[opt][1](self.opts[opt][2]))
            getattr(self, self.opts[opt][0]).setText(str(getattr(self, opt)))

    def init_GUI(self):
        layout = QVBoxLayout()

        self.tab = QTabWidget()
        self.tab.setGeometry(0, 0, 550, 550)
        self.tab.setMinimumSize(550, 550)
        self.tab.setCurrentIndex(0)
        self.init_GUI_CosmicRays()
        self.init_GUI_Sky()
        self.init_GUI_Extraction()
        self.init_GUI_Correction()
        layout.addWidget(self.tab)
        hl = QHBoxLayout()
        exposure = QPushButton('Exposure:')
        exposure.setFixedSize(100, 30)
        exposure.clicked.connect(self.changeExp)
        self.expchoose = QComboBox()
        self.expchoose.setFixedSize(400, 30)
        for s in self.parent.s:
            self.expchoose.addItem(s.filename)
        if len(self.parent.s) > 0:
            self.exp_ind = self.parent.s.ind
            self.expchoose.activated.connect(self.onExpChoose)
            self.expchoose.setCurrentIndex(self.exp_ind)
            self.onExpChoose(self.exp_ind)
        hl.addWidget(exposure)
        hl.addWidget(self.expchoose)
        hl.addStretch(0)
        layout.addLayout(hl)
        layout.addStretch(1)
        self.setLayout(layout)

    def init_GUI_CosmicRays(self):

        frame = QFrame(self)
        layout = QVBoxLayout()
        self.input = QTextEdit()
        self.input.setText('sigclip=2.0\nsigfrac=0.3\nobjlim=1.\npssl=0.0\ngain=10.0\nreadnoise=1.5\nsatlevel=65535.0\nniter=10\nsepmed=1\ncleantype=meanmask\nfsmode=median\npsffwhm=2.5\npsfsize=7\npsfk=None\npsfbeta=4.765')
        layout.addWidget(QLabel('cosmicray_lacosmic arguments:'))
        layout.addWidget(self.input)
        hl = QHBoxLayout()
        run = QPushButton('Run')
        run.setFixedSize(80, 30)
        run.clicked.connect(partial(self.cr, update='new'))
        add = QPushButton('Add')
        add.setFixedSize(80, 30)
        add.clicked.connect(partial(self.cr, update='add'))
        raw = QPushButton('From raw')
        raw.setFixedSize(80, 30)
        raw.clicked.connect(partial(self.cr, update='raw'))
        clear = QPushButton('Clear')
        clear.setFixedSize(100, 30)
        clear.clicked.connect(self.clear)
        hl.addWidget(run)
        hl.addWidget(add)
        hl.addWidget(raw)
        hl.addStretch(0)
        hl.addWidget(clear)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        fromexp = QPushButton('From exposure:')
        fromexp.setFixedSize(120, 30)
        fromexp.clicked.connect(partial(self.crfromexp))
        hl.addWidget(fromexp)
        self.expCrChoose = QComboBox()
        self.expCrChoose.setFixedSize(250, 30)
        for s in self.parent.s:
            self.expCrChoose.addItem(s.filename)
        if len(self.parent.s) > 0:
            self.exp_cr_ind = self.parent.s.ind
            self.expCrChoose.setCurrentIndex(self.exp_cr_ind)
        self.expCrChoose.currentIndexChanged.connect(partial(self.onExpChoose, name='exp_cr_ind'))
        hl.addWidget(fromexp)
        hl.addWidget(self.expCrChoose)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        expand = QPushButton('Expand:')
        expand.setFixedSize(120, 30)
        expand.clicked.connect(self.expand)
        self.expPixel = QLineEdit()
        self.expPixel.setFixedSize(40, 30)
        self.expPixel.textChanged.connect(partial(self.edited, 'exp_pixel'))
        self.expFactor = QLineEdit()
        self.expFactor.setFixedSize(40, 30)
        self.expFactor.textChanged.connect(partial(self.edited, 'exp_factor'))
        intelExpand = QPushButton('Intel. expand')
        intelExpand.setFixedSize(120, 30)
        intelExpand.clicked.connect(self.intelExpand)
        hl.addWidget(expand)
        hl.addWidget(self.expPixel)
        hl.addStretch(0)
        hl.addWidget(self.expFactor)
        hl.addWidget(intelExpand)
        layout.addLayout(hl)
        hl = QHBoxLayout()
        clean = QPushButton('Clean')
        clean.setFixedSize(120, 30)
        clean.clicked.connect(self.clean)
        hl.addWidget(clean)
        hl.addStretch(0)
        layout.addLayout(hl)
        hl = QHBoxLayout()
        extrapolate = QPushButton('Extrapolate')
        extrapolate.setFixedSize(120, 30)
        extrapolate.clicked.connect(partial(self.extrapolate, inplace=False))
        self.extrHeight = QLineEdit()
        self.extrHeight.setFixedSize(40, 30)
        self.extrHeight.textChanged.connect(partial(self.edited, 'extr_height'))
        self.extrWidth = QLineEdit()
        self.extrWidth.setFixedSize(40, 30)
        self.extrWidth.textChanged.connect(partial(self.edited, 'extr_width'))
        hl.addWidget(extrapolate)
        hl.addWidget(QLabel('h:'))
        hl.addWidget(self.extrHeight)
        hl.addWidget(QLabel('w:'))
        hl.addWidget(self.extrWidth)
        hl.addStretch(0)
        layout.addLayout(hl)
        frame.setLayout(layout)
        self.tab.addTab(frame, 'Cosmic Rays')

    def init_GUI_Extraction(self):

        frame = QFrame(self)
        layout = QVBoxLayout()

        hl = QHBoxLayout()
        trace = QPushButton('Trace each:')
        trace.setFixedSize(120, 30)
        trace.clicked.connect(partial(self.trace))
        self.traceStep = QLineEdit()
        self.traceStep.setFixedSize(50, 30)
        self.traceStep.textChanged.connect(partial(self.edited, 'trace_step'))
        hl.addWidget(trace)
        hl.addWidget(self.traceStep)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        traceFit = QPushButton('Fit trace')
        traceFit.setFixedSize(120, 30)
        traceFit.clicked.connect(partial(self.trace_fit))
        traceStat = QPushButton('Trace stats')
        traceStat.setFixedSize(120, 30)
        traceStat.clicked.connect(partial(self.trace_stat))
        hl.addWidget(traceFit)
        hl.addWidget(traceStat)
        hl.addStretch(0)
        layout.addLayout(hl)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("QFrame {border: 0.5px solid rgb(100,100,100);}")
        layout.addWidget(line)

        hl = QHBoxLayout()
        hl.addWidget(QLabel('Profile:'))
        self.extrProfile = QComboBox(self)
        self.extrProfile.addItems(['optimal', 'moffat', 'gaussian', 'rectangular'])
        self.extrProfile.setFixedSize(80, 30)
        #self.extrProfile.setCurrentText(self.extr_prof)
        hl.addWidget(self.extrProfile)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        self.slit = QCheckBox('Slit:')
        self.slit.setFixedSize(100, 30)
        self.slit.setChecked(True)
        hl.addWidget(self.slit)
        self.extrSlit = QLineEdit()
        self.extrSlit.setFixedSize(80, 30)
        self.extrSlit.textChanged.connect(partial(self.edited, 'extr_slit'))
        hl.addWidget(self.extrSlit)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        self.bary = QCheckBox('Bary. corr:')
        self.bary.setFixedSize(100, 30)
        self.bary.setChecked(True)
        hl.addWidget(self.bary)
        self.baryCorr = QLineEdit()
        self.baryCorr.setFixedSize(80, 30)
        self.baryCorr.textChanged.connect(partial(self.edited, 'bary_corr'))
        hl.addWidget(self.baryCorr)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        self.airVac = QCheckBox('Airvac. corr.')
        self.airVac.setFixedSize(100, 30)
        self.airVac.setChecked(True)
        hl.addWidget(self.airVac)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        self.removeCR = QCheckBox('Remove cosmics')
        self.removeCR.setFixedSize(100, 30)
        self.removeCR.setChecked(False)
        hl.addWidget(self.removeCR)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        extract = QPushButton('Extract')
        extract.setFixedSize(120, 30)
        extract.clicked.connect(partial(self.extract))
        hl.addWidget(extract)
        hl.addStretch(0)
        layout.addLayout(hl)
        layout.addStretch(0)
        frame.setLayout(layout)
        self.tab.addTab(frame, 'Extract')

    def init_GUI_Sky(self):

        frame = QFrame(self)
        layout = QVBoxLayout()

        hl = QHBoxLayout()
        hl.addWidget(QLabel('Window:'))
        self.extrWindow = QLineEdit()
        self.extrWindow.setFixedSize(40, 30)
        self.extrWindow.textChanged.connect(partial(self.edited, 'extr_window'))
        hl.addWidget(self.extrWindow)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel('Border indent:'))
        self.extrBorder = QLineEdit()
        self.extrBorder.setFixedSize(60, 30)
        self.extrBorder.textChanged.connect(partial(self.edited, 'extr_border'))
        hl.addWidget(self.extrBorder)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel('Profile confidence:'))
        self.extrConf = QLineEdit()
        self.extrConf.setFixedSize(60, 30)
        self.extrConf.textChanged.connect(partial(self.edited, 'extr_conf'))
        hl.addWidget(self.extrConf)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel('Model:'))
        self.skymodel = QComboBox()
        self.skymodel.setFixedSize(80, 30)
        self.skymodel.addItems(['median', 'polynomial', 'robust', 'wavy'])
        self.skymodeltype = 'wavy'
        self.skymodel.setCurrentText(self.skymodeltype)
        self.skymodel.activated.connect(self.skyModel)
        hl.addWidget(self.skymodel)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel('Poly order:'))
        self.skyPoly = QLineEdit()
        self.skyPoly.setFixedSize(60, 30)
        self.skyPoly.textChanged.connect(partial(self.edited, 'sky_poly'))
        hl.addWidget(self.skyPoly)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel('Smooth:'))
        self.skySmooth = QLineEdit()
        self.skySmooth.setFixedSize(60, 30)
        self.skySmooth.textChanged.connect(partial(self.edited, 'sky_smooth'))
        hl.addWidget(self.skySmooth)
        hl.addWidget(QLabel('reject at:'))
        self.skySmoothCoef = QLineEdit()
        self.skySmoothCoef.setFixedSize(60, 30)
        self.skySmoothCoef.textChanged.connect(partial(self.edited, 'sky_smooth_coef'))
        hl.addWidget(self.skySmoothCoef)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        calcsky = QPushButton('Calc Sky')
        calcsky.setFixedSize(120, 30)
        calcsky.clicked.connect(partial(self.sky))
        calcsky_simple = QPushButton('Calc Sky simple')
        calcsky_simple.setFixedSize(120, 30)
        calcsky_simple.clicked.connect(partial(self.sky_simple))
        hl.addWidget(calcsky)
        hl.addWidget(calcsky_simple)
        hl.addStretch(0)
        layout.addLayout(hl)
        layout.addStretch(0)
        frame.setLayout(layout)
        self.tab.addTab(frame, 'Sky model')

    def init_GUI_Correction(self):

        frame = QFrame(self)
        layout = QVBoxLayout()
        hl = QHBoxLayout()
        hl.addWidget(QLabel('Window:'))
        self.rescaleWindow = QLineEdit()
        self.rescaleWindow.setFixedSize(50, 30)
        self.rescaleWindow.textChanged.connect(partial(self.edited, 'rescale_window'))
        hl.addWidget(self.rescaleWindow)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        dispersion = QPushButton('Calc dispersion')
        dispersion.setFixedSize(140, 30)
        dispersion.clicked.connect(partial(self.dispersion))
        hl.addWidget(dispersion)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        rescale = QPushButton('Rescale from:')
        rescale.setFixedSize(120, 30)
        rescale.clicked.connect(partial(self.rescale))
        hl.addWidget(rescale)
        self.expResChoose = QComboBox()
        self.expResChoose.setFixedSize(250, 30)
        for s in self.parent.s:
            self.expResChoose.addItem(s.filename)
        if len(self.parent.s) > 0:
            self.exp_res_ind = self.parent.s.ind
            self.expResChoose.setCurrentIndex(self.exp_res_ind)
        self.expCrChoose.currentIndexChanged.connect(partial(self.onExpChoose, name='exp_res_ind'))
        hl.addWidget(rescale)
        hl.addWidget(self.expResChoose)
        hl.addStretch(0)
        layout.addLayout(hl)

        layout.addStretch(0)
        frame.setLayout(layout)
        self.tab.addTab(frame, 'Corrections')

    def edited(self, attr):
        try:
            setattr(self, attr, self.opts[attr][1](getattr(self, self.opts[attr][0]).text()))
        except:
            pass

    def skyModel(self):
        self.skymodeltype = self.skymodel.currentText()
        print(self.skymodeltype)

    def changeExp(self):
        self.exp_ind += 1
        if self.exp_ind >= len(self.parent.s):
            self.exp_ind = 0
        self.expchoose.setCurrentIndex(self.exp_ind)

    def onExpChoose(self, index, name='exp_ind'):
        setattr(self, name, index)
        if self.expchoose.currentText().endswith('.fits'):
            try:
                hdulist = fits.open(self.expchoose.currentText())
                for hdu in hdulist:
                    for attr in ['BARY', 'HIERARCH ESO QC VRAD BARYCOR']:
                        if attr in hdu.header:
                            self.baryCorr.setText("{:7.3f}".format(hdu.header[attr]).strip())
                            print('Set barycentric velocity:', hdu.header[attr])

            except:
                pass

    def updateExpChoose(self):
        self.expchoose.clear()
        for s in self.parent.s:
            self.expchoose.addItem(s.filename)
        self.expchoose.setCurrentIndex(self.exp_ind)
        self.expResChoose.clear()
        for s in self.parent.s:
            self.expResChoose.addItem(s.filename)
        self.expResChoose.setCurrentIndex(self.exp_res_ind)
        self.expCrChoose.clear()
        for s in self.parent.s:
            self.expCrChoose.addItem(s.filename)
        self.expCrChoose.setCurrentIndex(self.exp_cr_ind)

    def cr(self, update='new'):

        if update in ['new', 'add']:
            kwargs = {}
            for line in self.input.toPlainText().splitlines():
                if line.split('=')[1].replace('-', '', 1).replace('.', '', 1).strip().isdigit():
                    kwargs[line.split('=')[0]] = float(line.split('=')[1])
                else:
                    kwargs[line.split('=')[0]] = line.split('=')[1]

            self.parent.s[self.exp_ind].spec2d.cr_remove(update, **kwargs)

        elif update == 'raw':
            s = self.parent.s[self.exp_ind].spec2d
            if s.cr is None:
                s.cr = image(x=s.raw.x, y=s.raw.y, mask=np.zeros_like(s.raw.z))
            if s.raw.mask is not None:
                s.cr.mask = np.logical_or(s.cr.mask, s.raw.mask)

        self.parent.s.redraw()

    def clear(self):
        self.parent.s[self.exp_ind].spec2d.cr.mask = np.zeros_like(self.parent.s[self.exp_ind].spec2d.raw.z)
        self.parent.s.redraw()

    def crfromexp(self):
        self.parent.s[self.exp_ind].spec2d.cr.mask = np.copy(self.parent.s[self.exp_cr_ind].spec2d.cr.mask)
        self.parent.s.redraw()

    def expand(self):
        self.parent.s[self.exp_ind].spec2d.expand_mask(self.exp_pixel)
        self.parent.s.redraw()

    def intelExpand(self):
        self.parent.s[self.exp_ind].spec2d.intelExpand(self.exp_factor, self.exp_pixel)
        self.parent.s.redraw()

    def clean(self):
        self.parent.s[self.exp_ind].spec2d.clean()
        self.parent.s.redraw()
        self.updateExpChoose()

    def extrapolate(self, inplace=False):
        self.parent.s[self.exp_ind].spec2d.extrapolate(kind='new', extr_width=self.extr_width, extr_height=self.extr_height, sky=1)
        self.parent.s.redraw()
        self.updateExpChoose()

    def trace(self):
        s = self.parent.s[self.exp_ind]
        inds = np.where(s.cont_mask2d)[0]
        print(inds)
        s.spec2d.moffat_grid(2.35482 * self.extr_slit / 2 / np.sqrt(2 ** (1 / 4.765) - 1))
        for k, i in zip(np.arange(len(inds))[:-self.trace_step:self.trace_step], inds[:-self.trace_step:self.trace_step]):
            try:
                print(k)
                s.spec2d.profile(s.spec2d.raw.x[i+int(self.trace_step * 0.7)-4], s.spec2d.raw.x[i+int(self.trace_step * 0.7)+4],
                                 s.spec2d.raw.y[self.extr_border], s.spec2d.raw.y[-self.extr_border],
                                 x_0=s.cont2d.y[k], slit=self.extr_slit)
            except:
                pass

        self.parent.s[self.parent.s.ind].redraw()

    def trace_fit(self):
        self.parent.s[self.parent.s.ind].spec2d.fit_trace()
        self.parent.s.redraw()

    def trace_stat(self):
        trace = self.parent.s[self.parent.s.ind].spec2d.trace
        if trace is not None:
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(trace[0], trace[1])
            ax[1].plot(trace[0], trace[2])
            plt.show()

    def sky(self):

        s = self.parent.s[self.exp_ind]
        s.spec2d.sky_model(s.spec2d.raw.x[0], s.spec2d.raw.x[-1], border=self.extr_border, slit=self.extr_slit,
                           model=self.skymodeltype, window=self.extr_window, poly=self.sky_poly, conf=self.extr_conf,
                           smooth=self.sky_smooth, smooth_coef=self.sky_smooth_coef)

        self.parent.s.redraw()

    def sky_simple(self):
        s = self.parent.s[self.exp_ind]
        s.spec2d.sky_model_simple(s.spec2d.raw.x[0], s.spec2d.raw.x[-1], border=self.extr_border, conf=self.extr_conf)
        self.parent.s.redraw()

    def extract(self):
        self.bary_corr = float(self.baryCorr.text()) if self.bary.isChecked() else None
        s = self.parent.s[self.exp_ind]
        s.spec2d.extract(s.spec2d.raw.x[0], s.spec2d.raw.x[-1], slit=self.slit.isChecked() * self.extr_slit,
                         profile_type=self.extrProfile.currentText(), airvac=self.airVac.isChecked(), bary=self.bary_corr,
                         removecr=self.removeCR.isChecked(), extr_width=self.extr_width, extr_height=self.extr_height)

        self.updateExpChoose()
        self.parent.s.redraw(len(self.parent.s)-1)

    def dispersion(self):
        s = self.parent.s[self.exp_ind]
        x = s.spec.x()[s.cont_mask]
        y = s.spec.y()[s.cont_mask]
        err = s.spec.err()[s.cont_mask]

        std = []
        ref = np.arange(np.sum(s.cont_mask))
        for k, i in enumerate(np.where(s.cont_mask)[0]):
            mask = np.logical_and(ref > i-self.rescale_window / 2, ref < i+self.rescale_window / 2)
            std.append(np.std(y[mask] - s.cont.y[mask]) / np.mean(err[mask]))

        self.parent.s.append(Spectrum(self.parent, 'error_dispersion', data=[s.cont.x, np.asarray(std)]))
        print(len(s.cont.x), len(std))
        self.updateExpChoose()
        self.parent.s.redraw(len(self.parent.s)-1)

    def rescale(self):
        self.exp_res_ind = self.expResChoose.currentIndex()
        s = self.parent.s[self.exp_res_ind]
        inter = interp1d(s.cont.x, s.cont.y, fill_value='extrapolate')
        s = self.parent.s[self.exp_ind]
        s.spec.raw.err *= inter(s.spec.raw.x)
        self.parent.s.redraw(self.exp_ind)

    def keyPressEvent(self, event):
        super(extract2dWidget, self).keyPressEvent(event)
        key = event.key()

        if not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_D:
                if (QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier):
                    self.parent.extract2dwindow.close()

    def closeEvent(self, event):
        self.parent.extract2dwindow = None
        event.accept()

class fitContWidget(QWidget):
    def __init__(self, parent):
        super(fitContWidget, self).__init__()
        self.parent = parent
        self.init_GUI()
        self.init_Parameters()
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        self.setGeometry(200, 200, 550, 500)
        self.setWindowTitle('Continuum construction')

    def init_Parameters(self):
        self.opts = OrderedDict([
            ('cont_iter', ['contIter', int, 7]),
            ('cont_smooth', ['contSmooth', int, 501]),
            ('cont_clip', ['contClip', float, 2.0]),
            ('x_min', ['xmin', float, 3500]),
            ('x_max', ['xmax', float, 4500]),
            ('sg_order', ['sgOrder', int, 5])
        ])
        for opt in self.opts.keys():
            setattr(self, opt, self.opts[opt][1](self.opts[opt][2]))
            getattr(self, self.opts[opt][0]).setText(str(getattr(self, opt)))

    def init_GUI(self):
        layout = QVBoxLayout()

        self.tab = QTabWidget()
        self.tab.setGeometry(0, 0, 550, 200)
        self.tab.setMinimumSize(550, 200)
        self.tab.setCurrentIndex(0)
        self.init_GUI_Bsplain()
        self.init_GUI_SG()
        self.init_GUI_Smooth()
        self.init_GUI_Cheb()
        layout.addWidget(self.tab)

        hl = QHBoxLayout()
        hl.addWidget(QLabel('Iterations:'))
        self.contIter = QLineEdit()
        self.contIter.setFixedSize(50, 30)
        self.contIter.textChanged.connect(partial(self.edited, 'cont_iter'))
        hl.addWidget(self.contIter)
        hl.addStretch(1)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel('Smooth:'))
        self.contSmooth = QLineEdit()
        self.contSmooth.setFixedSize(50, 30)
        self.contSmooth.textChanged.connect(partial(self.edited, 'cont_smooth'))
        hl.addWidget(self.contSmooth)
        hl.addStretch(1)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        hl.addWidget(QLabel('Clipping:'))
        self.contClip = QLineEdit()
        self.contClip.setFixedSize(50, 30)
        self.contClip.textChanged.connect(partial(self.edited, 'cont_clip'))
        hl.addWidget(self.contClip)
        clip_group = QButtonGroup(self)
        self.positive = QRadioButton('posit.')
        self.positive.setChecked(True)
        clip_group.addButton(self.positive)
        hl.addWidget(self.positive)
        self.negative = QRadioButton('negat.')
        clip_group.addButton(self.negative)
        hl.addWidget(self.negative)
        self.absolute = QRadioButton('absol.')
        clip_group.addButton(self.absolute)
        hl.addWidget(self.absolute)
        hl.addStretch(1)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        data_group = QButtonGroup(self)
        self.spectrum = QRadioButton('spectrum')
        self.spectrum.setChecked(True)
        data_group.addButton(self.spectrum)
        hl.addWidget(self.spectrum)
        self.cont = QRadioButton('continuum')
        data_group.addButton(self.cont)
        hl.addWidget(self.cont)
        hl.addStretch(1)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        type_group = QButtonGroup(self)
        self.fullRange = QRadioButton('full')
        self.fullRange.setChecked(True)
        type_group.addButton(self.fullRange)
        hl.addWidget(self.fullRange)
        self.shownRange = QRadioButton('shown')
        type_group.addButton(self.shownRange)
        hl.addWidget(self.shownRange)
        self.windowRange = QRadioButton('window:')
        type_group.addButton(self.windowRange)
        hl.addWidget(self.windowRange)
        self.xmin = QLineEdit()
        self.xmin.setFixedSize(50, 30)
        self.xmin.textChanged.connect(partial(self.edited, 'x_min'))
        hl.addWidget(self.xmin)
        hl.addWidget(QLabel('..'))
        self.xmax = QLineEdit()
        self.xmax.setFixedSize(50, 30)
        self.xmax.textChanged.connect(partial(self.edited, 'x_max'))
        hl.addWidget(self.xmax)
        hl.addStretch(1)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        write_group = QButtonGroup(self)
        self.new = QRadioButton('new')
        self.new.setChecked(True)
        write_group.addButton(self.new)
        hl.addWidget(self.new)
        self.add = QRadioButton('add/overwrite')
        write_group.addButton(self.add)
        hl.addWidget(self.add)
        hl.addStretch(1)
        layout.addLayout(hl)
        hl = QHBoxLayout()
        exposure = QPushButton('Exposure:')
        exposure.setFixedSize(100, 30)
        exposure.clicked.connect(self.changeExp)
        self.expchoose = QComboBox()
        self.expchoose.setFixedSize(400, 30)
        for s in self.parent.s:
            self.expchoose.addItem(s.filename)
        if len(self.parent.s) > 0:
            self.exp_ind = self.parent.s.ind
            self.expchoose.currentIndexChanged.connect(self.onExpChoose)
            self.expchoose.setCurrentIndex(self.exp_ind)
        hl.addWidget(exposure)
        hl.addWidget(self.expchoose)
        hl.addStretch(0)
        layout.addStretch(1)
        layout.addLayout(hl)
        hl = QHBoxLayout()
        fit = QPushButton('Make it')
        fit.setFixedSize(100, 30)
        fit.clicked.connect(self.fit)
        hl.addWidget(fit)
        layout.addStretch(1)
        layout.addLayout(hl)
        self.setLayout(layout)

    def init_GUI_Bsplain(self):

        frame = QFrame(self)
        layout = QVBoxLayout()

        frame.setLayout(layout)
        self.tab.addTab(frame, 'Bspline')

    def init_GUI_SG(self):

        frame = QFrame(self)
        layout = QVBoxLayout()

        hl = QHBoxLayout()
        hl.addWidget(QLabel('Order:'))
        self.sgOrder = QLineEdit()
        self.sgOrder.setFixedSize(50, 30)
        self.sgOrder.textChanged.connect(partial(self.edited, 'sg_order'))
        hl.addWidget(self.sgOrder)
        hl.addStretch(1)
        layout.addLayout(hl)

        layout.addStretch(0)
        frame.setLayout(layout)
        self.tab.addTab(frame, 'SG')

    def init_GUI_Smooth(self):

        frame = QFrame(self)
        layout = QVBoxLayout()

        hl = QHBoxLayout()
        filter = QPushButton('Filter:')
        filter.setFixedSize(100, 30)
        filter.clicked.connect(self.changeFilter)
        self.filterchoose = QComboBox()
        self.filterchoose.setFixedSize(400, 30)
        self.filternames = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
        self.filterchoose.addItems(self.filternames)
        self.filterchoose.setCurrentIndex(0)
        hl.addWidget(filter)
        hl.addWidget(self.filterchoose)
        hl.addStretch(0)
        layout.addStretch(1)
        layout.addLayout(hl)

        frame.setLayout(layout)
        self.tab.addTab(frame, 'Smooth')

    def init_GUI_Cheb(self):

        frame = QFrame(self)
        layout = QVBoxLayout()

        frame.setLayout(layout)
        self.tab.addTab(frame, 'Chebyshev')

    def edited(self, attr):
        try:
            setattr(self, attr, self.opts[attr][1](getattr(self, self.opts[attr][0]).text()))
        except:
            pass

    def changeExp(self):
        self.exp_ind += 1
        if self.exp_ind >= len(self.parent.s):
            self.exp_ind = 0
        self.expchoose.setCurrentIndex(self.exp_ind)

    def changeFilter(self):
        ind = self.filterchoose.currentIndex() + 1 if self.filterchoose.currentIndex() < len(self.filternames)-1 else 0
        self.filterchoose.setCurrentIndex(ind)

    def onExpChoose(self, index, name='exp_ind'):
        setattr(self, name, index)

    def updateExpChoose(self):
        self.expchoose.clear()
        for s in self.parent.s:
            self.expchoose.addItem(s.filename)
        self.expchoose.setCurrentIndex(self.exp_ind)

    def fit(self):
        getattr(self, 'fit' + self.tab.tabText(self.tab.currentIndex()))()

    def fitBspline(self):
        x = self.getRange()
        self.parent.s[self.exp_ind].calcCont(method='Bspline', xl=x[0], xr=x[-1], iter=self.cont_iter, window=self.cont_smooth,
                                             clip=self.cont_clip, new=self.new.isChecked(), cont=self.cont.isChecked(), sign=self.sign())

    def fitSmooth(self):
        x = self.getRange()
        self.parent.s[self.exp_ind].calcCont(method='Smooth', xl=x[0], xr=x[-1], iter=self.cont_iter, window=self.cont_smooth,
                                             clip=self.cont_clip, filter=self.filternames[self.filterchoose.currentIndex()],
                                             new=self.new.isChecked(), cont=self.cont.isChecked(), sign=self.sign())

    def fitSG(self):
        x = self.getRange()
        self.parent.s[self.exp_ind].calcCont(method='SG', xl=x[0], xr=x[-1], iter=self.cont_iter, window=self.cont_smooth,
                                             clip=self.cont_clip, sg_order=self.sg_order, new=self.new.isChecked(),
                                             cont=self.cont.isChecked(), sign=self.sign())

    def sign(self):
        if self.positive.isChecked():
            return 1
        if self.negative.isChecked():
            return -1
        if self.absolute.isChecked():
            return 0

    def getRange(self):
        if self.fullRange.isChecked():
            x = [self.parent.s[self.exp_ind].spec.x()[0], self.parent.s[self.exp_ind].spec.x()[-1]]
        if self.shownRange.isChecked():
            x = self.parent.plot.vb.getState()['viewRange'][0]
        if self.windowRange.isChecked():
            x = [float(self.xmin.text()), float(self.xmax.text())]
        return x

    def closeEvent(self, event):
        self.parent.fitContWindow = None
        event.accept()

class SDSSentry():
    def __init__(self, name):
        self.name = name
        self.attr = ['name']

    def add_attr(self, attr):
        self.attr.append(attr)
    
    def __repr__(self):
        st = ''
        for a in self.attr:
            st += a + '=' + str(getattr(self, a)) + '\n'
        return st
        
    def __str__(self):
        return self.name

class loadSDSSwidget(QWidget):
    
    def __init__(self, parent):
        super(loadSDSSwidget, self).__init__()
        self.parent = parent
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        self.initUI()
        
    def initUI(self):      

        splitter = QSplitter(Qt.Orientation.Vertical)

        layout = QVBoxLayout(self)
        l = QHBoxLayout(self)
        l.addWidget(QLabel('Plate:', self))
        self.plate = QLineEdit(self)
        self.plate.setMaxLength(4)
        l.addWidget(self.plate)

        l.addWidget(QLabel('MJD:', self))
        self.mjd = QLineEdit(self)
        self.mjd.setMaxLength(5)
        l.addWidget(self.mjd)
        #self.MJD.move(20, 90)
        
        l.addWidget(QLabel('fiber:', self))
        self.fiber = QLineEdit(self)
        self.fiber.setMaxLength(4)
        l.addWidget(self.fiber)
        l.addStretch(1)

        l.addWidget(QLabel('or name:', self))
        self.name = QLineEdit(self)
        self.name.setMaxLength(30)
        self.name.setFixedSize(200, 30)
        l.addWidget(self.name)

        layout.addLayout(l)

        l = QHBoxLayout(self)
        self.load = QPushButton('Load', self)
        self.load.setFixedSize(150, 30)
        #self.load.resize(self.load.sizeHint())
        self.load.clicked.connect(self.loadspectrum)
        l.addWidget(self.load)
        self.add = QPushButton('Add', self)
        self.add.setFixedSize(150, 30)
        # self.load.resize(self.load.sizeHint())
        self.add.clicked.connect(partial(self.loadspectrum, append=True))
        l.addWidget(self.add)

        l.addStretch(1)

        layout.addLayout(l)
        layout.addStretch(1)

        widget = QWidget()
        widget.setLayout(layout)
        splitter.addWidget(widget)

        layout = QVBoxLayout(self)
        l = QHBoxLayout(self)
        l.addWidget(QLabel('Load list:'))
        self.filename = QLineEdit(self)
        self.filename.setMaxLength(100)
        self.filename.setFixedSize(600, 30)
        l.addWidget(self.filename)
        self.choosefile = QPushButton('Choose', self)
        self.choosefile.setFixedSize(100, 30)
        self.choosefile.clicked.connect(self.chooseFile)
        l.addWidget(self.choosefile)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        self.Astroquery = QCheckBox('Astroquery')
        self.Astroquery.clicked.connect(partial(self.selectCat, 'Astroquery'))
        l.addWidget(self.Astroquery)

        self.DR14 = QCheckBox('DR14')
        self.DR14.clicked.connect(partial(self.selectCat, 'DR14'))
        l.addWidget(self.DR14)

        self.DR12 = QCheckBox('DR12')
        self.DR12.clicked.connect(partial(self.selectCat, 'DR12'))
        l.addWidget(self.DR12)

        self.DR9Lee = QCheckBox('DR9Lee')
        self.DR9Lee.clicked.connect(partial(self.selectCat, 'DR9Lee'))
        l.addWidget(self.DR9Lee)
        l.addStretch(1)

        grp = QButtonGroup(self)
        for attr in ['Astroquery', 'DR14', 'DR12', 'DR9Lee']:
            grp.addButton(getattr(self, attr))
        getattr(self, self.parent.SDSScat).setChecked(True)

        layout.addLayout(l)

        hl = QHBoxLayout(self)

        self.sdsslist = QTextEdit('#enter list in PLATE, FIBER format')
        self.sdsslist.setFixedSize(250, 500)
        hl.addWidget(self.sdsslist)

        self.listPars = QWidget(self)
        self.scrolllayout = QVBoxLayout(self.listPars)
        self.scroll = None
        self.saved = {}
        self.updateScroll()
        hl.addWidget(self.listPars)

        vl = QVBoxLayout(self)
        self.preview = QTextEdit()
        self.preview.setFixedHeight(500)
        vl.addWidget(self.preview)

        l = QHBoxLayout(self)
        l.addWidget(QLabel('Plate:'))
        self.plate_col = QLineEdit('2')
        self.plate_col.setFixedSize(40, 30)
        l.addWidget(self.plate_col)

        l.addWidget(QLabel('Fiber:'))
        self.fiber_col = QLineEdit('4')
        self.fiber_col.setFixedSize(40, 30)
        l.addWidget(self.fiber_col)

        l.addWidget(QLabel('Name:'))
        self.name_col = QLineEdit()
        self.name_col.setFixedSize(40, 30)
        l.addWidget(self.name_col)

        l.addWidget(QLabel('Header:'))
        self.header = QLineEdit()
        self.header.setText('1')
        self.header.setFixedSize(40, 30)
        l.addWidget(self.header)
        l.addStretch(1)
        vl.addLayout(l)

        l = QHBoxLayout(self)
        l.addWidget(QLabel('Add columns:'))
        self.addcolumns = QLineEdit()
        self.addcolumns.setText('6')
        self.addcolumns.setFixedSize(100, 30)
        l.addWidget(self.addcolumns)
        l.addStretch(1)
        vl.addLayout(l)

        hl.addStretch(1)
        hl.addLayout(vl)

        layout.addLayout(hl)

        l = QHBoxLayout(self)
        self.loadlist = QPushButton('Load list', self)
        self.loadlist.setFixedSize(150, 30)
        # self.load.resize(self.load.sizeHint())
        self.loadlist.clicked.connect(self.loadList)
        l.addWidget(self.loadlist)
        l.addStretch(1)

        layout.addLayout(l)

        layout.addStretch(1)
        widget = QWidget()
        widget.setLayout(layout)
        splitter.addWidget(widget)

        splitter.setSizes([250, 1500])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)

        self.setGeometry(300, 300, 950, 900)
        self.setWindowTitle('load SDSS by Plate/MJD/Fiber or name')
        self.show()
        self.selectCat()

    def updateScroll(self):
        for s in self.saved.keys():
            try:
                self.scrolllayout.removeWidget(getattr(self, s))
                getattr(self, s).deleteLater()
            except:
                pass
        if self.scroll is not None:
            self.scrolllayout.removeWidget(self.scroll)
            self.scroll.deleteLater()

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        #self.scroll.setMaximumHeight(self.height()-150)
        self.scrollContent = QWidget(self.scroll)
        if hasattr(self, 'data'):
            l = QVBoxLayout()
            for par in self.saved.keys():
                setattr(self, str(par), QCheckBox(str(par)))
                getattr(self, str(par)).setChecked(self.saved[par])
                getattr(self, str(par)).clicked[bool].connect(partial(self.click, str(par)))
                l.addWidget(getattr(self, str(par)))
            l.addStretch()
            self.scrollContent.setLayout(l)
            self.scroll.setWidget(self.scrollContent)
        self.scrolllayout.addWidget(self.scroll)


    def loadspectrum(self, append=False):

        plate = None if len(self.mjd.text().strip()) == 0 else int(self.plate.text())
        MJD = None if len(self.mjd.text().strip()) == 0 else int(self.mjd.text())
        fiber = None if len(self.mjd.text().strip()) == 0 else int(self.fiber.text())#'{:0>4}'.format(self.fiber.text())
        name = self.name.text().strip()

        if self.parent.loadSDSS(plate=plate, MJD=MJD, fiber=fiber, name=name, append=append):
            pass
            #self.close()

    def chooseFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Import SDSS list', self.parent.SDSSfolder)

        if fname[0]:
            self.filename.setText(fname[0])
            self.parent.options('SDSSfolder', os.path.dirname(fname[0]))
            with open(fname[0], 'r') as f:
                try:
                    self.header.setText('1')
                    data = np.genfromtxt(self.filename.text(), names=True)
                    names = [n.lower() for n in data.dtype.names]
                    self.plate_col.setText(str(data.dtype.names.index(names.index('plate'))))
                    ind = names.index('fiber') if 'fiber' in names else names.index('fiberid')
                    self.fiber_col.setText(str(ind))
                except:
                    t = f.readlines()
                    self.preview.setPlainText(''.join(t))
                    ind = [len(s.split()) for s in t]
                    print(ind)
                    self.header.setText(str(np.argmax(ind)))

    def selectCat(self, cat=None):
        if cat != self.parent.SDSScat:
            if cat is None:
                cat = self.parent.SDSScat
            self.parent.options('SDSScat', cat)
            if cat == 'DR14':
                self.data = Table.read(r'C:\science\SDSS\DR14\DR14Q_v4_4.fits')
                default = ['SDSS_NAME', 'RA', 'DEC', 'PLATE', 'MJD', 'FIBERID', 'Z']
            if cat == 'DR12':
                self.data = Table(self.parent.IGMspec['BOSS_DR12']['meta'][()])
                default = ['SDSS_NAME', 'RA_GROUP', 'DEC_GROUP', 'PLATE', 'MJD', 'FIBERID', 'Z_VI']
            if cat == 'DR9Lee':
                self.data = Table.read(r'C:\science\SDSS\DR9_Lee\BOSSLyaDR9_cat.fits')
                default = ['SDSS_NAME', 'RA', 'DEC', 'PLATE', 'MJD', 'FIBERID', 'Z_VI']
            if cat == 'Astroquery':
                self.data = Table.read(r'C:\science\SDSS\DR14\DR14Q_v4_4.fits')
                default = ['SDSS_NAME', 'RA', 'DEC', 'PLATE', 'MJD', 'FIBERID', 'Z']
            self.saved = {}
            for par in self.data.colnames:
                self.saved[str(par)] = par in default
            self.updateScroll()

    def click(self, s):
        self.saved[s] = getattr(self, s).isChecked()

    def loadList(self):
        pars = [k for k, v in self.saved.items() if v]

        self.parent.SDSSdata = None
        inds, add_inds = [], []
        if self.filename.text().strip() == '':
            data = np.recarray((0,), dtype=[('plate', int), ('fiber', int)])
            for line in self.sdsslist.toPlainText().splitlines():
                if not line.startswith('#'):
                    data = np.append(data, np.array([(int(line.split()[0]), int(line.split()[1]))], dtype=data.dtype))
            if data.shape[0] > 0:
                plate, fiber = data['plate'], data['fiber']
            else:
                inds = np.arange(len(self.data['PLATE']))
        else:
            data = np.genfromtxt(self.filename.text(), dtype=None, skip_header=int(self.header.text()), encoding=None)
            if len(self.name_col.text().strip()) == 0:
                for p, f, i in zip(data['f{:d}'.format(int(self.plate_col.text())-1)], data['f{:d}'.format(int(self.fiber_col.text())-1)], range(len(data['f0']))):
                    print(p, f)
                    ind = np.where((self.data['PLATE'] == p) * (self.data['FIBERID'] == f))[0]
                    if len(ind) > 0:
                        inds.append(ind[0])
                        add_inds.append(i)
                    else:
                        print('missing:', p, f)
            else:
                for i, name in enumerate(data['f{:d}'.format(int(self.name_col.text())-1)]):
                    name = name.replace('J', '').replace('SDSS', '')
                    ra, dec = (name[:name.index('+')], name[name.index('+'):]) if '+' in name else (
                    name[:name.index('-')], name[name.index('-'):])
                    ra, dec = hms_to_deg(ra), dms_to_deg(dec)
                    if 'RA' in self.data.dtype.names:
                        RA, DEC = self.data['RA'], self.data['DEC']
                    elif 'RA_GROUP' in self.data.dtype.names:
                        RA, DEC = self.data['RA_GROUP'], self.data['DEC_GROUP']
                    ind = np.argmin((RA - ra) ** 2 + (DEC - dec) ** 2)
                    if (RA[ind] - ra) ** 2 + (DEC[ind] - dec) ** 2 < 0.3:
                        inds.append(ind)
                        add_inds.append(i)
                    #print(self.data['RA'], self.data['DEC'])
            if len(self.addcolumns.text().split()) > 0:
                add = data[:][['f{:d}'.format(int(i)-1) for i in self.addcolumns.text().split()]]
            else:
                add = None
        if len(self.data[pars][inds]) > 0:
            self.parent.SDSSdata = np.array(self.data[pars][inds])
            if add is not None:
                self.parent.SDSSdata = np.lib.recfunctions.merge_arrays([self.parent.SDSSdata, add[add_inds]], flatten=True, usemask=False)

        if self.parent.SDSSdata is not None:
            self.parent.SDSSlist = QSOlistTable(self.parent, 'SDSS')
            self.parent.SDSSlist.setdata(self.parent.SDSSdata)

    def keyPressEvent(self, qKeyEvent):
        if qKeyEvent.key() == Qt.Key.Key_Return:
            self.loadspectrum()

class SDSSPhotWidget(QWidget):
    def __init__(self, parent):
        super(SDSSPhotWidget, self).__init__()
        self.parent = parent
        self.setGeometry(100, 100, 2000, 1100)
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        self.show()


class observabilityWidget(QWidget):

    def __init__(self, parent):
        super(observabilityWidget, self).__init__()
        self.parent = parent
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        self.initUI()

    def initUI(self):

        layout = QVBoxLayout(self)
        l = QHBoxLayout(self)
        l.addWidget(QLabel('site:'))
        self.siteBox = QComboBox(self)
        self.siteBox.setFixedSize(150, 30)
        self.siteBox.addItems(astropy.coordinates.EarthLocation.get_site_names())
        self.siteBox.setCurrentText('Paranal Observatory')
        l.addWidget(self.siteBox)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        l.addWidget(QLabel('dates:  from '))
        self.datefrom = QLineEdit('2020-04-01 00:00')
        self.datefrom.setFixedSize(120, 30)
        l.addWidget(self.datefrom)
        l.addWidget(QLabel(' to '))
        self.dateto = QLineEdit('2020-10-01 23:59')
        self.dateto.setFixedSize(90, 30)
        l.addWidget(self.dateto)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        l.addWidget(QLabel('airmass:'))
        self.airmass = QLineEdit('1.3')
        self.airmass.setFixedSize(60, 30)
        l.addWidget(self.airmass)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout()
        self.loadSDSS = QCheckBox('load SDSS')
        l.addWidget(self.loadSDSS)
        l.addStretch(1)
        layout.addLayout(l)

        l = QHBoxLayout(self)
        self.loadList = QCheckBox('Load list:')
        l.addWidget(self.loadList)
        self.filename = QLineEdit(self)
        self.filename.setMaxLength(100)
        self.filename.setFixedSize(600, 30)
        l.addWidget(self.filename)
        self.choosefile = QPushButton('Choose', self)
        self.choosefile.setFixedSize(100, 30)
        self.choosefile.clicked.connect(self.chooseFile)
        l.addWidget(self.choosefile)
        l.addStretch(1)
        layout.addLayout(l)

        grp = QButtonGroup(self)
        for attr in ['loadSDSS', 'loadList']:
            grp.addButton(getattr(self, attr))
        self.loadSDSS.setChecked(True)

        layout.addLayout(l)

        vl = QVBoxLayout(self)
        self.preview = QTextEdit()
        self.preview.setFixedHeight(500)
        vl.addWidget(self.preview)

        l = QHBoxLayout(self)
        l.addWidget(QLabel('Name:'))
        self.name_col = QLineEdit()
        self.name_col.setFixedSize(40, 30)
        l.addWidget(self.name_col)

        l.addWidget(QLabel('RA:'))
        self.ra_col = QLineEdit()
        self.ra_col.setFixedSize(40, 30)
        l.addWidget(self.ra_col)

        l.addWidget(QLabel('Dec:'))
        self.dec_col = QLineEdit()
        self.dec_col.setFixedSize(40, 30)
        l.addWidget(self.dec_col)

        l.addWidget(QLabel('Plate:'))
        self.plate_col = QLineEdit('2')
        self.plate_col.setFixedSize(40, 30)
        l.addWidget(self.plate_col)

        l.addWidget(QLabel('Fiber:'))
        self.fiber_col = QLineEdit('4')
        self.fiber_col.setFixedSize(40, 30)
        l.addWidget(self.fiber_col)

        l.addWidget(QLabel('Header:'))
        self.header = QLineEdit()
        self.header.setText('1')
        self.header.setFixedSize(40, 30)
        l.addWidget(self.header)
        l.addStretch(1)
        vl.addLayout(l)

        layout.addLayout(vl)

        l = QHBoxLayout(self)
        self.calculateObs = QPushButton('Calculate', self)
        self.calculateObs.setFixedSize(150, 30)
        # self.load.resize(self.load.sizeHint())
        self.calculateObs.clicked.connect(self.calculate)
        l.addWidget(self.calculateObs)

        self.exportObs = QPushButton('Export', self)
        self.exportObs.setFixedSize(100, 30)
        # self.load.resize(self.load.sizeHint())
        self.exportObs.clicked.connect(self.export)
        l.addWidget(self.exportObs)
        l.addStretch(1)
        layout.addLayout(l)

        layout.addStretch(1)

        self.setLayout(layout)

        self.setGeometry(300, 300, 950, 900)
        self.setWindowTitle('calculate Observability')
        self.show()

    def chooseFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Import SDSS list', self.parent.SDSSfolder)

        if fname[0]:
            self.filename.setText(fname[0])
            self.parent.options('SDSSfolder', os.path.dirname(fname[0]))
            with open(fname[0], 'r') as f:
                try:
                    self.header.setText('1')
                    data = np.genfromtxt(self.filename.text(), names=True)
                    names = [n.lower() for n in data.dtype.names]
                    self.plate_col.setText(str(data.dtype.names.index(names.index('plate'))))
                    ind = names.index('fiber') if 'fiber' in names else names.index('fiberid')
                    self.fiber_col.setText(str(ind))
                except:
                    t = f.readlines()
                    self.preview.setPlainText(''.join(t))
                    ind = [len(s.split()) for s in t]
                    print(ind)
                    self.header.setText(str(np.argmax(ind)))
            self.loadList.setChecked(True)

    def click(self, s):
        self.saved[s] = getattr(self, s).isChecked()

    def loadList(self):
        pars = [k for k, v in self.saved.items() if v]

        self.parent.SDSSdata = None
        inds, add_inds = [], []
        if self.filename.text().strip() == '':
            data = np.recarray((0,), dtype=[('plate', int), ('fiber', int)])
            for line in self.sdsslist.toPlainText().splitlines():
                if not line.startswith('#'):
                    data = np.append(data, np.array([(int(line.split()[0]), int(line.split()[1]))], dtype=data.dtype))
            if data.shape[0] > 0:
                plate, fiber = data['plate'], data['fiber']
            else:
                inds = np.arange(len(self.data['PLATE']))
        else:
            data = np.genfromtxt(self.filename.text(), dtype=None, skip_header=int(self.header.text()), encoding=None)
            if len(self.name_col.text().strip()) == 0:
                for p, f, i in zip(data['f{:d}'.format(int(self.plate_col.text()) - 1)],
                                   data['f{:d}'.format(int(self.fiber_col.text()) - 1)], range(len(data['f0']))):
                    print(p, f)
                    ind = np.where((self.data['PLATE'] == p) * (self.data['FIBERID'] == f))[0]
                    if len(ind) > 0:
                        inds.append(ind[0])
                        add_inds.append(i)
                    else:
                        print('missing:', p, f)
            else:
                for i, name in enumerate(data['f{:d}'.format(int(self.name_col.text()) - 1)]):
                    name = name.replace('J', '').replace('SDSS', '')
                    ra, dec = (name[:name.index('+')], name[name.index('+'):]) if '+' in name else (
                        name[:name.index('-')], name[name.index('-'):])
                    ra, dec = hms_to_deg(ra), dms_to_deg(dec)
                    if 'RA' in self.data.dtype.names:
                        RA, DEC = self.data['RA'], self.data['DEC']
                    elif 'RA_GROUP' in self.data.dtype.names:
                        RA, DEC = self.data['RA_GROUP'], self.data['DEC_GROUP']
                    ind = np.argmin((RA - ra) ** 2 + (DEC - dec) ** 2)
                    if (RA[ind] - ra) ** 2 + (DEC[ind] - dec) ** 2 < 0.3:
                        inds.append(ind)
                        add_inds.append(i)
                    # print(self.data['RA'], self.data['DEC'])
            if len(self.addcolumns.text().split()) > 0:
                add = data[:][['f{:d}'.format(int(i) - 1) for i in self.addcolumns.text().split()]]
            else:
                add = None
        if len(self.data[pars][inds]) > 0:
            self.parent.SDSSdata = np.array(self.data[pars][inds])
            if add is not None:
                self.parent.SDSSdata = np.lib.recfunctions.merge_arrays([self.parent.SDSSdata, add[add_inds]],
                                                                        flatten=True, usemask=False)

        if self.parent.SDSSdata is not None:
            self.parent.SDSSlist = QSOlistTable(self.parent, 'SDSS')
            self.parent.SDSSlist.setdata(self.parent.SDSSdata)

    def calculate(self):
        observer = astroplan.Observer.at_site(self.siteBox.currentText())
        print(observer)
        time_range = astropy.time.Time([self.datefrom.text(), self.dateto.text()])
        print(time_range)
        constraints = [astroplan.AirmassConstraint(float(self.airmass.text()), 1.0), astroplan.AtNightConstraint.twilight_civil()]
        print(constraints)
        targets = []
        names = self.parent.SDSSdata.dtype.names[np.where(['name' in name.lower() for name in self.parent.SDSSdata.dtype.names])[0][0]]
        if self.loadSDSS.isChecked():
            for name in self.parent.SDSSdata[names]:
                name = name.replace('J', '').replace('SDSS', '').replace(':', '').replace('−', '-').strip()
                ra, dec = (name[:name.index('+')], name[name.index('+'):]) if '+' in name else (name[:name.index('-')], name[name.index('-'):])
                ra, dec = hms_to_deg(ra), dms_to_deg(dec)
                print(ra, dec)
                targets.append(FixedTarget(name=name, coord=astropy.coordinates.SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame='icrs')))
        else:
            pass

        if 1:
            table = astroplan.observability_table(constraints, observer, targets, time_range=time_range)
        else:
            table = np.ones([len(targets), 4])
        print(table[1][:])
        if 'obs' not in self.parent.SDSSdata.dtype.names:
            self.parent.SDSSdata = add_field(self.parent.SDSSdata, [('obs', bool)], table[1][:])
        else:
            self.parent.SDSSdata['obs'] = table[1][:]
        if 'time1.3' not in self.parent.SDSSdata.dtype.names:
            self.parent.SDSSdata = add_field(self.parent.SDSSdata, [('time1.3', float)], table[3][:])
        else:
            self.parent.SDSSdata['time1.3'] = table[3][:]

        self.parent.SDSSlist.close()
        self.parent.show_SDSS_list()

    def export(self, filename=None):
        #self.calculate()
        filename = QFileDialog.getSaveFileName(self, 'Export list', self.parent.SDSSfolder)
        with open(filename[0], 'w') as file:
            ascii.write(self.parent.SDSSdata, output=file, format='fixed_width_two_line')


    def closeEvent(self, event):
        self.parent.observability = None
        event.accept()

class ShowListImport(QWidget):
    def __init__(self, parent, cat=''):
        super().__init__()
        self.parent = parent

        self.move(400, 100)
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        self.table = QSOlistTable(self.parent, cat=cat, subparent=self, editable=False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        self.loadallButton = QPushButton("Show all")
        self.loadallButton.setFixedSize(90, 30)
        self.loadallButton.clicked[bool].connect(self.loadall)
        
        self.loadButton = QPushButton("Show")
        self.loadButton.setFixedSize(90, 30)
        self.loadButton.clicked[bool].connect(self.load)
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setFixedSize(90, 30)
        self.cancelButton.clicked[bool].connect(self.close)
        hbox = QHBoxLayout()
        hbox.addWidget(self.loadallButton)
        hbox.addStretch(1)
        hbox.addWidget(self.loadButton)
        hbox.addWidget(self.cancelButton)
        
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addLayout(hbox)
        self.setLayout(layout)
        
    def setdata(self, data):
        self.table.setdata(data)
        self.resize(self.width(), self.table.rowCount()*40+210)

    def load(self, loadall=False):
        flist = set(self.table.item(index.row(),0).text() for index in self.table.selectedIndexes())
        print(flist)
        with open(self.parent.importListFile) as f:
            if loadall:
                flist = f.read().splitlines()
            self.parent.importSpectrum(flist, dir_path=os.path.dirname(self.parent.importListFile)+'/')

    def loadall(self):
        self.load(loadall=True)


class ShowListCombine(QWidget):
    def __init__(self, parent, cat=''):
        super().__init__()
        self.parent = parent

        self.setStyleSheet(open(self.parent.parent.folder + 'config/styles.ini').read())
        self.table = QSOlistTable(self.parent.parent, cat=cat, subparent=self, editable=False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.setWidth = None

        self.update()
        self.show()

    def update(self):
        dtype = [('filename', str, 100), ('obs. date', str, 30),
                 ('wavelmin', np.float64), ('wavelmax', np.float64),
                 ('resolution', int)]
        zero = ('', '', np.nan, np.nan, 0)
        data = np.array([zero], dtype=dtype)
        self.edit_col = [4]
        for s in self.parent.parent.s:
            print(s.filename, s.date, s.wavelmin, s.wavelmax, s.resolution())
            data = np.insert(data, len(data), np.array(
                [('  ' + s.filename + '  ', '  ' + s.date + '  ', s.wavelmin, s.wavelmax, s.resolution())],
                dtype=dtype), axis=0)
        data = np.delete(data, (0), axis=0)
        self.table.setdata(data)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        if self.setWidth is None:
            self.setWidth = 120 + self.table.verticalHeader().width() + self.table.autoScrollMargin() * 2.5
            self.setWidth += np.sum([self.table.columnWidth(c) for c in range(self.table.columnCount())])

        self.table.resize(int(self.setWidth), int(self.table.rowCount() * 40 + 140))


class ExportDataWidget(QWidget):
    def __init__(self, parent, type):
        super().__init__()
        self.parent = parent
        self.type = type
        if self.type in ['export', 'export2d']:
            try:
                self.filename = self.parent.s[self.parent.s.ind].filename
            except:
                self.filename = None
        if self.type == 'save':
            self.filename = self.parent.options('filename_saved')
        self.initUI()
        
    def initUI(self):

        layout = QVBoxLayout()

        lbl = QLabel('filename:')
        self.setfilename = QLineEdit(self.filename)
        self.setfilename.resize(600, 25)
        self.setfilename.textChanged[str].connect(self.filenameChanged)

        self.chooseFile = QPushButton("Get file")
        self.chooseFile.clicked[bool].connect(self.chooseFileName)
        self.chooseFile.setFixedSize(70, 30)
        hbox_file = QHBoxLayout()
        hbox_file.addWidget(lbl)
        hbox_file.addWidget(self.setfilename)
        hbox_file.addWidget(self.chooseFile)
        #hbox_file.addStretch(1)
        layout.addLayout(hbox_file)

        if self.type == 'export':
            self.check = OrderedDict([('spectrum', 'Spectrum'), ('cont', 'Continuum'),
                                      ('fit', 'Fit model'), ('fit_comps', 'Fit comps'), ('trans', 'Transmission')])
            self.opt = self.parent.export_opt
        elif self.type == 'save':
            self.check = OrderedDict([('spectrum', 'Spectrum'), ('cont', 'Continuum'),
                                      ('points', 'Selected points'), ('fit', 'Fit model'),
                                      ('others', 'Other data'), ('fit_results', 'Fit results')])
            self.opt = self.parent.save_opt
        elif self.type == 'export2d':
            self.check = OrderedDict([('spectrum', 'Spectrum'), ('err', 'Error'),
                                      ('mask', 'Masked values'), ('cr', 'Cosmic Ray mask'),
                                      ('sky', 'Sky model'), ('trace', 'Trace')])
            self.opt = self.parent.export2d_opt

        for k, v in self.check.items():
            setattr(self, k, QCheckBox(v))
            if k in self.opt:
                getattr(self, k).setChecked(True)
            layout.addWidget(getattr(self, k))

        for k, v in self.check.items():
            getattr(self, k).stateChanged.connect(self.onChanged)

        if self.type == 'export':
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel('wavelenghts units:  '))
            self.wave_units = ['angstr', 'nm']
            self.waveunit = 'angstr'
            for s in self.wave_units:
                setattr(self, s, QCheckBox(s))
                getattr(self, s).clicked[bool].connect(partial(self.waveChanged, s))
                hbox.addWidget(getattr(self, s))
            self.waveChanged(self.waveunit)
            hbox.addStretch(1)
            layout.addLayout(hbox)

            hbox = QHBoxLayout()
            hbox.addWidget(QLabel('range:  '))
            self.exp_range = ['full', 'window']
            self.range = 'full'
            for s in self.exp_range:
                setattr(self, s, QCheckBox(s))
                getattr(self, s).clicked[bool].connect(partial(self.rangeChanged, s))
                hbox.addWidget(getattr(self, s))
            self.rangeChanged(self.range)
            hbox.addStretch(1)
            layout.addLayout(hbox)

            hbox = QHBoxLayout()
            hbox.addWidget(QLabel('fit:  '))
            self.fit_types = ['full', 'masked']
            self.fit_type = 'full'
            for s in self.fit_types:
                setattr(self, 'fit_'+s, QCheckBox(s))
                getattr(self, 'fit_'+s).clicked[bool].connect(partial(self.fitChanged, s))
                hbox.addWidget(getattr(self, 'fit_'+s))
            self.fitChanged(self.fit_type)
            hbox.addStretch(1)
            layout.addLayout(hbox)

            hbox = QHBoxLayout()
            self.cheb_applied = QCheckBox('chebyshev applied?')
            self.cheb_applied.setChecked(True)
            hbox.addWidget(self.cheb_applied)
            hbox.addStretch(1)
            layout.addLayout(hbox)

            hbox = QHBoxLayout()
            self.normalized = QCheckBox('normalized')
            self.normalized.setChecked(self.parent.normview)
            hbox.addWidget(self.normalized)
            hbox.addStretch(1)
            layout.addLayout(hbox)

        self.okButton = QPushButton(self.type.title())
        self.okButton.clicked[bool].connect(getattr(self, self.type))
        self.okButton.setFixedSize(80, 30)
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked[bool].connect(self.close)
        self.cancelButton.setFixedSize(80, 30)
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.cancelButton)
        
        layout.addStretch(1)
        layout.addLayout(hbox)
        self.setLayout(layout)
        
        self.setGeometry(200, 200, 800, 350)
        self.setWindowTitle(self.type.title() + ' Data')
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        self.show()
        
    def filenameChanged(self, text):
        try:
            self.filename = text
        except:
            pass
    
    def chooseFileName(self):
        fname = QFileDialog.getSaveFileName(self, 'Export spectrum', self.parent.work_folder)
        if fname[0]:
            self.filename = fname[0]
            self.setfilename.setText(self.filename)
            self.parent.options('filename_saved', self.filename)
            self.parent.options('work_folder', os.path.dirname(self.filename))
            self.parent.statusBar.setText('Filename is set' + self.filename)

    def onChanged(self):
        self.opt = []
        for k in self.check.keys():
            if getattr(self, k).isChecked():
                self.opt.append(k)

    def waveChanged(self, unit):
        self.waveunit = unit
        for s in self.wave_units:
            getattr(self, s).setChecked(False)
        getattr(self, unit).setChecked(True)

    def rangeChanged(self, unit):
        self.range = unit
        for s in self.exp_range:
            getattr(self, s).setChecked(False)
        getattr(self, unit).setChecked(True)

    def fitChanged(self, unit):
        self.fit_type = unit
        for s in self.fit_types:
            getattr(self, 'fit_'+s).setChecked(False)
        getattr(self, 'fit_'+unit).setChecked(True)

    def export(self):
        s = self.parent.s[self.parent.s.ind]
        kwargs = {'fmt':'%.5f', 'delimiter': ' '}
        unit = 1
        if self.waveunit == 'nm':
            unit = 10

        if self.fit_type == 'full':
            fit_mask = np.ones_like(s.fit.x(), dtype=bool)
        elif self.fit_type == 'masked':
            fit_mask = mask[s.fit_mask.x()]

        if self.range == 'window':
            print(s.spec.x())
            print(self.parent.plot.vb.getState()['viewRange'][0][0], self.parent.plot.vb.getState()['viewRange'][0][-1])
            mask = (s.spec.x() > self.parent.plot.vb.getState()['viewRange'][0][0]) * (s.spec.x() < self.parent.plot.vb.getState()['viewRange'][0][-1])
            cont_mask = (s.cont.x > self.parent.plot.vb.getState()['viewRange'][0][0]) * (s.cont.x < self.parent.plot.vb.getState()['viewRange'][0][-1])
            fit_mask *= (s.fit.x() > self.parent.plot.vb.getState()['viewRange'][0][0]) * (s.fit.x() < self.parent.plot.vb.getState()['viewRange'][0][-1])
        else:
            mask, cont_mask = np.isfinite(s.spec.x()), np.isfinite(s.cont.x)
        print(np.sum(mask))
        print(np.sum(s.cont_mask))
        print(np.sum(fit_mask))

        normview_saved = self.parent.normview
        if self.parent.normview != self.normalized.isChecked():
            self.parent.normalize(self.normalized.isChecked())

        if self.cheb_applied.isChecked() and self.parent.fit.cont_fit:
            cheb = interp1d(s.spec.raw.x[s.cont_mask], s.correctContinuum(s.spec.raw.x[s.cont_mask]), fill_value='extrapolate')
        else:
            cheb = interp1d(s.spec.raw.x[s.cont_mask], np.ones_like(s.spec.raw.x[s.cont_mask]), fill_value=1, bounds_error=False)

        print(self.spectrum.isChecked(), self.cont.isChecked(), self.fit.isChecked(), self.fit_comps.isChecked())
        if self.spectrum.isChecked():
            np.savetxt(self.filename, np.c_[s.spec.x()[mask] / unit, s.spec.y()[mask] * np.power(cheb(s.spec.x()[mask]), -self.parent.normview), s.spec.err()[mask], s.fit_mask.x()[mask]], **kwargs)
        if self.cont.isChecked():
            np.savetxt('_cont.'.join(self.filename.rsplit('.', 1)), np.c_[s.cont.x[cont_mask] / unit, s.cont.y[cont_mask] * np.power(cheb(s.cont.x[cont_mask]), 1 - 2 * self.parent.normview)], **kwargs)
            if len(s.cheb.disp[0].norm.x) > 0:
                np.savetxt('_cont_disp.'.join(self.filename.rsplit('.', 1)), np.c_[s.cheb.disp[0].x()[fit_mask] / unit, s.cheb.disp[0].y()[fit_mask] * np.power(cheb(s.fit.x()[fit_mask]), -self.parent.normview), s.cheb.disp[1].y()[fit_mask] * np.power(cheb(s.fit.x()[fit_mask]), -self.parent.normview)], **kwargs)
        if self.fit.isChecked():
            np.savetxt('_fit.'.join(self.filename.rsplit('.', 1)), np.c_[s.fit.x()[fit_mask] / unit, s.fit.y()[fit_mask] * np.power(cheb(s.fit.x()[fit_mask]), -self.parent.normview)], **kwargs)
            if len(s.fit.disp[0].norm.x) > 0:
                np.savetxt('_fit_disp.'.join(self.filename.rsplit('.', 1)), np.c_[s.fit.disp[0].x()[fit_mask] / unit, s.fit.disp[0].y()[fit_mask] * np.power(cheb(s.fit.x()[fit_mask]), -self.parent.normview), s.fit.disp[1].y()[fit_mask] * np.power(cheb(s.fit.x()[fit_mask]), -self.parent.normview)], **kwargs)
        #if self.fit.isChecked():
        #    np.savetxt('_fit_regions.'.join(self.filename.rsplit('.', 1)), np.c_[s.spec.x()[np.logical_and(mask, fit_mask)] / unit, (s.spec.y() / cheb(s.spec.x()))[np.logical_and(mask, fit_mask)]], **kwargs)
        if self.fit_comps.isChecked():
            for i, c in enumerate(s.fit_comp):
                if self.range == 'window':
                    fit_mask = (c.x() > self.parent.plot.vb.getState()['viewRange'][0][0]) * (c.x() < self.parent.plot.vb.getState()['viewRange'][0][-1])
                else:
                    fit_mask = np.ones_like(c.x(), dtype=bool)
                np.savetxt(f'_fit_comps_{i}.'.join(self.filename.rsplit('.', 1)), np.c_[c.x()[fit_mask] / unit, c.y()[fit_mask] / cheb(c.x()[fit_mask])], **kwargs)
                if len(c.disp[0].norm.x > 0):
                    if self.range == 'window':
                        fit_mask = (c.disp[0].norm.x > self.parent.plot.vb.getState()['viewRange'][0][0]) * (c.disp[0].norm.x < self.parent.plot.vb.getState()['viewRange'][0][-1])
                    else:
                        fit_mask = np.ones_like(c.disp[0].norm.x, dtype=bool)
                    np.savetxt(f'_fit_comps_{i}_disp.'.join(self.filename.rsplit('.', 1)), np.c_[c.disp[0].x()[fit_mask] / unit, c.disp[0].y()[fit_mask] * np.power(cheb(c.disp[0].x()[fit_mask]), -self.parent.normview), c.disp[1].y()[fit_mask] * np.power(cheb(c.disp[0].x()[fit_mask]), -self.parent.normview)], **kwargs)
            #print([[len(c.x()), len(c.y())] for c in s.fit_comp])
            #print([c.y()[fit_mask] / cheb(s.fit.x()[fit_mask]) for c in s.fit_comp])
            #np.savetxt('_fit_comps.'.join(self.filename.rsplit('.', 1)), np.column_stack([s.fit.x()[fit_mask] / unit] + [c.y()[fit_mask] / cheb(s.fit.x()[fit_mask]) for c in s.fit_comp]), **kwargs)
        if self.trans.isChecked():
            print("export tranmission")

            tell = self.parent.options("telluric")
            self.parent.options("telluric", False)
            self.parent.showFit(all=True)
            s = self.parent.s[self.parent.s.ind]
            if hasattr(s, "sky_cont") and len(s.sky_cont.x()) > 0:
                sky = s.sky_cont.norm.y
                x = s.sky_cont.norm.x
            else:
                x = s.fit.x()
                sky = np.ones_like(x)
            fint = interp1d(s.fit.line.norm.x, s.fit.line.norm.y, bounds_error=False, fill_value=1.0)
            np.savetxt('_trans.'.join(self.filename.rsplit('.', 1)), np.c_[x / unit, sky * fint(x) * np.power(cheb(x), -self.parent.normview)], **kwargs)

            self.parent.options("telluric", tell)

        if normview_saved != self.parent.normview:
            self.parent.normalize(normview_saved)

    def save(self):
        self.parent.save_opt = self.opt
        self.parent.saveFile(self.filename)
        self.parent.options('filename_saved', self.filename)
        self.close()

    def export2d(self):
        self.parent.export2dSpectrum(self.filename, self.opt)

    def closeEvent(self, event):
        self.parent.save_opt = self.opt


class combineWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.resize(1800, 1000)
        self.move(200, 100)
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())

        layout = QVBoxLayout()
        l = QHBoxLayout()

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('combine type: '))
        self.selectcombtype = QComboBox(self)
        self.selectcombtype.addItems(['Weighted mean', 'Median', 'Mean'])
        self.selectcombtype.setFixedSize(150, 30)
        self.selectcombtype.setCurrentIndex(0)
        hbox.addWidget(self.selectcombtype)
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addWidget(QLabel('wavelength scale: '))

        self.tab = QTabWidget()
        self.tab.setGeometry(0, 0, 550, 200)
        self.tab.setMinimumSize(550, 200)
        self.addItems()
        self.tab.setCurrentIndex(2)
        vbox.addWidget(self.tab)
        vbox.addStretch(1)
        l.addLayout(vbox)

        #self.expListView = QSOlistTable(self.parent, cat='fits', subparent=self, editable=False)
        self.expListView = ShowListCombine(self, cat='fits')
        l.addWidget(self.expListView)

        self.selectallButton = QPushButton("Select all")
        self.selectallButton.setFixedSize(90, 30)
        self.selectallButton.clicked[bool].connect(self.selectall)

        self.combineButton = QPushButton("Combine")
        self.combineButton.setFixedSize(90, 30)
        self.combineButton.clicked[bool].connect(self.combine)
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setFixedSize(90, 30)
        self.cancelButton.clicked[bool].connect(self.close)
        hbox = QHBoxLayout()
        hbox.addWidget(self.combineButton)
        hbox.addStretch(1)
        hbox.addWidget(self.selectallButton)
        #hbox.addWidget(self.cancelButton)

        layout.addLayout(l)
        layout.addLayout(hbox)
        self.setLayout(layout)

    def addItems(self):
        self.frames = ['Merged', 'Bin', 'Resolution', 'Log-linear', 'Load']
        self.tab.addTab(QFrame(self), 'Merged')

        frame = QFrame(self)
        l = QGridLayout()
        self.binsize = QLineEdit('0.025')
        l.addWidget(QLabel('Bin size, A:'), 0, 0)
        l.addWidget(self.binsize, 0, 1)
        l.addWidget(QLabel('First bin, A:'), 1, 0)
        firstbin = self.parent.s[self.parent.s.ind].spec.x()[0]
        self.zeropoint_bin = QLineEdit(str(firstbin))
        l.addWidget(self.zeropoint_bin, 1, 1)
        frame.setLayout(l)

        self.tab.addTab(frame, 'Bin')

        frame = QFrame(self)
        l = QGridLayout()
        l.addWidget(QLabel('Resolution: '), 0, 0)
        self.resolution = QLineEdit(str(int(self.parent.s[0].resolution())))
        l.addWidget(self.resolution, 0, 1)
        l.addWidget(QLabel('pixels pe FWHM: '), 1, 0)
        self.pp_fwhm = QLineEdit('3')
        l.addWidget(self.pp_fwhm, 1, 1)
        l.addWidget(QLabel('First bin, A:'), 2, 0)
        firstbin = self.parent.s.minmax()[0]
        self.zeropoint_res = QLineEdit(str(firstbin))
        l.addWidget(self.zeropoint_res, 2, 1)
        frame.setLayout(l)

        self.tab.addTab(frame, 'Resolution')

        frame = QFrame(self)
        l = QGridLayout()
        self.binsize_log = QLineEdit('0.0001')
        l.addWidget(QLabel('Bin size, in log A:'), 0, 0)
        l.addWidget(self.binsize_log, 0, 1)
        l.addWidget(QLabel('First bin, A:'), 1, 0)
        firstbin = self.parent.s.minmax()[0]
        self.zeropoint_log = QLineEdit(str(firstbin))
        l.addWidget(self.zeropoint_log, 1, 1)
        frame.setLayout(l)

        self.tab.addTab(frame, 'Log-linear')

        frame = QFrame(self)

        l = QHBoxLayout()
        self.file = QLineEdit('')
        self.file.setFixedHeight(30)
        l.addWidget(self.file)
        self.fromfile = QPushButton('load from file', self)
        self.fromfile.setFixedSize(130, 30)
        self.fromfile.clicked[bool].connect(self.loadfromfile)
        l.addWidget(self.fromfile)
        frame.setLayout(l)

        self.tab.addTab(frame, 'File')

    def loadfromfile(self):
        fname = QFileDialog.getOpenFileName(self, 'Import wavelength scale', '')

        if fname[0]:
            self.file.setText(fname[0])
            self.file_grid = np.genfromfile(fname[0], unpack=True, usecols=(0))

    def selectall(self):
        for i in range(len(self.parent.s)):
            self.expListView.table.selectRow(i)

    def combine(self):
        try:
            print([i.row() for i in self.expListView.table.selectionModel().selectedRows()])
            slist = [self.parent.s[i.row()] for i in self.expListView.table.selectionModel().selectedRows()]
        except:
            slist = self.parent.s
        print('slist:', len(slist))

        # make unified wavelength grid:
        if self.tab.currentIndex() == 0:
            x = set()
            for s in slist:
                print(len(s.spec.x()[np.logical_not(s.bad_mask.x())]))
                x.update(list(s.spec.x()[np.logical_not(s.bad_mask.x())]))
            x = sorted(x)
            x = np.asarray(x)

        elif self.tab.currentIndex() == 1:
            zero, binsize = float(self.zeropoint_bin.text()), float(self.binsize.text())
            num = int((self.parent.s.minmax()[1] - zero) / binsize)
            x = np.linspace(zero, zero + (num + 1) * binsize, num)

        elif self.tab.currentIndex() == 2:
            print('fixed res')
            zero = np.log10(float(self.zeropoint_res.text()))
            step = np.log10(1 + 1 / float(self.resolution.text()) / float(self.pp_fwhm.text()))
            print(zero, step)
            num = int((np.log10(self.parent.s.minmax()[1]) - zero) / step) + 1
            print(self.parent.s.minmax(), num)
            x = np.logspace(zero, zero + step * num, num + 1)

        elif self.tab.currentIndex() == 3:
            zero = np.log10(float(self.zeropoint_log.text()))
            step = float(self.binsize_log.text())
            num = int((np.log10(self.parent.s.minmax()[1]) - zero) / step) + 1
            x = np.logspace(zero, zero + step * num, num + 1)
            print(x[-1])

        elif self.tab.currentIndex() == 4:
            x = self.file_grid

        print('x: ', len(x), x)
        # calculate combined spectrum:
        comb = np.empty([len(slist), len(x)], dtype=float)
        comb.fill(np.nan)
        e_comb = np.empty([len(slist), len(x)], dtype=float)
        e_comb.fill(np.nan)

        for i, s in enumerate(slist):
            if 0:
                if 1:
                    spec = s.spec.y()[:]
                    spec[s.bad_mask.x()] = np.nan
                    spec = interp1d(s.spec.x(), spec, bounds_error=False, fill_value=np.nan)
                    err = s.spec.err()[:]
                    err[s.bad_mask.x()] = np.nan
                    err = interp1d(s.spec.x(), err, bounds_error=False, fill_value=np.nan)
                else:
                    spec = interp1d(s.spec.x()[np.logical_not(s.bad_mask.x())], s.spec.y()[np.logical_not(s.bad_mask.x())], bounds_error=False, fill_value=np.nan)
                    err = interp1d(s.spec.x()[np.logical_not(s.bad_mask.x())], s.spec.err()[np.logical_not(s.bad_mask.x())], bounds_error=False, fill_value=np.nan)
                comb[i] = spec(x)
                e_comb[i] = np.power(err(x), -1)
            else:
                #print(spectres.spectres(s.spec.x(), s.spec.y(), x, spec_errs=s.spec.err()))
                if (len(x) == len(s.spec.x())) and (np.max(np.abs(x - s.spec.x())) < 0.01 * np.max(np.diff(x))):
                    comb[i], e_comb[i] = s.spec.y()[:], s.spec.err()[:]
                else:
                    mask_s = (s.spec.err() != 0)
                    mask = (x > s.spec.x()[mask_s][2]) * (x < s.spec.x()[mask_s][-3])
                    comb[i][mask], e_comb[i][mask] = spectres.spectres(s.spec.x()[mask_s], s.spec.y()[mask_s], x[mask], spec_errs=s.spec.err()[mask_s])
                print(np.where(s.bad_mask.x())[0])
                print(s.spec.x()[np.where(s.bad_mask.x())[0]])
                print(np.searchsorted(x, s.spec.x()[np.where(s.bad_mask.x())[0]]))
                e_comb[i][np.searchsorted(x, s.spec.x()[np.where(s.bad_mask.x())[0]])] = np.nan

        print(comb, e_comb)

        typ = self.selectcombtype.currentText()
        print(typ)
        if typ == 'Median':
            y = np.nanmedian(comb, axis=0)
            err = np.power(np.nansum(np.power(e_comb, -2), axis=0), -0.5)

        if typ == 'Mean':
            y = np.nanmean(comb, axis=0)
            err = np.power(np.nansum(np.power(e_comb, -2), axis=0), -0.5)

        if typ == 'Weighted mean':
            w = np.power(e_comb, -2)
            y = np.nansum(comb * w, axis=0) / np.nansum(w, axis=0)
            err = np.power(np.nansum(w, axis=0), -0.5)
            #err = np.power(np.nansum(np.power(e_comb, -2), axis=0), -0.5)

        mask = np.logical_not(np.isnan(y))
        x, y, err = x[mask], y[mask], err[mask]
        # add combined spectrum to GU
        print(x, y, err)
        self.parent.s.append(Spectrum(self.parent, name='combined_'+typ.split()[0].lower()))
        self.parent.s[-1].set_data([x, y, err])
        self.parent.s.setSpec(new=True)


class rebinWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.resize(700, 500)
        self.move(400, 100)
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())

        self.treeWidget = QTreeWidget()
        self.treeWidget.setHeaderHidden(True)
        self.addItems(self.treeWidget)
        self.treeWidget.setColumnCount(3)
        self.treeWidget.setColumnWidth(0, 200)
        
        self.expchoose = QComboBox(self)
        for s in self.parent.s:
            self.expchoose.addItem(s.filename)
        print(self.parent.s.ind)
        self.expchoose.setCurrentIndex(self.parent.s.ind)
        self.expchoose.currentIndexChanged.connect(self.onExpChoose)

        self.okButton = QPushButton("Rebin")
        self.okButton.setFixedSize(70, 30)
        self.okButton.clicked[bool].connect(self.rebin)
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setFixedSize(70, 30)
        self.cancelButton.clicked[bool].connect(self.close)
        hbox = QHBoxLayout()
        hbox.addWidget(self.expchoose)
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.cancelButton)
        
        layout = QVBoxLayout()
        layout.addWidget(self.treeWidget)
        layout.addLayout(hbox)
        self.setLayout(layout)
        self.exp_ind = self.parent.s.ind
        
    def addItems(self, parent):
        self.d = {'fixednumber': 'Merge bins', 'fixedscale': 'Fixed scale', 'fixedres': 'Fixed Resolution',
             'loglinear': 'Log-linear scale', 'fromexp': 'From exposure', 'fromfile' : 'From file', 'convolve': 'Convolve'}
        for k, v in self.d.items():
            setattr(self, k+'_item', self.addParent(parent, v))
            #getattr(self, k+'_item').itemExpanded[bool].connect(partial(self.collapseAll, k))

        #self.fixednumber_item = self.addParent(parent, 'Merge bins', expanded=True)
        #self.fixedscale_item = self.addParent(parent, 'Fixed scale')
        #self.fixedres_item = self.addParent(parent, 'Fixed Resolution')
        #self.loglinear_item = self.addParent(parent, 'Log-linear scale')
        #self.fromfile_item = self.addParent(parent, 'From file')
        #self.convolve_item = self.addParent(parent, 'Convolve')
        
        self.addChild(self.fixednumber_item, 0, 'binnum', 'Bin number', 2)

        firstbin = self.parent.s[self.parent.s.ind].spec.x()[0] if len(self.parent.s) > 0 else 0
        self.addChild(self.fixedscale_item, 0, 'binsize', 'Bin size, A', 0.025)
        self.addChild(self.fixedscale_item, 1, 'zeropoint_bin', 'First bin, A', firstbin)
        
        self.addChild(self.fixedres_item, 0, 'resolution', 'Resolution', 50000)
        self.addChild(self.fixedres_item, 1, 'pp_fwhm', 'pixels per FWHM', 3)
        self.addChild(self.fixedres_item, 2, 'zeropoint_res', 'First bin, A', firstbin)
        
        self.addChild(self.loglinear_item, 0, 'binsize_log', 'step', 0.0001)
        self.addChild(self.loglinear_item, 1, 'zeropoint_log', 'First bin', np.log10(firstbin))

        self.fromexpchoose = QComboBox(self)
        for s in self.parent.s:
            self.fromexpchoose.addItem(s.filename)
        self.fromexpchoose.setCurrentIndex(self.parent.s.ind)
        item = QTreeWidgetItem(self.fromexp_item, [''])
        self.treeWidget.setItemWidget(item, 2, self.fromexpchoose)

        self.fromfile = QPushButton('Load from file', self)
        self.fromfile.clicked[bool].connect(self.loadfromfile)
        item = QTreeWidgetItem(self.fromfile_item, [''])
        ##item = QTreeWidgetItem(self.fromfile, )
        self.treeWidget.setItemWidget(item, 1, self.fromfile)

        self.addChild(self.convolve_item, 0, 'resol', 'Resolution', 50000)
        self.addChild(self.convolve_item, 1, 'res_b', 'FWHM [km/s]', 6)

        self.resol.textEdited.connect(partial(self.setResolution, 'resol'))
        self.res_b.textEdited.connect(partial(self.setResolution, 'res_b'))

        self.treeWidget.itemExpanded.connect(self.collapseAll)
        
    def addParent(self, parent, text, checkable=False, expanded=False):
        item = QTreeWidgetItem(parent, [text])
        if checkable:
            item.setCheckState(0, Qt.CheckState.Unchecked)
        else:
            #item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
        item.setExpanded(expanded)
        return item
        
    def addChild(self, parent, column, name, title, data):
        item = QTreeWidgetItem(parent, [title])
        setattr(self, name, QLineEdit())
        self.treeWidget.setItemWidget(item, 1, getattr(self, name))
        getattr(self, name).setText(str(data))
        #item.setData(1, Qt.UserRole, data)
        return item

    def collapseAll(self, excl):
        for k, v in self.d.items():
            if getattr(self, k+'_item') is not excl:
                self.treeWidget.collapseItem(getattr(self, k+'_item'))

    def setResolution(self, item):
        print(item)
        if item == 'resol':
            self.res_b.setText('{:.3f}'.format(299792.45 / float(self.resol.text())))
        elif item == 'res_b':
            self.resol.setText('{:.1f}'.format(299792.45 / float(self.res_b.text())))

    def loadfromfile(self):
        fname = QFileDialog.getOpenFileName(self, 'Import instrument function', '')

        if fname[0]:
            self.parent.instr_function = np.genfromfile(fname[0], unpack=True)
            
    def onExpChoose(self, index):
        self.exp_ind = index
        firstbin = self.parent.s[self.exp_ind].spec.x()[0]
        self.zeropoint_bin.setText(str(firstbin))
        self.zeropoint_res.setText(str(firstbin))
        self.zeropoint_log.setText(str(np.log10(firstbin)))

    def rebin_arr(self, a, factor):
        n = a.shape[0] // factor
        return a[:n*factor].reshape(a.shape[0] // factor, factor).sum(1)/factor 
    
    def rebin_err(self, a, factor):
        n = a.shape[0] // factor
        a = np.power(a[:n*factor].reshape(a.shape[0] // factor, factor), 2)
        return np.sqrt(a.sum(1))

    def rebin(self):
        if self.fixednumber_item.isExpanded():
            n = int(self.binnum.text())
            x = self.rebin_arr(self.parent.s[self.exp_ind].spec.x(), n)
            y = self.rebin_arr(self.parent.s[self.exp_ind].spec.y(), n)
            err = y / self.rebin_err(self.parent.s[self.exp_ind].spec.y()/self.parent.s[self.exp_ind].spec.err(), n)
            
            self.parent.s.append(Spectrum(self.parent, name='rebinned '+str(self.exp_ind+1), data=[x, y, err]))

        elif self.fixedscale_item.isExpanded():
            zero, binsize = float(self.zeropoint_bin.text()), float(self.binsize.text())
            num = int((self.parent.s[self.exp_ind].spec.x()[-1] - zero) / binsize)
            x = np.linspace(zero, zero + (num+1) * binsize, num)
            x = x[(x - binsize > self.parent.s[self.exp_ind].spec.raw.x[0]) * (x + binsize < self.parent.s[self.exp_ind].spec.raw.x[-1])]
            if self.parent.s[self.exp_ind].spec.raw.x[0] > x[0] - binsize or self.parent.s[self.exp_ind].spec.raw.x[-1] < x[-1] + binsize:
                self.parent.sendMessage('New wavelenght scale beyond the initial spectrum range. Please select appropriate zero point')
            else:
                y, err = spectres.spectres(self.parent.s[self.exp_ind].spec.raw.x, self.parent.s[self.exp_ind].spec.raw.y, x,
                                           spec_errs=self.parent.s[self.exp_ind].spec.raw.err)

                self.parent.s.append(Spectrum(self.parent, name='rebinned '+str(self.exp_ind+1), data=[x, y, err]))
                self.parent.s[-1].set_resolution(np.median(self.parent.s[-1].spec.raw.x)/(float(self.binsize.text()) * 2.5))

        elif self.fixedres_item.isExpanded():
            print('fixed res')
            zero = np.log10(float(self.zeropoint_res.text()))
            step = np.log10(1 + 1 / float(self.resolution.text()) / float(self.pp_fwhm.text()))
            num = int((np.log10(self.parent.s[self.exp_ind].spec.x()[-1]) - zero) / step)
            x = np.logspace(zero, zero+step*(num-1), num)
            x = x[(x - (x[1] - x[0]) > self.parent.s[self.exp_ind].spec.raw.x[0]) * (x + (x[-1] - x[-2]) < self.parent.s[self.exp_ind].spec.raw.x[-1])]
            if self.parent.s[self.exp_ind].spec.raw.x[0] > x[0] - (x[1] - x[0]) / 2 or self.parent.s[self.exp_ind].spec.raw.x[-1] < x[-1] + (x[-1] - x[-2]) / 2:
                self.parent.sendMessage('New wavelenght scale beyond the initial spectrum range. Please select appropriate zero point')
            else:
                y, err = spectres.spectres(self.parent.s[self.exp_ind].spec.raw.x, self.parent.s[self.exp_ind].spec.raw.y, x,
                                           spec_errs=self.parent.s[self.exp_ind].spec.raw.err)

                self.parent.s.append(Spectrum(self.parent, name='rebinned '+str(self.exp_ind+1)))
                self.parent.s[-1].set_data([x, y, err])
                self.parent.s[-1].set_resolution(float(self.resolution.text()))

        elif self.loglinear_item.isExpanded():
            print('loglinear')

        elif self.fromexp_item.isExpanded():

            ind = self.fromexpchoose.currentIndex()
            lmin = np.max([self.parent.s[ind].spec.raw.x[0], self.parent.s[self.exp_ind].spec.raw.x[0]])
            lmax = np.min([self.parent.s[ind].spec.raw.x[-1], self.parent.s[self.exp_ind].spec.raw.x[-1]])
            mask = np.logical_and(self.parent.s[self.exp_ind].spec.raw.x >= lmin, self.parent.s[self.exp_ind].spec.raw.x <= lmax)
            mask_r = np.logical_and(self.parent.s[ind].spec.raw.x >= self.parent.s[self.exp_ind].spec.raw.x[mask][0],
                                    self.parent.s[ind].spec.raw.x <= self.parent.s[self.exp_ind].spec.raw.x[mask][-1])
            x = self.parent.s[ind].spec.raw.x[mask_r]
            x = x[(x - (x[1] - x[0]) > self.parent.s[self.exp_ind].spec.raw.x[0]) * (x + (x[-1] - x[-2]) < self.parent.s[self.exp_ind].spec.raw.x[-1])]
            if self.parent.s[self.exp_ind].spec.raw.x[0] > x[0] - (x[1] - x[0]) / 2 or self.parent.s[self.exp_ind].spec.raw.x[-1] < x[-1] + (x[-1] - x[-2]) / 2:
                self.parent.sendMessage('New wavelenght scale beyond the initial spectrum range. Please select appropriate zero point')
            else:
                y, err = spectres.spectres(self.parent.s[self.exp_ind].spec.raw.x[mask], self.parent.s[self.exp_ind].spec.raw.y[mask],
                                           x, spec_errs=self.parent.s[self.exp_ind].spec.raw.err[mask])
                self.parent.s.append(Spectrum(self.parent, name='rebinned '+str(self.exp_ind+1)))
                self.parent.s[-1].set_data([x, y, err])

        elif self.fromfile_item.isExpanded():
            print('from file')

        elif self.convolve_item.isExpanded():
            x = self.parent.s[self.exp_ind].spec.x()
            print(float(self.resol.text()))
            self.parent.s.append(Spectrum(self.parent, name='convolved ' + str(self.exp_ind + 1)))
            y = convolveflux(x, self.parent.s[self.exp_ind].spec.y(), res=float(self.resol.text()), kind='direct')
            if self.parent.s[self.exp_ind].spec.raw.err is not None and self.parent.s[self.exp_ind].spec.raw.err.shape[0] == x.shape[0]:
                err = convolveflux(x, self.parent.s[self.exp_ind].spec.err(), res=float(self.resol.text()), kind='direct')
                self.parent.s[-1].set_data([x, y, err])
            else:
                self.parent.s[-1].set_data([x, y])
            if self.parent.s[self.exp_ind].resolution_linear[0] not in [0, None]:
                self.parent.s[-1].set_resolution(1 / np.sqrt(1 / float(self.resol.text())**2 + 1 / self.parent.s[self.exp_ind].resolution()**2))
            else:
                self.parent.s[-1].set_resolution(float(self.resol.text()))

        self.parent.s.redraw()
        self.parent.s[-1].specClicked()
        self.close()

class GenerateAbsWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setGeometry(300, 200, 500, 600)
        self.setWindowTitle('Generate Absorption System:')
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())
        self.initData()
        self.initGUI()

    def initData(self):
        self.opts = {'gen_template': str, 'gen_z': float, 'gen_xmin': float, 'gen_xmax': float,
                     'gen_resolution': float, 'gen_lyaforest': float, 'gen_Av': float, 'gen_z_Av': float,
                     'gen_snr': float, 'gen_num': int, 'gen_tau': float
                     }
        for opt, func in self.opts.items():
            # print(opt, self.parent.options(opt), func(self.parent.options(opt)))
            setattr(self, opt, func(self.parent.options(opt)))

    def initGUI(self):
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Orientation.Vertical)
        grid = QGridLayout()

        validator = QDoubleValidator()
        locale = QLocale('C')
        validator.setLocale(locale)
        # validator.ScientificNotation
        names = ['Template:', '', '', '',
                 'z:', '', '', '',
                 'x_min:', '', 'x_max:', '',
                 'resolution:', '', '', '',
                 'Lya-forest:', '', '', '',
                 'Extinction: Av:', '', 'z_Av:', '',
                 'noize', '', 'SNR:', ''
                 ]
        positions = [(i, j) for i in range(7) for j in range(4)]

        for position, name in zip(positions, names):
            if name == '':
                continue
            grid.addWidget(QLabel(name), *position)

        self.opt_but = OrderedDict([('gen_z', [1, 1]), ('gen_xmin', [2, 1]), ('gen_xmax', [2, 3]),
                                    ('gen_resolution', [3, 1]), ('gen_lyaforest', [4, 1]),
                                    ('gen_Av', [5, 1]), ('gen_z_Av', [5, 3]),
                                    ('gen_snr', [6, 3])
                                    ])
        for opt, v in self.opt_but.items():
            b = QLineEdit(str(getattr(self, opt)))
            b.setFixedSize(100, 30)
            b.setValidator(validator)
            b.textChanged[str].connect(partial(self.onChanged, attr=opt))
            grid.addWidget(b, v[0], v[1])

        self.template = QComboBox(self)
        self.templist = ['Selsing', 'VanDenBerk', 'HST', 'const', 'spectrum']
        self.template.addItems(self.templist)
        ind = self.templist.index(self.gen_template) if self.gen_template in self.templist else 0
        self.template.setCurrentIndex(ind)
        self.template.currentTextChanged.connect(self.onChanged)
        grid.addWidget(self.template, 0, 1)

        self.snr = QCheckBox('SNR')
        self.snr.setChecked(True)
        grid.addWidget(self.snr, 6, 2)

        widget = QWidget()
        widget.setLayout(grid)
        splitter.addWidget(widget)

        grid_boot = QGridLayout()
        names = ['num:', '',
                 'tau limit:', '',
                 ]
        positions = [(i, j) for i in range(7) for j in range(2)]

        for position, name in zip(positions, names):
            if name == '':
                continue
            grid_boot.addWidget(QLabel(name), *position)

        self.opt_but = OrderedDict([('gen_num', [0, 1]), ('gen_tau', [1, 1])
                                    ])
        for opt, v in self.opt_but.items():
            b = QLineEdit(str(getattr(self, opt)))
            b.setFixedSize(100, 30)
            b.setValidator(validator)
            b.textChanged[str].connect(partial(self.onChanged, attr=opt))
            grid_boot.addWidget(b, v[0], v[1])

        widget = QWidget()
        widget.setLayout(grid_boot)
        splitter.addWidget(widget)
        splitter.setSizes([700, 500])

        layout.addWidget(splitter)

        layout.addStretch(1)

        l = QHBoxLayout()
        self.bootButton = QPushButton("Boot")
        self.bootButton.clicked[bool].connect(self.boot)
        self.bootButton.setFixedSize(100, 30)
        self.showButton = QPushButton("Show")
        self.showButton.clicked[bool].connect(self.showBoot)
        self.showButton.setFixedSize(100, 30)
        self.okButton = QPushButton("Generate")
        self.okButton.clicked[bool].connect(self.generate)
        self.okButton.setFixedSize(100, 30)
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked[bool].connect(self.close)
        self.cancelButton.setFixedSize(100, 30)
        l.addWidget(self.bootButton)
        l.addWidget(self.showButton)
        l.addStretch(1)
        l.addWidget(self.okButton)
        l.addWidget(self.cancelButton)
        layout.addLayout(l)

        self.setLayout(layout)

        self.show()

    def onChanged(self, text, attr=None):
        if text in self.templist:
            attr = 'gen_template'
        if attr is not None:
            setattr(self, attr, self.opts[attr](text))
        self.parent.options(attr, self.opts[attr](text))

    def generate(self):
        snr = self.gen_snr if self.snr.isChecked() else None
        if self.parent.normview:
            self.parent.normalize()
        self.parent.generate(template=self.template.currentText(), z=self.gen_z,
                             xmin=self.gen_xmin, xmax=self.gen_xmax,
                             resolution=self.gen_resolution, snr=snr, lyaforest=self.gen_lyaforest,
                             Av=self.gen_Av, z_Av=self.gen_z_Av)

        self.close()

    def boot(self):
        if 1:
            res, lnprobs = [], []
        else:
            with open('temp/res.pickle', 'rb') as f:
                res = pickle.load(f)
        snr = self.gen_snr if self.snr.isChecked() else None

        self.parent.fit.save()

        for i in range(self.gen_num):
            print(i)
            self.parent.normalize(False)
            if self.parent.s.ind is not None:
                self.parent.s.remove(self.parent.s.ind)
            self.parent.fit.load()
            self.parent.generate(template='const', z=self.gen_z, fit=True,
                                 xmin=self.gen_xmin, xmax=self.gen_xmax,
                                 resolution=self.gen_resolution, snr=snr,
                                 lyaforest=self.gen_lyaforest, lycutoff=False, Av=self.gen_Av, z_Av=self.gen_z_Av, redraw=True)
            self.parent.normalize(True)
            if i == 0:
                self.parent.s.prepareFit(all=True)
                self.parent.s.calcFit(redraw=False)
                s = self.parent.s[self.parent.s.ind]
                m = s.fit.y() < np.exp(-self.gen_tau)
                m[0] = False
                m[-1] = False
                print(np.sum(m))
                regions = []
                regs = np.argwhere(np.abs(np.diff(m)))
                for r in range(0, len(regs), 2):
                    xmin, xmax = s.fit.x()[regs[r]], s.fit.x()[regs[r + 1]]
                    for line in self.parent.abs.activelist:
                        if xmin < line.line.l() * (1 + self.parent.fit.sys[0].z.val) < xmax:
                            regions.append([xmin, xmax])

            print(regions)
            s = self.parent.s[self.parent.s.ind]
            for r in regions:
                s.add_points(r[0], 1.5, r[1], -0.5, remove=False, redraw=False)
            s.set_fit_mask()
            self.parent.s.prepareFit(all=False)
            #self.parent.s.calcFit(redraw=False)

            self.parent.fit.shake()
            self.parent.fit.update()

            if 1:
                x, unc = self.parent.julia.fitLM(self.parent.julia_spec, self.parent.fit.list(), self.parent.julia_add,
                                                 method=self.parent.options("fit_method"),
                                                 tieds=self.parent.fit.tieds,
                                                 regular=int(self.parent.options("julia_grid")),
                                                 telluric=self.parent.options("telluric"),
                                                 tau_limit=self.parent.tau_limit,
                                                 accuracy=self.parent.accuracy,
                                                 toll=self.parent.options("fit_tolerance"))
                res.append(x)
                self.parent.s.calcFit(redraw=False)
                lnprobs.append(self.parent.s.chi2())
            else:
                self.parent.fitAbs(timer=False)
                self.parent.fit.fromJulia(res, unc)
                res.append([self.parent.fit.getValue(str(p)) for p in self.parent.fit.list_fit()])

        with open('temp/res.pickle', 'wb') as f:
            pickle.dump([str(p) for p in self.parent.fit.list_fit()], f)
            pickle.dump(res, f)
            pickle.dump(lnprobs, f)

        self.parent.fit.load()
        self.parent.s.redraw()

    def showBoot(self):
        with open('temp/res.pickle', 'rb') as f:
            names = [name.replace('_', ' ') for name in pickle.load(f)]
            res = np.asarray(pickle.load(f))
            lnprobs = np.asarray(pickle.load(f))
            print(res, lnprobs)

        if 0:
            c = ChainConsumer()
            c.add_chain(res, parameters=names)
            c.configure(smooth=True,
                        colors=['g'],
                        # cmap='Reds',
                        # marker_size=2,
                        cloud=True,
                        shade=True,
                        sigmas=[0, 1, 2, 3],
                        )
            c.configure_truth(ls='--', lw=1., c='black')  # c='darkorange')

            fig = c.plotter.plot(figsize=(20, 20),
                                 # filename="output/fit.png",
                                 display=True,
                                 truth=[self.parent.fit.getValue(str(p)) for p in self.parent.fit.list_fit()],
                                 )
        else:
            print(res.shape)
            print(res.reshape(-1, res.shape[-1]))
            fig = corner.corner(res.reshape(-1, res.shape[-1]),
                                labels=names,
                                show_titles=True,
                                plot_contours=True,
                                truths=[self.parent.fit.getValue(str(p)) for p in self.parent.fit.list_fit()],
                                )
            plt.show()

    def closeEvent(self, ev):
        for opt, func in self.opts.items():
            self.parent.options(opt, func(getattr(self, opt)))
        ev.accept()

class infoWidget(QWidget):
    def __init__(self, parent, title, file=None, text=None):
        super().__init__()
        self.parent = parent
        self.title = title
        self.file = file
        self.setWindowTitle(title)
        self.resize(700, 500)
        self.move(400, 100)

        layout = QVBoxLayout()
        self.text = QTextEdit()

        self.loadtxt(text=text)
        #self.text.setUpdatesEnabled(False)
        #self.text.setFixedSize(400, 300)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        self.okButton = QPushButton("Ok")
        self.okButton.setFixedSize(60, 30)
        self.okButton.clicked[bool].connect(self.close)
        hbox.addWidget(self.okButton)
        layout.addWidget(self.text)
        layout.addLayout(hbox)
        self.setLayout(layout)
        self.setStyleSheet(open(self.parent.folder + 'config/styles.ini').read())

    def loadtxt(self, text=None):
        if text is None:
            with open(self.file) as f:
                text = f.read()

        self.text.setText(text)


class messageWindow(QWidget):
    def __init__(self, parent, text='', timer=2000):
        super().__init__(parent=None)
        # make the window frameless
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.parent = parent
        self.text = text

        if 0:
            l = QHBoxLayout()
            l.addWidget(QLabel(text))
            self.setLayout(l)

        self.pos = [QApplication.screens()[0].geometry().width() / 2, QApplication.screens()[0].geometry().height() / 2]
        self.resize(400 + len(self.text) * 12, 50)
        self.move(int(self.pos[0] - self.size().width() / 2), int(self.pos[1] - self.size().height() / 2))

        self.fillColor = QColor(229, 23, 80, 120) #QColor(83, 148, 83, 105)
        self.penColor = QColor(70, 70, 70, 255)

        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.setDuration(timer)
        self.animation.finished.connect(self._onclose)
        self.animStarted = False

        if timer not in [None, 0]:
            QTimer.singleShot(timer, self._onclose)

        self.show()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        qp.setPen(self.penColor)
        qp.setBrush(self.fillColor)
        qp.drawRoundedRect(0, 0, self.size().width(), self.size().height(), 5, 5)

        font = QFont()
        font.setPixelSize(22)
        font.setBold(True)
        qp.setFont(font)
        qp.setPen(QColor(200, 200, 200))
        qp.drawText(int(self.size().width() / 2 - len(self.text) * 5), int(self.size().height() / 2 + 8), self.text)
        qp.end()

    def _onclose(self):

        if not self.animStarted:
            self.animation.start()
            self.animStarted = True
        #    event.ignore()
        else:
            #QWidget.closeEvent(self, event)
            self.hide()
            self.close()
            self.parent.message = None

class buttonpanel(QFrame):
    def __init__(self, parent):
        super().__init__()
        
        self.parent = parent
        self.initUI()

    def initUI(self):
        # >>> Redshift button
        lbl = QLabel('z =', self)
        lbl.move(30, 25)

        self.z_panel = QLineEdit(self)
        self.z_panel.move(55, 20)
        self.z_panel.textChanged[str].connect(self.zChanged)
        self.z_panel.returnPressed.connect(partial(self.zChanged, text=None))
        self.z_panel.setMaxLength(12)
        self.z_panel.resize(120, 30)
        validator = QDoubleValidator()
        validator.setLocale(QLocale('C'))
        self.z_panel.setValidator(validator)

        # >>> Normalized button
        self.normalize = QPushButton('Normalize', self)
        self.normalize.setCheckable(True)
        self.normalize.clicked[bool].connect(self.parent.normalize)
        self.normalize.move(180, 20)
        self.normalize.resize(110, 30)

        self.subtract = QPushButton('Subtract', self)
        self.subtract.setCheckable(True)
        self.subtract.clicked[bool].connect(partial(self.parent.normalize, action='subtract'))
        self.subtract.move(180, 60)
        self.subtract.resize(110, 30)

        self.fitbutton = QPushButton('Fit', self)
        self.fitbutton.setCheckable(True)
        self.fitbutton.clicked.connect(self.parent.fitLM)
        self.fitbutton.setStyleSheet('QPushButton::checked { background-color: rgb(168,66,195);}')
        self.fitbutton.move(300, 20)
        self.fitbutton.resize(70, 30)

        self.aod = QPushButton('AOD', self)
        self.aod.setCheckable(True)
        self.aod.clicked.connect(self.parent.aod) #partial(self.parent.normalize, action='aod'))
        self.aod.setStyleSheet('QPushButton::checked { background-color: rgb(168,66,195);}')
        self.aod.move(300, 60)
        self.aod.resize(70, 30)

        self.SAS = QPushButton('SAS', self)
        self.SAS.clicked.connect(partial(self.openURL, 'SAS'))
        self.SAS.move(450, 20)
        self.SAS.resize(70, 30)

        self.SkyS = QPushButton('SkyS', self)
        self.SkyS.clicked.connect(partial(self.openURL, 'SkyS'))
        self.SkyS.move(530, 20)
        self.SkyS.resize(70, 30)

        self.ESO = QPushButton('ESO', self)
        self.ESO.clicked.connect(self.getESO)
        self.ESO.move(450, 60)

        self.ESO.resize(70, 30)

        self.NIST = QPushButton('NIST', self)
        self.NIST.clicked.connect(self.getNIST)
        self.NIST.move(630, 20)
        self.NIST.resize(70, 30)

    def initStyle(self):
        self.setStyleSheet("""
            QFrame {
                border: 1px solid  #000;
            }
        """)

    def refresh(self):
        self.z_panel.setText("{:20.10f}".format(self.parent.z_abs).strip().rstrip("0"))
    
    def zChanged(self, text=None):
        if text is None:
            text = self.z_panel.text()
        self.parent.z_abs = float(text)
        try:
            if self.parent.plot.restframe:
                self.parent.plot.updateVelocityAxis()
            self.parent.abs.redraw()
        except:
            pass

    def openURL(self, typ):
        id = getIDfromName(self.parent.s[self.parent.s.ind].filename)
        print(id)
        if typ == 'SAS':
            url = QUrl('https://dr17.sdss.org/optical/spectrum/view?plateid={0:d}&mjd={1:d}&fiber={2:d}&run2d=any&zwarning=0&matches=any'.format(id[0], id[1], id[2]))
        elif typ == 'SkyS':
            url = QUrl('http://skyserver.sdss.org/dr17/en/tools/explore/Summary.aspx?plate={0:d}&fiber={1:d}&mjd={2:d}'.format(id[0], id[2], id[1]))
        if not QDesktopServices.openUrl(url):
            QMessageBox.warning(self, 'Open Url', 'Could not open url')

    def getESO(self):
        print(self.parent.s[self.parent.s.ind].filename)

    def getNIST(self):
        for r in self.parent.plot.regions:
            print(r.getRegion()[0], r.getRegion()[1])
            url = QUrl("https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=&limits_type=0&low_w={0:.2f}&upp_w={1:.2f}&unit=0&submit=Retrieve+Data&de=0&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on".format(r.getRegion()[0]/(1+self.parent.z_abs), r.getRegion()[1]/(1+self.parent.z_abs)))
            if not QDesktopServices.openUrl(url):
                QMessageBox.warning(self, 'Open Url', 'Could not open url')

class sviewer(QMainWindow):
    
    def __init__(self):
        super().__init__()

        #self.setWindowFlags(Qt.FramelessWindowHint)
        self.initStatus()
        self.setScreen()
        self.initUI()
        self.initStyles()

        self.initData()
        self.setWindowTitle('Spectro')
        self.setWindowIcon(QIcon(self.folder + 'images/spectro_logo.png'))

    def setScreen(self):
        screens = QApplication.screens()
        ind = int(self.options("display"))
        #for s in screens:
        #    print(s.geometry())
        if ind < len(screens):
            screen = screens[ind]
            self.move(screen.geometry().left(), screen.geometry().top())
            self.resize(screen.geometry().width(), screen.geometry().height())
            #print(self.fullscreen)
            #action = self.showFullScreen() if self.fullscreen else self.showMaximized()

    def initStyles(self):
        self.setStyleSheet(open(self.folder + 'config/styles.ini').read())

    def initStatus(self):
        self.folder = os.path.dirname(os.path.abspath(__file__)) + '/'
        QDir.addSearchPath('images', os.path.join(self.folder, 'images'))
        print(self.folder)
        self.t = Timer()
        self.setAcceptDrops(True)
        self.abs_H2_status = 0
        self.abs_DLA_status = 0
        self.abs_DLAmajor_status = 1
        self.abs_LowIoniz_status = 0
        self.abs_Molec_status = 0
        self.abs_SF_status = 1
        self.normview = False
        self.aodview = False
        if platform.system() == 'Windows':
            self.config = 'config/options.ini'
        elif platform.system() == 'Linux':
            self.config = 'config/options_linux.ini'
        self.developer = os.path.isfile(self.folder + 'config/developer.ini')
        self.blindMode = self.options('blindMode')
        self.SDSSfolder = self.options('SDSSfolder', config=self.config)
        self.SDSSDR14 = self.options('SDSSDR14', config=self.config)
        if self.SDSSDR14 is not None and os.path.isfile(self.SDSSDR14):
            self.SDSSDR14 = h5py.File(self.SDSSDR14, 'r')
        self.SDSSLeefolder = self.options('SDSSLeefolder', config=self.config)
        self.SDSSdata = []
        self.SDSSquery = None
        self.filters_status = {'SDSS': 0, 'Gaia': 0, '2MASS': 0, 'VISTA':0, 'UKIDSS': 0, 'WISE': 0, 'GALEX': 0}
        self.filters = {'SDSS': None, 'Gaia': None, '2MASS': None, 'VISTA': None, 'UKIDSS': None, 'WISE': None, 'GALEX': None}
        self.photo = None
        self.UVESSetup_status = False
        self.XQ100folder = self.options('XQ100folder', config=self.config)
        self.P94folder = self.options('P94folder', config=self.config)
        self.work_folder = self.options('work_folder', config=self.config)
        self.plot_set_folder = self.options('plot_set_folder', config=self.config)
        self.VandelsFile = self.options('VandelsFile', config=self.config)
        self.KodiaqFile = self.options('KodiaqFile', config=self.config)
        self.UVESfolder = self.options('UVESfolder', config=self.config)
        self.ErositaFile = self.options('ErositaFile', config=self.config)
        self.MALSfolder = self.options('MALSfolder', config=self.config)
        self.IGMspecFile = self.options('IGMspecFile', config=self.config)
        self.SFDMapPath = self.options('SFDMapPath', config=self.config)
        self.MCMC_output = 'output/mcmc.hdf5'
        self.z_abs = 0
        self.lines = lineList(self)
        #self.line_reper = line('HI', 1215.6701, 0.4164, 6.265e8, ref='???')
        self.regions = []
        self.show_residuals = self.options('show_residuals')
        self.show_2d = self.options('show_2d')
        self.bary, self.heli = 0, 0
        self.save_opt = ['cont', 'points', 'fit', 'others', 'fit_results']
        self.export_opt = ['cont', 'fit']
        self.export2d_opt = ['spectrum', 'err', 'mask', 'cr', 'sky', 'trace']
        self.num_between = int(self.options('num_between'))
        self.tau_limit = float(self.options('tau_limit'))
        self.accuracy = float(self.options('accuracy'))
        self.message = None
        self.julia = None
        if self.options("fitType") == "julia":
            if self.reload_julia() == False:
                self.options('fitType', 'uniform')
        # self.specview sets the type of plot representation
        for l in ['specview', 'selectview', 'linelabels', 'showinactive', 'show_osc', 'fitType', 'fitComp', 'fitview', 'comp_view', 'animateFit', 'fit_method']:
            setattr(self, l, self.options(l))
        self.polyDeg = int(self.options('polyDeg'))
        self.SDSScat = self.options('SDSScat')
        self.comp = 0
        self.fitprocess = None
        self.fitModel = None
        self.chooseFit = None
        self.preferences = None
        self.preferencesTabIndex = 0
        self.observability = None
        self.exp = None
        self.fitResults = None
        self.fitres = None
        self.showlines = None
        self.MCMC = None
        self.extract2dwindow = None
        self.fitContWindow = None
        self.rescale_ind = 0
        self.compositeQSO_status = False
        self.compositeGal_status = False
        self.ErositaWidget = None
        self.fullscreen = bool(self.options('fullscreen'))
        # this is to set the spectro_logo in the taskbar for Windows
        myapp = u'spectro.0.8'
        if platform.system() == 'Windows':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myapp)

    def initUI(self):
        #dbg = pg.dbg()
        # >>> create panel for plotting spectra
        self.plot = plotSpectrum(self)
        self.vb = self.plot.getPlotItem().getViewBox()
        self.s = Speclist(self)
        #self.plot.setFrameShape(QFrame.Shape.StyledPanel)
        
        self.panel = buttonpanel(self)
        self.panel.setFrameShape(QFrame.Shape.StyledPanel)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter_plot = QSplitter(Qt.Orientation.Vertical)
        self.splitter_plot.addWidget(self.plot)
        self.splitter_fit = QSplitter(Qt.Orientation.Horizontal)
        self.splitter_fit.addWidget(self.splitter_plot)
        self.splitter.addWidget(self.splitter_fit)

        splitter_2 = QSplitter(Qt.Orientation.Horizontal)
        splitter_2.addWidget(self.panel)
        self.console = Console(self)
        splitter_2.addWidget(self.console)
        splitter_2.setSizes([500, 500])
        self.splitter.addWidget(splitter_2)
        self.splitter.setSizes([1900, 100])

        self.setCentralWidget(self.splitter)
        
        # >>> create Menu
        self.initMenu()
        
        # create toolbar
        #self.toolbar = self.addToolBar('B-spline')
        #self.toolbar.addAction(Bspline)
        
        # >>> create status bar
        self.statusBarWidget = QStatusBar()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setStatusBar(self.statusBarWidget)
        self.statusBar = QLabel()
        splitter.addWidget(self.statusBar)
        self.componentBar = QLabel('')
        splitter.addWidget(self.componentBar)
        self.chiSquare = QLabel('')
        splitter.addWidget(self.chiSquare)
        self.MCMCprogress = QLabel('')
        splitter.addWidget(self.MCMCprogress)
        splitter.setSizes([1500, 200, 500, 500])
        splitter.setStyleSheet(open(self.folder + 'config/styles.ini').read() + "QSplitter::handle:horizontal {height: 1px; background: rgb(49,49,49);}")
        self.statusBarWidget.addWidget(splitter)

        self.statusBar.setText('Ready')

        self.draw()
        self.showMaximized()
        self.show()
         
    def initMenu(self):
        
        menubar = self.menuBar()
        #menubar.setIconSize(QSize(40, 20))
        menubar.setFixedHeight(37)
        #menubar.setNativeMenuBar(self)
        fileMenu = menubar.addMenu('&File')
        viewMenu = menubar.addMenu('&View')
        linesMenu = menubar.addMenu('&Lines')
        fitMenu = menubar.addMenu('&Fit')
        spec1dMenu = menubar.addMenu('&1d spec')
        spec2dMenu = menubar.addMenu('&2d spec')
        combineMenu = menubar.addMenu('&Combine')
        SDSSMenu = menubar.addMenu('&SDSS')
        if self.developer:
            samplesMenu = menubar.addMenu('&Samples')
        generateMenu = menubar.addMenu('&Generate')
        obsMenu = menubar.addMenu('&Observations')
        helpMenu = menubar.addMenu('&Help')
        
        # >>> create File Menu items
        # >>> create File Menu items
        clearAction = QAction('&Clear', self)
        clearAction.setStatusTip('Clear current session')
        clearAction.triggered.connect(self.clearSession)

        openAction = QAction('&Open...', self)
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open file')
        openAction.triggered.connect(self.showOpenDialog)
        
        saveAction = QAction('&Save', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save file')
        saveAction.triggered.connect(self.saveFilePressed)

        saveAsAction = QAction('&Save as...', self)
        saveAsAction.setStatusTip('Save file as')
        saveAsAction.triggered.connect(self.showSaveDialog)

        importAction = QAction('&Import spectrum...', self)
        importAction.setShortcut('Ctrl+I')
        importAction.setStatusTip('Import spectrum')
        importAction.triggered.connect(self.showImportDialog)

        importTelluric = QAction('&Import telluric/sky...', self)
        importTelluric.setStatusTip('Import telluric/sky/accompanying absorption')
        importTelluric.triggered.connect(self.showImportTelluricDialog)

        import2dAction = QAction('&Import 2d spectrum...', self)
        import2dAction.setStatusTip('Import 2d spectrum')
        import2dAction.triggered.connect(self.show2dImportDialog)

        importDispAction = QAction('&Import Profile Dispersion...', self)
        importDispAction.setStatusTip('Import the fit profile in a view of the confidence bands from Bayessian inference (obtained through MCMC widget)')
        importDispAction.triggered.connect(self.showDispImportDialog)

        exportAction = QAction('&Export spectrum...', self)
        exportAction.setStatusTip('Export spectrum')
        exportAction.triggered.connect(self.showExportDialog)

        export2dAction = QAction('&Export 2d spectrum...', self)
        export2dAction.setStatusTip('Export 2d spectrum')
        export2dAction.triggered.connect(self.show2dExportDialog)

        exportDataAction = QAction('&Export data...', self)
        exportDataAction.setStatusTip('Export data')
        exportDataAction.triggered.connect(self.showExportDataDialog)
        
        importList = QAction('&Import List...', self)
        importList.setStatusTip('Import list of spectra')
        importList.triggered.connect(self.showImportListDialog)

        importFolder = QAction('&Import Folder...', self)
        importFolder.setStatusTip('Import list of spectra from folder')
        importFolder.triggered.connect(self.showImportFolderDialog)

        exitAction = QAction('&Exit', self)
        #exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QApplication.instance().quit)

        fileMenu.addAction(clearAction)
        fileMenu.addSeparator()
        fileMenu.addAction(openAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(saveAsAction)
        fileMenu.addSeparator()
        fileMenu.addAction(importAction)
        fileMenu.addAction(importTelluric)
        fileMenu.addAction(import2dAction)
        fileMenu.addAction(importDispAction)
        fileMenu.addAction(importList)
        fileMenu.addAction(importFolder)
        fileMenu.addSeparator()
        fileMenu.addAction(exportAction)
        fileMenu.addAction(export2dAction)
        fileMenu.addAction(exportDataAction)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)
        
        # >>> create View Menu items

        exp = QAction('&Exposures', self)
        exp.setShortcut('F2')
        exp.setStatusTip('Show list of exposures')
        exp.triggered.connect(self.showExpList)

        self.showResiduals = QAction('&Residuals', self)
        self.showResiduals.setShortcut('F8')
        self.showResiduals.setStatusTip('Show/Hide Residuals panel')
        self.showResiduals.triggered.connect(partial(self.showResidualsPanel, show=None))
        self.showResidualsPanel(self.show_residuals)

        self.show2d = QAction('&2d spectrum', self)
        self.show2d.setShortcut('F9')
        self.show2d.setStatusTip('Show/Hide 2d spectrum panel')
        self.show2d.triggered.connect(partial(self.show2dPanel, show=None))
        self.show2dPanel(self.show_2d)

        preferences = QAction('&Preferences...', self)
        preferences.setStatusTip('Show preferences')
        preferences.setShortcut('F11')
        preferences.triggered.connect(self.showPreferences)

        showLines = QAction('&Plot line profiles', self)
        showLines.setShortcut('F7')
        showLines.setStatusTip('Plot line profiles using matplotlib')
        showLines.triggered.connect(partial(self.showLines, True))

        snapShot = QAction('&Plot snapshot', self)
        snapShot.setStatusTip('Snapshop of view using matplotlib')
        snapShot.triggered.connect(self.takeSnapShot)

        referenceAxis = QAction('&Switch reference axis', self)
        referenceAxis.setStatusTip('Switch top axis between velocity shift and restframe mode')
        referenceAxis.triggered.connect(self.switchReferenceAxis)

        self.FullScreen = QAction('&Full screen', self, checkable=True)
        self.FullScreen.setStatusTip('Switch to Fullscreen mode')
        self.FullScreen.setShortcut('F12')
        self.FullScreen.triggered.connect(self.switchFullScreen)
        self.FullScreen.setChecked(self.fullscreen)

        viewMenu.addAction(exp)
        viewMenu.addAction(self.showResiduals)
        viewMenu.addAction(self.show2d)
        viewMenu.addAction(preferences)
        viewMenu.addSeparator()
        viewMenu.addAction(showLines)
        viewMenu.addAction(snapShot)
        viewMenu.addAction(referenceAxis)
        viewMenu.addAction(self.FullScreen)

        # >>> create Line Menu items
        self.linesH2 = QAction('&H2 lines', self, checkable=True)
        self.linesH2.setStatusTip('Add H2 lines')
        self.linesH2.triggered.connect(partial(self.absLines, 'abs_H2_status'))
        self.linesH2.setChecked(self.abs_H2_status)

        self.linesDLA = QAction('&DLA', self, checkable=True)
        self.linesDLA.setStatusTip('Add extended list of DLA lines')
        self.linesDLA.triggered.connect(partial(self.absLines, 'abs_DLA_status'))
        self.linesDLA.setChecked(self.abs_DLA_status)

        self.linesDLAmajor = QAction('&DLA only major', self, checkable=True)
        self.linesDLAmajor.setStatusTip('Add list of major DLA lines')
        self.linesDLAmajor.triggered.connect(partial(self.absLines, 'abs_DLAmajor_status'))
        self.linesDLAmajor.setChecked(self.abs_DLAmajor_status)

        self.linesLowIon = QAction('&Low Ionization lines', self, checkable=True)
        self.linesLowIon.setStatusTip('Add Low Ionization lines associated with CNM/H2-bearing gas')
        self.linesLowIon.triggered.connect(partial(self.absLines, 'abs_LowIoniz_status'))
        self.linesLowIon.setChecked(self.abs_LowIoniz_status)

        self.linesMolec = QAction('&Minor molecules', self, checkable=True)
        self.linesMolec.setStatusTip('Add various molecular lines')
        self.linesMolec.triggered.connect(partial(self.absLines, 'abs_Molec_status'))
        self.linesMolec.setChecked(self.abs_Molec_status)

        self.linesSF = QAction('&Emission lines', self, checkable=True)
        self.linesSF.setStatusTip('Star-formation emission lines')
        self.linesSF.triggered.connect(partial(self.absLines, 'abs_SF_status'))
        self.linesSF.setChecked(self.abs_SF_status)

        linesChoice = QAction('&Choose lines', self)
        linesChoice.setStatusTip('Choose lines to indicate')
        linesChoice.triggered.connect(self.absChoicelines)

        hideAll = QAction('&Hide all', self)
        hideAll.setStatusTip('Remove all line indicators')
        hideAll.triggered.connect(self.hideAllLines)

        linesMenu.addAction(self.linesDLA)
        linesMenu.addAction(self.linesDLAmajor)
        linesMenu.addAction(self.linesH2)
        linesMenu.addAction(self.linesLowIon)
        linesMenu.addAction(self.linesMolec)
        linesMenu.addAction(self.linesSF)
        linesMenu.addSeparator()
        linesMenu.addAction(linesChoice)
        linesMenu.addAction(hideAll)

        # >>> create Fit Menu items
        
        setFit = QAction('&Fit model', self)
        setFit.setShortcut('F3')
        setFit.setStatusTip('set Fit model parameters')
        setFit.triggered.connect(self.setFitModel)

        chooseFitPars = QAction('&Fit parameters', self)
        chooseFitPars.setStatusTip('Choose particular fit parameters')
        chooseFitPars.setShortcut('F4')
        chooseFitPars.triggered.connect(self.chooseFitPars)

        showFit = QAction('&Show fit', self)
        showFit.setStatusTip('Show fit only near fitted points')
        showFit.setShortcut('F')
        showFit.triggered.connect(partial(self.showFit, -1, False))

        showFullFit = QAction('&Show full fit', self)
        showFullFit.setStatusTip('Show fit in all avaliable lines')
        showFullFit.setShortcut('Shift+F')
        showFullFit.triggered.connect(partial(self.showFit, -1, True))

        fitResults = QAction('&Fit results', self)
        fitResults.setStatusTip('Show fit results')
        fitResults.setShortcut('F6')
        fitResults.triggered.connect(self.showFitResults)

        fitLM = QAction('&Fit LM', self)
        fitLM.setStatusTip('Fit by Levenberg-Marquadt method')
        fitLM.triggered.connect(self.fitLM)

        fitMCMC = QAction('&Fit MCMC...', self)
        fitMCMC.setStatusTip('Fit by MCMC method')
        fitMCMC.setShortcut('F5')
        fitMCMC.triggered.connect(self.fitMCMC)

        fitGrid = QAction('&Grid fit', self)
        fitGrid.setStatusTip('Brute force calculation on the grid of parameters')
        fitGrid.triggered.connect(partial(self.profileLikelihood, num=None))

        fitCont = QAction('&Fit with cont...', self)
        fitCont.setStatusTip('Fit with cont unc')
        fitCont.triggered.connect(self.fitwithCont)

        resAnal = QAction('&Check structure in residuals', self)
        resAnal.setStatusTip('Analysis of structure in residuals')
        resAnal.triggered.connect(self.resAnal)

        stopFit = QAction('&Stop Fit', self)
        stopFit.setStatusTip('Stop fitting process')
        stopFit.triggered.connect(self.stopFit)

        tempFit = QAction('&Temp Fit', self)
        tempFit.setStatusTip('Tempopary button for specific actions')
        tempFit.triggered.connect(self.bootFit)

        fitCheb = QAction('&Fit Cheb', self)
        fitCheb.setStatusTip('Adjust continuum by Chebyshev polynomials')
        fitCheb.triggered.connect(partial(self.fitCheb, typ='cheb'))

        fitGP = QAction('&Fit Gaussian Process', self)
        fitGP.setStatusTip('Adjust continuum by gaussian Process')
        fitGP.triggered.connect(partial(self.fitGP))

        fitExt = QAction('&Fit Extinction...', self)
        fitExt.setStatusTip('Fit extinction')
        fitExt.triggered.connect(self.fitExt)

        fitGauss = QAction('&Fit by gaussian line', self)
        fitGauss.setStatusTip('Fit gauss')
        fitGauss.triggered.connect(partial(self.fitGauss, kind='integrate'))

        fitPower = QAction('&Power law fit', self)
        fitPower.setStatusTip('Fit by power law function')
        fitPower.triggered.connect(self.fitPowerLaw)

        fitPoly = QAction('&Polynomial fit', self)
        fitPoly.setStatusTip('Fit by polynomial function')
        fitPoly.triggered.connect(partial(self.fitPoly, None))

        fitMinEnvelope = QAction('&Envelope fit', self)
        fitMinEnvelope.setStatusTip('Find bottom envelope')
        fitMinEnvelope.triggered.connect(partial(self.fitMinEnvelope, res=200))

        H2Upper = QAction('&H2 upper limits', self)
        H2Upper.setStatusTip('Estimate H2 upper limits')
        H2Upper.triggered.connect(self.H2UpperLimit)

        AncMenu = QMenu('&Diagnostic plots', self)
        AncMenu.setStatusTip('Some ancillary for fit procedures')

        ExcDiag_t = QAction('&Excitation diagram with temperature', self)
        ExcDiag_t.setStatusTip('Show excitation diagram')
        ExcDiag_t.triggered.connect(partial(self.ExcDiag, temp=1))

        ExcDiag = QAction('&Excitation diagram', self)
        ExcDiag.setStatusTip('Show excitation diagram')
        ExcDiag.triggered.connect(partial(self.ExcDiag, temp=0))

        ExcTemp = QAction('&Excitation temperature', self)
        ExcTemp.setStatusTip('Calculate excitation temperature')
        ExcTemp.triggered.connect(partial(self.ExcitationTemp, ind=None, plot=True))

        MetalAbundance = QAction('&Metal abundance', self)
        MetalAbundance.setStatusTip('Show Metal abundances')
        MetalAbundance.triggered.connect(self.showMetalAbundance)

        fitMenu.addAction(setFit)
        fitMenu.addAction(chooseFitPars)
        fitMenu.addAction(showFit)
        fitMenu.addAction(showFullFit)
        fitMenu.addAction(fitResults)
        fitMenu.addSeparator()
        fitMenu.addAction(fitLM)
        fitMenu.addAction(fitMCMC)
        fitMenu.addAction(fitGrid)
        fitMenu.addAction(fitCont)
        fitMenu.addAction(resAnal)
        fitMenu.addAction(stopFit)
        fitMenu.addAction(tempFit)
        fitMenu.addSeparator()
        fitMenu.addAction(fitCheb)
        fitMenu.addAction(fitGP)
        fitMenu.addAction(fitExt)
        fitMenu.addAction(fitGauss)
        fitMenu.addAction(fitPower)
        fitMenu.addAction(fitPoly)
        fitMenu.addAction(fitMinEnvelope)
        fitMenu.addSeparator()
        fitMenu.addAction(H2Upper)
        fitMenu.addMenu(AncMenu)
        AncMenu.addAction(ExcDiag_t)
        AncMenu.addAction(ExcDiag)
        AncMenu.addAction(ExcTemp)
        AncMenu.addAction(MetalAbundance)

        # >>> create 1d spec Menu items

        fitCont = QAction('&Continuum...', self)
        fitCont.setStatusTip('Construct continuum using various methods')
        fitCont.setShortcut('Ctrl+C')
        fitCont.triggered.connect(partial(self.fitCont))

        CompositeQSO = QAction('&QSO composite', self)
        CompositeQSO.setStatusTip('Show composite qso spectrum')
        CompositeQSO.setShortcut('Ctrl+Q')
        CompositeQSO.triggered.connect(partial(self.showCompositeQSO))

        CompositeGal = QAction('&Galaxy template', self)
        CompositeGal.setStatusTip('Show galaxy template spectrum')
        CompositeGal.setShortcut('Ctrl+G')
        CompositeGal.triggered.connect(partial(self.showCompositeGal))

        rescaleErrs = QAction('&Adjust errors', self)
        rescaleErrs.setStatusTip('Adjust uncertainties to dispersion in the spectrum')
        rescaleErrs.triggered.connect(partial(self.rescale))

        stackLines = QAction('&Stack lines', self)
        stackLines.setStatusTip('Stack chosen absorption lines')
        stackLines.triggered.connect(partial(self.stackLines))

        FilterMenu = QMenu('&Filters:', self)
        FilterMenu.setStatusTip('Show the photometric filters')

        SDSSfilters = QAction('&SDSS filters', self, checkable=True)
        SDSSfilters.setStatusTip('Show SDSS filters')
        SDSSfilters.triggered.connect(partial(self.show_filters, name='SDSS'))
        SDSSfilters.setChecked(self.filters_status['SDSS'])

        Gaiafilters = QAction('&Gaia filters', self, checkable=True)
        Gaiafilters.setStatusTip('Show Gaia filters')
        Gaiafilters.triggered.connect(partial(self.show_filters, name='Gaia'))
        Gaiafilters.setChecked(self.filters_status['Gaia'])

        TwoMASSfilters = QAction('&2MASS filters', self, checkable=True)
        TwoMASSfilters.setStatusTip('Show 2MASS filters')
        TwoMASSfilters.triggered.connect(partial(self.show_filters, name='2MASS'))
        TwoMASSfilters.setChecked(self.filters_status['2MASS'])

        VISTAfilters = QAction('&VISTA filters', self, checkable=True)
        VISTAfilters.setStatusTip('Show VISTA filters')
        VISTAfilters.triggered.connect(partial(self.show_filters, name='VISTA'))
        VISTAfilters.setChecked(self.filters_status['VISTA'])

        UKIDSSfilters = QAction('&UKIDSS filters', self, checkable=True)
        UKIDSSfilters.setStatusTip('Show UKIDSS filters')
        UKIDSSfilters.triggered.connect(partial(self.show_filters, name='UKIDSS'))
        UKIDSSfilters.setChecked(self.filters_status['UKIDSS'])

        WISEfilters = QAction('&WISE filters', self, checkable=True)
        WISEfilters.setStatusTip('Show WISE filters')
        WISEfilters.triggered.connect(partial(self.show_filters, name='WISE'))
        WISEfilters.setChecked(self.filters_status['WISE'])

        GALEXfilters = QAction('&GALEX filters', self, checkable=True)
        GALEXfilters.setStatusTip('Show GALEX filters')
        GALEXfilters.triggered.connect(partial(self.show_filters, name='GALEX'))
        GALEXfilters.setChecked(self.filters_status['GALEX'])

        spec1dMenu.addAction(fitCont)
        spec1dMenu.addAction(CompositeQSO)
        spec1dMenu.addAction(CompositeGal)
        spec1dMenu.addAction(rescaleErrs)
        spec1dMenu.addSeparator()
        spec1dMenu.addAction(stackLines)
        spec1dMenu.addSeparator()
        spec1dMenu.addMenu(FilterMenu)
        FilterMenu.addAction(SDSSfilters)
        FilterMenu.addAction(Gaiafilters)
        FilterMenu.addAction(TwoMASSfilters)
        FilterMenu.addAction(UKIDSSfilters)
        FilterMenu.addAction(WISEfilters)
        FilterMenu.addAction(GALEXfilters)

        # >>> create 2d spec Menu items

        extract = QAction('&Extract', self)
        extract.setStatusTip('extract 1d spectrum from 2d spectrum')
        extract.setShortcut('Ctrl+D')
        extract.triggered.connect(self.extract2d)

        spec2dMenu.addAction(extract)

        # >>> create Combine Menu items
        
        expList = QAction('&Exposure list', self)
        expList.setStatusTip('show Exposure list')
        expList.triggered.connect(self.showExpListCombine)
                
        selectCosmics = QAction('&Select cosmic', self)        
        selectCosmicsUVESSet = QAction('&Load Settings', self)
        selectCosmics.triggered.connect(self.selectCosmics)
        
        calcSmooth = QAction('&Smooth', self)        
        calcSmooth.setStatusTip('Smooth exposures')
        calcSmooth.triggered.connect(self.calcSmooth)
        
        coscaleExp = QAction('&Coscale', self)
        coscaleExp.setStatusTip('Coscale exposures')
        coscaleExp.triggered.connect(self.coscaleExposures)

        shiftExp = QAction('&Shift', self)
        shiftExp.setStatusTip('Shift exposure')
        shiftExp.triggered.connect(self.shiftExposure)

        rescaleExp = QAction('&Rescale', self)
        rescaleExp.setStatusTip('Rescale exposure')
        rescaleExp.triggered.connect(self.rescaleExposure)

        rescaleErrs = QAction('&Rescale errs', self)
        rescaleErrs.setStatusTip('Rescale uncertainties')
        rescaleErrs.triggered.connect(self.rescaleErrs)

        combine = QAction('&Combine...', self)
        combine.setStatusTip('Combine exposures')
        combine.triggered.connect(self.combine)

        rebin = QAction('&Rebin...', self)
        rebin.setStatusTip('Rebin exposures')
        rebin.triggered.connect(self.rebin)
        
        combineMenu.addAction(expList)
        combineMenu.addSeparator()
        combineMenu.addAction(selectCosmics)
        combineMenu.addAction(calcSmooth)
        combineMenu.addAction(coscaleExp)
        combineMenu.addAction(shiftExp)
        combineMenu.addAction(rescaleExp)
        combineMenu.addAction(rescaleErrs)
        combineMenu.addSeparator()
        combineMenu.addAction(combine)
        combineMenu.addAction(rebin)


        # >>> create SDSS Menu items
        loadSDSS = QAction('&load SDSS', self)
        loadSDSS.setStatusTip('Load SDSS by Plate/fiber')
        loadSDSS.triggered.connect(self.showSDSSdialog)

        SDSSMenu.addAction(loadSDSS)
        SDSSMenu.addSeparator()

        if self.developer:
            SDSSLeelist = QAction('&DR9 Lee list', self)
            SDSSLeelist.setStatusTip('load SDSS DR9 Lee database')
            SDSSLeelist.triggered.connect(self.loadSDSSLee)

            importSDSSlist = QAction('&import SDSS list', self)
            importSDSSlist.setStatusTip('import SDSS list from file')
            importSDSSlist.triggered.connect(partial(self.importSDSSlist, filename=None))

            showSDSSlist = QAction('&show SDSS list', self)
            showSDSSlist.setStatusTip('show SDSS list')
            showSDSSlist.triggered.connect(self.show_SDSS_list)

            SDSSSearchH2 = QAction('&Search H2', self)
            SDSSSearchH2.setStatusTip('Search H2 absorption systems')
            SDSSSearchH2.triggered.connect(self.search_H2)

            SDSSH2cand = QAction('&Show H2 cand.', self)
            SDSSH2cand.setStatusTip('Show H2 cand.')
            SDSSH2cand.triggered.connect(self.show_H2_cand)

            SDSSStack = QAction('&Stack', self)
            SDSSStack.setStatusTip('Calculate SDSS Stack spectrum')
            SDSSStack.triggered.connect(self.calc_SDSS_Stack_Lee)

            SDSSDLA = QAction('&DLA search', self)
            SDSSDLA.setStatusTip('Search for DLA systems')
            SDSSDLA.triggered.connect(self.calc_SDSS_DLA)

            SDSSPhot = QAction('&SDSS photometry', self)
            SDSSPhot.setStatusTip('Show SDSS photometry window')
            SDSSPhot.triggered.connect(self.SDSSPhot)

            SDSSMenu.addAction(SDSSLeelist)
            SDSSMenu.addAction(importSDSSlist)
            SDSSMenu.addAction(showSDSSlist)
            SDSSMenu.addSeparator()
            SDSSMenu.addAction(SDSSSearchH2)
            SDSSMenu.addAction(SDSSH2cand)
            SDSSMenu.addSeparator()
            SDSSMenu.addAction(SDSSStack)
            SDSSMenu.addAction(SDSSDLA)
            SDSSMenu.addSeparator()
            SDSSMenu.addAction(SDSSPhot)

            # >>> create Samples Menu items
            XQ100list = QAction('&XQ100 list', self)
            XQ100list.setStatusTip('load XQ100 list')
            XQ100list.triggered.connect(self.showXQ100list)

            P94list = QAction('&P94 list', self)
            P94list.setStatusTip('load P94 list')
            P94list.triggered.connect(self.showP94list)

            DLAlist = QAction('&DLA list', self)
            DLAlist.setStatusTip('load DLA list')
            DLAlist.triggered.connect(self.showDLAlist)

            LyaforestMenu = QMenu('&Lyaforest', self)

            Lyalist = QAction('&Lyaforest sample', self)
            Lyalist.setStatusTip('load lya forest sample')
            Lyalist.triggered.connect(self.showLyalist)
            LyaforestMenu.addAction(Lyalist)

            Lyalines = QAction('&Lyaforest line', self)
            Lyalines.setStatusTip('load lya forest lines')
            Lyalines.triggered.connect(self.showLyalines)
            LyaforestMenu.addAction(Lyalines)

            Vandels = None
            if self.VandelsFile is not None and os.path.isfile(self.VandelsFile):
                Vandels = QAction('&Vandels', self)
                Vandels.setStatusTip('load Vandels catalog')
                Vandels.triggered.connect(self.showVandels)

            Kodiaq = None
            if self.KodiaqFile is not None and os.path.isfile(self.KodiaqFile):
                Kodiaq = QAction('&KODIAQ DR2', self)
                Kodiaq.setStatusTip('load Kodiaq DR2 catalog')
                Kodiaq.triggered.connect(self.showKodiaq)

            UVES = None
            if self.UVESfolder is not None and os.path.isdir(self.UVESfolder):
                UVES = QAction('&UVES ADP QSO', self)
                UVES.setStatusTip('load QSO sample from UVES ADP')
                UVES.triggered.connect(self.showUVES)

            Erosita = None
            if self.ErositaFile is not None and os.path.isfile(self.ErositaFile):
                Erosita = QAction('&Erosita-SDSS', self)
                Erosita.setStatusTip('load Erosita-SDSS matched sample')
                Erosita.triggered.connect(self.showErosita)

            IGMspecMenu = None
            if self.IGMspecFile is not None and os.path.isfile(self.IGMspecFile):
                IGMspecMenu = QMenu('&IGMspec', self)
                IGMspecMenu.setStatusTip('Data from IGMspec database')
                try:
                    self.IGMspec = h5py.File(self.IGMspecFile, 'r')
                    for i in self.IGMspec.keys():
                        item = QAction('&'+i, self)
                        item.triggered.connect(partial(self.showIGMspec, i, None))
                        IGMspecMenu.addAction(item)
                except:
                    pass

            MALS_gal = None
            if self.MALSfolder is not None and os.path.isfile(self.MALSfolder + 'catalog.csv'):
                MALS_gal = QAction('&MALS_galactic', self)
                MALS_gal.setStatusTip('load MALS galactic sample')
                MALS_gal.triggered.connect(self.showMALS)

            samplesMenu.addAction(XQ100list)
            samplesMenu.addAction(P94list)
            samplesMenu.addAction(DLAlist)
            samplesMenu.addMenu(LyaforestMenu)
            if Vandels is not None:
                samplesMenu.addAction(Vandels)
            if Kodiaq is not None:
                samplesMenu.addAction(Kodiaq)
            if UVES is not None:
                samplesMenu.addAction(UVES)
            if Erosita is not None:
                samplesMenu.addAction(Erosita)
            if MALS_gal is not None:
                samplesMenu.addAction(MALS_gal)
            samplesMenu.addSeparator()
            if IGMspecMenu is not None:
                samplesMenu.addMenu(IGMspecMenu)

        # >>> create Generate Menu items
        addAbsSystem = QAction('&Generate with model', self)
        addAbsSystem.setStatusTip('generate spectrum based on the current fit model')
        addAbsSystem.triggered.connect(self.add_abs_system)
        
        addDustSystem = QAction('&Apply dust extinction', self)
        addDustSystem.setStatusTip('apply dust extinction')
        addDustSystem.triggered.connect(self.add_dust_system)

        colorColorPlot = QAction('&color-color', self)
        colorColorPlot.setStatusTip('show color-color generation module')
        colorColorPlot.triggered.connect(self.colorColorPlot)

        generateMenu.addAction(addAbsSystem)
        generateMenu.addAction(addDustSystem)
        generateMenu.addSeparator()
        generateMenu.addAction(colorColorPlot)

        # >>> create Obervations Menu items
        UVESMenu = QMenu('&UVES', self)
        UVESMenu.setStatusTip('methods for UVES/VLT')

        self.UVESSetup = QAction('&Setup', self, checkable=True)
        self.UVESSetup.setStatusTip('&Choose appropriate Setup')
        self.UVESSetup.triggered.connect(self.chooseUVESSetup)
        self.UVESSetup.setChecked(self.UVESSetup_status)

        UVESetc = QAction('&load ETC data', self)
        UVESetc.setStatusTip('Add data from UVES ETC')
        UVESetc.triggered.connect(self.addUVESetc)

        UVESMenu.addAction(self.UVESSetup)
        UVESMenu.addAction(UVESetc)
        obsMenu.addMenu(UVESMenu)

        observability = QAction('&Observability', self)
        observability.setStatusTip('&Calculate observability for given targets')
        observability.triggered.connect(self.showObservability)

        obsMenu.addSeparator()
        obsMenu.addAction(observability)

        # >>> create Help Menu items
        howto = QAction('&How to ...', self)
        howto.setShortcut('F1')
        howto.setStatusTip('How to do')
        howto.triggered.connect(self.info_howto)
        
        tutorial = QAction('&Tutorial', self)        
        tutorial.setStatusTip('Some tutorial')
        tutorial.triggered.connect(self.info_tutorial)
        
        about = QAction('&About', self)
        about.setStatusTip('About the program')
        about.triggered.connect(self.info_about)
        
        helpMenu.addAction(howto)
        helpMenu.addAction(tutorial)
        helpMenu.addSeparator()
        helpMenu.addAction(about)

    def initData(self):
        self.fit = fitPars(self)
        self.atomic = atomicData()
        self.atomic.readdatabase()
        self.abs = absSystemIndicator(self)
        for s in ['H2', 'DLAmajor', 'DLA', 'Molec', 'SF']:
            self.absLines('abs_'+s+'_status', value=getattr(self, 'abs_'+s+'_status'))

        filename = self.options('loadfile', config=self.config)
        if os.path.exists(filename):
            import importlib.util
            spec = importlib.util.spec_from_file_location("load", filename)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            foo.loadaction(self)

    def options(self, opt, value=None, config='config/options.ini'):
        """
        Read and write options from the config file
        """
        with open(self.folder + config) as f:
            s = f.readlines()

        for i, line in enumerate(s):
            if len(line.split()) > 0 and opt == line.split()[0] and not any([line.startswith(st) for st in ['#', '!', '$', '%']]):
                if value is None:
                    if len(line.split()) > 2:
                        if line.split()[2] == 'None':
                            return None
                        elif line.split()[2] == 'True':
                            return True
                        elif line.split()[2] == 'False':
                            return False
                        else:
                            return ' '.join(line.split()[2:])
                    else:
                        return ''
                else:
                    setattr(self, opt, value)
                    s[i] = "{0:20}  :  {1:} \n".format(opt, value)
                break
        else:
            return None
            #return 'option {0} was not found'.format(opt)

        with open(self.folder + 'config/options.ini', 'w') as f:
            for line in s:
                f.write(line)

    def reload_julia(self):
        #t = Timer("julia")
        self.sendMessage("Julia was not imported. Importing")
        #try:
        #    reload(Julia)
        #except:
        if 0:
            from julia.api import Julia
            #t.time("imp")
            print("compiled modules: ", platform.system() != 'Linux')
            Julia(compiled_modules=(platform.system() != 'Linux'), optimize=3)  # .Julia()

            from julia import Main

            #t.time("comp")
            self.julia = Main
            #t.time("second im")
        else:
            from juliacall import Main as julia
            self.julia = julia

        self.julia.include(self.folder + "profiles.jl")

        #t.time("include")
        self.options('juliaFit', True)
        return True

        #except:
        #    self.sendMessage("There was a problems to import Julia was not imported.")
        #    self.julia = None
        #    self.options("fitType", "uniform")
        #    return False


    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   GUI routines
    # >>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            # Workaround for OSx dragging and dropping

            filelist = []
            for url in event.mimeData().urls():
                filelist.append(str(url.toLocalFile()))
                print('drop:', str(url.toLocalFile()))
            if str(url.toLocalFile()).endswith('.spv'):
                self.openFile(str(url.toLocalFile()))
            else:
                self.importSpectrum(filelist, append=True)
        else:
            event.ignore()

    def mousePressEvent(self, event):
        #if self.b_status:
        pass
        
    def mouseMoveEvent(self, event):
        pass
    
    def draw(self):
        # >>> add zero line level
        self.zeroline = pg.InfiniteLine(0.0, 0, pen=pg.mkPen(color=(148, 103, 189), width=1, style=Qt.PenStyle.DashLine))
        if 0:
            self.zeroline.setMovable(1)
            self.zeroline.setHoverPen(color=(214, 39, 40), width=3)
        self.vb.addItem(self.zeroline)

    def sendMessage(self, text='', timer=2000):
        if self.message is not None:
            self.message.animation.stop()
            self.message.animStarted = True
            self.message.close()

        self.message = messageWindow(self, text=text, timer=timer)

    #def closeMessage(self):
    #    self.message.close()
    #    self.message = None

    def absLines(self, status='', sig=True, value=None, verbose=False):

        if verbose:
            print(status, value)

        if value is not None:
            setattr(self, status, value)
        else:
            setattr(self, status, 1 - getattr(self, status))
        if value != 0:
            if status == 'abs_H2_status':
                lines, color, va = self.atomic.list(['H2j'+str(i) for i in range(3)]), (229, 43, 80), 'down'
            if status == 'abs_DLA_status':
                lines, color, va = self.atomic.DLA_list(), (105, 213, 105), 'down'
            if status == 'abs_DLAmajor_status':
                lines, color, va = self.atomic.DLA_major_list(), (105, 213, 105), 'down'
            if status == 'abs_LowIoniz_status':
                lines, color, va = self.atomic.Low_Ioniz_list(), (39, 140, 245), 'down'
            if status == 'abs_Molec_status':
                lines, color, va = self.atomic.Molecular_list(), (255, 111, 63), 'down'
            if status == 'abs_SF_status':
                lines, color, va = self.atomic.EmissionSF_list(), (0, 204, 255), 'up'

            if verbose:
                print('linelist:', lines)

            if getattr(self, status):
                self.abs.add(lines, color=color, va=va)
            else:
                self.abs.remove(lines)

    def absChoicelines(self):
        d = {'H2': ['J='+str(i) for i in range(10)]}
        d = {k: [] for k in self.atomic.keys()}
        self.choiceLinesWindow = choiceLinesWidget(self, d)
        self.choiceLinesWindow.show()

    def hideAllLines(self):
        self.console.exec_command('hide all')
        for s in ['H2', 'DLA', 'DLAmajor', 'Molec']:
            getattr(self, 'lines'+s).setChecked(False)
            setattr(self, 'abs_' + s + '_status', False)

    def setz_abs(self, text):
        self.z_abs = float(text)
        self.panel.z_panel.setText("{:20.10f}".format(self.z_abs).strip().rstrip("0"))
        if self.plot.restframe:
            self.plot.updateVelocityAxis()
        self.abs.redraw()

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   File menu routines
    # >>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def clearSession(self):

        # >>> remove regions:
        for r in reversed(self.plot.regions[:]):
            self.plot.regions.remove(r)
        self.plot.regions = regionList(self.plot)

        # >>> remove doublets:
        for d in reversed(self.plot.doublets[:]):
            d.remove()
        self.plot.doublets = doubletList(self.plot)

        for s in self.s:
            s.remove()
        self.s = Speclist(self)

        self.lines = lineList(self)

        self.plot.remove_pcRegion()
        if self.fitModel is not None:
            self.fitModel.close()

        if self.fitResults is not None:
            self.fitResults.close()

        if self.photo is not None:
            for f in self.photo:
                for fi in f:
                    if fi in self.plot.vb.addedItems:
                        self.plot.vb.removeItem(fi)
            self.photo = None

        self.fit = fitPars(self)

        self.setz_abs(0)

    def showOpenDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', self.work_folder)
        
        if fname[0]:

            self.openFile(fname[0])
            self.statusBar.setText('Data was read from ' + fname[0])
            self.showFit()


    def openFile(self, filename, zoom=True, skip_header=0, remove_regions=False, remove_doublets=False):

        if remove_regions:
            for r in reversed(self.plot.regions[:]):
                self.plot.regions.remove(r)
            self.plot.regions = regionList(self.plot)

        if remove_doublets:
            for d in reversed(self.plot.doublets[:]):
                d.remove()
            self.plot.doublets = doubletList(self.plot)

        folder = os.path.dirname(filename)

        with open(filename) as f:
            d = f.readlines()

        i = -1 + skip_header
        while (i < len(d)-1):
            i += 1
            if '%' in d[i] or any([x in d[i] for x in ['spect', 'Bcont', 'fitting']]):
                if '%' in d[i]:
                    specname = d[i][1:].strip()
                    print(specname)
                    try:
                        ind = [s.filename for s in self.s].index(specname)
                    except:
                        ind = -1
                        try:
                            if all([slash not in specname for slash in ['/', '\\']]):
                                specname = folder + '/' + specname
                            if not self.importSpectrum(specname, append=True):
                                st = re.findall(r'spec-\d{4}-\d{5}-\d+', specname)
                                if len(st) > 0:
                                    self.loadSDSS(plate=int(st[0].split('-')[1]), fiber=int(st[0].split('-')[3]))
                            ind = len(self.s) - 1
                        except:
                            pass
                    i += 1
                else:
                    ind = 0

                if i > len(d) - 1:
                    break

                if ind == -1 and 'spectrum' in d[i]:
                    n = int(d[i].split()[1])
                    if n > 0:
                        x, y, err = [], [], []
                        if n > 0:
                            for t in range(n):
                                i += 1
                                w = d[i].split()
                                x.append(float(w[0]))
                                y.append(float(w[1]))
                                if len(w) > 2:
                                    err.append(float(w[2]))
                        self.importSpectrum(specname, spec=[np.asarray(x), np.asarray(y), np.asarray(err)], append=True)
                        ind = len(self.s) - 1

                if ind > -1:
                    while all([x not in d[i] for x in ['%', '----', 'doublet', 'region', 'fit_model']]):
                        if 'Bcont' in d[i]:
                            self.s[ind].spline = gline()
                            n = int(d[i].split()[1])
                            if n > 0:
                                for t in range(n):
                                    i += 1
                                    w = d[i].split()
                                    self.s[ind].spline.add(x=float(w[0]), y=float(w[1]))
                                self.s[ind].calc_spline()

                        if 'fitting_points' in d[i]:
                            self.s[ind].mask.set(x=np.zeros_like(self.s[ind].spec.x(), dtype=bool))
                            n = int(d[i].split()[1])
                            if n > 0:
                                i += 1
                                w = [float(line.split()[0]) for line in d[i:i+n]]
                                self.s[ind].add_exact_points(w, redraw=False)
                                i += n-1
                                self.s[ind].mask.normalize()
                        if 'bad_pixels' in d[i]:
                            self.s[ind].bad_mask.set(x=np.zeros_like(self.s[ind].spec.x(), dtype=bool))
                            n = int(d[i].split()[1])
                            if n > 0:
                                i += 1
                                w = [float(line.split()[0]) for line in d[i:i + n]]
                                self.s[ind].add_exact_points(w, redraw=False, bad=True)
                                i += n - 1
                                self.s[ind].bad_mask.normalize()

                        if 'sky' in d[i]:
                            self.importTelluric(d[i].split()[1])

                        if '2d' in d[i]:
                            self.import2dSpectrum(d[i].split()[1], ind=ind)

                        if 'resolution' in d[i]:
                            print(d[i].split()[1], )
                            if ".." in d[i].split()[1]:
                                print(d[i].split()[1].split(".."))
                                self.s[ind].set_resolution(float(d[i].split()[1].split("..")[0]), float(d[i].split()[1].split("..")[1]))
                            else:
                                self.s[ind].set_resolution(float(d[i].split()[1]))

                        if 'lsf_type' in d[i]:
                            self.s[ind].lsf_type = d[i].split()[1]

                        if 'scaling_factor' in d[i]:
                            if float(d[i].split()[1]) != 1:
                                self.s[ind].rescale(float(d[i].split()[1]))

                        i += 1

                        if i > len(d) - 1:
                            break

            if '%' in d[i]:
                i -= 1

            if 'regions' in d[i]:
                ns = int(d[i].split()[1])
                for r in range(ns):
                    self.plot.regions.add('..'.join(d[i+1+r].split()))

            if 'doublets' in d[i]:
                ns = int(d[i].split()[1])
                for r in range(ns):
                    i += 1
                    #print(d[i].split()[0], float(d[i].split()[1]))
                    if len(d[i].split()) == 2:
                        self.plot.doublets.append(Doublet(self.plot, name=d[i].split()[0], z=float(d[i].split()[1])))
                    else:
                        self.plot.doublets.append(Doublet(self.plot, name=d[i].split()[0], z=float(d[i].split()[1])))
                        for l in d[i].split()[2:]:
                            print(l.split('_')[0], float(l.split('_')[1]))
                            self.plot.doublets[-1].add_line(float(l.split('_')[1]), name=l.split('_')[0])

            if 'lines' in d[i]:
                self.lines = lineList(self)
                ns = int(d[i].split()[1])
                for r in range(ns):
                    i += 1
                    self.lines.add(d[i].strip())
                    self.abs.lines[-1].setActive(bool=True)

            if 'fit_model' in d[i]:
                self.plot.remove_pcRegion()
                self.fit = fitPars(self)
                num = int(d[i].split()[1])
                for k in range(num):
                    i += 1
                    self.fit.readPars(d[i])
                if num > 0:
                    self.setz_abs(self.fit.sys[0].z.val)
                self.fit.showLines()

            if 'fit_tieds' in d[i]:
                self.fit.tieds = {}
                num = int(d[i].split()[1])
                for k in range(num):
                    i += 1
                    self.fit.addTieds(d[i].strip().split()[0], d[i].strip().split()[1])

            if 'fit_exclude' in d[i]:
                for k in range(len(self.fit.sys)):
                    self.fit.sys[k].exclude = []
                num = int(d[i].split()[1])
                for k in range(num):
                    i += 1
                    #print(int(d[i].split()[0]))
                    self.fit.sys[int(d[i].split()[0])].exclude.append(' '.join(d[i].split()[1:]))


            #print(d[i])
            #if 'cheb' in d[i]:
            #    self.fit.cont_num = int(d[i].split()[1])
            #    for k in range(self.fit.cont_num):
            #        i += 1
            #        self.fit.cont[k].left, self.fit.cont[k].right = float(d[i].split()[0]), float(d[i].split()[1])

            if 'fit_results' in d[i]:
                num = int(d[i].split()[1])
                for k in range(num):
                    i += 1
                    print(d[i])
                    self.fit.setValue(self.atomic.correct_name(d[i].split()[0]), d[i].split()[2], 'unc')

        if zoom:
            try:
                self.plot.set_range(self.s[self.s.ind].spline.x[0], self.s[self.s.ind].spline.x[-1])
            except:
                pass

        self.work_folder = os.path.dirname(filename)
        self.options('work_folder', self.work_folder)
        self.options('filename_saved', filename)
        self.s.redraw(self.s.ind)

        self.sendMessage(f'File {filename} was opened')

    def saveFile(self, filename, save_name=True):
        if not filename.endswith('.spv'):
            filename += '.spv'

        if os.path.isfile(filename):
            copyfile(filename, self.folder + '/temp/backup.spv')

        with open(filename, 'w') as f:

            if any([opt in self.save_opt for opt in ['spectrum', 'cont', 'points']]):
                for s in self.s:
                    if save_name:
                        print(os.path.dirname(os.path.realpath(filename)))
                        print(os.path.dirname(os.path.realpath(s.filename)))
                        print(os.path.dirname(os.path.realpath(filename)) == os.path.dirname(os.path.realpath(s.filename)))
                        if os.path.dirname(os.path.realpath(filename)) == os.path.dirname(os.path.realpath(s.filename)):
                            print(os.path.basename(os.path.realpath(s.filename)))
                            f.write('%{}\n'.format(os.path.basename(os.path.realpath(s.filename))))
                        else:
                            f.write('%{}\n'.format(s.filename))

                    # >>> save spectra
                    if 'spectrum' in self.save_opt:
                        num = len(s.spec.x())
                        if num > 0:
                            if self.normview:
                                f.write('norm_spectrum:  {}\n'.format(num))
                            else:
                                f.write('spectrum:  {}\n'.format(num))
                            for x, y, err in zip(s.spec.x(), s.spec.y(), s.spec.err()):
                                f.write('{0:10.4f}  {1:10.4f}  {2:10.4f} \n'.format(x, y, err))

                    # >>> save cont
                    if 'cont' in self.save_opt:
                        f.write('Bcont:  {}\n'.format(s.spline.n))
                        if s.spline.n > 0:
                            for x, y in zip(s.spline.x, s.spline.y):
                                f.write('{0:10.4f}  {1:10.4e} \n'.format(x, y))

                    # >>> save fitting points:
                    if 'points' in self.save_opt:
                        num = np.sum(s.mask.x())
                        f.write('fitting_points:   {}\n'.format(num))
                        if num > 0:
                            for x in s.spec.x()[s.mask.x()]:
                                f.write('{:.5f}\n'.format(x))

                    # >>> save bad points:
                    if 'points' in self.save_opt:
                        f.write('bad_pixels:   {}\n'.format(np.sum(s.bad_mask.x())))
                        if np.sum(s.bad_mask.x()) > 0:
                            for x in s.spec.x()[s.bad_mask.x()]:
                                f.write('{:.5f}\n'.format(x))

                    # >>> save resolution amd scaling factor:
                    if 'others' in self.save_opt:
                        if s.resolution_linear[0] not in [0, None]:
                            f.write('resolution:   {}\n'.format(s.resolution(out='spv')))
                        if s.lsf_type != 'gauss':
                            f.write('lsf_type:   {}\n'.format(s.lsf_type))
                        if s.scaling_factor not in [0, None]:
                            f.write('scaling_factor:   {}\n'.format(s.scaling_factor))

                    # >>> save sky spectrum:
                    if 'others' in self.save_opt:
                        if s.sky.filename not in [None, '']:
                            f.write('sky:   {}\n'.format(s.sky.filename))

                    # >>> save 2d spectrum:
                    if 'others' in self.save_opt:
                        if s.spec2d is not None and s.spec2d.filename not in [None, '']:
                            f.write('2d:   {}\n'.format(s.spec2d.filename))

                    f.write('-------------------------\n')

            # >>> save other parameters
            if 'others' in self.save_opt:
                if len(self.plot.regions) > 0:
                    f.write('regions:   ' + str(len(self.plot.regions)) + '\n')
                    for r in self.plot.regions:
                        mi, ma = r.getRegion()
                        f.write('{0:11.5f} {1:11.5f} \n'.format(mi, ma))

                if len(self.plot.doublets) > 0:
                    f.write('doublets:   ' + str(len(self.plot.doublets)) + '\n')
                    for d in self.plot.doublets:
                        f.write(str(d)+'\n')

                if len(self.lines) > 0:
                    f.write('lines:   ' + str(len(self.lines)) + '\n')
                    for l in self.lines:
                        f.write('{0}\n'.format(l))

            # >>> save fit model:
            if 'fit' in self.save_opt:
                pars = self.fit.list()
                f.write('fit_model: {0:}\n'.format(len(pars)))
                for p in pars:
                    f.write(p.str() + '\n')

            # >>> save fit model tieds:
            if 'fit' in self.save_opt:
                if len(self.fit.tieds.keys()) > 0:
                    f.write('fit_tieds: {0:}\n'.format(len(self.fit.tieds.keys())))
                    for k, v in self.fit.tieds.items():
                        f.write(' '.join([k, v]) + '\n')

            if 'fit' in self.save_opt:
                excl = [' '.join([str(i), line]) for i, sys in enumerate(self.fit.sys) for line in sys.exclude]
                if len(excl) > 0:
                    f.write('fit_exclude: {0:}\n'.format(len(excl)))
                    for line in excl:
                        f.write(line + '\n')

            # >>> save cheb parameters:
            #if 'fit' in self.save_opt:
            #    if self.fit.cont_fit:
            #        f.write('cheb: {0:}\n'.format(self.fit.cont_num))
            #        for i in range(self.fit.cont_num):
            #            f.write('{0:.2f} {1:.2f} \n'.format(self.fit.cont[i].left, self.fit.cont[i].right))

            # >>> save fit result:
            if 'fit_results' in self.save_opt:
                pars = self.fit.list_fit()
                if any([p.unc.minus > 0 and p.unc.plus > 0 for p in pars]):
                    f.write('fit_results: {0:}\n'.format(len(pars)))
                    for p in pars:
                        f.write(str(p) + ' = ' + p.fitres(latex=True, showname=True) + '\n')

        self.work_folder = os.path.dirname(filename)
        self.options('work_folder', self.work_folder)

        self.statusBar.setText('Data is saved to ' + filename)

    def saveFilePressed(self):
        if self.options('filename_saved') is None:
            self.showSaveDialog()
        else:
            self.saveFile(self.options('filename_saved'))

    def showSaveDialog(self):
        self.exportData = ExportDataWidget(self, 'save')

    def showImportDialog(self):

        fname = QFileDialog.getOpenFileName(self, 'Import spectrum', self.work_folder)

        if fname[0]:
            
            self.importSpectrum(fname[0])
            self.abs.redraw()
            self.statusBar.setText('Spectrum is imported from ' + fname[0])

    def showImportTelluricDialog(self):

        fname = QFileDialog.getOpenFileName(self, 'Import telluric/sky/accompanying spectrum', self.work_folder)

        if fname[0]:
            self.importTelluric(fname[0])
            self.abs.redraw()
            self.statusBar.setText('Telluric/sky/accompaying spectrum is imported from ' + fname[0])

    def show2dImportDialog(self):

        fname = QFileDialog.getOpenFileName(self, 'Import 2d spectrum', self.work_folder)

        if fname[0]:
            self.import2dSpectrum(fname[0])
            self.statusBar.setText('2d spectrum is imported from ' + fname[0])

    def showDispImportDialog(self):

        fname = QFileDialog.getOpenFileName(self, 'Import confidence band profiles', self.work_folder)

        if fname[0]:
            self.load_fit_disp(fname[0])
            self.statusBar.setText('Confidence band profile is imported from ' + fname[0])

    def load_fit_disp(self, filename=''):

        self.julia.include(self.folder + "MCMC.jl")

        self.s.prepareFit(-1, all=all)
        self.s.calcFit(recalc=True)
        self.s.calcFitComps(recalc=True)

        if self.fitType == 'julia':
            x, fit_disp, fit_comp_disp, cheb_disp = self.julia.load_disp(filename)
            for i, s in enumerate(self.s):
                if s.fit.line.norm.n > 0:
                    x[i] = np.asarray(x[i])
                    #if s.fit.line.norm.n > 0:
                    self.s[i].fit.disp[0].set(x=x[i], y=np.asarray(fit_disp[i][:, 0]))
                    self.s[i].fit.disp[1].set(x=x[i], y=np.asarray(fit_disp[i][:, 1]))
                    if self.fit.cont_fit:
                        self.s[i].cheb.disp[0].set(x=x[i], y=np.asarray(cheb_disp[i][:, 0]))
                        self.s[i].cheb.disp[1].set(x=x[i], y=np.asarray(cheb_disp[i][:, 1]))
                    for k, sys in enumerate(self.fit.sys):
                        if len(fit_comp_disp[i][k]) > 0:
                            self.s[i].fit_comp[k].disp[0].set(x=x[i], y=np.asarray(fit_comp_disp[i][k][:, 0]))
                            self.s[i].fit_comp[k].disp[1].set(x=x[i], y=np.asarray(fit_comp_disp[i][k][:, 1]))
                        else:
                            self.s[i].fit_comp[k].disp[0].set(x=self.s[i].fit.disp[0].norm.x, y=self.s[i].fit.disp[0].norm.y)
                            self.s[i].fit_comp[k].disp[1].set(x=self.s[i].fit.disp[1].norm.x, y=self.s[i].fit.disp[1].norm.y)

        print("load disp done")

    def importSpectrum(self, filelist, spec=None, mask=None, header=0, dir_path='', scale_factor=1, append=False, corr=True):

        if not append:
            for s in self.s:
                s.remove()
            self.s = Speclist(self)

        if len(self.s) == 0:
            append = False

        if isinstance(filelist, str):
            filelist = [filelist]

        if self.normview:
            self.normalize()

        for line in filelist:
            filename = line.strip()
            s = Spectrum(self, name=filename)

            telluric = None

            if spec is None:

                if filename.endswith('tar.gz'):
                    tar = tarfile.open(filename, 'r:gz')
                    for m in tar.getmembers():
                        if m.name.endswith('.fits') or m.name.endswith('.fit'):
                            filename = tar.extractfile(m)
                            hdulist = fits.open(filename)
                else:
                    hdulist = None
                    if ':' not in filename:
                        filename = dir_path + filename

                if 'IGMspec' in filename:
                    if self.IGMspecFile is not None:
                        s1 = filename.split('/')
                        data = self.IGMspec[s1[1]]
                        d = np.empty([len(data['meta']['IGM_ID'])], dtype=[('SPEC_FILE', str, 100)])
                        d['SPEC_FILE'] = np.array([x[:] for x in data['meta']['SPEC_FILE']])
                        ind = [i for i, d in enumerate(d['SPEC_FILE']) if s1[2] in d][0]
                        s.set_data([data['spec'][ind]['wave'], data['spec'][ind]['flux'], data['spec'][ind]['sig']], mask=data['spec'][ind]['and_mask'])
                        if s1[1] == 'KODIAQ_DR1':
                            s.spec.raw.clean(min=-1, max=2)
                            s.set_data()
                        s.set_resolution(data['meta']['R'][ind])

                elif hdulist is not None or filename.endswith('.fits') or filename.endswith('.fit'):
                    if hdulist is None:
                        hdulist = fits.open(filename)
                    if 'INSTRUME' in hdulist[0].header:
                        #try:
                        if 'XSHOOTER' in hdulist[0].header['INSTRUME']:
                            prihdr = hdulist[1].data
                            scale = 10 if prihdr[0][0][0] < 2000 else 1
                            s.set_data([prihdr[0][0][:] * scale, prihdr[0][1][:] * 1e17, prihdr[0][2][:] * 1e17])

                        if any([instr in hdulist[0].header['INSTRUME'] for instr in ['UVES', 'VIMOS']]):
                            prihdr = hdulist[1].data
                            l = prihdr[0][0][:]
                            coef = 1e17 if 'VIMOS' in hdulist[0].header['INSTRUME'] else 1
                            s.set_data([l, prihdr[0][1][:]*coef, prihdr[0][2][:]*coef])
                            if 'SPEC_RES' in hdulist[0].header:
                                s.set_resolution(hdulist[0].header['SPEC_RES'])
                            if 'DATE-OBS' in hdulist[0].header:
                                s.date = hdulist[0].header['DATE-OBS']
                            print(s.resolution_linear, s.date)

                        if 'STIS' in hdulist[0].header['INSTRUME']:
                            prihdr = hdulist[1].data

                            if hdulist[1].header['NAXIS2'] == 1:
                                if 'HLSPID' in hdulist[0].header and (hdulist[0].header['HLSPID'].strip() == 'HSLA' or hdulist[0].header['HLSPID'].strip() == 'ULLYSES'):
                                    s.set_data([prihdr['WAVELENGTH'][0], prihdr['FLUX'][0], prihdr['ERROR'][0]])
                                else:
                                    s.set_data([prihdr['WAVE'][0], prihdr['FLUX'][0], prihdr['ERROR'][0]])
                            else:
                                if 1:
                                    for k in range(hdulist[1].header['NAXIS2']):
                                        s = Spectrum(self, name=filename + f'_{k}')
                                        s.set_data([prihdr['WAVELENGTH'][k], prihdr['FLUX'][k], prihdr['ERROR'][k]])
                                        res = 0 if "SPECRES" not in hdulist[0].header else hdulist[0].header["SPECRES"]
                                        s.set_resolution(res)
                                        if k < hdulist[1].header['NAXIS2'] - 1:
                                            s.wavelmin, s.wavelmax = np.min(s.spec.raw.x), np.max(s.spec.raw.x)
                                            s.rescale(1e13)
                                            self.s.append(s)

                                else:
                                    for k in [0, 1]:
                                        s = Spectrum(self, name=filename + f'_{k}')
                                        r = range(k, hdulist[1].header['NAXIS2'], 2)
                                        s.set_data([np.concatenate([np.r_[prihdr['WAVELENGTH'][i], 2 * prihdr['WAVELENGTH'][i][-1] - prihdr['WAVELENGTH'][i][-2]] for i in r]),
                                                    np.concatenate([np.r_[prihdr['FLUX'][i], np.inf] for i in r]),
                                                    np.concatenate([np.r_[prihdr['ERROR'][i], np.inf] for i in r])])

                                        res = 0 if "SPECRES" not in hdulist[0].header else hdulist[0].header["SPECRES"]
                                        s.set_resolution(res)
                                        if k == 0:
                                            s.wavelmin, s.wavelmax = np.min(s.spec.raw.x), np.max(s.spec.raw.x)
                                            s.rescale(1e13)
                                            self.s.append(s)

                            s.rescale(1e13)

                        if 'COS' in hdulist[0].header['INSTRUME']:
                            prihdr = hdulist[1].data
                            print(prihdr['WAVELENGTH'])
                            s.set_data([[np.r_[prihdr['WAVELENGTH'][i], 2 * prihdr['WAVELENGTH'][i][-1] - prihdr['WAVELENGTH'][i][-2]] for i in range(hdulist[1].header['NAXIS2'])],
                                        np.concatenate([np.r_[prihdr['FLUX'][i], np.inf] for i in range(hdulist[1].header['NAXIS2'])]),
                                        np.concatenate([np.r_[prihdr['ERROR'][i], np.inf] for i in range(hdulist[1].header['NAXIS2'])]),
                                        ])
                            s.filename = filename
                            s.wavelmin, s.wavelmax = np.min(s.spec.raw.x), np.max(s.spec.raw.x)
                            s.set_resolution(20000)
                            s.rescale(1e13)
                            s.lsf_type = 'cos'
                            #self.s.append(s)
                            #s = Spectrum(self, name=filename+'_2')
                            #s.set_data([prihdr['WAVELENGTH'][1],
                            #            1e15 * prihdr['FLUX'][1],
                            #            1e15 * prihdr['ERROR'][1]
                            #            ])
                            # for l, f, e in zip(prihdr['WAVELENGTH'], prihdr['FLUX'], prihdr['ERROR']):
                            #    s.set_data()

                        if 'MagE' in hdulist[0].header['INSTRUME']:
                            for k in [0, 1]:
                                s = Spectrum(self, name=filename + f'_{k}')
                                print(len(hdulist) - 1)
                                r = range(k+1, len(hdulist) - 1, 2)
                                print([i for i in r])
                                print(np.concatenate([np.r_[hdulist[i].data['OPT_WAVE'], 2 * hdulist[i].data['OPT_WAVE'][-1] - hdulist[i].data['OPT_WAVE'][-2]] for i in r]))
                                s.set_data([np.concatenate([np.r_[hdulist[i].data['OPT_WAVE'], 2 * hdulist[i].data['OPT_WAVE'][-1] - hdulist[i].data['OPT_WAVE'][-2]] for i in r]),
                                            np.concatenate([np.r_[hdulist[i].data['OPT_COUNTS'], np.inf] for i in r]),
                                            np.concatenate([np.r_[hdulist[i].data['OPT_COUNTS_SIG'], np.inf] for i in r])])
                                if k == 0:
                                    s.wavelmin, s.wavelmax = np.min(s.spec.raw.x), np.max(s.spec.raw.x)
                                    self.s.append(s)

                        if 'ESPRESSO' in hdulist[0].header['INSTRUME']:
                            prihdr = hdulist[1].data
                            print(prihdr['WAVE'])
                            s.set_data([prihdr['WAVE'][0], prihdr['FLUX'][0] * 1e17, prihdr['ERR'][0] * 1e17])

                        if 'FUV' in hdulist[0].header['INSTRUME']:
                            prihdr = hdulist[1].data
                            s.set_data([prihdr['WAVE'], prihdr['FLUX'] * 1e17, prihdr['ERROR'] * 1e17])

                        if '4MOST' in hdulist[0].header['INSTRUME']:
                            prihdr = hdulist[1].data[0]
                            s.set_data([prihdr[0], prihdr[1] * 1e17, prihdr[2] * 1e17])
                        try:
                            if corr:
                                s.bary_vel = hdulist[0].header['HIERARCH ESO QC VRAD BARYCOR']
                                s.apply_shift(s.bary_vel)
                                s.airvac()
                                s.spec.raw.interpolate()
                        except:
                            pass
                        #except:
                        #    print('fits file was not loaded')
                        #    return False

                    elif 'TELESCOP' in hdulist[0].header:
                        if 'SDSS' in hdulist[0].header['TELESCOP']:
                            data = hdulist[1].data
                            DR9 = 0
                            print(hdulist[0].header['VERSUTIL'])
                            if 0 and hdulist[0].header['VERSUTIL'].strip() == 'v5_3_0':
                                l = 10 ** (hdulist[0].header['COEFF0'] + hdulist[0].header['COEFF1'] * np.arange(hdulist[1].header['NAXIS2']))
                                fl = hdulist[0].data[0]
                                cont = hdulist[0].data[1]
                                sig = hdulist[0].data[2]
                            elif DR9:
                                res_st = int((data.field('LOGLAM')[0] - self.LeeResid[0][0]) * 10000)
                                print('SDSS:', res_st)
                                #mask = data.field('MASK_COMB')[i_min:i_max]
                                l = 10 ** data.field('LOGLAM')
                                fl = data.field('FLUX')
                                cont = (data.field('CONT') * self.LeeResid[1][res_st:res_st+len(l)]) #/ data.field('DLA_CORR')
                                sig = (data.field('IVAR')) ** (-0.5) / data.field('NOISE_CORR')
                            else:
                                l = 10**data.field('loglam')
                                fl = data.field('flux')
                                sig = (data.field('ivar'))**(-0.5)
                                cont = data.field('model')
                                print(l, fl, sig)
                            s.set_data([l, fl, sig])
                            s.cont.set_data(l, cont)
                            s.set_resolution(2000)
                        elif 'FUSE' in hdulist[0].header['TELESCOP']:
                            x = np.linspace(hdulist[0].header['CRVAL1'], hdulist[0].header['CRVAL1']+hdulist[0].header['CDELT1']*(hdulist[0].header['NAXIS1']-1), hdulist[0].header['NAXIS1'])
                            print(len(x), len(hdulist[0].data))
                            s.set_data([x, hdulist[0].data*1e18])

                    elif 'ORIGIN' in hdulist[0].header:
                        if hdulist[0].header['ORIGIN'] == 'ESO-MIDAS':
                            prihdr = hdulist[1].data
                            s.set_data([prihdr['LAMBDA']*10, prihdr['FLUX'], prihdr['ERR']])
                    elif 'DLA_PASQ' in hdulist[0].header:
                        prihdr = hdulist[0].data
                        x = np.logspace(hdulist[0].header['CRVAL1'], hdulist[0].header['CRVAL1']+0.0001*hdulist[0].header['NAXIS1'], hdulist[0].header['NAXIS1'])
                        s.set_data([x, prihdr[0], prihdr[1]])
                    elif 'HIERARCH ESO PRO CATG' in hdulist[0].header:
                        #print(hdulist[0].header['HIERARCH ESO PRO CATG'])
                        if hdulist[0].header['HIERARCH ESO PRO CATG'] == 'MOS_SCIENCE_REDUCED':
                            x = np.linspace(hdulist[0].header['CRVAL1'], hdulist[0].header['CRVAL1']+hdulist[0].header['CDELT1']*(hdulist[0].header['NAXIS1']-1), hdulist[0].header['NAXIS1'])
                            s.set_data([x, hdulist[0].data[0]*1e20, np.ones_like(x)])
                    elif 'UVES_popler' in str(hdulist[0].header['HISTORY']):
                        header = hdulist[0].header
                        if 'LOGLIN' in header['CTYPE1']:
                            x = 10 ** (header['CRVAL1'] + np.arange(header['NAXIS1'] + 1 - header['CRPIX1']) * header['CD1_1'])

                        elif 'LINEAR' in header['CTYPE1']:
                            pass
                        err = hdulist[0].data[2, :]
                        err[err < 0] = 0
                        s.set_data([x, hdulist[0].data[0, :], err])
                    else:
                        prihdr = hdulist[1].data
                        if 1:
                            #print(prihdr.dtype)
                            s.set_data([prihdr['lam'] * 10000, prihdr['trans']])
                        else:
                            print(type(prihdr), prihdr.field('LAMBDA'))
                            s.set_data([prihdr['LAMBDA'] * 10000, prihdr['FLUX']])
                        try:
                            if 'BINTABLE' in hdulist[2].header['XTENSION']:
                                prihdr = hdulist[2].data
                                s.set_data([prihdr[0][0][:], prihdr[0][1][:], prihdr[0][2][:]])
                        except:
                            print('aborted Hadi fits')
                            return False
                elif filename.endswith('.hdf5'):
                    f = h5py.File(filename, 'r')
                    for l in ['wavelength', 'wave', 'x']:
                        if l in f.keys():
                            wave = np.asarray(f[l][:], dtype=float)
                    for l in ['flux', 'f', 'y']:
                        if l in f.keys():
                            flux = f[l][:]
                    err = None
                    for l in ['err', 'unc', 's', 'error', 'errors']:
                        if l in f.keys():
                            err = f[l][:]
                    if err is not None:
                        s.set_data([wave, flux, err])
                    else:
                        s.set_data([wave, flux])
                elif filename.endswith('.spec'):
                    data = np.genfromtxt(filename, comments='#', unpack=True)
                    if np.median(data[1]) < 1e-15:
                        s.set_data([data[0], data[1] * 1e17, data[2] * 1e17])
                    else:
                        s.set_data([data[0], data[1], data[2]])

                elif filename.endswith('.csv') and 'MALS' in filename:
                    data = np.genfromtxt(filename, comments='#', skip_header=1, unpack=True, usecols=(1, 2), delimiter=',')
                    s.set_data([data[0] / 1e9, data[1]])
                elif filename.endswith('.dat') and ',' in open(filename, 'r').readline():
                    data = np.genfromtxt(filename, comments='#', unpack=True, delimiter=',')
                    s.set_data([data[0], data[1], data[2]])
                else:
                    # >>>> other ascii data, e.g. .dat, .txt, etc
                    try:
                        data = np.genfromtxt(filename, comments='#', unpack=True)
                        if len(data) >= 3:
                            s.set_data(data[:3])
                        elif len(data) == 2:
                            s.set_data(data)
                        if len(data) == 4:
                            telluric = np.transpose(np.c_[data[0], data[3]])


                    except Exception as inst:
                        #print(type(inst))    # the exception instance
                        #print(inst.args)     # arguments stored in .args
                        #print(inst)
                        #print('aborted dat')
                        #raise Exception
                        return False

            else:
                s.set_data(spec, mask=mask)

            s.wavelmin = np.min(s.spec.raw.x)
            s.wavelmax = np.max(s.spec.raw.x)
            self.s.append(s)
            self.importTelluric(data=telluric)

        print("Append:", append)
        if append:
            self.plot.vb.disableAutoRange()
            self.s.redraw()
        else:
            self.s.draw()
            self.plot.vb.setRange(xRange=self.s.minmax())

        for name, status in self.filters_status.items():
            if status:
                m = max([max(s.spec.raw.y) for s in self.s])
                for f in self.filters[name]:
                    f.update(m)

    def importTelluric(self, filename=None, data=None):
        if filename is not None:
            data = np.genfromtxt(filename, unpack=True)

        print("telluric:", data, len(self.s))
        if data is not None:
            if len(self.s) > 0:
                print(len(self.s), self.s.ind)
                for attr in ['g_sky', 'g_sky_cont']:
                    if hasattr(self.s[self.s.ind], attr) and getattr(self.s[self.s.ind], attr) in self.vb.addedItems:
                            self.vb.removeItem(getattr(self.s[self.s.ind], attr))
                try:
                    #print(data.shape[0])
                    m = np.r_[np.diff(data[0]) != 0, True]
                    if data.shape[0] == 2:
                        self.s[self.s.ind].sky.set(data[0][m], data[1][m])
                    elif data.shape[0] > 3:
                        self.s[self.s.ind].sky.set(data[0][m], data[-1][m])
                    #print(self.s[self.s.ind].sky.y())
                    print('load')
                    self.s[self.s.ind].sky.interpolate()
                    print('inter')
                    self.s[self.s.ind].sky.filename = filename
                    self.s[self.s.ind].update_sky()
                    print('redraw')
                    #self.s.redraw()
                    #print('update')
                    #self.s[self.s.ind].update_sky()
                except:
                    self.sendMessage("Sky/telluric/nuisance absorption spectrum was not imported. Check the file")
        print(len(self.s), self.s.ind)


    def import2dSpectrum(self, filelist, spec=None, header=0, dir_path='', ind=None, append=False):

        if isinstance(filelist, str):
            filelist = [filelist]

        for line in filelist:
            filename = line.split()[0]
            if ind is not None:
                self.s.ind = ind
            else:
                self.s.append(Spectrum(self, name=filename))
                self.s.ind = len(self.s) - 1

            s = self.s[self.s.ind]

            if spec is None:
                if ':' not in filename:
                    filename = dir_path + filename
                if filename.endswith('.fits'):
                    with fits.open(filename, memmap=False) as hdulist:
                        if 'CUNIT1' in hdulist[0].header:
                            if 'A' in hdulist[0].header['CUNIT1']:
                                k = 1
                            elif 'nm' in hdulist[0].header['CUNIT1']:
                                k = 10
                        elif 'ORIGIN' in hdulist[0].header and 'sviewer' in hdulist[0].header['ORIGIN']:
                            k = 1
                        else:
                            k = 10
                        #print(k, 'CUNIT1' in hdulist[0].header)
                        x = np.linspace(hdulist[0].header['CRVAL1'] * k,
                                        (hdulist[0].header['CRVAL1'] + hdulist[0].header['CDELT1'] *
                                        (hdulist[0].header['NAXIS1']-1)) * k,
                                        hdulist[0].header['NAXIS1'])
                        y = np.linspace(hdulist[0].header['CRVAL2'],
                                        hdulist[0].header['CRVAL2'] + hdulist[0].header['CDELT2'] *
                                        (hdulist[0].header['NAXIS2']-1),
                                        hdulist[0].header['NAXIS2'])

                        for k in ['BARY', 'HELI']:
                            inds = [k in h for h in hdulist[0].header.keys()]
                            if any(inds):
                                setattr(self, k.lower(), hdulist[0].header[np.where(inds)[0][0]])
                                #print(x)
                        #print('ORIGIN' in hdulist[0].header, 'sviewer' in hdulist[0].header['ORIGIN'])
                        if 'INSTRUME' in hdulist[0].header and 'XSHOOTER' in hdulist[0].header['INSTRUME']:
                            err, mask = None, None
                            factor = 1 #1e17
                            for h in hdulist[1:]:
                                if 'EXTNAME' not in h.header or h.header['EXTNAME'].strip() == 'ERRS':
                                    err = h.data * factor
                                elif h.header['EXTNAME'].strip() == 'QUAL':
                                    mask = h.data.astype(bool)
                            #print(err, mask)
                            s.spec2d.set(x=x, y=y, z=hdulist[0].data * factor, err=err, mask=mask)

                        elif 'INSTRUME' in hdulist[0].header and hdulist[0].header['INSTRUME'] == 'SCORPIO-2':
                            err, mask = None, None
                            s.spec2d.set(x=x, y=y, z=hdulist[0].data * 1e17, err=err, mask=mask)

                        if 'ORIGFILE' in hdulist[0].header and 'VANDELS' in hdulist[0].header['ORIGFILE']:
                            s.spec2d.set(x=x, y=y[:-1], z=hdulist[0].data[:-1, :])

                        if 'ORIGIN' in hdulist[0].header and 'sviewer' in hdulist[0].header['ORIGIN']:
                            err, mask, cr, sky, sky_mask, trace = None, None, None, None, None, None
                            for h in hdulist[1:]:
                                print(h.header['EXTNAME'])
                                if h.header['EXTNAME'].strip() == 'err':
                                    err = h.data
                                if h.header['EXTNAME'].strip() == 'mask':
                                    mask = h.data.astype(bool)
                                if h.header['EXTNAME'].strip() == 'cr':
                                    cr = h.data
                                if h.header['EXTNAME'].strip() == 'sky':
                                    sky = h.data
                                if h.header['EXTNAME'].strip() == 'sky_mask':
                                    sky_mask = h.data
                                if h.header['EXTNAME'].strip() == 'trace':
                                    trace = h.data
                            s.spec2d.set(x=x, y=y, z=hdulist[0].data, err=err, mask=mask)
                            if cr is not None:
                                s.spec2d.cr = image(x=x, y=y, mask=cr)
                            if sky is not None:
                                if sky_mask is None:
                                    if cr is not None:
                                        sky_mask = cr
                                    elif mask is not None:
                                        sky_mask = mask
                                s.spec2d.sky = image(x=x, y=y, z=sky, mask=sky_mask)
                            if trace is not None:
                                s.spec2d.trace = trace

                elif filename.endswith('.dat'):
                    s.spec2d = None

            else:
                s.spec2d.set(x=spec[0], y=spec[1], z=spec[2])

        if s.spec2d is not None:
            s.spec2d.filename = filename

        self.s.redraw()

    def showExportDialog(self):

        fname = QFileDialog.getSaveFileName(self, 'Export spectrum', self.work_folder)

        if fname[0]:
            self.exportSpectrum(fname[0])
            self.statusBar.setText('Spectrum is written to ' + fname[0])

    def show2dExportDialog(self):

        self.exportData = ExportDataWidget(self, 'export2d')
        self.exportData.show()

    def showExportDataDialog(self):

        self.exportData = ExportDataWidget(self, 'export')
        self.exportData.show()
              
    def exportSpectrum(self, filename):
        if len(self.s[self.s.ind].spec.err()) > 0:
            data = np.c_[self.s[self.s.ind].spec.x(), self.s[self.s.ind].spec.y(), self.s[self.s.ind].spec.err()]
        else:
            data = np.c_[self.s[self.s.ind].spec.x(), self.s[self.s.ind].spec.y()]
        np.savetxt(filename, data, fmt='%10.5f')

    def export2dSpectrum(self, filename, opts=[]):
        s = self.s[self.s.ind].spec2d
        hdul = fits.HDUList()
        hdr = fits.Header()
        hdr['ORIGIN'] = 'sviewer'
        hdr['CRPIX1'] = 1.0
        hdr['CRVAL1'] = s.raw.x[0]
        hdr['CDELT1'] = s.raw.x[1] - s.raw.x[0]
        hdr['CTYPE1'] = 'LINEAR'
        hdr['CRPIX2'] = 1.0
        hdr['CRVAL2'] = s.raw.y[0]
        hdr['CDELT2'] = s.raw.y[1] - s.raw.y[0]
        hdr['CTYPE2'] = 'LINEAR'
        hdr['BARY'] = self.bary
        hdr['HELI'] = self.heli

        hdr_c = fits.Header()
        for opt in opts:
            hdr['EXTNAME'] = opt
            hdr_c['EXTNAME'] = opt
            if opt == 'spectrum':
                hdul.append(fits.ImageHDU(data=s.raw.z, header=hdr))
            if opt == 'err':
                if s.raw.err is not None:
                    hdul.append(fits.ImageHDU(data=s.raw.err, header=hdr))
            if opt == 'mask':
                if s.raw.mask is not None:
                    hdul.append(fits.ImageHDU(data=s.raw.mask.astype(int), header=hdr))
            if opt == 'cr':
                if s.cr is not None and s.cr.mask is not None:
                    hdul.append(fits.ImageHDU(data=s.cr.mask.astype(int), header=hdr))
            if opt == 'sky':
                if s.sky is not None:
                    hdul.append(fits.ImageHDU(data=s.sky.z, header=hdr_c))
                    hdr_c['EXTNAME'] = 'sky_mask'
                    hdul.append(fits.ImageHDU(data=s.sky.mask, header=hdr_c))
            if opt == 'trace':
                if s.trace is not None:
                    hdul.append(fits.ImageHDU(data=s.trace, header=hdr_c))
        hdul.writeto(filename, overwrite=True)

    def showImportListDialog(self):

        fname = QFileDialog.getOpenFileName(self, 'Import list of spectra', self.work_folder)

        if fname[0]:
            self.importListSpectra(fname[0])
            self.abs.redraw()
            self.statusBar.setText('Spectra are imported from list' + fname[0])

    def showImportFolderDialog(self):

        fname = QFileDialog.getExistingDirectory(self, "Select Directory", self.work_folder)
        print(fname)

        if fname:
            self.importFolder(fname)

    def importFolder(self, fname):
        self.work_folder = fname
        self.options('work_folder', self.work_folder)
        print([f for f in os.listdir(fname)])
        flist = os.listdir(fname)
        for fl in flist:
            if '#' in fl or '!' in fl or '%' in fl:
                flist.remove(fl)
        self.importSpectrum(flist, dir_path=fname+'/')
        self.plot.vb.enableAutoRange()
        self.abs.redraw()
        self.statusBar.setText('Spectra are imported from folder ' + fname[0])

    def importListSpectra(self, filename):
        
        self.importListFile = filename
        dir_path = os.path.dirname(filename)+'/'
        
        with open(filename) as f:
            flist = f.read().splitlines()
            for fl in flist:
                if '#' in fl or '!' in fl or '%' in fl:
                    flist.remove(fl)
            self.importSpectrum(flist, dir_path=dir_path)

        # correct error in the list given by parameters in the line
        for fl in flist:
            if len(fl.split()) > 2:
                if not any([x in fl for x in ['#', '!', '%']]):
                    for s in self.s:
                        if s.filename == fl.split()[0]:
                            s.spec.raw.err *= float(fl.split()[2])

        self.plot.vb.enableAutoRange()
        
    def showExportListDialog(self):

        fname = QFileDialog.getOpenFileName(self, 'Export list of spectra', self.work_folder)

        if fname[0]:
            self.statusBar.setText('Spectrum list are written to ' + fname[0])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   View routines
    # >>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def showExpList(self):
        if self.exp is None:
            self.exp = expTableWidget(self)
            self.exp.show()
        else:
            self.exp.close()

    def showResidualsPanel(self, show=None):
        if show is None:
            self.show_residuals = not self.show_residuals
        else:
            self.show_residuals = show
        self.options('show_residuals', bool(self.show_residuals))
        if self.show_residuals:
            self.residualsPanel = residualsWidget(self)
            self.splitter_plot.insertWidget(0, self.residualsPanel)
            if self.show_2d:
                self.splitter_plot.setSizes([450, 1000, 1000])
            else:
                self.splitter_plot.setSizes([450, 1900])
            if len(self.s) > 0:
                self.s.redraw()
        else:
            if hasattr(self, 'residualsPanel'):
                self.residualsPanel.hide()
                self.residualsPanel.deleteLater()
                del self.residualsPanel

    def show2dPanel(self, show=None):
        print(show)
        if show is None:
            self.show_2d = not self.show_2d
        else:
            self.show_2d = show
        self.options('show_2d', bool(self.show_2d))
        if self.show_2d:
            self.spec2dPanel = spec2dWidget(self)
            if self.show_residuals:
                self.splitter_plot.insertWidget(1, self.spec2dPanel)
                self.splitter_plot.setSizes([450, 1000, 1000])
            else:
                self.splitter_plot.insertWidget(0, self.spec2dPanel)
                self.splitter_plot.setSizes([1000, 1000])
            if len(self.s) > 0:
                self.s.redraw()
        else:
            if hasattr(self, 'spec2dPanel'):
                self.spec2dPanel.hide()
                self.spec2dPanel.deleteLater()
                del self.spec2dPanel

    def showPreferences(self):
        if self.preferences is None:
            self.preferences = preferencesWidget(self)
        else:
            self.preferences.close()

    def showLines(self, show=True):
        if self.showlines is None:
            self.showlines = showLinesWidget(self)
            if show:
                self.showlines.show()
        else:
            self.showlines.close()

    def takeSnapShot(self):
        self.snap = snapShotWidget(self)
        #self.snap.show()

    def switchReferenceAxis(self):
        self.plot.restframe = 1 - self.plot.restframe
        if self.plot.restframe:
            self.abs.set_reference()
        else:
            if hasattr(self.abs, 'reference'):
                self.abs.set_reference(self.abs.reference)
            else:
                self.abs.set_reference()
        self.plot.updateVelocityAxis()

    def switchFullScreen(self):
        self.fullscreen = 1 - self.fullscreen
        self.options('fullscreen', bool(self.fullscreen))
        if self.fullscreen:
            self.showFullScreen()
        else:
            self.showMaximized()
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   Observational routines
    # >>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    def chooseUVESSetup(self):
        if not self.UVESSetup.isChecked():
            self.UVESSetup_status = -1
        else:
            if hasattr(self, 'UVES_setup') and self.UVES_setup.gobject in self.vb.addedItems:
                self.vb.removeItem(self.UVES_setup.gobject)
            if hasattr(self, 'UVES_setup') and self.UVES_setup.label in self.vb.addedItems:
                self.vb.removeItem(self.UVES_setup.label)

            self.UVESSetups = UVESSetups()
            if (self.UVESSetup_status == -1):
                self.UVESSetup_status = 0
            self.UVES_setup = UVESSet(self, name=list(self.UVESSetups.items())[self.UVESSetup_status][0])

            m = max([max(s.spec.y()) for s in self.s])
            self.UVES_setup.set_gobject(m)
            self.vb.addItem(self.UVES_setup.gobject)
            self.vb.addItem(self.UVES_setup.label)

    def addUVESetc(self):
        fname = QFileDialog.getOpenFileName(self, 'Import ETC data', self.work_folder)
        self.work_folder = os.path.dirname(fname[0])
        self.options('work_folder', self.work_folder)
        data = np.genfromtxt(fname[0], skip_header=2, unpack=True)
        s = Spectrum(self, name=fname[0])
        s.set_data([data[6]*10, data[12], np.ones_like(data[6])])
        s.wavelmin = np.min(s.spec.x())
        s.wavelmax = np.max(s.spec.x())
        self.s.append(s)
        self.s.redraw()

    def showObservability(self):
        if self.observability is None:
            self.observability = observabilityWidget(self)
        else:
            self.observability.close()

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   Fit routines
    # >>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def normalize(self, state=None, action='normalize'):
        if self.normview != state:
            if state == None:
                self.normview = not self.normview
            else:
                self.normview = state
            self.s.normalize(action=action)
            if action == 'normalize':
                self.panel.normalize.setChecked(self.normview)
                self.panel.subtract.setEnabled(not self.normview)
                #self.panel.aod.setEnabled(not self.normview)
            elif action == 'subtract':
                self.panel.subtract.setChecked(self.normview)
                self.panel.normalize.setEnabled(not self.normview)
                self.panel.aod.setEnabled(not self.normview)
                self.panel.fitbutton.setEnabled(not self.normview)
            elif action == 'aod':
                self.panel.aod.setChecked(self.normview)
                self.panel.normalize.setEnabled(not self.normview)
                self.panel.subtract.setEnabled(not self.normview)
                self.panel.fitbutton.setEnabled(not self.normview)
            # self.parent.abs.redraw()
            x = self.plot.vb.getState()['viewRange'][0]
            self.plot.vb.enableAutoRange()
            try:
                self.plot.set_range(x[0], x[-1])
                self.abs.redraw()
            except:
                pass

    def setFitModel(self):
        if self.fitModel is None:
            self.fitModel = fitModelWidget(self)
            self.fitModel.show()
        else:
            self.fitModel.close()

    def chooseFitPars(self):

        if self.chooseFit is None:
            self.chooseFit = chooseFitParsWidget(self)
            self.splitter_fit.insertWidget(1, self.chooseFit)
            self.splitter_fit.setSizes([2500, 170])
            self.chooseFit.show()
        else:
            self.chooseFit.close()
            self.chooseFit = None

    def showFit(self, ind=-1, all=True):
        f = not self.normview
        if f:
            self.normalize()
        print("showFit", ind)
        self.s.prepareFit(ind=ind, all=all)
        self.s.calcFit(ind=ind, redraw=True)
        self.s.calcFitComps(ind=ind)
        self.s.chi2()
        if f:
            self.normalize()
        try:
            self.fitModel.refresh()
        except:
            pass
        self.s.redraw()

    def setFit(self, comp=-1):
        for par in self.fit.list():
            if comp == -1 or (par.sys is not None and par.sys.ind == comp):
                par.fit = par.vary
            else:
                par.fit = False
        print(self.fit.list_fit())

    def profileLikelihood(self, num=None, line=None):
        if not self.normview:
            self.normalize()

        if line is not None:
            print(line.l(), line.f(), line.name)
            if line.name in self.fit.sys[self.comp].sp.keys():
                line.logN = self.fit.sys[self.comp].sp[line.name].N.max
                line.b = self.fit.sys[self.comp].sp[line.name].b.max
                line.z = self.fit.sys[self.comp].z.val
                t = tau(line=line)
                t.calctau(verbose=False)
                xmin, xmax = t.x[0] - (t.x[-1] - t.x[0]), t.x[-1] + (t.x[-1] - t.x[0])

                self.normalize(False)
                ind = self.s.ind
                m = (self.s[ind].spec.x() > xmin) * (self.s[ind].spec.x() < xmax)
                s = Spectrum(self, name='temp', resolution=self.s[ind].resolution())
                s.set_data([self.s[ind].spec.x()[m], self.s[ind].spec.norm.y[m], self.s[ind].spec.norm.err[m]])
                s.wavelmin, s.wavelmax = np.min(s.spec.raw.x), np.max(s.spec.raw.x)
                s.cont_mask = np.ones_like(s.spec.raw.x, dtype=bool)
                s.cont.set_data(s.spec.raw.x, self.s[ind].cont.inter(s.spec.raw.x))
                self.s.append(s)
                #self.calc_cont()
                xmin, xmax = t.x[t.tau > 0.03][0], t.x[t.tau > 0.03][-1]
                self.s[self.s.ind].add_points(xmin, 1.5, xmax, -0.5, remove=False, redraw=False)
                self.normalize(True)
                ind, exp_ind, all = -1, self.s.ind, True
                #saved_pars = {str(p): p for p in self.fit.list()}
                self.fit.save()
                addinfo = {}
                for p in self.fit.list():
                    if str(p).split('_')[0] in ['b', 'N'] and int(str(p).split('_')[1]) == self.comp and len(str(p).split('_')) > 2 and str(p).split('_')[2].strip() in line.name:
                        if str(p).split('_')[0] == 'b':
                            addinfo = {str(p): self.fit.getValue(str(p), 'addinfo')}
                            self.fit.setValue(str(p), '', 'addinfo')
                        self.fit.setValue(str(p), 1, 'vary')
                        self.fit.setValue(str(p), 1, 'fit')
                    else:
                        self.fit.setValue(str(p), 0, 'fit')
        else:
            ind, exp_ind, all = -1, -1, True

        #self.s.prepareFit(ind=ind, all=all)

        if 1:
            if len(self.fit.list_fit()) == 1:
                p = str(self.fit.list_fit()[0])
                print(p)
                if num is not None:
                    pg = np.linspace(self.fit.getValue(p, 'min'), self.fit.getValue(p, 'max'), num)
                else:
                    pg = np.linspace(self.fit.getValue(p, 'min'), self.fit.getValue(p, 'max'), int((self.fit.getValue(p, 'max') - self.fit.getValue(p, 'min')) / self.fit.getValue(p, 'step'))+1)
                lnL = np.zeros(pg.size)
                for i, v in enumerate(pg):
                    print(i, v)
                    self.fit.setValue(p, v)
                    self.s.prepareFit(ind=ind, exp_ind=exp_ind, all=False)
                    self.s.calcFit(ind=ind, exp_ind=exp_ind, recalc=True)
                    lnL[i] = self.s.chi2(exp_ind=exp_ind)

                d = distr1d(pg, np.exp(np.min(lnL.flatten())-lnL))
                d.dopoint()
                d.plot(conf=0.683)

                self.fit.setValue(p, d.point)

            if len(self.fit.list_fit()) == 2:
                p1, p2 = str(self.fit.list_fit()[0]), str(self.fit.list_fit()[1])
                psv1, psv2 = self.fit.getValue(p1), self.fit.getValue(p2)

                if num is not None:
                    pg1 = np.linspace(self.fit.getValue(p1, 'min'), self.fit.getValue(p1, 'max'), num)
                    pg2 = np.linspace(self.fit.getValue(p2, 'min'), self.fit.getValue(p2, 'max'), num)
                else:
                    if 1:
                        pg1 = np.linspace(self.fit.getValue(p1, 'val') - 3 * self.fit.getValue(p1, 'step'), self.fit.getValue(p1, 'val') + 3 * self.fit.getValue(p1, 'step'), 20)
                        pg2 = np.linspace(self.fit.getValue(p2, 'val') - 3 * self.fit.getValue(p2, 'step'), self.fit.getValue(p2, 'val') + 3 * self.fit.getValue(p2, 'step'), 20)
                    else:
                        pg1 = np.linspace(self.fit.getValue(p1, 'min'), self.fit.getValue(p1, 'max'), int((self.fit.getValue(p1, 'max') - self.fit.getValue(p1, 'min')) / self.fit.getValue(p1, 'step'))+1)
                        pg2 = np.linspace(self.fit.getValue(p2, 'min'), self.fit.getValue(p2, 'max'), int((self.fit.getValue(p2, 'max') - self.fit.getValue(p2, 'min')) / self.fit.getValue(p2, 'step'))+1)
                print(pg1, pg2)
                lnL = np.zeros((pg2.size, pg1.size))
                for i1, v1 in enumerate(pg1):
                    print(i1)
                    self.fit.setValue(p1, v1, check=False)
                    for i2, v2 in enumerate(pg2):
                        self.fit.setValue(p2, v2, check=False)
                        self.s.prepareFit(ind=ind, exp_ind=exp_ind, all=True)
                        self.s.calcFit(ind=ind, exp_ind=exp_ind, recalc=True)
                        lnL[i2, i1] = self.s.chi2(exp_ind=exp_ind)

                d = distr2d(pg1, pg2, np.exp(np.min(lnL.flatten())-lnL))
                #d = distr2d(pg1, pg2, np.exp(- lnL / np.min(lnL.flatten())))
                d.plot_contour(conf_levels=[0.683, 0.954], xlabel=p1.replace('_', ' '), ylabel=p2.replace('_', ' '))

                self.fit.setValue(p1, d.point[0])
                self.fit.setValue(p2, d.point[1])

        self.s.calcFit(ind=ind, exp_ind=exp_ind, recalc=True, redraw=True)
        self.s.chi2(exp_ind=exp_ind)

        if line is not None:
            self.fit.load()
            for k, v in addinfo.items():
                self.fit.setValue(k, v, 'addinfo')
            self.normalize(False)
            #self.s.remove(self.s.ind)
            #self.s.redraw()
        plt.show()

    def fitLM(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.panel.fitbutton.setChecked(True)
        if self.chiSquare.text().strip() == '':
            print('showFit')
            self.showFit()

        if self.animateFit:
            if 1:
                self.thread = threading.Thread(target=self.fitAbs, args=(), kwargs={'comp': comp}, daemon=True)
                self.thread.start()
            else:
                self.fitprocess = Process(target=self.fitAbs)  # s, args=(comp,))
                self.fitprocess.daemon = True
                self.fitprocess.start()
        else:
            if self.options('fitType') == 'julia':
                self.fitJulia()
            else:
                self.fitAbs()
        self.panel.fitbutton.setChecked(False)
        QApplication.restoreOverrideCursor()

    def fitJulia(self, **kwargs):

        #self.reload_julia()
        self.s.prepareFit(all=False)
        #self.julia_spec = self.julia.prepare(self.s, self.julia_pars)
        res, unc, converged = self.julia.fitLM(self.julia_spec, self.fit.list(), self.julia_add, tieds=self.fit.tieds,
                                               opts=kwargs,
                                               blindMode=self.blindMode,
                                               method=self.options("fit_method"),
                                               grid_type=self.options("julia_grid"),
                                               grid_num=int(self.options("julia_grid_num")),
                                               binned=self.options("julia_binned"),
                                               telluric=self.options("telluric"),
                                               tau_limit=self.tau_limit,
                                               accuracy=self.accuracy,
                                               toll=float(self.options("fit_tolerance"))
                                               )

        s = self.fit.fromJulia(res, unc)

        if not converged:
            print("Minimization is not converged. You need to continue...")
            self.sendMessage("Minimization is not converged. You need to continue")
            s += "Minimization is not converged. You need to continue..."
            if self.blindMode:
                self.console.set("Minimization is not converged. You need to continue...")

        if not self.blindMode:
            self.console.set(s)

        self.showFit(all=False)

    def fitAbs(self, timer=True, redraw=True):
        t = Timer(verbose=True) if 1 else False

        def fcn2min(params):
            for p in params:
                name = params[p].name.replace('l4', '****').replace('l3', '***').replace('l2', '**').replace('l1', '*') #this line is added since lmfit doesn't recognize '*' mark\
                self.fit.setValue(name, self.fit.pars()[name].ref(params[p].value))
            self.fit.update(redraw=False)
            if timer:
                t.time(None)

            self.s.prepareFit(all=False)
            self.s.calcFit(recalc=True, redraw=self.animateFit)

            if timer:
                tim = t.time('out')
                if self.animateFit:
                    t.sleep(max(0, 0.02-tim))

            return self.s.chi()

        # create a set of Parameters
        params = Parameters()
        for par in self.fit.list():
            if not par.fit or not par.vary:
                par.unc = None
        for par in self.fit.list_fit():
            p = str(par).replace('****', 'l4').replace('***', 'l3').replace('**', 'l2').replace('*', 'l1')  #this line is added since lmfit doesn't recognize '*' mark
            value, pmin, pmax = par.ref()  #par.val, par.min, par.max
            if 'cf' in p:
                pmin, pmax = 0, 1
            params.add(p, value=value, min=pmin, max=pmax)

        # do fit, here with leastsq model
        minner = Minimizer(fcn2min, params,  nan_policy='propagate', calc_covar=True)
        #kws = {'options': {'maxiter': 2000}}
        if 0:
            chi0 = fcn2min(params)
            print('chi0', np.sum(chi0**2))
            for p in params.keys():
                params[p].value += (params[p].max - params[p].min) / 1000
                chi = fcn2min(params)
                print(p, (params[p].max - params[p].min) / 1000, np.sum(chi**2))
                if np.allclose(chi0, chi):
                    print('not shifted', p)

        result = minner.minimize(method=self.fit_method)

        print(result.message)
        self.console.set(result.message)
        print(dir(result))
        print(vars(result))
        # calculate final result
        #print(result.success, result.var_names, result.params, result.covar, result.errorbars, result.message)
        #final = data + result.residual

        # write error report

        #ci = conf_interval(minner, result)
        #printfuncs.report_ci(ci)

        self.showFit(all=False)

        self.fit.fromLMfit(result)

        if not self.blindMode:
            report_fit(result)
            self.console.set(fit_report(result))

        return fit_report(result)

    def fitMCMC(self):
        if self.MCMC is None:
            self.MCMC = fitMCMCWidget(self)
            self.MCMC.show()
        else:
            self.MCMC.activateWindow()

    def stopFit(self):
        """
        stop executing fit process
        """
        if self.fitprocess is not None:
            self.fitprocess.terminate()
            self.fitprocess.join()
            self.fitprocess = None
        if self.thread.is_alive():
            self.thread.join()

    def bootFit(self):
        print('boot')
        if 1:
            res = []
        else:
            with open('temp/res.pickle', 'rb') as f:
                res = pickle.load(f)
            print(res)
        if 1:
            for i in range(100):
                print(i)
                self.normalize(False)
                self.openFile("C:/science/Noterdaeme/HE0001/MgII_boot.spv")
                self.s.remove(self.s.ind)
                self.generate(template='const', z=self.fit.sys[0].z.val, fit=True, xmin=4054, xmax=4078, resolution=55000, snr=20,
                             lyaforest=0.0, lycutoff=False, Av=0.0, Av_bump=0.0, z_Av=0.0, redraw=True)
                self.normalize(True)
                self.console.exec_command('x 4054 4078')
                s = self.s[self.s.ind]
                s.add_points(4060.350, 1.5, 4060.60, -0.5, remove=False, redraw=False)
                s.add_points(4070.770, 1.5, 4071, -0.5, remove=False, redraw=False)
                # self.parent.s[i].add_points(self.mousePoint_saved.x(), self.mousePoint_saved.y(), self.mousePoint.x(), self.mousePoint.y(), remove=False)
                s.set_fit_mask()
                s.update_points()
                s.set_res()
                self.fit.cf_0.val = 0.95
                self.fitAbs(timer=False)
                print(self.fit.list_fit())
                print([self.fit.getValue(str(p)) for p in self.fit.list_fit()])
                res.append([self.fit.getValue(str(p)) for p in self.fit.list_fit()])
            with open('temp/res.pickle', 'wb') as f:
                pickle.dump(res, f)
        print(res)

    def showFitResults(self):
        if not self.blindMode:
            if self.fitResults is None:
                self.fitResults = fitResultsWidget(self)
                self.fitResults.show()
            else:
                self.fitResults.refresh()
        else:
            self.sendMessage("Blind mode is on. Disable it in Preference menu (F11)")

    def exportLine(self, folder):
        print("[Exporting optical depth of the selected line to the file:]", folder)
        print(folder)
        if self.abs.reference.line is not None:
            print(self.abs.reference.line.name, self.abs.reference.line.l())
            print(self.z_abs)
            res_saved = copy.copy(self.s[self.s.ind].resolution_linear)
            self.s[self.s.ind].set_resolution(0)
            self.s.prepareFit(ind=-1, exp_ind=self.s.ind, all=True, selected_line=self.abs.reference.line)
            #self.s.calcFit(ind=-1, exp_ind=self.s.ind, recalc=True)
            x, flux, bins, binned = self.julia.calc_spectrum(self.julia_spec[self.s.ind],
                                                            self.julia_pars,
                                                            comp=0,
                                                            grid_type=self.options("julia_grid"),
                                                            grid_num=int(self.options("julia_grid_num")),
                                                            binned=self.options("julia_binned"),
                                                            telluric=self.options("telluric"),
                                                            tau_limit=self.tau_limit,
                                                            accuracy=self.accuracy,
                                                            optical_depth=True)
            x = (np.asarray(x) / (self.abs.reference.line.l() * (1 + self.z_abs)) - 1) * 299792.46
            m = (x > -1000) * (x < 1000)
            print(folder + '/' + self.abs.reference.line.name + '_' + str(self.abs.reference.line.l()) + '.dat')
            np.savetxt(folder + '/' + self.abs.reference.line.name + '_' + str(self.abs.reference.line.l()) + '.dat', np.c_[x[m] , np.asarray(flux)[m]], fmt='%.3f')
            self.s[self.s.ind].set_resolution(*res_saved)
        else:
            self.sendMessage("Reference line is not specified")

    def resAnal(self):
        res = self.s[self.s.ind].res
        plot = 0
        self.residualsPanel.struct(clear=True)
        comps = []
        for ic, comp in enumerate(self.s[self.s.ind].fit_comp):
            fit = comp.line.norm
            inds = [ind for ind in argrelextrema(fit.y, np.less)[0] if fit.y[ind] < 0.95]
            #print(inds)
            indf = np.argwhere(np.abs(np.diff(fit.y < 0.98))).flatten()
            k, lines = 0, 0
            for ind in inds[:]:
                xind = [ind + k * np.min(k * (indf - ind)[k * (indf - ind) > 0]) for k in [1, -1]]
                xmin, xmax = np.min(xind), np.max(xind)
                m = (res.x >= fit.x[xmin]) * (res.x <= fit.x[xmax])
                #w = np.asarray([fit.y[np.argmin(np.abs(fit.x - x))] for x in res.x[m]])
                x, y = res.x[m] - fit.x[ind], res.y[m]
                if len(y) > 6:
                    lines += 1
                    #r = np.random.randn(len(x))
                    # r = gaussian_filter(r * 1.5, 0.7)
                    if plot:
                        fig, ax = plt.subplots()
                        ax.scatter(x, y)
                        #ax.scatter(x, r, color='0.9')
                        #ax.scatter(x, gaussian_filter(r, 0.7), color='0.5')
                        #ax.scatter(x, np.convolve(r, [1, 1, 1], 'same'), color='0.5')
                        ax.set_title("{:.2f}".format(fit.x[ind]))

                    fft = np.fft.fft(y)
                    boot = []
                    for i in range(1000):
                        boot.append(np.abs(np.fft.fft(np.random.randn(len(x)))))
                    boot = np.asarray(boot)

                    if 0:
                        for i in range(len(fft)):
                            fig, ax2 = plt.subplots()
                            ax2.hist(boot[:, i])
                            ax2.axvline(np.abs(fft)[i], color='tomato', lw=8, alpha=0.5)
                    #print(np.quantile(boot, 0.95, axis=0))

                    fft[:int(len(fft) * 4 / 5)] = 0
                    #fft[-2:] = 0
                    fft[np.abs(fft) < np.quantile(boot, 0.99, axis=0)] = 0
                    if np.sum(np.abs(fft)) > 0:
                        print(fit.x[ind], 'detected:', np.sum(np.abs(fft)))
                        k += 1
                        self.residualsPanel.struct(x=res.x[m], y=np.fft.ifft(fft).real)

                    if plot:
                        ax.plot(x, np.fft.ifft(fft).real)

            if k / len(lines) > 0.5:
                comps.append(ic)
                print(f'anomaly detected in {ic} component')

        if len(comps) > 0:
            self.statusBar.setText('stucture in residuals detected in components: ' + ' '.join([str(c) for c in comps]))
        else:
            self.statusBar.setText('there is no evident structure in residuals')

        if plot:
            plt.show()

    def aod(self):
        self.aodview = 1 - self.aodview
        if self.aodview:
            self.savenormview = self.normview
            if self.normview:
                self.normview = False
        self.normalize(action='aod')
        if not self.aodview:
            if self.savenormview:
                self.normalize()

    def aod_ratios(self):
        print('aod_ratios')
        if self.normview and len(self.plot.regions) == 1:
            range = self.plot.regions[0].getRegion()
            mask = (self.s[self.s.ind].spec.x() > range[0]) * (self.s[self.s.ind].spec.x() < range[1])
            v = vel_offset(self.s[self.s.ind].spec.x()[mask], self.abs.reference.line.l() * (self.z_abs + 1))
            print(v)

            for line in self.abs.activelist:
                print(line.line, line.line != self.abs.reference.line, line.exp)
                if line.line != self.abs.reference.line:
                    v_1 = vel_offset(self.s[line.exp].spec.x(), line.line.l() * (self.z_abs + 1))
                    y_1 = spectres.spectres(v_1, self.s[line.exp].spec.y(), v)
                    print(y_1)
                    m = (v_1 >= v[0]) * (v_1 <= v[-1])
                    if 1:
                        plt.step(v, np.log(self.s[line.exp].spec.y()[mask]) / np.log(y_1), where='mid')
                    else:
                        plt.step(v, -np.log(self.s[line.exp].spec.y()[mask]), where='mid')
                        plt.step(v, -np.log(y_1), where='mid')
                    print(self.abs.reference.line.f(), line.line.f())
                    plt.axhline(self.abs.reference.line.f() / line.line.f(), ls='--', color='k')
                    plt.axhline(1, ls='--', color='tab:red')
                    #plt.ylim([-0.1, max(1, self.abs.reference.line.f() / line.line.f()) * 1.3])
                    plt.show()

    def calc_cont(self):
        x = self.plot.vb.getState()['viewRange'][0]
        self.s[self.s.ind].calcCont(method='Smooth', xl=x[0], xr=x[-1], iter=3, clip=3, filter='hanning',
                                    window=int(np.sum((self.s[self.s.ind].spec.x() > x[0]) * (self.s[self.s.ind].spec.x() < x[-1])) / 2),
                                    )

    def fitCheb(self, typ='cheb'):
        """
        fit Continuum using specified model.
            - kind        : can be 'cheb', 'GP',
        """
        s = self.s[0]
        mask = (s.spec.x() > self.fit.cont_left) * (s.spec.x() < self.fit.cont_right)
        fit = s.fit.f(s.spec.x())
        mask = np.logical_and(fit > 0.05, s.fit_mask.x)
        x = s.norm.x[mask]
        y = s.norm.y[mask] / fit[mask]
        w = s.norm.err[mask] / fit[mask]
        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=w, fmt='o')

        if typ == 'cheb':
            cheb = np.polynomial.chebyshev.Chebyshev.fit(x, y, self.fit.cont_num - 1, w=1.0/w)
            poly = np.polynomial.chebyshev.cheb2poly([c for c in cheb])
            for i, c in enumerate(cheb):
                self.fit.setValue('cont_' + str(i), c)
            ax.plot(x, self.s[0].correctContinuum(x), '-r')

        plt.show()

    def fitGP(self):
        """
        fit Continuum using specified model.
            - kind        : can be 'cheb', 'GP',
        """
        s = self.s[0]
        mask = (s.spec.x() > self.fit.cont_left) * (s.spec.x() < self.fit.cont_right)
        fit = s.fit.f(s.spec.x())
        mask = np.logical_and(fit > 0.05, s.fit_mask.x)
        x = s.norm.x[mask]
        y = s.norm.y[mask] / fit[mask]
        w = s.norm.err[mask] / fit[mask]
        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=w, fmt='o')

        model = pyGPs.GPR()
        model.getPosterior(x, y)
        model.optimize(x, y)
        z = np.linspace(x[0], x[-1], len(x) * 2)
        model.predict(z)
        ym = np.reshape(model.ym, (model.ym.shape[0],))
        ys2 = np.reshape(model.ys2, (model.ys2.shape[0],))
        ax.plot(z, ym, color='g', ls='-', lw=3.)
        # print(z, ym, ym - 2. * np.sqrt(ys2), ym + 2. * np.sqrt(ys2))
        ax.fill_between(z, ym - 2. * np.sqrt(ys2), ym + 2. * np.sqrt(ys2),
                        facecolor='g', alpha=0.4, linewidths=0.0)
        # model.plot()

        plt.show()

    def fitExt(self):
        self.fitExtWindow = fitExtWidget(self)
        self.fitExtWindow.show()

    def fitwithCont(self, st, n=10, priors=[]):
        reg = []
        #priors = {'b_0_HDj0': a(14.8, 3.1, 3.1, 'd'), 'b_1_HDj0': a(3.0, 2.0, 2.0, 'd')}
        #priors = {'b_0_HDj0': a(3.9, 0.3, 0.3, 'd')}
        self.fit.save()
        for ind, s in enumerate(self.s):
            inds = np.where(np.diff(s.mask.x()) == 1)[0]
            if len(inds) > 0:
                for i_s, i_e in zip(inds[::2], inds[1::2]):
                    reg.append([ind, [i_s, i_e], np.copy(s.spec.norm.y[i_s:i_e]), np.mean(s.spec.norm.err[i_s:i_e])])
        res = []
        for i in range(n):
            print(i)
            self.fit.load()
            #for p in priors.keys():
            #    self.fit.setValue(p, priors[p].rvs(repr='dec')[0])
            for r in reg:
                self.s[r[0]].spec.norm.y[r[1][0]:r[1][1]] = r[2] + r[3] * np.random.randn()
            self.fitAbs(timer=False)
            print(self.fit.list_fit())
            print([self.fit.getValue(str(p)) for p in self.fit.list_fit()])
            res.append([self.fit.getValue(str(p)) for p in self.fit.list_fit()])

        for i in range(len(res[0])):
            x = np.asarray([r[i] for r in res])
            np.savetxt('temp/HD.dat', x)
            z = np.linspace(np.min(x) - (np.max(x) - np.min(x)) / 4, np.max(x) + (np.max(x) - np.min(x)) / 4, 100)
            kernel = gaussian_kde(x)
            d = distr1d(z, kernel(z), name=self.fit.list_fit()[i])
            d.stats(latex=2)
            d.plot(conf=0.683)
        plt.show()

    def fitGauss(self, kind='integrate'):
        """
        fit spectrum with simple gaussian line (emission)
        """
        print(kind)
        for s in self.s:
            n = np.sum(s.mask.x())
            if n > 0:
                mean = self.abs.reference.line.l() * (1 + self.z_abs)
                x, y, err = np.array(s.spec.x()[s.mask.x()], dtype=float), np.array(s.spec.y()[s.mask.x()], dtype=float), np.array(s.spec.err()[s.mask.x()], dtype=float)
                if self.normview:
                    cont = np.zeros_like(x, dtype=float)
                else:
                    cont = np.array(s.cont.y[s.mask.x()[s.cont_mask]], dtype=float)
                np.savetxt('output/fit_gauss_spec.dat', np.c_[s.spec.x(), s.spec.y(), s.spec.err(), s.mask.x()])
                np.savetxt('output/fit_gauss_cont.dat', np.c_[s.cont.x, s.cont.y])
                #np.savetxt('output/fit_gauss_data.dat', np.c_[x, y, err])
                ymax = np.max(y - cont)
                m = y - cont > ymax / 2
                if len(x[m]) > 1:
                    fwhm = (x[m][-1] - x[m][0])
                else:
                    fwhm = x[np.where(y == ymax)[0]] - x[np.where(y == ymax)[0]+1]
                sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
                amp = ymax * np.sqrt(2 * np.pi) * sigma
                mean = np.mean(x)
                #print(amp, sigma, mean, y - cont)

                def gaussian(x, amp, cen, disp):
                    """1-d gaussian: gaussian(x, amp, cen, disp)"""
                    return amp / ((2 * np.pi)**0.5 * disp) * np.exp(-(x - cen) ** 2 / (2 * disp ** 2))

                def gauss_integ(x_int, amp, cen, disp):
                    errf = erf((x_int - cen) / np.sqrt(2) / disp)
                    return amp / 2 * (errf[1:] - errf[:-1]) / np.diff(x_int)

                def fcn2min(params, x, y, err, cont):
                    return (y - (cont + gaussian(x, params['amp'].value, params['cen'].value, params['disp'].value))) / err

                def fcn2min_integ(params, x_int, y, err, cont):
                    return ((y - cont) - gauss_integ(x_int, params['amp'].value, params['cen'].value, params['disp'].value)) / err

                params = Parameters()
                names = ['amp', 'cen', 'disp']
                for name, value in zip(names, [amp, mean, sigma]):
                    params.add(name, value=value, min=0, max=np.inf)

                if 1:
                    if kind == 'integrate':
                        x_bin = np.insert(np.insert(x[:-1] + np.diff(x) / 2, 0, x[0] - (x[1] - x[0]) / 2), len(x), x[-1] + (x[-1] - x[-2]) / 2)
                        #print(np.diff(x), x_bin)
                        minner = Minimizer(fcn2min_integ, params, fcn_args=(x_bin, y, err, cont))
                    else:
                        minner = Minimizer(fcn2min, params, fcn_args=(x, y, err, cont))
                    result = minner.minimize(method='leastsq')
                    for par in params:
                        params[par].value = result.params[par].value
                        #print(fit_report(result))
                    self.console.set(fit_report(result))
                if 0:
                    minner = Minimizer(fcn2min, params, fcn_args=(x, y, err, cont), calc_covar=True)
                    result = minner.minimize(method='emcee')
                if 1:
                    def lnprob(params, x_int, y, err, cont):
                        return -0.5 * np.sum((((y - cont) - gauss_integ(x_int, params[0], params[1], params[2])) / err ) ** 2)

                    ndim, nwalkers, nsteps = 3, 100, 2000
                    p0 = np.asarray([result.params[par].value + np.random.randn(nwalkers) * result.params[par].stderr for par in params]).transpose()
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[x_bin, y, err, cont])
                    sampler.run_mcmc(p0, nsteps)
                    samples = sampler.chain[:, int(nsteps/2):, :].reshape((-1, ndim))
                    print('quantiles:', np.quantile(samples, [0.1, 0.5, 0.9], axis=0))
                    print(samples.shape)
                    samples = np.c_[samples, samples[:, 2] * 2 * np.sqrt(2 * np.log(2)) / samples[:, 1] * 299792.46]
                    print(samples.shape)
                    samples[:, 2] = samples[:, 0] / (2 * np.pi)**0.5 / samples[:, 2]
                    c = ChainConsumer()
                    names, truth = ['Area', 'Centroid', 'Amplitude', 'FWHM'], {'Area': params['amp'].value, 'Centroid': params['cen'].value, 'Amplitude': params['amp'].value / (2 * np.pi)**0.5 / params['disp'].value, 'FWHM': np.sqrt(2 * np.log(2)) * params['disp'].value * 2}
                    pd.DataFrame(data=samples, columns=names).to_pickle('output/mcmc_fitGauss.pickle')
                    c.add_chain(Chain(samples=pd.DataFrame(data=samples, columns=names),  # walkers=nwalkers,
                                    name="emission line posteriors",
                                    # parameters=names,
                                    smooth=True,
                                    # colors='tab:red',
                                    # cmap='Reds',
                                    # marker_size=2,
                                    plot_cloud=True,
                                    shade=True,
                                    sigmas=[0, 1, 2, 3],
                                    ))
                    #c.configure_truth(ls='--', lw=1., c='lightblue')  # c='darkorange')
                    if truth is not None:
                        c.add_truth(Truth(location=truth))
                    c.set_plot_config(PlotConfig(blind=False,
                            #flip=True,
                            #labels={"A": "$A$", "B": "$B$", "C": r"$\alpha^2$"},
                            #contour_label_font_size=12,
                            ))
                    figure = c.plotter.plot(figsize=(20, 20),
                                            # filename="output/fit.png",
                                            #display=True,
                                            )
                    table = c.analysis.get_latex_table(caption="Results for emission line fit", label="tab:results")
                    print(table)
                    self.console.set(table)
                    plt.show()
                amp, mean, sigma = result.params['amp'].value, result.params['cen'].value, result.params['disp'].value
                c = interp1d(x, cont, fill_value='extrapolate')
                x = np.linspace(x[0], x[-1], 100)
                x_bin = np.insert(x + (x[1] - x[0]) / 2, 0, x[0] - (x[1] - x[0]) / 2)
                #self.plot.add_line(x, c(x) + gauss_integ(x_bin, amp, mean, sigma))
                self.plot.add_line(x, c(x) + gaussian(x, amp, mean, sigma))

    def fitPowerLaw(self):
        if not self.normview:
            s = self.s[self.s.ind]
            x = np.log10(s.spline.x)
            y = np.log10(s.spline.y)

            p = np.polyfit(x, y, 1)

            x = np.logspace(np.log10(s.spec.x()[0]), np.log10(s.spec.x()[-1]), 100)
            y = np.power(10, p[1] + np.log10(x)*p[0])
            s.cont.set_data(x=x, y=y)
            s.redraw()

    def fitPoly(self, deg=None, typ='cheb'):
        """
        Fitting the selected points region with Polynomial function
        :param deg:  degree of polynmial function
        :param typ:  type of polynomial function, can be 'cheb' for Chebyshev, 'x' for simple polynomial
        :return: None
        """

        s = self.s[self.s.ind]
        x = s.spec.x()[s.fit_mask.x()]
        y = s.spec.y()[s.fit_mask.x()]
        w = s.spec.err()[s.fit_mask.x()]

        #self.fit.cont_fit = True
        #s.redraw()

        if deg is not None:
            self.options('polyDeg', deg)

        if typ == 'x':
            p = np.polyfit(x, y, self.polyDeg, w=1.0/w)
        elif typ == 'cheb':
            cheb = np.polynomial.chebyshev.Chebyshev.fit(x, y, self.polyDeg, w=1.0/w)

        x = np.linspace(x[0], x[-1], 100)

        if typ == 'x':
            y = np.polyval(p, x)
        elif typ == 'cheb':
            base = (x - x[0]) * 2 / (x[-1] - x[0]) - 1
            y = np.polynomial.chebyshev.chebval(base, [c for c in cheb])

        if self.normview:
            s.set_cheb(x=x, y=y)
        else:
            s.cont_mask = (s.spec.raw.x > x[0]) & (s.spec.raw.x < x[-1])
            s.cont.set_data(x=x, y=y)
            s.cont.interpolate()
            s.cont.set_data(x=s.spec.raw.x[s.cont_mask], y=s.cont.inter(s.spec.raw.x[s.cont_mask]))
            s.g_cont.setData(x=s.cont.x, y=s.cont.y)

    def fitMinEnvelope(self, res=200):

        s = self.s[self.s.ind]
        x, y = s.spec.x(), s.spec.y()
        if 1:
            mask = np.ones_like(x, dtype=bool)
            for r in self.regions:
                mask *= 1 - (x > r[0]) * (x < r[1])
            x, y = x[mask], y[mask]
        #w = s.spec.err()[s.fit_mask.x()]

        fig, ax = plt.subplots()
        ax.plot(x, y, '-b')

        # >>> convolve flux
        y = convolveflux(x, y, res=res, kind='direct')
        ax.plot(x, y, '--g')

        # >>> find local minima
        inds = np.where(np.r_[True, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:], True])[0]
        for i, c in zip(range(3), ['gold', 'magenta', 'red']):
            ax.plot(x[inds], y[inds], 'o', c=c)

            ys = sg.savitzky_golay(y[inds], window_size=5, order=3)
            inter = interp1d(x[inds], ys, bounds_error=False, fill_value=(ys[0], ys[-1]))
            ax.plot(x, inter(x), '-', c=c)

            inds = np.delete(inds, np.where((ys - y[inds]) / np.std(ys - y[inds]) < -1)[0])

        plt.show()

    def H2UpperLimit(self):
        """
        Estimate the upper limits on H2 column density
        :return:
        """
        f = not self.normview
        if f:
            self.normalize()

        def recalcfit(self):
            self.s.prepareFit(-1, all=True)
            self.s.calcFit(-1, redraw=True)
            self.s[self.s.ind].mask.set((self.s[self.s.ind].fit.inter(self.s[self.s.ind].spec.x()) < 0.99) * (self.s[self.s.ind].spec.y() - 1 < 2 * self.s[self.s.ind].spec.err()))
            self.s[self.s.ind].set_fit_mask()

        z_grid = np.linspace(self.fit.sys[0].z.min, self.fit.sys[0].z.max, int((self.fit.sys[0].z.max-self.fit.sys[0].z.min)/self.fit.sys[0].z.step)+1)
        N_grid = np.linspace(self.fit.sys[0].Ntot.min, self.fit.sys[0].Ntot.max, int((self.fit.sys[0].Ntot.max - self.fit.sys[0].Ntot.min) / self.fit.sys[0].Ntot.step)+1)
        z_save = self.fit.sys[0].z.val
        for N in N_grid:
            self.fit.setValue('Ntot_0', N)
            print(N)
            for z in z_grid:
                self.fit.setValue('z_0', z)
                recalcfit(self)
                print(z, np.sum(self.s[self.s.ind].mask.x()), np.sum(self.s.chi() > 0), np.sum(self.s.chi() > 2))
                if np.sum(self.s[self.s.ind].mask.x()) < 40 or (np.sum(self.s.chi() > 0) * 0.0455 > np.sum(self.s.chi() > 2)):
                    break
            if z == z_grid[-1]:
                break
            else:
                z_save = z
        else:
            print('Limit is not reached. Please checked Ntot boundaries.')
            return
        self.fit.setValue('z_0', z_save)
        self.fit.setValue('Ntot_0', N_grid[np.argwhere(N_grid == N)[0]-1])
        recalcfit(self)
        print(np.sum(self.s[self.s.ind].mask.x()), np.sum(self.s.chi() > 0), np.sum(self.s.chi() > 2))

        self.statusBar.setText('H2 upper limit is: ' + str(self.fit.sys[0].Ntot.val))

    def ExcitationTemp(self, levels=[0, 1, 2], E=None, ind=None, plot=True, ax=None):
        from ..excitation_temp import ExcitationTemp

        text = None
        for i, sys in enumerate(self.fit.sys):
            if ind is None or ind == i:
                if all(['H2j'+str(x) in sys.sp.keys() for x in levels]):
                    levels = [0, 1, 2]
                    # print(Temp.col_dens(num=4, Temp=92, Ntot=21.3))
                    n = [sys.sp['H2j'+str(x)].N.unc for x in levels]
                    if any([ni.val == 0 for ni in n]):
                        n = [a(sys.sp['H2j'+str(x)].N.val, 0, 0) for x in levels]
                    temp = ExcitationTemp('H2', n)

                if any(['COj' + str(x) in sys.sp.keys() for x in levels]):
                    #levels = np.arange(3)
                    levels = [int(k[k.index('j')+1:]) for k in sys.sp.keys() if k.startswith('CO') and int(k[k.index('j')+1:]) in levels]
                    print("levels:", levels)
                    n = [sys.sp['COj' + str(x)].N.unc for x in levels]
                    if any([ni.val == 0 for ni in n]):
                        n = [a(sys.sp['COj' + str(x)].N.val, 0, 0) for x in levels]
                    temp = ExcitationTemp('CO', n)
                temp.calc(method='emcee')
                #temp.temp, temp.ntot = temp.res['temp'].val, temp.res['ntot'].val
                temp.temp_to_slope()

                temp.plot_temp(ax=ax, energy='cm-1')

                if E is None:
                    E = temp.E
                elif isinstance(E, (float, int, np.floating)):
                    E = np.asarray([0, E])
                elif isinstance(E, list):
                    E = np.asarray(E)

                if ax is not None:
                    if isinstance(temp.slope, (float, int, np.floating)):
                        #ax.plot(E / 1.4388 * 1.5, temp.slope * E * 1.5 + temp.zero, '--k', lw=1.5)
                        text = [E[-1] / 1.4388 * 1.5, temp.slope * E[-1] * 1.5 + temp.zero,
                                r'T$_{' + ''.join([str(l) for l in levels]) + r'}$=' + "{:.1f}".format(temp.temp) + r'$\,$K']
                    elif temp.slope.type == 'm':
                        #ax.plot(E / 1.4388 * 1.5, temp.slope.val * E * 1.5 + temp.zero.val, '--k', lw=1.5)
                        text = [E[-1] / 1.4388 * 1.5, temp.slope.val * E[-1] * 1.5 + temp.zero.val,
                                r'T$_{' + ''.join([str(l) for l in levels]) + r'}$=' + temp.latex(f=0, base=0)+r'$\,$K']
                    elif temp.slope.type == 'u':
                        E = np.linspace(E[0], E[1], 5)
                        #ax.errorbar(E / 1.4388 * 1.5, (temp.slope.val - temp.slope.minus) * E * 1.5 + temp.zero.val, fmt='--k', yerr=0.1, lolims=E>0, lw=1.5, capsize=0, zorder=0 )
                        text = [E[-1] / 1.4388 * 1.5, temp.slope.val * E[-1] * 1.5 + temp.zero.val,
                                r'T$_{' + ''.join([str(l) for l in levels]) + r'}>' + '{:.0f}'.format(temp.temp.dec().val) + r'\,$K']
                    text[2] = ax.text(text[0], text[1], text[2], va='top', ha='right', fontsize=16)
        if plot:
            plt.show()

        if text is None:
            text = [0, 0, ax.text(0, 0, "not constrained", va='top', ha='right', fontsize=16)]
        return text

    def ExcDiag(self, temp=1):
        """
        Show H2 excitation diagram for the selected component
        """
        data_H2 = np.genfromtxt(self.folder + "/data/H2/energy_X.dat", comments='#', unpack=True)
        def data_CO(nu, j):
            we, we_xe, Be, De = 2169.81358, 13.28831, 1.93128087, 6.12147e-06
            return (nu + .5) * we - (nu + .5) ** 2 * we_xe + Be * j * (j + 1) - De * j * (j + 1)

        fig, ax = plt.subplots(figsize=(6, 7))
        num_sys = 0
        text = []
        species = ''
        for sys, color in zip(self.fit.sys, [c['color'] for c in plt.rcParams["axes.prop_cycle"]]):
            label = 'sys_'+str(self.fit.sys.index(sys)+1)
            label = 'z = '+str(sys.z.str(attr='val')[:8])
            if any(['H2' in name for name in sys.sp.keys()]):
                species = 'H2'
                num_sys += 1
                for nu, marker in zip([0, 1], ['o', 's']):
                    x, y = [], []
                    for sp in sys.sp:
                        if 'H2' in sp:
                            if 'v' in sp:
                                v, j = int(sp[sp.index('v')+1:]), int(sp[sp.index('j')+1:sp.index('v')])
                            else:
                                v, j = 0, int(sp[sp.index('j')+1:])
                            if v == nu:
                                m = np.logical_and(data_H2[0] == v, data_H2[1] == j)
                                x.append(float(data_H2[2][m]))
                                #x.append(self.atomic[sp].energy)
                                if sys.sp[sp].N.unc.val != 0:
                                    y.append(deepcopy(sys.sp[sp].N.unc).log()) # - np.log10(self.atomic[sp].statw()))
                                else:
                                    y.append(a(sys.sp[sp].N.val, 0.2, 0.2, 'l')) # - np.log10(self.atomic[sp].statw()))
                                y[-1].log()
                                y[-1].val = sys.sp[sp].N.val - np.log10(self.atomic[sp].statw())
                    arg = np.argsort(x)
                    x = np.array(x)[arg]
                    y = np.array(y)[arg]

                    if len(x) > 0:
                        p = ax.plot(x, [v.val for v in y], marker, markersize=1, color=color) #, label='sys_' + str(self.fit.sys.index(sys)))
                        ax.errorbar(x, [v.val for v in y], yerr=[[v.minus for v in y], [v.plus for v in y]], fmt=marker, color=p[0].get_color(), label=label)
                #temp = self.H2ExcitationTemp(levels=[0, 1], ind=self.fit.sys.index(sys), plot=False, ax=ax)
                if temp:
                    text.append(self.ExcitationTemp(levels=[0, 1], ind=self.fit.sys.index(sys), plot=False, ax=ax))

            if any([name.startswith('CO') for name in sys.sp.keys()]):
                species = 'CO'
                num_sys += 1
                x, y = [], []
                for sp in sys.sp:
                    if sp.startswith('CO'):
                        if 'v' in sp:
                            nu, j = int(sp[sp.index('v')+1:]), int(sp[sp.index('j')+1:sp.index('v')])
                        else:
                            nu, j = 0, int(sp[sp.index('j')+1:])
                        x.append(data_CO(nu, j) - data_CO(0, 0))
                        #x.append(self.atomic[sp].energy)
                        if sys.sp[sp].N.unc.val != 0:
                            y.append(deepcopy(sys.sp[sp].N.unc).log()) # - np.log10(self.atomic[sp].statw()))
                        else:
                            y.append(a(sys.sp[sp].N.val, 0.2, 0.2, 'l')) # - np.log10(self.atomic[sp].statw()))
                        y[-1].log()
                        y[-1].val = sys.sp[sp].N.val - np.log10(self.atomic[sp].statw())
                arg = np.argsort(x)
                x = np.array(x)[arg]
                y = np.array(y)[arg]

                p = ax.plot(x, [v.val for v in y], 'o', markersize=1) #, label='sys_' + str(self.fit.sys.index(sys)))
                ax.errorbar(x, [v.val for v in y], yerr=[[v.minus for v in y], [v.plus for v in y]],  fmt='o', color = p[0].get_color(), label=label)
                if temp:
                    text.append(self.ExcitationTemp(levels=[0, 1, 2, 3], ind=self.fit.sys.index(sys), plot=False, ax=ax))


        #if len(text) > 0:
        #    adjust_text([t[2] for t in text], [t[0] for t in text], [t[1] for t in text], ax=ax)
        ax.set_xlabel(r'Energy, cm$^{-1}$')
        ax.set_ylabel(r'$\log\, N$ / g')
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        #ax.xaxis.set_major_locator(self.x_locator)
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        #ax.yaxis.set_major_locator(self.y_locator)

        print(num_sys)
        if num_sys > 1:
            ax.legend(loc='best')
        fig.tight_layout()
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + f'/output/{species}_Exc.pdf', bbox_inches='tight')
        plt.show()
        self.statusBar.setText('Excitation diagram for {0:s} rotational level for {1:d} component is shown'.format(species, self.comp))

    def showMetalAbundance(self, species=[], component=1, dep_ref='ZnII', HI=a(21,0,0)):
        """
        Show metal abundances, metallicity and depletion based on the fit
        """
        colors = ['royalblue', 'orangered', 'seagreen', 'darkmagenta', 'skyblue', 'paleviotelred', 'chocolate']

        names = set()
        for sys in self.fit.sys:
            for sp in sys.sp.keys():
                if sp in species:
                    names.add(sp)

        refs = set(names)
        if 0:
            for sys in self.fit.sys:
                refs = refs & sys.sp.keys()
            inds = np.argsort([condens_temperature(name) for name in names])
            names = [names[i] for i in inds]

        names = list(names)
        print(names, refs)

        if len(refs) > 0:
            ref = list(refs)[0]

            print(names, refs, ref)

            for sys in self.fit.sys:
                for sp in sys.sp.keys():
                    if sys.sp[sp].N.unc is None or sys.sp[sp].N.unc.val == 0:
                        sys.sp[sp].N.unc = a(sys.sp[sp].N.val, 0, 0)

            if component:
                fig, ax = plt.subplots()

                for sys in self.fit.sys:
                    color = colors[self.fit.sys.index(sys)]
                    m = metallicity(ref, sys.sp[ref].N.unc, 22.0)
                    for i, sp in enumerate(names):
                        if sp in sys.sp.keys():
                            y = metallicity(sp, sys.sp[ref].N.unc, 22.0) / m
                            ax.scatter(i, y.val, c=color)
                            ax.errorbar([i], [y.val], yerr=[[y.minus], [y.plus]], c=color)
                ax.set_xticks(np.arange(len(names)))
                ax.set_xticklabels(names)
                plt.draw()

                fig, ax = plt.subplots()

                for sys in self.fit.sys:
                    color = colors[self.fit.sys.index(sys)]
                    if dep_ref in sys.sp.keys():
                        for k, v in sys.sp.items():
                            y = depletion(k, v.N.unc, sys.sp[dep_ref].N.unc, ref=dep_ref)
                            ax.scatter(names.index(k), y.val, c=color)
                            ax.errorbar([names.index(k)], [y.val], yerr=[[y.minus], [y.plus]], c=color)

                ax.set_xticks(np.arange(len(names)))
                ax.set_xticklabels(names)
                plt.show()

            sp = {}
            for name in names:
                print(name, name not in self.fit.total.sp.keys())
                if name not in self.fit.total.sp.keys():
                    sp[name] = a(0, 0, 'd')
                    for sys in self.fit.sys:
                        if name in sys.sp.keys():
                            print(name, sys.sp[name].N.unc)
                            sp[name] += sys.sp[name].N.unc
                    sp[name].log()
                    self.fit.total.addSpecies(name)
                    self.fit.total.sp[name].N.unc = sp[name]
                else:
                    sp[name] = self.fit.total.sp[name].N.unc

            res = {}
            for k, v in sp.items():
                try:
                    if dep_ref == '':
                        res[k] = [v, metallicity(k, v, HI)]
                    else:
                        res[k] = [v, metallicity(k, v, HI), depletion(k, v, sp[dep_ref], ref=dep_ref)]
                    print('SMA', k, res[k])
                except:
                    res[k] = [v]

            return res

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   1d spec routines
    # >>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def fitCont(self):
        if self.fitContWindow is None:
            self.fitContWindow = fitContWidget(self)
            self.fitContWindow.show()
        else:
            self.fitContWindow.close()

    def showCompositeQSO(self):
        if self.compositeQSO_status % 2 == 0:
            self.compositeQSO = CompositeSpectrum(self, kind='QSO', z=self.z_abs)
        else:
            if hasattr(self, 'compositeQSO'):
                self.compositeQSO.remove()

    def showCompositeGal(self):
        if self.compositeGal_status % 2 == 0:
            self.compositeGal = CompositeSpectrum(self, kind='Galaxy', z=self.z_abs)
        else:
            if hasattr(self, 'compositeGal'):
                self.compositeGal.remove()

    def rescale(self):
        if self.rescale_ind == 0:

            s = self.s[self.s.ind]

            #x = s.spec.raw.x[s.cont_mask]
            y = s.spec.raw.y[s.cont_mask] / s.cont.y
            #err = s.spec.raw.err[s.cont_mask] / s.cont.y

            window = 20
            mean = smooth(y, window_len=window, window='flat', mode='same')
            square = smooth(y**2, window_len=window, window='flat', mode='same')
            std = np.sqrt(square - mean**2)

            mask_0 = (y < 1.5 * std)
            mask_1 = (np.abs(y - 1) < 1.5 * std)

            self.s.append(Spectrum(self, 'error_0', data=[s.cont.x[mask_0], std[mask_0], std[mask_0]/np.sqrt(window)]))
            self.s.append(Spectrum(self, 'error_1', data=[s.cont.x[mask_1], std[mask_1], std[mask_1]/np.sqrt(window)]))
            self.s.redraw(len(self.s) - 1)

        if self.rescale_ind == 1:

            s0 = self.s[self.s.find('error_0')]
            s1 = self.s[self.s.find('error_1')]

            inter1 = interp1d(s1.cont.x, s1.cont.y, fill_value='extrapolate')
            if s0.cont.n > 0:
                inter0 = interp1d(s0.cont.x, s0.cont.y, fill_value='extrapolate')
            else:
                inter0 = inter1
            s = self.s[self.s.ind]
            y = s.spec.raw.y / s.cont.y
            s.spec.raw.err = s.cont.y * (inter0(s.spec.raw.x) * (1 - np.abs(y)) + inter1(s.spec.raw.x) * np.abs(y))

            self.s.redraw()

        self.rescale_ind = 1 - self.rescale_ind

    def stackLines(self):
        if not self.normview:
            self.normalize(True)

        s = self.s[self.s.ind]
        dv = 200
        x = np.linspace(-dv, dv, int(2 * dv / 299792.46 * s.resolution()) * 4)
        y, err, fs = np.zeros_like(x), np.zeros_like(x), 0
        for line in self.lines:
            print(line)
            name = ' '.join(line.split()[:2])
            for l in self.abs.lines:
                if name == str(l.line):
                    wavelength, f = l.line.l(), l.line.f()
            xv = (s.spec.x() / wavelength / (1 + self.z_abs) - 1) * 299792.46
            mask = (xv > x[0] - 299792.46 / s.resolution()) * (xv < x[-1] + 299792.46 / s.resolution())
            yi, erri = spectres.spectres((s.spec.x()[mask] / wavelength / (1 + self.z_abs) - 1) * 299792.46, s.spec.y()[mask], x, spec_errs=s.spec.err()[mask])
            y += (1 - yi) * f / erri ** 2
            fs += f
            err += 1.0 * f / erri ** 2
        print(x, y, err)
        y /= err
        err /= fs
        err = 1 / np.sqrt(err)
        if self.normview:
            self.normalize(False)
        s = Spectrum(self, name='stack')
        s.set_data([x, 1 - y, err])

        s.wavelmin = np.min(s.spec.raw.x)
        s.wavelmax = np.max(s.spec.raw.x)
        self.s.append(s)
        self.plot.vb.disableAutoRange()
        self.s.redraw()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   2d spec routines
    # >>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def extract2d(self):

        if self.extract2dwindow is None:
            self.extract2dwindow = extract2dWidget(self)
            self.extract2dwindow.show()
        else:
            self.extract2dwindow.close()

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   Combine routines
    # >>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def showExpListCombine(self):
        
        if hasattr(self, 'importListFile'):
            dtype = [('filename', str, 100), ('DATE-OBS', str, 20),
                     ('WAVELMIN', np.float64), ('WAVELMAX', np.float64),
                     ('EXPTIME', np.float64), ('SPEC_RES', int)]
            zero = ('', '', np.nan, np.nan, np.nan, 0)
            x = np.array([zero], dtype=dtype)
            i = 0
            with open(self.importListFile) as f:
                flist = f.read().splitlines() 
                dir_path = os.path.dirname(self.importListFile)+'/'
                for filename in flist:
                    if ':' not in filename:
                        filename = dir_path+filename
                    if '.fits' in filename:
                        if i:
                            x = np.insert(x, len(x), zero, axis=0)
                        i = 1
                        x[-1]['filename'] = os.path.basename(filename)
                        hdulist = fits.open(filename)
                        header = hdulist[0].header
                        for d in dtype[1:]:
                            try:
                                x[-1][d[0]] = header[d[0]]
                            except:
                                pass
            self.statusBar.setText('List of fits was loaded')
        
            if len(x) > 0:
                self.expListView = ShowListImport(self, 'fits')
                self.expListView.setdata(x)
                self.expListView.show()
            
    def selectCosmics(self):
        self.s.selectCosmics()
        
    def calcSmooth(self):
        self.s.calcSmooth()
        
    def coscaleExposures(self):
        self.s.coscaleExposures()

    def shiftExposure(self):
        pass

    def rescaleExposure(self):
        pass

    def rescaleErrs(self):
        print('rescale err', self.s.ind)
        fig, ax = plt.subplots()
        s = self.s[self.s.ind]
        x = s.spec.raw.x[s.cont_mask]
        print(s.cont_mask)
        f = (s.spec.raw.y[s.cont_mask] - s.cont.y) / s.spec.raw.err[s.cont_mask]
        ax.hist(f, bins=np.linspace(np.min(f), np.max(f), int((np.max(f) - np.min(f))/0.3)+1))
        m = np.abs(f) < 5
        mean, std = np.mean(f[m]), np.std(f[m])
        print(mean, std)
        xmin, xmax = mean - 1 * std, mean + 3 * std
        m = (xmin < f) * (f < xmax)
        kde = gaussian_kde(f[m])
        x = np.linspace(xmin, xmax, int(np.sqrt(len(f[m]))))
        fig, ax = plt.subplots()
        ax.plot(x, kde(x), '-r')

        if 1:
            n = len(x)

            def gauss(x, a, x0, sigma):
                return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
            popt, pcov = curve_fit(gauss, x, kde(x), p0=[2 * std**2, mean, std])

            y = gauss(x, popt[0], popt[1], popt[2])
            ax.plot(x, y, '-k')
            print(popt)

        self.s[self.s.ind].spec.raw.err *= popt[2]
        plt.show()

    def crosscorrExposures(self, i1, i2, dv=50):
        self.s[i1].crosscorrExposures(i2, dv=dv)

    def combine(self, typ='mean'):
        """
        combine exposures
        parameters:
            - typ        :  type of combine, can be either 'mean', 'weighted mean' or 'median' 
        """

        self.combineWidget = combineWidget(self)
        self.combineWidget.show()

    def rebin(self):
        
        self.rebinWidget = rebinWidget(self)
        self.rebinWidget.show()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   SDSS routines
    # >>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def loadSDSS(self, plate=None, MJD=None, fiber=None, name=None, z_abs=0, append=False, gal_ext=False, erosita=False):
        out = True
        if name is not None and len(name) > 0:
            name = name.replace('J', '').replace('SDSS', '').replace(':', '').replace('−', '-').strip()
            ra, dec = (name[:name.index('+')], name[name.index('+'):]) if '+' in name else (name[:name.index('-')], name[name.index('-'):])
            ra, dec = hms_to_deg(ra), dms_to_deg(dec)
        else:
            plate, MJD, fiber = int(plate), int(MJD), int(fiber)
            query = aqsdss.SDSS.query_specobj(plate=plate, fiberID=fiber, mjd=MJD)
            if query is not None and len(query) == 1:
                ra, dec = query[0]['ra'], query[0]['dec']
            else:
                gal_ext = 0

        if gal_ext:
            Av_gal = 3.1 * sfdmap.SFDMap(self.SFDMapPath).ebv(ra, dec)
        else:
            Av_gal = 0

        print('RA:', ra, "DEC:", dec)
        print('Av_gal:', Av_gal)

        if erosita:
            print('erosita')
            try:
                filename = os.path.dirname(self.ErositaFile) + '/spectra/spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(int(plate), int(MJD), int(fiber))
                if os.path.exists(filename):
                    qso = fits.open(filename)
                    ext = add_ext(10 ** qso[1].data['loglam'][:], Av_gal)
                    self.importSpectrum('spec-{0:05d}-{1:05d}-{2:04d}'.format(plate, MJD, fiber),
                                        spec=[10 ** qso[1].data['loglam'][:], qso[1].data['flux'][:] / ext, np.sqrt(1.0 / qso[1].data['ivar'][:]) / ext],
                                        mask=qso[1].data['and_mask'])
                resolution = 1800
            except:
                out = False
        elif self.SDSScat == 'DR12':
            try:
                sdss = self.IGMspec['BOSS_DR12']

                if name is None or name == '':
                    ind = np.where((sdss['meta']['PLATE'] == plate) & (sdss['meta']['FIBERID'] == fiber))[0][0]
                else:
                    ind = np.argmin((sdss['meta']['RA_GROUP'] - ra) ** 2 + (sdss['meta']['DEC_GROUP'] - dec) ** 2)
                print(sdss['meta'][ind]['SPEC_FILE'].decode('UTF-8'))
                ext = add_ext(sdss['spec'][ind]['wave'], Av_gal)
                self.importSpectrum(sdss['meta'][ind]['SPEC_FILE'].decode('UTF-8'), spec=[sdss['spec'][ind]['wave'], sdss['spec'][ind]['flux'] / ext,
                                                 sdss['spec'][ind]['sig'] / ext], append=append)
                resolution = int(sdss['meta'][ind]['R'])
            except:
                out = False
        elif self.SDSScat == 'DR14':
            #try:
            sdss = self.SDSSDR14['meta']
            if name is None or name.strip() == '':
                ind = np.where((sdss['meta']['PLATE'] == plate) & (sdss['meta']['FIBERID'] == fiber))
                if len(ind[0]) > 0:
                    ind = np.where((sdss['meta']['PLATE'] == plate) & (sdss['meta']['FIBERID'] == fiber))[0][0]
                else:
                    self.sendMessage(f"There is no {plate}-{MJD}-{fiber} spectrum in database")
            else:
                ind = np.argmin((sdss['meta']['RA'] - ra) ** 2 + (sdss['meta']['DEC'] - dec) ** 2)
            plate, MJD, fiber = sdss['meta']['PLATE'][ind], sdss['meta']['MJD'][ind], sdss['meta']['FIBERID'][ind]
            spec = self.SDSSDR14['data/{0:05d}/{2:04d}/{1:05d}'.format(plate, MJD, fiber)]
            ext = add_ext(10**spec['loglam'][:], Av_gal)
            self.importSpectrum('spec-{0:05d}-{1:05d}-{2:04d}'.format(plate, MJD, fiber),
                                spec=[10**spec['loglam'][:], spec['flux'][:] / ext, np.sqrt(1.0/spec['ivar'][:]) / ext],
                                mask=(spec['and_mask'][:] != 0), append=append)
            resolution = 1800
            #except:
            #    print('error')
            #    out = False
        elif self.SDSScat == 'DR9Lee':
            pass
        elif self.SDSScat == 'Astroquery':
            if self.SDSSquery is None:
                self.SDSSquery = aqsdss.SDSS
            if name is not None and len(name) > 0:
                qso = self.SDSSquery.get_spectra(coordinates=astropy.coordinates.SkyCoord(ra, dec, frame='icrs', unit='deg'), radius='4″')
            else:
                qso = self.SDSSquery.get_spectra(plate=plate, fiberID=fiber, mjd=MJD)
            #mask = qso[1].data['ivar'] > 0
            if qso is not None:
                print(qso[0])
                qso = qso[0]
                plate, MJD, fiber = qso[0].header['PLATEID'], qso[0].header['MJD'], qso[0].header['FIBERID']
                ext = add_ext(10 ** qso[1].data['loglam'][:], Av_gal)
                self.importSpectrum('spec-{0:05d}-{1:05d}-{2:04d}'.format(plate, MJD, fiber),
                                    spec=[10 ** qso[1].data['loglam'][:], qso[1].data['flux'][:] / ext, np.sqrt(1.0 / qso[1].data['ivar'][:]) / ext],
                                    mask=qso[1].data['and_mask'], append=append)
                resolution = 1800
            else:
                out = False
        else:
            out = False
        if out:
            self.s[-1].set_resolution(resolution)
            self.vb.enableAutoRange()
            self.z_abs = z_abs
            self.abs.redraw()
            self.statusBar.setText('Spectrum is imported: ' + self.s[-1].filename)
        return out

    def loadSDSSLee(self):
        self.LeeResid = np.loadtxt('C:/Science/SDSS/DR9_Lee/residcorr_v5_4_45.dat', unpack=True)
        self.importSDSSlist('C:/science/SDSS/DR9_Lee/BOSSLyaDR9_cat.fits')
        self.SDSSlist = QSOlistTable(self, 'SDSSLee')
        self.SDSSdata = self.SDSSdata[:]
        self.SDSSlist.setdata(self.SDSSdata)
        
    def showSDSSdialog(self):
        
        self.load_SDSS = loadSDSSwidget(self)
        self.load_SDSS.show()

    def importSDSSlist(self, filename=None):
        if filename is None:
            filename = QFileDialog.getOpenFileName(self, 'Load SDSS list...', self.SDSSfolder)[0]
            print(filename)
        if 1:
            if any([s in filename for s in ['.dat', '.txt', '.list', '.tsv']]):
                with open(filename) as f:
                    n = np.min([len(line.split()) for line in f])
                print(n)
                self.SDSSdata = np.genfromtxt(filename, names=True, dtype=None, usecols=range(n), comments='#', encoding=None)
                print(self.SDSSdata)
            elif '.fits' in filename:
                hdulist = fits.open(filename)
                data = hdulist[1].data
                self.SDSSdata = np.array(hdulist[1].data)
            elif '.csv' in filename:
                print(filename)
                self.SDSSdata = pd.read_csv(filename, keep_default_na=False).to_records(index=False)
            elif '.xlsx' in filename:
                print(filename)
                self.SDSSdata = pd.read_excel(filename, keep_default_na=False).to_records(index=False)
        else:
            self.SDSSdata = []
            data = np.genfromtxt(filename, names=True, dtype=None, unpack=True)
            for d in data:
                SDSSunit = SDSSentry(d['name'])
                for attr in data.dtype.names:
                    SDSSunit.add_attr(attr)
                    setattr(SDSSunit, attr, d[attr])
                self.SDSSdata.append(SDSSunit)

        self.show_SDSS_list()

    def show_SDSS_list(self):
        if hasattr(self, 'SDSSdata'):
            self.SDSSlist = QSOlistTable(self, 'SDSS')
            #self.SDSSlist.show()
            self.SDSSlist.setdata(self.SDSSdata)
        else:
            self.statusBar.setText('No SDSS list is loaded')

    def search_H2(self):
        search_H2(self, z_abs=self.z_abs)

    def show_H2_cand(self):
        self.mw = MatplotlibWidget(size=(200, 100), dpi=100)
        self.mw.move(QPointF(100, 100))
        self.mw.show()
        figure = self.mw.getFigure()
        self.s[self.s.ind].calc_norm()
        if self.s[self.s.ind].norm.n > 0:
            self.SDSS.load_spectrum([self.s[self.s.ind].norm.x, self.s[self.s.ind].norm.y, self.s[self.s.ind].norm.err])
            self.SDSS.H2_cand.z = self.z_abs
            self.SDSS.plot_candidate(fig=figure, normalized=True)
        self.mw.draw()



    def calc_SDSS_Stack_Lee(self, typ=['cont'], ra=None, dec=None, lmin=3.5400, lmax=4.0000, snr=2.5):
        """
        Subroutine to calculate SDSS QSO stack spectrum

        parameters:
            typ        -  what type of Stack
            ra         -  Right Ascension mask (tuple of two values e.g. (20, 40) or None)
            dec        -  Declination mask (tuple of two values e.g. (20, 40) or None)
            lmin       -  loglambda minimum boundary
            lmax       -  loglambda maximum boundary
            snr        -  SNR threshold
        """

        # >>> prepare stack class to write
        delta = 0.0001
        num = int((lmax-lmin)/delta+1)
        stack = Stack(num)
        l_min = 1041
        l_max = 1185
        calc_stack = 1
        for s in self.s:
            s.remove()
        self.s = Speclist(self)
        self.specview = 'line'
        self.s.append(Spectrum(self, name='stack_cont'))
        self.s.append(Spectrum(self, name='stack_poly'))
        self.s.append(Spectrum(self, name='stack_zero'))
        self.s.append(Spectrum(self, name='cont'))
        self.s.append(Spectrum(self, name='poly'))
        self.s.append(Spectrum(self, name='mask'))

        # >>> apply mask:
        print(self.SDSSdata.dtype, self.SDSSdata['RA'], self.SDSSdata['DEC'])
        mask = np.ones(len(self.SDSSdata), dtype=bool)
        print(np.sum(mask))
        plt.hist(self.SDSSdata['RA']/180-1)
        plt.hist(self.SDSSdata['DEC'])
        plt.show()

        #ra, dec = (140, 150), (0, 60)
        if ra is not None:
            mask *= (self.SDSSdata['RA'] > ra[0]) * (self.SDSSdata['RA'] < ra[1])
        print(np.sum(mask))
        if dec is not None:
            mask *= (self.SDSSdata['DEC'] > dec[0]) * (self.SDSSdata['RA'] < dec[1])
        print(np.sum(mask))
        if snr is not None:
            mask *= self.SDSSdata['SNR_LYAF'] > snr
        print(np.sum(mask))

        if 1:
            fig, ax = plt.subplots(subplot_kw=dict(projection='aitoff'))
            ax.scatter(self.SDSSdata['RA']/180, self.SDSSdata['DEC']/180-1, s=5, marker='+')
            plt.grid(True)
            plt.show()
        else:
            for i, s in enumerate(self.SDSSdata):
                print(i, 'SNR:', s['SNR_LYAF'])
                if s['SNR_LYAF'] > snr:
                    fiber = '{:04d}'.format(int(s['FIBERID']))
                    plate = s['PLATE']
                    MJD = s['MJD']
                    filename = self.SDSSLeefolder + str(plate) + '/' + 'speclya-{0}-{1}-{2}.fits'.format(plate, MJD, fiber)
                    hdulist = fits.open(filename)

                    z = s['Z_VI']
                    hdulist[1].verify('fix')
                    if np.isnan(float(hdulist[1].header['CH_CONT'])) == False:
                        data = hdulist[1].data
                        i_min = int(max(0, (np.log10((1+z)*l_min)-data.field('LOGLAM')[0])*10000))
                        i_max = int(max(0, (np.log10((1+z)*l_max)-data.field('LOGLAM')[0])*10000))
                        if i_max > i_min:
                            res_st = int((data.field('LOGLAM')[0] - self.LeeResid[0][0])*10000)
                            #print(res_st)
                            mask = np.logical_not(data.field('MASK_COMB')[i_min:i_max])
                            #l = (10**(data.field('LOGLAM')[i_min:i_max])/lya - 1)
                            #l = 10**(data.field('LOGLAM')[i_min:i_max])
                            l = data.field('LOGLAM')[i_min:i_max]
                            corr = self.LeeResid[1][i_min+res_st:i_max+res_st] / data.field('DLA_CORR')[i_min:i_max]
                            cont = data.field('CONT')[i_min:i_max]
                            fl = data.field('FLUX')[i_min:i_max]
                            sig = (data.field('IVAR')[i_min:i_max])**(-0.5) / data.field('NOISE_CORR')[i_min:i_max]

                            p = np.polyfit(l[mask], fl[mask], 1)  # , w=np.power(sig[mask], -1))
                            p1 = np.polyfit(l[mask], fl[mask], 1, w=np.power(sig[mask], -1))
                            poly = (p[0] * l + p[1])
                            poly_sig = (p1[0] * l + p1[1])

                            imin = int(round((l[0]-lmin)/delta))
                            imax = int(round((l[-1]-lmin)/delta)+1)
                            stack.mask[imin:imax] += mask
                            stack.cont[imin:imax] += cont * mask
                            stack.poly[imin:imax] += poly * mask
                            stack.sig[imin:imax] += fl / corr / cont * mask
                            stack.sig_p[imin:imax] += fl / corr / poly * mask
                            stack.zero[imin:imax] += (fl / corr - poly) * mask / np.std((fl / corr - poly) * mask, ddof=1)

                            ston = np.mean(fl / sig)

                            stack.mask_w[imin:imax] += mask * ston
                            stack.cont_w[imin:imax] += cont * mask * ston
                            stack.poly_w[imin:imax] += poly * mask * ston
                            stack.sig_w[imin:imax] += fl / corr / cont * mask * ston
                            stack.sig_p_w[imin:imax] += fl / corr / poly * mask * ston
                            stack.zero_w[imin:imax] += (fl / corr - poly) * mask * ston / np.std((fl / corr - poly) * mask, ddof=1)
                            if 0:
                                fig, ax = plt.subplots()
                                ax.plot(l, fl, label='flux')
                                ax.plot(l, sig, label='flux/cont')
                                ax.plot(l, mask, label='mask')
                                ax.plot(l, cont, label='cont')
                                ax.plot(l, poly, label='poly')
                                ax.plot(l, corr, label='corr')
                                ax.plot(l, (fl - poly) / np.std((fl - poly) * mask, ddof=1), label='zerot')
                                ax.legend(loc='best')
                                plt.show()
        stack.masked()
        l = np.power(10, lmin+delta*np.arange(stack.n))
        for i, attr in enumerate(stack.attrs + ['mask']):
            self.s[i].set_data([l, getattr(stack, attr)])
        self.s.redraw()
        self.vb.enableAutoRange(axis=self.vb.XAxis)
        self.vb.setRange(yRange=(-2, 2))

        stack.save(l)
        if 0:
            f_Stack = open('Stack_DR9.dat', 'w')
            for i in range(len(stack)):
                f_Stack.write("%6.4f %10.4f %8.0f\n" % (lmin+delta*i, stack[i], smask[i]))

    def calc_SDSS_DLA(self):
        s_fit = SDSS_fit(self, timer=True)
        analyse = [] #['pre', 'learn']
        prefix = '_'+str(len(self.SDSSdata))

        if 'pre' in analyse:
            for i, s in enumerate(self.SDSSdata):
                SNR_Thres = 3
                print(i, 'SNR:', s['SNR_LYAF'])
                if s['SNR_LYAF'] > SNR_Thres:
                    fiber = '{:04d}'.format(int(s['FIBERID']))
                    plate = s['PLATE']
                    MJD = s['MJD']
                    s_fit.add_spec(s['SDSS_NAME'], s['Z_VI'], plate, MJD, fiber)
            print(s_fit.data)

            s_fit.preprocess()
            s_fit.SDSS_prepare()
            s_fit.SDSS_remove_outliers()
            s_fit.savetofile(['Y', 'V', 'qso'], prefix=prefix)
        else:
            s_fit.load(['Y', 'V', 'qso'], prefix=prefix)

        # plot non-NaN pixels fraction:
        if 0:
            fig, ax = plt.subplots()
            ax.plot(s_fit.l, np.sum(~np.isnan(s_fit.Y), axis=0)/s_fit.Y.shape[0])
            plt.show(block=False)

        if 'learn' in analyse:
            s_fit.calc_mean()
            s_fit.calc_covar()
            s_fit.savetofile(['m', 'M', 'w'], prefix=prefix)
        else:
            s_fit.load(['m', 'K', 'w'], prefix=prefix)

        s_fit.toGUI('mean')
        s_fit.toGUI('w')
        s_fit.plot('covar')

    def show_filters(self, name='SDSS'):
        if self.filters[name] is None:
            if name == 'SDSS':
                self.filters[name] = [SpectrumFilter(self, f) for f in ['u', 'g', 'r', 'i', 'z']]
            elif name == 'Gaia':
                self.filters[name] = [SpectrumFilter(self, f) for f in ['G', 'G_BP', 'G_RP']]
            elif name == 'VISTA':
                self.filters[name] = [SpectrumFilter(self, f) for f in ['Y_VISTA', 'J_VISTA', 'H_VISTA', 'Ks_VISTA']]
            elif name == '2MASS':
                self.filters[name] = [SpectrumFilter(self, f) for f in ['J_2MASS', 'H_2MASS', 'Ks_2MASS']]
            elif name == 'UKIDSS':
                self.filters[name] = [SpectrumFilter(self, f) for f in ['Z_UKIDSS', 'Y_UKIDSS', 'J_UKIDSS', 'H_UKIDSS', 'K_UKIDSS']]
            elif name == 'WISE':
                self.filters[name] = [SpectrumFilter(self, f) for f in ['W1', 'W2', 'W3', 'W4']]
            elif name == 'GALEX':
                self.filters[name] = [SpectrumFilter(self, f) for f in ['NUV', 'FUV']]

        self.filters_status[name] = not self.filters_status[name]
        if self.filters_status[name]:
            try:
                m = max([max(s.spec.y) for s in self.s])
            except:
                m = 1
            for f in self.filters[name]:
                f.set_gobject(m)
                self.vb.addItem(f.gobject)
                self.vb.addItem(f.label)
        else:
            for f in self.filters[name]:
                if f.gobject in self.vb.addedItems:
                    self.vb.removeItem(f.gobject)
                if f.label in self.vb.addedItems:
                    self.vb.removeItem(f.label)

    def SDSSPhot(self):
        if 1:
            self.SDSS_phot = SDSSPhotWidget(self)
        else:
            if self.filters['SDSS'] is None:
                self.filters['SDSS'] = [SpectrumFilter(self, f) for f in ['u', 'g', 'r', 'i', 'z']]
            data = self.IGMspec['BOSS_DR12']
            num = len(data['meta']['Z_VI'])
            out = np.zeros([num, 5])
            for i, d in enumerate(data['spec']):
                out[i] = [f.get_value(x=d['wave'], y=d['flux']) for f in self.filters['SDSS']]
            out = np.insert(out, 0, data['meta']['THING_ID'], axis=1)
            np.savetxt('temp/sdss_photo.dat', out, fmt='%9i %.2f %.2f %.2f %.2f %.2f')

    def makeH2Stack(self, **kwargs): #beta=-0.9, Nmin=16, Nmax=22, norm=0, b=4, load=True, draw=True):
        return makeH2Stack(self, **kwargs)

    def H2StackFit(self, **kwargs):
        H2StackFit(self, **kwargs)

    def makeHIStack(self, **kwargs):
        return makeHIStack(self, **kwargs)

    def HIStackFitPower(self, **kwargs):
        HIStackFitPower(self, **kwargs)

    def HIStackFitGamma(self, load=True, draw=True):
        HIStackFitGamma(self, load=load, draw=draw)
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   Samples routines
    # >>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    def showXQ100list(self):
        
        if not hasattr(self, 'XQ100data'):
            self.XQ100data = load_QSO()
            dtype = [('id', str, 10), ('z_em', np.float64), ('HighRes', str, 1), ('cont', str, 1),
                    ('DLA', list), ('LLS', list)
                     ]
            x = []
            for i, q in enumerate(self.XQ100data):
                x.append(tuple([getattr(q, a[0]) for a in dtype]))
            #print(x)
            x = np.array(x, dtype=dtype)
            #print(x)
            self.XQ100data = x
            self.statusBar.setText('XQ100 list was loaded')
        if len(self.XQ100data) > 0:
            self.XQ100list = QSOlistTable(self, 'XQ100')
            self.XQ100list.setdata(self.XQ100data)            
    
    def showP94list(self):
        
        if not hasattr(self, 'P94data'):
            from H2_summary import load_P94
            self.P94data = load_P94()
            dtype = [('name', str, 20), ('z_em', np.float64), ('z_dla', str, 9),
                     ('HI', np.float64), ('H2', np.float64), ('Me', np.float64), ('SiII', np.float64),
                     ('SII', np.float64), ('ZnII', np.float64), ('FeII', np.float64)]
            x = []
            for i, q in enumerate(self.P94data):
                row = []
                for a in dtype:
                    if a[0] == 'z_dla':
                        row.append('{:.7f}'.format(q.z_dla))
                    elif a[0] in ['H2', 'HI', 'Me', 'SiII', 'SII', 'ZnII', 'FeII']:
                        row.append(q.e[a[0]].col.val)
                    else:
                        row.append(getattr(q, a[0]))
                print(row)
                x.append(tuple(row))
            #print(x)
            x = np.array(x, dtype=dtype)
            self.P94data = x
            self.statusBar.setText('P94 list was loaded')
        
        if len(self.P94data) > 0:
            self.P94list = QSOlistTable(self, 'P94')
            self.P94list.setdata(self.P94data)

    def showDLAlist(self):

        if not hasattr(self, 'DLAdata'):
            dtype = [('plate', int), ('MJD', int), ('fiber', int),
                     ('z_DLA', np.float64), ('HI', np.float64)]

            self.DLAdata = np.genfromtxt('C:/science/Noterdaeme/DLAs/DLA_catalogs/Garnett/Garnett_ESDLAs/DLA22.dat', unpack=False, dtype=dtype)
            self.statusBar.setText('P94 list was loaded')

        if len(self.DLAdata) > 0:
            self.DLAlist = QSOlistTable(self, 'DLA')
            self.DLAlist.setdata(self.DLAdata)

    def showLyalist(self):

        filename = self.options('Lyasamplefile')
        if os.path.isfile(filename):
            with open(filename) as f:
                n = np.min([len(line.split()) for line in f])

            self.Lyadata = np.genfromtxt(filename, names=True, unpack=False, usecols=range(n), delimiter='\t',
                                         dtype = ('U20', 'U20', 'U20', float, float, float, float, float, float, float, float, 'U100'),
                                         )
            self.statusBar.setText('Lya sample was loaded')

        if len(self.Lyadata) > 0:
            self.Lyalist = QSOlistTable(self, 'Lya', folder=os.path.dirname(filename))
            self.Lyalist.setdata(self.Lyadata)

    def showLyalines(self):

        filename = self.options('Lyasamplefile').replace('sample.dat', 'lines.dat')
        if os.path.isfile(filename):
            self.Lyalines = np.genfromtxt(filename, names=True, unpack=False,
                                         dtype = (float, float, float, float, float, float, float, 'U20', 'U30', 'U30'),
                                         )
            self.statusBar.setText('Lya lines data was loaded')

        if len(self.Lyalines) > 0:
            self.Lyalinestable = QSOlistTable(self, 'Lyalines', folder=os.path.dirname(filename))
            mask = np.ones_like(self.Lyalines['t'], dtype=bool)
            #mask = (self.Lyalines['t'] != 'b') * (self.Lyalines['chi'] < 1.3)
            self.Lyalinestable.setdata(self.Lyalines[mask])
            #self.data = add_field(self.Lyalines[mask], [('ind', int)], np.arange(len(self.Lyalines[mask])))

    def showVandels(self):
        data = np.genfromtxt(self.VandelsFile, delimiter=',', names=True,
                             dtype=('U19', '<f8', '<f8', '<f8', 'U12', '<f8', 'U10', '<f8', 'U9', '<i4', '<f8', '<f8', '<f8', '<f8', 'U24', 'U9', 'U9', 'U9', 'U50'))
        self.VandelsTable = QSOlistTable(self, 'Vandels', folder=os.path.dirname(self.VandelsFile))
        self.VandelsTable.setdata(data)

    def showKodiaq(self):
        data = np.genfromtxt(self.KodiaqFile, names=True,
                             dtype=('U17', 'U30', 'U25', 'U14', 'U17', 'U10', 'U15', 'U9', '<f8', '<f8', '<f8', '<i4'))
        self.KodiaqTable = QSOlistTable(self, 'Kodiaq', folder=os.path.dirname(self.KodiaqFile))
        self.KodiaqTable.setdata(data)

    def showUVES(self):
        self.UVESTable = QSOlistTable(self, 'UVES', folder=self.UVESfolder)
        data = np.genfromtxt(self.UVESfolder+'list.dat', names=True, delimiter='\t',
                             dtype=('U20', '<f8', '<f8', '<i4', '<i4', 'U5', 'U20', 'U200'))
        self.UVESTable.setdata(data)

    def showErosita(self):
        if self.ErositaWidget is None:
            self.ErositaWidget = ErositaWidget(self)

    def showMALS(self):
        self.MALSTable = QSOlistTable(self, 'MALS', folder=self.MALSfolder)
        #data = np.genfromtxt(self.MALSfolder + 'catalog.csv', names=True, delimiter=',', dtype=('U20', '<f8'))
        data = pd.read_csv(self.MALSfolder + 'catalog.csv', index_col=False).to_records(index=False)
        self.MALSTable.setdata(data)

    def showIGMspec(self, cat, data=None):
        self.IGMspecTable = IGMspecTable(self, cat)
        print(cat, data)
        self.IGMspecTable.setdata(data=data)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   Generation routines
    # >>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    def loadSDSSmedian(self):
        self.importSpectrum('data/SDSS/medianQSO.dat', header=2)
        self.z_abs = 0
        self.vb.enableAutoRange()
        self.abs.redraw()
        self.statusBar.setText('median VanDen Berk spectrum is imported')
   
    def loadHSTmedian(self):
        self.importSpectrum('data/SDSS/hst_composite.dat', header=2)
        self.z_abs = 0
        self.vb.enableAutoRange()
        self.abs.redraw()
        self.statusBar.setText('median HST spectrum is imported')
    
    def add_abs_system(self):
        self.generateAbs = GenerateAbsWidget(self)
        
    def add_dust_system(self):
        pass

    def generate(self, template='current', z=0, fit=True, xmin=3500, xmax=10000, resolution=2000, snr=None,
                 lyaforest=0.0, lycutoff=True, Av=0.0, Av_bump=0.0, z_Av=0.0, redraw=True):

        if self.normview:
            self.normalize(False)
        if template in ['Selsing', 'SDSS', 'VanDenBerk', 'HST', 'const']:
            s = Spectrum(self, name='mock')
            if template == 'Selsing':
                data = np.genfromtxt(self.folder + "data/SDSS/Selsing2016.dat", skip_header=0, unpack=True)
                fill_value = 'extrapolate'
            elif template == 'SDSS':
                data = np.genfromtxt(self.folder + "data/SDSS/medianQSO.dat", skip_header=2, unpack=True)
                fill_value = (1.3, 0.5)
            elif template == 'VanDenBerk':
                data = np.genfromtxt(self.folder + "data/SDSS/QSO_composite.dat", unpack=True)
                fill_value = 'extrapolate'
            elif template == 'HST':
                data = np.genfromtxt(self.folder + "data/SDSS/hst_composite.dat", skip_header=2, unpack=True)
                fill_value = 'extrapolate'
            elif template == 'const':
                data = np.ones((2, 10))
                data[0] = np.linspace(xmin / (1 + z), xmax / (1 + z), 10)
                fill_value = 1
            data[0] *= (1 + z)
            inter = interp1d(data[0], data[1], bounds_error=False, fill_value=fill_value, assume_sorted=True)
            s.set_resolution(resolution)
            bin = (xmin + xmax) / 2 / resolution / 4
            x = np.linspace(xmin, xmax, int((xmax - xmin) / bin))
            #debug(len(x), 'lenx')
            s.set_data([x, inter(x), np.ones_like(x) * 0.01])
            self.s.append(s)
            self.s.ind = len(self.s) - 1
        s = self.s[self.s.ind]
        s.cont.x, s.cont.y = np.copy(s.spec.raw.x), np.copy(s.spec.raw.y)
        s.cont.n = len(s.cont.y)
        s.cont_mask = np.logical_not(np.isnan(s.spec.raw.x))
        s.spec.normalize()
        s.mask.normalize()


        if lyaforest > 0 or Av > 0 or lycutoff:
            y = s.spec.raw.y
            if lyaforest > 0:
                y *= add_LyaForest(x=s.spec.raw.x, z_em=z, factor=lyaforest, kind='lines')
            if Av > 0:
                y *= add_ext_bump(x=s.spec.raw.x, z_ext=z_Av, Av=Av, Av_bump=Av_bump)
            if lycutoff:
                y *= add_LyaCutoff(x=s.spec.raw.x, z=z)
            s.spec.set(y=y)

        if fit and len(self.fit.sys) > 0:
            self.s.prepareFit(all=True)
            #s.findFitLines(all=all, debug=False)
            self.s.calcFit(recalc=True, redraw=False, timer=False)
            #s.calcFit(recalc=True, redraw=False)

            #s.findFitLines(all=True, debug=False)
            #s.calcFit_julia(recalc=True, redraw=False)
            s.fit.line.norm.interpolate(fill_value=1.0)
            s.spec.set(y=s.spec.raw.y * s.fit.line.norm.inter(s.spec.raw.x))

        if 0:
            def rebin(a, factor):
                n = a.shape[0] // factor
                return a[:n * factor].reshape(a.shape[0] // factor, factor).sum(1) / factor
            s.spec.set(x=rebin(s.spec.raw.x, 3), y=rebin(s.spec.raw.x, 3))
            s.cont.y = rebin(s.cont.y, 3)

        if snr is not None:
            s.spec.set(y=s.spec.raw.y + s.cont.y * np.random.normal(0.0, 1.0 / snr, s.spec.raw.n))
            s.spec.set(err=s.cont.y / snr)
            s.spec.raw.interpolate()

        if redraw:
            self.s.redraw()
            self.vb.enableAutoRange()

        for name, status in self.filters_status.items():
            if status:
                m = max([max(self.s[self.s.ind].spec.y())])
                for f in self.filters[name]:
                    f.update(m)

        if self.filters_status['SDSS']:
            d = {}
            for f in self.filters['SDSS']:
                d[f.name] = f.value
            return d

    def colorColorPlot(self):
        self.colorcolor = colorColorWidget(self)

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>
    # >>>   Help program routines
    # >>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    def info_howto(self):
        self.howto = infoWidget(self, 'How to', file=os.path.dirname(os.path.realpath(__file__)) + '/help/howto.txt')
        self.howto.show()
    
    def info_tutorial(self):
        self.tutorial = infoWidget(self, 'Tutorial', file=os.path.dirname(os.path.realpath(__file__)) + '/help/tutorial.txt')
        self.tutorial.show()

    def info_about(self):
        self.about = infoWidget(self, 'About program', file=os.path.dirname(os.path.realpath(__file__)) + '/help/about.txt')
        self.about.show()

    def closeEvent(self, event):
        
        if 1:
            msgBox = QMessageBox(self)
            msgBox.setText("Are you sure want to quit?")
            msgBox.setIcon(QMessageBox.Icon.Question)
            msgBox.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msgBox.setStyleSheet("QLabel{min-width:500 px; font-size: 22px;} QPushButton{width:200px; font-size:22px;}")
            msgBox.setBaseSize(QSize(900, 120))
            reply = msgBox.exec()
            if reply == QMessageBox.StandardButton.Yes:
                event.accept()
            else:
                event.ignore()   
            
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    #ex = Main()
    sys.exit(app.exec_())
