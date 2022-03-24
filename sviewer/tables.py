# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:45:49 2016

@author: Serj
"""
from astropy.io import fits
import fileinput
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QVariant
from PyQt5.QtWidgets import QHeaderView, QSizePolicy
from scipy.interpolate import interp1d

from spectro.sdss import SDSS
from .external import spectres
from .fit import fitPars
from .lyaforest import Lyaforest_scan, plotLyalines
from .stack import catalog
from .utils import add_field, slice_fields

def _defersort(fn):
    def defersort(self, *args, **kwds):
        # may be called recursively; only the first call needs to block sorting
        setSorting = False
        if self._sorting is None:
            self._sorting = self.isSortingEnabled()
            setSorting = True
            self.setSortingEnabled(False)
        try:
            return fn(self, *args, **kwds)
        finally:
            if setSorting:
                self.setSortingEnabled(self._sorting)
                self._sorting = None

    return defersort

class TableWidget(pg.TableWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @_defersort
    def setRow(self, row, vals):
        if row > self.rowCount() - 1:
            self.setRowCount(row + 1)
        for col in range(len(vals)):
            val = vals[col]
            item = self.itemClass(val, row)
            if col in self.edit_col:
                item.setEditable(True)
            else:
                item.setEditable(False)
            sortMode = self.sortModes.get(col, None)
            if sortMode is not None:
                item.setSortMode(sortMode)
            format = self._formats.get(col, self._formats[None])
            item.setFormat(format)
            self.items.append(item)
            self.setItem(row, col, item)
            item.setValue(val)  # Required--the text-change callback is invoked
            # when we call setItem.


class expTableWidget(TableWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
        #self.resize(100, 800)
        self.move(500, 300)
        self.setStyleSheet(open('config/styles.ini').read())
        #self.setWindowFlags(Qt.FramelessWindowHint)
        self.setSortingEnabled(False)
        self.setWidth = None
        self.update()
        self.show()
        self.setSortingEnabled(True)
        self.setWindowTitle('List of Exposures')
        #selectionModel = self.selectionModel()
        #selectionModel.selectionChanged.connect(self.cell_Clicked)
        self.cellDoubleClicked.connect(self.row_clicked)
        self.cellClicked.connect(self.cell_Clicked)
        self.cellChanged.connect(self.cell_Changed)
        #self.resizeColumnsToContents()
        self.horizontalHeader().sortIndicatorChanged.connect(self.rewrite)

    def update(self):
        dtype = [('filename', np.str_, 100), ('obs. date', np.str_, 30),
                 ('wavelmin', np.float_), ('wavelmax', np.float_),
                 ('resolution', np.int_)]
        zero = ('', '', np.nan, np.nan, 0)
        data = np.array([zero], dtype=dtype)
        self.edit_col = [4]
        for s in self.parent.s:
            print(s.filename, s.date, s.wavelmin, s.wavelmax, s.resolution)
            data = np.insert(data, len(data), np.array([('  '+s.filename+'  ', '  '+s.date+'  ', s.wavelmin, s.wavelmax, s.resolution)], dtype=dtype), axis=0)
        if len(self.parent.s) > 0:
            data = np.delete(data, (0), axis=0)
        self.setData(data)
        self.resizeColumnsToContents()
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        if len(self.parent.s) > 0:
            self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        if self.setWidth is None:
            self.setWidth = 120 + self.verticalHeader().width() + self.autoScrollMargin() * 2.5
            self.setWidth += sum([self.columnWidth(c) for c in range(self.columnCount())])
        self.resize(self.setWidth, self.rowCount()*40+140)
        if len(self.parent.s) > 0:
            self.selectRow(self.parent.s.ind)

    def keyPressEvent(self, event):
        super(TableWidget, self).keyPressEvent(event)
        key = event.key()

        if key == Qt.Key_Up or key == Qt.Key_Down:
            self.row_clicked()

        if key == Qt.Key_F3:
            self.close()

        if key == Qt.Key_Delete:
            self.parent.s.remove(self.parent.s.ind)

    def row_clicked(self):
        #self.parent.s.ind = self.currentItem().row()
        #self.parent.s.redraw(self.currentItem().row())
        #self.selectRow(self.parent.s.ind)
        self.parent.s.setSpec(self.currentItem().row())

    def cell_Clicked(self):
        if self.currentItem().row() != self.parent.s.ind:
            self.row_clicked()

    def cell_Changed(self):
        if self.parent.s[self.currentItem().row()].resolution != int(self.cell_value('resolution')):
            self.parent.s[self.currentItem().row()].resolution = int(self.cell_value('resolution'))
            print(self.parent.s[self.currentItem().row()].resolution)
            self.parent.s.calcFit(-1, recalc=True)
            self.parent.s.calcFitComps()
            self.parent.s.redraw()

    def cell_value(self, columnname):

        row = self.currentItem().row()

        # loop through headers and find column number for given column name
        headercount = self.columnCount()
        for x in range(0, headercount, 1):
            headertext = self.horizontalHeaderItem(x).text()
            if columnname == headertext:
                matchcol = x
                break

        cell = self.item(row, matchcol).text()  # get cell at row, col

        return cell

    def rewrite(self):
        names = [s.filename for s in self.parent.s]
        inds = [names.index(self.item(row, 0).text().strip()) for row in range(self.rowCount())]
        self.parent.s.rearrange(inds)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F2:
            self.close()
        super(expTableWidget, self).keyPressEvent(event)

    def closeEvent(self, event):
        self.parent.exp = None
        event.accept()

class QSOlistTable(pg.TableWidget):
    def __init__(self, parent, cat=None, folder=None, subparent=None, editable=True):
        super().__init__(editable=editable)
        self.setStyleSheet(open('config/styles.ini').read())
        self.parent = parent     
        self.subparent = subparent
        self.folder = folder
        self.setSortingEnabled(False)
        self.cat = cat
        self.resize(100, 800)
        self.move(500, 300)
        self.show()
        self.format = None
        if 'SDSS' == self.cat:
            self.setWindowTitle('SDSS list of files')
            self.format = {'SDSS_NAME': '%s', 'PLATE': '%5d', 'MJD': '%5d', 'FIBERID': '%4d',
                           'Z_QSO': '%.3f', 'Z_ABS': '%.3f', 'RA': '%.4f', 'DEC': '%.4f',
                           'u': '%.2f', 'g': '%.2f', 'r': '%.2f', 'z': '%.2f', 'i': '%.2f'}
                           #'f5': '%.6f'}
        if 'SDSSLee' == self.cat:
            self.setWindowTitle('SDSS DR9 Lee list of files')
        if 'XQ100' in self.cat:
            self.setWindowTitle('XQ100 list of files')
        if 'P94' in self.cat:
            self.setWindowTitle('P94 list of files')
        if 'DLA' in self.cat:
            self.setWindowTitle('DLA list of files')
        if 'Lya' == self.cat:
            self.setWindowTitle('Lya forest sample')
            self.format = {'name': '%s', 'date': '%s', 'sample': '%s', 'z_qso': '%.3f',
                           'lambda_min': '%.1f', 'lambda_max': '%.1f', 'z_min': '%.4f', 'z_max': '%.4f',
                           'SNR': '%3d', 'resolution': '%5d', 'DLA': '%1d', 'comment': '%s'}
        if 'Lyalines' == self.cat:
            self.setWindowTitle('Lya forest lines sample')
            self.format = {'z': '%.7f', 'N': '%.3f', 'Nerr': '%.3f', 'b': '%.3f', 'berr': '%.3f',
                           'snr': '%.2f', 'chi': '%.3f', 't': '%s', 'name': '%s', 'comment': '%s'}
            self.parent.console.exec_command('load HI')
            self.parent.fit.setValue('z_0', 1, 'min')
            self.parent.fit.setValue('z_0', 9, 'max')
            self.parent.fit.setValue('b_0_HI', 200, 'max')
            self.filename_saved = ''
        if self.cat == 'Vandels':
            self.setWindowTitle('Vandels catalog')
            if 0:
                self.parent.console.exec_command('hide all')
                for sp in ['HI', 'CIV', 'SiIV', 'CII']:
                    self.parent.console.exec_command('show '+sp)
        if self.cat == 'Kodiaq':
            self.setWindowTitle('KODIAQ DR2 catalog')
        if self.cat == 'UVES':
            self.setWindowTitle('UVES ADP QSO catalog')
        if self.cat == 'Erosita':
            self.setWindowTitle('Erosita-SDSS sample')
            self.format = {'SDSS_NAME_fl': '%s', 'PLATE_fl': '%5d', 'MJD_fl': '%5d', 'FIBERID_fl': '%4d', 'z': '%.6f',
                           'RA_fin': '%.7f', 'DEC_fin': '%.7f', 'srcname_fin': '%s', 'F_X_int': '%.4e',
                           'ML_FLUX_ERR_0': '%.4e', 'DET_LIKE_0': '%.3f'}
        if self.cat is None:
            self.setWindowTitle('QSO list of files')
        self.cellDoubleClicked.connect(self.row_clicked)
        self.cellPressed.connect(self.editCell)
        self.edit_item = [-1, -1]
        self.previous_item = None
        self.verticalHeader().setDefaultAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        if 'SDSS' in self.cat:
            self.contextMenu.addAction('Save spectra').triggered.connect(self.saveSpectra)
            self.contextMenu.addAction('Make stack').triggered.connect(self.makeStackSDSS)
        if self.cat == 'Lya':
            self.contextMenu.addSeparator()
            self.contextMenu.addAction('Save data').triggered.connect(self.saveData)
            self.contextMenu.addAction('Save continuum').triggered.connect(self.saveCont)
            self.contextMenu.addAction('Save norm').triggered.connect(self.saveNorm)
            self.contextMenu.addAction('Lya scan').triggered.connect(partial(self.scan, do=None))
            self.contextMenu.addAction('Lya fit').triggered.connect(partial(self.scan, do='fit'))
        if self.cat == 'Lyalines':
            self.contextMenu.addSeparator()
            self.contextMenu.addAction('Plot data').triggered.connect(self.plotLines)
            self.contextMenu.addAction('Save doublets').triggered.connect(self.saveDoublets)
            self.contextMenu.addAction('Update line').triggered.connect(self.updateLine)
            self.cellChanged.connect(self.saveLines)
        if self.cat == 'Vandels':
            self.contextMenu.addSeparator()
            self.contextMenu.addAction('Save continuum').triggered.connect(self.saveCont)
            self.contextMenu.addAction('Stack').triggered.connect(self.makeStack)
            self.cellChanged.connect(self.saveVandels)
        if self.cat == 'UVES':
            self.contextMenu.addSeparator()
            self.contextMenu.addAction('Show header').triggered.connect(self.showHeader)
            self.cellChanged.connect(self.saveUVES)
        if self.cat == 'Erosita':
            self.contextMenu.addSeparator()
            #self.contextMenu.addAction('Show header').triggered.connect(self.showHeader)
            #self.cellChanged.connect(self.saveUVES)

    def setdata(self, data):
        self.data = data
        #print(self.data.dtype)
        self.setData(data)
        if self.format is not None:
            for k, v in self.format.items():
                if self.columnIndex(k) is not None:
                    self.setFormat(v, self.columnIndex(k))
        self.resizeColumnsToContents()
        self.horizontalHeader().setResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setResizeMode(0, QHeaderView.ResizeToContents)
        self.horizontalHeader().setResizeMode(1, QHeaderView.ResizeToContents)
        w = 180 + self.verticalHeader().width() + self.autoScrollMargin()*1.5
        w += sum([self.columnWidth(c) for c in range(self.columnCount())])
        self.resize(w, self.size().height())
        if self.subparent is not None:
            self.subparent.resize(w, self.size().height()+50)
        self.setSortingEnabled(True)

    def editCell(self, row, col):
        self.edit_item = [row, col]

    def saveSpectra(self):
        if 'SDSS' in self.cat:
            # folder = QFileDialog.getExistingDirectory(self, "Select Directory", self.parent.work_folder)
            folder = r'C:\\Temp\\SDSS\\'
            with open(folder + 'list.list', 'w') as f:
                f.write(str(self.rowCount()) + '\n')
                f.write(folder + '\n')
                sdss = self.parent.IGMspec['BOSS_DR12']
                for i in range(self.rowCount()):
                    print(i)
                    fiber = '{:04d}'.format(int(self.cell_value('FIBER', row=i)))
                    plate = self.cell_value('PLATE', row=i)
                    MJD = self.cell_value('MJD', row=i)
                    if 0:
                        ind = np.where((sdss['meta']['PLATE'] == int(plate)) & (sdss['meta']['FIBERID'] == int(fiber)))[0][0]
                        self.parent.importSpectrum(sdss['meta'][ind]['SPEC_FILE'].decode('UTF-8'),
                                                   spec=[sdss['spec'][ind]['wave'], sdss['spec'][ind]['flux'],
                                                         sdss['spec'][ind]['sig']])
                    else:
                        print('C:/science/Noterdaeme/ESDLA/DR14/data/spSpec-{:}-{:05d}-{:04d}.fits'.format(MJD, int(plate), int(fiber)))
                        self.parent.importSpectrum(
                            'C:/science/Noterdaeme/ESDLA/DR14/data/spSpec-{:}-{:05d}-{:04d}.fits'.format(MJD, int(plate), int(fiber)))
                    try:
                        os.makedirs(folder + plate)
                    except:
                        pass
                    filename = r'\spec-{0}-{1}-{2}.dat'.format(plate, MJD, fiber)
                    self.parent.exportSpectrum(folder + plate + filename)
                    z_QSO, z_DLA, nHI = self.cell_value('z_QSO', row=i), self.cell_value('z_DLA', row=i), self.cell_value('NHI', row=i)
                    f.write('{:47} {:20} {:4} {:5} {:4} {:5} {:5}    {:5} \n'.format(plate + filename, plate + '-' + MJD + '-' + fiber,
                                                                                     plate, MJD, fiber, z_QSO, z_DLA, nHI))

    def makeStackSDSS(self):
        x = np.linspace(1000, 1600, 1800)
        cat = catalog(x, name='DLA', num=100)
        for idx in self.selectedIndexes():
            if idx.column() == 0:
                name = self.cell_value('SDSS_NAME', row=idx.row())[2:-1]
                print(name, self.cell_value('PLATE', row=idx.row()), self.cell_value('FIBERID', row=idx.row()))
                self.parent.loadSDSS(plate=self.cell_value('PLATE', row=idx.row()), fiber=self.cell_value('FIBERID', row=idx.row()))
                self.parent.openFile("C:/science/Noterdaeme/Coronographic/intervening/DR9/J" + name + '.spv')
                self.parent.normalize()
                spec = self.parent.s[self.parent.s.ind].spec
                cat.add(spec.x(), spec.y(), spec.err(), np.ones_like(spec.x(), dtype=bool), z=float(self.cell_value('f5', row=idx.row())))
        cat.finalize()
        cat.stack()
        print(cat.x, cat.fl)
        cat.save('output/stack.hdf5', stack=True)
        self.parent.importSpectrum('stack', spec=[cat.x, cat.fl])

    def saveData(self):
        with open(self.folder + '/sample_saved.dat', 'w') as f:
            if self.format is not None:
                f.write('\t'.join([i for i in self.format.keys()])+'\n')
                for d in self.data:
                    f.write(' \t'.join([(self.format[f] % i) for i, f in zip(d, self.format)])+'\n')
            else:
                data = np.empty([],dtype=self.data.dtype)
                for row in range(self.rowCount()):
                    rowdata = []
                    for column in range(self.columnCount()):
                        item = self.item(row, column)
                        if item is not None:
                            rowdata.append(item.text())
                        else:
                            rowdata.append('')
                    data = np.append(data, np.array([tuple([r for r in rowdata])], dtype=data.dtype))
                fmt = ' '.join(['{:'+''.join(i for i in r[1] if not i.isdigit()).replace('<', '').replace('U', 's')+'}' for r in data.dtype.descr])
                f.write('\t'.join([name for name in data.dtype.names])+'\n')
                for d in data:
                    print(d, tuple(d))
                    line = '\t'.join([f.format(d) for f, d in zip(fmt.split(), d)])
                    f.write(line+'\n')

    def saveCont(self, name=None):
        if name is None or isinstance(name, bool):
            name = 'name' if self.cat != 'Vandels' else 'id'
            if self.cell_value(name).strip() in self.parent.s[-1].filename:
                name = self.cell_value(name).strip()
            filename = self.folder + '/cont/' + name
            self.parent.save_opt = ['cont', 'others']
            self.parent.saveFile(filename, save_name=False)

    def saveNorm(self):
        self.parent.normalize(True)
        x, y, err = self.parent.s[-1].spec.norm.x, self.parent.s[-1].spec.norm.y, self.parent.s[-1].spec.norm.err
        mask = (x > 1215.6701 * (1 + float(self.cell_value('z_min')))) * (x < 1215.6701 * (1 + float(self.cell_value('z_max'))))
        for r in self.parent.plot.regions:
            mi, ma = r.getRegion()
            mask *= (x < mi) | (x > ma)
        if np.sum(mask) > 0:
            inds = np.append(np.where(np.diff(mask.astype(int)) < 0)[0], np.where(np.diff(mask.astype(int)) > 0)[0]+1)
            y[inds] = np.zeros(len(inds))
            err[inds] = 0.01 * np.ones(len(inds))
            data = np.c_[x[mask], y[mask], err[mask]]
            np.savetxt(self.folder+'/norm/' + self.cell_value('name') + '.dat', data, fmt='%10.4f %12.4f %12.4f')
            return data
        else:
            print('norm not saved for: '+ self.cell_value('name'))

    def scan(self, do=None):
        if do is None:
            do = 'corr' if len(self.selectedIndexes()) > 1 else 'all'
        rows = set()
        for idx in self.selectedIndexes():
            print('row', idx.row())
            if idx.row() not in rows:
                rows.add(idx.row())
                self.row_clicked(idx.row())
                data = self.saveNorm()
                Lyaforest_scan(self.parent, data.transpose(), do=do)
        plt.show()

    def plotLines(self):
        self.lyalines = plotLyalines(self)
        self.lyalines.set_data(slice_fields(self.data, ['N', 'b', 'Nerr', 'berr', 'comment']))
        self.lyalines.show()

    def saveDoublets(self):
        if self.cell_value('name').strip() in self.parent.s[-1].filename:
            filename = self.folder+'/cont/'+self.cell_value('name').strip().replace('.dat', '')
            self.parent.save_opt = ['cont', 'others']
            self.parent.saveFile(filename, save_name=False)

    def updateLine(self):
        row = self.currentItem().row()
        print(row)
        sp = self.parent.fit.sys[0].sp['HI']
        #for col, val in zip(['N', 'Nerr', 'b', 'berr'], [sp.N.val, sp.N.step, sp.b.val, sp.b.step]):
        #    self.set_cell_value(row, col, '{0:6.3f}'.format(val))
        for line in fileinput.input(self.folder + '/lines.dat', inplace=True):
            if len(line.split()) > 7:
                if line.split()[0] == self.item(row, 0).text() and line.split()[8] == self.item(row, 8).text():
                    s = '{0:9.7f}  {1:6.3f}   {2:5.3f}  {3:6.3f}   {4:5.3f}'.format(self.parent.fit.sys[0].z.val, sp.N.val, sp.N.step, sp.b.val,sp.b.step) + line[41:-1]
                    print(s)
                else:
                    print(line.replace('\n', ''))
        self.setRow(row, s.split())

    def saveLines(self, row, col):
        if row == self.edit_item[0] and col == self.edit_item[1]:
            if self.horizontalHeaderItem(col).text() == 'comment':
                print(row, col)
                for line in fileinput.input(self.folder + '/lines.dat', inplace=True):
                    if len(line.split()) > 7:
                        if line.split()[0] == self.item(row, 0).text() and line.split()[8] == self.item(row, 8).text():
                            print(line[:91] + self.item(row, col).text())
                        else:
                            print(line.replace('\n', ''))
            try:
                self.lyalines.set_data()
                filename = self.parent.options('Lyasamplefile').replace('sample.dat', 'lines.dat')
                if os.path.isfile(filename):
                    self.data =  np.genfromtxt(filename, names=True, unpack=True, dtype=(float, float, float, float, float, float, float, 'U20', 'U30', 'U30'))
                    self.lyalines.set_data(slice_fields(self.data, ['N', 'b', 'Nerr', 'berr', 'comment']))
            except:
                pass

    def saveVandels(self, row, col):

        if row == self.edit_item[0] and col == self.edit_item[1]:
            if self.horizontalHeaderItem(col).text() in ['type', 'lya', 'break', 'comment']:
                print(row, col)
                for line in fileinput.input(self.folder + '/VANDELS.csv', inplace=True):
                #for line in open(self.folder + '/VANDELS.csv'):
                    s = line.split(',')
                    if s[0] == self.item(row, 0).text():
                        print(','.join(s[:col] + [self.item(row, col).text()] + s[col+1:]).replace('\n', ''))
                    else:
                        print(line.replace('\n', ''))

    def makeStack(self):
        l = np.linspace(900, 2000, 2500)
        s, serr, sw = np.zeros_like(l), np.zeros_like(l), np.zeros_like(l)
        print(np.where(self.data['lya'] == 'n')[0])
        for i, d in enumerate(self.data):
            if (d['lya'] == 'n' and d['break'] == 'n') and d['zflg'] in [2, 3, 4, 14]:
                print(d['zspec'])
                if 0:
                    hdulist = fits.open(self.folder + '/' + d['FILENAME'])
                    prihdr = hdulist[1].data
                    x, f, err = prihdr[0][0] / (1+float(d['zspec'])), prihdr[0][1]*1e17, prihdr[0][2]*1e17
                else:
                    self.parent.importSpectrum(self.folder + '/' + d['FILENAME'])
                    if os.path.isfile(self.folder + '/cont/' + d['FILENAME'].replace('.fits', '.spv')):
                        self.parent.openFile(self.folder + '/cont/' + d['FILENAME'].replace('.fits', '.spv'), remove_regions=True, remove_doublets=True)
                        self.parent.normalize()
                    x, f, err = self.parent.s[self.parent.s.ind].spec.x() / (1 + float(d['zspec'])), self.parent.s[self.parent.s.ind].spec.y(), self.parent.s[self.parent.s.ind].spec.err()
                    if 1:
                        mask = np.ones_like(x, dtype=bool)
                        for r in self.parent.regions:
                            print('regions', r)
                            mask *= 1 - (x > r[0]) * (x < r[1])
                        x, f, err = x[mask], f[mask], err[mask]
                    print(x, f, err)
                    if 1:
                        m = (x > 1410) * (x < 1510)
                        n = np.mean(f[m])
                        ston = n / np.sqrt(np.mean(err[m]**2))
                        w = 1 / (ston**(-2) + 0.1**2)
                    print(n, ston, w)
                    #print(x, f, err)
                    mask = (l > x[2]) * (l < x[-3])
                    f, err = spectres.spectres(x, f, l[mask], spec_errs=err)
                    m = np.logical_not(err == 0)
                    s[mask] += m * (f / n) * w
                    err = (err / n)**(-2)
                    err[np.isinf(err)] = 0
                    serr[mask] += err * w
                    sw[mask] += m * w
        print(l, s, serr)
        self.parent.importSpectrum('stack', spec=[l, s / sw, (serr / sw)**(-.5)])

    def showHeader(self, row):
        hdul = fits.open(self.folder + self.cell_value('filename', row=row).strip())
        print(repr(hdul[0].header))
        from .sviewer import infoWidget
        self.header = infoWidget(self.parent, 'Header of '+self.cell_value('name', row=row).strip(), text=repr(hdul[0].header))
        self.header.show()

        if row == self.edit_item[0] and col == self.edit_item[1]:
            if self.horizontalHeaderItem(col).text() in ['z', 'z_DLA', 'Lyaf', 'comment']:
                for line in fileinput.input(self.folder + '/list.dat', inplace=True):
                    # for line in open(self.folder + '/VANDELS.csv'):
                    s = line[:-1].split('\t')

                    if s[0].strip() == self.item(row, 0).text():
                        lst = s[:col] + [self.item(row, col).text()] + s[col + 1:]
                        print('{:>20}\t{:>10}\t{:>10}\t{:>12}\t{:>12}\t{:>5}\t{:>20}\t{:5}'.format(*lst))
                    else:
                        print(line.replace('\n', ''))

    def row_clicked(self, row=None):

        colInd = self.currentItem().column()
        if row is not None:
            self.selectRow(row)

        load_spectrum = 0
        if 'SDSS' == self.cat:
            if colInd == 0:
                fiber_name = self.horizontalHeaderItem(np.where(['fiber' in self.horizontalHeaderItem(x).text().lower() for x in range(self.columnCount())])[0][0]).text()
                plate_name = self.horizontalHeaderItem(np.where(['plate' in self.horizontalHeaderItem(x).text().lower() for x in range(self.columnCount())])[0][0]).text()
                mjd_name = self.horizontalHeaderItem(np.where(['mjd' in self.horizontalHeaderItem(x).text().lower() for x in range(self.columnCount())])[0][0]).text()

                if 1:
                    self.parent.loadSDSS(plate=int(self.cell_value(plate_name)), MJD=int(self.cell_value(mjd_name)), fiber=int(self.cell_value(fiber_name)))
                    load_spectrum = 0
                else:
                    self.parent.importSpectrum('C:/science/Noterdaeme/ESDLA/DR14/data/spSpec-{:}-{:05d}-{:04d}.fits'.format(MJD, int(plate), int(fiber)))
                print('spectrum loaded')
                if 1:
                    ind = np.where((self.data[plate_name] == int(self.cell_value(plate_name))) & (self.data[fiber_name] == int(self.cell_value(fiber_name))))[0][0]
                    for attr in ['Z', 'Z_VI', 'Z_PCE', 'Z_CIV', 'z_em', 'Z_QSO']:
                        if attr in self.data.dtype.names:
                            self.parent.setz_abs(self.data[attr][ind])
                            break

                if 0:
                    self.parent.console.exec_command('load HIH2')
                    print(float(self.cell_value('NHI')), self.parent.fit.sys[0].sp['HI'].N.val)
                    self.parent.fit.sys[0].sp['HI'].N.val = float(self.cell_value('NHI'))
                    self.parent.fit.sys[0].sp['H2j0'].N.val = float(self.cell_value('H2')) - 0.2
                    self.parent.fit.sys[0].sp['H2j1'].N.val = float(self.cell_value('H2')) - 0.1

        if 'SDSSLee' == self.cat:
            if colInd == 0:
                fiber = '{:04d}'.format(int(self.cell_value('FIBERID')))
                plate = self.cell_value('PLATE')
                MJD = self.cell_value('MJD')
                filename = self.parent.SDSSLeefolder + plate + '/' + 'speclya-{0}-{1}-{2}.fits'.format(plate, MJD, fiber)
                #self.deleteSpectrum()
                self.parent.SDSS = SDSS(self.cell_value('SDSS_NAME'), plate, MJD, fiber, self.cell_value('Z_VI'))
                self.parent.SDSS.file = filename
                load_spectrum = 1
                
        if 'XQ100' in self.cat:
            XQ100comb = 0
            if self.horizontalHeaderItem(self.currentItem().column()).text() in ['id', 'DLA']:
                load_spectrum = 1
                if XQ100comb:
                    filename = self.parent.XQ100folder + str(self.cell_value('id')) + '/' + str(self.cell_value('id'))+ '_rescale.fits'
                else:
                    filename = self.parent.XQ100folder + str(self.cell_value('id')) + '/' + str(self.cell_value('id'))+ '.f'
                    filename = [filename.replace('.f', '_uvb.fits'), 
                                filename.replace('.f', '_vis.fits'),
                                filename.replace('.f', '_nir.fits')]
                    print(filename)
                try:
                    self.parent.setz_abs(self.cell_value('DLA')[1:-1].split(',')[0])
                except:
                    pass
                
        if 'P94' in self.cat:
            comb = 0
            if self.horizontalHeaderItem(self.currentItem().column()).text() in ['name']:
                load_spectrum = 1
                if comb:
                    filename = self.parent.P94folder + str(self.cell_value('name')) + '/' + str(self.cell_value('id'))+ '_rescale.fits'
                else:
                    filename = []
                    for ind in ['UVB', 'VIS', 'NIR']:
                        for file in os.listdir(self.parent.P94folder + 'fits/'):
                            if file.find(str(self.cell_value('name'))) > -1 and file.endswith(ind+'.fits'):
                                filename.append(self.parent.P94folder + 'fits/' + file)
                try:
                    self.parent.setz_abs(self.cell_value('z_dla')[1:-1].split(',')[0])
                except:
                    pass
            
            if self.horizontalHeaderItem(self.currentItem().column()).text() in ['H2']:
                print(self.parent.P94folder + str(self.cell_value('name')) + '/H2.sss')
                self.parent.openFile(self.parent.P94folder + str(self.cell_value('name')) + '/H2.sss', zoom=self.parent.s[0].cont.n==0)
                
                if 0:
                    H2_energy = [0, 118.4950, 354.3903, 705.4968, 1168.5301, 1739.1118,
                                 2411.7682, 3179.9316, 4035.9347, 4971.0273, 5975.3566]
                    n = 6
                    H2tot = float(self.cell_value('H2'))

                    if 0:
                        Temp = 70
                        gf = [np.exp(-H2_energy[i]/0.695/Temp)*(2*i+1)*((i%2)*2+1) for i in range(n+1)]
                        H2j = [spectro.atomic.e('H2', H2tot + np.log10(gf[i]) - np.log10(sum(gf)), J=i) for i in range(n+1)]
                    else:
                        if H2tot < 19.7:
                            ref_col = [18.54, 18.22, 18.25, 16.62, 14.84, 13.94, 13.86, 13.00, 13.00]
                        else:
                            ref_col = [19.93, 19.83, 19.25, 16.47, 15.15, 13.95, 13.00, 13.00]
                        s = np.log10(sum(np.power(10, ref_col)))
                        H2j = [spectro.atomic.e('H2', H2tot - s + ref_col[i], J=i) for i in range(n+1)]
                    print([j.logN for j in H2j])
                    H2 = H2list.Malec(n)
                    for line in H2:
                        line.logN = H2j[line.J_l].logN
                        line.b = 5
                    self.parent.s.fit(H2, z=float(self.cell_value('z_dla')))

            if self.horizontalHeaderItem(self.currentItem().column()).text() in ['SII', 'SiII', 'ZnII', 'FeII']:
                self.parent.console.exec_command('show SII')
                print(self.parent.P94folder + str(self.cell_value('name')) + '/' + self.horizontalHeaderItem(self.currentItem().column()).text() + '.sss')
                self.parent.openFile(self.parent.P94folder + str(self.cell_value('name')) + '/' + self.horizontalHeaderItem(self.currentItem().column()).text() + '.sss')
                self.parent.normalize(True)
                self.parent.showFit()

        if 'DLA' == self.cat:
            if colInd == 0:
                sdss = self.parent.IGMspec['/BOSS_DR12']
                ind = np.where(np.logical_and(sdss['meta']['FIBERID'] == int(self.cell_value('fiber')), sdss['meta']['PLATE'] == int(self.cell_value('plate'))))[0][0]
                print(sdss['meta'][ind]['SPEC_FILE'].decode())
                self.parent.importSpectrum(sdss['meta'][ind]['SPEC_FILE'].decode(),
                                           spec=[sdss['spec'][ind]['wave'], sdss['spec'][ind]['flux'],
                                                 sdss['spec'][ind]['sig']])
                mask = (sdss['spec'][ind]['wave'] != 0)
                inter = interp1d(sdss['spec'][ind]['wave'][mask], sdss['spec'][ind]['co'][mask], bounds_error=False, fill_value='extrapolate')
                self.parent.s[-1].cont.set_data(x=self.parent.s[-1].spec.x(), y=inter(self.parent.s[-1].spec.x()))
                self.parent.s[-1].cont_mask = np.ones_like(self.parent.s[-1].spec.x(), dtype=bool)
                self.parent.setz_abs(self.cell_value('z_DLA'))
                self.parent.fit = fitPars(self.parent)
                self.parent.fit.addSys(z=float(self.cell_value('z_DLA')))
                self.parent.fit.sys[0].addSpecies('HI')
                self.parent.fit.sys[0].sp['HI'].b.set(4)
                self.parent.fit.sys[0].sp['HI'].b.vary = False
                self.parent.fit.sys[0].sp['HI'].N.set(float(self.cell_value('HI')))
                try:
                    self.parent.showFit()
                except:
                    pass
                #self.parent.s[-1].resolution = 2000
                self.parent.statusBar.setText('Spectrum is imported: ' + self.parent.s[-1].filename)

        if 'Lya' == self.cat:
            if colInd == 0:
                filename = self.cell_value('name').strip()
                #for r in self.parent.plot.regions:
                #    r.remove()
                #self.parent.regions = []
                self.parent.importSpectrum(self.folder + '/spectra/' + filename + '.dat')
                #self.parent.s[-1].spec.raw.clean(min=-1, max=2)
                self.parent.s[-1].set_data()
                self.parent.s[-1].resolution = float(self.cell_value('resolution'))
                self.parent.s.redraw()
                try:
                    with open(self.folder + '/cont/' + filename + '.spv') as f:
                        skip_header = 1 if '%' in f.readline() else 0
                    self.parent.openFile(self.folder + '/cont/' + filename + '.spv', skip_header=skip_header, remove_regions=True, remove_doublets=True)
                except:
                    pass
                self.parent.plot.vb.setYRange(-0.1, 1.2)

        if 'Lyalines' == self.cat:
            if colInd in [0, 1]:
                filename = self.cell_value('name').strip()
                if self.filename_saved != filename:
                    self.parent.normview = False
                    if 0:
                        self.parent.importSpectrum(self.folder + '/norm/' + filename)
                        self.parent.s[-1].spec.raw.clean(min=-1, max=2)
                        self.parent.s[-1].set_data()
                        self.parent.s[-1].add_spline([self.parent.s[-1].spec.raw.x[0], self.parent.s[-1].spec.raw.x[-1]], [1, 1])
                    else:
                        self.parent.importSpectrum(self.folder + '/spectra/' + filename)
                        #self.parent.s[-1].spec.raw.clean(min=-1, max=10)
                        self.parent.s[-1].set_data()
                        with open(self.folder + '/cont/' + filename.replace('.dat', '.spv')) as f:
                            skip_header = 1 if '%' in f.readline() else 0
                        self.parent.openFile(self.folder + '/cont/' + filename.replace('.dat', '.spv'), skip_header=skip_header, remove_regions=True, remove_doublets=True)
                    self.parent.normalize()
                    self.parent.s[-1].resolution = 48000 #float(self.cell_value('resolution'))
                    self.filename_saved = filename
                #self.parent.s.redraw()
                self.parent.s[-1].mask.set(x=np.zeros_like(self.parent.s[-1].spec.x(), dtype=bool))
                self.parent.s[-1].set_fit_mask()
                self.parent.s[-1].update_points()
                self.parent.s[-1].set_res()
                self.parent.fit.setValue('z_0', 1, 'min')
                self.parent.fit.setValue('z_0', 9, 'max')
                self.parent.fit.setValue('z_0', float(self.cell_value('z')))
                self.parent.fit.setValue('N_0_HI', float(self.cell_value('N')))
                self.parent.fit.setValue('b_0_HI', float(self.cell_value('b')))
                self.parent.showfullfit = True
                self.parent.showFit()
                l = 1215.6701 * (float(self.cell_value('z')) + 1)
                self.parent.plot.vb.setXRange(l * 0.999, l * 1.001)
                self.parent.plot.vb.setYRange(-0.1, 1.2)
                if 'me' in self.cell_value('comment'):
                    self.parent.setz_abs(self.cell_value('comment').split('_')[2])

        if self.cat == 'Vandels':
            if colInd in [0]:
                if self.previous_item is not None:
                    self.saveCont(name=self.item(self.previous_item[0], self.columnIndex('id')).text())
                filename = self.folder + '/' + self.cell_value('FILENAME')
                self.parent.importSpectrum(filename)
                self.parent.vb.enableAutoRange()
                self.parent.import2dSpectrum(filename.replace('.fits', '_2D.fits'), ind=0)
                self.parent.setz_abs(self.cell_value('zspec'))
                self.parent.s[self.parent.s.ind].spec2d.raw.getQuantile(quantile=0.95)
                self.parent.console.exec_command('2d levels -0.01 0.01')

                if os.path.isfile(self.folder + '/cont/' + self.cell_value('id').strip() + '.spv'):
                    self.parent.openFile(self.folder + '/cont/' + self.cell_value('id').strip() + '.spv', remove_regions=True, remove_doublets=True)

                self.parent.s[self.parent.s.ind].rebinning(32)

        if self.cat == 'Kodiaq':
            if colInd in [0]:
                filename = self.folder + '/' + self.cell_value('qso') + '/' + self.cell_value('pi_date') + '/' + self.cell_value('spec_prefix')
                hdulist = fits.open(filename+'_f.fits')
                y = hdulist[0].data
                x = np.power(10, hdulist[0].header['CRVAL1'] + np.arange(len(y)) * hdulist[0].header['CDELT1'])
                hdulist = fits.open(filename + '_e.fits')
                err = hdulist[0].data
                mask = np.logical_and(y > -5, y < 5)
                self.parent.importSpectrum(self.cell_value('spec_prefix'), spec=[x[mask], y[mask], err[mask]])
                self.parent.vb.enableAutoRange()

        if self.cat == 'UVES':
            if colInd in [0, 6]:
                filename = self.folder + self.cell_value('filename').strip()
                hdul = fits.open(filename)
                y = hdul[0].data
                x = hdul[0].header['CRVAL1'] + np.arange(hdul[0].header['NAXIS1']) * hdul[0].header['CDELT1']
                mask = y != np.nan
                self.parent.importSpectrum(self.cell_value('name'), spec=[x[mask], y[mask]])
                self.parent.vb.enableAutoRange()

        if 'Erosita' == self.cat:
            if colInd == 0:
                if all([self.columnIndex(name) is not None for name in ['PLATE_fl', 'FIBERID_fl', 'MJD_fl']]):
                    plate, MJD, fiber = int(self.cell_value('PLATE_fl')), int(self.cell_value('MJD_fl')), int(self.cell_value('FIBERID_fl'))
                elif self.columnIndex('SDSS_NAME_fl') is not None:
                    plate, MJD, fiber = self.parent.ErositaWidget.getSDSSind(name=self.cell_value('SDSS_NAME_fl'))
                else:
                    plate = None

                if plate is not None:
                    self.parent.loadSDSS(plate=plate, MJD=MJD, fiber=fiber, gal_ext=True)
                    self.parent.statusBar.setText('Spectrum is imported: ' + self.parent.s[-1].filename)

                    if self.columnIndex('z') is not None:
                        self.parent.setz_abs(self.cell_value('z'))
                        if self.parent.compositeQSO_status % 2:
                            self.parent.compositeQSO.z = float(self.cell_value('z'))
                            self.parent.compositeQSO.calc_scale()
                            self.parent.compositeQSO.redraw()
                        if self.parent.compositeGal_status % 2:
                            self.parent.compositeGal.z = float(self.cell_value('z'))
                            self.parent.compositeGal.calc_scale()
                            self.parent.compositeGal.redraw()

                    self.parent.ErositaWidget.index(name=self.cell_value('SDSS_NAME_fl')  , ext=False)
                # self.parent.s[-1].resolution = 2000

        if load_spectrum:

            self.parent.importSpectrum(filename)
            self.parent.vb.enableAutoRange()

            if 'XQ100' in self.cat:
                if XQ100comb:
                    print(filename[:filename.rfind('/')]+'/cont.sss')
                    try:
                        self.parent.openFile(filename[:filename.rfind('/')]+'/cont.sss')
                    except:
                        pass
                else:
                    for f, s in zip(filename, self.parent.s):
                        hdulist = fits.open(f)
                        prihdr = hdulist[1].data
                        try:
                            s.cont.add(prihdr[0][0][:]*10, prihdr[0][3][:]*1e17)
                            s.g_cont.setData(x=s.cont.x, y=s.cont.y)
                        except:
                            pass
                        
                    if self.horizontalHeaderItem(self.currentItem().column()).text() == 'DLA':
                        self.DLAtable = absTable(self)
                        XQ100data = load_QSO()
                        for qso in XQ100data:
                            if self.item(self.currentItem().row(), 0).text() == str(qso.id):
                                break
                        data = np.array([(a.z, a.logN) for a in qso.DLA], dtype=[('z', np.str_, 9), ('N_HI', np.str_, 5)])
                        print(data)
                        self.DLAtable.setdata(data)

            if 'P94' in self.cat:               
                self.parent.setz_abs(self.cell_value('z_dla'))

            if isinstance(filename, str):
                self.parent.statusBar.setText('Spectrum is imported: ' + filename)
            elif isinstance(filename, list):
                self.parent.statusBar.setText('Spectra are imported from list: ' + str(filename))

        self.previous_item = [self.currentItem().row(), self.currentItem().column()]

    def columnIndex(self, columnname):
        l = [self.horizontalHeaderItem(x).text() for x in range(self.columnCount())]
        if columnname in l:
            return l.index(columnname)
        else:
            return None

    def set_cell_value(self, row, columnname, text):
        print(self.columnIndex(columnname), text)
        self.item(row, self.columnIndex(columnname)).setText(text)

    def cell_value(self, columnname, row=None):

        if row is None:
            row = self.currentItem().row()

        if self.columnIndex(columnname) is not None:
            cell = self.item(row, self.columnIndex(columnname)).text()   # get cell at row, col
            return cell

    def getColumnData(self, column):
        ind = self.columnIndex(column)
        if ind is not None:
            return [self.item(row, ind).text() for row in range(self.rowCount())]

    def getRowIndex(self, column=None, value=None):
        d = self.getColumnData(column)
        if d is not None and value in d:
            return d.index(value)

    def selectRow(self, row):
        for col in range(self.columnCount()):
            self.item(row, col).setSelected(True)

    def show_H2_cand(self):
        
        self.mw = MatplotlibWidget(size=(200, 100), dpi=100)
        self.mw.move(QPoint(100,100))
        self.mw.show()
        figure = self.mw.getFigure()
        self.s[self.s.ind].calc_norm()
        if self.s[self.s.ind].norm.n > 0:
            self.SDSS.load_spectrum([self.s[self.s.ind].norm.x, self.s[self.s.ind].norm.y, self.s[self.s.ind].norm.err])
            self.SDSS.H2_cand.z = self.z_abs
            self.SDSS.plot_candidate(fig=figure, normalized=True)
        self.mw.draw()

class IGMspecTable(pg.TableWidget):
    def __init__(self, parent, cat=None, subparent=None, editable=False):
        super().__init__(editable=editable)
        self.setStyleSheet(open('config/styles.ini').read())
        self.parent = parent
        self.subparent = subparent
        self.setSortingEnabled(True)
        self.cat = cat
        self.resize(100, 800)
        self.move(500, 300)
        self.show()
        self.setWindowTitle(self.cat)
        self.cellDoubleClicked.connect(self.row_clicked)
        self.verticalHeader().setDefaultAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        self.data = self.parent.IGMspec[self.cat]

    def setdata(self, data=None):
        print(data)
        if data is None:
            dtype = [('SPEC_FILE', np.str_, 100), ('RA_GROUP', np.float_), ('DEC_GROUP', np.float_), ('IGM_ID', np.int_),
                     ('DATE-OBS', np.str_, 10), ('zem_GROUP', np.float_), ('R', np.int_)]
            data = np.empty([len(self.data['meta']['IGM_ID'])], dtype=dtype)
            data = add_field(data, [('ind', int)], np.arange(len(self.data['meta']['IGM_ID'])))
            for d in dtype:
                if isinstance(d[1], str):
                    data[d[0]] = np.array([x[:] for x in self.data['meta'][d[0]]])
                else:
                    data[d[0]] = self.data['meta'][d[0]]

        print(data)
        self.setData(data)
        self.resizeColumnsToContents()
        self.horizontalHeader().setResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setResizeMode(0, QHeaderView.ResizeToContents)
        self.horizontalHeader().setResizeMode(1, QHeaderView.ResizeToContents)
        w = 180 + self.verticalHeader().width() + self.autoScrollMargin() * 1.5
        w += sum([self.columnWidth(c) for c in range(self.columnCount())])
        self.resize(w, self.size().height())
        if self.subparent is not None:
            self.subparent.resize(w, self.size().height() + 50)

    def row_clicked(self):

        if self.currentItem().column() == 0:

            #ind = [x.decode() for x in self.data['meta']['SPEC_FILE']].index(self.cell_value('SPEC_FILE'))
            ind = int(self.cell_value('ind'))
            print(ind, self.data['meta']['SPEC_FILE'][ind])
            print('IGMspec/'+self.cat+'/'+self.cell_value('SPEC_FILE'))
            self.parent.importSpectrum('IGMspec/'+self.cat+'/'+self.cell_value('SPEC_FILE'), spec=[self.data['spec'][ind]['wave'], self.data['spec'][ind]['flux'], self.data['spec'][ind]['sig']])
            if self.cat == 'KODIAQ_DR1':
                self.parent.s[-1].spec.raw.clean(min=-1, max=2)
            try:
                self.parent.s[-1].resolution = int(self.cell_value('R'))
            except:
                pass
            self.parent.s[-1].set_data()
            self.parent.s.redraw()
            self.parent.statusBar.setText('Spectrum is imported: ' + self.cell_value('SPEC_FILE'))

    def cell_value(self, columnname):

        row = self.currentItem().row()

        # loop through headers and find column number for given column name
        headercount = self.columnCount()
        for x in range(0, headercount, 1):
            headertext = self.horizontalHeaderItem(x).text()
            if columnname == headertext:
                matchcol = x
                break

        cell = self.item(row, matchcol).text()  # get cell at row, col

        return cell

class absTable(pg.TableWidget):
    def __init__(self, parent, cat=None):
        super().__init__(editable=True)
        self.parent = parent      
        self.resize(300, 800)
        self.move(200, 100)
        self.show()
        self.setWindowTitle('choose DLA from list')
        if 0:
            self.cellDoubleClicked.connect(self.row_clicked)
        else:
            self.verticalHeader().sectionDoubleClicked.connect(self.row_clicked)
    
    def setdata(self, data):
        print(data)
        self.setData(data)
        w = 80 + self.verticalHeader().width() + self.autoScrollMargin()*1.5
        w += sum([self.columnWidth(c) for c in range(self.columnCount())])
        h = 5 + self.horizontalHeader().height() + self.autoScrollMargin()*1.5
        h += sum([self.rowHeight(c) for c in range(self.rowCount())])
        print(w, h, self.verticalHeader().height(), self.autoScrollMargin(), self.rowCount(), [self.rowHeight(c) for c in range(self.rowCount())])
        self.resize(w, h)
        
    def row_clicked(self):
        self.parent.parent.setz_abs(self.cell_value('z'))
        self.add_abs()
    
    def cell_value(self, columnname):
        row = self.currentItem().row()
        #loop through headers and find column number for given column name
        headercount = self.columnCount()
        for x in range(0,headercount,1):
            headertext = self.horizontalHeaderItem(x).text()
            if columnname == headertext:
                matchcol = x
                break
            
        cell = self.item(row,matchcol).text()   # get cell at row, col
        
        return cell
        
    def add_abs(self):
        HI = HIlist.HIset(25)
        for lines in HI:
            lines.logN = float(self.cell_value('N_HI'))
            lines.b = 15
        self.parent.parent.s.fit(HI, z=float(self.cell_value('z')))
          
