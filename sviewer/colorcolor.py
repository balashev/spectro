from astropy.io import fits
import numpy as np
import pyqtgraph as pg
from collections import OrderedDict
from functools import partial
from itertools import combinations
from PyQt5.QtCore import QLocale
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
                             QLabel, QCheckBox, QGridLayout, QFrame, QTextEdit)
from scipy.stats import gaussian_kde

from .sdss_fit import SDSS_to_asinh


class colorColorPlot(pg.PlotItem):
    def __init__(self, parent, axis):
        super(colorColorPlot, self).__init__()
        self.parent = parent
        self.name = axis[0]+axis[1]
        self.x_axis = axis[0]
        self.y_axis = axis[1]
        self.setLabel('bottom', self.x_axis.replace('_', '-'))
        self.setLabel('left', self.y_axis.replace('_', '-'))

    def plotData(self, x=None, y=None, color=(255, 255, 255), size=10, rescale=True):
        if x is not None and y is not None:
            brush = pg.mkBrush(*color, 255)
            if size < 3:
                item = pg.ScatterPlotItem(x, y, brush=brush, pen=pg.mkPen(None), size=size, pxMode=True)
            else:
                item = pg.ScatterPlotItem(x, y, brush=brush, size=size, pxMode=True)
            self.addItem(item)
            if rescale:
                self.enableAutoRange()
            else:
                self.disableAutoRange()
            return item

    def plotIsoCurves(self, data, color=(255, 255, 255), rescale=True, levels=[0.5, 0.1, 0.01]):
        print(self.name)
        if data is not None:
            item = []
            xmin, xmax, ymin, ymax = np.min(data[0]), np.max(data[0]), np.min(data[1]), np.max(data[1])
            print(xmin, xmax, ymin, ymax)
            pen = pg.mkPen(*color, 255)
            num = 100.0
            if 0:
                kde = gaussian_kde(data)


                X, Y = np.mgrid[xmin:xmax:complex(0,num), ymin:ymax:complex(0,num)]
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.log10(np.reshape(kde(positions).T, X.shape))
                print(Z)

                item = pg.ImageItem()
                item.setImage(Z, levels=(-10, 0))
                item.translate(-num/2 + (xmax + xmin) / 2, -num / 2 + (ymax + ymin) / 2)
                item.scale((xmax-xmin)/num, (ymax-ymin)/num)
                self.addItem(item)
            else:
                H, X, Y = np.histogram2d(data[0], data[1], bins=num, normed=True, range=[[xmin, xmax], [ymin, ymax]])
                for l in levels:
                    i = pg.IsocurveItem(H, level=l, pen=pen)
                    i.scale((xmax - xmin) / num, (ymax - ymin) / num)
                    i.translate(- num/2 + (xmax+xmin)/2 * num/(xmax-xmin), - num/2 + (ymax+ymin)/2 * num/(ymax-ymin))
                    item.append(i)
                    self.addItem(i)

            if rescale:
                self.enableAutoRange()
            else:
                self.disableAutoRange()
            return item

class colorColorWidget(QWidget):
    def __init__(self, parent):
        super(colorColorWidget, self).__init__()
        self.parent = parent
        self.setStyleSheet(open('config/styles.ini').read())
        self.setGeometry(100, 120, 2200, 1300)

        self.initData()
        self.initGUI()
        self.show()
        #self.addSDSSQSO()
        #self.addCustom()

    def initData(self):
        self.colors = ['g_u', 'r_u', 'g_r', 'r_i', 'i_g']
        self.combs = combinations(self.colors, 2)

        self.loadSDSSPSF('spec')
        self.sdss = self.parent.IGMspec['BOSS_DR12']
        self.opts = {'traceNum': int,
                     'z_em_min': float, 'z_em_max': float,
                     'z_abs_min': float, 'z_abs_max': float,
                     'HI_min': float, 'HI_max': float,
                     'H2_min': float, 'H2_max': float,
                     'HI_min': float, 'HI_max': float,
                     'Av_min': float, 'Av_max': float,
                     'Av_bump_min': float, 'Av_bump_max': float
                     }
        for opt, func in self.opts.items():
            #print(opt, self.parent.options(opt), func(self.parent.options(opt)))
            setattr(self, opt, func(self.parent.options(opt)))

    def initGUI(self):
        layout = QHBoxLayout()

        v = QVBoxLayout()

        line = QFrame()
        line.setFixedSize(300, 1)
        line.setStyleSheet('background-color: rgb(149,149,149)')

        widget = QWidget()
        grid = QGridLayout()

        addTrace = QPushButton('add trace')
        addTrace.setFixedSize(100, 30)
        addTrace.clicked[bool].connect(self.addTrace)
        grid.addWidget(addTrace, 9, 0)

        self.color = pg.ColorButton()
        self.color.setFixedSize(60, 30)
        self.color.setColor(color=(225, 200, 50))
        grid.addWidget(self.color, 8, 0)

        names = ['Num:', '', '', '',
                 'Show', '', '', '',
                 'z_em:', '', '...', '',
                 'z_abs', '', '...', '',
                 'HI', '', '...', '',
                 'H2', '', '...', '',
                 'Av', '', '...', '',
                 'Av_bump', '', '...', '',
                 ]

        positions = [(i, j) for i in range(8) for j in range(4)]

        for position, name in zip(positions, names):
            if name == '' or (position[1] == 0 and position[0] not in [0,2]):
                continue
            label = QLabel(name)
            width = 30 if name == '...' else 45
            label.setFixedSize(width, 30)
            grid.addWidget(label, *position)

        self.opt_but = OrderedDict([('traceNum', [0, 1]),
                                    ('z_em_min', [2, 1]), ('z_em_max', [2, 3]),
                                    ('z_abs_min', [3, 1]), ('z_abs_max', [3, 3]),
                                    ('HI_min', [4, 1]), ('HI_max', [4, 3]),
                                    ('H2_min', [5, 1]), ('H2_max', [5, 3]),
                                    ('Av_min', [6, 1]), ('Av_max', [6, 3]),
                                    ('Av_bump_min', [7, 1]), ('Av_bump_max', [7, 3]),
                                   ])
        validator = QDoubleValidator()
        validator.setLocale(QLocale('C'))

        for opt, pos in self.opt_but.items():
            b = QLineEdit(str(getattr(self, opt)))
            b.setFixedSize(40, 30)
            b.setValidator(validator)
            b.textChanged[str].connect(partial(self.onChanged, attr=opt))
            grid.addWidget(b, pos[0], pos[1])

        for position, name in zip(positions, names):
            if position[1] == 0 and position[0] not in [0,2]:
                s = '' if position[0] == 1 else ':'
                setattr(self, name, QCheckBox(name + s))
                getattr(self, name).setFixedSize(80, 30)
                grid.addWidget(getattr(self, name), *position)

        self.Show.setChecked(False)
        self.z_abs.setChecked(True)
        self.z_abs.stateChanged.connect(self.checkEvent)
        self.HI.setChecked(True)
        self.H2.setChecked(True)
        self.Av.setChecked(False)
        self.Av_bump.setChecked(False)

        widget.setFixedSize(250, 400)
        widget.setLayout(grid)
        v.addWidget(widget)

        v.addWidget(line)

        self.addSDSSStars = QCheckBox('add SDSS stars')
        self.addSDSSStars.setFixedSize(200, 30)
        self.addSDSSStars.stateChanged.connect(self.addStellarLocus)
        self.addSDSSStars.setChecked(False)
        v.addWidget(self.addSDSSStars)

        self.addSDSSQSO = QCheckBox('add SDSS QSO:')
        self.addSDSSQSO.setFixedSize(160, 30)
        self.addSDSSQSO.stateChanged.connect(self.addQSOSDSS)
        self.addSDSSQSO.setChecked(False)
        v.addWidget(self.addSDSSQSO)
        h = QHBoxLayout()
        self.z_SDSS_min = QLineEdit('2.6')
        self.z_SDSS_min.setFixedSize(50, 30)
        h.addWidget(self.z_SDSS_min)
        label = QLabel('...')
        label.setFixedSize(15, 30)
        h.addWidget(label)
        self.z_SDSS_max = QLineEdit('2.8')
        self.z_SDSS_max.setFixedSize(50, 30)
        h.addWidget(self.z_SDSS_max)
        h.addStretch(1)
        v.addLayout(h)

        v.addWidget(line)

        v.addWidget(QLabel('Select by criterion:'))
        self.criteria = QTextEdit('')
        self.criteria.setFixedSize(300, 200)
        self.criteria.setText('r-i 0.6 0.8 \ni-g -2.0 -0.8\ng-r 0.2 0.6')
        v.addWidget(self.criteria)
        self.select = QPushButton('Apply')
        self.select.setFixedSize(80, 30)
        self.select.clicked[bool].connect(self.applyCriteria)
        v.addWidget(self.select)

        v.addWidget(line)

        v.addWidget(QLabel('Select by name:'))
        self.names = QTextEdit('')
        self.names.setFixedSize(300, 200)
        self.names.setText('7093-0632 \n6439-0160')
        v.addWidget(self.names)
        self.shownames = QPushButton('Show')
        self.shownames.setFixedSize(80, 30)
        self.shownames.clicked[bool].connect(self.applyNames)
        v.addWidget(self.shownames)
        v.addStretch(1)
        layout.addLayout(v)

        self.plot = pg.GraphicsWindow(size=(1900, 1800))

        self.p = []
        for comb in self.combs:
            p = colorColorPlot(self, comb)
            setattr(self, p.name, p)
            self.p.append(getattr(self, p.name))
            self.plot.addItem(self.p[-1], row=(len(self.p)-1) // 4, col=(len(self.p)-1) % 4 )

        self.setLinks()
        layout.addWidget(self.plot)
        self.setLayout(layout)

    def onChanged(self, text, attr=None):
        if attr is not None:
            setattr(self, attr, self.opts[attr](text))
            self.parent.options(attr, value=text)

    def checkEvent(self):
        self.HI.setChecked(self.z_abs.isChecked())
        self.H2.setChecked(self.z_abs.isChecked())
        self.Av.setChecked(self.z_abs.isChecked())
        self.Av_bump.setChecked(self.z_abs.isChecked())
        self.HI.setEnabled(self.z_abs.isChecked())
        self.H2.setEnabled(self.z_abs.isChecked())
        self.Av.setEnabled(self.z_abs.isChecked())
        self.Av_bump.setEnabled(self.z_abs.isChecked())

    def setLinks(self):
        for p in self.p:
            for p1 in self.p:
                if p.name != p1.name:
                    if p1.x_axis == p.x_axis:
                        p.setXLink(p1)
                    if p1.y_axis == p.y_axis:
                        p.setYLink(p1)

    def addData(self, data, typ='points', color=None, size=10, rescale=True, levels=None):
        """
        add data structurized numpy array:
        like, data['u'] = [22.4, 21.5, ...], data['g'] = [20.3,  20.2, ...]
        typ can be 'points' or 'isocurves'
        """

        #for k, v in data.items():
        #    data[k] = np.array(v)

        d = {}

        for p in self.p:
            c = p.x_axis.split('_')
            x = data[c[0]] - data[c[1]]
            c = p.y_axis.split('_')
            y = data[c[0]] - data[c[1]]
            if typ == 'points':
                d[p.name] = p.plotData(x=x, y=y, color=color, size=size, rescale=rescale)
            elif typ == 'isocurves':
                d[p.name] = p.plotIsoCurves(data=np.vstack([x,y]), color=color, rescale=rescale, levels=levels)
                #d[p.name] = p.plotData(x=x, y=y, color=color, size=size, rescale=rescale)

        return d

    def addTrace(self):
        self.parent.console.exec_command('load HI')
        if not self.HI.isChecked():
            del self.parent.fit.sys[0].sp['HI']
        if not self.H2.isChecked():
            del self.parent.fit.sys[0].sp['H2j0']
            del self.parent.fit.sys[0].sp['H2j1']

        if not self.parent.SDSS_filters_status:
            self.parent.show_SDSS_filters()

        zem_grid = np.linspace(self.z_em_min, self.z_em_max, self.traceNum)
        zabs_grid = np.linspace(self.z_abs_min, self.z_abs_max, self.traceNum)
        HI_grid = np.linspace(self.HI_min, self.HI_max, self.traceNum)
        H2_grid = np.linspace(self.H2_min, self.H2_max, self.traceNum)
        Av_grid = np.linspace(self.Av_min, self.Av_max, self.traceNum)
        Av_bump_grid = np.linspace(self.Av_bump_min, self.Av_bump_max, self.traceNum)
        data = []
        for zem, zabs, HI, H2, Av, Av_bump in zip(zem_grid, zabs_grid, HI_grid, H2_grid, Av_grid, Av_bump_grid):
            try:
                print(zabs, zem, HI, H2, Av)
                self.parent.fit.sys[0].z.val = zabs
                if self.HI.isChecked():
                    self.parent.fit.sys[0].sp['HI'].N.val = HI
                if self.H2.isChecked():
                    self.parent.fit.sys[0].sp['H2j0'].N.val = H2 - 0.3
                    self.parent.fit.sys[0].sp['H2j1'].N.val = H2 - 0.3
            except:
                pass
            if not self.Av.isChecked():
                Av = 0
            if not self.Av_bump.isChecked():
                Av_bump = 0
            d = self.parent.generate(template='HST', z=zem,
                                     fit=(self.HI.isChecked() and self.H2.isChecked()),
                                     xmin=3000, xmax=10000, resolution=2000, lyaforest=1.0,
                                     Av=Av, Av_bump=Av_bump, z_Av=zabs)
            data.append([d[k] for k in ['u', 'g', 'r', 'i', 'z']])

            if not self.Show.isChecked():
                self.parent.s.remove()

        data = np.core.records.array(list(tuple(np.array(data).transpose())), dtype=([('u', 'f4'), ('g', 'f4'), ('r', 'f4'), ('i', 'f4'), ('z', 'f4')]))
        print('trace', data)
        self.addData(data, color=self.color.color('byte')[:3], rescale=False)

    def addStellarLocus(self):
        if self.addSDSSStars.isChecked():
            f = fits.open('C:/Users/Serj/Desktop/specObj-dr8.fits')
            mask = (f[1].data['CLASS'] == 'STAR')
            sdss = SDSS_to_asinh(f[1].data['SPECTROFLUX'][mask][:30000])
            data = np.core.records.array(list(tuple(sdss.transpose())), dtype=([('u', 'f4'), ('g', 'f4'), ('r', 'f4'), ('i', 'f4'), ('z', 'f4')]))
            self.stars = self.addData(data, typ='isocurves', color=(100, 100, 100), size=1, levels=[0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
            print(self.stars)
        else:
            try:
                for p in self.p:
                    if isinstance(self.stars[p.name], list):
                        for i in self.stars[p.name]:
                            p.removeItem(i)
                    else:
                        p.removeItem(self.stars[p.name])
            except:
                pass

    def loadSDSSPSF(self, source='spec'):
        if source == 'photo':
            self.PSF = np.core.records.array(list(tuple(SDSS_to_asinh(sdss['meta']['PSFFLUX']).transpose())),
                                             dtype=([('u', 'f4'), ('g', 'f4'), ('r', 'f4'), ('i', 'f4'), ('z', 'f4')]))

        if source == 'spec':
            self.PSF = np.genfromtxt('temp/sdss_photo.dat', usecols=(1, 2, 3, 4, 5),
                                     dtype=([('u', 'f4'), ('g', 'f4'), ('r', 'f4'), ('i', 'f4'), ('z', 'f4')]))

    def addQSOSDSS(self, source='spec'):
        if self.addSDSSQSO.isChecked():
            mask = np.ones(len(self.sdss['meta']['THING_ID']), dtype=bool)
            if source == 'photo':
                mask[90644] = 0
                mask[195799] = 0
                mask[197031] = 0

            mask = np.logical_and(mask, np.logical_and(self.sdss['meta']['Z_VI'] > float(self.z_SDSS_min.text()),
                                  self.sdss['meta']['Z_VI'] < float(self.z_SDSS_max.text())))
            self.loadSDSSPSF(source)
            self.QSO = self.addData((self.PSF[mask]), typ='isocurves', color=(100, 100, 200), size=1, levels=[0.5, 0.1, 0.05, 0.01])
        else:
            try:
                for p in self.p:
                    if isinstance(self.QSO[p.name], list):
                        for i in self.QSO[p.name]:
                            p.removeItem(i)
                    else:
                        p.removeItem(self.QSO[p.name])
            except:
                pass


    def applyNames(self):
        spec = self.names.toPlainText().split('\n')
        mask = np.zeros(len(self.sdss['meta']), dtype=bool)
        for s in spec:
            print(int(s.split('-')[0]), int(s.split('-')[1]))
            mask = np.logical_or(mask, np.logical_and(int(s.split('-')[0]) == self.sdss['meta']['PLATE'],
                                                      int(s.split('-')[1]) == self.sdss['meta']['FIBERID']))

        if np.sum(mask) > 0:
            self.showSelected(mask, color=(100, 200, 100), size=9)

    def applyCriteria(self):
        criteria = self.criteria.toPlainText().split('\n')
        mask = np.ones(len(self.sdss['meta']), dtype=bool)

        for c in criteria:
            print(c)
            if 'z' in c.split()[0]:
                mask = np.logical_and(mask, np.logical_and(self.sdss['meta']['Z_VI'] > float(c.split()[1]),
                                                           self.sdss['meta']['Z_VI'] < float(c.split()[2])))

            elif 'BAL' in c.split()[0]:
                pass

            elif '-' in c.split()[0]:
                c1, c2 = c.split()[0].split('-')
                d = self.PSF[c1] - self.PSF[c2]
                mask = np.logical_and(mask, np.logical_and(d > float(c.split()[1]), d < float(c.split()[2])))

            elif c.split()[0] in ['u', 'g', 'r', 'z', 'i']:
                mask = np.logical_and(mask, np.logical_and(self.PSF[c.split()[0]] > float(c.split()[1]),
                                                           self.PSF[c.split()[0]] < float(c.split()[2])))
        print(self.PSF)
        if np.sum(mask) > 0:
            self.showSelected(mask, color=(200, 100, 100), size=5)

    def showSelected(self, mask, color=(200, 100, 100), size=3):

        ind = np.where(mask)[0]
        dtype = [('SPEC_FILE', np.str_, 100), ('THING_ID', np.int_), ('Z_VI', np.float_),
                 ('u', np.float_), ('g', np.float_), ('r', np.float_), ('i', np.float_), ('z', np.float_)]
        data = np.empty([len(ind)], dtype=dtype)
        for d in dtype:
            if isinstance(d[1], str):
                data[d[0]] = np.array([x[:] for x in self.sdss['meta'][mask][d[0]]])
            elif d[0][0].isupper():
                data[d[0]] = self.sdss['meta'][mask][d[0]]
            else:
                data[d[0]] = self.PSF[mask][d[0]]
        if len(data) > 0:
            self.parent.showIGMspec('BOSS_DR12', data=data)

        try:
            for p in self.p:
                p.removeItem(self.selected[p.name])
        except:
            pass

        self.selected = self.addData(self.PSF[mask], color=color, size=size, rescale=False)



