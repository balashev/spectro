from astropy import constants as const
from collections import OrderedDict
from functools import partial
from itertools import combinations
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (QFont)
from PyQt5.QtWidgets import (QWidget, QLabel, QGridLayout, QPushButton,
                             QApplication, QHBoxLayout, QVBoxLayout)

import pyqtgraph as pg

from ..atomic import *
from .fit import par
from .utils import Timer

class atomic_data(OrderedDict):
    def __init__(self):
        super().__init__()
        self.readMorton()
        self.readH2(j=[0,1,2,3,4,5,6])
        #self.readHD()
        #.compareH2()

    def readMorton(self):
        with open('data/Morton2003.dat', 'r') as f:
            ind = 0
            while True:
                l = f.readline()
                if l == '':
                    break
                if ind == 1:
                    name = l.split()[0].replace('_', '')
                    self[name] = e(name)
                    block = []
                    while l[:3] != '***':
                        l = f.readline()
                        block.append(l)
                    levels = set()
                    for l in block:
                        if l[34:35] == '.' and 'MltMean' not in l:
                            levels.add(float(l[31:38]))
                    levels = sorted(levels)
                    for i, lev in enumerate(levels):
                        if i>0:
                            name = name+'*'
                            self[name] = e(name)
                        for l in block:
                            add = 0
                            if l[23:24] == '.':
                                #print(l)
                                if name == 'HI' or name == 'DI':
                                    if 'MltMean' in l:
                                        add = 1
                                elif 'MltMean' not in l:
                                    add = 1
                                if add == 1 and l[79:88].strip() != '' and l[59:68].strip() != '' and float(l[31:38]) == lev:
                                    self[name].lines.append(line(name, float(l[19:29]), float(l[79:88]), float(l[59:68]), ref='Morton2003'))
                    ind = 0

                if l[:3] == '***':
                    ind = 1

    def readH2(self, nu=0, j=[0,1], energy=None):
        if 0:
            self.readH2Abgrall(nu=nu, j=j, energy=energy)
        else:
            if nu == 0:
                self.readH2Malec(j=j)

    def readH2Malec(self, j=[0,1]):
        """
        read Malec calalogue 2010 data for H2

        parameters:
            - n    : if list - specify rotational levels to read
                     if int - read J_l<=n
        """
        x = np.genfromtxt(r'data/H2/energy_X.dat', comments='#', unpack=True)

        if isinstance(j, int) or isinstance(j, float):
            j = [j]
        mask = np.zeros_like(x[1], dtype=bool)
        for j in j:
            mask = np.logical_or(mask, x[1]==j)
        mask = np.logical_and(mask, x[0] <= 0)

        with open(os.path.dirname(os.path.realpath(__file__)) + r'/data/H2/H2MalecCat.dat', newline='') as f:
            f.readline()
            data = f.readlines()

        x = x[:, mask]
        x = np.transpose(x)
        for xi in x:
            nu_l, j_l = int(xi[0]), int(xi[1])
            name = 'H2' + 'j' +  str(j_l)
            if xi[0] > 0:
                name += 'n' + str(nu_l)
            print(name)
            self[name] = e(name)
            self[name].energy = float(xi[2])
            self[name].stats = (2 * (j_l % 2) + 1) * (2 * j_l + 1)
            self[name].lines = AtomicList.Malec(n=[j_l])

    def readH2Abgrall(self, nu=0, j=[0,1], energy=None):
        """
        read data of H2 transitions from file.
        parameters:
            - nu       :   threshold of vibrational level
            - j        :   threshold of rotational level
            - energy   :   threshold of energy

            any H2 level specified as
            H2_j_nu, e.g. H2_3_1
            if nu is not given then nu=0
        """
        x = np.genfromtxt(r'data/H2/energy_X.dat', comments='#', unpack=True)
        B = np.genfromtxt(r'data/H2/energy_B.dat', comments='#', unpack=True)
        Cp = np.genfromtxt(r'data/H2/energy_C_plus.dat', comments='#', unpack=True)
        Cm = np.genfromtxt(r'data/H2/energy_C_minus.dat', comments='#', unpack=True)
        B_t = np.genfromtxt(r'data/H2/transprob_B.dat', comments='#', unpack=True)
        Cp_t = np.genfromtxt(r'data/H2/transprob_C_plus.dat', comments='#', unpack=True)
        Cm_t = np.genfromtxt(r'data/H2/transprob_C_minus.dat', comments='#', unpack=True)
        e2_me_c = const.e.gauss.value ** 2 / const.m_e.cgs.value / const.c.cgs.value
        if energy is None:
            if isinstance(j, int) or isinstance(j, float):
                mask = np.logical_and(x[0] <= nu, x[1] == j)
            if isinstance(j, list):
                mask = np.logical_and(x[0] <= nu, np.logical_and(x[1] >= j[0], x[1] <= j[1]))
        else:
            mask = x[2] < energy
        x = x[:,mask]
        x = np.transpose(x)
        fout = open(r'data/H2/lines.dat', 'w')
        for xi in x:
            nu_l, j_l = int(xi[0]), int(xi[1])
            name = 'H2' + 'j' +  str(j_l)
            if xi[0] > 0:
                name += 'n' + str(nu_l)
            self[name] = e(name)
            self[name].energy = float(xi[2])
            self[name].stats = (2 * (j_l % 2) + 1) * (2 * j_l + 1)
            print(name, self[name].energy, self[name].stats)
            for t, en, note in zip([B_t, Cm_t, Cp_t], [B, Cm, Cp], ['L', 'W', 'W']):
                m = np.logical_and(t[4] == nu_l, t[5] == j_l)
                for ti in t[:,m].transpose():
                    nu_u, j_u = int(ti[1]), int(ti[2])
                    m1 = np.logical_and(en[0] == nu_u, en[1] == j_u)
                    l = 1e8 / (en[2, m1][0] - self[name].energy)
                    g_u = (2 * (j_l % 2) + 1) * (2 * j_u + 1)
                    f = (l * 1e-8) ** 2 / 8 / np.pi ** 2 / e2_me_c * ti[6] * g_u / self[name].stats
                    m = np.logical_and(t[1] == nu_u, t[2] == j_u)
                    g = np.sum(t[6, m])
                    self[name].lines.append(line(name, l, f, g, ref='Abgrall', j_l=j_l, nu_l=nu_l, j_u=j_u, nu_u=nu_u))
                    self[name].lines[-1].band = note
                    if j_l >= 0:
                        fout.write("{:8s} {:2d} {:2d} {:2d} {:2d} {:10.5f} {:.2e} {:.2e} \n".format(str(self[name].lines[-1]).split()[1], nu_l, j_l, nu_u, j_u, l, f, g))
                        #print(str(self[name].lines[-1]), nu_l, j_l, nu_u, j_u, l, f, g)

        fout.close()

    def compareH2(self):
        if 0:
            mal = Atomiclist.Malec()
            for line in mal:
                for l1 in self['H2j'+str(line)[-1]].lines:
                    #print(line)
                    #print(l1)
                    if str(line) == str(l1): #and np.abs(l1.f/line.f-1) > 0.2:
                        print(line, l1.f/line.f, line.f, l1.f)
            input()
        else:
            out = open(r'C:/Users/Serj/Desktop/H2MalecCat_comparison.dat', 'w')
            with open(r'C:/science/python/spectro/data/H2MalecCat.dat', 'r') as f:
                for line in f.readlines()[1:]:
                    m = line.split()
                    for l1 in self['H2j' + m[4]].lines:
                        if l1.nu_u == int(m[2]) and m[3] in str(l1) and m[1] in str(l1):
                            #out.write(line[:54] + '{0:12.10f}   {1:6.1e} '.format(l1.f, l1.g) + line[77:])
                            out.write(line[:-1] + '{0:12.10f}   {1:6.1e}\n'.format(l1.f, l1.g))
            out.close()

    def readHD(self):
        HD = np.genfromtxt(r'data/molec/HD.dat', skip_header=2, usecols=[0,1,2,3,4,5,6], unpack=True, dtype=None)
        j_u = {'R': 1, 'Q': 0, 'P': -1}
        for j in range(3):
            name = 'HDj'+str(j)
            self[name] = e(name)
            for l in HD:
                if l[0] == j:
                    #name = 'HD ' + l[1].decode('UTF-8') + str(l[2]) + '-0' + l[3].decode('UTF-8') + '(' + str(int(l[0])) + ')'
                    self[name].lines.append(line(name, l[4], l[5], l[6], ref='', j_l=l[0], nu_l=0, j_u=l[0] + j_u[l[3].decode('UTF-8')], nu_u=l[2]))
                    self[name].lines[-1].band = l[1].decode('UTF-8')

    def readHF(self):
        HF = np.genfromtxt(r'data/molec/HF.dat', skip_header=1, unpack=True, dtype=None)
        self['HF'] = e('HF')
        for l in HF:
            self['HF'].lines.append(line(l[4].decode('UTF-8'), l[6], l[7], l[8], ref='', j_l=l[0], nu_l=l[1], j_u=l[2], nu_u=l[3]))

class absSystemIndicator():
    def __init__(self, parent):
        self.parent = parent
        self.lines = []
        self.update()

    def add(self, lines, color=(0, 0, 0)):
        for line in lines:
            if line not in self.linelist:
                l = LineLabel(self, line, self.parent.linelabels, color=color)
                self.parent.vb.addItem(l)
                self.lines.append(l)
        self.redraw()

    def remove(self, lines=None, el=None):
        if el is None and lines is None:
            lines = self.lines
        if el is not None:
            lines = [l for l in self.lines if str(l.line).startswith(el)]
        if not isinstance(lines, list):
            lines = [lines]
        for line in lines:
            for i in reversed(range(len(self.lines))):
                if line == self.lines[i].line:
                    self.parent.vb.removeItem(self.lines[i])
                    try:
                        self.parent.lines.remove(str(line))
                    except:
                        pass
                    self.lines.pop(i)
                    break
        self.redraw()

    def changeStyle(self):
        lines = self.lines[:]
        for line in lines:
            l, c = line.line, line.color
            self.parent.vb.removeItem(line)
            l = LineLabel(self, l, self.parent.linelabels, color=c)
            self.parent.vb.addItem(l)
            self.lines.append(l)
        self.redraw()

    def update(self):
        self.linelist = [l.line for l in self.lines]
        self.linenames = [str(l.line) for l in self.lines]
        self.activelist = [l for l in self.lines if l.active]

    def index(self, linename):

        if linename in self.linenames:
            return self.linenames.index(linename)
        else:
            return -1

    def redraw(self, z=None):
        self.update()
        if z is not None:
            self.parent.z_abs = z
            self.parent.panel.z_panel.setText(str(z))
        if hasattr(self.parent, 's') and len(self.parent.s) > 0:
            for line in self.lines:
                line.setActive()
                line.redraw(self.parent.z_abs)

class LineLabel(pg.TextItem):
    def __init__(self, parent, line, graphicType, **kwrds):
        self.parent = parent
        self.saved_color = kwrds['color']
        pg.TextItem.__init__(self, text='', anchor=(0.5, -1), fill=pg.mkBrush(0, 0, 0, 0), **kwrds)
        self.graphicType = graphicType
        if self.graphicType == 'short':
            self.arrow = pg.ArrowItem(angle=90, headWidth=0.5, headLen=0, tailLen=30, brush=pg.mkBrush(255, 0, 0, 255),
                                  pen=pg.mkPen(0, 0, 0, 0), anchor=(0.5, -0.5))
        elif self.graphicType == 'infinite':
            self.arrow = pg.InfiniteLine(angle=90, pen=pg.mkPen(color=kwrds['color'], width=.5, style=Qt.SolidLine), label='') #style=Qt.DashLine
        self.arrow.setParentItem(self)
        self.setFont(QFont("SansSerif", 10))
        self.line = line
        self.setActive()

    def setActive(self, bool=None):
        if bool is not None:
            self.active = bool
        else:
            self.active = True if str(self.line) in self.parent.parent.lines else False
        if self.parent.parent.show_osc:
            self.setText(str(self.line)+' {:.4f}'.format(self.line.f))
        else:
            self.setText(str(self.line))

        if self.active:
            self.border = pg.mkPen(40, 10, 0, 255, width=1)
            self.fill = pg.mkBrush(255, 69, 0, 255)
            self.setColor(pg.mkColor(255, 255, 255, 255))
            if self.graphicType == 'infinite':
                self.arrow.setPen(pg.mkPen(color=self.fill.color(), width=1, style=Qt.SolidLine))
            #self.setColor(pg.mkColor(255, 69, 0, 255))
        else:
            self.border = pg.mkPen(0, 0, 0, 0, width=0)
            self.fill = pg.mkBrush(0, 0, 0, 0)
            self.setColor(self.saved_color)
            if self.graphicType == 'infinite':
                self.arrow.setPen(pg.mkPen(color=self.saved_color, width=.5, style=Qt.SolidLine))
        #self.paint()

    def redraw(self, z):
        ypos = self.parent.parent.s[self.parent.parent.s.ind].spec.inter(self.line.l * (1 + z))
        if ypos == 0:
            for s in self.parent.parent.s:
                ypos = s.spec.inter(self.line.l * (1 + z))
                if ypos != 0:
                    break
        self.setPos(self.line.l * (1 + z), ypos)

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
            self.parent.parent.z_abs += (pos.x() - self.st_pos) / self.line.l
            self.parent.parent.panel.refresh()
            self.parent.parent.line_reper = self.line
            self.parent.parent.plot.updateVelocityAxis()
            self.parent.redraw()
            ev.accept()

    def mouseClickEvent(self, ev):

        if QApplication.keyboardModifiers() == Qt.ShiftModifier:
            self.parent.parent.line_reper = self.line
            self.parent.parent.plot.updateVelocityAxis()
            ev.accept()
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.parent.remove(self.line)
            del self
        if ev.double():
            self.setActive(not self.active)
            if self.active and str(self.line) not in self.parent.parent.lines:
                self.parent.parent.lines.append(str(self.line))
            if not self.active and str(self.line) in self.parent.parent.lines:
                self.parent.parent.lines.remove(str(self.line))
            ev.accept()

    def clicked(self, pts):
        print("clicked: %s" % pts)

    def __hash__(self):
        return hash(str(self.line.l) + str(self.line.f))

    def __eq__(self, other):
        if self.line == other:
            return True
        else:
            return False

class choiceLinesWidget(QWidget):
    def __init__(self, parent, d):
        super().__init__()
        self.parent = parent
        self.d = d
        self.resize(1700, 1200)
        self.move(400, 100)
        self.setStyleSheet(open('config/styles.ini').read())
        layout = QVBoxLayout()
        self.grid = QGridLayout()
        layout.addLayout(self.grid)

        self.grid.addWidget(QLabel('HI:'), 0, 0)

        self.showAll = QPushButton("Show all")
        self.showAll.setFixedSize(100, 30)
        self.showAll.clicked[bool].connect(partial(self.showLines, 'all'))
        self.hideAll = QPushButton("HIde all")
        self.hideAll.setFixedSize(100, 30)
        self.hideAll.clicked[bool].connect(partial(self.showLines, 'none'))
        self.okButton = QPushButton("Ok")
        self.okButton.setFixedSize(100, 30)
        self.okButton.clicked[bool].connect(self.close)
        hbox = QHBoxLayout()

        hbox.addWidget(self.showAll)
        hbox.addWidget(self.hideAll)
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)

        layout.addLayout(hbox)
        self.setLayout(layout)

    def addItems(self, parent):
        for s in self.d.items():
            self.par_item = self.addParent(parent, s[0], expanded=True)
            if len(s[1]) > 1:
                for c in s[1]:
                    self.addChild(self.par_item, 1, c.replace('=', ''), c, 2)

    def showLines(self, lines):

        pass
        # self.close()

        # self.close()

class Doublet():
    def __init__(self, parent, name=None, z=None, color=pg.mkColor(45, 217, 207)):
        self.parent = parent
        self.read()
        self.active = False
        self.pen = pg.mkPen(color=color, width=0.5, style=Qt.SolidLine)
        self.name = name
        self.z = z
        self.temp = None
        if self.name is not None and self.z is not None:
            self.draw(add=False)

    def read(self):
        self.doublet = {}
        with open('data/doublet.dat') as f:
            for line in f:
                self.doublet[line.split()[0]] = [float(d) for d in line.split()[1:]]

    def draw(self, add=True):
        self.l = self.doublet[self.name]
        self.line, self.label = [], []
        for l in self.l:
            self.line.append(pg.InfiniteLine(l * (1 + self.z), angle=90, pen=self.pen))
            self.parent.vb.addItem(self.line[-1])
            anchor = (0, 1) if 'DLA' not in self.name else (0, 0)
            self.label.append(doubletLabel(self, self.name, l, angle=90, anchor=anchor))
            self.parent.vb.addItem(self.label[-1])
        if add and self.name.strip() in ['CIV', 'SiIV', 'AlIII', 'FeII', 'MgII']:
            self.parent.doublets.append(Doublet(self.parent, name='DLA', z=self.z))

    def redraw(self):
        #self.determineY()
        if self.active:
            self.pen = pg.mkPen(225, 215, 0, width=2)
        else:
            self.pen = pg.mkPen(45, 217, 207, width=0.5)
        for l, line, label in zip(self.l, self.line, self.label):
            line.setPos(l * (1 + self.z))
            line.setPen(self.pen)
            label.redraw()

    def remove(self):
        if self.temp is not None:
            self.remove_temp()
        else:
            for line, label in zip(self.line, self.label):
                self.parent.vb.removeItem(line)
                self.parent.vb.removeItem(label)
            self.parent.doublets.remove(self)

        del self

    def draw_temp(self, x):
        self.line_temp = pg.InfiniteLine(x, angle=90, pen=pg.mkPen(color=(44, 160, 44), width=2, style=Qt.SolidLine))
        self.parent.vb.addItem(self.line_temp)
        self.temp = []
        for lines in self.doublet.values():
            for d in combinations(lines, 2):
                for i in [-1,1]:
                    x = self.line_temp.value() * (d[0] / d[1])**i
                    self.temp.append(pg.InfiniteLine(x, angle=90, pen=pg.mkPen(color=(160, 80, 44), width=1, style=Qt.SolidLine)))
                    self.parent.vb.addItem(self.temp[-1])

    def remove_temp(self):
        if self.temp is not None:
            self.parent.vb.removeItem(self.line_temp)
            for t in self.temp:
                self.parent.vb.removeItem(t)
        self.temp = None

    def determineY(self):
        s = self.parent.parent.s[self.parent.parent.s.ind]
        imin, imax = s.spec.index([self.l1*(1+self.z), self.l2*(1+self.z)])[:]
        imin, imax = max(0, int(imin - (imax-imin)/2)), min(int(imax + (imax-imin)/2), s.spec.n())
        self.y = np.median(s.spec.y()[imin:imax])*1.5

    def set_active(self, active=True):
        if active:
            for d in self.parent.doublets:
                d.set_active(False)
        self.active = active
        self.parent.parent.setz_abs(self.z)
        self.redraw()

    def find(self, x1, x2, toll=9e-2):
        """
        Function which found most appropriate doublet using two wavelengths.
        parameters:
            - x1        :  first wavelength
            - x2        :  second wavelength
            - toll      :  tollerance for relative position

        """
        x1, x2 = np.min([x1, x2]), np.max([x1, x2])
        diff = 1-x1/x2

        res, ind = [], []
        for k, v in self.doublet.items():
            for d in combinations(v, 2):
                if -toll < 1 - (diff / (1 - d[0] / d[1])) < toll:
                    res.append(1- (diff / (1- d[0]/d[1])))
                    ind.append((k, d[0]))
        self.remove_temp()
        if len(res) > 0:
            i = np.argmin(np.abs(res))
            self.name = ind[i][0] #.decode('UTF-8').replace('_', '')
            self.z = x1 / ind[i][1] - 1
            self.parent.parent.console.exec_command('show '+self.name)
            self.parent.parent.setz_abs(self.z)
            self.draw()
        else:
            self.parent.doublets.remove(self)
            del self

class doubletLabel(pg.TextItem):
    def __init__(self, parent, name, line, **kwrds):
        self.parent = parent
        self.name = name
        self.line = line
        pg.TextItem.__init__(self, text=self.name, fill=pg.mkBrush(0, 0, 0, 0), **kwrds)
        self.setFont(QFont("SansSerif", 8))
        self.determineY()
        self.redraw()

    def determineY(self):
        s = self.parent.parent.parent.s[self.parent.parent.parent.s.ind]
        imin, imax = s.spec.index([self.line * (1 + self.parent.z) * (1 - 0.001), self.line * (1 + self.parent.z) * (1 + 0.001)])[:]
        imin, imax = max(0, int(imin - (imax - imin) / 2)), min(int(imax + (imax - imin) / 2), s.spec.n())
        if imin < imax:
            self.y = np.median(s.spec.y()[imin:imax]) * 1.5
        else:
            self.y = s.spec.y()[imin-1] * 1.5

    def redraw(self):
        self.determineY()
        self.setText(self.name + ' ' + str(self.line)[:str(self.line).index('.')] + '   z=' + str(self.parent.z)[:6])
        self.setColor(self.parent.pen.color())
        self.setPos(self.line * (1 + self.parent.z), self.y)

    def mouseDragEvent(self, ev):

        if QApplication.keyboardModifiers() == Qt.ShiftModifier:
            if ev.button() != Qt.LeftButton:
                ev.ignore()
                return

            pos = self.getViewBox().mapSceneToView(ev.scenePos())
            if not ev.isStart():
                self.parent.z += (pos.x() - self.st_pos) / self.line
            self.st_pos = pos.x()
            self.parent.redraw()
            ev.accept()

    def mouseClickEvent(self, ev):

        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.parent.remove()
            ev.accept()

        if ev.double():
            self.parent.set_active(not self.parent.active)


    def clicked(self, pts):
        print("clicked: %s" % pts)

class pcRegion():
    def __init__(self, parent, ind, x1=None, x2=None):
        self.parent = parent
        self.setInd(ind)
        self.color = pg.mkColor(220, 20, 60)
        if x1 is not None and x2 is not None:
            if isinstance(x1, float) and isinstance(x2, float):
                self.x1, self.x2 = x1, x2
                self.value = 0.1
            else:
                self.x1, self.x2 = np.min([x1.x(), x2.x()]), np.max([x1.x(), x2.x()])
                self.value = (x1.y() + x2.y()) / 2
        else:
            self.x1, self.x2 = 3000, 9000
            self.value = 0.0
        self.draw()
        if not self.parent.parent.fit.cf_fit:
            self.parent.parent.fit.cf_fit = True
        try:
            self.parent.parent.fitModel.cf.setExpanded(self.parent.parent.fit.cf_fit)
            self.parent.parent.fitModel.addChild('cf', 'cf_' + str(ind))
        except:
            pass
        self.parent.parent.fit.cf_num += 1
        if x1 is not None and x2 is not None:
            self.parent.parent.fit.add('cf_' + str(self.parent.parent.fit.cf_num))
            self.updateFitModel()

    def draw(self):
        self.gline = pg.PlotCurveItem(x=[self.x1, self.x2], y=[self.value, self.value], pen=pg.mkPen(color=self.color), clickable=True)
        self.gline.sigClicked.connect(self.lineClicked)
        self.parent.vb.addItem(self.gline)
        self.label = cfLabel(self, color=self.color)
        self.parent.vb.addItem(self.label)

    def redraw(self):
        self.gline.setData(x=[self.x1, self.x2], y=[self.value, self.value])
        self.label.redraw()

    def setInd(self, ind):
        self.ind = ind
        self.name = 'cf_' + str(ind)
        self.labelname = 'LFR ' + str(ind)

    def updateFromFit(self):
        self.value = getattr(self.parent.parent.fit, self.name).val
        self.x1 = getattr(self.parent.parent.fit, self.name).min
        self.x2 = getattr(self.parent.parent.fit, self.name).max
        self.redraw()

    def updateFitModel(self):
        #print(self.name, self.value, self.x1, self.x2)
        self.parent.parent.fit.setValue(self.name, self.value)
        self.parent.parent.fit.setValue(self.name, self.x1, 'min')
        self.parent.parent.fit.setValue(self.name, self.x2, 'max')
        try:
            self.parent.parent.fitModel.refresh()
        except:
            pass

    def remove(self):
        self.parent.vb.removeItem(self.gline)
        self.parent.vb.removeItem(self.label)
        self.parent.pcRegions.remove(self)
        try:
            for i in range(self.ind, len(self.parent.pcRegions) + 1):
                self.parent.parent.fitModel.cf.removeChild(getattr(self.parent.parent.fitModel, 'cf_' + str(i)))
        except:
            pass

        if self.ind < len(self.parent.pcRegions):
            for i in range(self.ind, len(self.parent.pcRegions)):
                print(i)
                self.parent.pcRegions[i].setInd(i)
                cf = getattr(self.parent.parent.fit, 'cf_' + str(i + 1))
                setattr(self.parent.parent.fit, 'cf_' + str(i), par(self, 'cf_' + str(i), cf.val, cf.min, cf.max, cf.step, addinfo=cf.addinfo))
                self.parent.pcRegions[i].redraw()

        self.parent.parent.fit.remove('cf_' + str(len(self.parent.pcRegions)))
        self.parent.parent.fit.cf_num = len(self.parent.pcRegions)
        if self.parent.parent.fit.cf_num == 0:
            self.parent.parent.fit.cf_fit = False

        try:
            if self.ind < len(self.parent.pcRegions):
                for i in range(self.ind, len(self.parent.pcRegions)):
                    self.parent.parent.fitModel.addChild('cf', 'cf_' + str(i))

            self.parent.parent.fitModel.refresh()
        except:
            pass

        del self

    def lineClicked(self):
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.remove()

class cfLabel(pg.TextItem):
    def __init__(self, parent,  **kwrds):
        self.parent = parent
        pg.TextItem.__init__(self, text=self.parent.labelname, anchor=(0, 1), fill=pg.mkBrush(0, 0, 0, 0), **kwrds)
        self.setFont(QFont("SansSerif", 12))
        self.redraw()

    def redraw(self):
        self.setText(self.parent.labelname)
        self.setPos(self.parent.x1 + (self.parent.x2 - self.parent.x1)*0.1, self.parent.value)

    def mouseDragEvent(self, ev):

        if QApplication.keyboardModifiers() == Qt.ShiftModifier:
            if ev.button() != Qt.LeftButton:
                ev.ignore()
                return

            pos = self.parent.parent.parent.vb.mapSceneToView(ev.pos())
            if ev.isStart():
                # We are already one step into the drag.
                # Find the point(s) at the mouse cursor when the button was first
                # pressed:
                self.st_pos = pos
            self.parent.x1 += (pos.x() - self.st_pos.x())
            self.parent.x2 += (pos.x() - self.st_pos.x())
            self.parent.value += (pos.y() - self.st_pos.y())
            self.parent.updateFitModel()
            self.parent.redraw()
            ev.accept()

    def mouseClickEvent(self, ev):

        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.parent.remove()
            ev.accept()

    def clicked(self, pts):
        print("clicked: %s" % pts)