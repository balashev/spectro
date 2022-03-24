from astropy.io import ascii
from collections import OrderedDict
from functools import partial
from io import StringIO
import numpy as np
from PyQt5.QtCore import Qt, QSize, QLocale
from PyQt5.QtGui import QIcon, QDoubleValidator
from PyQt5.QtWidgets import (QApplication, QAction, QCheckBox, QComboBox, QFrame,
                             QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton,
                             QTabBar, QTabWidget, QTextEdit, QTreeWidget,
                             QTreeWidgetItem, QScrollArea, QVBoxLayout,
                             QWidget, QToolButton
                             )
import pyqtgraph as pg
from .utils import Timer
from ..a_unc import a
from ..pyratio import pyratio

class FLineEdit(QLineEdit):
    def __init__(self, parent, text='', var=None, name=''):
        super().__init__(str(text))
        self.parent = parent
        self.var = var
        self.name = name
        if self.name != '' and 'val' in self.name:
            self.returnPressed.connect(self.calcFit)
        self.setLength()
        if 1:
            validator = QDoubleValidator()
            locale = QLocale('C')
            #locale.RejectGroupSeparator = 1
            validator.setLocale(locale)
            validator.ScientificNotation
            if self.name[:1] == 'z':
                validator.setDecimals(8)
            if self.name[:1] in ['b', 'N', 'd', 'm', 'c', 'r', 'l', 's']:
                validator.setDecimals(4)
            self.setValidator(validator)

    def setLength(self, l=None):
        if l is None:
            if self.name[:1] == 'z':
                self.setMaxLength(10)
            if self.name[:1] in ['b', 'N', 'd', 'm', 'c', 'r', 'l', 's']:
                self.setMaxLength(7)
        else:
            self.setMaxLength(l)

    def calcFit(self):
        if self.var in ['N', 'b', 'z', 'kin', 'turb', 'v', 'c', 'Ntot', 'logn', 'logT', 'rad', 'CMB']:
            self.parent.parent.s.reCalcFit() #self.parent.tab.currentIndex())
        elif self.var in ['mu', 'dtoh'] or any([x in self.var for x in ['me', 'dispz', 'disps', 'res', 'st']]):
            self.parent.parent.s.prepareFit()
            self.parent.parent.s.calcFit()

class fitTabBar(QTabBar):
    def __init__(self, parent=None):
        QTabBar.__init__(self, parent)
        self.parent = parent
        self.setStyleSheet(open('config/styles.ini').read())

    def mouseReleaseEvent(self, ev):
        super(fitTabBar, self).mouseReleaseEvent(ev)
        cycles = self.swap([int(self.tabText(i)[4:]) for i in range(self.count())])
        for inds in cycles:
            for i in range(len(inds)-1):
                self.parent.parent.fit.swapSys(inds[i], inds[i+1])
                #self.moveTab(min(inds[i], inds[i + 1]), max(inds[i], inds[i + 1]))
        for i in range(self.count()):
            self.setTabText(i, 'sys ' + str(i))
            self.parent.tab.widget(i).ind = i
        self.parent.refresh()


    def swap(self, inds):
        unvisited = set(inds)
        cycle = []
        while unvisited:
            j = i = unvisited.pop()
            c = [i]
            while True:
                j = inds[j]
                if j == i:
                    break
                c.append(j)
                unvisited.remove(j)
            if len(c) > 1:
                cycle.append(c)
        return cycle

class chooseSystemPC(QToolButton):
    def __init__(self, parent, name):
        super(chooseSystemPC, self).__init__()
        self.parent = parent
        self.name = name
        self.setFixedSize(90, 30)
        self.toolmenu = QMenu(self)
        self.sys = []
        self.update()
        self.setText(self.currentText())
        self.setMenu(self.toolmenu)
        self.setPopupMode(QToolButton.InstantPopup)

    def update(self):
        try:
            self.toolmenu.removeAction(self.setall)
        except:
            pass
        try:
            for i in reversed(range(len(self.sys))):
                self.toolmenu.removeAction(self.sys[i])
                self.sys.remove(self.sys[i])
        except:
            pass
        if self.parent.parent.fit.cf_num == 1:
            self.setall = QAction("all", self.toolmenu)
            self.setall.setCheckable(True)
            self.setall.triggered.connect(partial(self.set, sys=None))
            self.toolmenu.addAction(self.setall)

        for i in range(len(self.parent.parent.fit.sys)):
            self.sys.append(QAction("sys " + str(i), self.toolmenu))
            self.sys[i].triggered.connect(partial(self.set, sys=i))
            self.toolmenu.addAction(self.sys[i])
            self.sys[i].setCheckable(True)
            if str(i) in self.parent.parent.fit.getValue(self.name, 'addinfo').split('_')[0].split('sys'):
                self.sys[i].setChecked(True)
            #print(self.parent.parent.fit.getValue(self.name, 'addinfo'), self.parent.parent.fit.getValue(self.name, 'addinfo').startswith('all'))
            if self.parent.parent.fit.getValue(self.name, 'addinfo').startswith('all') and hasattr(self, 'setall'):
                self.setall.setChecked(True)
            self.setText(self.currentText())

    def currentText(self):
        if hasattr(self, 'setall') and self.setall.isChecked():
            return 'all'
        else:
            return ''.join([f'sys{i}' for i, s in enumerate(self.sys) if s.isChecked()])

    def set(self, sys=None):
        if sys == None:
            if self.setall.isChecked():
                for s in self.sys:
                    s.setChecked(False)
            else:
                self.setall.setChecked(True)
        else:
            #for i in range(self.parent.parent.fit.cf_num):
            #    if self.name != 'cf_'+str(i):
            #        self.parent.parent.fit.setValue('cf_'+str(i), self.parent.parent.fit.getValue('cf_'+str(i), 'addinfo').replace('sys'+str(sys), ''), 'addinfo'),
            #        getattr(self.parent, 'cf_' + str(i) + '_applied').update()
            if hasattr(self, 'setall'):
                self.setall.setChecked(False)

        self.setText(self.currentText())
        exp = self.parent.parent.fit.getValue(self.name, 'addinfo').split('_')
        exp = '_'+exp[1] if len(exp) > 1 else '_all'
        self.parent.parent.fit.setValue(self.name, self.currentText() + exp, 'addinfo')

class fitModelWidget(QWidget):
    def __init__(self, parent):
        self.refr = False
        super(fitModelWidget, self).__init__()
        self.parent = parent
        self.setStyleSheet(open('config/styles.ini').read())

        self.tab = QTabWidget(movable=True)
        self.tabBar = fitTabBar(self)
        self.tab.setTabBar(self.tabBar)
        self.tabBar.setMouseTracking(False)
        self.tabBar.setMovable(True)
        self.tab.setGeometry(0, 0, 800, 900)
        self.tab.setMinimumSize(900, 300)
        self.tabIndex = 1
        #t = Timer('fitmodel')
        self.tabNum = len(self.parent.fit.sys)
        for i in range(self.tabNum):
            sys = fitModelSysWidget(self, i)
            #t.time(i)
            self.tab.addTab(sys, "sys {:}".format(i))
            #t.time(i)

        self.tab.currentChanged.connect(self.onTabChanged)
        self.tab.setCurrentIndex(self.parent.comp)
        self.parent.componentBar.setText("{:d} component".format(self.parent.comp))

        self.addFixedTreeWidget()

        speciesbox = QHBoxLayout()
        lbl = QLabel('type: ')
        speciesbox.addWidget(lbl)
        self.inputSpecies = QLineEdit()
        self.inputSpecies.setFixedSize(60, 30)
        self.inputSpecies.returnPressed.connect(self.addSpec)
        speciesbox.addWidget(self.inputSpecies)
        lbl = QLabel(' or ')
        speciesbox.addWidget(lbl)
        self.chooseSpecies = QComboBox()
        self.chooseSpecies.setFixedSize(80, 30)
        self.chooseSpecies.addItems(['choose...'] + list(self.parent.atomic.keys()))
        self.chooseSpecies.activated[str].connect(self.selectSpecies)
        speciesbox.addWidget(self.chooseSpecies)

        self.addall = QCheckBox('add to all')
        self.addall.setChecked(False)
        speciesbox.addWidget(self.addall)

        self.tied = QComboBox()
        self.tied.setFixedSize(80, 30)
        sp = set([sp for sys in self.parent.fit.sys for sp in sys.sp.keys()])
        self.tied.addItems(['tied...'] + list(sp))
        speciesbox.addWidget(self.tied)

        self.btied = QCheckBox('b')
        self.btied.setChecked(False)
        speciesbox.addWidget(self.btied)

        self.Ntied = QCheckBox('log N')
        self.Ntied.setChecked(False)
        speciesbox.addWidget(self.Ntied)

        self.Nscale = QLineEdit("")
        self.Nscale.setFixedSize(60, 30)
        self.Nscale.returnPressed.connect(self.addSpec)
        speciesbox.addWidget(self.Nscale)

        speciesbox.addStretch(1)
        # self.setLayout(layout)

        rangebox = QHBoxLayout()
        self.setrange = QPushButton("Set range:")
        self.setrange.setFixedSize(80, 30)
        self.setrange.clicked[bool].connect(self.setRange)
        lbl = QLabel("   unc x")
        lbl.setFixedSize(40, 30)
        self.rangeValue = QLineEdit("1")
        self.rangeValue.setFixedSize(60, 30)
        self.rangeValue.returnPressed.connect(self.setRange)
        rangebox.addWidget(self.setrange)
        rangebox.addWidget(lbl)
        rangebox.addWidget(self.rangeValue)
        rangebox.addStretch(1)

        self.addSys = QPushButton("Add system")
        self.addSys.setFixedSize(120, 30)
        self.addSys.clicked[bool].connect(self.addSystem)
        self.delSys = QPushButton("Del system")
        self.delSys.setFixedSize(120, 30)
        self.delSys.clicked[bool].connect(self.delSystem)
        self.okButton = QPushButton("OK")
        self.okButton.setFixedSize(70, 30)
        self.okButton.clicked[bool].connect(self.close)
        hbox = QHBoxLayout()
        hbox.addWidget(self.addSys)
        hbox.addWidget(self.delSys)
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)

        layout_left = QVBoxLayout()
        layout_left.addWidget(self.tab)
        layout_left.addWidget(QLabel('add species: '))
        layout_left.addLayout(speciesbox)
        layout_left.addLayout(rangebox)
        layout_left.addSpacing(50)
        #layout.addStretch(1)
        layout_left.addLayout(hbox)
        layout_right = QVBoxLayout()
        layout_right.addWidget(QLabel('add additional tied: '))
        self.tieWindow = QTextEdit()
        self.tieWindow.setFixedSize(400, 120)
        self.tieWindow.textChanged.connect(partial(self.updateTieWindow, init=False))
        self.updateTieWindow(init=True)
        layout_right.addWidget(self.tieWindow)
        layout_right.addWidget(self.treeWidget)
        #layout_right.addStretch(1)

        l = QHBoxLayout()
        l.addLayout(layout_left)
        l.addLayout(layout_right)

        self.setLayout(l)

        self.setGeometry(100, 100, 1900, 900)
        self.setWindowTitle('Fit model')
        self.show()
        self.refresh()
        self.refr = True

    def addFixedTreeWidget(self):
        self.treeWidget = QTreeWidget()
        self.treeWidget.move(0, 0)
        self.treeWidget.setHeaderHidden(True)
        self.treeWidget.setColumnCount(12)
        for i, w in enumerate([170, 10, 30, 70, 80, 70, 80, 10, 80, 70, 100, 60]):
            self.treeWidget.setColumnWidth(i, w)

        attr = ['val', 'min', 'max', 'step']
        sign = ['value: ', 'range: ', '....', 'step: ']

        self.cont = self.addParent(self.treeWidget, 'Continuum', expanded=self.parent.fit.cont_fit)
        self.cont.name = 'cont'

        self.cont_m = QTreeWidgetItem(self.cont)
        self.cont_m.name = 'cont_m'
        self.cont_m.setTextAlignment(3, Qt.AlignRight)
        self.cont_m.setText(3, 'reg num: ')
        self.cont_num = FLineEdit(self, str(self.parent.fit.cont_num))
        self.cont_num.setFixedSize(30, 30)
        self.cont_num.returnPressed.connect(self.numContChanged)
        self.treeWidget.setItemWidget(self.cont_m, 4, self.cont_num)
        self.cont_reg = QPushButton('from regions')
        self.cont_reg.setFixedSize(100, 30)
        self.cont_reg.clicked.connect(self.fromRegions)
        self.treeWidget.setItemWidget(self.cont_m, 5, self.cont_reg)
        self.cont_hier = QTreeWidgetItem(self.cont)
        self.addChild('cont', 'hcont', text_val='hier factor')
        #self.treeWidget.setItemWidget(self.cont_m, 5, self.cont_reg)
        self.cont.setExpanded(self.parent.fit.cont_fit)
        for i in range(self.parent.fit.cont_num):
            self.addContParent(self.cont, 'region_' + str(i), expanded=self.parent.fit.cont_fit)

        for s, name in zip(['mu', 'dtoh'], ['mp/me', 'D/H']):
            setattr(self, s + '_p', self.addParent(self.treeWidget, name, expanded=hasattr(self.parent.fit, s)))
            getattr(self, s + '_p').name = s
            self.addChild(s + '_p', s)
        #self.treeWidget.itemChanged.connect(self.stateChanged)

        self.me = self.addParent(self.treeWidget, 'Metallicity', expanded=self.parent.fit.me_num > 0)
        self.me.name = 'me'

        self.me_m = QTreeWidgetItem(self.me)
        self.me_m.setTextAlignment(3, Qt.AlignRight)
        self.me_m.setText(3, 'num: ')
        self.me_num = FLineEdit(self, str(self.parent.fit.me_num))
        self.me_num.setFixedSize(30, 30)
        self.me_num.returnPressed.connect(self.numMeChanged)
        self.treeWidget.setItemWidget(self.me_m, 4, self.me_num)
        for i in range(self.parent.fit.me_num):
            self.addChild('me', 'me_' + str(i), text_val='Metal. ' + str(i))

        self.res = self.addParent(self.treeWidget, 'Resolution', expanded=self.parent.fit.res_num > 0)
        self.res.name = 'res'

        self.res_m = QTreeWidgetItem(self.res)
        self.res_m.setTextAlignment(3, Qt.AlignRight)
        self.res_m.setText(3, 'num: ')
        self.res_num = FLineEdit(self, str(self.parent.fit.res_num))
        self.res_num.setFixedSize(30, 30)
        self.res_num.returnPressed.connect(self.numResChanged)
        self.treeWidget.setItemWidget(self.res_m, 4, self.res_num)
        for i in range(self.parent.fit.res_num):
            self.addChild('res', 'res_' + str(i), text_val='res ' + str(i))
        #self.res.setExpanded(self.parent.fit.res_fit)

        self.cf = self.addParent(self.treeWidget, 'Covering factors', expanded=self.parent.fit.cf_fit)
        self.cf.name = 'cf'

        self.cf_m = QTreeWidgetItem(self.cf)
        self.cf_m.setTextAlignment(3, Qt.AlignRight)
        self.cf_m.setText(3, 'num: ')
        self.cf_num = FLineEdit(self, str(self.parent.fit.cf_num))
        self.cf_num.setFixedSize(30, 30)
        self.cf_num.returnPressed.connect(self.numCfChanged)
        self.treeWidget.setItemWidget(self.cf_m, 4, self.cf_num)
        for i in range(self.parent.fit.cf_num):
            self.addChild('cf', 'cf_' + str(i), text_val='cf ' + str(i))
        self.cf.setExpanded(self.parent.fit.cf_fit)

        self.disp = self.addParent(self.treeWidget, 'Dispersion', expanded=self.parent.fit.disp_num > 0)
        self.disp.name = 'disp'

        self.disp_m = QTreeWidgetItem(self.disp)
        self.disp_m.setTextAlignment(3, Qt.AlignRight)
        self.disp_m.setText(3, 'num: ')
        self.disp_num = FLineEdit(self, str(self.parent.fit.disp_num))
        self.disp_num.setFixedSize(30, 30)
        self.disp_num.returnPressed.connect(self.numDispChanged)
        self.treeWidget.setItemWidget(self.disp_m, 4, self.disp_num)
        for i in range(self.parent.fit.disp_num):
            self.addChild('disp', 'dispz_' + str(i), text_val='zero p. ' + str(i))
            self.addChild('disp', 'disps_' + str(i), text_val='slope ' + str(i))
        #self.disp.setExpanded(self.parent.fit.disp_fit)

        self.stack = self.addParent(self.treeWidget, 'Stack', expanded=self.parent.fit.stack_num > 0)
        self.stack.name = 'stack'
        self.stack_m = QTreeWidgetItem(self.stack)
        self.stack_m.setTextAlignment(3, Qt.AlignRight)
        self.stack_m.setText(3, 'num: ')
        self.stack_num = FLineEdit(self, str(self.parent.fit.stack_num))
        self.stack_num.setFixedSize(30, 30)
        self.stack_num.returnPressed.connect(self.numStackChanged)
        self.treeWidget.setItemWidget(self.stack_m, 4, self.stack_num)
        for i in range(self.parent.fit.stack_num):
            self.addChild('stack', 'sts_' + str(i), text_val='slope ' + str(i))
            self.addChild('stack', 'stNl_' + str(i), text_val='N_low ' + str(i))
            self.addChild('stack', 'stNu_' + str(i), text_val='N_up ' + str(i))

        self.treeWidget.itemExpanded.connect(self.stateChanged)
        self.treeWidget.itemCollapsed.connect(self.stateChanged)

    def addParent(self, parent, text, checkable=False, expanded=False, checked=False):
        item = QTreeWidgetItem(parent, [text])
        item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
        item.setExpanded(expanded)
        #item.name = text
        return item

    def addContParent(self, parent, text, checkable=False, expanded=True, checked=False):
        item = QTreeWidgetItem(parent, [text])
        item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
        item.name = text
        ind = int(text.split('_')[1])
        if not hasattr(self.parent.fit, 'cont_' + str(ind) + '_0'):
            self.parent.fit.add('cont_' + str(ind) + '_0')
        item.cont_m = QTreeWidgetItem(item)
        item.cont_m.setTextAlignment(3, Qt.AlignRight)
        item.cont_m.setText(3, 'poly num: ')
        item.cont_num = FLineEdit(self, str(self.parent.fit.cont[ind].num))
        item.cont_num.setFixedSize(30, 30)
        item.cont_num.returnPressed.connect(partial(self.numContRegionChanged, ind))
        self.treeWidget.setItemWidget(item.cont_m, 4, item.cont_num)
        item.cont_m.setTextAlignment(5, Qt.AlignRight)
        item.cont_m.setText(5, 'range: ')
        item.cont_left = FLineEdit(self, '{0:6.1f}'.format(self.parent.fit.cont[ind].left).strip())
        item.cont_left.textEdited.connect(partial(self.contRange, ind))
        self.treeWidget.setItemWidget(item.cont_m, 6, item.cont_left)
        item.cont_m.setTextAlignment(7, Qt.AlignRight)
        item.cont_m.setText(7, '...')
        item.cont_right = FLineEdit(self, '{0:6.1f}'.format(self.parent.fit.cont[ind].right).strip())
        item.cont_right.textEdited.connect(partial(self.contRange, ind))
        self.treeWidget.setItemWidget(item.cont_m, 8, item.cont_right)
        item.cont_m.setTextAlignment(9, Qt.AlignRight)
        item.cont_m.setText(9, 'disp: ')
        item.cont_disp = FLineEdit(self, '{0:5.3f}'.format(self.parent.fit.cont[ind].disp).strip())
        item.cont_disp.textEdited.connect(partial(self.contDisp, ind))
        self.treeWidget.setItemWidget(item.cont_m, 10, item.cont_disp)
        setattr(self, item.name + '_applied_exp', QComboBox(self))
        combo = getattr(self, item.name + '_applied_exp')
        combo.setFixedSize(70, 30)
        combo.activated.connect(partial(self.setApplied, name=item.name))
        self.treeWidget.setItemWidget(item.cont_m, 11, combo)
        setattr(self, item.name, item)
        item.setExpanded(self.parent.fit.cont_fit)
        for i in range(self.parent.fit.cont[ind].num):
            self.addChild(text, 'cont_' + str(ind) + '_' + str(i))
        self.parent.fit.cont[ind].update()
        return item

    def addChild(self, parent, name, text_val='value'):
        if 'cf' in name:
            attr = ['val', 'left', 'right', 'step']
            sign = [text_val + ': ', 'range: ', '....']
        else:
            attr = ['val', 'min', 'max', 'step']
            sign = [text_val + ': ', 'range: ', '....', 'step: ']

        if not hasattr(self.parent.fit, name):
            self.parent.fit.add(name)
            if 'hcont' in name:
                self.parent.fit.setValue('hcont', 0, 'vary')
        setattr(self, name, QTreeWidgetItem(getattr(self, parent)))
        for i in [1, 3, 5, 7, 9]:
            getattr(self, name).setTextAlignment(i, Qt.AlignRight)
        for k in range(4):
            if 'cf' not in name or k < 3:
                getattr(self, name).setText(2 * k + 3, sign[k])
                var = name if attr[k] == 'val' else None
                setattr(self, name + '_' + attr[k], FLineEdit(self, getattr(getattr(self.parent.fit, name), attr[k]), var=var, name=name + '_' + attr[k]))
                getattr(self, name + '_' + attr[k]).textChanged[str].connect(partial(self.onChanged, name, attr[k]))
                self.treeWidget.setItemWidget(getattr(self, name), 2 * k + 4, getattr(self, name + '_' + attr[k]))
            else:
                setattr(self, name + '_' + attr[3], FLineEdit(self, getattr(getattr(self.parent.fit, name), attr[3]), var=var, name=name + '_' + attr[3]))
                getattr(self, name + '_' + attr[3]).textChanged[str].connect(partial(self.onChanged, name, attr[3]))
                self.treeWidget.setItemWidget(getattr(self, name), 9, getattr(self, name + '_' + attr[3]))

                setattr(self, name + '_applied', chooseSystemPC(self, name))
                #getattr(self, name + '_applied').triggered.connect(partial(self.setApplied, name=name))
                self.treeWidget.setItemWidget(getattr(self, name), 10, getattr(self, name + '_applied'))

            if any([x in name for x in ['res', 'cf', 'dispz', 'sts']]) and k > 2:
                setattr(self, name + '_applied_exp', QComboBox(self))
                combo = getattr(self, name + '_applied_exp')
                combo.setFixedSize(70, 30)
                combo.activated.connect(partial(self.setApplied, name=name))
                self.treeWidget.setItemWidget(getattr(self, name), 11, combo)
        setattr(self, name + '_vary', QCheckBox(self))
        getattr(self, name + '_vary').stateChanged.connect(partial(self.varyChanged, name))
        getattr(self, name + '_vary').setChecked(self.parent.fit.getValue(name, 'vary'))
        self.treeWidget.setItemWidget(getattr(self, name), 2, getattr(self, name + '_vary'))
        if not getattr(self, parent).isExpanded():
            print('remove', name)
            self.parent.fit.remove(name)

    def setApplied(self, name):
        combo = getattr(self, name + '_applied_exp')
        sp1 = combo.currentText()
        print(name, sp1) #, self.parent.fit.getValue(name, 'addinfo'))
        if 'applied' in sp1:
            sp1 = ''

        if 'cf' in name:
            sp = getattr(self, name + '_applied').currentText()
            print(sp + '_' + sp1)
            self.parent.fit.setValue(name, sp + '_' + sp1, 'addinfo')
            self.parent.fit.getValue(name, 'addinfo')
        elif 'region' in name:
            self.parent.fit.cont[int(name.split('_')[1])].exp = int(getattr(self, name + '_applied_exp').currentText().split('_')[1])
            for i in range(self.parent.fit.cont[int(name.split('_')[1])].num):
                info = self.parent.fit.getValue('cont_' + name.split('_')[1] + '_' + str(i), 'addinfo')
                self.parent.fit.setValue('cont_' + name.split('_')[1] + '_' + str(i), info.split('_')[0] + '_' + sp1, 'addinfo')
        elif 'sts' in name:
            self.parent.fit.setValue(name, sp1, 'addinfo')
            self.parent.fit.setValue(sp1, 0, 'vary')
        else:
            self.parent.fit.setValue(name, sp1, 'addinfo')
        self.refresh()

    def updateMe(self, excl=''):
        try:
            #self.res.setExpanded(self.parent.fit.res_num > 0)
            self.me_num.setText(str(self.parent.fit.me_num))
        except:
            pass

    def updateRes(self, excl=''):
        try:
            #self.res.setExpanded(self.parent.fit.res_num > 0)
            self.res_num.setText(str(self.parent.fit.res_num))
        except:
            pass
        if self.parent.fit.res_num > 0:
            for res in self.parent.fit.list():
                if 'res' in str(res):
                    try:
                        combo = getattr(self, str(res) + '_applied_exp')
                        combo.clear()
                        sp = list(['exp_' + str(i) for i in range(len(self.parent.s))])
                        combo.addItems(sp)
                        s = getattr(res, 'addinfo')
                        if s != '' and s in sp:
                            combo.setCurrentText(s)
                        else:
                            combo.setCurrentIndex(0)
                    except:
                        pass

    def updateCF(self, excl=''):
        try:
            self.cf.setExpanded(self.parent.fit.cf_fit)
            self.cf_num.setText(str(self.parent.fit.cf_num))
        except:
            pass
        if self.parent.fit.cf_fit:
            for cf in self.parent.fit.list():
                if 'cf' in str(cf):
                    try:
                        names = ['val', 'left', 'right']
                        for attr in names:
                            if str(cf) + '_' + attr != excl:
                                getattr(self, str(cf) + '_' + attr).setText(str(getattr(cf, attr)))

                        combo = getattr(self, cf.name + '_applied')
                        combo.update()
                        combo = getattr(self, cf.name + '_applied_exp')
                        combo.clear()
                        sp = list(['exp' + str(i) for i in range(len(self.parent.s))])
                        combo.addItems(['all'] + sp)
                        s = getattr(cf, 'addinfo').split('_')[1]
                        if s != '' and s in sp:
                            combo.setCurrentText(s)
                        else:
                            combo.setCurrentIndex(0)
                    except:
                        pass

    def updateDisp(self, excl=''):
        try:
            #self.disp.setExpanded(self.parent.fit.disp_num > 0)
            self.disp_num.setText(str(self.parent.fit.disp_num))
        except:
            pass
        if self.parent.fit.disp_num > 0:
            for disp in self.parent.fit.list():
                if 'disp' in str(disp):
                    try:
                        names = ['val', 'max', 'min']
                        for attr in names:
                            if str(disp) + '_' + attr != excl:
                                getattr(self, str(disp) + '_' + attr).setText(str(getattr(disp, attr)))
                    except:
                        pass

                if 'dispz' in str(disp):
                    try:
                        combo = getattr(self, str(disp) + '_applied_exp')
                        combo.clear()
                        sp = list(['exp_' + str(i) for i in range(len(self.parent.s))])
                        combo.addItems(sp)
                        s = getattr(disp, 'addinfo')
                        if s != '' and s in sp:
                            combo.setCurrentText(s)
                        else:
                            combo.setCurrentIndex(0)
                    except:
                        pass

    def updateStack(self, excl=''):
        try:
            #self.disp.setExpanded(self.parent.fit.disp_num > 0)
            self.stack_num.setText(str(self.parent.fit.stack_num))
        except:
            pass
        if self.parent.fit.stack_num > 0:
            for stack in self.parent.fit.list():
                if 'st' in str(stack):
                    try:
                        names = ['val', 'max', 'min']
                        for attr in names:
                            if str(stack) + '_' + attr != excl:
                                getattr(self, str(stack) + '_' + attr).setText(str(getattr(stack, attr)))
                    except:
                        pass
                    if 'sts' in str(stack):
                        try:
                            combo = getattr(self, str(stack) + '_applied_exp')
                            combo.clear()
                            sp = list([str(f) for f in self.parent.fit.list() if any([t in str(f) for t in ['N_', 'Ntot']])])
                            combo.addItems([''] + sp)
                            s = getattr(stack, 'addinfo')
                            if s != '' and s in sp:
                                combo.setCurrentText(s)
                                self.parent.fit.setValue(s, 0, 'vary')
                            else:
                                combo.setCurrentIndex(0)
                        except:
                            pass

    def updateCont(self, excl=''):
        try:
            self.cont_m.setExpanded(self.parent.fit.cont_fit)
            self.cont_num.setText(str(self.parent.fit.cont_num))
        except:
            pass
        try:
            if 0:
                f = 1 - self.parent.fit.getValue('hcont', 'vary')
                for attr in ['val', 'min', 'max', 'step']:
                    getattr(self, 'hcont' + '_' + attr).setEnabled(1 - f)
                for i in range(self.parent.fit.cont_num):
                    getattr(self, 'region_' + str(i)).cont_num.setEnabled(f)
                    for k in range(self.parent.fit.cont[i].num):
                        name = 'cont_' + str(i) + '_' + str(k)
                        self.parent.fit.setValue(name, f, 'vary')
                        getattr(self, name + '_vary').setChecked(f)
                        for attr in ['val', 'min', 'max', 'step', 'vary']:
                            getattr(self, name + '_' + attr).setEnabled(f)
        except:
            pass
        if self.parent.fit.cont_fit:
            for i, c in enumerate(self.parent.fit.cont):
                try:
                    combo = getattr(self, 'region_' + str(i) + '_applied_exp')
                    combo.clear()
                    sp = list(['exp_' + str(s) for s in range(len(self.parent.s))])
                    combo.addItems(sp)
                    s = 'exp_' + str(getattr(c, 'exp'))
                    if s != '' and s in sp:
                        combo.setCurrentText(s)
                    else:
                        combo.setCurrentIndex(0)
                except:
                    pass

    def updateTieWindow(self, init=False):
        if init == False:
            text = self.tieWindow.toPlainText()
            self.parent.fit.tieds = {}
            for line in text.split('\n'):
                if not line.startswith('#'):
                    l = line.strip().split()
                    if len(l) == 2:
                        self.parent.fit.addTieds(l[0], l[1])
            self.update()
        else:
            head = '# tie par1 to par2 by printing: <par1> <par2>'
            self.tieWindow.setText('\n'.join([head] + [' '.join([k, v]) for k, v in self.parent.fit.tieds.items()]))

    def setRange(self):
        x = float(self.rangeValue.text())
        for p in self.parent.fit.list_fit():
            if p.unc.val != 0:
                self.parent.fit.setValue(str(p), p.unc.val - p.unc.minus * x, attr='min')
                self.parent.fit.setValue(str(p), p.unc.val + p.unc.plus * x, attr='max')
            else:
                self.parent.fit.setValue(str(p), p.val - p.step * x, attr='min')
                self.parent.fit.setValue(str(p), p.val + p.step * x, attr='max')
        self.update()

    def addSystem(self):
        self.tabNum += 1
        z = self.parent.z_abs if self.tab.currentIndex() == -1 else None
        self.parent.fit.addSys(self.tab.currentIndex(), z=z)
        sys = fitModelSysWidget(self, len(self.parent.fit.sys)-1)
        self.tab.addTab(sys, "sys {:}".format(self.tabNum-1))
        self.tab.setCurrentIndex(len(self.parent.fit.sys)-1)
        self.parent.s.refreshFitComps()
        self.parent.s.reCalcFit(self.tab.currentIndex())
        if hasattr(self.parent, 'chooseFit') and self.parent.chooseFit is not None:
            self.parent.chooseFit.update()
        self.onTabChanged()

    def delSystem(self):
        self.tabNum -= 1
        self.parent.fit.delSys(self.tab.currentIndex())
        self.tab.removeTab(self.tab.currentIndex())
        for i in range(self.tabNum):
            self.tab.setTabText(i, "sys {:}".format(i))
            self.tab.widget(i).ind = i
        self.parent.s.refreshFitComps()
        self.parent.showFit()
        if hasattr(self.parent, 'chooseFit'):
            self.parent.chooseFit.update()
        self.onTabChanged()

    def onTabChanged(self):
        if self.tab.currentWidget() is not None:
            self.tab.currentWidget().refresh()
            self.parent.comp = self.tab.currentIndex()
            self.parent.componentBar.setText("{:d} component".format(self.parent.comp))
            self.parent.s.redrawFitComps()
            self.parent.abs.redraw(z=self.parent.fit.sys[self.parent.comp].z.val)

    def varyChanged(self, name=''):
        if hasattr(self.parent.fit, name):
            setattr(getattr(self.parent.fit, name), 'vary', getattr(self, name + '_vary').isChecked())
        if self.refr:
            self.refresh('varyChanged')

    def stateChanged(self, item):
        if item.name in ['mu', 'dtoh']:
            if item.isExpanded():
                self.parent.fit.add(item.name)
            else:
                self.parent.fit.remove(item.name)

        if item.name == 'cont':
            self.parent.fit.cont_fit = self.cont.isExpanded()
            self.parent.fit.cont_num = int(self.cont_num.text())
            print(self.parent.fit.cont_num)
            if item.isExpanded():
                self.parent.fit.add('hcont')
            else:
                self.parent.fit.remove('hcont')
            for i in reversed(range(self.parent.fit.cont_num)):
                getattr(self, 'region_' + str(i)).setExpanded(item.isExpanded())
                if item.isExpanded():
                    self.addContParent(self.cont, 'region_' + str(i), expanded=self.parent.fit.cont_fit)
                else:
                    self.cont.removeChild(getattr(self, 'region_' + str(i)))

        if 'region' in item.name:
            for i in range(self.parent.fit.cont[int(item.name.split('_')[1])].num):
                if item.isExpanded():
                    self.parent.fit.add('cont_' + item.name.split('_')[1] + '_' + str(i))
                else:
                    self.parent.fit.remove('cont_' + item.name.split('_')[1] + '_' + str(i))


        if item.name == 'res':
            #self.parent.fit.cf_num = int(self.cf_num.text())
            for i in range(self.parent.fit.res_num):
                if self.res.isExpanded():
                    self.parent.fit.add('res_'+str(i))
                else:
                    self.parent.fit.remove('res_'+str(i))

        if item.name == 'cf':
            self.parent.fit.cf_fit = self.cf.isExpanded()
            #self.parent.fit.cf_num = int(self.cf_num.text())
            for i in range(self.parent.fit.cf_num):
                if self.cf.isExpanded():
                    self.parent.fit.add('cf_'+str(i))
                else:
                    self.parent.fit.remove('cf_'+str(i))

        if item.name == 'disp':
            #self.parent.fit.disp_fit = self.disp.isExpanded()
            for i in range(self.parent.fit.disp_num):
                if self.disp.isExpanded():
                    self.parent.fit.add('dispz_'+str(i))
                    self.parent.fit.add('disps_' + str(i))
                else:
                    self.parent.fit.remove('dispz_'+str(i))
                    self.parent.fit.remove('disps_' + str(i))

        if self.refr:
            self.refresh('stateChanged')

    def fromRegions(self):
        if int(self.cont_num.text()) <= len(self.parent.plot.regions):
            self.parent.fit.cont_num = len(self.parent.plot.regions)
            self.cont_num.setText(str(self.parent.fit.cont_num))
            #self.cont_num.setText('0')
            #self.numContChanged()
            self.parent.fit.cont_fit = 1

        for i, r in enumerate(self.parent.plot.regions):
            if not hasattr(self, 'region_' + str(i)):
                self.addContParent(self.cont, 'region_' + str(i), expanded=self.parent.fit.cont_fit)
            self.parent.fit.cont[i].left, self.parent.fit.cont[i].right = r.getRegion()
            s = self.parent.s[self.parent.fit.cont[i].exp]
            if self.parent.normview:
                m = (s.spec.x() > self.parent.fit.cont[i].left) * (s.spec.x() < self.parent.fit.cont[i].right)
                self.parent.fit.cont[i].disp = np.nanmean(s.spec.err()[m] / s.cont.y[m])
            else:
                m = (s.spec.x() > self.parent.fit.cont[i].left) * (s.spec.x() < self.parent.fit.cont[i].right)
                self.parent.fit.cont[i].disp = np.nanmean(s.spec.err()[m])
            self.parent.fit.cont[i].update()
            getattr(self, 'region_' + str(i)).cont_left.setText('{0:6.1f}'.format(self.parent.fit.cont[i].left).strip())
            getattr(self, 'region_' + str(i)).cont_right.setText('{0:6.1f}'.format(self.parent.fit.cont[i].right).strip())
            getattr(self, 'region_' + str(i)).cont_disp.setText('{0:6.2f}'.format(self.parent.fit.cont[i].disp).strip())
        #self.parent.fit.cont_num = len(self.parent.plot.regions)
        #self.cont_num.setText(str(self.parent.fit.cont_num))

    def numContChanged(self):
        k = int(self.cont_num.text())
        sign = 1 if k > self.parent.fit.cont_num else -1
        if k != self.parent.fit.cont_num:
            rang = range(self.parent.fit.cont_num, k) if sign == 1 else range(self.parent.fit.cont_num-1, k-1, -1)
            for i in list(rang):
                if k > self.parent.fit.cont_num:
                    self.addContParent(self.cont, 'region_' + str(i), expanded=self.parent.fit.cont_fit)
                else:
                    #for l in range(self.parent.fit.cont[k].num):
                    self.parent.fit.cont.remove(i)
                    getattr(self, 'cont').removeChild(getattr(self, 'region_' + str(i)))
                    delattr(self, 'region_' + str(i))
            self.parent.fit.cont_num = k
            if self.refr:
                self.refresh()
        if self.parent.fit.cont_num < 1:
            self.parent.fit.cont_fit = 0
        else:
            self.parent.fit.cont_fit = 1

    def numContRegionChanged(self, ind):
        k = int(getattr(self, 'region_' + str(ind)).cont_num.text())
        if k == -1:
            pass
        else:
            sign = 1 if k > self.parent.fit.cont[ind].num else -1
            if k != self.parent.fit.cont[ind].num:
                rang = range(self.parent.fit.cont[ind].num, k) if sign == 1 else range(self.parent.fit.cont[ind].num-1, k-1, -1)
                for i in list(rang):
                    if k > self.parent.fit.cont[ind].num:
                        self.addChild('region_' + str(ind), 'cont_' + str(ind) + '_' + str(i), text_val='cont_' + str(ind) + '_' + str(i))
                        self.parent.fit.cont[ind].update()
                    else:
                        self.parent.fit.remove('cont_' + str(ind) + '_' + str(i))
                        getattr(self, 'region_' + str(ind)).removeChild(getattr(self, 'cont_' + str(ind) + '_' + str(i)))
                self.parent.fit.cont[ind].num = k
                if self.refr:
                    self.refresh()

    def numResChanged(self):
        k = int(self.res_num.text())
        sign = 1 if k > self.parent.fit.res_num else -1
        if k != self.parent.fit.res_num:
            rang = range(self.parent.fit.res_num, k) if sign == 1 else range(self.parent.fit.res_num-1, k-1, -1)
            for i in list(rang):
                if k > self.parent.fit.res_num:
                    self.parent.fit.add('res_' + str(i))
                    self.addChild('res', 'res_' + str(i), text_val='res ' + str(i))
                else:
                    self.parent.fit.remove('res_' + str(i))
                    getattr(self, 'res').removeChild(getattr(self, 'res_' + str(i)))
            self.parent.fit.res_num = k
            if self.refr:
                self.refresh()

    def numMeChanged(self):
        k = int(self.me_num.text())
        sign = 1 if k > self.parent.fit.me_num else -1
        if k != self.parent.fit.me_num:
            rang = range(self.parent.fit.me_num, k) if sign == 1 else range(self.parent.fit.me_num-1, k-1, -1)
            for i in list(rang):
                if k > self.parent.fit.me_num:
                    self.parent.fit.add('me_' + str(i))
                    self.addChild('me', 'me_' + str(i), text_val='Metal. ' + str(i))
                else:
                    self.parent.fit.remove('me_' + str(i))
                    getattr(self, 'me').removeChild(getattr(self, 'me_' + str(i)))
            self.parent.fit.me_num = k
            if self.refr:
                self.refresh()

    def numCfChanged(self):
        k = int(self.cf_num.text())
        sign = 1 if k > self.parent.fit.cf_num else -1
        if k != self.parent.fit.cf_num:
            rang = range(self.parent.fit.cf_num, k) if sign == 1 else range(self.parent.fit.cf_num-1, k-1, -1)
            for i in list(rang):
                if k > self.parent.fit.cf_num:
                    #self.parent.fit.add('cf_'+str(i))
                    self.parent.plot.add_pcRegion(self.parent.vb.viewRange()[0][0], self.parent.vb.viewRange()[0][1])
                else:
                    print(i)
                    self.parent.fit.remove('cf_' + str(i))
                    getattr(self, 'cf').removeChild(getattr(self, 'cf_' + str(i)))
                    self.parent.plot.remove_pcRegion(i)
            self.parent.fit.cf_num = k
            if self.refr:
                self.refresh()

    def numDispChanged(self):
        k = int(self.disp_num.text())
        sign = 1 if k > self.parent.fit.disp_num else -1
        if k != self.parent.fit.disp_num:
            rang = range(self.parent.fit.disp_num, k) if sign == 1 else range(self.parent.fit.disp_num-1, k-1, -1)
            for i in list(rang):
                if k > self.parent.fit.disp_num:
                    for attr in ['dispz', 'disps']:
                        self.parent.fit.add(attr + '_' + str(i))
                        self.addChild('disp', attr + '_' + str(i), text_val=attr + ' ' + str(i))
                else:
                    for attr in ['dispz', 'disps']:
                        self.parent.fit.remove(attr + '_' + str(i))
                        getattr(self, 'disp').removeChild(getattr(self, attr + '_' + str(i)))
            self.parent.fit.disp_num = k
            if self.refr:
                self.refresh()

    def numStackChanged(self):
        k = int(self.stack_num.text())
        sign = 1 if k > self.parent.fit.stack_num else -1
        if k != self.parent.fit.stack_num:
            rang = range(self.parent.fit.stack_num, k) if sign == 1 else range(self.parent.fit.stack_num-1, k-1, -1)
            for i in list(rang):
                if k > self.parent.fit.stack_num:
                    for attr, name in zip(['sts', 'stNl', 'stNu'], ['slope', 'N_low', 'N_up']):
                        self.parent.fit.add(attr + '_' + str(i))
                        self.addChild('stack', attr + '_' + str(i), text_val=name + ' ' + str(i))
                else:
                    for attr in ['sts', 'stNl', 'stNu']:
                        self.parent.fit.remove(attr + '_' + str(i))
                        getattr(self, 'stack').removeChild(getattr(self, attr + '_' + str(i)))
            self.parent.fit.stack_num = k
            if self.refr:
                self.refresh()

    def contRange(self, ind):
        self.parent.fit.cont[ind].left = float(getattr(self, 'region_' + str(ind)).cont_left.text())
        self.parent.fit.cont[ind].right = float(getattr(self, 'region_' + str(ind)).cont_right.text())
        self.parent.fit.cont[ind].update()

    def contDisp(self, ind):
        self.parent.fit.cont[ind].disp = float(getattr(self, 'region_' + str(ind)).cont_disp.text())
        self.parent.fit.cont[ind].update()

    def onChanged(self, s, attr):
        if s in ['mu', 'dtoh'] or any([c in s for c in ['me', 'cont', 'res', 'cf', 'disp', 'sts', 'stNu', 'stNl']]):
            print(hasattr(self.parent.fit, s))
            if hasattr(self.parent.fit, s):
                setattr(getattr(self.parent.fit, s), attr, float(getattr(self, s + '_' + attr).text()))

        if 'res' in s:
            print(getattr(self.parent.fit, s).addinfo, getattr(self.parent.fit, s).addinfo[4:])
            self.parent.s[int(getattr(self.parent.fit, s).addinfo[4:])].resolution = self.parent.fit.getValue(s)

        if self.refr:
            self.refresh(s + '_' + attr)

    def refresh(self, excl='', refresh=None, ):
        #print('refresh:', self.refr)
        names = ['val', 'max', 'min', 'step']
        for s in ['mu', 'dtoh']:
            if hasattr(self.parent.fit, s) and hasattr(self, s + '_vary'):
                getattr(self, s + '_vary').setChecked(getattr(getattr(self.parent.fit, s), 'vary'))
                for attr in names:
                    if s + '_' + attr != excl:
                        getattr(self, s + '_' + attr).setText(getattr(self.parent.fit, s).str(attr))
                    if attr != 'val':
                        getattr(self, s+'_'+attr).setEnabled(getattr(getattr(self.parent.fit, s), 'vary'))
        self.updateRes(excl=excl)
        self.updateCF(excl=excl)
        self.updateDisp(excl=excl)
        self.updateCont(excl=excl)
        self.updateStack(excl=excl)
        if self.tab.currentIndex() > -1:
            self.tab.currentWidget().refresh()

    def selectSpecies(self, text):
        self.addSpecies(text)
        self.chooseSpecies.setCurrentIndex(0)

    def addSpec(self):
        self.addSpecies(self.inputSpecies.text())
        self.inputSpecies.setText('')

    def addSpecies(self, s):
        if len(self.parent.fit.sys) == 0:
            self.addSystem()
            self.tab.setCurrentIndex(0)

        if s not in self.parent.atomic.keys() and 'H2j' in s:
            try:
                self.parent.atomic.readH2(j=int(s[3:]))
            except:
                pass

        if s in self.parent.atomic.keys() and self.tab.currentIndex() > -1:
            for i, sys in enumerate(self.parent.fit.sys):
                if self.addall.isChecked() or self.tab.currentIndex() == sys.ind:
                    self.tab.setCurrentIndex(i)
                    if sys.addSpecies(s):
                        self.tab.currentWidget().addSpecies(s)
                        self.tab.currentWidget().refresh()
                        if self.tied.currentText().strip() != 'tied...':
                            if self.btied.isChecked():
                                self.parent.fit.sys[i].sp[s].b.addinfo = self.tied.currentText().strip()
                                self.tab.currentWidget().updateTieds()
                            if self.Ntied.isChecked() and self.Nscale.text().strip() != '':
                                self.parent.fit.sys[i].sp[s].N.val = self.parent.fit.sys[i].sp[self.tied.currentText().strip()].N.val + float(self.Nscale.text().strip())
                        self.tab.currentWidget().refresh()
                self.parent.fit.showLines([s])

    def upIndex(self):
        pass

    def downIndex(self):
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F and QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.close()
            #
        super(fitModelWidget, self).keyPressEvent(event)

    def closeEvent(self, event):
        self.parent.fitModel = None
        self.onTabChanged()
        event.accept()

class fitModelSysWidget(QFrame):
    """
    Widget for selecting and setting the parameters of fitting system
    """
    def __init__(self, parent, ind):
        super(fitModelSysWidget, self).__init__()
        self.t = Timer('fitrefresh')
        self.init = True
        self.parent = parent
        self.ind = ind
        self.fit = self.parent.parent.fit
        self.initGUI()
        self.setStyleSheet(open('config/styles.ini').read())
        #self.setFixedHeight(550)
        #self.show()
        self.init = False
        #self.refresh()

    def initGUI(self):
        layout = QVBoxLayout()
        self.treeWidget = QTreeWidget()
        self.treeWidget.move(0, 0)
        self.treeWidget.setHeaderHidden(True)
        layout.addWidget(self.treeWidget)

        self.species = {}
        self.treeWidget.setColumnCount(13)
        for i, w in enumerate([110, 10, 30, 60, 80, 70, 80, 10, 80, 70, 80, 30, 50]):
            self.treeWidget.setColumnWidth(i, w)

        # ------------------ z --------------------------------
        attr = ['val', 'min', 'max', 'step']
        sign = ['val: ', 'range: ', '....', 'step: ']
        self.z = self.addParent(self.treeWidget, 'redshift', checkable=True, expanded=True)
        cons_vary = all([hasattr(self.fit.sys[self.ind], attr) for attr in ['turb', 'kin']])
        Ncons_vary = all([hasattr(self.fit.sys[self.ind], attr) for attr in ['Ntot', 'logT', 'logn', 'rad', 'CMB']])
        self.cons = self.addParent(self.treeWidget, 'b tied', checkable=True, expanded=cons_vary)
        self.Ncons = self.addParent(self.treeWidget, 'N tied', checkable=True, expanded=Ncons_vary)
        for s, name in zip(['z', 'turb', 'kin', 'Ntot', 'logT', 'logn', 'logf', 'rad', 'CMB'], ['z', 'b_turb', 'Tkin, K', 'N_tot', 'logT', 'logn', 'logf', 'UV/dist', 'CMB']):
            if s == 'z':
                item = QTreeWidgetItem(self.z)
            elif s in ['turb', 'kin']:
                item = QTreeWidgetItem(self.cons)
                if not cons_vary:
                    self.fit.sys[self.ind].add(s)
            else:
                item = QTreeWidgetItem(self.Ncons)
                if not Ncons_vary:
                    self.fit.sys[self.ind].add(s)
            for i in [1, 3, 5, 7, 9]:
                item.setTextAlignment(i, Qt.AlignRight)
            for k in range(4):
                item.setText(2 * k + 3, sign[k].replace('val', name))
                var = s if attr[k] == 'val' else None
                setattr(self, s + '_' + attr[k], FLineEdit(self.parent, getattr(getattr(self.fit.sys[self.ind], s), attr[k]), var=var, name=s+'_'+attr[k]))
                getattr(self, s + '_' + attr[k]).textChanged[str].connect(partial(self.onChanged, s, attr[k], species=None))
                self.treeWidget.setItemWidget(item, 2 * k + 4, getattr(self, s + '_' + attr[k]))
            setattr(self, s + '_vary', QCheckBox())
            getattr(self, s + '_vary').setChecked(getattr(getattr(self.fit.sys[self.ind], s), 'vary'))
            getattr(self, s + '_vary').stateChanged.connect(self.varyChanged)
            if (s in ['turb', 'kin'] and not cons_vary) or (s in ['Ntot', 'logT', 'logn', 'logf', 'rad', 'CMB'] and not Ncons_vary):
                self.fit.sys[self.ind].remove(s)

            self.treeWidget.setItemWidget(item, 2, getattr(self, s + '_vary'))

            if s == 'z':
                item = QTreeWidgetItem(getattr(self, s))
                item.setText(3, 'v shift: ')
                item.setTextAlignment(3, Qt.AlignRight)
                self.vshift = FLineEdit(self.parent, var='v')
                self.vshift.returnPressed.connect(self.zshift)
                self.treeWidget.setItemWidget(item, 4, self.vshift)
                item.setText(5, 'v range: ')
                item.setTextAlignment(5, Qt.AlignRight)
                self.vrange = FLineEdit(self.parent)
                self.vrange.textEdited[str].connect(self.zrange)
                self.treeWidget.setItemWidget(item, 6, self.vrange)

                item.setText(9, 'v step: ')
                item.setTextAlignment(9, Qt.AlignRight)
                self.vstep = FLineEdit(self.parent)
                self.vstep.textEdited[str].connect(self.zstep)
                self.treeWidget.setItemWidget(item, 10, self.vstep)

        for k in self.fit.sys[self.ind].sp.keys():
            self.addSpecies(k)

        self.treeWidget.itemExpanded.connect(self.stateChanged)
        self.treeWidget.itemCollapsed.connect(self.stateChanged)
        if self.Ncons.isExpanded():
            self.fit.sys[self.ind].pyratio(init=False)
        else:
            self.fit.sys[self.ind].pr = None
        self.setLayout(layout)
        self.refresh()

    def addParent(self, parent, text, checkable=False, expanded=False, checked=False):
        item = QTreeWidgetItem(parent, [text])
        item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
        item.setExpanded(expanded)
        return item

    def addSpecies(self, species):
        if species not in self.species.keys():
            attr = ['val', 'min', 'max', 'step']
            sign = ['val', 'range: ', '....', 'step: ']
            self.species[species] = self.addParent(self.treeWidget, species, checkable=True, expanded=True, checked=True)
            setattr(self, species + '_del', QPushButton('', self))
            button = getattr(self, species + '_del')
            button.setFixedSize(24, 24)
            button.setStyleSheet('QPushButton { background-color:  rgb(49,49,49)}')
            button.clicked.connect(lambda: self.removeSpecies(species))
            button.setIcon(QIcon('images/cross.png'))
            button.setIconSize(QSize(24, 24))
            self.treeWidget.setItemWidget(self.species[species], 12, button)
            for s in ['b', 'N']:
                item = QTreeWidgetItem(self.species[species])

                sp = species + '_' + s + '_vary'
                setattr(self, sp, QCheckBox())
                getattr(self, sp).setChecked(getattr(getattr(self.fit.sys[self.ind].sp[species], s), 'vary'))
                getattr(self, sp).stateChanged.connect(self.varyChanged)
                self.treeWidget.setItemWidget(item, 2, getattr(self, sp))

                name = 'b: ' if s == 'b' else 'logN: '
                for i in [1, 3, 5, 7, 9]:
                    item.setTextAlignment(i, Qt.AlignRight)
                for k in range(4):
                    item.setText(2 * k + 3, sign[k].replace('val', name))
                    sp = species + '_' + s + '_' + attr[k]
                    var = s if attr[k] == 'val' else None
                    setattr(self, sp, FLineEdit(self.parent, var=var, name=s+'_'+attr[k]))
                    getattr(self, sp).textEdited[str].connect(partial(self.onChanged, s, attr[k], species=species))
                    self.treeWidget.setItemWidget(item, 2 * k + 4, getattr(self, sp))
                if s == 'b':
                    setattr(self, species + '_btied', QComboBox(self))
                    combo = getattr(self, species + '_btied')
                    combo.setFixedSize(70, 30)
                    combo.activated.connect(partial(self.setbTied, species=species))
                    self.treeWidget.setItemWidget(item, 12, combo)
                if s == 'N':
                    setattr(self, species + '_Ntied', QComboBox(self))
                    combo = getattr(self, species + '_Ntied')
                    combo.setFixedSize(70, 30)
                    combo.activated.connect(partial(self.setNTied, species=species))
                    self.treeWidget.setItemWidget(item, 12, combo)
                    #setattr(self, species + '_me', QCheckBox('me'))
                    #getattr(self, species + '_me').setFixedSize(70, 30)
                    #getattr(self, species + '_me').clicked.connect(partial(self.setMe, species=species))
                    #self.treeWidget.setItemWidget(item, 12, getattr(self, species + '_me'))
            self.updateTieds()

    def updateTieds(self):
        for species in self.species.keys():
            combo = getattr(self, species + '_btied')
            combo.clear()
            combo.addItems(['tie to'])
            if self.cons.isExpanded():
                combo.addItems(['consist'])
            sp = list(self.species.keys())
            combo.addItems(sp)
            combo.removeItem([combo.itemText(i) for i in range(combo.count())].index(species))
            try:
                s = getattr(getattr(self.fit.sys[self.ind].sp[species], 'b'), 'addinfo').strip()
                if s != '' and (s in sp or s == 'consist'):
                    combo.setCurrentText(s)
                else:
                    combo.setCurrentIndex(0)
            except:
                pass

            combo = getattr(self, species + '_Ntied')
            combo.clear()
            combo.addItems(['tie to'])
            if self.parent.parent.fit.me_num > 0:
                for i in range(self.parent.parent.fit.me_num):
                    combo.addItems(['me_' + str(i)])
            if hasattr(self.parent.parent.fit, 'dtoh'):
                combo.addItems(['DtoH'])
            if self.Ncons.isExpanded():
                combo.addItems(['Ntot'])
            try:
                s = getattr(getattr(self.fit.sys[self.ind].sp[species], 'N'), 'addinfo').strip()
                if s in ['Ntot', 'DtoH'] or 'me' in s:
                    combo.setCurrentText(s)
                else:
                    combo.setCurrentIndex(0)
            except:
                pass

    def removeSpecies(self, species):
        root = self.treeWidget.invisibleRootItem()
        root.removeChild(self.species[species])
        del self.species[species]
        del self.fit.sys[self.ind].sp[species]
        for sp in self.fit.sys[self.ind].sp.keys():
            if self.fit.sys[self.ind].sp[sp].b.addinfo == species:
                setattr(getattr(self.fit.sys[self.ind].sp[sp], 'b'), 'addinfo', '')
                setattr(getattr(self.fit.sys[self.ind].sp[sp], 'b'), 'vary', True)

        try:
            self.parent.parent.chooseFit.update()
        except:
            pass
        self.updateTieds()
        self.refresh()
        #del self.species[species]

    def zshift(self):
        self.fit.sys[self.ind].zshift(float(self.vshift.text()))
        self.parent.parent.s.reCalcFit(self.ind)
        self.refresh()

    def zrange(self):
        self.fit.sys[self.ind].zrange(abs(float(self.vrange.text())))
        self.refresh()

    def zstep(self):
        try:
            self.fit.sys[self.ind].z.step = (abs(float(self.vstep.text()))) / 299792.458 * (1 + self.fit.sys[self.ind].z.val)
        except:
            pass
        self.refresh()

    def varyChanged(self):
        for s in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB', 'sts', 'stNl', 'stNu']:
            if hasattr(self.fit.sys[self.ind], s):
                setattr(getattr(self.fit.sys[self.ind], s), 'vary', getattr(self, s + '_vary').isChecked())
            #print('state:', getattr(getattr(self.fit.sys[self.ind], s), 'vary'))

        if len(self.species) > 0:
            for k, v in self.species.items():
                for s in ['b', 'N']:
                    setattr(getattr(self.fit.sys[self.ind].sp[k], s), 'vary', getattr(self, k + '_' + s + '_vary').isChecked())

        self.refresh()

    def stateChanged(self, item):
        if item == self.cons:
            for s in ['turb', 'kin']:
                if self.cons.isExpanded():
                    self.fit.sys[self.ind].add(s)
                else:
                    self.fit.sys[self.ind].remove(s)
        if item == self.Ncons:
            for s in ['Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
                if self.Ncons.isExpanded():
                    self.fit.sys[self.ind].add(s)
                else:
                    self.fit.sys[self.ind].remove(s)
            if self.Ncons.isExpanded():
                self.fit.sys[self.ind].pyratio(init=True)
            else:
                self.fit.sys[self.ind].pr = None
        if item == self.z:
            if not self.z.isExpanded():
                self.z.setExpanded(True)
        self.updateTieds()
        self.refresh()

    def onChanged(self, s, attr, species=None):
        if s in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
            setattr(getattr(self.fit.sys[self.ind], s), attr, float(getattr(self, s + '_' + attr).text()))
        if s in ['b', 'N']:
            setattr(getattr(self.fit.sys[self.ind].sp[species], s), attr, float(getattr(self, species + '_' + s + '_' + attr).text()))
        if attr == 'val':
            excl = species + '_' + s + '_' + attr if species is not None else s + '_' + attr
            self.refresh(excl, what=s)

    def setbTied(self, species):
        combo = getattr(self, species + '_btied')
        sp = combo.currentText()
        if 'tie' in sp:
            sp = ''

        setattr(getattr(self.fit.sys[self.ind].sp[species], 'b'), 'addinfo', sp)
        getattr(self.fit.sys[self.ind].sp[species], 'b').check()
        self.refresh(what='b')

    def setNTied(self, species):
        combo = getattr(self, species + '_Ntied')
        sp = combo.currentText()
        if 'tie' in sp:
            sp = ''

        setattr(getattr(self.fit.sys[self.ind].sp[species], 'N'), 'addinfo', sp)
        getattr(self.fit.sys[self.ind].sp[species], 'N').check()
        self.refresh(what='Ncons')

    def setMe(self, species):
        info = 'me' if getattr(self, species + '_me').isChecked() else ''
        setattr(getattr(self.fit.sys[self.ind].sp[species], 'N'), 'addinfo', info)

        self.refresh(what='me', ind='all')

    def refresh(self, excl='', what='all', ind=None):
        #print('refresh', excl, self.ind)
        if ind == None:
            ind = self.ind

        if not self.init:
            self.fit.update(what=what, ind=ind)

            names = ['val', 'max', 'min', 'step']
            for s in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT', 'logf', 'rad', 'CMB']:
                if hasattr(self.fit.sys[self.ind], s):
                    getattr(self, s + '_vary').setChecked(getattr(getattr(self.fit.sys[self.ind], s), 'vary'))
                    for attr in names:
                        if s + '_' + attr != excl:
                            getattr(self, s + '_' + attr).setText(getattr(self.fit.sys[self.ind], s).str(attr))
                        if attr != 'val':
                            getattr(self, s + '_' + attr).setEnabled(getattr(getattr(self.fit.sys[self.ind], s), 'vary'))

            if len(self.species) > 0:
                for k, v in self.fit.sys[self.ind].sp.items():
                    try:
                        for s in ['b', 'N']:
                            p = getattr(self.fit.sys[self.ind].sp[k], s)
                            vary = getattr(p, 'vary')
                            tied = False
                            if s == 'b':
                                tied = (p.addinfo != '')
                                getattr(self, k + '_' + s + '_vary').setEnabled(not tied)
                                ind = getattr(self, k + '_btied').findText(p.addinfo) if tied else 0
                                getattr(self, k + '_btied').setCurrentIndex(ind)
                            if s == 'N':
                                tied = (p.addinfo != '')
                                #if 'HI' not in k:
                                #    getattr(self, k + '_me').setChecked(tied)
                                getattr(self, k + '_' + s + '_vary').setEnabled(not tied)
                                ind = getattr(self, k + '_btied').findText(p.addinfo) if tied else 0
                                getattr(self, k + '_btied').setCurrentIndex(ind)
                            getattr(self, k + '_' + s + '_vary').setChecked(vary and not tied)
                            for attr in names:
                                if k + '_' + s + '_' + attr != excl:
                                    getattr(self, k + '_' + s + '_' + attr).setText(p.str(attr))
                                getattr(self, k + '_' + s + '_' + attr).setEnabled((attr == 'val' or vary) and not tied)

                    except:
                        pass

            self.updateTieds()

            #if self.parent.parent.chooseFit is not None:
            #    self.parent.parent.chooseFit.update()


class fitResultsWidget(QWidget):
    def __init__(self, parent):
        super(fitResultsWidget, self).__init__()
        self.parent = parent
        self.res = None
        self.view = 'plain'
        self.init_GUI()
        self.setStyleSheet(open('config/styles.ini').read())

        self.setGeometry(200, 100, 750, 900)
        self.setWindowTitle('Fit results')
        self.refresh()

    def init_GUI(self):
        layout = QVBoxLayout()
        self.output = QTextEdit()
        layout.addWidget(self.output)
        hl = QHBoxLayout()
        self.latexTable = QPushButton('View:')
        menu = QMenu()
        menu.setStyleSheet(open('config/styles.ini').read())
        plainView = QAction("Plain", menu)
        plainView.triggered.connect(partial(self.refresh, 'plain'))
        menu.addAction(plainView)
        latexView = QAction("Latex", menu)
        latexView.triggered.connect(partial(self.refresh, 'latex'))
        menu.addAction(latexView)
        tableView = QAction("Table", menu)
        tableView.triggered.connect(partial(self.refresh, 'table'))
        menu.addAction(tableView)
        classView = QAction("Class", menu)
        classView.triggered.connect(partial(self.refresh, 'class'))
        menu.addAction(classView)
        self.latexTable.setMenu(menu)
        self.latexTable.setFixedSize(100, 30)
        self.latexTable.clicked.connect(self.refresh)
        self.vert = QCheckBox('vertical')
        self.vert.setChecked(True)
        self.vert.clicked.connect(self.refresh)
        self.tiedN = QCheckBox('Tied N')
        self.tiedN.setChecked(False)
        self.tiedN.clicked.connect(self.refresh)
        self.showb = QCheckBox('Show b')
        self.showb.setChecked(False)
        self.showb.clicked.connect(self.refresh)
        self.showv = QCheckBox('Delta v, z_ref:')
        self.showv.setChecked(False)
        self.showv.clicked.connect(self.refresh)
        self.zref = FLineEdit(self, 0.0, var='z', name='z_ref')
        #self.zref = QLineEdit(self.z_ref)
        self.zref.setFixedSize(80, 30)
        self.zref.setEnabled(self.showv.isChecked())
        self.zref.returnPressed.connect(self.refresh)
        self.vcomp = QComboBox(self)
        self.vcomp.setFixedSize(40, 30)
        self.vcomp.addItems([str(i+1) for i in range(len(self.parent.fit.sys))])
        self.vcomp.activated.connect(self.set_zref)
        if len(self.parent.fit.sys) > 0:
            self.comp = 1
            self.zref.setText(str(self.parent.fit.sys[self.comp - 1].z.val))
            self.vcomp.setCurrentIndex(self.comp)

        self.showLFR = QCheckBox('LFR')
        self.showLFR.setChecked(True)
        self.showLFR.clicked.connect(self.refresh)
        hl.addWidget(self.latexTable)
        hl.addWidget(self.vert)
        hl.addWidget(self.tiedN)
        hl.addWidget(self.showb)
        hl.addWidget(self.showv)
        hl.addWidget(self.zref)
        hl.addWidget(self.vcomp)
        hl.addWidget(self.showLFR)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        self.showtotal = QCheckBox('Total')
        self.showtotal.setChecked(False)
        self.showtotal.clicked.connect(self.refresh)
        self.showme = QCheckBox('Metallicity')
        self.showme.setChecked(False)
        self.showme.clicked.connect(self.refresh)
        self.HI = a(20.47, 0.01, 0.01)
        self.HIvalue = QLineEdit(self.HI.latex(eqs=0))
        self.HIvalue.setFixedSize(150, 30)
        self.HIvalue.setEnabled(self.showme.isChecked())
        self.HIvalue.returnPressed.connect(self.refresh)
        self.showdep = QCheckBox('Depletion')
        self.showdep.setChecked(False)
        self.showdep.clicked.connect(self.refresh)
        self.depref = 'ZnII'
        self.depRef = QLineEdit(self.depref)
        self.depRef.setFixedSize(30, 30)
        self.depRef.setEnabled(self.showdep.isChecked())
        self.depRef.returnPressed.connect(self.refresh)
        self.showratios = QCheckBox('Ratios')
        self.showratios.setChecked(False)
        self.showratios.clicked.connect(self.refresh)
        self.ratios = QLineEdit()
        self.ratios.setFixedSize(90, 30)
        self.ratios.setEnabled(self.showratios.isChecked())
        self.ratios.returnPressed.connect(self.refresh)
        hl.addWidget(self.showtotal)
        hl.addWidget(self.showme)
        hl.addWidget(self.HIvalue)
        hl.addWidget(self.showdep)
        hl.addWidget(self.depRef)
        hl.addWidget(self.showratios)
        hl.addWidget(self.ratios)
        hl.addStretch(0)
        layout.addLayout(hl)

        hl = QHBoxLayout()
        restoval = QPushButton('set Values')
        restoval.clicked.connect(self.restoval)
        hl.addWidget(restoval)
        hl.addStretch(0)
        layout.addLayout(hl)

        self.setLayout(layout)

    def set_zref(self):
        self.comp = self.vcomp.currentIndex() + 1
        if len(self.parent.fit.sys) > 0:
            self.z_ref = self.parent.fit.sys[self.comp-1].z.val
            self.zref.setText(str(self.z_ref))

    def refresh(self, view=None):
        self.zref.setEnabled(self.showv.isChecked() * len(self.parent.fit.sys) > 0)
        self.vcomp.setEnabled(self.showv.isChecked() * len(self.parent.fit.sys) > 0)
        if len(self.parent.fit.sys) > 0:
            self.z_ref = float(self.zref.text())
            self.comp = self.vcomp.currentIndex() + 1
        self.HIvalue.setEnabled(self.showme.isChecked())
        self.HI = a(self.HIvalue.text())
        self.depRef.setEnabled(self.showdep.isChecked())
        self.depref = self.depRef.text() if self.showdep.isChecked() else ''
        self.ratios.setEnabled(self.showratios.isChecked())
        if isinstance(view, str):
            self.view = view
        if self.view == 'plain':
            self.text()
        if self.view == 'latex':
            self.latex()
        if self.view == 'table':
            self.latex(view='widget')
        if self.view == 'class':
            self.classView()

    def text(self):
        s = ''
        for p in self.parent.fit.list():
            s += p.fitres() + '\n'
        self.output.setText(s)

    def latex(self, view=''):
        fit = self.parent.fit
        if (self.showtotal.isChecked() or self.showme.isChecked() or self.showdep.isChecked()): #and self.res is None:
            self.res = self.parent.showMetalAbundance(component=0, dep_ref=self.depref, HI=self.HI)

        sps = OrderedDict()
        for sys in fit.sys:
            for sp in sys.sp.keys():
                sps[sp] = 1
        #names = ['par'] + ['comp '+str(i+1) for i in range(len(fit.sys))]

        d = ['comp', 'z']
        if self.showv.isChecked():
            d += ['$\Delta$v, km/s']
        if self.showb.isChecked():
            d += ['b, km/s']
        if self.tiedN.isChecked():
            d += [r'$\log n [\rm cm^{-3}]$']
            d += [r'$\log T [\rm cm^{-3}]$']
            d += [r'$\log N_{\rm tot}$']
        d += list([r'$\log N$(' + s + ')' for s in sps.keys()])
        if self.showtotal.isChecked() and any([all([el in sp for sp in sys.sp.keys()]) for el in ['H2', 'CO', 'HD', 'CI']]):
            d += [r'$\log N_{\rm tot}$']
        if self.showratios.isChecked():
            d += [r'$\log N$({0:s})/$N$({1:s})'.format(s.split('/')[0], s.split('/')[1]) for s in self.ratios.text().split()]
        if self.showLFR.isChecked() and self.parent.fit.cf_fit:
            d += [r'LFR']
        data = [d]

        for sys in fit.sys:
            d = [str(fit.sys.index(sys)+1)]
            d.append(sys.z.fitres(latex=True, showname=False))

            if self.showv.isChecked():
                d.append('{:.1f}'.format((sys.z.val - self.z_ref)/(1 + self.z_ref) * 299792.46))

            if self.showb.isChecked():
                sp = sys.sp[list(sys.sp.keys())[0]]
                if sp.b.addinfo != '':
                    sp = sys.sp[sp.b.addinfo]
                d.append(sp.b.fitres(latex=True, dec=2, showname=False))

            if self.tiedN.isChecked():
                d.append(sys.logn.fitres(latex=True, dec=2, showname=False))
                d.append(sys.logT.fitres(latex=True, dec=2, showname=False))
                d.append(sys.Ntot.fitres(latex=True, dec=2, showname=False))

            for sp in sps.keys():
                if sp in sys.sp.keys() and 'Ntot' not in sys.sp[sp].N.addinfo:
                    sys.sp[sp].N.unc.log()
                    d.append(sys.sp[sp].N.fitres(latex=True, dec=2, showname=False))
                else:
                    d.append(' ')

            t = ''
            if self.showtotal.isChecked():
                for el in ['H2', 'CO', 'HD', 'CI', 'SiII']:
                    if el in sys.total.keys():
                        d.append(sys.total[el].N.fitres(latex=True, dec=2, showname=False))
                        t = el
                    elif all([el in sp for sp in sys.sp.keys()]):
                        n = a(0, 0, 0)
                        for v in sys.sp.values():
                            n += v.N.unc
                        sys.addSpecies(el + 't')
                        sys.sp[el + 't'].N.set(n.val)
                        sys.sp[el + 't'].N.set(n, attr='unc')
                        d.append(sys.sp[el + 't'].N.fitres(latex=True, dec=2, showname=False))
                        del sys.sp[el + 't']
                        t = el

            if self.showratios.isChecked():
                for s in self.ratios.text().split():
                    sp = s.split('/')
                    print(sys.sp[sp[0]].N.unc / sys.sp[sp[1]].N.unc)
                    d.append((sys.sp[sp[0]].N.unc / sys.sp[sp[1]].N.unc).latex(f=2))

            if self.showLFR.isChecked() and fit.cf_fit:
                for i in range(fit.cf_num):
                    if 'sys'+str(fit.sys.index(sys)) in fit.getPar('cf_'+str(i)).addinfo:
                        d.append(fit.getPar('cf_'+str(i)).fitres(latex=True, showname=False))
                        break
                else:
                    d += ['']

            data.append(d)

        for show, ind, name in zip(['showtotal', 'showme', 'showdep'], [0, 1, 2], [r'$\log N_{\rm tot}$', r'$\rm [X/H]$', r'$\rm [X/' + self.depref + ']$']):
            if getattr(self, show).isChecked():
                if ind > 0 or (ind == 0 and len(self.parent.fit.sys) > 1):
                    d = [name, '']
                    for show in ['showv', 'showb', 'tiedN']:
                        if getattr(self, show).isChecked():
                            d += ['']
                    for sp in sps.keys():
                        d.append(self.res[sp][ind].log().latex())
                    print('total:', t)
                    if t != '':
                        if t not in fit.total.sp.keys():
                            n = a(0, 0, 0, 'd')
                            for sp in sps.keys():
                                n += self.res[sp][0].log()
                            d += [n.log().latex()]
                        else:
                            d += [fit.total.sp[t].N.unc.latex()]
                    data.append(d)
                    print(d)

        #print(data)
        if view == 'widget':
            if hasattr(self, 'table') and self.table is not None:
                self.table.close()
            self.table = pg.TableWidget(sortable=False)
            self.table.show()
            self.table.resize(1000, 1000)
            self.table.setStyleSheet(open('config/styles.ini').read())
            self.table.setWindowTitle('Fit results')
            self.table.setData(data)

        if self.vert.isChecked():
            data = [list(i) for i in zip(*data)]

        #print('data:', data[0], data[1:])
        output = StringIO()
        ascii.write([list(i) for i in zip(*data[1:])], output, names=data[0], format='latex')
        #ascii.write(data, output, names=names, format='latex')
        table = output.getvalue()
        #print(table)
        self.output.setText(table)
        output.close()

    def classView(self):
        s = 'q.comp = []\n'
        for sys in self.parent.fit.sys:
            s += sys.z.fitres(classview=True) + '\n'
            for el in ['H2', 'CO', 'HD', 'CI']:
                if any([el in sp for sp in sys.sp.keys()]):
                    n = a(0,0,0)
                    for k, v in sys.sp.items():
                        if k.startswith(el):
                            n += v.N.unc
                    sys.addSpecies(el+'t')
                    sys.sp[el + 't'].N.set(n.val)
                    sys.sp[el + 't'].N.set(n, attr='unc')
                    s += sys.sp[el + 't'].N.fitres(classview=True).replace('t', '') + '\n'
                    del sys.sp[el + 't']
            for k, v in sys.sp.items():
                n = k if v.b.addinfo == '' else v.b.addinfo
                b = ", b=" + sys.sp[n].b.fitres(aview=True) + ")"
                s += v.N.fitres(classview=True).replace(')', '') + b + '\n'
            s += 'q.comp.append(co)\n'
        self.output.setText(s)

    def restoval(self):
        for p in self.parent.fit.pars():
            print(p)
            res = self.parent.fit.getValue(p, 'unc')
            print(res)
            self.parent.fit.setValue(p, res.val)
            self.parent.fit.setValue(p, (res.plus + res.minus) / 2, 'step')

    def keyPressEvent(self, event):
        super(fitResultsWidget, self).keyPressEvent(event)
        key = event.key()

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_T:
                if (QApplication.keyboardModifiers() == Qt.ControlModifier):
                    self.parent.fitResults.close()

    def closeEvent(self, event):
        self.parent.fitResults = None
        event.accept()



class chooseFitParsWidget(QWidget):
    """
    Widget for choose fitting parameters during the fit.
    """
    def __init__(self, parent, closebutton=True):
        super().__init__()
        self.parent = parent
        #self.resize(700, 900)
        #self.move(400, 100)
        self.setStyleSheet(open('config/styles.ini').read())

        self.saved = []
        #if hasattr(self.parent, 'fit'):
        #    for par in self.parent.fit.list():
        #        self.saved[str(par)] = par.fit

        layout = QVBoxLayout()

        self.selectAll = QPushButton("Select all")
        self.selectAll.setFixedSize(110, 30)
        self.selectAll.clicked[bool].connect(partial(self.select, True))
        self.deselectAll = QPushButton("Deselect all")
        self.deselectAll.setFixedSize(110, 30)
        self.deselectAll.clicked[bool].connect(partial(self.select, False))
        layout.addWidget(self.selectAll)
        layout.addWidget(self.deselectAll)
        self.scroll = None

        self.layout = QVBoxLayout()
        self.update()
        layout.addLayout(self.layout)


        if closebutton:
            self.okButton = QPushButton("Close")
            self.okButton.setFixedSize(110, 30)
            self.okButton.clicked[bool].connect(self.ok)
            hbox = QHBoxLayout()
            hbox.addWidget(self.okButton)
            hbox.addStretch()
            layout.addLayout(hbox)

        self.setLayout(layout)

    def update(self):
        for s in self.saved:
            try:
                self.layout.removeWidget(getattr(self, s))
                getattr(self, s).deleteLater()
            except:
                pass
        if self.scroll is not None:
            self.layout.removeWidget(self.scroll)
            self.scroll.deleteLater()

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        #self.scroll.setMaximumHeight(self.height()-150)
        self.scrollContent = QWidget(self.scroll)
        if hasattr(self.parent, 'fit'):
            l = QVBoxLayout()
            self.saved = []
            for par in self.parent.fit.list():
                self.saved.append(str(par))
                setattr(self, str(par), QCheckBox(str(par)))
                getattr(self, str(par)).setChecked(par.fit & par.vary)
                getattr(self, str(par)).setEnabled(par.vary)
                getattr(self, str(par)).clicked[bool].connect(partial(self.click, str(par)))
                l.addWidget(getattr(self, str(par)))
            l.addStretch()
            self.scrollContent.setLayout(l)
            self.scroll.setWidget(self.scrollContent)
        self.layout.addWidget(self.scroll)

    def select(self, s):
        if hasattr(self.parent, 'fit'):
            for par in self.parent.fit.list():
                par.fit = par.vary & s
                getattr(self, str(par)).setChecked(par.fit)
        self.updateShow()

    def click(self, s):
        self.parent.fit.setValue(s, getattr(self, s).isChecked(), attr='fit')
        self.updateShow()

    def updateShow(self):
        try:
            self.parent.MCMC.chooseShow.update()
        except:
            pass

    def ok(self):
        self.hide()
        self.parent.chooseFit = None
        self.deleteLater()

    def cancel(self):
        for par in self.parent.fit.list():
            par.fit = self.saved[str(par)]
        self.close()

class chooseShowParsWidget(QWidget):
    """
    Widget for choose fitting parameters during the fit.
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setStyleSheet(open('config/styles.ini').read())

        self.saved = []

        layout = QVBoxLayout()

        self.selectAll = QPushButton("Select all")
        self.selectAll.setFixedSize(110, 30)
        self.selectAll.clicked[bool].connect(partial(self.select, True))
        self.deselectAll = QPushButton("Deselect all")
        self.deselectAll.setFixedSize(110, 30)
        self.deselectAll.clicked[bool].connect(partial(self.select, False))
        layout.addWidget(self.selectAll)
        layout.addWidget(self.deselectAll)
        self.scroll = None
        self.layout = QVBoxLayout()
        self.update(init=True)
        layout.addLayout(self.layout)

        self.setLayout(layout)

    def update(self, init=False):
        for s in self.saved:
            try:
                self.layout.removeWidget(getattr(self, s))
                getattr(self, s).deleteLater()
            except:
                pass
        if self.scroll is not None:
            self.layout.removeWidget(self.scroll)
            self.scroll.deleteLater()

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        #self.scroll.setMaximumHeight(self.height()-150)
        self.scrollContent = QWidget(self.scroll)
        if hasattr(self.parent.parent, 'fit'):
            l = QVBoxLayout()
            self.saved = []
            for par in self.parent.parent.fit.list():
                par.show = (par.show | init) & par.fit & par.vary
                if par.fit & par.vary:
                    self.saved.append(str(par))
                    setattr(self, str(par), QCheckBox(str(par)))
                    getattr(self, str(par)).setChecked(par.show)
                    getattr(self, str(par)).clicked[bool].connect(partial(self.click, str(par)))
                    l.addWidget(getattr(self, str(par)))
            l.addStretch(0)
            self.scrollContent.setLayout(l)
            self.scroll.setWidget(self.scrollContent)
        self.layout.addWidget(self.scroll)

    def select(self, s):
        if hasattr(self.parent.parent, 'fit'):
            for par in self.parent.parent.fit.list_fit():
                par.show = s
                getattr(self, str(par)).setChecked(par.show)

    def click(self, s):
        self.parent.parent.fit.setValue(s, getattr(self, s).isChecked(), attr='show')
