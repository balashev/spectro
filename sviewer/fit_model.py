from functools import partial
from PyQt5.QtCore import Qt, QSize, QLocale
from PyQt5.QtGui import QIcon, QDoubleValidator
from PyQt5.QtWidgets import (QWidget, QTreeWidget, QTabWidget, QHBoxLayout,
                             QVBoxLayout, QTreeWidgetItem, QLineEdit, QFrame,
                             QLabel, QPushButton, QCheckBox, QComboBox,
                             QScrollArea, QApplication, QTextEdit)

from .utils import Timer
from ..pyratio import pyratio

class FLineEdit(QLineEdit):
    def __init__(self, parent, text='', var=None, name=''):
        super().__init__(str(text))
        self.parent = parent
        self.var = var
        self.name = name
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
            if self.name[:1] in ['b', 'N', 'd', 'm', 'c', 'r', 'l']:
                validator.setDecimals(5)
            self.setValidator(validator)

    def setLength(self, l=None):
        if l is None:
            if self.name[:1] == 'z':
                self.setMaxLength(10)
            if self.name[:1] in ['b', 'N', 'd', 'm', 'c', 'r', 'l']:
                self.setMaxLength(9)
        else:
            self.setMaxLength(l)

    def calcFit(self):
        if self.var in ['N', 'b', 'z', 'kin', 'turb', 'v', 'c', 'Ntot', 'logn', 'logT']:
            self.parent.parent.s.reCalcFit(self.parent.tab.currentIndex())
        elif self.var in ['mu', 'me', 'dtoh', 'res']:
            self.parent.parent.s.prepareFit()
            self.parent.parent.s.calcFit()

class fitModelWidget(QWidget):
    def __init__(self, parent):
        self.refr = False
        super(fitModelWidget, self).__init__()
        self.parent = parent
        self.setStyleSheet(open('config/styles.ini').read())

        self.tab = QTabWidget()
        self.tab.setGeometry(0, 0, 1050, 900)
        self.tab.setMinimumSize(1050, 300)
        self.tabIndex = 1
        t = Timer('fitmodel')
        self.tabNum = len(self.parent.fit.sys)
        for i in range(self.tabNum):
            sys = fitModelSysWidget(self, i)
            t.time(i)
            self.tab.addTab(sys, "sys {:}".format(i+1))
            t.time(i)

        self.tab.currentChanged.connect(self.onTabChanged)
        self.tab.setCurrentIndex(self.parent.comp)

        self.addFixedTreeWidget()

        speciesbox = QHBoxLayout()
        lbl = QLabel('type: ')
        speciesbox.addWidget(lbl)
        self.inputSpecies = QLineEdit()
        self.inputSpecies.setFixedSize(100, 30)
        self.inputSpecies.returnPressed.connect(self.addSpec)
        speciesbox.addWidget(self.inputSpecies)
        lbl = QLabel('  or  ')
        speciesbox.addWidget(lbl)
        self.chooseSpecies = QComboBox()
        self.chooseSpecies.setFixedSize(100, 30)

        self.chooseSpecies.addItems(['choose...'] + list(self.parent.atomic.keys()))

        self.chooseSpecies.activated[str].connect(self.selectSpecies)
        speciesbox.addWidget(self.chooseSpecies)
        speciesbox.addStretch(1)
        # self.setLayout(layout)

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

        layout = QVBoxLayout()
        layout.addWidget(self.tab)
        layout.addWidget(QLabel('add species: '))
        layout.addLayout(speciesbox)
        layout.addSpacing(50)
        #layout.addStretch(1)
        layout.addLayout(hbox)
        l = QHBoxLayout()
        l.addLayout(layout)
        l.addWidget(self.treeWidget)

        self.setLayout(l)

        self.setGeometry(200, 200, 2050, 900)
        self.setWindowTitle('Fit model')
        self.show()
        self.refresh()
        self.refr = True

    def addFixedTreeWidget(self):
        self.treeWidget = QTreeWidget()
        self.treeWidget.move(0, 0)
        self.treeWidget.setHeaderHidden(True)
        self.treeWidget.setColumnCount(12)
        for i, w in enumerate([150, 30, 40, 70, 110, 90, 110, 20, 110, 70, 80, 10]):
            self.treeWidget.setColumnWidth(i, w)

        attr = ['val', 'min', 'max', 'step']
        sign = ['value: ', 'range: ', '....', 'step: ']
        self.cont = self.addParent(self.treeWidget, 'Continuum', expanded=self.parent.fit.cont_fit)
        self.cont.name = 'cont'

        self.cont_m = QTreeWidgetItem(self.cont)
        self.cont_m.setTextAlignment(3, Qt.AlignRight)
        self.cont_m.setText(3, 'num: ')
        self.cont_num = FLineEdit(self, str(self.parent.fit.cont_num))
        self.cont_num.setFixedSize(30, 30)
        self.cont_num.returnPressed.connect(self.numContChanged)
        self.treeWidget.setItemWidget(self.cont_m, 4, self.cont_num)
        self.cont_left = FLineEdit(self, str(self.parent.fit.cont_left))
        self.cont_left.textEdited.connect(self.contRange)
        self.cont_m.setTextAlignment(5, Qt.AlignRight)
        self.cont_m.setText(5, 'range: ')
        self.treeWidget.setItemWidget(self.cont_m, 6, self.cont_left)
        self.cont_right = FLineEdit(self, str(self.parent.fit.cont_right))
        self.cont_right.textEdited.connect(self.contRange)
        self.cont_m.setTextAlignment(7, Qt.AlignRight)
        self.cont_m.setText(7, '...')
        self.treeWidget.setItemWidget(self.cont_m, 8, self.cont_right)
        for i in range(self.parent.fit.cont_num):
            self.addChild('cont', 'cont' + str(i))
        self.cont.setExpanded(self.parent.fit.cont_fit)

        for s, name in zip(['mu', 'me', 'dtoh', 'res'], ['mp/me', 'metallicity', 'D/H', 'resolution']):
            print(s+'_p')
            setattr(self, s + '_p', self.addParent(self.treeWidget, name, expanded=hasattr(self.parent.fit, s)))
            getattr(self, s + '_p').name = s
            self.addChild(s + '_p', s)
        #self.treeWidget.itemChanged.connect(self.stateChanged)

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
            self.addChild('cf', 'cf_' + str(i))
        self.cf.setExpanded(self.parent.fit.cf_fit)

        self.treeWidget.itemExpanded.connect(self.stateChanged)
        self.treeWidget.itemCollapsed.connect(self.stateChanged)

    def addParent(self, parent, text, checkable=False, expanded=False, checked=False):
        item = QTreeWidgetItem(parent, [text])
        item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
        item.setExpanded(expanded)

        return item

    def addChild(self, parent, name):
        attr = ['val', 'min', 'max', 'step']
        sign = ['value: ', 'range: ', '....', 'step: ']
        if not hasattr(self.parent.fit, name):
            self.parent.fit.add(name)
        setattr(self, name, QTreeWidgetItem(getattr(self, parent)))
        for i in [1, 3, 5, 7, 9]:
            getattr(self, name).setTextAlignment(i, Qt.AlignRight)
        for k in range(4):
            if 'cf' not in name or k < 3:
                getattr(self, name).setText(2 * k + 3, sign[k])
                var = name if attr[k] == 'val' else None
                setattr(self, name + '_' + attr[k], FLineEdit(self, getattr(getattr(self.parent.fit, name), attr[k]), var=var, name=name+'_'+attr[k]))
                getattr(self, name + '_' + attr[k]).textChanged[str].connect(partial(self.onChanged, name, attr[k]))
                self.treeWidget.setItemWidget(getattr(self, name), 2 * k + 4, getattr(self, name + '_' + attr[k]))
            else:
                setattr(self, name + '_applied', QComboBox(self))
                combo = getattr(self, name + '_applied')
                combo.setFixedSize(70, 30)
                # print('create combo:', species)
                combo.activated.connect(partial(self.setApplied, name=name))
                self.treeWidget.setItemWidget(getattr(self, name), 10, combo)
                setattr(self, name + '_applied_exp', QComboBox(self))
                combo = getattr(self, name + '_applied_exp')
                combo.setFixedSize(70, 30)
                # print('create combo:', species)
                combo.activated.connect(partial(self.setApplied, name=name))
                self.treeWidget.setItemWidget(getattr(self, name), 11, combo)

        setattr(self, name + '_vary', QCheckBox(self))
        getattr(self, name + '_vary').stateChanged.connect(partial(self.varyChanged, name))
        getattr(self, name + '_vary').setChecked(True)
        self.treeWidget.setItemWidget(getattr(self, name), 2, getattr(self, name + '_vary'))
        if not getattr(self, parent).isExpanded():
            self.parent.fit.remove(name)

    def setApplied(self, name):
        combo = getattr(self, name + '_applied')
        sp = combo.currentText()
        if 'applied' in sp:
            sp = ''
        combo = getattr(self, name + '_applied_exp')
        sp1 = combo.currentText()
        if 'applied' in sp1:
            sp1 = ''
        setattr(getattr(self.parent.fit, name), 'addinfo', sp+'_'+sp1)
        print(sp+'_'+sp1)
        self.refresh()

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
                        names = ['val', 'max', 'min']
                        for attr in names:
                            if str(cf) + '_' + attr != excl:
                                getattr(self, str(cf) + '_' + attr).setText(str(getattr(cf, attr)))

                        combo = getattr(self, cf.name + '_applied')
                        combo.clear()
                        sp = list(['sys'+ str(i) for i in range(len(self.parent.fit.sys))])
                        combo.addItems(['all']+sp)
                        s = getattr(cf, 'addinfo').split('_')[0]
                        if s != '' and s in sp:
                            combo.setCurrentText(s)
                        else:
                            combo.setCurrentIndex(0)
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

    def addSystem(self):
        self.tabNum += 1
        z = self.parent.z_abs if self.tab.currentIndex() == -1 else None
        self.parent.fit.addSys(self.tab.currentIndex(), z=z)
        sys = fitModelSysWidget(self, len(self.parent.fit.sys)-1)
        self.tab.addTab(sys, "sys {:}".format(self.tabNum))
        self.tab.setCurrentIndex(len(self.parent.fit.sys)-1)
        self.parent.s.refreshFitComps()
        self.parent.s.reCalcFit(self.tab.currentIndex())
        if hasattr(self.parent, 'chooseFit'):
            self.parent.chooseFit.update()
        self.onTabChanged()

    def delSystem(self):
        self.tabNum -= 1
        self.parent.fit.delSys(self.tab.currentIndex())
        self.tab.removeTab(self.tab.currentIndex())
        for i in range(self.tabNum):
            self.tab.setTabText(i, "sys {:}".format(i + 1))
            self.tab.widget(i).ind = i
        self.parent.s.refreshFitComps()
        self.parent.showFit()
        if hasattr(self.parent, 'chooseFit'):
            self.parent.chooseFit.update()
        self.onTabChanged()

    def onTabChanged(self):
        self.tab.currentWidget().refresh()
        self.parent.comp = self.tab.currentIndex()
        self.parent.s.redrawFitComps()
        self.parent.abs.redraw(z=self.parent.fit.sys[self.parent.comp].z.val)

    def varyChanged(self, name=''):
        if 0:
            for s in ['mu', 'me', 'dtoh', 'res']:
                if hasattr(self.parent.fit, s):
                    setattr(getattr(self.parent.fit, s), 'vary', getattr(self, s+'_vary').isChecked())
            if self.parent.fit.cont_num > 0:
                for i in range(self.parent.fit.cont_num):
                    s = 'cont' + str(i)
                    if hasattr(self.parent.fit, s):
                        setattr(getattr(self.parent.fit, s), 'vary', getattr(self, s + '_vary').isChecked())
            if self.parent.fit.cf_num > 0:
                for i in range(self.parent.fit.cf_num):
                    s = 'cf_' + str(i)
                    if hasattr(self.parent.fit, s):
                        setattr(getattr(self.parent.fit, s), 'vary', getattr(self, s + '_vary').isChecked())
        else:
            if hasattr(self.parent.fit, name):
                setattr(getattr(self.parent.fit, name), 'vary', getattr(self, name + '_vary').isChecked())
        if self.refr:
            self.refresh('varyChanged')

    def stateChanged(self, item):
        print('stateChaged', item.name)
        if item.name in ['mu', 'me', 'dtoh', 'res']:
            if item.isExpanded():
                self.parent.fit.add(item.name)
            else:
                self.parent.fit.remove(item.name)
        if item.name == 'cont':
            self.parent.fit.cont_fit = self.cont.isExpanded()
            self.parent.fit.cont_num = int(self.cont_num.text())
            for i in range(self.parent.fit.cont_num):
                if self.cont.isExpanded():
                    self.parent.fit.add('cont'+str(i))
                else:
                    self.parent.fit.remove('cont'+str(i))

        if item.name == 'cf':
            self.parent.fit.cf_fit = self.cf.isExpanded()
            #self.parent.fit.cf_num = int(self.cf_num.text())
            for i in range(self.parent.fit.cf_num):
                if self.cf.isExpanded():
                    self.parent.fit.add('cf_'+str(i))
                else:
                    self.parent.fit.remove('cf_'+str(i))
        if self.refr:
            self.refresh('stateChanged')

    def numContChanged(self):
        k = int(self.cont_num.text())
        sign = 1 if k > self.parent.fit.cont_num else -1
        if k != self.parent.fit.cont_num:
            rang = range(self.parent.fit.cont_num, k) if sign == 1 else range(self.fit.cont_num-1, k-1, -1)
            for i in list(rang):
                if k > self.parent.fit.cont_num:
                    self.parent.fit.add('cont'+str(i))
                    self.addChild('cont', 'cont'+str(i))
                else:
                    self.parent.fit.remove('cont' + str(i))
                    getattr(self, 'cont').removeChild(getattr(self, 'cont' + str(i)))
            self.parent.fit.cont_num = k
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

    def contRange(self):
        self.parent.fit.cont_left = float(self.cont_left.text())
        self.parent.fit.cont_right = float(self.cont_right.text())

    def onChanged(self, s, attr):
        print('onChanged', s, attr, self.refr)
        if s in ['mu', 'me', 'dtoh', 'res'] or any([c in s for c in ['cont', 'cf']]):
            print(hasattr(self.parent.fit, s))
            if hasattr(self.parent.fit, s):
                setattr(getattr(self.parent.fit, s), attr, float(getattr(self, s + '_' + attr).text()))
                print(getattr(getattr(self.parent.fit, s), attr))

        if s == 'res':
            self.parent.s[self.parent.s.ind].resolution = self.parent.fit.res.val

        if self.refr:
            self.refresh(s + '_' + attr)

    def refresh(self, excl='', refresh=None):
        print('refresh:', self.refr)
        names = ['val', 'max', 'min', 'step']
        for s in ['mu', 'me', 'dtoh']:
            if hasattr(self.parent.fit, s) and hasattr(self, s + '_vary'):
                getattr(self, s + '_vary').setChecked(getattr(getattr(self.parent.fit, s), 'vary'))
                for attr in names:
                    if s + '_' + attr != excl:
                        getattr(self, s + '_' + attr).setText(getattr(self.parent.fit, s).str(attr))
                    if attr != 'val':
                        getattr(self, s+'_'+attr).setEnabled(getattr(getattr(self.parent.fit, s), 'vary'))
        self.updateCF(excl=excl)
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

        if s not in self.parent.atomic.keys() and 'H2j' in s:
            try:
                self.parent.atomic.readH2(j=int(s[3:]))
            except:
                pass

        if s in self.parent.atomic.keys() and self.tab.currentIndex() > -1:
            if self.parent.fit.sys[self.tab.currentIndex()].addSpecies(s):
                self.tab.currentWidget().addSpecies(s)
                self.tab.currentWidget().refresh()

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

class fitResultsWidget(QTextEdit):
    def __init__(self, parent):
        super(fitResultsWidget, self).__init__()
        self.parent = parent
        self.setStyleSheet(open('config/styles.ini').read())

        self.setGeometry(200, 200, 750, 900)
        self.setWindowTitle('Fit results')
        self.refresh()

    def refresh(self):
        s = ''
        for p in self.parent.fit.list():
            print(p, p.fitres())
            s += p.fitres() + '\n'
        self.setText(s)

    def keyPressEvent(self, event):
        super(QTextEdit, self).keyPressEvent(event)
        key = event.key()

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_T:
                if (QApplication.keyboardModifiers() == Qt.ControlModifier):
                    self.parent.fitResults.close()

    def closeEvent(self, event):
        self.parent.fitResults = None
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
        for i, w in enumerate([150, 30, 40, 70, 110, 90, 110, 20, 110, 70, 80, 50, 50, 50]):
            self.treeWidget.setColumnWidth(i, w)

        # ------------------ z --------------------------------
        attr = ['val', 'min', 'max', 'step']
        sign = ['val: ', 'range: ', '....', 'step: ']
        self.z =  self.addParent(self.treeWidget, 'redshift', checkable=True, expanded=True)
        cons_vary = all([hasattr(self.fit.sys[self.ind], attr) for attr in ['turb', 'kin']])
        Ncons_vary = all([hasattr(self.fit.sys[self.ind], attr) for attr in ['Ntot', 'logT', 'logn']])
        self.cons = self.addParent(self.treeWidget, 'b tied', checkable=True, expanded=cons_vary)
        self.Ncons = self.addParent(self.treeWidget, 'N tied', checkable=True, expanded=Ncons_vary)
        for s, name in zip(['z', 'turb', 'kin', 'Ntot', 'logn', 'logT'], ['z','b_turb', 'Tkin, K', 'N_tot', 'logn', 'logT']):
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
                setattr(self, s + '_' + attr[k], FLineEdit(self.parent, getattr(getattr(self.fit.sys[self.ind], s), attr[k]), var=var))
                getattr(self, s + '_' + attr[k]).textChanged[str].connect(partial(self.onChanged, s, attr[k], species=None))
                self.treeWidget.setItemWidget(item, 2 * k + 4, getattr(self, s + '_' + attr[k]))
            setattr(self, s + '_vary', QCheckBox())
            getattr(self, s + '_vary').setChecked(getattr(getattr(self.fit.sys[self.ind], s), 'vary'))
            getattr(self, s + '_vary').stateChanged.connect(self.varyChanged)
            if (s in ['turb', 'kin'] and not cons_vary) or (s in ['Ntot', 'logn', 'logT'] and not Ncons_vary):
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
                self.vrange.returnPressed.connect(self.zrange)
                self.treeWidget.setItemWidget(item, 6, self.vrange)

                item.setText(9, 'v step: ')
                item.setTextAlignment(9, Qt.AlignRight)
                self.vstep = FLineEdit(self.parent)
                self.vstep.returnPressed.connect(self.zstep)
                self.treeWidget.setItemWidget(item, 10, self.vstep)

        for k in self.fit.sys[self.ind].sp.keys():
            self.addSpecies(k)

        self.treeWidget.itemExpanded.connect(self.stateChanged)
        self.treeWidget.itemCollapsed.connect(self.stateChanged)
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
            if hasattr(self.parent.parent.fit, 'me'):
                combo.addItems(['me'])
            if self.Ncons.isExpanded():
                combo.addItems(['Ntot'])
            try:
                s = getattr(getattr(self.fit.sys[self.ind].sp[species], 'N'), 'addinfo').strip()
                if s in ['Ntot', 'me']:
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
        self.parent.parent.chooseFit.update()
        self.updateTieds()
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
            self.fit.sys[self.ind].z.step = (abs(float(self.vstep.text()))) / 299792.458
        except:
            pass
        self.refresh()

    def varyChanged(self):
        for s in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT']:
            if hasattr(self.fit.sys[self.ind], s):
                setattr(getattr(self.fit.sys[self.ind], s), 'vary', getattr(self, s + '_vary').isChecked())
            #print('state:', getattr(getattr(self.fit.sys[self.ind], s), 'vary'))

        if len(self.species) > 0:
            for k, v in self.species.items():
                for s in ['b', 'N']:
                    setattr(getattr(self.fit.sys[self.ind].sp[k], s), 'vary', getattr(self, k + '_' + s + '_vary').isChecked())

        self.refresh()

    def stateChanged(self):
        for s in ['turb', 'kin']:
            if self.cons.isExpanded():
                self.fit.sys[self.ind].add(s)
            else:
                self.fit.sys[self.ind].remove(s)
        for s in ['Ntot', 'logn', 'logT']:
            if self.Ncons.isExpanded():
                self.fit.sys[self.ind].add(s)
            else:
                self.fit.sys[self.ind].remove(s)
        if self.Ncons.isExpanded():
            self.fit.sys[self.ind].pyratio(init=True)
        else:
            self.fit.sys[self.ind].pr = None
        if not self.z.isExpanded():
            self.z.setExpanded(True)
        self.updateTieds()
        self.refresh()

    def onChanged(self, s, attr, species=None):
        if s in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT']:
            setattr(getattr(self.fit.sys[self.ind], s), attr, float(getattr(self, s + '_' + attr).text()))
        if s in ['b', 'N']:
            setattr(getattr(self.fit.sys[self.ind].sp[species], s), attr, float(getattr(self, species + '_' + s + '_' + attr).text()))
        excl = species + '_' + s + '_' + attr if species is not None else s + '_' + attr
        self.refresh(excl)

    def setbTied(self, species):
        combo = getattr(self, species + '_btied')
        sp = combo.currentText()
        if 'tie' in sp:
            sp = ''

        setattr(getattr(self.fit.sys[self.ind].sp[species], 'b'), 'addinfo', sp)
        getattr(self.fit.sys[self.ind].sp[species], 'b').check()
        self.refresh()

    def setNTied(self, species):
        combo = getattr(self, species + '_Ntied')
        sp = combo.currentText()
        if 'tie' in sp:
            sp = ''

        setattr(getattr(self.fit.sys[self.ind].sp[species], 'N'), 'addinfo', sp)
        getattr(self.fit.sys[self.ind].sp[species], 'N').check()
        self.refresh()

    def setMe(self, species):
        info = 'me' if getattr(self, species + '_me').isChecked() else ''
        setattr(getattr(self.fit.sys[self.ind].sp[species], 'N'), 'addinfo', info)

        self.refresh()

    def refresh(self, excl=''):
        if not self.init:
            self.fit.update()

            names = ['val', 'max', 'min', 'step']
            for s in ['z', 'turb', 'kin', 'Ntot', 'logn', 'logT']:
                if hasattr(self.fit.sys[self.ind], s):
                    getattr(self, s+'_vary').setChecked(getattr(getattr(self.fit.sys[self.ind], s), 'vary'))
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

            if self.parent.parent.chooseFit is not None:
                self.parent.parent.chooseFit.update()

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
        del self.parent.chooseFit
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
