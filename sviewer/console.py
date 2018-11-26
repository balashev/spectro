import sys
import copy
from mendeleev import element
import numpy as np
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (QFont, QTextCursor)
from PyQt5.QtWidgets import (QTextEdit)
import pyqtgraph as pg
from statsmodels.stats.weightstats import DescrStatsW

from ..a_unc import a
from ..atomic import metallicity, depletion, abundance
from .utils import Timer, roman


class Console(QTextEdit):
    def __init__(self, parent):
        self.parent = parent
        QTextEdit.__init__(self)
        font = QFont("courier new", 8)
        font.setStyleHint(QFont.TypeWriter)
        self.document().setDefaultFont(font)
        self.displayPrompt()
        self.initStyle()
        self.addhistory()

    def initStyle(self):
        self.setStyleSheet("""

        """)

    def displayPrompt(self):
        """
        Displays the command prompt at the bottom of the buffer,
        and moves cursor there.
        """

        #prompt = "<span style=\" font-size:8pt; font-weight:600; color:#3399bb;\" > >>> </span>"

        # get the current format
        format = self.textCursor().charFormat()
        # modify it
        format.setBackground(Qt.red)
        format.setForeground(Qt.blue)
        # apply it
        self.textCursor().setCharFormat(format)
        prompt = '>>> '
        self.append(prompt)

        self.moveCursor(QTextCursor.End)
        self.promptBlockNumber = self.textCursor().blockNumber()
        self.promptColumnNumber = self.textCursor().columnNumber()
        self.promptPosition = self.textCursor().position()

    def clear(self):
        cursor = self.textCursor()
        cursor.select(QTextCursor.LineUnderCursor)
        cursor.removeSelectedText()
        cursor.movePosition(QTextCursor.Up, QTextCursor.MoveAnchor, 1)
        cursor.movePosition(QTextCursor.EndOfLine)
        cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 1)
        cursor.removeSelectedText()
        self.setTextCursor(cursor)
        self.displayPrompt()

    def set(self, s):
        self.append(str(s))
        self.moveCursor(QTextCursor.End)
        self.displayPrompt()

    def addhistory(self, command=None):
        f = open('config/console.list', 'r')
        self.history = f.read().splitlines()
        f.close()
        if command is not None:
            if len(self.history) > 20:
                del self.history[0]
            self.history.append(command)
            f = open('config/console.list', 'w')
            for h in self.history:
                f.write(h + '\n')
            f.close()
        self.hist_pos = len(self.history) - 1

    def getCurrentCommand(self):
        """
        Gets the current command written on the command prompt.
        """
        # Select the command.
        block = self.document().findBlockByNumber(self.promptBlockNumber)
        print(block.text()[self.promptColumnNumber:])
        return block.text()[self.promptColumnNumber:]

    def isInEditionZone(self):
        return self.promptBlockNumber == self.textCursor().blockNumber() and \
               self.promptColumnNumber <= self.textCursor().columnNumber()

    def keyPressEvent(self, event):
        # Run the command if enter was pressed.
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            event.accept()
            command = self.getCurrentCommand()
            self.addhistory(command)

            self.hist_pos = len(self.history)
            if len(command) > 0:
                result = ""
                try:
                    result = self.exec_command(command)
                    if result == None:
                        result = eval(str(command), globals(), locals())
                    else:
                        pass
                except Exception as error:
                    result = str(error)
                self.append(str(result))
                self.moveCursor(QTextCursor.End)
            self.displayPrompt()
            return

        if event.key() == Qt.Key_Backspace:
            event.accept()
            if not self.isInEditionZone():
                self.moveCursor(QTextCursor.End)
                # First backspace just moves the cursor.
                return
            # If there is something to be deleted, delete it.
            if self.promptColumnNumber < self.textCursor().columnNumber():
                self.textCursor().deletePreviousChar()
            return

        if event.key() == Qt.Key_Left:
            event.accept()
            if not self.isInEditionZone():
                self.moveCursor(QTextCursor.End)
                # First backspace just moves the cursor.
                return
            if self.promptColumnNumber < self.textCursor().columnNumber():
                mode = QTextCursor.MoveAnchor
                if event.modifiers() & Qt.ShiftModifier:
                    mode = QTextCursor.KeepAnchor
                self.moveCursor(QTextCursor.Left, mode)
            return

        if event.key() == Qt.Key_Right:
            event.accept()
            if not self.isInEditionZone():
                self.moveCursor(QTextCursor.End)
                # First backspace just moves the cursor.
                return
            mode = QTextCursor.MoveAnchor
            if event.modifiers() & Qt.ShiftModifier:
                mode = QTextCursor.KeepAnchor
            self.moveCursor(QTextCursor.Right, mode)
            return

        if event.key() == Qt.Key_Up:
            event.accept()
            self.hist_pos -= 1
            self.hist_pos = max(0, self.hist_pos)
            self.clear()
            self.textCursor().insertText(self.history[self.hist_pos])

        if event.key() == Qt.Key_Down:
            event.accept()
            self.hist_pos += 1
            self.hist_pos = min(len(self.history)-1, self.hist_pos)
            self.clear()
            self.textCursor().insertText(self.history[self.hist_pos])

        if len(event.text()) > 0:
            event.accept()
            if not self.isInEditionZone():
                self.moveCursor(QTextCursor.End)
            self.textCursor().insertText(event.text())

    def exec_command(self, command):
        args = command.split()

        if args[0] == 'z':
            if len(args) == 2:
                self.parent.abs.redraw(float(args[1]))

        elif args[0] == 'load':
            if len(args) == 2:
                self.parent.openFile('data/templates/'+args[1]+'.spv')
                self.parent.fit.sys[0].z.val = self.parent.z_abs
                self.parent.fit.sys[0].zrange(1000.0)

        elif args[0] == 'save':
            if len(args) == 2:
                self.parent.saveFile('data/templates/'+args[1]+'.spv')

        elif args[0] == 'show':

            lines, color = [], (23, 190, 207)
            if args[1] == 'all':
                for s in self.parent.atomic.keys():
                    if '*' not in s:
                        lines += self.parent.atomic.list(s)
            elif args[1] == 'full':
                lines = self.parent.atomic.list(s)
                #for s in self.parent.atomic.keys():
                #    lines += self.parent.atomic[s].lines

            elif args[1] == 'H2' or 'H2j' in args[1]:
                color = (229, 43, 80)
                energy = None
                if 'H2j' in args[1]:
                    j = [int(args[1][3:])]
                else:
                    if len(args) == 2:
                        j = np.arange(3)
                    elif len(args) == 3:
                        energy = float(args[2]) if float(args[2]) > 100 else None
                        j = [int(args[2])]
                    elif len(args) > 3:
                        print(args[2:])
                        j = [int(j) for j in args[2:]]
                print(j)
                #self.parent.atomic.readH2(j=j, energy=energy)
                #for k in self.parent.atomic.keys():
                for j in j:
                    lines += self.parent.atomic.list('H2j{:d}'.format(j))

            elif args[1] == 'HD' or 'HDj' in args[1]:
                if 'HDj' in args[1]:
                    j = [int(args[1][3:])]
                else:
                    if len(args) == 2:
                        j = [0]
                    elif len(args) > 2:
                        j = [int(j) for j in args[2:]]

                #self.parent.atomic.readHD()
                for j in j:
                    lines += self.parent.atomic.list('HDj{:d}'.format(j))
                color = (254, 255, 63)

            elif args[1] == 'CO' or 'COj' in args[1]:
                if 'COj' in args[1]:
                    j = [int(args[1][3:])]
                else:
                    if len(args) == 2:
                        j = [0,1,2,3]
                    elif len(args) > 2:
                        j = [int(j) for j in args[2:]]

                #self.parent.atomic.readHD()
                for j in j:
                    lines += self.parent.atomic.list('COj{:d}'.format(j))

            elif args[1] == 'HF':
                for k in self.parent.atomic.keys():
                    if 'HF' in k:
                        lines += self.parent.atomic.list('HF')
                color = (255, 69, 69)
            else:
                if args[1] in self.parent.atomic.keys():
                    lines += self.parent.atomic.list(args[1])

            if args[1] != 'H2':
                try:
                    f = float(args[2])
                except:
                    f = 0
                for l in reversed(lines):
                    if l.f() > f:
                        del l
            self.parent.abs.add(lines, color=color)

            return ''

        elif args[0] == 'hide':

            if args[1] == 'all':
                #self.parent.abs.remove(self.parent.abs.lines)
                for l in self.parent.abs.lines:
                    self.parent.vb.removeItem(l)
                self.parent.abs.lines = []
                self.parent.lines = []

            if args[1] in self.parent.atomic.keys():
                self.parent.abs.remove(el=args[1])

            if args[1] in ['H2', 'HD', 'CO']:
                if len(args) == 2:
                    self.parent.abs.remove(el=args[1])
                elif len(args) == 3:
                    self.parent.abs.remove(el=args[1]+'j{:1d}'.format(int(args[2])))

            return ''

        elif args[0] in self.parent.atomic.keys():

            print(args[0])
            try:
                el = element(roman.ion(args[0])[0])
                s = '{:.3f} eV \n'.format(el.ionenergies[roman.int(roman.ion(args[0])[1])])
            except:
                s = ''
            try:
                f = float(args[1])
            except:
                f = 0
            lines, lf = [], []

            for l in self.parent.atomic.list(args[0]):
                if l.f() > f:
                    lines.append(l)
                    lf.append(l.f())
            for i in np.argsort(lf):
                s += str(lines[i]) + ',  l={:.5f}, f={:.2e},  g={:.1e} \n'.format(lines[i].l(), lines[i].f(), lines[i].g())
            return s

        elif any(s in args[0] for s in ['HD', 'H2']):
            self.exec_command('show '+ args[0])

        elif args[0] == 'fit':
            if len(args) == 1:
                #self.parent.setFit()
                for par in self.parent.fit.list():
                    par.fit = par.vary
                self.parent.fitLM()
            else:
                if args[1] == 'grid':
                    num = int(args[2]) if len(args) == 3 else 20
                    self.parent.fitGrid(num=num)
            return 'fit is started'

        elif args[0] == 'fitcomp':
            print(self.parent.comp)
            #self.parent.setFit(self.parent.comp)
            self.parent.setFit(comp=self.parent.comp)
            self.parent.fitLM(self.parent.comp)
            return 'fit of component is started'

        elif args[0] == 'fitmcmc':
            self.parent.fitMCMC()
            return 'MCMC fit is started'

        elif args[0] == 'fitcont':
            if len(args) > 1 and args[1] in ['cheb', 'GP']:
                self.parent.fitCont(typ=args[1])
            else:
                self.parent.fitCont()
            return 'estimate the continuum'

        elif args[0] in ['y', 'x']:
            if args[0] == 'y':
                self.parent.plot.vb.setYRange(float(args[1]), float(args[2]))
            elif args[0] == 'x':
                self.parent.plot.vb.setXRange(float(args[1]), float(args[2]))

        elif args[0] == 'logN':
            print(args[1])
            if len(args) == 2:
                c = np.arange(len(self.parent.fit.sys))
            elif len(args) > 2:
                c = [int(i) for i in args[2:]]
            N = a(0,0,0, 'd')
            for i in c:
                if args[1] in self.parent.fit.sys[i].sp:
                    self.parent.fit.sys[i].sp[args[1]].N.unc.val = self.parent.fit.sys[i].sp[args[1]].N.val
                    N += self.parent.fit.sys[i].sp[args[1]].N.unc
                if args[1] in ['H2', 'HD', 'CO']:
                    for s, v in self.parent.fit.sys[i].sp.items():
                        if args[1] in s:
                            v.N.unc.val = v.N.val
                            print(v.N.unc)
                            N += v.N.unc
                            print(N)
            N.log()
            print(N)
            return N

        elif args[0] == 'Me':
            c = np.arange(len(self.parent.fit.sys))
            N = a(0,0,0, 'd')
            for i in c:
                if args[1] in self.parent.fit.sys[i].sp:
                    self.parent.fit.sys[i].sp[args[1]].N.unc.val = self.parent.fit.sys[i].sp[args[1]].N.val
                    N += self.parent.fit.sys[i].sp[args[1]].N.unc
            N.log()
            print(args[1], N.log())
            return metallicity(args[1], N.log(), a(float(args[2]), 0, 0))

        elif args[0] == 'depletion':
            if len(args) == 3:
                c = np.arange(len(self.parent.fit.sys))
            elif len(args) > 3:
                c = [int(i) for i in args[3:]]
            N = [a(0, 0, 0, 'd'), a(0, 0, 0, 'd')]
            for i in c:
                for k in range(2):
                    if args[k+1] in self.parent.fit.sys[i].sp:
                        self.parent.fit.sys[i].sp[args[k+1]].N.unc.val = self.parent.fit.sys[i].sp[args[k+1]].N.val
                        N[k] += self.parent.fit.sys[i].sp[args[k+1]].N.unc
            print(N)
            return depletion(args[1], N[0], N[1], ref=args[2])

        elif args[0] == 'abundance':
            N, me = float(args[2]), float(args[3])
            N, me = np.max([N, me]), np.min([N, me])
            N, me = np.max([N, me]), np.min([N, me])
            print(N, me)
            return abundance(args[1], N, me)

        elif args[0] == 'rescale':
            if len(args) == 3 or len(args) == 2:
                mask = np.logical_and(self.parent.s[self.parent.s.ind].spec.raw.x > self.parent.plot.vb.viewRange()[0][0],
                                      self.parent.s[self.parent.s.ind].spec.raw.x < self.parent.plot.vb.viewRange()[0][1])
                if args[1] == 'y' or len(args) == 2:
                    self.parent.s[self.parent.s.ind].spec.raw.y[mask] *= float(args[-1])
                    self.parent.s.redraw()
                    #if self.parent.s[self.parent.s.ind].spline.n > 0:
                    #    self.parent.s[self.parent.s.ind].spline.y *= float(args[-1])
                    #    self.parent.s[self.parent.s.ind].calc_spline()
                if args[1] == 'err' or len(args) == 2:
                    self.parent.s[self.parent.s.ind].spec.raw.err[mask] *= float(args[-1])
                    self.parent.s.redraw()
                if args[1] == 'spline':
                    if self.parent.s[self.parent.s.ind].spline.n > 0:
                        self.parent.s[self.parent.s.ind].spline.y *= float(args[-1])
                        self.parent.s[self.parent.s.ind].calc_spline()

        elif args[0] == 'shift':
            if len(args) == 2:
                self.parent.s[self.parent.s.ind].spec.raw.x += float(args[1])
                self.parent.s[self.parent.s.ind].spec.norm.x += float(args[1])
                self.parent.s.redraw()
            if len(args) == 3 and args[1] == 'v':
                self.parent.s[self.parent.s.ind].spec.raw.x *= (1 + float(args[2])/299792.458)
                self.parent.s[self.parent.s.ind].spec.norm.x *= (1 + float(args[2])/299792.458)
                self.parent.s.redraw()

        elif args[0] == 'set':
            if len(args) == 3:
                mask = np.logical_and(self.parent.s[self.parent.s.ind].spec.raw.x > self.parent.plot.vb.viewRange()[0][0],
                                      self.parent.s[self.parent.s.ind].spec.raw.x < self.parent.plot.vb.viewRange()[0][1])
                if args[1] == 'err':
                    if len(self.parent.s[self.parent.s.ind].spec.raw.err) == 0:
                        self.parent.s[self.parent.s.ind].spec.raw.err = np.ones_like(self.parent.s[self.parent.s.ind].spec.raw.x) * float(args[2])
                    self.parent.s[self.parent.s.ind].spec.raw.err[mask] = float(args[2])
                    self.parent.s.redraw()

                if args[1] == 'cont':
                    self.parent.s[self.parent.s.ind].set_spline([self.parent.s[self.parent.s.ind].spec.raw.x[mask][0], self.parent.s[self.parent.s.ind].spec.raw.x[mask][-1]], [float(args[2]), float(args[2])])
                    #self.parent.s.redraw()

        elif args[0] == 'divide':
            if len(args) == 3:
                i1, i2 = int(args[1]), int(args[2])
                x = self.parent.s[i1].spec.x()
                y = self.parent.s[i1].spec.y() / self.parent.s[i2].spec.inter(x)
                self.parent.importSpectrum('{:}_divided_by_{:}'.format(i1, i2), spec=[x, y], append=True)
                #self.parent.s[-1].normalize()

        elif args[0] == 'subtract':
            if len(args) == 3:
                i1, i2 = int(args[1]), int(args[2])
                x = self.parent.s[i1].spec.x()
                y = self.parent.s[i1].spec.y() - self.parent.s[i2].spec.inter(x)
                self.parent.importSpectrum('{:}_subtracted_from_{:}'.format(i2, i1), spec=[x, y], append=True)
                #self.parent.s[-1].normalize()

        elif args[0] == 'coscale':
            if len(args) == 2 and args[1] == 'cont':
                print(self.parent.s[self.parent.s.ind].cont.y, len(self.parent.s[self.parent.s.ind].spec.raw.y),
                      len(self.parent.s[self.parent.s.ind].cont.y))

                self.parent.s[self.parent.s.ind].spec.raw.y *= self.parent.s[self.parent.s.ind].cont.y
                self.parent.s[self.parent.s.ind].spec.raw.err *= self.parent.s[self.parent.s.ind].cont.y

            if len(args) == 3:
                self.parent.coscaleExposures(int(args[1]), int(args[2]))

        elif args[0] == 'crosscorr':
            if len(args) > 2:
                if len(args) == 3:
                    self.parent.crosscorrExposures(int(args[1]), int(args[2]))
                else:
                    self.parent.crosscorrExposures(int(args[1]), int(args[2]), dv=float(args[3]))

        elif args[0] == 'apply':
            if len(args) > 1:
                if args[1] == 'vac':
                    self.parent.s[self.parent.s.ind].airvac()
                if args[1] == 'helio':
                    try:
                        float(args[2])
                        self.parent.s[self.parent.s.ind].apply_shift(float(args[2]))
                    except:
                        pass


        elif args[0] == 'lines':

            if args[1] == 'save':
                with open('data/lines/saved.dat', 'w') as f:
                    for line in self.parent.lines:
                        f.write(line + '\n')

            elif args[1] == 'load':
                lines, addlines = [], []
                if len(args) == 2 or (len(args) == 3 and args[-1] in ['0', '1']):
                    with open('data/lines/saved.dat', 'r') as f:
                        lines = f.readlines()
                elif len(args) == 3 or (len(args) == 4 and args[-1] in ['0', '1']):
                    with open('data/lines/' + args[2] + '.dat', 'r') as f:
                        lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    for l in self.parent.atomic[line.split()[0]].lines:
                        if str(l) == line:
                            print(l)
                            addlines += [l]
                    if 'H2j' in line:
                        #self.parent.atomic.readH2Abgrall(j=int(line.split()[0][-1]))
                        for l in self.parent.atomic[line.split()[0]].lines:
                            if str(l) == line:
                                print(l)
                                addlines += [l]
                if len(args) > 2 and args[-1] == '1':
                    self.parent.lines = [str(l) for l in addlines]
                self.parent.abs.add(addlines, color=(23, 190, 97))

            elif args[1] == 'all':
                self.parent.lines = [str(l.line) for l in self.parent.abs.lines]

            elif args[1] == 'none':
                self.parent.lines = []

            self.parent.abs.redraw()

        elif args[0] == 'savefit':
            try:
                self.parent.vb.removeItem(self.savefit)
            finally:
                fit = self.parent.s[self.parent.s.ind].fit
                self.savefit = pg.PlotCurveItem(x=fit.x(), y=fit.y(), pen=pg.mkPen(0,208,96, width=2))
                self.parent.vb.addItem(self.savefit)

        elif args[0] == 'hidefit':
            self.parent.vb.removeItem(self.savefit)

        elif args[0] == 'ston':
            s = 'region        ston'
            for r in self.parent.plot.regions:
                s += str(r)
                xmin, xmax = float(str(r).split('..')[0]), float(str(r).split('..')[1])
                spec = self.parent.s[self.parent.s.ind].spec
                if self.parent.normview:
                    mask = (xmin < spec.x()) * (spec.x() < xmax)
                    if np.sum(mask) > 0:
                        s += '  ' + str(np.mean(spec.y()[mask]/spec.err()[mask])) + '\n'
                    else:
                        s += ' there is no continuum \n'
            return s

        elif args[0] == 'stats':
            st = 'region             snr       err_disp  snr/disp\n'
            for r in self.parent.plot.regions:
                st += str(r)
                xmin, xmax = float(str(r).split('..')[0]), float(str(r).split('..')[1])
                s = self.parent.s[self.parent.s.ind]
                if self.parent.normview:
                    mask = (xmin < s.spec.x()) * (s.spec.x() < xmax)
                    if np.sum(mask) > 0:
                        ston, disp = np.mean(s.spec.y()[mask]/s.spec.err()[mask]), np.std(s.spec.y()[mask]-1) / np.mean(s.spec.err()[mask])
                        st += '   {0:6.2f}    {1:6.3f}    {2:6.2f}\n'.format(ston, disp, ston/disp)
                    else:
                        st += ' there is no continuum \n'
                else:
                    mask = (xmin < s.spec.x()[s.cont_mask]) * (s.spec.x()[s.cont_mask] < xmax)
                    if np.sum(mask) > 0:
                        ston, disp = np.mean(s.spec.y()[s.cont_mask][mask] / s.spec.err()[s.cont_mask][mask]), np.std(s.spec.y()[s.cont_mask][mask] - s.cont.y[mask]) / np.mean(s.spec.err()[s.cont_mask][mask])
                        st += '   {0:6.2f}    {1:6.3f}    {2:6.2f}\n'.format(ston, disp, ston/disp)
                    else:
                        st += ' there is no continuum \n'

            return st

        elif args[0] == 'level':
            st = 'region               mean        weigthed    weighted err\n'
            for r in self.parent.plot.regions:
                st += str(r)
                xmin, xmax = float(str(r).split('..')[0]), float(str(r).split('..')[1])
                spec = self.parent.s[self.parent.s.ind].spec
                mask = (xmin < spec.x()) * (spec.x() < xmax)
                waver = DescrStatsW(spec.y()[mask], weights=1.0/spec.err()[mask], ddof=1)
                if np.sum(mask) > 0:
                    st += ' {0:10.3f}  {1:10.3f}  {2:10.3f}\n'.format(np.average(spec.y()[mask]), waver.mean, waver.std)
            return st

        elif args[0] == 'ew':
            s = ''
            for r in self.parent.plot.regions:
                s += str(r)
                xmin, xmax = float(str(r).split('..')[0]), float(str(r).split('..')[1])
                fit = self.parent.s[self.parent.s.ind].fit
                if self.parent.normview:
                    mask = (xmin < fit.x()) * (fit.x() < xmax)
                    if np.sum(mask) > 0:
                        s += '  ' + str(np.trapz(1 - fit.y()[mask], x=fit.x()[mask])) + '\n'
            return s

        elif args[0] == '2d':
            s = self.parent.s[self.parent.s.ind].spec2d

            if args[1] == 'scale':
                s.set(x=s.raw.x, y=s.raw.y, z=s.raw.z*float(args[2]))

            if args[1] == 'levels':
                if len(args) == 4:
                    s.raw.setLevels(float(args[2]), float(args[3]))
                elif len(args) == 5:
                    if args[2] == 'err':
                        s.raw.setLevels(float(args[3]), float(args[4]), attr='err')
                    if args[2] == 'sky':
                        s.sky.setLevels(float(args[3]), float(args[4]))


            self.parent.s.redraw()

        elif args[0] == 'Tkin':
            levels = [0, 1]
            if len(args) > 1:
                levels = [int(a) for a in args[1:]]
            self.parent.H2ExcitationTemp(levels=levels)

        elif args[0].isdigit():
            self.parent.s.setSpec(int(args[0]))

        else:
            return None
