#!/usr/bin/env python

from adjustText import adjust_text
from bisect import bisect_left
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from mendeleev import element
import numpy as np
from pathlib import Path
from scipy import interpolate
import sys
sys.path.append('D:/science/python')
import colors as col
from .sviewer.utils import roman
from .atomic import atomicData

class rectangle():
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.right = self.left + width
        self.bottom = self.top - height
        self.data = [left, top-height, width, height]
    
    def __str__(self):
        return str(list([self.left, self.bottom, self.right-self.left, self.top-self.bottom]))
        
    def __repr__(self):
        return str(list([self.left, self.bottom, self.right-self.left, self.top-self.bottom]))
        
class rect_param():
    """
    class for setting position and size of axis object of the regular grid plot:
    parameters:
        - n_rows      : number of rows
        - n_cols      : number of columns 
        - row_offset  : offset between rows
        - col_offset  : offset between columns
        - width       : width of the grid
        - height      : height of the grid
        - v_indend    : 
        - order       : order of the panel, 'v' for vertical and 'h' for horizontal
    """
    def __init__(self, n_rows=1, n_cols=1, row_offset=0, col_offset=0, 
                 width=1.0, height=1.0,  order='v', v_indent=0.05, h_indent=0.03):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.row_offset = row_offset
        self.col_offset = col_offset
        self.width = width
        self.height = height
        self.v_indent = v_indent
        self.h_indent = h_indent
        self.order = order

class plot_spec(list):
    """
    Class to plot the figures with multiply panels each of them consists of line profiles:

    - Each element of the class is panel with line profile.

    Methods:
        - readlist    :  read the path to lines from the file with the list
        -
    """
    def __init__(self, arg, vel_scale=False, font=14, font_labels=16, fit=True, figure=None, gray_out=False, show_comps=True, show_err=True, error_cap=5):
        self.vel_scale = vel_scale
        self.name_pos = [0.02, 0.19]
        self.add_ioniz = False
        self.font = font
        self.font_labels = font_labels
        self.fit = fit
        self.figure = figure
        self.color_total = col.tableau10[3]
        self.show_comps = show_comps
        self.show_err = show_err
        self.error_cap = error_cap
        self.gray_out = gray_out
        self.order = 'v'

        if isinstance(arg, int):
            for i in range(arg):
                self.append(plotline(self, fit=fit, figure=self.figure))
        if isinstance(arg, str):
            self.filename = arg
            self.readlist()
        self.comps = []
        self.comp_names = None

    def readlist(self):
        with open(self.filename,'r') as f:
            for line in f:
                print(line)
                if line.strip() and '#' not in line:
                    self.append(plotline(self, line.split()[0], bool(int(line.split()[1])), figure=self.figure))
            #self = [line for line in f if line.strip() and '#' not in line]
                    
    def specify_rects(self, rect_pars):
        """
        function to form rects from the list of regular grid plot objects <rect_param>:
        parameters:
            - rect_pars   : list of rect_param object. can be just one object.
            
            example:
                rects = [rect_param(n_rows=2, n_cols=1, order='v', height=0.2, row_offset=0.02),
                         rect_param(n_rows=10, n_cols=2, order='v', height=0.75)
                         ]
                ps.specify_rects(rects)
        """
        rects = []
        if not isinstance(rect_pars, list):
            rect_pars = [rect_pars]
        left = rect_pars[0].v_indent 
        top = 1 
        for r in rect_pars:
            panel_h = (r.height - r.h_indent - r.row_offset * (r.n_rows - 1)) / r.n_rows
            panel_w = (r.width - r.v_indent - r.col_offset * (r.n_cols - 1)) / r.n_cols
            for i in range(r.n_rows * r.n_cols):
                if r.order is 'v':
                    col = i // r.n_rows
                    row = i % r.n_rows
                if r.order is 'h':
                    col = i % r.n_cols
                    row = i // r.n_cols
                rects.append(rectangle(left + (panel_w + r.col_offset) * col, top - (panel_h + r.row_offset) * row, panel_w, panel_h))
                #[left+(panel_w+r.col_offset)*col, top-panel_h*(row+1)-r.row_offset*row, panel_w, panel_h])
            top -= r.height
            top -= r.h_indent
            #if r.row_offset == 0:
            #    top -= r.h_indent

        self.rect = rect_pars[0]
        for i, line in enumerate(self):
            line.rect = rects[i]
   
    def specify_comps(self, *args):
        self.comps = np.array(args)
                
    def specify_styles(self, color_total=None, color=None, ls=None, lw=None, lw_total=2, lw_spec=1.0, ls_total='solid',
                       disp_alpha=0.7, res_style='scatter'):

        if color_total is not None:
            if np.max(list(color_total)) > 1:
                color_total = tuple(c / 255 for c in color_total)
            self.color_total = color_total

        num = len(self.comps+1)
        if color is None:
            cmap = plt.get_cmap('rainbow')
            color = cmap(np.linspace(0, 0.85, num))
            if 1:
                for i in range(num):
                    color[i] = [max(0, c-0.20) for c in color[i][:3]]+[1.]
            if 0:
                color_add = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:cyan', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:brown', 'tab:red', 'dodgerblue']
                for i in range(min(len(color_add), num)):
                    color[i] = color_add[i]
        else:
            for i, c in enumerate(color):
                if np.max(list(c)) > 1:
                    color[i] = tuple(ci / 255 for ci in c)
        self.color = color[:]

        #d = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdot': '-:'}
        if ls is None:
            ls = ['-'] * num
        if isinstance(ls, (str)):
            ls = [ls] * num
        self.ls = ls[:]

        if lw is None:
            lw = [0.5] * num
        if isinstance(lw, (float, int)):
            lw = [lw] * num
        self.lw = lw[:]

        self.lw_total = lw_total
        self.lw_spec = lw_spec
        self.ls_total = ls_total
        self.disp_alpha = disp_alpha
        self.res_style = res_style

    def set_limits(self, x_min, x_max, y_min, y_max):
        for p in self:
            p.x_min, p.x_max, p.y_min, p.y_max = x_min, x_max, y_min, y_max

    def set_ticklabels(self, xlabels=None, ylabels=None, xlabel=r'v [km s$^{-1}$]', ylabel='Normalized flux'):
        """
        set axis labels
        parmeters:
            - xlabels      : manualy specify which panel show x_label
                             type: list
            - ylabels      : manualy specify which panel show y_label
                             type: list
        """
        for p in self:
            col = [p1 for p1 in self if abs(p1.rect.left - p.rect.left)<0.01]
            if all([abs(p1.rect.top-p.rect.bottom)>0.001 for p1 in col]):
                p.xticklabels = 1

        for p in self:
            if all([abs(p1.rect.right - p.rect.left) > 0.001 for p1 in self]):
                p.yticklabels = 1

        for p in self:
            p.xlabel = None
            p.ylabel = None

        if xlabels is not None:
            for i, p in enumerate(self):
                if i in xlabels:
                    p.xlabel = xlabel
        else:
            print('ticks', self.rect.order)
            if self.rect.order == 'h':
                inds = list(range(len(self)-self.rect.n_cols, len(self)))
            if self.rect.order == 'v':
                inds = np.append(self.rect.n_rows * np.arange(1, len(self) // self.rect.n_rows+1) - 1, len(self)-1)
            for i, p in enumerate(self):
                if i in inds:
                    p.xlabel = xlabel

        if ylabels is not None:
            for i, p in enumerate(self):
                if i in ylabels:
                    p.ylabel = ylabel
        else:
            for i, p in enumerate(self):
                if self.rect.order == 'h':
                    k = (len(self) - 1)  // self.rect.n_cols
                    if k == 0:
                        inds = [0]
                    elif k == 1:
                        inds = [0, self.rect.n_cols]
                    else:
                        inds = [(k // 2) * self.rect.n_cols]
                if self.rect.order == 'v':
                    k = min(self.rect.n_rows, len(self))
                    if k == 0:
                        inds = [0]
                    elif k == 1:
                        inds = [0, 1]
                    else:
                        inds = [(k // 2)]
            for i, p in enumerate(self):
                if i in inds:
                    p.ylabel = ylabel

    def set_ticks(self, x_tick=100, x_num=10, y_tick=1, y_num=10):
        for p in self:
            p.x_minorLocator = AutoMinorLocator(x_num)
            p.x_locator = MultipleLocator(x_tick)
            p.y_minorLocator = AutoMinorLocator(y_num)
            p.y_locator = MultipleLocator(y_tick)

class data():
    def __init__(self):
        pass

    def mask(self, xmin, xmax):
        imin = max(0, bisect_left(self.x, xmin)-1)
        imax = min(len(self.x), bisect_left(self.x, xmax)+1)
        mask = np.zeros_like(self.x, dtype=bool)
        mask[imin:imax] = True
        #mask = np.logical_and(self.x > xmin, self.x < xmax)

        self.x = self.x[mask]
        self.y = self.y[mask]
        if hasattr(self, 'err'):
            self.err = self.err[mask]
        if hasattr(self, 'comp'):
            for i in range(len(self.comp)):
                self.comp[i] = self.comp[i][mask]
        return mask

class plotline():
    """
    Class which specify the panel with line profile
    """
    def __init__(self, parent, filename=None, fit=False, figure=None):
        self.parent = parent
        self.fig = figure
        if filename is not None: 
            self.filename = [filename]
            with open(filename, 'r') as f:
                self.wavelength = float(f.readline())
                name = f.readline()
                try:
                    self.el = element(name[:name.find('_')].strip())
                    io = name[(name.find('_')+1):][:name[(name.find('_')+1):].find('_')].strip()
                    self.ion_state = roman.int(roman.ion(io)[1])
                except:
                    pass
                self.name = name.replace('_','').replace('\n', '') 
                self.correctname()
            self.loaddata()
        else:
            self.filename = None
            self.name = ''
            self.wavelength = 0
        self.show_fit = fit
        self.add_errors = True
        self.show_err = self.parent.show_err
        self.gray_out = self.parent.gray_out
        self.add_cont = True
        self.rect = []
        self.cont_range = []
        self.cont = []
        self.yticklabels = None
        self.xticklabels = None
        self.ylabel = None
        self.xlabel = None
        self.label = None
        self.x_min, self.x_max, self.y_min, self.y_max = 0, 0, 0, 0
        self.x_minorLocator = AutoMinorLocator(10)
        self.x_locator = MultipleLocator(100)
        self.y_minorLocator = AutoMinorLocator(10)
        self.y_locator = MultipleLocator(1)
        self.x_formatter = None
        self.y_formatter = None
        self.add_residual = True
        self.font = self.parent.font
        self.font_labels = self.parent.font_labels
        self.vel_scale = self.parent.vel_scale
        self.name_pos = self.parent.name_pos[:]
        self.show_comps = self.parent.show_comps
        self.sig = 2

    def loaddata(self, d=None, f=None, fit_comp=None, fit_disp=None, fit_comp_disp=None, verbose=False):

        if self.filename is not None:
            print('filename:', self.filename)
            d = np.genfromtxt(self.filename[0], skip_header=2, unpack=True)
            if Path(self.filename[0].replace('.dat', '_fit.dat')).exists():
                f = np.genfromtxt(self.filename[0].replace('.dat', '_fit.dat'), skip_header=2, unpack=True)

        if d is not None:
            self.spec = data()
            self.spec.x = d[0, :]
            self.spec.y = d[1, :]
            self.spec.err = d[2, :]
            if len(d) > 3:
                self.points = d[3, :]

        if f is not None:
            self.fit = data()
            self.fit.x = np.insert(np.append(f[0, :], self.spec.x[-1]), 0, self.spec.x[0])
            self.fit.y = np.insert(np.append(f[1, :], 1), 0, 1)

        if fit_comp is not None:
            self.fit_comp = []
            for c in fit_comp:
                fit = data()
                fit.x, fit.y = c[0], c[1]
                self.fit_comp.append(fit)
            self.num_comp = len(self.fit_comp)

        if fit_disp is not None:
            self.fit_disp = [data(), data()]
            for i, d in enumerate(fit_disp):
                self.fit_disp[i].x = d[0]
                self.fit_disp[i].y = d[1]
        else:
            self.fit_disp = None

        if fit_comp_disp is not None:
            self.fit_comp_disp = ['']*len(fit_comp_disp)
            for k in range(len(fit_comp_disp)):
                self.fit_comp_disp[k] = [data(), data()]
                for i, d in enumerate(fit_comp_disp[k]):
                    self.fit_comp_disp[k][i].x = d[0]
                    self.fit_comp_disp[k][i].y = d[1]
        else:
            self.fit_comp_disp = None

        if f is None:
            self.fit = None
            self.show_fit = False
            self.num_comp = 0


        if verbose:
            print(self.spec.x, self.spec.y)

    def correctname(self):
        if self.name is not None:
            self.name = self.name[:self.name.find('.')]
            self.name = self.name.replace(' n=1', '')
            self.name = self.name.replace(' n=2 ', '*')
            self.name = self.name.replace(' n=3 ', '*')
            self.name = self.name.replace('1215', r' Ly-$\alpha$')
            self.name = self.name.replace('1025', r' Ly-$\beta$')
            self.name = self.name.replace('972', r' Ly-$\gamma$')
            self.name = self.name.replace('949', r' Ly-$\delta$')
            self.name = self.name.replace('937', r' Ly-5')
            self.name = self.name.replace('930', r' Ly-6')
            self.name = self.name.replace('925', r' Ly-7')
            self.name = self.name.replace('922', r' Ly-8')
            self.name = self.name.replace('920', r' Ly-9')
            self.name = self.name.replace('919', r' Ly-10')
            self.name = self.name.replace('917', r' Ly-11')
            self.name = self.name.replace('916', r' Ly-12')
            l = self.name.find('I')
            if l == -1:
                l = self.name.find('V')
            self.name = self.name[:l]+' '+self.name[l:]

    def plot_line(self):
        """
        Plot the line profile panel
        """
        if self.fig is not None:
            self.ax = self.fig.add_axes(self.rect.data)
        else:
            self.ax = plt.axes(self.rect.data)

        #t = Timer(self.name)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>> auto range of plot
        if self.x_min == 0 and self.x_max == 0:
            # >>> recalculate to velocity offset if necessary
            if not self.vel_scale:
                self.x_min, self.x_max = self.spec.x[0], self.spec.x[-1]
            else:
                self.x_min, self.x_max = (self.spec.x[0]/self.wavelength/(1+self.parent.z_ref)-1)*299794.26, (self.spec.x[-1]/self.wavelength/(1+self.parent.z_ref)-1)*299794.26
            self.y_min, self.y_max = min(self.spec.y), max(self.spec.y)

        # >>> correct continuum
        if len(self.cont) > 0:
            self.correct_cont()

        # >>> recalculate to velocity offset if necessary
        if self.vel_scale:
            self.spec.x = (self.spec.x / self.wavelength / (1 + self.parent.z_ref) - 1) * 299794.26

        # >>> mask only selected wavelength range
        self.points = self.points[self.spec.mask(self.x_min, self.x_max)]

        # >>> plot spectrum
        if self.show_err:
            elinewidth, ecolor, capsize = 0.5, None, self.parent.error_cap
        else:
            elinewidth, ecolor, capsize = 0, '1.0', 0

        if self.gray_out:
            self.ax.errorbar(self.spec.x, self.spec.y, self.spec.err, lw=self.parent.lw_spec, elinewidth=elinewidth, drawstyle='steps-mid', color='0.5', ecolor=ecolor, capsize=capsize, zorder=0)
            k = (self.points == 0)
            self.spec.y[k] = np.NaN
        self.ax.errorbar(self.spec.x, self.spec.y, self.spec.err, lw=self.parent.lw_spec, elinewidth=elinewidth, drawstyle='steps-mid',  color='k', ecolor=ecolor, capsize=capsize, zorder=1)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>> plot fit
        if self.show_fit:
            self.plot_fit()

        # >>> add residuals
        if self.add_residual and self.show_fit:
            self.plot_residuals()

        # >>> set axis ranges:
        self.ax.axis([self.x_min, self.x_max, self.y_min, self.y_max])

        # >>> set ticks labels:
        if self.yticklabels is None:
            self.ax.set_yticklabels([])
        if self.xticklabels is None:
            self.ax.set_xticklabels([])

        # >>> specify ticks:
        self.ax.xaxis.set_minor_locator(self.x_minorLocator)
        self.ax.xaxis.set_major_locator(self.x_locator)
        self.ax.yaxis.set_minor_locator(self.y_minorLocator)
        self.ax.yaxis.set_major_locator(self.y_locator)
        self.ax.tick_params(which='both', width=1)
        self.ax.tick_params(which='major', length=5)
        self.ax.tick_params(which='minor', length=3)
        self.ax.tick_params(axis='both', which='major', labelsize=self.font-2)

        # >>> set axis ticks formater:
        if self.y_formatter is not None:
            self.ax.yaxis.set_major_formatter(FormatStrFormatter(self.y_formatter))
        if self.x_formatter is not None:
            self.ax.xaxis.set_major_formatter(FormatStrFormatter(self.x_formatter))

        # >>> set axis labels:
        if self.ylabel is not None:
            self.ax.set_ylabel(self.ylabel, fontsize=self.font, labelpad=2)
        if self.xlabel is not None:
            self.ax.set_xlabel(self.xlabel, fontsize=self.font, labelpad=2)

        # >>> add lines
        self.ax.plot([self.x_min, self.x_max], [0.0, 0.0], 'k--', lw=0.5)
        if self.show_comps and self.show_fit:
            for k in range(self.num_comp):
                v = (self.parent.comps[k] - self.parent.z_ref) * 299794.26 / (1 + self.parent.z_ref)
                self.ax.plot([v, v], [self.y_min, self.y_max], color=self.parent.color[k], linestyle=':', lw=1.0) #self.parent.lw[k])
                if self.parent.comp_names is not None:
                    self.ax.text(v, null_res-1.5*delt_res, self.parent.comp_names[k], fontsize=self.font_labels-2,
                    color=self.parent.color[k], backgroundcolor='w', clip_on=True, ha='center', va='top', zorder=21)
                if 0 and 'HI' in self.name:
                    ax.plot([v-81.6, v-81.6], [self.y_min, self.y_max], color=self.parent.color[k], linestyle=':', lw=self.parent.lw[k])

        # >>> add text
        if self.name_pos is not None:
            self.ax.text(self.name_pos[0], self.name_pos[1], str(self.name).strip(), ha='left', va='top', fontsize=self.font_labels, transform=self.ax.transAxes)
            if self.label is not None:
                self.ax.text(1 - self.name_pos[0], self.name_pos[1], str(self.label).strip(), ha='right', va='top', fontsize=self.font_labels, transform=self.ax.transAxes)

        if self.parent.add_ioniz:
            el = element(self.el.name)
            print(el, self.ion_state, el.ionenergies)
            ion_en = "{:.3f} eV".format(el.ionenergies[self.ion_state])
            print(ion_en)
            self.ax.text(1-self.name_pos[0], self.name_pos[1], str(ion_en), ha='right', va='top', fontsize=self.font_labels, transform=self.ax.transAxes)

        return self.ax
    
    def plot_region(self):
        self.ax = plt.axes(self.rect.data)
        # >>> auto range of plot
        print(self.x_min, self.x_max)
        if self.x_min == 0 and self.x_max == 0:
            self.x_min, self.x_max = self.spec.x[0], self.spec.x[-1]
            self.y_min, self.y_max = min(self.spec.y), max(self.spec.y)

        # >>> correct continuum
        if len(self.cont) > 0:
            self.correct_cont()

        # >>> plot spectrum
        if self.show_err:
            elinewidth, ecolor, capsize = 0.5, '0.3', self.parent.error_cap
        else:
            elinewidth, ecolor, capsize = 0, None, 0

        if self.gray_out:
            if self.add_errors:
                self.ax.errorbar(self.spec.x, self.spec.y, self.spec.err, lw=1, elinewidth=elinewidth, drawstyle='steps-mid',
                            color='k', ecolor='0.3', capsize=capsize, zorder=1)
            else:
                self.ax.errorbar(self.spec.x, self.spec.y, lw=1, elinewidth=elinewidth, drawstyle='steps-mid',
                            color='k', capsize=capsize, zorder=1)
            k = (self.points == 0)
            self.spec.y[k] = np.NaN

        self.ax.errorbar(self.spec.x, self.spec.y, self.spec.err, lw=1, elinewidth=elinewidth, drawstyle='steps-mid',
                        color='k', ecolor=ecolor, capsize=capsize, zorder=0)

        # >>> correct continuum
        try:
            if self.add_cont:
                cont = np.genfromtxt('_cont.'.join(self.filename[0].rsplit('.', 1)), unpack=True)
                self.ax.plot(cont[0], cont[1], '-', color=col.tableau10[0])
        except:
            pass

        # >>> plot fit
        if self.show_fit:
            self.plot_fit()

        # >>> add residuals
        if self.add_residual and self.fit:
            self.plot_residuals()

        # >>> set axis ranges:
        self.ax.axis([self.x_min, self.x_max, self.y_min, self.y_max])

        # >>> set ticks labels:
        if self.yticklabels is None:
            self.ax.set_yticklabels([])
        if self.xticklabels is None:
            self.ax.set_xticklabels([])

        # >>> specify ticks:
        self.ax.xaxis.set_minor_locator(self.x_minorLocator)
        self.ax.xaxis.set_major_locator(self.x_locator)
        self.ax.yaxis.set_minor_locator(self.y_minorLocator)
        self.ax.yaxis.set_major_locator(self.y_locator)
        self.ax.tick_params(which='both', width=1)
        self.ax.tick_params(which='major', length=5)
        self.ax.tick_params(which='minor', length=3)
        self.ax.tick_params(axis='both', which='major', labelsize=self.font-2)

        # >>> set axis ticks formater:
        if self.y_formatter is not None:
            self.ax.yaxis.set_major_formatter(FormatStrFormatter(self.y_formatter))
        else:
            self.ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        if self.x_formatter is not None:
            self.ax.xaxis.set_major_formatter(FormatStrFormatter(self.x_formatter))

        # >>> set axis labels:
        if self.ylabel is not None:
            self.ax.set_ylabel(self.ylabel, fontsize=self.font)
        if self.xlabel is not None:
            self.ax.set_xlabel(self.xlabel, fontsize=self.font, labelpad=-4)

        # >>> add lines
        self.ax.plot([self.x_min, self.x_max], [0.0, 0.0], 'k--', lw=0.5)

        # >>> add text
        if self.name_pos is not None:
            self.ax.text(self.name_pos[0], self.name_pos[1], str(self.name).strip(), ha='left', va='top', fontsize=self.font, transform=self.ax.transAxes)
            if self.label is not None:
                self.ax.text(1 - self.name_pos[0], self.name_pos[1], str(self.label).strip(), ha='right', va='top', fontsize=self.font_labels, transform=self.ax.transAxes)

        return self.ax

    def plot_fit(self):

        # >>> recalculate to velocity offset if necessary
        if self.vel_scale:
            self.fit.x = (self.fit.x / self.wavelength / (1 + self.parent.z_ref) - 1) * 299794.26

        # >>> mask only selected wavelength range
        self.fit.mask(self.x_min, self.x_max)

        # >>> plot fit components
        if self.show_comps:
            for k in range(self.num_comp):
                if self.fit_disp is None:
                    if self.vel_scale:
                        self.fit_comp[k].x = (self.fit_comp[k].x / self.wavelength / (1 + self.parent.z_ref) - 1) * 299794.26
                    self.fit_comp[k].mask(self.x_min, self.x_max)
                    self.ax.plot(self.fit_comp[k].x, self.fit_comp[k].y, color=self.parent.color[k], ls=self.parent.ls[k], lw=self.parent.lw[k], zorder=10)
                else:
                    if self.vel_scale:
                        self.fit_comp_disp[k][0].x = (self.fit_comp_disp[k][0].x / self.wavelength / (1 + self.parent.z_ref) - 1) * 299794.26
                        self.fit_comp_disp[k][1].x = (self.fit_comp_disp[k][1].x / self.wavelength / (1 + self.parent.z_ref) - 1) * 299794.26
                    self.fit_comp_disp[k][0].mask(self.x_min, self.x_max)
                    self.fit_comp_disp[k][1].mask(self.x_min, self.x_max)
                    self.ax.fill_between(self.fit_comp_disp[k][0].x, self.fit_comp_disp[k][0].y, self.fit_comp_disp[k][1].y, fc=self.parent.color[k], alpha=self.parent.disp_alpha, zorder=11)
                    self.ax.plot(self.fit_comp_disp[k][0].x, self.fit_comp_disp[k][0].y, color=self.parent.color[k], ls=self.parent.ls[k], lw=self.parent.lw[k], zorder=10)
                    self.ax.plot(self.fit_comp_disp[k][0].x, self.fit_comp_disp[k][1].y, color=self.parent.color[k], ls=self.parent.ls[k], lw=self.parent.lw[k], zorder=10)

        # >>> plot joint fit
        if self.fit_disp is None:
            self.ax.plot(self.fit.x, self.fit.y, color=self.parent.color_total, ls=self.parent.ls_total, lw=self.parent.lw_total, zorder=11)
        else:
            # >>> plot fit dispersion
            if self.vel_scale:
                self.fit_disp[0].x = (self.fit_disp[0].x / self.wavelength / (1 + self.parent.z_ref) - 1) * 299794.26
                self.fit_disp[1].x = (self.fit_disp[1].x / self.wavelength / (1 + self.parent.z_ref) - 1) * 299794.26
            self.fit_disp[0].mask(self.x_min, self.x_max)
            self.fit_disp[1].mask(self.x_min, self.x_max)
            self.ax.fill_between(self.fit_disp[0].x, self.fit_disp[0].y, self.fit_disp[1].y, fc=self.parent.color_total, alpha=self.parent.disp_alpha, zorder=11)
            self.ax.plot(self.fit_disp[0].x, self.fit_disp[0].y, color=self.parent.color_total, ls=self.parent.ls_total, lw=self.parent.lw_total, zorder=10)
            self.ax.plot(self.fit_disp[0].x, self.fit_disp[1].y, color=self.parent.color_total, ls=self.parent.ls_total, lw=self.parent.lw_total, zorder=10)

    def plot_residuals(self):
        color_res = col.tableau20[7]
        color_linres = 'lightseagreen'  # 'mediumpurple' #col.tableau20[5]
        null_res = self.y_max + (self.y_max - self.y_min) * 0.10
        delt_res = (self.y_max - self.y_min) * 0.08
        self.y_max = self.y_max + self.add_residual * (self.y_max - self.y_min) * 0.28
        self.ax.axhline(null_res, color=color_linres, ls='-', lw=0.5, zorder=0)
        self.ax.axhline(null_res + delt_res, color=color_linres, ls='--', lw=0.5, zorder=0)
        self.ax.axhline(null_res - delt_res, color=color_linres, ls='--', lw=0.5, zorder=0)
        # ax.add_patch(patches.Rectangle((0.94*self.x_max, null_res-1.1*delt_res), 0.04*self.x_max, 0.2*delt_res, edgecolor='none', facecolor='w', zorder=20))
        # ax.add_patch(patches.Rectangle((0.94*self.x_max, null_res+0.9*delt_res), 0.04*self.x_max, 0.2*delt_res, edgecolor='none', facecolor='w', zorder=20))
        x_pos = self.x_max - 0.01 * (self.x_max - self.x_min)
        self.ax.text(x_pos, null_res + delt_res, r'+' + str(self.sig) + '$\sigma$', fontsize=self.font - 2,
                color=color_linres, backgroundcolor='w', clip_on=True, ha='right', va='center', zorder=1)
        self.ax.text(x_pos, null_res - delt_res, r'-' + str(self.sig) + '$\sigma$', fontsize=self.font - 2,
                color=color_linres, backgroundcolor='w', clip_on=True, ha='right', va='center', zorder=1)
        # print(self.fit.x, self.fit.y)
        try:
            if sum(self.points) > 0:
                k = (self.points != 1)
                size = 10
                if self.fit_disp is None:
                    fit = interpolate.interp1d(self.fit.x, self.fit.y, bounds_error=False, fill_value=1)
                    y = np.ma.masked_where(k, (
                                (self.spec.y - fit(self.spec.x)) / self.spec.err) / self.sig * delt_res + null_res)
                    if self.parent.res_style == 'scatter':
                        self.ax.scatter(self.spec.x, y, color=color_res, s=size)
                    if self.parent.res_style == 'step':
                        self.ax.step(self.spec.x, y, where='mid', color=color_res, ls=self.parent.ls_total,
                                lw=self.parent.lw_total)
                    # ax.step(self.spec.x[k], ((self.spec.y[k] - fit_0(self.spec.x[k])) / self.spec.err[k]) / self.sig * delt_res + null_res, lw=1, where='mid', color=color_res)
                else:
                    fit_0 = interpolate.interp1d(self.fit_disp[0].x, self.fit_disp[0].y, bounds_error=False,
                                                 fill_value=1)
                    fit_1 = interpolate.interp1d(self.fit_disp[0].x, self.fit_disp[1].y, bounds_error=False,
                                                 fill_value=1)
                    if self.parent.res_style == 'scatter':
                        y = np.ma.masked_where(k, ((self.spec.y - (fit_0(self.spec.x) + fit_1(
                            self.spec.x)) / 2) / self.spec.err) / self.sig * delt_res + null_res)
                        self.ax.scatter(self.spec.x, y, color=color_res, s=size)
                    elif self.parent.res_style == 'step':
                        y0 = np.ma.masked_where(k, ((self.spec.y - fit_0(
                            self.spec.x)) / self.spec.err) / self.sig * delt_res + null_res)
                        y1 = np.ma.masked_where(k, ((self.spec.y - fit_1(
                            self.spec.x)) / self.spec.err) / self.sig * delt_res + null_res)
                        self.ax.fill_between(self.spec.x, y0, y1, step='mid', facecolor=color_res, ls=self.parent.ls_total,
                                        lw=self.parent.lw_total, alpha=0.9, edgecolor=color_res)
        except:
            pass


    def correct_cont(self):
        mask = (self.spec.x > self.cont_range[0]) * (self.spec.x < self.cont_range[1])
        corr = np.ones_like(self.spec.x)
        if len(x[mask]) > 0:
            base = (self.spec.x[mask] - self.spec.x[mask][0]) * 2 / (self.spec.x[mask][-1] - self.spec.x[mask][0]) - 1
            corr[mask] = np.polynomial.chebyshev.chebval(base, self.cont)
        print(corr)

        sum_cont = np.zeros(len(self.spec.x))
        for k in range(len(data[0])):
            for l in range(len(self.cont)):
                sum_cont[k] = sum_cont[k] + self.cont[l] * np.cos((l) * np.arccos(
                    -1 + (self.spec.x[k] - self.cont_range[0]) / (self.cont_range[1] - self.cont_range[0]) * 2))
        print(sum_cont - corr)
        self.spec.y = self.spec.y / sum_cont
        self.spec.err = self.spec.err / sum_cont

        if self.show_fit:
            # >>> correct continuum
            sum_cont = np.zeros(len(self.fit.x))
            for k in range(len(self.fit.x)):
                for l in range(len(self.cont)):
                    sum_cont[k] = sum_cont[k] + self.cont[l]*np.cos((l)*np.arccos(-1+(self.fit.x[k]-self.cont_range[0])/(self.cont_range[1]-self.cont_range[0])*2))
                self.fit.y = self.fit.y / sum_cont
                for c in self.fit.comp:
                    c = c / sum_cont

    def showH2(self, levels=[0, 1, 2, 3, 4, 5], pos=0.84, dpos=0.03, color='cornflowerblue', show_ticks=True, kind='full'):
        if 1:
            ymin, ymax = self.ax.get_ylim()
            pos, dpos = ymin + pos * (ymax - ymin), dpos * (ymax - ymin)
            xmin, xmax = self.ax.get_xlim()
        lines = atomicData.H2(levels)
        lines = [l for l in lines if l.l()*(1+self.parent.z_ref) > xmin and l.l()*(1+self.parent.z_ref) < xmax]
        s = set([str(line).split()[1][:str(line).split()[1].index('-')] for line in lines])
        #print(lines)
        texts = []
        for band in s:
            if ('L' in band and (int(band.split('-')[0][1:]) % 2 == 0 or (np.max(levels) < 5 and int(band.split('-')[0][1:]) < 10))):
                pos_y = pos
            elif 'W' in band:
                pos_y = pos - 2*dpos - 2*dpos * (ymax - ymin)
            else:
                pos_y = pos - dpos - dpos * (ymax - ymin)

            b_lines = [line for line in lines if band in str(line)]
            b_lines = [line for line in b_lines if line.l()*(1 + self.parent.z_ref) > xmin and line.l()*(1 + self.parent.z_ref) < xmax]
            if str(min(levels)) in [str(line).split()[0][3:] for line in b_lines]:
                l = [line.l() for line in b_lines]
                if 1:
                    if show_ticks:
                        for line in b_lines:
                            if line.j_l in levels:
                                self.ax.plot([line.l() * (1 + self.parent.z_ref), line.l() * (1 + self.parent.z_ref)], [pos_y, pos_y + dpos], lw=1.0, color=color, ls='-')
                                if kind == 'full':
                                    texts.append(self.ax.text(line.l() * (1 + self.parent.z_ref), pos_y + dpos, str(line.j_l), ha='center', va='bottom',
                                                 fontsize=self.parent.font - 4, color=color))

                        self.ax.plot([np.min(l) * (1 + self.parent.z_ref), np.max(l) * (1 + self.parent.z_ref)], [pos_y + dpos, pos_y + dpos], lw=1.0, color=color, ls='-')
                if kind == 'short':
                    self.ax.text(np.min(l) * (1 + self.parent.z_ref), pos_y + dpos, band + '-0', ha='left', va='bottom', fontsize=self.parent.font-4, color=color)
                elif kind == 'full':
                    self.ax.text(np.min(l) * (1 + self.parent.z_ref), pos_y + dpos, band + ':$\,\,$', ha='right', va='bottom',
                            fontsize=self.parent.font - 4, color=color)
        print(texts)
        adjust_text(texts, only_move={'text': 'x'})

    def __str__(self):
        return 'plot line object: ' + str(self.name) + ', ' + str(self.wavelength)
    
    def __repr__(self):
        return 'plot line object: ' + str(self.name) + ', ' + str(self.wavelength)
   
if __name__ == '__main__':
    if 1:
        ps = plot_spec(50)
        rects = rect_param(n_rows=10, n_cols=5, order='h', height=0.75)
        ps.specify_rects(rects)
        ps.set_ticklabels()
        fig = plt.figure(1, figsize=(13, 16))
        for p in ps:
            ax = p.plot_line()
            ax.text(0.5, 0.5, ps.index(p), fontsize=20)
        plt.show()   
    if 0:        
        ps = plot_spec('D:/science/Zavarigin/DtoH/J031115-172247/linelist_both.dat')
        rects = [rect_param(n_rows=2, n_cols=1, order='v', height=0.2, row_offset=0.02),
                rect_param(n_rows=10, n_cols=2, order='v', height=0.75)
                ]
        ps.specify_rects(rects)
        ps.set_limits(x_min=-250, x_max=150, y_min=-0.1, y_max=1.2)
        ps.set_ticklabels()
        ps.specify_comps(3.73395103, 3.73443226)
        ps.specify_styles()
        
        fig = plt.figure(1, figsize=(13, 16))
        for p in ps:
            if 'alpha' in p.name:
                print(p.name)
                p.x_min, p.x_max = -1800, 1800
                p.x_minorLocator, p.x_locator = AutoMinorLocator(10), MultipleLocator(1000)
            if 'beta' in p.name:
                p.x_min, p.x_max = -500, 500
                p.x_minorLocator, p.x_locator = AutoMinorLocator(10), MultipleLocator(300)
            ax = p.plot_line()
            #ax.text(0.5, 0.5, p.name, fontsize=20)
        plt.show()    

    