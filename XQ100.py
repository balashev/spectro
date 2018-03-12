import matplotlib.pyplot as plt
import numpy as np
import os
import re
from astropy import constants as const
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator
from lxml import etree
from scipy.interpolate import splev, splrep
from .profiles import convolveflux
from .atomic import atomicData
from .absorption_systems import HI

folder = r'D:/science/qso/XQ100/official/'

color = []
color.append('k')

style = []
style.append('-')

font = 16

class QSO():

    def __init__(self, ident, z_em):
        """
        - id         : short id of qso, like J0000-0000
        - z_em       : redshift of Quasar
        - name       : other names of Quasar (NED, e.t.c)
        - SIMBAD     : SIMBAD identifier
        - Mag        : Magnitude
        - HighRes    : is high resolution spectra available?
        - spectra    : spectrum
        - norm       : normalized spectrum
        - DLA        : list of DLA in spectra
        - LLS        : list of LLS in spectra
        - DtoH       : posibility to estimate D to H ratio
        - comment    : comment
        """
        self.id = ident
        self.z_em = float(z_em)
        self.z_lit = 0
        self.file = ''
        self.name = []
        self.SIMBAD = ''
        self.Mag = ''
        self.HighRes = ''
        self.spectra = []
        self.norm = []
        self.DLA = []
        self.LLS = []
        self.DtoH = None
        self.cont = ''
        self.comment = ''


    def load_spectrum(self, norm=0):
        """
        parameters:
            - norm   : if norm=1 load normalized spectra, if 1 load raw spectra
        """

        print(folder + self.id + '/' + self.id + '.dat')
        if norm == 0:
            self.spectrum = np.genfromtxt(folder+self.id+'\\'+self.id+'.dat', unpack=1)

        if norm == 1:
            self.norm = np.genfromtxt(file, unpack=1)

    def calc_norm(self):
        filename = folder + self.id + '/cont.sss'
        with open(filename) as f:
            i = 1
            while f.readline().find('Bcont') == -1:
                i += 1
            n = int(f.readline())
            x, y = np.genfromtxt(filename, skip_header=i+1, max_rows=n, unpack=1)

        tck = splrep(x, y)
        self.norm[:] = self.spectrum[:]
        splain = splev(self.norm[0], tck)
        self.norm[1] = self.spectrum[1] / splain
        self.norm[2] = self.spectrum[2] / splain

    def set_file(self, file):

        self.file = file

    def add_DLA(self, z):
        for w in re.findall('[0-9]\.[0-9]+', z):
            self.DLA.append(HI(float(w)))
        #print(self.DLA)

    def add_LLS(self, z):
        for w in re.findall('[0-9]\.[0-9]+', z):
            self.LLS.append(HI(float(w)))
        #print(self.LLS)

    def plot_DLA(self, z, logN=None, v_corr=0, v_space=500, normalized=False, 
                 save=False, debug=False, verbose=False):
        """
        parameters:
            - z           : redshift of the DLA
            - logN        : column density of HI absorber
            - v_corr      : velocity offset for correction of the z_DLA, in km/s
            - v_space     : x_axis range
            - normalized  : plot normalized specrtum
            - save        : save figure
        """
        c = const.c.cgs.value
        z_DLA = (1 + z) * (1 + v_corr * 1e5/ c) - 1

        n_rows, n_cols = 15, 2 #int(len(lines) / 2) + 1

        # set x and y axis scale
        x_minorLocator = AutoMinorLocator(9)
        x_locator = MultipleLocator(100)
        # y_minorLocator = AutoMinorLocator(9)
        # y_locator = MultipleLocator(1.0)
        y_min = -0.1
        y_max = 1.4

        width = 35
        h_size = width * n_cols + 1
        height = 20
        v_size = height * n_rows + 1
        
        lines = atomicData.DLA_minor()
        num = len(lines)
        
        if n_cols > 1 and n_rows > 1:
            fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 40))

            l = -1
            col = 0
            for line in lines:
                lambda_0 = line.l * (1 + z_DLA)
                if verbose:
                    print(line, lambda_0)
                
                if str(line) == 'HI 1215':
                    vel_space = max(v_space, lorenz_range(0.05, logN=logN, line=line) + 100)
                    x_minorLocator = AutoMinorLocator(4)
                    x_locator = MultipleLocator(500)
                elif str(line) == 'HI 1025':
                    vel_space = max(v_space, lorenz_range(0.05, logN=logN, line=line) + 100)
                    x_minorLocator = AutoMinorLocator(3)
                    x_locator = MultipleLocator(400)
                else:
                    vel_space = v_space
                    x_minorLocator = AutoMinorLocator(9)
                    x_locator = MultipleLocator(100)

                # find y_min, y_max:
                if verbose:
                    print('range: ', lambda_0 * (1 - vel_space * 1e5 / c), lambda_0 * (1 + vel_space * 1e5 / c))
                    
                i_min, i_max = np.searchsorted(self.norm[0], lambda_0 * (1 - vel_space * 1e5 / c)), np.searchsorted(
                    self.norm[0], lambda_0 * (1 + vel_space * 1e5 / c))
                # print(i_min, i_max)
                if i_min < i_max:
                    if line.name == 'HI' and col == 0:
                        col = 1
                        l = -1
                    
                    l = l + 1
                    row = l

                    if line.name == 'HI':
                        y_min, y_max = -0.1, 1.3
                    else:
                        y_min, y_max = minmax(self.norm[1][i_min:i_max])
                    # y_min, y_max = np.amin(qso[1][i_min:i_max]), np.amax(qso[1][i_min:i_max])

                    axs = ax[row, col]
                    # >>> plot spectrum
                    axs.errorbar((self.norm[0][i_min:i_max] / lambda_0 - 1) * c / 1e5,
                                 self.norm[1][i_min:i_max],self.norm[2][i_min:i_max],
                                 lw=0.75, elinewidth=0.5, drawstyle='steps-mid',
                                 color='k', ecolor='0.3', capsize=3)

                    # >>> set axis
                    axs.axis([-vel_space, vel_space, y_min, y_max])

                    # >>> set labels
                    if col == 0 and col == n_cols / 2:
                        axs.set_ylabel('Normalized flux', fontsize=font)
                    if row == n_rows - 1:
                        axs.set_xlabel('v [km/s]', fontsize=font)

                    # >>> set text
                    axs.text(-vel_space * 0.9, y_min, line, color='red', fontsize=font, 
                             ha='left', va='bottom')

                    # >>> set ticks
                    axs.xaxis.set_minor_locator(x_minorLocator)
                    axs.xaxis.set_major_locator(x_locator)
                    axs.yaxis.set_major_locator(MaxNLocator(nbins=4))
                    axs.tick_params(which='major', length=5, width=1, labelsize=font - 2)
                    axs.tick_params(which='minor', length=3, width=1)
                    
                    # >>> set lines
                    axs.plot([-vel_space, vel_space], [0.0, 0.0], 'k--', lw=0.5)
                    axs.plot([0, 0], [y_min, y_max], color='#aa0000', linestyle='--', lw=1.5)
                    if line.name == 'HI':
                        axs.plot([-81.6, -81.6], [y_min, y_max], color='#0000aa', linestyle='--', lw=1.5)

                    # >>> plot absorption lines
                    if logN is not None:
                        wscale = np.linspace(self.norm[0][i_min], self.norm[0][i_max], (i_max-i_min)*3)
                        if line.name == 'HI':
                            # add HI lines
                            v_0 = (wscale / lambda_0 - 1) * c / 1e5
                            tau = calctau(v_0, line.l, line.f, line.g, logN, 20.0, vel=True)
                            axs.plot(v_0, convolveflux(v_0, np.exp(-tau), res=20000, vel=True), '-r', lw=1.5)
                            # add DI lines
                            v = (wscale / lambda_0 / (1 - 81.6e5 / c) - 1) * c / 1e5
                            tau = calctau(v, line.l, line.f, line.g, logN - 4.55, 19.0, vel=True)
                            
                            axs.plot(v_0, convolveflux(v_0, np.exp(-tau), res=20000, vel=True), '-b', lw=1.5)

            fig.suptitle(q.id + ', z=' + str(z), fontsize=14)
        print(q.id + '.pdf')
        plt.savefig(q.id + '.pdf', bbox_inches='tight', pad_inches=0.1)
        plt.show()
        

    def plot_LLS(self):
        pass

    def print_code(self):
        #print(self.id, self.z_em)
        print('# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('# add {}'.format(self.id))
        print('q = QSO(\'{}\', {})'.format(self.id, self.z_em))
        print('q.name = ' + str(self.name))
        print('q.Mag = \'' + str(self.Mag) + '\'')
        print('q.HighRes = \'' + str(self.HighRes) + '\'')
        print('q.z_lit = ' + str(self.z_lit))
        for d in self.DLA:
            print('q.DLA.append(HI(' + str(d.z)+'))')
        for d in self.LLS:
            print('q.LLS.append(HI(' + str(d.z) + '))')
        print('q.comment = \'' + str(self.comment) + '\'')
        print('Q.append(q)')
        print('')
    
    def htmlrow(self, keys):
        st = '<tr>'
        for w in keys.split():
            st += '<td>' + str(getattr(self, w)) + r'</td>'
        st += r'</tr>'
        return st.replace('[', '').replace(']', '').replace('None', '').replace('\'', '')
    
    def __str__(self):
        return self.id
    
    def __repr__(self):
        return self.id
        
def load_QSO():
    Q = []

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0003-2603
    q = QSO('J0003-2603', 4.125)
    q.name = ['[HB89] 0000-263']
    q.Mag = 'NED=18.?'
    q.HighRes = 'Y'
    q.z_lit = 4.01
    q.DLA.append(HI(3.390191, 21.4))
    q.DLA.append(HI(3.054227, 20.0))
    q.comment = 'high DLA'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0006-6208
    q = QSO('J0006-6208', 4.44)
    q.name = ['BR J0006-6208']
    q.Mag = 'NED=18.3,R=19.3'
    q.HighRes = 'Y'
    q.z_lit = 4.46
    q.DLA.append(HI(3.775934, logN=20.9))
    q.DLA.append(HI(3.202631, logN=20.8))
    q.LLS.append(HI(4.357900, logN=18.0))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0030-5129
    q = QSO('J0030-5129', 4.173)
    q.name = ['BR J0030-5129']
    q.Mag = 'R=18.6'
    q.HighRes = 'N'
    q.z_lit = 4.17
    q.DLA.append(HI(2.452397, logN=21.0))
    q.comment = 'no LLS'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0034+1639
    q = QSO('J0034+1639', 4.292)
    q.name = ['PSS J0034+1639']
    q.Mag = 'R_APM=18.0'
    q.HighRes = 'N'
    q.z_lit = 4.29
    q.DLA.append(HI(3.753871, logN=20.3))
    q.DLA.append(HI(4.25231, logN=20.9))
    q.DLA.append(HI(4.2833438, logN=20.9))
    q.comment = 'high DLA'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0042-1020
    q = QSO('J0042-1020', 3.863)
    q.name = ['SDSS J004219.74-102009.4']
    q.Mag = 'APM_R=18.2,r=18.7'
    q.HighRes = 'N'
    q.z_lit = 3.88
    q.DLA.append(HI(2.754484, logN=20.1))
    q.LLS.append(HI(3.628688, logN=18.4))
    q.cont = 'Y'
    q.comment = 'OI deficient!!'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0048-2442
    q = QSO('J0048-2442', 4.083)
    q.name = ['BR 0035-25(BRI J0048-2442)']
    q.Mag = 'NED=18.9'
    q.HighRes = 'N'
    q.z_lit = 4.15
    q.DLA.append(HI(3.757833, logN=19.7))
    q.DtoH = '1DLA'
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0056-2808
    q = QSO('J0056-2808', 3.635)
    q.name = ['[HB89] 0053-284']
    q.Mag = 'NED=18.3'
    q.HighRes = 'N'
    q.z_lit = 3.62
    q.LLS.append(HI(3.580316, logN=18.5))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0057-2643
    q = QSO('J0057-2643', 3.661)
    q.name = ['[HB89] 0055-269']
    q.Mag = 'NED=17.1'
    q.HighRes = 'Y'
    q.z_lit = 3.66
    q.comment = 'no Ly cutoff'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0100-2708
    q = QSO('J0100-2708', 3.546)
    q.name = ['PMN J0100-2708']
    q.Mag = 'R=18.7'
    q.HighRes = 'N'
    q.z_lit = 3.52
    q.DLA.append(HI(3.242792, logN=19.8))
    q.DtoH = '1DLA'
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0113-2803
    q = QSO('J0113-2803', 4.314)
    q.name = ['BRI J0113-2803']
    q.Mag = 'R=18.7'
    q.HighRes = 'N'
    q.z_lit = 4.30
    q.DLA.append(HI(3.890575, logN=19.4))
    q.DLA.append(HI(3.106263, logN=21))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0117+1552
    q = QSO('J0117+1552', 4.243)
    q.name = ['PSS J0117+1552']
    q.Mag = 'V=18.6'
    q.HighRes = 'N'
    q.z_lit = 4.24
    q.DLA.append(HI(2.522548, logN=20.1))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0121+0347
    q = QSO('J0121+0347', 4.125)
    q.name = ['PSS J0121+0347']
    q.Mag = 'R=18.3, V=17.9'
    q.HighRes = 'Y'
    q.z_lit = 4.13
    q.LLS.append(HI(3.805467))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0124+0044
    q = QSO('J0124+0044', 3.837)
    q.name = ['SDSS J0124+0044']
    q.Mag = 'APM_R=17.5?,r=17.9'
    q.HighRes = 'Y'
    q.z_lit = 3.84
    q.DLA.append(HI(3.077774, logN=20.2))
    q.DLA.append(HI(2.261129, logN=20.7))
    q.comment = 'CrII and MnII abs'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0132+1341
    q = QSO('J0132+1341', 4.152)
    q.name = ['PSS J0132+1341']
    q.Mag = 'R_APM=18.5'
    q.HighRes = 'N'
    q.z_lit = 4.16
    q.DLA.append(HI(3.936262, logN=20.3))
    q.LLS.append(HI(4.10523))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0133+0400
    q = QSO('J0133+0400', 4.185)
    q.name = ['PSS J0133+0400']
    q.Mag = 'R_APM=18.3'
    q.HighRes = 'Y'
    q.z_lit = 4.15
    q.DLA.append(HI(4.115345, logN=19.6))
    q.DLA.append(HI(3.995523, logN=20.1))
    q.DLA.append(HI(3.772417, logN=20.5))
    q.DLA.append(HI(3.692212, logN=20.5))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0137-4224
    q = QSO('J0137-4224', 3.971)
    q.name = ['BRI J0137-4224']
    q.Mag = 'NED=18.46'
    q.HighRes = 'Y'
    q.z_lit = 3.97
    q.DLA.append(HI(3.665325, logN=19.1))
    q.DLA.append(HI(3.10083, logN=19.9))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0153-0011
    q = QSO('J0153-0011', 4.195)
    q.name = ['SDSS J015339.60-001104.8']
    q.Mag = 'APM_R=18.0,r=18.9'
    q.HighRes = 'N'
    q.z_lit = 4.19
    q.LLS.append(HI(3.87517))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0211+1107
    q = QSO('J0211+1107', 3.973)
    q.name = ['PSS J0211+1107']
    q.Mag = 'APM_R=18.2'
    q.HighRes = 'N'
    q.z_lit = 3.98
    q.DLA.append(HI(3.501615, logN=19.9))
    q.DLA.append(HI(3.1415687, logN=19.9))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0214-0517
    q = QSO('J0214-0517', 3.977)
    q.name = ['PMN J0214-0517']
    q.Mag = 'APM_R=18.4'
    q.HighRes = 'N'
    q.z_lit = 3.99
    q.DLA.append(HI(3.721297, logN=20.6))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0234-1806
    q = QSO('J0234-1806', 4.305)
    q.name = ['BR J0234-1806']
    q.Mag = 'NED=18.8'
    q.HighRes = 'N'
    q.z_lit = 4.31
    q.DLA.append(HI(3.693897, logN=20.4))
    q.DLA.append(HI(4.228031, logN=19.5))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0244-0134
    q = QSO('J0244-0134', 4.055)
    q.name = ['BRI 0241-0146']
    q.Mag = 'APM_R=17.8'
    q.HighRes = 'Y'
    q.z_lit = 4.05
    q.LLS.append(HI(3.966595))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0247-0556
    q = QSO('J0247-0556', 4.233)
    q.name = ['BR 0245-0608']
    q.Mag = 'NED=18.6'
    q.HighRes = 'Y'
    q.z_lit = 4.24
    q.LLS.append(HI(4.138486))
    q.comment = 'a lot of metal lines'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0248+1802
    q = QSO('J0248+1802', 4.439)
    q.name = ['PSS J0248+1802']
    q.Mag = 'APM_R=17.7'
    q.HighRes = 'Y'
    q.z_lit = 4.42
    q.LLS.append(HI(4.129949))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0255+0048
    q = QSO('J0255+0048', 4.003)
    q.name = ['SDSS J025518.57+004847.4']
    q.Mag = 'APM_R=18.3,r=19.0'
    q.HighRes = 'Y'
    q.z_lit = 4.01
    q.DLA.append(HI(3.252838, logN=20.5))
    q.DLA.append(HI(3.914426, logN=21.3))
    q.LLS.append(HI(3.450383))
    q.DtoH = '1DLA'
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0307-4945
    q = QSO('J0307-4945', 4.716)
    q.name = ['BR J0307-4945']
    q.Mag = 'NED=18.8'
    q.HighRes = 'Y'
    q.z_lit = 4.72
    q.DLA.append(HI(4.468128, logN=20.6))
    q.DLA.append(HI(3.591209, logN=20.3))
    q.LLS.append(HI(4.211639, logN=19))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0311-1722
    q = QSO('J0311-1722', 4.034)
    q.name = ['BR J0311-1722']
    q.Mag = 'APM_R=17.7'
    q.HighRes = 'N'
    q.z_lit = 4.04
    q.DLA.append(HI(3.734179, logN=20.2))
    q.DtoH = '1DLA'
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0403-1703
    q = QSO('J0403-1703', 4.227)
    q.name = ['BR 0401-1711']
    q.Mag = 'NED=18.7'
    q.HighRes = 'Y'
    q.z_lit = 4.23
    q.LLS.append(HI(4.2306717, 18.0))
    q.DtoH = '1LLS'
    q.cont = 'Y'
    q.comment = 'z_LLS > z_QSO, very large Lya emission line'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0415-4357
    q = QSO('J0415-4357', 4.073)
    q.name = ['BR J0415-4357']
    q.Mag = 'NED=18.8'
    q.HighRes = 'N'
    q.z_lit = 4.07
    q.DLA.append(HI(3.80774, logN=20.2))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0424-2209
    q = QSO('J0424-2209', 4.329)
    q.name = ['BR 0424-2209']
    q.Mag = 'NED=17.9, no APM'
    q.HighRes = 'Y'
    q.z_lit = 4.32
    q.DLA.append(HI(2.982721, logN=21.4))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0525-3343
    q = QSO('J0525-3343', 4.385)
    q.name = ['BR 0523-3345']
    q.Mag = 'APM_R=18.4'
    q.HighRes = 'N'
    q.z_lit = 4.41
    q.LLS.append(HI(4.067916))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0529-3552
    q = QSO('J0529-3552', 4.172)
    q.name = ['BR J0529-3552']
    q.Mag = 'APM_R=18.3'
    q.HighRes = 'N'
    q.z_lit = 4.17
    q.DLA.append(HI(3.68437, logN=20.1))
    q.LLS.append(HI(4.065729))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0529-3526
    q = QSO('J0529-3526', 4.418)
    q.name = ['BR J0529-3526']
    q.Mag = 'NED=18.9'
    q.HighRes = 'N'
    q.z_lit = 4.41
    q.DLA.append(HI(3.571748, logN=20.1))
    q.comment = 'Stange DLA at 5560'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0714-6455
    q = QSO('J0714-6455', 4.464)
    q.name = ['BR J0714-6455']
    q.Mag = 'APM_R=18.3'
    q.HighRes = 'N'
    q.z_lit = 4.46
    q.LLS.append(HI(4.459323, 18.0))
    q.LLS.append(HI(3.750112, 18.0))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0747+2739
    q = QSO('J0747+2739', 4.133)
    q.name = ['SDSS J074711.15+273903.3']
    q.Mag = 'APM_R=17.2,r=18.5'
    q.HighRes = 'Y'
    q.z_lit = 4.17
    q.DLA.append(HI(3.900842, logN=20.5))
    q.DLA.append(HI(3.42396, logN=20.7))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0755+1345
    q = QSO('J0755+1345', 3.663)
    q.name = ['SDSS J075552.41+134551.1']
    q.Mag = 'APM_R=18.8,r=18.6'
    q.HighRes = 'N'
    q.z_lit = 3.67
    q.DLA.append(HI(2.485556, logN=20.0))
    q.LLS.append(HI(3.096194))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0800+1920
    q = QSO('J0800+1920', 3.948)
    q.name = ['SDSS J080050.27+192058.9']
    q.Mag = 'APM_R=18.3, r=20.0'
    q.HighRes = 'N'
    q.z_lit = 3.96
    q.DLA.append(HI(3.946544, logN=20.4))
    q.DLA.append(HI(3.429224, logN=20.0))
    q.comment = 'DLA at z_em'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0818+0958
    q = QSO('J0818+0958', 3.656)
    q.name = ['SDSS J081855.78+095848.0']
    q.Mag = 'APM_R=17.7,r=17.9'
    q.HighRes = 'N'
    q.z_lit = 3.67
    q.DLA.append(HI(3.30593, logN=21))
    q.LLS.append(HI(3.531259))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0833+0959
    q = QSO('J0833+0959', 3.716)
    q.name = ['SDSS J083322.50+095941.2']
    q.Mag = 'APM_R=18.5,r=18.8'
    q.HighRes = 'N'
    q.z_lit = 3.75
    q.LLS.append(HI(3.050229))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0835+0650
    q = QSO('J0835+0650', 4.007)
    q.name = ['SDSS J083510.92+065052.8']
    q.Mag = 'APM_R=18.0,r=18.5'
    q.HighRes = 'N'
    q.z_lit = 3.99
    q.DLA.append(HI(3.955743, logN=20.3))
    q.DLA.append(HI(3.6020198, logN=19.7))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0839+0318
    q = QSO('J0839+0318', 4.23)
    q.name = ['SDSS J083941.45+031817.0']
    q.Mag = 'APM_R=17.9, r=18.9'
    q.HighRes = 'N'
    q.z_lit = 4.25
    q.DLA.append(HI(4.098013, logN=19.6))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0920+0725
    q = QSO('J0920+0725', 3.646)
    q.name = ['SDSS J092041.76+072544.0']
    q.Mag = 'APM_R=18.5,r=18.6'
    q.HighRes = 'N'
    q.z_lit = 3.64
    q.DLA.append(HI(2.237169, logN=20.7))
    q.LLS.append(HI(3.059595))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0935+0022
    q = QSO('J0935+0022', 3.747)
    q.name = ['SDSS J093556.91+002255.6']
    q.Mag = 'APM_R=17.8,r=18.7'
    q.HighRes = 'N'
    q.z_lit = 3.75
    q.LLS.append(HI(3.562382))
    q.LLS.append(HI(3.305247))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0937+0828
    q = QSO('J0937+0828', 3.703)
    q.name = ['SDSS J093714.48+082858.6']
    q.Mag = 'APM_R=18.2,r=18.7'
    q.HighRes = 'N'
    q.z_lit = 3.70
    q.DLA.append(HI(3.12899, logN=19.6))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0955-0130
    q = QSO('J0955-0130', 4.418)
    q.name = ['BRI 0952-0115']
    q.Mag = 'NED=18.7'
    q.HighRes = 'Y'
    q.z_lit = 4.43
    q.DLA.append(HI(4.021088, logN=20.5))
    q.DLA.append(HI(3.475441, logN=20.1))
    q.LLS.append(HI(4.433464, logN=20.5))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J0959+1312
    q = QSO('J0959+1312', 4.092)
    q.name = ['SDSS J095937.11+131215.4']
    q.Mag = 'APM_R=16.9'
    q.HighRes = 'N'
    q.z_lit = 4.06
    q.DLA.append(HI(3.912422, logN=20.0))
    q.comment = ''
    Q.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1013+0650
    q = QSO('J1013+0650', 3.808)
    q.name = ['SDSS J101347.29+065015.6']
    q.Mag = 'APM_R=18.4, r=18.9'
    q.HighRes = 'N'
    q.LLS.append(HI(3.489365, logN=19.5))
    q.z_lit = 3.79
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1018+0548
    q = QSO('J1018+0548', 3.515)
    q.name = ['SDSS J101818.45+054822.8']
    q.Mag = 'APM_R=18.1,r=18.8'
    q.HighRes = 'N'
    q.z_lit = 3.52
    q.DLA.append(HI(3.38445, logN=19.6))
    q.DtoH = '1DLA'
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1020+0922
    q = QSO('J1020+0922', 3.64)
    q.name = ['SDSS J102040.62+092254.2']
    q.Mag = ' APM_R=18.0,r=18.4 '
    q.HighRes = 'N'
    q.z_lit = 3.64
    q.DLA.append(HI(2.748895, logN=20.1))
    q.DLA.append(HI(2.593157, logN=21.5))
    q.LLS.append(HI(3.106269, logN=18))
    q.DtoH = '1LLS'
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1024+1819
    q = QSO('J1024+1819', 3.524)
    q.name = ['SDSS J102456.61+181908.7']
    q.Mag = ' APM_R=17.9 '
    q.HighRes = 'N '
    q.z_lit = 3.53
    q.DLA.append(HI(2.298305, logN=21.4))
    q.LLS.append(HI(3.188573, logN=18))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1032+0927
    q = QSO('J1032+0927', 3.985)
    q.name = ['SDSS J103221.11+092748.9']
    q.Mag = ' APM_R=17.9, r=18.8'
    q.HighRes = 'N'
    q.z_lit = 3.99
    q.DLA.append(HI(3.804089, logN=19.8))
    q.DtoH = '1DLA'
    q.cont = 'Y'
    q.comment = 'perfect DtoH'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1036-0343
    q = QSO('J1036-0343', 4.531)
    q.name = ['BR 1033-0327']
    q.Mag = ' APM_R=18.5'
    q.HighRes = 'N'
    q.z_lit = 4.51
    q.DLA.append(HI(4.174784, logN=19.9))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1034+1102
    q = QSO('J1034+1102', 4.269)
    q.name = ['SDSS J103446.54+110214.5']
    q.Mag = ' APM_R=18.2, r=18.9'
    q.HighRes = 'N'
    q.z_lit = 4.27
    q.LLS.append(HI(3.541946, logN=18.3))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1037+0704
    q = QSO('J1037+0704', 4.127)
    q.name = ['SDSS J103732.38+070426.2']
    q.Mag = ' APM_R=18.3,r=18.5'
    q.HighRes = 'N'
    q.z_lit = 4.10
    q.LLS.append(HI(4.012133))
    q.LLS.append(HI(3.281879))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1037+2135
    q = QSO('J1037+2135', 3.626)
    q.name = ['SDSS J103730.33+213531.3']
    q.Mag = ' APM_R=17.7'
    q.HighRes = 'N'
    q.z_lit = 3.63
    q.DLA.append(HI(1.919930, logN=19.5))
    q.cont = 'Y'
    q.comment = 'no LLS'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1042+1957
    q = QSO('J1042+1957', 3.63)
    q.name = ['SDSS J104234.01+195718.6']
    q.Mag = ' APM_R=18.1'
    q.HighRes = 'N'
    q.z_lit = 3.64
    q.LLS.append(HI(3.269873))
    q.cont = 'Y'
    q.comment = 'no LLS'
    Q.append(q)
       
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1053+0103
    q = QSO('J1053+0103', 3.663)
    q.name = ['SDSS J105340.75+010335.6']
    q.Mag = 'APM_R=19.1,r=18.5'
    q.HighRes = 'N'
    q.z_lit = 3.65
    q.LLS.append(HI(3.044643))
    q.comment = 'no Ly cutoff'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1054+0215
    q = QSO('J1054+0215', 3.971)
    q.name = ['SDSS J105434.17+021551.9']
    q.Mag = ' APM_R=18.0, r=18.8'
    q.HighRes = 'N'
    q.z_lit = 3.97
    q.LLS.append(HI(3.936313))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1057+1910
    q = QSO('J1057+1910', 4.128)
    q.name = ['SDSS J105705.37+191042.8']
    q.Mag = ' APM_R=17.9, r=18.7'
    q.HighRes = 'N'
    q.z_lit = 4.10
    q.DLA.append(HI(3.373576, logN=20.2))
    q.LLS.append(HI(4.062002))
    q.LLS.append(HI(3.798568))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1058+1245
    q = QSO('J1058+1245', 4.341)
    q.name = ['SDSS J105858.38+124554.9']
    q.Mag = ' APM_R=17.6, V=18'
    q.HighRes = 'N'
    q.z_lit = 4.33
    q.DLA.append(HI(3.430422, logN=20.5))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1103+1004
    q = QSO('J1103+1004', 3.607)
    q.name = ['SDSS J110352.73+100403.1']
    q.Mag = ' APM_R=18.6,r=18.7'
    q.HighRes = 'N'
    q.z_lit = 3.61
    q.LLS.append(HI(3.241318, logN=18.0))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1110+0244
    q = QSO('J1110+0244', 4.146)
    q.name = ['SDSS J111008.61+024458.0']
    q.Mag = ' APM_R=17.6,r=18.3'
    q.HighRes = 'N'
    q.z_lit = 4.12
    q.LLS.append(HI(3.475846, logN=18.0))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1108+1209
    q = QSO('J1108+1209', 3.678)
    q.name = ['SDSS J110855.47+120953.3']
    q.Mag = ' APM_R=18.4,r=18.6'
    q.HighRes = 'Y'
    q.z_lit = 3.67
    q.DLA.append(HI(3.39633, logN=20.7))
    q.DLA.append(HI(3.544972, logN=20.8))
    q.comment = 'stange incr. of flux at 5588A'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1111-0804
    q = QSO('J1111-0804', 3.922)
    q.name = ['BRI 1108-0747']
    q.Mag = 'APM_R=18.8, NED=18.1'
    q.HighRes = 'Y'
    q.z_lit = 3.92
    q.DLA.append(HI(3.607829, logN=20.3))
    q.DLA.append(HI(3.481739, logN=19.9))
    q.LLS.append(HI(3.813805))
    q.LLS.append(HI(3.760536))
    q.comment = ''
    Q.append(q)
  
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1117+1311
    q = QSO('J1117+1311', 3.622)
    q.name = ['SDSS J111701.89+131115.4']
    q.Mag = ' APM_R=18.3,r=18.4'
    q.HighRes = 'N'
    q.z_lit = 3.62
    q.LLS.append(HI(3.275453, logN=18.0))
    q.LLS.append(HI(3.023322, logN=18.0))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1126-0126
    q = QSO('J1126-0126', 3.634)
    q.name = ['SDSS J112617.40-012632.6']
    q.Mag = ' APM_R=18.7,r=18.9'
    q.HighRes = 'N'
    q.z_lit = 3.61
    q.LLS.append(HI(3.289812, logN=18.4))
    q.DtoH = '1LLS'
    q.cont = 'Y'
    q.comment = 'nice LLS lines at edge'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1126-0124
    q = QSO('J1126-0124', 3.765)
    q.name = ['SDSS J112634.28-012436.9']
    q.Mag = ' APM_R=18.5, r=19.0'
    q.HighRes = 'N'
    q.z_lit = 3.74
    q.LLS.append(HI(3.544876))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1135+0842
    q = QSO('J1135+0842', 3.834)
    q.name = ['SDSS J113536.40+084218.9']
    q.Mag = ' APM_R=18.3,r=18.3'
    q.HighRes = 'N'
    q.z_lit = 3.83
    q.comment = 'no LLS'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1201+1206
    q = QSO('J1201+1206', 3.522)
    q.name = ['[HB89] 1159+123']
    q.Mag = ' APM_R=17.3,r=17.4'
    q.HighRes = 'Y'
    q.z_lit = 3.51
    q.DLA.append(HI(1.9962, logN=19.6))
    q.LLS.append(HI(3.52643, logN=18.4))
    q.LLS.append(HI(2.795457, logN=18.2))
    q.DtoH = '2LLS'
    q.cont = 'Y'
    q.comment = 'LLS/BAL at z>z_QSO (CII but no OI/SiII)'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1202-0054
    q = QSO('J1202-0054', 3.592)
    q.name = ['SDSS J120210.08-005425.4']
    q.Mag = ' APM_R=18.5'
    q.HighRes = 'N'
    q.z_lit = 3.59
    q.DLA.append(HI(2.660711, logN=19.6))
    q.LLS.append(HI(3.15089))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1248+1304
    q = QSO('J1248+1304', 3.721)
    q.name = ['SDSS J124837.31+130440.9']
    q.Mag = ' APM_R=18.1,r=18.6'
    q.HighRes = 'N'
    q.z_lit = 3.72
    q.LLS.append(HI(3.559132, logN=18.2))
    q.LLS.append(HI(3.406102, logN=18.2))
    q.cont = 'Y'
    q.comment = 'stange 1LLS'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1249-0159
    q = QSO('J1249-0159', 3.629)
    q.name = ['SDSS J124957.23-015928.8']
    q.Mag = ' APM_R=17.5,r=17.6'
    q.HighRes = 'Y'
    q.z_lit = 3.63
    q.LLS.append(HI(3.525420, logN=18.2))
    q.LLS.append(HI(3.102140, logN=18.2))
    q.comment = 'contaminated 1LLS, no metals'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1304+0239
    q = QSO('J1304+0239', 3.648)
    q.name = ['SDSS J130452.57+023924.8']
    q.Mag = ' APM_R=18.6,r=18.4'
    q.HighRes = 'N'
    q.z_lit = 3.65
    q.DLA.append(HI(3.210205, logN=19.5))
    q.LLS.append(HI(3.336886, logN=18.2))
    q.comment = 'metalrich DLA'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1312+0841
    q = QSO('J1312+0841', 3.731)
    q.name = ['SDSS J131242.87+084105.1']
    q.Mag = ' APM_R=18.4,r=18.6'
    q.HighRes = 'N'
    q.z_lit = 3.74
    q.DLA.append(HI(2.65955, logN=20.5))
    q.LLS.append(HI(3.31002, logN=18.2))
    q.LLS.append(HI(3.013328, logN=18.2))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1320-0523
    q = QSO('J1320-0523', 3.717)
    q.name = ['2MASS J1320299-052335']
    q.Mag = ' APM_R=17.8'
    q.HighRes = 'Y'
    q.z_lit = 3.70
    q.LLS.append(HI(3.576551, logN=17.6))
    q.LLS.append(HI(2.831156, logN=17.6))
    q.comment = 'Stange shift of metal lines in 2LLS'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1323+1405
    q = QSO('J1323+1405', 4.054)
    q.name = ['SDSS J132346.05+140517.6']
    q.Mag = ' APM_R=18.6, r=19.0'
    q.HighRes = 'N'
    q.z_lit = 4.04
    q.LLS.append(HI(3.826578, logN=18.0))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1330-2522
    q = QSO('J1330-2522', 3.948)
    q.name = ['BR J1330-2522']
    q.Mag = ' APM_R=18.5'
    q.HighRes = 'Y'
    q.z_lit = 3.95
    q.DLA.append(HI(3.080478, logN=19.9))
    q.DLA.append(HI(2.654113, logN=19.3))
    q.LLS.append(HI(3.709446, logN=18))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1331+1015
    q = QSO('J1331+1015', 3.852)
    q.name = ['SDSS J133150.69+101529.4']
    q.Mag = 'APM_R=18.8,r=18.6'
    q.HighRes = 'N'
    q.z_lit = 3.85
    q.LLS.append(HI(3.653237))
    q.LLS.append(HI(3.399963))
    q.comment = ''
    Q.append(q)
  
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1332+0052
    q = QSO('J1332+0052', 3.508)
    q.name = ['SDSS J133254.51+005250.6']
    q.Mag = ' APM_R=18.4,r=18.3'
    q.HighRes = 'Y'
    q.z_lit = 3.51
    q.LLS.append(HI(3.421158))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1352+1303
    q = QSO('J1352+1303', 3.706)
    q.name = ['SDSS J135247.98+130311.5']
    q.Mag = ' APM_R=18.4,r=18.2'
    q.HighRes = 'N'
    q.z_lit = 3.70
    q.LLS.append(HI(3.005157, logN=18.3))
    q.cont = 'Y'
    q.comment = 'gradual Ly cutoff from z~3.7, MgII not aligned with HI'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1336+0243
    q = QSO('J1336+0243', 3.801)
    q.name = ['SDSS J133653.44+024338.1']
    q.Mag = ' APM_R=18.6,r=18.7'
    q.HighRes = 'N'
    q.z_lit = 3.80
    q.DLA.append(HI(2.691157, logN=19.7))
    q.DtoH = '1DLA'
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1401+0244
    q = QSO('J1401+0244', 4.408)
    q.name = ['SDSS J1401+0244']
    q.Mag = ' APM_R=18.4,r=19.0'
    q.HighRes = 'N'
    q.z_lit = 4.44
    q.DLA.append(HI(3.020025, logN=19.9))
    q.LLS.append(HI(4.286631, logN=18.5))
    q.comment = ''
    Q.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1416+1811
    q = QSO('J1416+1811', 3.593)
    q.name = ['SDSS J141608.39+181144.0']
    q.Mag = ' APM_R=18.2'
    q.HighRes = 'N'
    q.z_lit = 3.59
    q.DLA.append(HI(2.663178, logN=19.6))
    q.DLA.append(HI(2.227585, logN=21.5))
    q.comment = 'CrII, ZnII, MnII lines in 2DLA'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1421-0643
    q = QSO('J1421-0643', 3.688)
    q.name = ['PKS B1418-064']
    q.Mag = 'R=18.5'
    q.HighRes = 'Y'
    q.z_lit = 3.689
    q.DLA.append(HI(3.448329, logN=20.4))
    q.DtoH = '1DLA'
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1442+0920
    q = QSO('J1442+0920', 3.532)
    q.name = ['SDSS J144250.12+092001.5']
    q.Mag = 'APM_R=17.2,r=17.5'
    q.HighRes = 'N'
    q.z_lit = 3.53
    q.LLS.append(HI(3.025575))
    q.LLS.append(HI(3.013126))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1445+0958
    q = QSO('J1445+0958', 3.562)
    q.name = ['SDSS J1445 +0958']
    q.Mag = 'r=17.9 '
    q.HighRes = 'Y'
    q.z_lit = 3.5203
    q.comment = 'no LLS, little metal lines'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1503+0419
    q = QSO('J1503+0419', 3.692)
    q.name = ['SDSS J150328.88+041949.0']
    q.Mag = ' APM_R=18.0,r=18.1'
    q.HighRes = 'N'
    q.z_lit = 3.66
    q.LLS.append(HI(3.558186, logN=17.6))
    q.LLS.append(HI(3.147267, logN=18.2))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1517+0511
    q = QSO('J1517+0511', 3.555)
    q.name = ['SDSS J151756.18+051103.5']
    q.Mag = ' APM_R=18.3'
    q.HighRes = 'N'
    q.z_lit = 3.56
    q.DLA.append(HI(2.687985, logN=21.3))
    q.LLS.append(HI(3.342199, logN=18.5))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1524+2123
    q = QSO('J1524+2123', 3.6)
    q.name = ['SDSS J152436.08+212309.1']
    q.Mag = ' APM_R=17.3'
    q.HighRes = 'N'
    q.z_lit = 3.61
    q.LLS.append(HI(3.46339, logN=18.8))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1542+0955
    q = QSO('J1542+0955', 3.986)
    q.name = ['SDSS J154237.71+095558.8']
    q.Mag = ' APM_R=18.2,r=18.9'
    q.HighRes = 'N'
    q.z_lit = 3.99
    q.LLS.append(HI(3.330878))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1552+1005
    q = QSO('J1552+1005', 3.722)
    q.name = ['SDSS J155255.03+100538.3']
    q.Mag = ' APM_R=18.6,r=18.9'
    q.HighRes = 'N'
    q.z_lit = 3.73
    q.DLA.append(HI(3.665921, logN=20.7))
    q.DLA.append(HI(3.600765, logN=21.1))
    q.DLA.append(HI(3.442262, logN=19.2))
    q.comment = '3DLA: low HI but high metals'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1621-0042
    q = QSO('J1621-0042', 3.711)
    q.name = ['SDSS J1621-0042']
    q.Mag = ' APM_R=17.7,r=17.3'
    q.HighRes = 'Y'
    q.z_lit = 3.70
    q.DLA.append(HI(3.104085, logN=19.8))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1633+1411
    q = QSO('J1633+1411', 4.365)
    q.name = ['SDSS J163319.63+141142.0']
    q.Mag = ' APM_R=18.7,r=19.0'
    q.HighRes = 'Y'
    q.z_lit = 4.33
    q.DLA.append(HI(2.880290, logN=20.4))
    q.DLA.append(HI(2.594261, logN=20.5))
    q.comment = 'complex LLS at z~4.3'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1658-0739
    q = QSO('J1658-0739', 3.749)
    q.name = ['CGRaBS J1658-0739']
    q.Mag = ' R=18.7'
    q.HighRes = 'N'
    q.z_lit = 3.74
    q.LLS.append(HI(3.687712, logN=18))
    q.LLS.append(HI(3.547231, logN=18))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J1723+2243
    q = QSO('J1723+2243', 4.531)
    q.name = ['PSS J1723+2243']
    q.Mag = ' R=18.2'
    q.HighRes = 'N'
    q.z_lit = 4.52
    q.DLA.append(HI(3.696843, logN=20.5))
    q.comment = 'rich DLA'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J2215-1611
    q = QSO('J2215-1611', 3.994)
    q.name = ['BR 2212-1626']
    q.Mag = ' APM_R=18.1'
    q.HighRes = 'Y'
    q.z_lit = 3.99
    q.DLA.append(HI(3.70137, logN=19.5))
    q.DLA.append(HI(3.661782, logN=20.2))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J2216-6714
    q = QSO('J2216-6714', 4.479)
    q.name = ['BR 2213-6729']
    q.Mag = 'APM_R=18.6'
    q.HighRes = 'Y'
    q.z_lit = 4.47
    q.LLS.append(HI(4.286342, logN=18))
    q.LLS.append(HI(3.979620, logN=18))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J2239-0552
    q = QSO('J2239-0552', 4.556)
    q.name = ['2MASSi J2239536-055219']
    q.Mag = ' APM_R=18.3'
    q.HighRes = 'Y'
    q.z_lit = 4.56
    q.DLA.append(HI(4.078804, logN=20.4))
    q.LLS.append(HI(4.36026))
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J2251-1227
    q = QSO('J2251-1227', 4.157)
    q.name = ['BR 2248-1242']
    q.Mag = ' APM_R=18.6'
    q.HighRes = 'N'
    q.z_lit = 4.16
    q.DLA.append(HI(3.45723, logN=19.9))
    q.comment = 'not zero flux at all lines!'
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J2344+0342
    q = QSO('J2344+0342', 4.248)
    q.name = ['PSS J2344+0342']
    q.Mag = 'APM_R=18.2'
    q.HighRes = 'Y'
    q.z_lit = 4.24
    q.DLA.append(HI(3.8848463, logN=19.7))
    q.DLA.append(HI(3.220192, logN=21.2))
    q.cont = 'Y'
    q.comment = ''
    Q.append(q)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add J2349-3712
    q = QSO('J2349-3712', 4.219)
    q.name = ['BR J2349-3712']
    q.Mag = 'APM_R=18.7'
    q.HighRes = 'N'
    q.z_lit = 4.21
    q.DLA.append(HI(3.6923081, logN=20.1))
    q.LLS.append(HI(4.089958))
    q.comment = ''
    Q.append(q)

    return Q

def load_table(htmlfile):

    data = []

    with open(htmlfile) as f:
        s = f.read()
        #s = f.readlines()[122:224]
    s = s[s.find('<table'):s.find('</table>')+8]
    s = s.replace(r'<tbody>', '').replace(r'</tbody>', '')
    #print(s)
    table = etree.XML(s)
    rows = iter(table)
    headers = [col.text for col in next(rows)]
    for row in rows:
        values = [col.text for col in row]
        #print(values)
        d = dict(zip(headers, values))
        #print(d)
        data.append(QSO(d['id'], d['z_PCA']))
        data[-1].name.append(d['Object'].strip())
        data[-1].Mag = d['Mag']
        data[-1].HighRes = d['HighRes']
        data[-1].z_lit = d['z_lit']
        try:
            data[-1].comment = d['comm']
        except:
            pass
        try:
            data[-1].add_DLA(d['z_DLA'])
        except:
            pass
        try:
            data[-1].add_LLS(d['D/H'])
        except:
            pass
        #print(data[-1].name)
        data[-1].print_code()

    return data


def printHTMLtable(htmlfile):
    
    print("print html table in file: ", htmlfile)
    keys = 'id name z_lit Mag HighRes DLA LLS DtoH cont comment'
    
    st = '<tr><td>N</td>'
    for k in keys.split():
        st += '<td>{}</td>'.format(k)
    st += '</tr>\n'

    for i, q in enumerate(XQ100):
        st += q.htmlrow(keys).replace('<tr>', '<tr><td>{}</td>'.format(i+1)) + '\n'
    
    with open(htmlfile, 'r') as f:
        s = f.read()

    s = s[:s.find('<tbody>')+7] + st + s[s.find('</tbody>'):]
    with open(htmlfile, 'w') as f:
        f.write(s)
    

if __name__ == '__main__':
    
    if 0:
        XQ100 = load_table("D:\science\QSO\XQ100\XQ100.html")
    else:
        XQ100 = load_QSO()
    
    if 0:    
        printHTMLtable("D:\science\QSO\XQ100\XQ100.html")
    
    # print([x.DLA for x in XQ100])
    i = 0
    for q in XQ100:
        #q.print_code()
        if 0:
            if os.path.isfile(folder + q.id + '/cont.sss'):
                print(q.id, q.cont)
                i += 1
        if 0:
            if q.id.find("J1517") > -1:
                print(q.DLA)
                q.load_spectrum()
                q.calc_norm()
                i = 0
                if 0:
                    ax = q.plot_DLA(q.DLA[i].z, q.DLA[i].logN, v_corr=0, v_space=500, save=True)
                else:
                    ax = q.plot_DLA(q.LLS[i].z, q.LLS[i].logN, v_corr=0, v_space=500, save=True)
        if 0:
            if q.DtoH is not None:
                print(q.id)
                q.load_spectrum()
                q.calc_norm()
                words = q.DtoH.split()
                for w in words:
                    i = int(w[:1])
                    print(i)
                    if w[1:] == 'DLA':
                        ax = q.plot_DLA(q.DLA[i-1].z, q.DLA[i-1].logN, v_corr=0, v_space=500, save=True)
                    elif w[1:] == 'LLS':
                        ax = q.plot_DLA(q.LLS[i-1].z, q.LLS[i-1].logN, v_corr=0, v_space=500, save=True)
    if 1:
        i = 0
        d = []
        for q in XQ100:
            if len(q.DLA) > 0:
                for dla in q.DLA:
                    d.append(dla.logN)
                    if dla.logN >= 20.3:
                        i += 1
                        print(i, q.id, dla, dla.logN)
                        
        print(len(d))
        plt.hist(d, bins=20)
    print(i)