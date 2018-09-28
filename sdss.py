
import astropy.constants as const
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator

from .atomic import atomicData
from .profiles import tau, convolveflux

class SDSS():

    def __init__(self, name, plate, MJD, fiber, z_em):
        """
        - name       : SDSS name of Quasar
        - plate      : SDSS plate
        - MJD        : SDSS MJD
        - fiber      : SDSS fiber
        - z_em       : redshift of Quasar
        - SIMBAD     : SIMBAD identifier
        - spectra    : spectrum
        - norm       : normalized spectrum
        - DLA        : list of DLA in spectra
        - LLS        : list of LLS in spectra
        """
        
        self.name = name
        self.ra = 0
        self.dec = 0
        self.plate = ''
        self.MJD = ''
        self.fiber = ''
        self.z_em = float(z_em)
        self.file = ''
        self.SIMBAD = ''
        self.u = 0.0
        self.g = 0.0
        self.r = 0.0
        self.z = 0.0
        self.i = 0.0
        self.spectrum = []
        self.norm = []
        self.DLA = []
        self.obs = ''
        self.comment = ''
        self.H2_cand = []


    def load_spectrum(self, spec=None, norm=0):
        """
        :param
            - norm   : if norm=1 load normalized spectra, if 1 load raw spectra
        """
        
        if spec == None:
            print(self.file)
            if norm:
                self.norm = np.genfromtxt(self.file, unpack=1)
            else:
                self.spectrum = np.genfromtxt(self.file, unpack=1)
        else:
            self.spectrum = spec
            
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

    def plot_candidate(self, logN=None, v_corr=0, v_space=800, normalized=True, 
                       font=16, save=False, fig=None):
        """
        parameters:
            - z           : redshift of the DLA
            - logN        : column density of HI absorber
            - v_corr      : velocity offset for correction of the z_DLA, in km/s
            - v_space     : x_axis range
            - normalized  : plot normalized specrtum
            - save        : save figure
            - fig         : figure object for plotting
        """
        c = const.c.cgs.value
        z_DLA = (1 + self.H2_cand.z) * (1 + v_corr * 1e5/ c) - 1
        print(z_DLA)
        
        if fig is None:
            fig = plt.figure(figsize=(20, 10))
        # >>> plot H2 panel
        #ax = plt.axes([0.07, 0.50, 0.90, 0.45])
        ax = fig.add_axes([0.07, 0.50, 0.90, 0.45])

        i_min = np.searchsorted(self.spectrum[0], 910 * (1+z_DLA)) 
        i_max = np.searchsorted(self.spectrum[0], 1150 * (1+z_DLA))
        if i_min < i_max:
            # >>> plot spectrum
            ax.errorbar(self.spectrum[0][i_min:i_max],
                        self.spectrum[1][i_min:i_max],
                        self.spectrum[2][i_min:i_max],
                        lw=0.75, elinewidth=0.5, drawstyle='steps-mid',
                        color='k', ecolor='0.3', capsize=3)
            
            # >>> plot H2 candidate profile
            H2 = atomicData().H2(1)
            print(H2)
            
            N = self.H2_cand.H2.col.val
            t = 50
            st = 9*np.exp(- 118.5 / 0.695 / 50)
            n = np.log10(np.array([1/(1+st), st/(1+st)]) * 10**N)
            print(n)
            
            x = np.linspace(self.spectrum[0][i_min], self.spectrum[0][i_max], (i_max - i_min)*10)
            t = np.zeros_like(x)
            for line in H2:
                tl = tau(l=line.l(), f=line.f(), g=line.g(), logN=n[line.j_l], b=3, z=self.H2_cand.z)
                t += tl.calctau(x)
            
            I = convolveflux(x, np.exp(-t), res=2600)
            ax.plot(x, I, '-r', lw=2)
            ax.set_xlim([self.spectrum[0][i_min], self.spectrum[0][i_max]])
            for line in H2:
                if line.j_l == 0 and line.l()*(1+self.H2_cand.z) > self.spectrum[0][i_min]:
                    ax.text(line.l()*(1+self.H2_cand.z), 1.1, str(line)[3:str(line).find('-0')+2], va='bottom', ha='center', color='r', fontsize=12)
        ax.set_ylabel('Normalized flux', fontsize=font)
        
        # >>> plot metal lines panels
        lines = atomicData.DLA_SDSS_H2()
        num = len(lines)
        
        left = 0.07
        width = 0.90/num
        bottom = 0.05
        height = 0.40
        rect_w = [[left+width*i, bottom, width-0.01, height] for i in range(num)]
        
        for i in range(num):
            ax = fig.add_axes(rect_w[i])
            print(lines[i].name, lines[i].l)
            lambda_0 = lines[i].l() * (1 + z_DLA)
            
            vel_space = v_space
            x_minorLocator = AutoMinorLocator(5)
            x_locator = MultipleLocator(500)

            #print(lambda_0, lambda_0 * (1 - vel_space * 1e5 / c), lambda_0 * (1 + vel_space * 1e5 / c))
            i_min = np.searchsorted(self.spectrum[0], lambda_0 * (1 - vel_space * 1e5 / c))
            i_max = np.searchsorted(self.spectrum[0], lambda_0 * (1 + vel_space * 1e5 / c))
            
            if i_min < i_max:
                
                if normalized:
                    y_min, y_max = -0.1, 1.3
                else:
                    y_min, y_max = np.min(self.spectrum[1][i_min:i_max]), np.max(self.spectrum[1][i_min:i_max])
                    y_min, y_max = (y_min-y_max)*0.1, y_max+(y_max-y_min)*0.2
                
                # >>> plot spectrum
                ax.errorbar((self.spectrum[0][i_min:i_max] / lambda_0 - 1) * c / 1e5,
                                       self.spectrum[1][i_min:i_max],self.spectrum[2][i_min:i_max],
                                       lw=0.75, elinewidth=0.5, drawstyle='steps-mid',
                                       color='k', ecolor='0.3', capsize=3)

                # >>> set axis
                #axs.axis([-vel_space, vel_space, y_min, y_max])
                ax.set_xlim([-vel_space, vel_space])
                ax.set_ylim([y_min, y_max])
                
                # >>> set labels
                if i == 0:
                    ax.set_ylabel('Normalized flux', fontsize=font)
                ax.set_xlabel('v [km/s]', fontsize=font)

                # >>> set text
                ax.text(-vel_space * 0.9, y_min, str(lines[i]), color='red',
                                   fontsize=font, horizontalalignment='left', verticalalignment='bottom')

                # >>> set ticks
                ax.xaxis.set_minor_locator(x_minorLocator)
                ax.xaxis.set_major_locator(x_locator)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.tick_params(which='major', length=5, width=1, labelsize=font - 2)
                ax.tick_params(which='minor', length=3, width=1)
                
                # >>> set lines
                ax.plot([-vel_space, vel_space], [0.0, 0.0], 'k--', lw=0.5)
                ax.axvline(0, color='#aa0000', linestyle='--', lw=1.5)
                
                
        #fig.suptitle(self.name + ', z=' + str(z_DLA), fontsize=font+2)
        if save:
            print(self.name + '.pdf')
            plt.savefig(self.name + '.pdf', bbox_inches='tight', pad_inches=0.1)
            plt.show()
        

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
        print('q.comment = \'' + str(self.comment) + '\'')
        print('Q.append(q)')
        print('')

if __name__ == '__main__':
    pass
    