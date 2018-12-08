import astropy.constants as const
from collections import OrderedDict
from functools import wraps
import h5py
import numpy as np
import re
import os
from mendeleev import element

if __name__ == '__main__':
    import sys
    sys.path.append('C:/science/python')
    from spectro.a_unc import a
else:
    from .a_unc import a

class e():
    """
    Class for species: 
    
        purpose: set observational data for species 
        
        dependences:    a_unc 
                        mendeleev
                        re
                        
        
        to initialize species:
            1. e('SiII', 19.0, 0.2, 0.2)
            2. e('SiII', a(19.0, 0.2, 0.2))
            3. e('SiII', '19.0\pm0.2')
            4. e('SiII', '19.0^{+0.2}_{-0.2}')
            
            where 19.0, 0.2, 0.2 sets column density of species.
            column density will be saved as <a> object in self.col value.
            
        to initialize upper/lower limit:
            1. e('SiII*', '<19')
            2. e('SiII*', a(19, t='u')),  see also description of <a> object
        
        Molecular species (H2, HD, CO, ...) can be specified as well:
            1. e('H2', 21.0, 0.1, 0.1, J=0)
            2. e('H2', 14.0, 0.1, 0.1, J=0, nu=1)
            
            nu=0 by default
            
        you can also specify b parameter for species, using <b> keyword
            1. e('H2', 18.0, 0.1, 0.1, J=4, b=(2.4, 0.4, 0.8))
            
        notes:
        1. ionization states gets automatically by get_ioniz() method
        2. a list of method are provided coupled with mendeleev package.
        
    """
    def __init__(self, *args, **kwargs):
        
        # representation name SiII, OI*, HD
        self.name = args[0]
        # short desciprion of element, like Si
        self.el = self.get_element_name()
        
        # set column density:
        if len(args) == 1:
            self.col = a()
        elif len(args) == 2:
            if isinstance(args[1], str):
                self.col = a(args[1])
            elif isinstance(args[1], a):
                self.col = args[1]
            elif isinstance(args[1], (int, float)):
                self.col = a(args[1], 0, 0)
        elif len(args) == 4:
            self.col = a(args[1], args[2], args[3])
        else:
            self.col = a(args[1], args[2], args[3], )
        if 'f' in kwargs.keys():
            if kwargs['f'] in ['d', 'dec']:
                self.col.repr = 'dec'
                self.col.default_format = 'dec'
            if kwargs['f'] in ['l', 'log']:
                self.col.repr = 'log'
                self.col.default_format = 'log'

        self.nu = None
        self.J = None
        if 'j' in self.name:
            self.J = int(self.name[int(self.name.index('j'))+1:])
        self.b = None
        
        for k in kwargs.items():
            if k[0] == 'b' and isinstance(k[1], (list, tuple)):
                setattr(self, k[0], a(k[1][0], k[1][1], k[1][2], 'd'))
            else:
                setattr(self, k[0], k[1])
        
        # refresh representation for molecules
        self.name = self.__repr__()
        
        self.mend = None
        
        self.fine = self.name.count('*')
        
        self.statw()
        self.ionstate()

        self.lines = []

        # for line modeling:
        self.logN = self.col.val
        if self.b is not None:
            self.b_ = self.b.val
        
        # some additional parameters:        
        self.mod = 0
        
    def get_element_name(self):
        st = self.name[:]
        for s in ['I', 'V', 'X', '*']:
            st = st.replace(s, '')
            
        return st
        
    def __eq__(self, other):
        return self.name == other.name and self.ion == other.ion

    def __repr__(self):
        st = self.name
        if self.name in ['H2', 'HD', 'CO']:
            if self.J is not None:
                st +=  'j' + str(self.J)
            if self.nu is not None:
                st += 'v' + str(self.nu)
        return st
        
    def __str__(self):
        return self.__repr__() + ' ' + str(self.col)
    
    def info(self):
        """
        print info about element
        """
        seq = ['name', 'el', 'ioniz', 'fine', 'J', 'nu', 'col']
        attrs = vars(self).copy()
        for s in seq:
            print('{:}: {:}'.format(s, attrs[s]))
            del attrs[s]
        for a in attrs.items():
            print('{:}: {:}'.format(a[0], a[1]))
        
    def statw(self):
        """
        return statistical weight of the element
        """
        self.stat = None
        if 'H2' in self.name:
            if self.J is not None:
                self.stat = (2*self.J+1)*((self.J%2)*2+1)

        elif 'HD' in self.name:
            if self.J is not None:
                self.stat = (2*self.J+1)
        
        elif 'CO' in self.name:
            if self.J is not None:
                self.stat = (2*self.J+1)

        elif 'SiII' in self.name:
            self.stat = (self.fine+1)*2
            
        elif 'CII' in self.name:
            self.stat = (self.fine+1)*2
        
        elif 'CI' in self.name:
            self.stat = self.fine*2+1
        
        elif 'OI' in self.name:
            self.stat = self.fine*2+1
        
        return self.stat
        
    def ionstate(self):
        """
        Method to set ionization state by name
        """
        self.ioniz = 0
        m = re.search(r"[IXV]", self.name)
        if m:
            string = self.name[m.start():]
            roman = [['X',10], ['IX',9], ['V',5], ['IV',4], ['I',1]]
            for letter, value in roman:
                while string.startswith(letter):
                    self.ioniz += value
                    string = string[len(letter):]
                    
        return self.ioniz
    
    def mendeleev(self, elem=None):
        if elem is None:
            elem = self.el
        if self.mend is None:
            self.mend = element(elem)

    def atom(self):
        self.mendeleev()        
        return self.mend.atomic_number
    
    def ion_energy(self):
        self.mendeleev()
        return self.mend.ionenergies[self.ioniz]
        
    def mass(self):
        if 'D' not in self.name:
            self.mendeleev()
            return self.mend.mass
        else:
            self.mendeleev('H')
            return self.mend.mass * 2

    def cond_temp(self):
        return elem.CondensTemperature(self.name+'_')
        
    @classmethod
    def read_line(cls, descr):
        """
        Method to read species data from the line
        Used by reading data from file
        """
        s = cls()
        s.readline(descr)
        return s

    def readline(self, descr):
        """
        Method to read species data from the line
        Used by reading data from file
        """
        words = descr.split()
        self.descr = descr

        self.name = words[0]
        i = 1

        if self.name in ['H2', 'HD', 'CO']:

            if len(words) == 4:
                self.J = -1
                self.nu = -1
            else:
                if len(words) == 6:
                    self.nu = int(words[i])
                    i += 1
                elif len(words) == 5:
                    self.nu = 0
                self.J = int(words[i])
                self.stat = (2*self.J+1)*(2*(self.J%2)+1)
                i += 1

        if self.name in ['OI', 'CI', 'SiII']:
            if len(words) == 5:
                self.J = int(words[1])
                i += 1
            else:
                self.J = -1

        self.get_ioniz()
        self.col = a(float(words[i]), float(words[i+1]), float(words[i+2]))
        

class line():
    """
    General class for working with absorption lines
    """
    def __init__(self, name, l, f, g, logN=None, b=None, z=0, nu_u=None, j_u=None, nu_l=None, j_l=None, descr='', ref=''):
        self.name = name
        self.wavelength = [float(l)]
        self.oscillator = [float(f)]
        self.gamma = [float(g)]
        self.logN = logN
        self.b = b
        self.z = z
        self.nu_u = nu_u
        self.j_u = j_u
        self.nu_l = nu_l
        self.j_l = j_l
        self.descr = descr
        self.ref = [ref]
        self.band = ''

    def add(self, l, f, g, ref=''):
        self.wavelength.append(l)
        self.oscillator.append(f)
        self.gamma.append(g)
        self.ref.append(ref)

    def _with_ref(func):
        @wraps(func)
        def wrapped(inst, *args, **kwargs):
            ind = inst.ind(*args, **kwargs)
            if ind is not None:
                return func(inst)[inst.ind(*args, **kwargs)]
            else:
                return func(inst)[0]

        return wrapped

    def ind(self, ref=None):
        if ref is None:
            return 0
        else:
            if ref in self.ref:
                return self.ref.index(ref)
            else:
                return None

    @_with_ref
    def l(self, ref=None):
        return self.wavelength

    @_with_ref
    def f(self, ref=None):
        return self.oscillator

    @_with_ref
    def g(self, ref=None):
        return self.gamma

    def set_Ju(self):
        d = {'O': -2, 'P': -1, 'Q': 0, 'R': 1, 'S': 2}
        self.j_u = self.j_l + d[self.rot]
    
    def __repr__(self):
        if any([ind in self.name for ind in ['H2', 'HD', 'CO']]):
            d = {-2: 'O', -1: 'P', 0: 'Q', 1: 'R', 2: 'S'}
            return '{0} {1}{2}-{3}{4}{5}'.format(self.name, self.band, self.nu_u, self.nu_l, d[self.j_u - self.j_l], self.j_l)
        else:
            return self.name + ' ' + str(self.l())[:str(self.l()).find('.')]
    
    def __str__(self):
        if any([ind in self.name for ind in ['H2', 'HD', 'CO']]) and self.j_l is not None:
            d = {-2: 'O', -1: 'P', 0: 'Q', 1: 'R', 2: 'S'}
            return '{0} {1}{2}-{3}{4}{5}'.format(self.name, self.band, self.nu_u, self.nu_l, d[self.j_u-self.j_l], self.j_l)
        else:
            return self.name + ' ' + str(self.l())[:str(self.l()).find('.')+3]
    
    def __eq__(self, other):
        if str(self) == str(other) and self.z == self.z: # and self.l() == other.l():
            return True
        else:
            return False

class HIlist(list):
    """
    Class specify sets of HI absorption lines
    """
    def __init__(self):
        data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__))+r'/data/HI_data.dat', skip_header=2, names=True)   
        for d in data:
            if d['u_l'] == 1:
                name = r'HI Ly-\alpha'
            elif d['u_l'] == 2:
                name = r'HI Ly-\beta'
            elif d['u_l'] == 3:
                name = r'HI Ly-\gamma'
            elif d['u_l'] == 4:
                name = r'HI Ly-\delta'
            else:
                name = 'HI Ly-'+str(d['u_l'])
            self.append(line(name, d['lambda'], d['f'], d['gamma']))    
        
    @classmethod
    def HIset(cls, n=None):
        sample = cls()
        if n == None:
            n = len(sample)
        if n > len(sample):
            raise(IndexError, 'Value n should not exceed 30')
            n = len(sample)
        return sample[:n]
        
         
class atomicData(OrderedDict):
    """
    Class read and specify the sets of various atomic lines
    """
    def __init__(self):
        super().__init__()
        #self.readdatabase()

    def list(self, els=None, linelist=None):
        lines = []

        if els is not None and isinstance(els, str):
            els = [els]
        if els is None and linelist is None:
            els = self.data.keys()
        if els is None and linelist is not None:
            els = np.unique([l.split()[0] for l in linelist])

        if self.data is None:
            if els is not None:
                for e in els:
                    if e in self.keys():
                        for l in self[e].lines:
                            lines.append(l)
                    else:
                        print(e, 'is not found in atomic data')
            else:
                for e in self.values():
                    for l in e.lines:
                        if linelist is None or str(l) in linelist:
                            lines.append(l)

        else:
            for e in els:
                if e in self.data.keys():
                    for i, ref in enumerate(self.data[e]['ref']):
                        l = self.data[e]['lines'][str(i)][0]
                        l = line(e, l[0], l[1], l[2], ref=l[3])
                        for attr in ['j_l', 'nu_l', 'j_u', 'nu_u']:
                            if 'None' not in ref[attr]:
                                setattr(l, attr, int(ref[attr]))
                        if 'None' not in ref['band']:
                            setattr(l, 'band', ref['band'])
                        if linelist is None or any([str(l) in lin or lin in str(l) for lin in linelist]):
                            lines.append(l)

        return lines

    def set_specific(self, linelist):
        s = []
        for el in linelist:
            #print(el)
            for line in self:
                if str(line)[:len(el)] == el:
                    s.append(line)
                    break
            else:
                print(el, 'is not found in atomic data')
        return s

    def read_DLA(self):
        with open(os.path.dirname(os.path.realpath(__file__))+r'/data/DLA.dat', newline='') as f:
            f.readline()
            n = int(f.readline())
            data = f.readlines()
        for d in data:
            words = d.split()
            self.append(line(words[0], words[1], words[2], words[3]))
        #print(l[-1])

    def readMorton(self):
        with open('data/Morton2003.dat', 'r') as f:
            ind = 0
            while True:
                l = f.readline()
                if l == '':
                    break
                if ind == 1:
                    name = l.split()[0].replace('_', '')
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
                        if name not in self.keys():
                            self[name] = e(name)
                            self[name].lines = []
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
                                    lin = line(name, float(l[19:29]), float(l[79:88]), float(l[59:68]), ref='Morton2003')
                                    if lin not in self[name].lines:
                                        self[name].lines.append(lin)
                                    else:
                                        self[name].lines[self[name].lines.index(lin)].add(float(l[19:29]), float(l[79:88]), float(l[59:68]), ref='Morton2003')
                    ind = 0

                if l[:3] == '***':
                    ind = 1

    def readCashman(self):
        with open('data/Cashman2017.dat', 'r') as f:
            for l in f:
                name = ''.join(l[3:10].split())
                if name not in self.keys():
                    self[name] = e(name)
                    self[name].lines = []
                lam = float(l[90:99]) if l[78:89] else float(l[78:89])
                lin = line(name, lam, float(l[105:113]), 1e+8, ref='Cashman2017')
                if lin not in self[name].lines:
                    self[name].lines.append(lin)
                else:
                    self[name].lines[self[name].lines.index(lin)].add(lam, float(l[105:113]), 1e+8, ref='Cashman2017')

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
        x = np.genfromtxt(os.path.dirname(os.path.realpath(__file__))+r'/data/H2/energy_X.dat', comments='#', unpack=True)

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
            self.read_maleccat(n=[j_l])

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
                mask = np.logical_and(x[0] <= nu, np.logical_and(x[1] >= j[0], x[1] <= j[-1]))
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
                        print(line, l1.f()/line.f(), line.f(), l1.f())
            input()
        else:
            out = open(r'C:/Users/Serj/Desktop/H2MalecCat_comparison.dat', 'w')
            with open(r'C:/science/python/spectro/data/H2MalecCat.dat', 'r') as f:
                for line in f.readlines()[1:]:
                    m = line.split()
                    for l1 in self['H2j' + m[4]].lines:
                        if l1.nu_u == int(m[2]) and m[3] in str(l1) and m[1] in str(l1):
                            #out.write(line[:54] + '{0:12.10f}   {1:6.1e} '.format(l1.f, l1.g) + line[77:])
                            out.write(line[:-1] + '{0:12.8f}   {1:6.1e}  {2:6.2f}\n'.format(l1.f(), l1.g(), l1.f()/float(m[8])))
            out.close()

    def readHD(self):
        HD = np.genfromtxt(r'data/molec/HD.dat', skip_header=1, usecols=[0,1,2,3,4,5,6], unpack=True, names=True, dtype=None)
        j_u = {'R': 1, 'Q': 0, 'P': -1}
        for j in range(3):
            name = 'HDj'+str(j)
            self[name] = e(name)
            self[name].lines = []
            mask = HD['level'] == j
            for l in HD[mask]:
                #name = 'HD ' + l[1].decode('UTF-8') + str(l[2]) + '-0' + l[3].decode('UTF-8') + '(' + str(int(l[0])) + ')'
                self[name].lines.append(line(name, l['lambda'], l['f'], l['gamma'], ref='', j_l=l['level'], nu_l=0, j_u=l['level'] + j_u[l['PorR'].decode('UTF-8')], nu_u=l['band']))
                self[name].lines[-1].band = l['LW'].decode('UTF-8')

    def readCO(self):
        CO = np.genfromtxt(r'data/CO_data_Dapra.dat', skip_header=1, unpack=True, names=True, dtype=None)
        for i in np.unique(CO['level']):
            name = 'COj'+str(i)
            self[name] = e(name)
            self[name].lines = []
            mask = CO['level'] == i
            for l in CO[mask]:
                self[name].lines.append(line(name, l['lambda'], l['f'], l['gamma'], ref='', j_l=i, nu_l=0, j_u=i + l['PQR'], nu_u=l['band']))
                self[name].lines[-1].band = l['name'].decode()

    def readHF(self):
        HF = np.genfromtxt(r'data/molec/HF.dat', skip_header=1, unpack=True, dtype=None)
        self['HF'] = e('HF')
        for l in HF:
            self['HF'].lines.append(line(l[4].decode('UTF-8'), l[6], l[7], l[8], ref='', j_l=l[0], nu_l=l[1], j_u=l[2], nu_u=l[3]))

    def makedatabase(self):
        f = h5py.File('data/atomic.hdf5', 'w')
        self.readMorton()
        #self.readCashman()
        self.readH2(j=[0,1,2,3,4,5,6])
        self.readHD()
        self.readCO()
        self.readHF()
        self.read_Molecular()
        self.read_EmissionSF()
        for el in self.keys():
            grp = f.create_group(el)
            dt = h5py.special_dtype(vlen=str)
            ds = grp.create_dataset('ref', shape=(len(self[el].lines),),
                                    dtype=np.dtype([('l', float), ('ind', int), ('descr', dt), ('j_l', dt),
                                                       ('nu_l', dt), ('j_u', dt), ('nu_u', dt), ('band', dt)]))
            lines = grp.create_group('lines')
            #ref = np.empty(len(self[el].lines), dtype=[('l', float), ('ind', int), ('descr', 'S')])
            for i, l in enumerate(self[el].lines):
                print(str(l), l.band)
                ds[i] = (l.l(), i, l.descr, str(l.j_l), str(l.nu_l), str(l.j_u), str(l.nu_u), str(l.band))
                lin = lines.create_dataset(str(i), shape=(len(l.ref),), dtype = np.dtype([('l', float), ('f', float), ('g', float), ('ref', dt)]))
                for k in range(len(l.ref)):
                    lin[k] = (l.wavelength[k], l.oscillator[k], l.gamma[k], l.ref[k])
                #ref[i] = (l.l(), i, l.descr, str(l.j_l), str(l.nu_l), str(l.j_u), str(l.nu_u)) #.encode("ascii", "ignore"))

    def readdatabase(self):
        self.data = h5py.File(os.path.dirname(os.path.realpath(__file__)) + r'/data/atomic.hdf5', 'r')
        for el in self.data.keys():
            self[el] = e(el)

    def read_maleccat(self, n=-1):
        """
        read Malec calalogue 2010 data for H2

        parameters:
            - n    : if list - specify rotational levels to read
                     if int - read J_l<=n
        """
        with open(os.path.dirname(os.path.realpath(__file__)) + r'/data/H2/H2MalecCat.dat', newline='') as f:
            f.readline()
            data = f.readlines()
        for d in data:
            words = d.split()
            if words[4] != '7':
                if (isinstance(n, list) and int(words[4]) in n) or (
                            isinstance(n, int) and int(words[4]) <= n) or n == -1:
                    l = line('H2j' + str(int(words[4])), words[5], words[8], words[9])
                    l.band = words[1]
                    l.nu_u = int(words[2])
                    l.nu_l = 0
                    l.rot = words[3]
                    l.j_l = int(words[4])
                    l.set_Ju()
                    self['H2j'+words[4]].lines.append(l)

    def read_Molecular(self):
        with open(os.path.dirname(os.path.realpath(__file__))+r'/data/Molecular_data.dat', newline='') as f:
            for i in range(3):
                f.readline()
            data = f.readlines()
        for d in data:
            words = d.split()
            if words[0] not in self:
                self[words[0]] = e(words[0])
            self[words[0]].lines.append(line(words[0], words[1], words[2], words[3], descr=' '.join(words[4:])))

    def read_EmissionSF(self):
        with open(os.path.dirname(os.path.realpath(__file__))+r'/data/EmissionSFLines.dat', newline='') as f:
            for i in range(3):
                f.readline()
            data = f.readlines()
        for d in data:
            words = d.split()
            if words[0] not in self:
                self[words[0]] = e(words[0])
            self[words[0]].lines.append(line(words[0], words[1], words[2], words[3], descr=' '.join(words[4:])))

    def Molecular_list(self):
        data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/data/Molecular_data.dat', skip_header=3, usecols=(0), dtype=(str))
        return self.list(np.unique(data))

    def DLA_list(self, lines=True):
        linelist = ['HI 1215',
                    'HI 1025',
                    'HI 972',
                    'HI 949',
                    'HI 937',
                    'HI 930',
                    'HI 926',
                    'HI 923',
                    'HI 920',
                    'HI 919',
                    'HI 918',
                    'HI 917',
                    'HI 916',
                    'HI 915',
                    'AlII 1670',
                    'AlIII 1854',
                    'AlIII 1862',
                    'CI 1276',
                    'CI 1277',
                    'CI 1280',
                    'CI 1328',
                    'CI 1560',
                    'CI 1656',
                    'CII 1036',
                    'CIII 977',
                    'CIV 1548',
                    'CIV 1550',
                    'CrII 2026.2',
                    'CrII 2056',
                    'CrII 2062.23',
                    'CrII 2066',
                    'FeII 1062.15',
                    'FeII 1063.1',
                    'FeII 1063.97',
                    'FeII 1081',
                    'FeII 1096',
                    'FeII 1121.97',
                    'FeII 1125',
                    'FeII 1133.6',
                    'FeII 1142.36',
                    'FeII 1143.2',
                    'FeII 1144',
                    'FeII 1260.53',
                    'FeII 1608',
                    'FeII 2249',
                    'FeII 2260',
                    'FeII 2344',
                    'FeII 2374',
                    'FeII 2382',
                    'FeII 2586',
                    'FeII 2600',
                    'FeIII 1122.53',
                    'MgI 2026.4',
                    'MgI 2852',
                    'MgII 2796',
                    'MgII 2803',
                    'MnII 2576',
                    'NI 1134.16',
                    'NI 1134.41',
                    'NI 1134.98',
                    'NI 1199.54',
                    'NI 1200.22',
                    'NI 1200.70',
                    'NII 1083.99',
                    'NIII 989.79',
                    'NV 1238',
                    'NV 1242',
                    'NiII 1317',
                    'NiII 1370',
                    'NiII 1454',
                    'NiII 1709',
                    'NiII 1741',
                    'NiII 1751',
                    'OI 988.57',
                    'OI 988.65',
                    'OI 988.77',
                    'OI 1039',
                    'OI 1302',
                    'OVI 1031',
                    'OVI 1037',
                    'PII 1152',
                    'SII 1250',
                    'SII 1253',
                    'SII 1259.51',
                    'SIII 1012'
                    'SIII 1190.21',
                    'SIV 1062.66',
                    'SiII 989.87',
                    'SiII 1020',
                    'SiII 1190.41',
                    'SiII 1193',
                    'SiII 1260.42',
                    'SiII 1304',
                    'SiII 1526',
                    'SiII 1808',
                    'SiIII 1206',
                    'SiIV 1393',
                    'SiIV 1402',
                    'TiII 1910.61',
                    'TiII 1910.95',
                    'CII 1334.53',
                    'CII* 1335.71',
                    'CII 1334.53',
                    'ZnII 2026.13',
                    'ZnII 2062.66',
                    'CII 1036',
                    ]

        if lines:
            return self.list(linelist=linelist)
        else:
            return linelist

    def DLA_major_list(self, lines=True):
        linelist = ['HI 1215',
                    'HI 1025',
                    'HI 972',
                    'HI 949',
                    'HI 937',
                    'HI 930',
                    'HI 926',
                    'HI 923',
                    'HI 920',
                    'HI 919',
                    'HI 918',
                    'HI 917',
                    'HI 916',
                    'HI 915',
                    'CIII 977',
                    'OI 988.57',
                    'OI 988.65',
                    'OI 988.77',
                    'SiII 989.87',
                    'NIII 989.79',
                    'OVI 1031',
                    'CII 1036',
                    'OVI 1037',
                    'OI 1039',
                    'FeII 1063.1',
                    'NII 1083.99',
                    'FeII 1096',
                    'FeII 1121.97',
                    'FeII 1125',
                    'NI 1134.16',
                    'NI 1134.41',
                    'NI 1134.98',
                    'FeII 1144',
                    'SiII 1190.41',
                    'SiII 1193',
                    'NI 1199.54',
                    'NI 1200.22',
                    'NI 1200.70',
                    'SiIII 1206',
                    'SiII 1260.42',
                    'FeII 1260.53',
                    'OI 1302',
                    'SiII 1304',
                    'CII 1334',
                    'SiIV 1393',
                    'SiIV 1402',
                    'SiII 1526',
                    'CIV 1548',
                    'CIV 1550',
                    'FeII 1608',
                    'AlII 1670',
                    'FeII 2344',
                    'FeII 2374',
                    'FeII 2382',
                    'FeII 2586',
                    'FeII 2600',
                    'MgII 2796',
                    'MgII 2803',
                    'MgI 2852',
                    ]
        if lines:
            return self.list(linelist=linelist)
        else:
            return linelist

    def DLA_SDSS_H2_list(self, lines=True):
        linelist = ['SiII 1260',
                    'OI 1302',
                    'SiII 1304',
                    'CII 1334',
                    'AlII 1670',
                    'FeII 2344',
                    'ZnII 2026',
                    'CIV 1548',
                    ]
        if lines:
            return self.list(linelist=linelist)
        else:
            return linelist

    def EmissionSF_list(self, lines=True):
        linelist = ['HI 6562',
                    'HI 4861',
                    'HI 4340',
                    'HI 4101',
                    'HI 3970',
                    'HI 3889',
                    'OIII 5006',
                    'OIII 4958',
                    'OII 3726',
                    'OII 3728',
                    'NII 6548',
                    'NII 6583',
                    ]
        if lines:
            return self.list(linelist=linelist)
        else:
            return linelist

    @classmethod
    def DLA(cls, lines=True):
        print('DLA')
        s = cls()
        s.readdatabase()
        return s.DLA_list(lines=lines)

    @classmethod
    def DLA_major(cls, lines=True):
        print('DLA major')
        s = cls()
        s.readdatabase()
        return s.DLA_major_list(lines)


    @classmethod
    def DLA_SDSS_H2(cls, lines=True):
        s = cls()
        s.readdatabase()
        return s.DLA_SDSS_H2_list(lines=lines)

    @classmethod
    def MinorMolecular(cls):
        s = cls()
        s.read_Molecular()
        return s

    @classmethod
    def H2(cls, n=7):
        """
        create instance of H2list from Malec calalogue 2010

        parameters:
            - n    : if list - specify rotational levels to read
                     if int - read J_l<=n
        """
        s = cls()
        s.readdatabase()
        if isinstance(n, int):
            return s.list(['H2j'+str(i) for i in range(n)])
        elif isinstance(n, (list, tuple)):
            return s.list(['H2j' + str(i) for i in n])



def condens_temperature(name):
    """
    return Condensation Temperature (at 50 %) of the element onto dust grain for
    # solar abundance nebula at the pressure 10^-4 bar. (from Lodders 2003)
    parameters:
        - name         :   Name of the species

    return: t
        - t            :   Condensation temperature
    """
    name = e(name).get_element_name()
    d = {
    'Li': 1142,
    'Be': 1452,
    'B': 908,
    'C': 40,
    'N': 123,
    'O': 180,
    'F': 734,
    'Ne': 9.1,
    'Na': 958,
    'Mg': 1336,
    'Al': 1653,
    'Si': 1310,
    'P': 1229,
    'S': 664,
    'Cl': 948,
    'Ar': 47,
    'K': 1006,
    'Ca': 1517,
    'Sc': 1659,
    'Ti': 1582,
    'V': 1429,
    'Cr': 1296,
    'Mn': 1158,
    'Fe': 1334,
    'Co': 1352,
    'Ni': 1353,
    'Cu': 1037,
    'Zn': 726,
    'Ga': 968,
    'Ge': 883,   
    'As': 1065,
    'Se': 697,
    'Br': 546,
    'Kr': 52,
    'Rb': 800,
    'Sr': 1464,
    'Y': 1659,
    'Zr': 1741,
    'Nb': 1559,
    'Mo': 1590,
    'Ru': 1551,
    'Rh': 1392,
    'Pd': 1324,
    'Ag': 996,
    'Cd': 652,
    'In': 536,
    'Sn': 704,
    'Sb': 979,   
    'Te': 709,
    'I': 535,
    'Xe': 68,
    'Cs': 799,
    'Ba': 1455,
    'La': 1578,
    'Ce': 1478,
    'Pr': 1582,
    'Nd': 1602,
    'Sm': 1590,
    'Eu': 1356,
    'Gd': 1659,
    'Tb': 1659, 
    'Dy': 1659,
    'Ho': 1659,
    'Er': 1659,
    'Tm': 1659,
    'Yb': 1487,
    'Lu': 1659,
    'Hf': 1684,
    'Ta': 1573,
    'W': 1789,
    'Re': 1821,
    'Os': 1812,
    'Ir': 1603,
    'Pt': 1408,
    'Au': 1060, 
    'Hg': 252,
    'Tl': 532,
    'Pb': 727,
    'Bi': 746,
    'Th': 1659,
    'U': 1610
    }
    try:
        return d[name] 
    except:
        return None
    
def Asplund2009(name, relative=True):
    d = {
    'H': [12,0,0],
    'He': [10.93,0.01,0.01], #be carefull see Asplund 2009
    'Li': [1.05,0.10,0.10],
    'Be': [1.38,0.09,0.09],
    'B': [2.70,0.20,0.20],
    'C': [8.43, 0.05, 0.05], 
    'N': [7.83,0.05,0.05],
    'O': [8.69,0.05,0.05],
    'F': [4.56,0.30,0.30],
    'Ne': [7.93,0.10,0.10],  #be carefull see Asplund 2009
    'Na': [6.24,0.04,0.04],
    'Mg': [7.60,0.04,0.04],
    'Al': [6.45,0.03,0.03],
    'Si': [7.51,0.03,0.03],
    'P': [5.41,0.03,0.03],
    'S': [7.12,0.03,0.03],
    'Cl': [5.50,0.30,0.30],
    'Ar': [6.40,0.13,0.13],  #be carefull see Asplund 2009
    'K': [5.03,0.09,0.09],
    'Ca': [6.34,0.04,0.04],
    'Sc': [3.15,0.04,0.04],
    'Ti': [4.95,0.05,0.05],
    'V': [3.93,0.08,0.08],
    'Cr': [5.64,0.04,0.04],
    'Mn': [5.43,0.04,0.04],
    'Fe': [7.50,0.04,0.04],
    'Co': [4.99,0.07,0.07],
    'Ni': [6.22,0.04,0.04],
    'Cu': [4.19,0.04,0.04],
    'Zn': [4.56,0.05,0.05],
    'Ga': [3.04,0.09,0.09],
    'Ge': [3.65,0.10,0.10]
    }
    if relative:
        ref = 12
    else:
        ref = 0
        for i in d.values():
            ref += 10**i[0]
        ref = np.log10(ref)

    try:
        x = d[name]
    except KeyError:
        raise Exception(KeyError, 'there is no abundance in Asplund2009 for a given name')

    x[0] = x[0] - ref
    return x

def metallicity(element, logN, logNH, mode=None):
    """
    Return the value of the metallicity using Asplund 2009 data
    parameters:
            - element     : name of the element
            - logN        : column density of element
            - logNH       : column density of HI
            - mode        : if mode == 'Solar' include solar uncertainties in result
    """
    element = e(element).get_element_name()
    if isinstance(logN, a) or isinstance(logNH, a):
        if not isinstance(logNH, a):
            logNH = 10**logNH
        if mode == 'Solar':
            return (logN / logNH / a(Asplund2009(element), 'l')).log()
        else:
            return (logN / logNH / 10**(Asplund2009(element)[0])).log()
    
    if isinstance(logN, float) and isinstance(logNH, float):
        return logN - logNH - Asplund2009(element)[0]


def abundance(element, logNH, me, mode=None):
    """
    Return the value of the abundance of element at specified metallicity using Asplund 2009 data
    parameters:
            - element     : name of the element
            - logNH       : column density of HI
            - me          : metallicity
            - mode        : if mode == 'Solar' include solar uncertainties in result
    return: N
            - N           : abundance of element in log, or a() object
    
    """
    element = e(element).get_element_name()
    if isinstance(me, a) or isinstance(logNH, a):
        if mode == 'Solar':
            return (logNH * me * a(Asplund2009(element), 'l')).log()
        else:
            return (logNH * me * 10**(Asplund2009(element)[0])).log()

    if isinstance(me, float) or isinstance(logNH, float):
        return me + logNH + Asplund2009(element)[0]


def depletion(element, logN, logNref, ref='Zn', mode=None):
    """
    Return the value of the depletion of element using Asplund 2009 data
    parameters:
            - element     : name of the element
            - logN        : column density of element
            - logNH       : column density of reference element for the depletion
            - ref         : name of the reference element, Zn is default
            - mode        : if mode == 'Solar' include solar uncertainties in result
    """
    element = e(element).get_element_name()
    element_ref = e(ref).get_element_name()
    if isinstance(logN, a) or isinstance(logNref, a):
        if mode == 'Solar':
            return (logN / logNref * a(Asplund2009(element_ref), 'l') / a(Asplund2009(element), 'l')).log()
        else:
            return (logN / logNref * 10 ** (Asplund2009(element_ref)[0] - Asplund2009(element)[0])).log()

def doppler(element, turb, kin):
    """
        Return the value doppler parameter for the given species, given turbulence and kinetic temperature 
        parameters:
                - element     : name of the species
                - turb        : turbulence doppler parameter, in km/s
                - kin         : Kinetic temperature, in K
        Output: b
                - b           : doppler parameter in km/s
                
        Example:  doppler('OI', 3.4, 20000)
    """
    return np.sqrt(turb**2 + 0.0164 * kin / e(element).mass())

if __name__ == '__main__':
    if 0:
        d = DLAlist.DLA_SDSS()
        d = DLAlist()
        print(d)    
    
    if 0:
        H2 = H2list.Malec(1)  
        for line in H2:
            print(line)
    if 0:
        HI = HIlist.HIset()
        for line in HI:
            print(line, line.l, line.f, line.g)
    if 0:
        N = a('14.47^{+0.09}_{-0.07}', 'l')
        print(N)
        HI = a('19.88^{+0.01}_{-0.01}', 'l')
        print(metallicity('O_I', N, HI))

    if 0:
        HI = a('19.88^{+0.01}_{-0.01}', 'l')
        me = a('-1.2^{+0.1}_{-0.1}', 'l')
        print(abundance('OI', HI, me))

    if 0:
        A = atomicData()
        #A.readCO()
        #print(A.list('SiII'))
        A.makedatabase()
        #A.makedatabase()

    if 1:
        A = atomicData()
        A.readH2Abgrall(j=[0,1,2,3,4,5,6,7])
        print(A.keys())
        #A.readH2(j=[0,1,2,3,4,5,6,7])
        A.compareH2()