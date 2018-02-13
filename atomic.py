import numpy as np
import re
import os
from .a_unc import a
from mendeleev import element

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
            if 'f' in kwargs.keys():
                self.col = a(args[1], args[2], args[3], kwargs['f'])
            else:
                self.col = a(args[1], args[2], args[3], )
        
        self.nu = None
        self.J = None
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
                st +=  '_' + str(self.J)
            if self.nu is not None:
                st += '_' + str(self.nu)
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
    def __init__(self, name, l, f, g, logN=None, b=None, z=0, nu_u=None, j_u=None, nu_l=None, j_l=None, descr=None, ref=''):
        self.name = name
        self.l = float(l)
        self.f = float(f)
        self.g = float(g)
        self.logN = logN
        self.b = b
        self.z = z
        self.nu_u = nu_u
        self.j_u = j_u
        self.nu_l = nu_l
        self.j_l = j_l
        self.descr = descr
        self.ref = ref
        self.band = ''

    def set_Ju(self):
        d = {'O': -2, 'P': -1, 'Q': 0, 'R': 1, 'S': 2}
        self.j_u = self.j_l + d[self.rot]
    
    def __repr__(self):
        if any([ind in self.name for ind in ['H2', 'HD', 'CO']]):
            d = {-2: 'O', -1: 'P', 0: 'Q', 1: 'R', 2: 'S'}
            return '{0} {1}{2}-{3}{4}{5}'.format(self.name, self.band, self.nu_u, self.nu_l, d[self.j_u - self.j_l], self.j_l)
        else:
            return self.name + ' ' + str(self.l)[:str(self.l).find('.')]
    
    def __str__(self):
        if any([ind in self.name for ind in ['H2', 'HD', 'CO']]):
            d = {-2: 'O', -1: 'P', 0: 'Q', 1: 'R', 2: 'S'}
            return '{0} {1}{2}-{3}{4}{5}'.format(self.name, self.band, self.nu_u, self.nu_l, d[self.j_u-self.j_l], self.j_l)
        else:
            return self.name + ' ' + str(self.l)[:str(self.l).find('.')+3]
    
    def __eq__(self, other):
        if str(self) == str(other) and self.l == other.l and self.z == self.z:
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
        
         
class AtomicList(list):
    """
    Class read and specify the sets of various atomic lines
    """
    def __init__(self):
        pass

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
                                    self.append(line(name, float(l[19:29]), float(l[79:88]), float(l[59:68]), ref='Morton2003'))
                    ind = 0

                if l[:3] == '***':
                    ind = 1

    def read_maleccat(self, n=-1):
        """
        read Malec calalogue 2010 data for H2

        parameters:
            - n    : if list - specify rotational levels to read
                     if int - read J_l<=n
        """
        with open(os.path.dirname(os.path.realpath(__file__)) + r'/data/H2MalecCat.dat', newline='') as f:
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
                    self.append(l)

    def read_Molecular(self):
        with open(os.path.dirname(os.path.realpath(__file__))+r'/data/Molecular_data.dat', newline='') as f:
            for i in range(3):
                f.readline()
            data = f.readlines()
        for d in data:
            words = d.split()
            self.append(line(words[0], words[1], words[2], words[3], descr=' '.join(words[4:])))

    @classmethod
    def DLA(cls, lines=True):
        print('DLA')
        s = cls()
        s.readMorton()
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
            return s.set_specific(linelist)
        else:
            return linelist

    @classmethod
    def DLA_major(cls, lines=True):
        print('DLA major')
        s = cls()
        s.readMorton()
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
            return s.set_specific(linelist)
        else:
            return linelist

    @classmethod
    def DLA_SDSS_H2(cls, lines=True):
        s = cls()
        s.readMorton()
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
            return s.set_specific(linelist)
        else:
            return linelist

    @classmethod
    def MinorMolecular(cls):
        s = cls()
        s.read_Molecular()
        return s

    @classmethod
    def Malec(cls, n=7):
        """
        create instance of H2list from Malec calalogue 2010

        parameters:
            - n    : if list - specify rotational levels to read
                     if int - read J_l<=n
        """
        s = cls()
        s.read_maleccat(n)
        return s

    

def condens_temperature(name):
    """
    return Condensation Temperature (at 50 %) of the element onto dust grain for
    # solar abundance nebula at the pressure 10^-4 bar. (from Lodders 2003)
    parameters:
        - name         :   Name of the species

    return: t
        - t            :   Condensation temperature
    """
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
        if mode == 'Solar':
            return logN / logNH / a(Asplund2009(element), 'l')
        else:
            return logN / logNH / 10**(Asplund2009(element)[0])
    
    if isinstance(logN, float) or isinstance(logNH, float):
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
            return logN / logNref * a(Asplund2009(element_ref), 'l') / a(Asplund2009(element), 'l')
        else:
            return logN / logNref * 10 ** (Asplund2009(element_ref)[0] - Asplund2009(element)[0])

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
    if 1:
        HI = HIlist.HIset()
        for line in HI:
            print(line, line.l, line.f, line.g)
    if 0:
        N = a('14.47^{+0.09}_{-0.07}', 'l')
        print(N)
        HI = a('19.88^{+0.01}_{-0.01}', 'l')
        print(metallicity('O_I', N, HI))

    if 1:
        HI = a('19.88^{+0.01}_{-0.01}', 'l')
        me = a('-1.2^{+0.1}_{-0.1}', 'l')
        print(abundance('OI', HI, me))