import re
import sys, os
sys.path.append('D:/science/python')
from a_unc import a
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
        if len(args) == 2:
            if isinstance(args[1], 'str'):
                self.col = a(args[1])
            elif isinstance(args[1], a):
                self.col = args[1]
        elif len(args) == 4:
            self.col = a(args[1], args[2], args[3])
        
        self.nu = None
        self.J = None
        self.b = None
        
        for k in kwargs.items():
            if k[0] == 'b' and isinstance(k[1], (list, tuple)):
                setattr(self, k[0], a(k[1]))
            else:
                setattr(self, k[0], k[1])
        
        # refresh representation for molecules
        self.name = self.__repr__()
        
        self.mend = None
        
        self.fine = self.name.count('*')
        
        self.statw()
        self.ionstate()
        
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
        return self.__repr__() + self.col
    
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
    
    def mendeleev(self):
        if self.mend is None:
            self.mend = element(self.el)
            
    def atom(self):
        self.mendeleev()        
        return self.mend.atomic_number
    
    def ion_energy(self):
        self.mendeleev()
        return self.mend.ionenergies[self.ioniz]
        
    def mass(self):
        self.mendeleev()
        return self.mend.mass

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