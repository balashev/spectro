if __name__ == '__main__':
    import sys
    sys.path.append('C:/science/python')
    from spectro.a_unc import a
else:
    from .a_unc import a


class species():
    """
    Class for species. 
        Sets data for species 
    dependences mendeleev package
    """
    def __init__(self, name='', col=0, plus=0, minus=0):
        self.name = name
        self.elem = ''
        self.nu = -1
        self.J = -1
        self.stat = -1
        self.col = a(col,plus,minus)
        self.mod = 0
        self.ioniz = 0
        self.descr = self.name
        if col != 0:
            self.descr += ' '+str(col)
        # for line modeling:
        self.logN = 0
        self.b = 0
        self.b_ind = 0
        
    def __repr__(self):
        return self.descr
    
    def __str__(self):
        return self.descr
        
    def get_ioniz(self):
        """
        Method to set ionization state by name
        """
        if self.name not in ['H2', 'HD', 'CO']:
            self.ioniz = 0
            m = re.search(r"[IXV]", self.name)
            if m:
                self.elem = self.name[:m.start()]
                string = self.name[m.start():]
                roman = [['X',10], ['IX',9], ['V',5], ['IV',4], ['I',1]]
                for letter, value in roman:
                    while string.startswith(letter):
                        self.ioniz += value
                        string = string[len(letter):]

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

class DLA():
    """
    Class for DLA absorber
    
    """
    def __init__(self, z):
        self.z = float(z)
        self.el = []
        
    def standart(self):
        self.add(species('HI', 20.3))
        self.add(species('DI', 20.3-4.55))
    
    def add(self, spec):
        self.el.append(spec)
        self.__setattr__(spec.name, spec)

class HI():
    """
    Simple class for HI absorber
    
    """
    def __init__(self, z, logN=20.3):
        self.z = z
        self.logN = logN
    
    def __repr__(self):
        return str(self.z)

def vel_offset(x, line):
    R = x / line
    return (R ** 2 - 1) / (R ** 2 + 1) * 299792.46
    # return (x / (1 + z_qso) / lines['CaII_0'][0] - 1) * 299792.458

def deltaV(z_em, z_abs):
    """
    return the doppler shift, between emission and absorption redshift
    :param z_em: emission redshift
    :param z_abs: absorption redshift
    :return: deltaV in km/s
    """
    R = (1 + z_abs) / (1 + z_em)
    return (R**2 - 1) / (R**2 + 1) * 299792.46

if __name__ == '__main__':
    print(deltaV(2.811124, 2.811156))
    print(deltaV(2.811124, 2.811126))
    print(deltaV(2.811124, 2.811132))
    