import numpy as np
from collections import OrderedDict

class data(float):
    def __new__(cls, val, ref='', units=''):
        return float.__new__(cls, val)

    def __init__(self, val, ref='', units=''):
        self.ref = ref
        self.units = units

    def __str__(self):
        return '{0:f} [{1:s}] u={2:s}'.format(self, self.ref, self.units)

    def __repr__(self):
        return self.__str__()

class level():
    def __init__(self, J=None, nu=None, E=None, ref=None, units=None):
        self.J = int(J)
        self.nu = int(nu)
        self.E = data(E, ref, units)
        self.vE = {}
        self.add(energy=self.E, ref=ref)

    def add(self, energy=None, ref=None, units=None):
        self.vE[ref] = data(energy, ref=ref, units=units)

    def __str__(self):
        return "J={:d} nu={:d} E={:f}".format(self.J, self.nu, self.E)

    def __repr__(self):
        return self.__str__()

class state():
    def __init__(self, name, filename=None):
        self.name = name
        if filename is not None:
            self.readdata(filename)
        self.levels = np.empty((40, 40), dtype=np.object)

    def level(self, J, nu):
        return self.levels[J, nu]

    def readCloudy(self):
        file = 'data/H2/energy_'+self.name+'.dat'
        with open(file) as f:
            ref = f.readline().replace('#', '').strip()
        data = np.genfromtxt(file, comments='#')
        for d in data:
            l = level(J=d[1], nu=d[0], E=d[2], ref=ref, units='cm-1')
            self.levels[l.J, l.nu] = l

class H2data(OrderedDict):
    def __init__(self):
        super().__init__()
        self.readMorton()
        self.readH2Abgrall(j=7)
        #self.compareH2()

    def readMorton(self):
        with open('data/Morton2003.dat', 'r') as f:
            ind = 0
            while True:
                l = f.readline()
                if l == '':
                    break
                if ind == 1:
                    name = l.split()[0].replace('_', '')
                    self[name] = e(name)
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
                            self[name] = e(name)
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
                                    self[name].lines.append(line(name, float(l[19:29]), float(l[79:88]), float(l[59:68]), ref='Morton2003'))
                    ind = 0

                if l[:3] == '***':
                    ind = 1

    def readH2Abgrall(self, nu=0, j=5, energy=None):
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
        x = np.genfromtxt(r'data\H2\energy_X.dat', comments='#', unpack=True)
        B = np.genfromtxt(r'data\H2\energy_B.dat', comments='#', unpack=True)
        Cp = np.genfromtxt(r'data\H2\energy_C_plus.dat', comments='#', unpack=True)
        Cm = np.genfromtxt(r'data\H2\energy_C_minus.dat', comments='#', unpack=True)
        B_t = np.genfromtxt(r'data\H2\transprob_B.dat', comments='#', unpack=True)
        Cp_t = np.genfromtxt(r'data\H2\transprob_C_plus.dat', comments='#', unpack=True)
        Cm_t = np.genfromtxt(r'data\H2\transprob_C_minus.dat', comments='#', unpack=True)
        e2_me_c = const.e.gauss.value ** 2 / const.m_e.cgs.value / const.c.cgs.value
        if energy is None:
            mask = np.logical_and(x[0] <= nu, x[1] <= j)
        else:
            mask = x[2] < energy
        x = x[:,mask]
        x = np.transpose(x)
        fout = open(r'data\H2\lines.dat', 'w')
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
        mal = H2list.Malec()
        for line in mal:
            for l1 in self['H2j'+str(line)[-1]].lines:
                #print(line)
                #print(l1)
                if str(line) == str(l1): #and np.abs(l1.f/line.f-1) > 0.2:
                    print(line, l1.f/line.f, line.f, l1.f)
        input()

if __name__ == '__main__':

    X = state('X')
    X.readCloudy()
    B = state('B')
    B.readCloudy()
    C_minus = state('C_minus')
    C_minus.readCloudy()
    C_plus = state('C_plus')
    C_plus.readCloudy()
    for j in range(0, 6):
        print(X.levels[j+2, 0].E - X.levels[j, 0].E)