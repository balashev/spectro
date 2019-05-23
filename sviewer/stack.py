import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from .utils import smooth

class catalog:
    def __init__(self, x, name='cat', DLA=None, num=100, sigma=0.1):
        self.x = x
        self.name = name
        self.DLA = DLA
        self.sigma = sigma
        self.f, self.e, self.mask = np.zeros([num, len(x)]), np.zeros([num, len(x)]), np.zeros([num, len(x)], dtype=bool)
        self.norm, self.w, self.inds = np.zeros([num]), np.zeros([num]), np.arange(num)
        self.capacity = num
        self.size = 0
        self.median = None
        self.dla_mask = None
        self.set_mask()

    def set_mask(self, z=None, N=None):
        self.smask = {}
        if z is not None:
            self.smask['z_abs'] = z
        if N is not None:
            self.smask['NHI'] = N

    def calc_mask(self):
        self.m = np.ones(len(self.DLA), dtype=bool)
        for i, d in enumerate(self.DLA):
            for k, v in self.smask.items():
                if not v[0] < d[k] < v[1]:
                    self.m[i] *= False

    def load_median(self, filename):
        f = h5py.File(filename.replace('.hdf5', '') + '.hdf5', 'r')
        self.median = interp1d(f['wavelength'], f['flux'], bounds_error=False, fill_value=0, assume_sorted=True)

    def load_dla_mask(self, filename, kind='calc', sigma=2.0, expand=2):
        if kind == 'calc':
            x, f, e = np.genfromtxt(filename, unpack=1)
            m = (1 - f) > (sigma * e)
            # expand in both sides by <expand> pixels:
            mask = np.copy(m)
            for p in range(-expand - 1, expand + 1):
                if p < 0:
                    m1 = np.insert(m[:p], [0] * np.abs(p), 0)
                if p > 0:
                    m1 = np.insert(m[p:], [m.shape[0] - p] * p, 0)
                mask = np.logical_or(mask, m1)
                mask = np.logical_and(mask, np.logical_and(f > 0.4, f < 1.0))
            self.dla = interp1d(x, f, bounds_error=False, fill_value=1, assume_sorted=True)

        if kind == 'read':
            x, mask = np.genfromtxt(filename, usecols=(0, 1))

        #mask = np.asarray(mask, dtype=bool)
        self.dla_mask = interp1d(x, mask, bounds_error=False, fill_value=0, assume_sorted=True)
        #plt.plot(x, self.dla(x))
        #plt.plot(x, self.dla_mask(x), 'o')
        #splt.show()
        #self.dla_mask = np.asarray(interp1d(x, mask, bounds_error=False, fill_value=0, assume_sorted=True)(self.x), dtype=bool)
        #print(self.x[self.dla_mask])

    def add_interpolated(self, f, e, mask, ind=0):
        if self.size == self.capacity:
            print(self.size, self.capacity)
            self.capacity *= 2
            #newdata = np.zeros((self.capacity, len(self.x)))
            #newdata[:self.size, :] = self.data
            #self.data = newdata

        m_norm = ((1300 < self.x) * (self.x < 1383)) | ((1408 < self.x) * (self.x < 1500))
        #m = (f[m] != 0) * (np.isfinite(f[m]))
        m = ~mask[m_norm]
        N = np.sum(m)
        if N > 0 and N > np.sum(m_norm) * 0.8:
            n = np.mean(np.ma.array(f[m_norm], mask=~m))
            s = n / np.sqrt(np.mean(np.ma.array(e[m_norm], mask=~m)**(2)))
            if s >= 1:
                self.f[self.size], self.e[self.size], self.mask[self.size] = f[:], e[:], mask[:]
                self.norm[self.size], self.w[self.size], self.inds[self.size] = n, 1 / (s**(-2) + self.sigma**2), ind
                #print(self.norm[self.size], self.w[self.size])
                self.size += 1
            else:
                #print('s less 1:', s)
                pass
        else:
            #print('missed:', N, np.sum(m_norm))
            pass

    def add(self, x, f, e, mask, ind=0, z=0.0, corr=None, add_mask=None, remove_ouliers=True):
        f, e, m = interp1d(x[mask], f[mask], bounds_error=False, fill_value=0, assume_sorted=True), \
                  interp1d(x[mask], 1 / e[mask], bounds_error=False, fill_value=0, assume_sorted=True), \
                  interp1d(x, mask, bounds_error=False, fill_value=0, assume_sorted=True)

        fl, el = f(self.x * (1 + z)), e(self.x * (1 + z))
        if corr is not None:
            fl, el = fl / corr, el / corr

        mask = np.logical_or(~np.isfinite(el), np.logical_or(~np.isfinite(fl), m(self.x * (1 + z)) == 0))

        if remove_ouliers:
            dla_mask = self.dla_mask(self.x) == 0
            cont = interp1d(self.x[dla_mask], smooth(fl[dla_mask], window_len=101, window='hanning', mode='same'), bounds_error=False, fill_value=1, assume_sorted=True)(self.x)
            econt = interp1d(self.x[dla_mask], smooth(el[dla_mask], window_len=101, window='hanning', mode='same'), bounds_error=False, fill_value=1, assume_sorted=True)(self.x)
            mask = np.logical_or(mask, np.logical_or(fl / cont > 2 + 3 * econt,  fl / cont < - 3 * econt))
            #print(np.sum(mask))
            if 0:
                plt.plot(self.x, fl)
                plt.plot(self.x, mask, 'o')
                plt.plot(self.x, cont)
                plt.plot(self.x, econt)
                plt.ylim(0, 5)
                plt.show()
        if add_mask is not None:
            mask = np.logical_and(mask, add_mask)
        self.add_interpolated(fl, el, mask, ind=ind)

    def make_qsos(self, num=1, kind='QSO', apply_lyaforest=False, apply_DLA=False, SDSS=None):
        for i, d in enumerate(self.DLA[:num]):
            name = 'data/{0:05d}/{2:04d}/{1:05d}'.format(d['PLATE'], d['MJD'], d['FIBER'])
            #print(name)
            if i % 1000 == 0:
                print(i)

            x, f, e, mask = 10 ** SDSS[name+'/loglam'][:], SDSS[name+'/flux'][:], SDSS[name+'/ivar'][:], SDSS[name+'/and_mask'][:] == 0

            corr, m = np.ones_like(self.x, dtype=float), None

            if kind == 'QSO':
                z = d['zqso']
                if apply_lyaforest:
                    corr[self.x < 1215.67] = corr[self.x < 1215.67] * np.exp(-0.0018 * (self.x[self.x < 1215.67] * (1 + d['zqso']) / 1215.67)**3.92)
                if apply_DLA:
                    m = np.asarray(self.dla_mask(self.x / (1 + d['z_abs']) * (1 + z)), dtype=bool)
                    if 0:
                        corr[m] *= self.dla(self.x[m] / (1 + d['z_abs']) * (1 + z))
                    #corr = np.ones_like(x, dtype=float)

            if kind == 'DLA':
                z = d['z_abs']
                if self.median is not None:
                    corr[self.x > 920] *= self.median(self.x[self.x > 920] / (1 + d['zqso']) * (1 + z))
                if apply_lyaforest:
                    x_0 = 1215.67 * (1 + z) / (1 + d['zqso'])
                    corr[self.x < x_0] = corr[self.x < x_0] * np.exp(-0.0018 * (self.x[self.x < x_0] * (1 + z) / 1215.67) ** 3.92)
                if apply_DLA:
                    pass

            self.add(x, f, e, mask, ind=i, z=z, corr=corr, add_mask=m)

            if 0:
                self.mask[self.size - 1] *= (corr > 0)
                m = self.mask[self.size - 1]
                self.f[self.size-1][m] = self.f[self.size-1][m] / corr[m]
                self.e[self.size-1][m] = self.e[self.size-1][m] / corr[m]

    def finalize(self):
        for attr in ['f', 'e']:
            setattr(self, attr, np.ma.array(getattr(self, attr)[:self.size], mask=self.mask[:self.size]))

        for attr in ['norm', 'w', 'inds']:
            setattr(self, attr, getattr(self, attr)[:self.size])

    def stack(self, mode='w_mean'):
        m = self.m[self.inds]
        if mode == 'mean':
            self.fl = np.ma.sum(self.f[m] / self.norm[m, None] * self.w[m, None], axis=0) / np.ma.sum(self.w[m, None], axis=0)

        if mode == 'w_mean':
            w = self.e[m]**(-1) * self.w[m, None]
            self.fl = np.ma.sum(self.f[m] / self.norm[m, None] * w, axis=0) / np.ma.sum(w, axis=0)

        if mode == 'median':
            self.fl = np.ma.median(self.f[m] / self.norm[m, None], axis=0)

        self.el = (np.ma.sum(self.e[m] ** (-2) / self.norm[m, None] * self.w[m, None], axis=0) / np.ma.sum(self.w[m, None], axis=0))**(-2)

    def stats(self):

        self.plot(self.n, self.w, 'o', alpha=0.2)
        #self.sig = np.ma.std(self.f, axis=0)
        self.ff = np.ma.filled(np.ma.sort(self.f / self.norm[:, None], axis=0), np.nan)
        self.sig = [np.nanpercentile(self.ff, 16, axis=0), np.nanpercentile(self.ff, 84, axis=0)]
        plt.fill_between(self.x, self.sig[0], self.sig[1], color='dodgerblue', alpha=0.4)
        plt.plot(self.x, self.fl)
        plt.show()

    def inter(self):
        return interp1d(self.x, self.fl, bounds_error=False, fill_value=0, assume_sorted=True)

    def save(self, filename=None, stack=False):
        if filename is None:
            filename = self.name if ~stack else self.name + '_stack'
        f = h5py.File(filename.replace('.hdf5', '') + '.hdf5', 'w')

        if stack:
            f['wavelength'], f['flux'] = self.x, self.fl
        else:
            for attr in ['f', 'e', 'norm', 'w', 'inds']:
                f[attr] = getattr(self, attr)
        f.close()

    def load(self, filename=None):
        if filename is None:
            filename = self.name
        f = h5py.File(filename.replace('.hdf5', '')+'.hdf5', 'r')

        # >> use dtype to get rid of the Memory error in stack calculations:
        dtype = {'f': np.float32, 'e': np.float32, 'norm': np.float32, 'w': np.float32, 'inds': np.int}

        for attr in ['f', 'e', 'norm', 'w', 'inds']:
            if attr in f:
                setattr(self, attr, np.ma.array(f[attr][:], mask=(f[attr][:] == 0), dtype=dtype[attr]))
        for attr, name in zip(['x', 'fl'], ['wavelength', 'flux']):
            if name in f:
                setattr(self, attr, f[name][:])
        f.close()