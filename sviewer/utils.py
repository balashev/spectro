# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:25:04 2016

@author: Serj
"""
import astropy.constants as ac
from astropy.modeling.functional_models import Moffat1D
import inspect
from itertools import compress
from math import atan2,degrees
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.stats import rv_continuous
import sys
sys.path.append('C:/science/python')
from spectro.sviewer.utils import *
import threading
import time

def include(filename):
    if os.path.exists(filename):
        exec(open(filename).read())

class Timer:
    """
    class for timing options
    """
    def __init__(self, name='', verbose=True):
        self.start = time.time()
        self.name = name
        self.verbose = verbose
        if self.name != '':
            self.name += ': '

    def restart(self):
        self.start = time.time()
    
    def time(self, st=None):
        s = self.start
        self.start = time.time()
        if st is not None and self.verbose:
            print(self.name + str(st) + ':', self.start - s)
        return self.start - s
        
    def get_time_hhmmss(self, st):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        print(st, time_str)
        return time_str

    def sleep(self, t=0):
        time.sleep(t)

class MaskableList(list):
    """
    make list to be maskable like numpy arrays
    """
    def __getitem__(self, index):
        try:
            return super(MaskableList, self).__getitem__(index)
        except TypeError:
            return MaskableList(compress(self, index))

    def uniqueappend(self, other):
        for line in other:
            if line not in self:
                self.append(line)
        return self

def debug(o, name=None):
    s = '' if name is None else name+': '
    print(s + str(o) + ' ('+inspect.stack()[1][3]+')')

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='█'):
    """
    Call in a loop to create terminal progress bar 
    (taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console)
    parameters:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def add_field(a, descr, vals=None):
    """
    Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields
      vals  -- a numpy array to be added in the field. If None - nothing to add

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("`A' must be a structured numpy array")
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    b[descr[0][0]] = vals
    return b

def slice_fields(a, fields):
    """
    Slice numpy structured array

    Arguments:
        - a         : a structured numpy array
    :param fields:
    :return:
    """
    dtype2 = np.dtype({name:a.dtype.fields[name] for name in fields})
    return np.ndarray(a.shape, dtype2, a, 0, a.strides)

def hms_to_deg(coord):
    if ':' in coord:
        h, m, s = int(coord.split(':')[0]), int(coord.split(':')[1]), float(coord.split(':')[2])
    elif 'h' in coord:
        h, m, s = int(coord[:coord.index('h')]), int(coord[coord.index('h')+1:coord.index('m')]), float(coord[coord.index('m')+1:])
    else:
        h, m, s = int(coord[:2]), int(coord[2:4]), float(coord[4:])
    return (h * 3600 + m * 60 + s) / 240

def dms_to_deg(coord):
    if ':' in coord:
        d, m, s = int(coord.split(':')[0]), int(coord.split(':')[1]), float(coord.split(':')[2])
    elif 'h' in coord:
        d, m, s = int(coord[:coord.index('d')]), int(coord[coord.index('d')+1:coord.index('m')]), float(coord[coord.index('m')+1:])
    else:
        d, m, s = int(coord[:3]), int(coord[3:5]), float(coord[5:])
    sign = -1 if coord[0] == '-' else 1
    return sign * (np.abs(d) + (m * 60 + s) / 3600)

#Label line with line2D label data
def labelLine(line, x, label=None, align=True, xpos=0, ypos=0, **kwargs):

    ax = line.get_axes()
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_axis_bgcolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    print(x, y, label, trans_angle)
    return ax.text(x+xpos, y+ypos, label, rotation=trans_angle, **kwargs)

def labelLines(lines, align=True, xvals=None, **kwargs):

    ax = lines[0].get_axes()
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin, xmax, len(labLines)+2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)


class roman():
    def __init__(self):
        self.table = [['M', 1000], ['CM', 900], ['D', 500], ['CD', 400], ['C', 100], ['XC', 90], ['L', 50], ['XL', 40],
                      ['X', 10], ['IX', 9], ['V', 5], ['IV', 4], ['I', 1]]

    def int_to_roman(self, integer):
        """
        Convert arabic number to roman number.
        parameter:
            - integer         :  a number to convert
        return: r
            - r               :  a roman number
        """
        parts = []
        for letter, value in self.table:
            while value <= integer:
                integer -= value
                parts.append(letter)
        return ''.join(parts)

    def roman_to_int(self, string):
        """
        Convert roman number to integer.
        parameter:
            - string          :  a roman to convert
        return: i
            - i               :  an integer
        """
        result = 0
        for letter, value in self.table:
            while string.startswith(letter):
                result += value
                string = string[len(letter):]
        return result

    def separate_ion(self, string):
        ind = np.min([string[1:].index(letter) for letter, value in self.table if letter in string[1:]]) + 1
        return string[:ind], string[ind:]

    @classmethod
    def int(cls, string):
        s = cls()
        return s.roman_to_int(string)

    @classmethod
    def roman(cls, integer):
        s = cls()
        return s.int_to_roman(integer)

    @classmethod
    def ion(cls, string):
        s = cls()
        return s.separate_ion(string)

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

# ---------------------------------------------------------
# fitting functions

class moffat_func(rv_continuous):
    "Moffat spectral distribution"

    def _pdf(self, x):
        # return 1.198436723 / gamma ** 2 * (1 + (x / gamma)**2) ** (-4.765)
        return 1.131578879 * (1 + x ** 2) ** (-4.765)

def moffat_fit(x, a, x_0, gamma, c):
    moffat = Moffat1D(a, x_0, gamma, 4.765)
    return moffat(x) + c

# ---------------------------------------------------------
# fitting functions

def smooth(x, window_len=11, window='hanning', mode='valid'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode=mode)[window_len-1:-window_len+1]
    return y

def flux_to_mag(flux, x, filter_name):
    """
    Calculate the flux at x in the filter given by filter_name
    :param flux:  flux in 1e-17 erg/cm^2/s/Hz
    :param x:     wavelengths in Angstrem
    :param filter_name:
    :return:    magnitude of the flux in corresponding filter
    """
    fil = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/data/SDSS/' + filter_name + '.dat', skip_header=6, usecols=(0, 1), unpack=True)
    mask = np.logical_and(x > fil[0][0], x < fil[0][-1])
    fil = interp1d(fil[0], fil[1], bounds_error=False, fill_value=0, assume_sorted=True)
    b = {'u': 1.4e-10, 'g': 0.9e-10, 'r': 1.2e-10, 'i': 1.8e-10, 'z': 7.4e-10}
    m = - 2.5 / np.log(10) * (np.arcsinh(
        flux[mask] * x[mask] ** 2 / ac.c.to('Angstrom/s').value / 3.631e-20 / 2 / b[filter_name]) + np.log(
        b[filter_name]))
    return np.trapz(m * fil(x[mask]), x=x[mask]) / np.trapz(fil(x[mask]), x=x[mask])
