from lmfit import Minimizer, Parameters, report_fit, fit_report, conf_interval, printfuncs
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import pickle
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize, least_squares
from scipy.signal import correlate, argrelextrema

from ..profiles import tau, convolveflux, fisherbN
from ..atomic import line
from .external import spectres
from .fit import fitPars
from .graphics import Spectrum
from .utils import Timer
import emcee

def correl(y, fit, err=None):
    stride = y.strides[0]
    y_strides = as_strided(y, shape=[len(y) - len(fit) + 1, len(fit)], strides=[stride, stride]) - fit
    if err is not None:
        err_strides = as_strided(err, shape=[len(err) - len(fit) + 1, len(fit)], strides=[stride, stride])
        return np.sum(np.power(np.divide(y_strides, err_strides), 2), axis=1)
    else:
        return np.sum(np.power(y_strides, 2), axis=1)

def correl3(y, fit, mask, err=None):
    #t = Timer()
    stride = y.strides[0]
    #t.time('1')
    y_s = as_strided(y, shape=[len(y) - fit.shape[0] + 1, fit.shape[0], 4], strides=[stride, stride, 0])
    #print(y_s.shape)
    #t.time('2')
    y_s = y_s - fit
    #t.time('3')
    if err is not None:
        err_strides = as_strided(err, shape=[len(err) - fit.shape[0] + 1, fit.shape[0], 4], strides=[stride, stride, 0])
        #t.time('4')
        #s = np.sum(np.divide(y_s, err_strides) ** 2 * mask, axis=1)
        #t.time('5')
        return np.sum(np.divide(y_s, err_strides) ** 2 * mask, axis=1)
    else:
        return np.sum(y_s**2 * mask, axis=1)

def correlnew(y, fit, mask, err=None):
    #t = Timer()
    stride = y.strides[0]
    #t.time('1')
    y_s = as_strided(y, shape=[len(y) - fit.shape[0] + 1, fit.shape[0], 4], strides=[stride, stride, 0])
    #t.time('2')
    y_s = np.multiply(y_s, fit)
    #t.time('3')
    return np.sum(y_s * mask, axis=1)

def Lyaforest_scan(parent, data, do='all'):
    """
    Scan for individual Lya forest lines and fit them
    paremeters:
        -  data         :  spectrum to fit, 2d numpy array
        -  do           :  type of calculations, can be '
                                'all'  -  whole fitting process
                                'corr' -  only cross correlate the spectrum
                                'fit'  -  only fit lines using loaded correlation array from "pickle/<filename>.pkl" file
    """

    sample = np.genfromtxt('C:/Users/ksush/work/Lyasample/lines.dat', usecols=(0, 8), skip_header=1, dtype=[('z', '<f8'), ('name', '|S40')], unpack=True)

    print('scan')
    t = Timer()

    # >>> prepare spectrum
    qsoname = os.path.basename(parent.s[0].filename)
    if 0:
        x = np.linspace(data[0][1], data[0][-2], int((data[0][-2]-data[0][1])/0.017))
    else:
        x = np.logspace(np.log10(data[0][1]), np.log10(data[0][-2]), int((data[0][-2]-data[0][1])/0.017))
    y, err = spectres.spectres(data[0], data[1], x, spec_errs=data[2])
    print('1')
    s = Spectrum(parent, name='rebinned')
    parent.normview = False
    print('1')
    s.set_data([x, y, err])
    s.spec.norm.set_data(x, y, err=err)
    s.cont.set_data(x, np.ones_like(x))
    s.resolution = parent.s[0].resolution
    print('2')
    if 1:
        mask = (np.abs((y - 1) / err) < 5) * (err > 0)
        window = 300
        box = np.ones(window) / window
        snr = interp1d(x[mask], np.convolve(y[mask]/err[mask], box, mode='same'), fill_value='extrapolate')
    print('3')
    parent.normview = True
    #parent.s.append(s)
    parent.s.ind = 0
    #parent.s.redraw()

    lya = 1215.6701
    zmin = x[0] / lya - 1
    zmax = x[-1] / lya - 1
    print(zmin, zmax)

    koef_red = 4
    flux_limit = 0.97

    t.time('prepare')

    typ = {0: 'c', 1: 'r', 2: 'l', 3: 'b'}

    if do in ['all', 'corr']:
        # >>> make Lya line grid
        if 0:
            N_grid, b_grid, xf, f = makeLyagrid_uniform(N_range=[13.0, 14.5], b_range=[10, 30], N_num=5, b_num=5, resolution=s.resolution)
        else:
            max_ston = np.max(snr(np.linspace(x[0], x[-1], 50)))
            print('max_ston:', max_ston)
            print('res:',s.resolution)
            N_grid, b_grid, xf, f = makeLyagrid_fisher(N_range=[12.5, 14.6], b_range=[8, 50], z=(zmin+zmax)/2, ston=max_ston, resolution=s.resolution, plot=1)
        #N_grid, b_grid, xf, f = makeLyagrid(N_range=[14.39, 14.39], b_range=[28, 28], N_num=1, b_num=1, resolution=s.resolution)
        xl = xf * x[int(len(x)/2)] / 1215.6701
        t.time('make Lya grid')

        # >>> calc fit line
        show_corr = 0
        lines = []
        for i, N in enumerate(N_grid):
            print(i)
            for k, b in enumerate(b_grid):
                # >>> prepare fit
                inter = interp1d(xl, f[i,k])
                mask = f[i,k] < flux_limit
                imin, imax = np.argmin(np.abs(x - xl[mask][0])), np.argmin(np.abs(x - xl[mask][-1]))
                yf = inter(x[imin:imax])
                mask = np.ones([len(yf), 4])
                mask[:, 1] = (yf < 1 - (1 - np.min(yf)) / 2) | (x[imin:imax] > x[(imin + imax) // 2])
                mask[:, 2] = (yf < 1 - (1 - np.min(yf)) / 2) | (x[imin:imax] < x[(imin + imax) // 2])
                mask[:, 3] = (yf < 1 - (1 - np.min(yf)) / 3)
                yf = np.repeat(yf[:,np.newaxis], 4, axis=1)

                ind = np.argmin(np.abs(x-xl[np.argmin(f[i,k])]))
                x_corr = x[ind-imin:ind-imax+1]

                #t.time('prepare fit')

                corr = correl3(y, yf, mask, err) / np.sum(mask, axis=0) / koef_red
                #print(corr.shape, corr)
                if show_corr:
                    for l in range(4):
                        s = Spectrum(parent, name='correlation_'+typ[l])
                        if 1:
                            s.set_data([x_corr, corr[:, l]])
                            s.spec.norm.set_data(x_corr, corr[:, l])
                        else:
                            pos = np.argmin(np.abs(x - 3954.761))
                            s.set_data([x[imin - ind + pos:imax - ind + pos], yf[:, l] * mask[:, l]])
                            s.spec.norm.set_data(x[imin - ind + pos:imax - ind + pos], yf[:, l]* mask[:, l])
                        parent.s.append(s)
                        parent.s.ind = 1
                        parent.s.redraw()

                #t.time('correlate')
                mask = np.where(corr < 1)
                if len(mask[0]) > 0:
                    inds = np.hsplit(np.array(mask), np.where(np.diff(mask[0]) > 1)[0] + 1)
                    #print('inds:', inds)
                    for ind in inds:
                        #print('ind:', tuple(ind))
                        if 0 in ind[1]:
                            ii = np.where(ind[1] == 0)
                            imin = ii[0][np.argmin(corr[tuple(ind)][ii])]
                        else:
                            imin = np.argmin(corr[tuple(ind)])
                        lines.append((x_corr[ind[0][imin]] / 1215.6701 - 1, i, k, corr[ind[0][imin], ind[1][imin]], typ[ind[1, imin]]))
                #t.time('find local minima')
            t.time('loop')
        with open('C:/Users/ksush/work/Lyasample/pickle/' + qsoname, 'wb') as fil:
            pickle.dump((N_grid, b_grid, xf, f, lines), fil)
        t.time('fit lines')
    else:
        with open('C:/Users/ksush/work/Lyasample/pickle/' + qsoname, 'rb') as fil:
            N_grid, b_grid, xf, f, lines = pickle.load(fil)

    zi = [l[0] for l in lines]
    corr = [l[3] for l in lines]
    c = [l[4] for l in lines]
    sortind = np.argsort(zi)
    zi = np.array(zi)[sortind]
    corr = np.array(corr)[sortind]
    c = np.array(c)[sortind]
    ind = np.where(np.diff(zi) > 0.0003)[0] + 1
    ind = np.insert(ind, 0, 0)
    ind = np.append(ind, len(zi)-1)
    types, ix = [], []
    for i in range(1, len(ind)):
        if len(corr[ind[i-1]:ind[i]]) > 0:
            #print(corr[ind[i-1]:ind[i]], np.argmin(corr[ind[i-1]:ind[i]]), ind[i-1]+np.argmin(corr[ind[i-1]:ind[i]]))
            ix.append(ind[i-1]+np.argmin(corr[ind[i-1]:ind[i]]))
            if 'c' in c[ind[i - 1]:ind[i]]:
                types.append('c')
            else:
                types.append(c[np.floor_divide(ind[i - 1] + ind[i], 2)])
    ind = ix

    # >>> fit lines
    #print(ind)
    #print(types)
    check_doublicates = 0
    dynamic_mask = 1

    if do in ['all', 'fit']:
        print('dynamic mask status:',dynamic_mask)
        parent.fit = fitPars(parent)
        if check_doublicates:
            old_lines = np.genfromtxt('C:/Users/ksush/work/Lyasample//lines.dat', names=True, dtype=None)
        if 0:
            filename = open('C:/Users/ksush/work/Lyasample/lines.dat', 'a')
        else:
            filename = open('C:/Users/ksush/work/Lyasample/lines/'+qsoname, 'w')

        def fcn2min(params, x, y, err, line):
            line.z, line.logN, line.b = params['z'].value, params['N'].value, params['b'].value
            line.calctau()
            inter = interp1d(line.x, convolveflux(line.x, np.exp(-line.tau), res=parent.s[0].resolution), bounds_error=False, fill_value=1)
            return (y - inter(x)) / err

        def VoigtProfile(x, z, logN, b):
            line_.z, line_.logN, line_.b = z, logN, b
            line_.calctau()
            inter = interp1d(line_.x, convolveflux(line_.x, np.exp(-line_.tau), res=parent.s[0].resolution),
                             bounds_error=False, fill_value=1)
            return inter(x)

        def VoigtProfile_mask(params, x, y, err, line, typ):
            line_.z, line_.logN, line_.b = tuple(params)
            #line.z, line.logN, line.b = params['z'].value, params['N'].value, params['b'].value
            line_.calctau()
            inter = interp1d(line_.x, convolveflux(line_.x, np.exp(-line_.tau), res=parent.s[0].resolution), bounds_error=False, fill_value=1)


            intx=inter(x)
            #print(line_.z, line_.logN, line_.b)
            if 0:
                chi = (y - intx) / err
                #print(line.z, line.logN, line.b, np.sum(chi ** 2) / (len(y) - 3))
                return np.sum(chi**2)/(len(y)-3)
            else:
                mask0 = intx <= flux_limit
                #print(len(x[mask]),len(x_over))
                if (len(x[mask0]) <= 3)|(len(x[(mask0)*(mask_over)])<len(x[(mask0)])):
                    #print(len(summary), line_.z, line_.logN, line_.b, typ)
                    summary.append(1e10)
                    #print('chi:', summary[-1])
                    #print('min chi:', np.min(summary), np.argmin(summary))
                    return 1e10

                else:
                    xmin, xmax = x[mask0][0], x[mask0][-1]
                    if typ == 'l':
                        mask2 = intx <= 1 - (1 - np.min(intx)) / 2
                        if len(x[mask2])>=len(x[mask0]):
                            #print('calculated mask is larger than flux-limited mask')
                            #print(len(summary), line_.z, line_.logN, line_.b, typ)
                            summary.append(1e10)
                            #print('chi:', summary[-1])
                            #print('min chi:', np.min(summary), np.argmin(summary))
                            return 1e10
                        else:
                            #xmax = x[np.max(np.where(np.abs(np.diff(mask2)) > 0)[0])]
                            xmax = x[np.max(np.where(np.diff(mask2) > 0)[0])]
                            # print('min ind, max ind')
                            # print(np.diff(mask2))
                            # print(np.min(np.where(np.abs(np.diff(mask2)) > 0)[0]),np.max(np.where(np.abs(np.diff(mask2)) > 0)[0]))
                            # print(np.min(np.where(np.diff(mask2) > 0)[0]), np.max(np.where(np.diff(mask2) > 0)[0]))

                    elif typ == 'r':
                        mask2 = intx <= 1 - (1 - np.min(intx)) / 2
                        if len(x[mask2]) >= len(x[mask0]):
                            #print('calculated mask is larger than flux-limited mask')
                            #print(len(summary), line_.z, line_.logN, line_.b, typ)
                            summary.append(1e10)
                            #print('chi:', summary[-1])
                            #print('min chi:', np.min(summary), np.argmin(summary))
                            return 1e10
                        else:
                            #xmin = x[np.min(np.where(np.abs(np.diff(mask2)) > 0)[0])+1]
                            xmin = x[np.min(np.where(np.diff(mask2) > 0)[0]) + 1]
                    elif typ == 'b':
                        mask2 = intx <= 1 - (1 - np.min(intx)) / 3
                        #print(np.shape(mask2),np.count_nonzero(mask2))
                        #print('non zeros', np.count_nonzero(np.diff(mask2)))
                        if len(x[mask2])>=len(x[mask0]):
                            #print('calculated mask is larger than flux-limited mask')
                            #print(len(summary), line_.z, line_.logN, line_.b, typ)
                            summary.append(1e10)
                            #print('chi:', summary[-1])
                            #print('min chi:', np.min(summary), np.argmin(summary))
                            return 1e10
                        else:
                            # xmin, xmax = x[np.min(np.where(np.abs(np.diff(mask2)) > 0)[0])+1], x[
                            #     np.max(np.where(np.abs(np.diff(mask2)) > 0)[0])]
                            xmin, xmax = x[np.min(np.where(np.diff(mask2) > 0)[0]) + 1], x[
                                np.max(np.where(np.diff(mask2) > 0)[0])]


                    mask = (x >= xmin) * (x <= xmax) * (err > 0)
                    if len(x[mask])<=3:
                        #print(len(summary), line_.z, line_.logN, line_.b, typ)
                        summary.append(1e10)
                        #print('chi:', summary[-1])
                        #print('min chi:', np.min(summary), np.argmin(summary))
                        return 1e10

                    intx, y, err = intx[mask], y[mask], err[mask]
                    chi = (y - intx) / err
                    if 1:
                        #print(len(summary), line_.z, line_.logN, line_.b, typ)
                        #print(len(x_over), len(x[mask]), np.sum(chi ** 2) / (len(y) - 3))
                        summary.append(np.sum(chi ** 2) / (len(y) - 3))
                        #print('chi:', summary[-1])
                        #print('min chi:', np.min(summary), np.argmin(summary))

                    return np.sum(chi**2)/(len(y)-3)


        # create a set of Parameters
        params = Parameters()
        params.add('z', value=2, min=0, max=10)
        params.add('N', value=13, min=11, max=20)
        params.add('b', value=20, min=5, max=200)

        def calc_mask(l, typ):
            mask = f[l[1], l[2]] < flux_limit
            xmin, xmax = xf[mask][0], xf[mask][-1]
            if typ == 'l':
                mask2 = f[l[1], l[2]] < 1 - (1 - np.min(f[l[1], l[2]])) / 2
                xmax = xf[np.max(np.where(np.diff(mask2) > 0)[0])]
            elif typ == 'r':
                mask2 = f[l[1], l[2]] < 1 - (1 - np.min(f[l[1], l[2]])) / 2
                xmin = xf[np.min(np.where(np.diff(mask2) > 0)[0])]
            elif typ == 'b':
                mask2 = f[l[1], l[2]] < 1 - (1 - np.min(f[l[1], l[2]])) / 3
                xmin, xmax = xf[np.min(np.where(np.diff(mask2) > 0)[0])], xf[np.max(np.where(np.diff(mask2) > 0)[0])]
            elif typ == 'over':
                # imin, imax = np.argmin(np.abs(xf - (2*xmin - xmax))), np.argmin(
                #      np.abs(xf -(2*xmax - xmin)))
                imin, imax = np.argmin(np.abs(xf - (3*xmin - xmax) / 2)), np.argmin(
                    np.abs(xf -(3*xmax - xmin) / 2))
                xmin, xmax = xf[imin], xf[imax]
            mask = (x > xmin * (1 + l[0])) * (x < xmax * (1 + l[0])) * (err > 0 )

            return mask

        #---------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        #to update old mask for drawing actual profiles for dynamic_mask regime

        def update_mask(z,N,b,typ):
            line_ = tau(line=line('lya', l=1215.6701, f=0.4164, g=6.265e8, z=z, logN=N, b=b), resolution=parent.s[0].resolution)
            line_.calctau()
            flux = convolveflux(line_.x, np.exp(-line_.tau), res=parent.s[0].resolution)
            x, y, err = parent.s[0].spec.x(), parent.s[0].spec.y(), parent.s[0].spec.err()
            inter = interp1d(line_.x, flux, bounds_error=False, fill_value=1)

            #xx = xf*(1+z)
            #intx = inter(xx)
            intx = inter(x)
            #print(len(xx), len(x))

            mask = intx <= flux_limit
            #xmin, xmax = xx[mask][0], xx[mask][-1]
            xmin, xmax = x[mask][0], x[mask][-1]
            if typ == 'l':
                mask2 = intx <= 1 - (1 - np.min(intx)) / 2
                xmax2 = x[np.max(np.where(np.diff(mask2) > 0)[0])]
                #xmax2 = x[np.max(np.where(np.abs(np.diff(mask2)) > 0)[0])]
                xmax=xmax2
            elif typ == 'r':
                mask2 = intx <= 1 - (1 - np.min(intx)) / 2
                xmin2 = x[np.min(np.where(np.diff(mask2) > 0)[0])+1]
                #xmin2 = x[np.min(np.where(np.abs(np.diff(mask2)) > 0)[0])+1]
                xmin=xmin2
            elif typ == 'b':
                mask2 = intx <= 1 - (1 - np.min(intx)) / 3
                xmin2, xmax2 = x[np.min(np.where(np.diff(mask2) > 0)[0])+1], x[np.max(np.where(np.diff(mask2) > 0)[0])]
                #xmin2, xmax2 = x[np.min(np.where(np.abs(np.diff(mask2)) > 0)[0])+1], x[np.max(np.where(np.abs(np.diff(mask2)) > 0)[0])]
                xmin, xmax =xmin2,xmax2


            mask = (x >= xmin ) * (x <= xmax ) * (err > 0 )

            return mask

        #---------------------------------------------------------------------------------------------------------------

        m = np.zeros_like(parent.s[0].spec.x(), dtype=bool)
        lines = [lines[i] for i in sortind]
        if len(lines) > 0:
            i = 0
            while i < len(ind):
                x, y, err = parent.s[0].spec.x(), parent.s[0].spec.y(), parent.s[0].spec.err()
                #print(i, ind[i], types[i])
                l, typ = lines[ind[i]], types[i]

                mask = calc_mask(l, types[i])
                if ind[i] < ind[-1]:
                    mask_next = calc_mask(lines[ind[i+1]], types[i+1])
                else:
                    mask_next = np.zeros_like(mask)
                if np.sum(mask * mask_next) > 4:
                    i += 1
                else:
                    mask_over = calc_mask(l, 'over')
                    x_over, y_over, err_over = x[mask_over], y[mask_over], err[mask_over]
                    x_stat, y_stat, err_stat = x[mask], y[mask], err[mask]
                    global line_

                    line_ = tau(z=l[0], logN=N_grid[l[1]], b=b_grid[l[2]], resolution=parent.s[0].resolution)
                    if 1 or (1 - np.min(f[l[1], l[2]])) / np.mean(err) > 3:
                        save_N, save_b = line_.logN, line_.b

                        params['z'].value, params['N'].value, params['b'].value = line_.z, line_.logN, line_.b
                        params['z'].min, params['z'].max = line_.z * (1 - 5 / parent.s[0].resolution), line_.z * (1 + 5 / parent.s[0].resolution)

                        #dN, db, F, Fmin = fisherbN(save_N, save_b, [line('lya', l=1215.6701, f=0.4164, g=6.265e8, z=line_.z)], ston=float(snr(1215.67 * (1 + line_.z))), resolution=parent.s[0].resolution)





                        # ===============================================================================================
                        # new minimization within varied mask, calculated on each (z,N,b) point
                        # errors were estimated via Fisher matrix
                        if dynamic_mask:
                            summary=[]
                            #print(types[i])
                            # result = minimize(VoigtProfile_mask, x0=(l[0], save_N, save_b), method='tnc', args=(x, y, err, line_, types[i]),
                            #                     bounds=((params['z'].min,params['z'].max),(params['N'].min, params['N'].max),(params['b'].min, params['b'].max)))
                            #minimization with tnc method does not return min of chi.....use another method, fer example Nelder-Mead
                            # result = minimize(VoigtProfile_mask, x0=(l[0], save_N, save_b), method='tnc', args=(x_over, y_over, err_over, line_, types[i]),
                            #                    bounds=((params['z'].min,params['z'].max),(params['N'].min, params['N'].max),(params['b'].min, params['b'].max)))
                            result = minimize(VoigtProfile_mask, x0=(l[0], save_N, save_b), method='Nelder-Mead', args=(x, y, err, line_, types[i]),
                                              bounds=((params['z'].min,params['z'].max),(params['N'].min, params['N'].max),(params['b'].min, params['b'].max)))



                            z, N, b = tuple(result.x)
                            Nerr, berr, F, Fmin = fisherbN(N, b,
                                                       [line('lya', l=1215.6701, f=0.4164, g=6.265e8, z=z)],
                                                       ston=float(snr(1215.67 * (1 + z))),
                                                       resolution=parent.s[0].resolution)
                            chi = result.fun


                        #===============================================================================================
                        #old minimization within static mask, calculated from save_N, save_b and z:
                        else:
                            popt, pcov = curve_fit(VoigtProfile, x_stat, y_stat, p0=(l[0], save_N, save_b), sigma=err_stat, method='dogbox', bounds=([params['z'].min, params['N'].min, params['b'].min], [params['z'].max, params['N'].max, params['b'].max]))
                            z, N, Nerr, b, berr = popt[0], popt[1], np.sqrt(pcov[1, 1]), popt[2], np.sqrt(pcov[2, 2])
                            #z,N,b =l[0], save_N, save_b
                            chi = np.sum(((VoigtProfile(x_stat, z, N, b) - y_stat) / err_stat) ** 2) / (len(x_stat) - 3)
                        # ----------------------------------------------------------------------------------------------

                        if chi < 3 and N / Nerr > 5 and b / berr > 5: #and Nerr != 0 and berr != 0 and np.sum(m * mask) == 0:
                            plt.errorbar(N, b, fmt='o', xerr=Nerr, yerr=berr, color='k')
                            plt.arrow(save_N, save_b, N - save_N, b - save_b, fc='orangered', ec='orangered')
                            lyb = True
                            if lyb:
                                lyb = False
                                for lylines in [line('lyb', l=1025.7223, f=0.07912, g=1.897e8, z=z, logN=N, b=b), line('lyg', l=972.5368, f=0.02900, g=8.127e7, z=z, logN=N, b=b)]:
                                    line_ = tau(line=lylines, resolution=parent.s[0].resolution)
                                    line_.calctau()
                                    flux = convolveflux(line_.x, np.exp(-line_.tau), res=parent.s[0].resolution)
                                    #x, y, err = parent.s[0].spec.x(), parent.s[0].spec.y(), parent.s[0].spec.err()
                                    m_lyb = (x > line_.x[0]) * (x < line_.x[-1])
                                    inter = interp1d(line_.x, flux, bounds_error=False, fill_value=1)
                                    m1 = (y[m_lyb] > inter(x[m_lyb])) * (err[m_lyb] > 0)
                                    #print(np.sum(m_lyb), np.sum(m1))
                                    if np.sum(m1) > 10 and np.sum(((y[m_lyb][m1] - inter(x[m_lyb])[m1])/err[m_lyb][m1])**2)/np.sum(m1) > 4:
                                        fig, ax = plt.subplots()
                                        ax.plot(x[m_lyb], y[m_lyb])
                                        ax.plot(x[m_lyb], inter(x[m_lyb]), '-r')
                                        lyb = True
                                        break
                            if not lyb:
                                if dynamic_mask:
                                    mask_upd = update_mask(z,N,b,typ)
                                    m =np.logical_or(m, mask_upd)
                                    #m = np.logical_or(m, mask_over)
                                    if 1:#z>3.787 and z<3.789:temporary block
                                        line_ = tau(
                                            line=line('lya', l=1215.6701, f=0.4164, g=6.265e8, z=z, logN=N, b=b),
                                            resolution=parent.s[0].resolution)
                                        line_.calctau()
                                        flux = convolveflux(line_.x, np.exp(-line_.tau), res=parent.s[0].resolution)
                                        #x, y, err = parent.s[0].spec.x(), parent.s[0].spec.y(), parent.s[0].spec.err()
                                        inter = interp1d(line_.x, flux, bounds_error=False, fill_value=1)
                                        intx = inter(x[mask_upd])
                                        chi_hand=np.sum((intx-y[mask_upd])**2 /err[mask_upd]**2)/(len(x[mask_upd])-3)

                                        if np.abs(chi_hand-chi)> 10e-15:
                                        #if 1:#chi_hand != chi:
                                            print('bad line:')
                                            print('fit:',z,N,b,typ)
                                            print('start:',save_N, save_b)
                                            print(intx)
                                            print(inter(x[mask_over]))
                                            print(1 - (1 - np.min(intx)) / 2, 1 - (1 - np.min(intx)) / 3)
                                            #print(y[mask_upd])
                                            print(len(y[mask_over]),len(y[mask_upd]))
                                            print(chi,chi_hand)
                                            print('--------------------')
                                        # else:
                                        #     print('good line:')
                                        #     print(z, N, b, typ)
                                        #     print(chi, chi_hand)
                                else:
                                    m = np.logical_or(m, mask)

                                if check_doublicates:
                                    if len(np.where(np.abs(z - old_lines['z'][old_lines['name'] == qsoname.encode()])*300000 < 20)[0]) > 0:
                                        print(np.where(np.abs(z - old_lines['z'][old_lines['name'] == qsoname.encode()])*300000 < 20)[0])
                                    #np.where((old_lines['name'] == qsoname) and np.abs(z - old_lines['z']) * 300 > 20:
                                #if len(sample['z']) == 0 or len(np.where(np.abs(sample['z'][qsoname.encode() == sample['name']] - z) < 0.0001)[0]) == 0:
                                if dynamic_mask:
                                    filename.write('{:9.7f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.2f} {:6.3f} {:5d} {:2s} {:30s} {:30s}\n'.format(z, N, Nerr, b, berr, float(snr(1215.67*(1+z))), chi, len(x[mask_upd]), typ, qsoname, '-'))
                                else:
                                    filename.write(
                                        '{:9.7f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.2f} {:6.3f} {:5d} {:2s} {:30s} {:30s}\n'.format(
                                            z, N, Nerr, b, berr, float(snr(1215.67 * (1 + z))), chi, len(x[mask]),
                                            typ, qsoname, '-'))

                                parent.fit.addSys(z=z)
                                parent.fit.sys[len(parent.fit.sys)-1].addSpecies('HI')
                                parent.fit.setValue('N_{:d}_HI'.format(len(parent.fit.sys)-1), N)
                                parent.fit.setValue('b_{:d}_HI'.format(len(parent.fit.sys)-1), b)
                    else:
                        plt.errorbar(line.logN, line.b, fmt='o', color='r')
                        print('discarded:', line.logN, line.b)
                i += 1
        t.time('fit lines')

        filename.close()

        print(parent.fit.list())
        print('mask', len(m))
        parent.s[0].mask.set(x=m)
        parent.s.prepareFit()
        parent.s.calcFit()
        parent.s.redraw()

def makeLyagrid_uniform(N_range=[13., 14], b_range=[20, 30], N_num=30, b_num=30, resolution=50000):

    line = tau(resolution=0)
    x = np.linspace(line.l * (1 - 3 * b_range[-1]/300000), line.l * (1 + 3 * b_range[-1]/300000), 501)

    N_grid = np.linspace(N_range[0], N_range[-1], N_num)
    b_grid = np.linspace(b_range[0], b_range[-1], b_num)
    flux = np.empty([N_num, b_num, len(x)])
    for i, N in enumerate(N_grid):
        line.logN = N
        for k, b in enumerate(b_grid):
            line.b = b
            line.calctau0()
            f = np.exp(-line.calctau(x))
            flux[i,k,:] = convolveflux(x, f, res=resolution, kind='astropy')

    return N_grid, b_grid, x, flux

def makeLyagrid_fisher(N_range=[13., 14], b_range=[20, 30], ston=10, z=0, resolution=50000, plot=0):

    koef = 8
    print(N_range, b_range,ston)

    lines = [line('lya', l=1215.6701, f=0.4164, g=6.265e8, z=z)]
    N_grid, b_grid = [N_range[0]], [b_range[0]]
    while N_grid[-1] < N_range[1]:
        Nmin = 10
        for b in np.linspace(b_range[0], b_range[1], 10):
            dN, db, F, Fmin = fisherbN(N_grid[-1], b, lines, ston=ston, resolution=resolution)
            Nmin = min(Nmin, dN)
        N_grid.append(N_grid[-1] + koef * Nmin)
    while b_grid[-1] < b_range[1]:
        bmin = 10
        for N in np.linspace(N_range[0], N_range[1], 10):
            dN, db, F, Fmin = fisherbN(N, b_grid[-1], lines, ston=ston, resolution=resolution)
            bmin = min(bmin, db)
        b_grid.append(b_grid[-1] + koef * bmin)
    print(len(N_grid), len(b_grid))

    if 0 or plot:
        fig, ax = plt.subplots()
        for N in N_grid[:]:
            for b in b_grid[:]:
              dN, db, F, Fmin = fisherbN(N, b, lines, ston=ston, resolution=resolution)
              ax.errorbar(N, b, xerr=dN, yerr=db, fmt='o', color='k')
        plt.show()

    l = tau(resolution=0)
    x = np.linspace(l.l * (1 - 3 * b_range[-1] / 300000), l.l * (1 + 3 * b_range[-1] / 300000), 501)

    flux = np.empty([len(N_grid), len(b_grid), len(x)])
    for i, N in enumerate(N_grid):
        l.logN = N
        for k, b in enumerate(b_grid):
            l.b = b
            l.calctau0()
            f = np.exp(-l.calctau(x))
            flux[i,k,:] = convolveflux(x, f, res=resolution, kind='astropy')

    return N_grid, b_grid, x, flux

class plotLyalines(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29))
        self.initstatus()
        self.vb = self.getViewBox()

    def initstatus(self):
        self.s_status = False
        self.selected_point = None

    def set_data(self, data=None):
        if data is None:
            for attr in ['err', 'points', 'metals', 'bad', 'check']:
                try:
                    self.vb.removeItem(getattr(self, attr))
                except:
                    pass
        else:
            self.data = data
            mask = data['comment'] == '-'
            self.err = pg.ErrorBarItem(x=self.data['N'], y=self.data['b'], width=self.data['Nerr'], height=self.data['berr'])
            self.points = pg.ScatterPlotItem(self.data['N'][mask], self.data['b'][mask], symbol='o', pen={'color': 0.8, 'width': 1}, brush=pg.mkBrush(100, 100, 200))
            self.vb.addItem(self.err)
            self.vb.addItem(self.points)
            mask = np.asarray(['me' in d for d in data['comment']])
            self.metals = pg.ScatterPlotItem(self.data['N'][mask], self.data['b'][mask], symbol='o', pen={'color': 0.0, 'width': 1}, brush=pg.mkBrush(223, 31, 223))
            self.vb.addItem(self.metals)
            mask = np.asarray(['bad' in d for d in data['comment']])
            self.bad = pg.ScatterPlotItem(self.data['N'][mask], self.data['b'][mask], symbol='o', pen={'color': 0.0, 'width': 1}, brush=pg.mkBrush(253, 255, 63))
            self.vb.addItem(self.bad)
            mask = data['comment'] == 'checked'
            self.check = pg.ScatterPlotItem(self.data['N'][mask], self.data['b'][mask], symbol='o', pen={'color': 0.0, 'width': 1}, brush=pg.mkBrush(162, 209, 91))
            self.vb.addItem(self.check)


    def mousePressEvent(self, event):
        super(plotLyalines, self).mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            if self.s_status:
                self.mousePoint = self.vb.mapSceneToView(event.pos())
                r = self.vb.viewRange()
                self.ind = np.argmin(((self.mousePoint.x() - self.data['N']) / (r[0][1] - r[0][0]))**2   + ((self.mousePoint.y() - self.data['b']) / (r[1][1] - r[1][0]))**2)
                if self.selected_point is not None:
                    self.vb.removeItem(self.selected_point)
                self.selected_point = pg.ScatterPlotItem(x=[self.data['N'][self.ind]], y=[self.data['b'][self.ind]], symbol='o', size=15,
                                                        pen={'color': 0.8, 'width': 1}, brush=pg.mkBrush(230, 100, 10))
                self.vb.addItem(self.selected_point)
                N = [float(self.parent.item(i,1).text()) for i in range(self.parent.rowCount())]
                b = [float(self.parent.item(i,3).text()) for i in range(self.parent.rowCount())]
                ind = np.argmin((b - self.data['b'][self.ind])**2 + (N - self.data['N'][self.ind])**2)
                #ind = np.where(np.logical_and(b == self.data['b'][self.ind], N == self.data['N'][self.ind]))[0][0]
                #ind = np.where(np.logical_and(self.parent.data['b'] == self.data['b'][self.ind], self.parent.data['N'] == self.data['N'][self.ind]))[0][0]
                self.parent.setCurrentCell(0,0)
                self.parent.row_clicked(ind)

    def keyPressEvent(self, event):
        super(plotLyalines, self).keyPressEvent(event)
        key = event.key()

        if not event.isAutoRepeat():
            if event.key() == Qt.Key_S:
                self.s_status = True

    def keyReleaseEvent(self, event):
        super(plotLyalines, self).keyReleaseEvent(event)
        key = event.key()

        if not event.isAutoRepeat():

            if event.key() == Qt.Key_S:
                self.s_status = False