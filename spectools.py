# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 18:01:14 2016

@author: Serj
"""

import numpy as np
from astropy.io import fits
import os, sys
sys.path.append('C:/science/python')
from astrolib import helcorr
from glob import glob
from vac_helio import vac_helio

class exposure():
    def __init__(self, folder, name=None, instr=None, gelio=True, mask=False):
        self.folder = folder
        if name is not None:
            self.name = name
        else:
            self.name = folder[folder.rfind('\\')+1:]
        self.readfits()
        self.instr = instr
        self.gelio = gelio
        self.mask = mask
        print(self.name, self.folder)
        
    def readfits(self):
        path = self.folder
        if os.path.exists(path+'/list.dat'):
            f_in = open(folder+'/list.dat', 'r')
            self.list_of_fits = []   
            for l in f_in:
                self.list_of_fits.append(str(l).replace('\n', ''))
        else:
            print(path)
            self.list_of_fits = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.fits'))]    
    
    def fitstoascii(self):
        with open(self.folder+r'\file.list','w') as f_list:
            for file in self.list_of_fits:
                f_list.write(file.replace('.fits', '.dat') + '\n')
                
        if self.instr == 'UVES':
            self.fitstoascii_UVES()
                    
    
    def fitstoascii_UVES(self):            
        for file in self.list_of_fits:
            s = file.replace('.fits', '.dat')
            hdulist = fits.open(file)
            if self.mode == 'CRVAL':
                header = hdulist[0].header
                print(header)
                print(header['CRVAL1'], header['CRPIX1'])
                prihdr = hdulist[0].data
                print(prihdr[0], prihdr[1], prihdr[2], prihdr[3])
                l = np.power(10, np.arange(header['NAXIS1'])*header['CD1_1']+header['CRVAL1'])
                f = prihdr[0]
                e = prihdr[1]
                print(l, f, e)
            elif self.mode == 'ADP':
                header = hdulist[1].header
                print(header)
                prihdr = hdulist[1].data
                print(prihdr)
                x = prihdr.field(0)
                n = x.size
                l = prihdr.field(0)[0]
                f = prihdr.field(1)[0]
                e = prihdr.field(2)[0]
                print(l, f, e)
                
            if self.mask:
                mask = (l > 3300) * (f != 0)
                l, f, e = l[mask], f[mask], e[mask]
                print(l, f, e, mask)
            
            if self.gelio:
                long = hdulist[0].header['HIERARCH ESO TEL GEOLON']
                lat  = hdulist[0].header['HIERARCH ESO TEL GEOLAT']
                elev = hdulist[0].header['HIERARCH ESO TEL GEOELEV']
                ra   = hdulist[0].header['RA']
                dec  = hdulist[0].header['DEC']
                mjd  = hdulist[0].header['MJD-OBS']
                rjd = mjd + 0.5
                jd  = rjd + 2400000.0
                exptime = hdulist[0].header['EXPTIME']
                corrhel, hjd = helcorr(long,lat,elev,ra/15.0,dec, rjd+exptime/86400.0/2)
                v_helio = corrhel
                print(v_helio)
                l = vac_helio(l, v_helio)
            
            print('>>> writing to ascii file ...')
            
            np.savetxt(s, np.array([l, f, e]).transpose(), fmt='%.4f')
                    
if __name__ == "__main__":
    exp = exposure(r'D:\science\QSO\UVES\J031115-172247', instr='UVES', gelio=False)
    exp.mode = 'CRVAL'
    exp.fitstoascii()