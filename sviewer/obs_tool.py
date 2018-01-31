import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import astroplan
import astropy.units as u
import calendar

from astroplan import FixedTarget, Observer, moon
from astroplan.plots import plot_airmass, plot_sky
from astroplan import AirmassConstraint, AtNightConstraint
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
from astropy.table import Table
from astropy.time import Time
from collections import OrderedDict
from PyQt5.QtGui import QFont
from pytz import timezone

from .graphics import gline

class UVESSetups(OrderedDict):
    def __init__(self):
        #super(self, OrderedDict).__init__()
        self.initData()

    def initData(self):
        self['DIC1_CD1_CD3_1'] = [[303, 346, 388], [476, 580, 684]]
        self['DIC1_CD2_CD3_2'] = [[326, 390, 454], [458, 564, 668]]
        self['DIC1_CD1_CD3_1'] = [[303, 346, 388], [458, 564, 668]]
        self['DIC1_CD2_CD3_2'] = [[326, 390, 454], [476, 580, 684]]
        self['DIC2_CD1_CD4_1'] = [[303, 346, 388], [565, 760, 946]]
        self['DIC2_CD2_CD4_1'] = [[326, 390, 454], [565, 760, 946]]
        self['DIC2_CD2_CD4_2'] = [[373, 437, 499], [565, 760, 946]]
        self['DIC2_CD2_CD4_3'] = [[373, 437, 499], [660, 860, 1060]]
        self['DIC2_CD1_CD4_1'] = [[303, 346, 388], [660, 860, 1060]]
        self['DIC2_CD2_CD4_1'] = [[326, 390, 454], [660, 860, 1060]]
        self['BLUE_CD1'] = [[303, 346, 388]]
        self['BLUE_CD2'] = [[373, 437, 499]]
        self['RED_CD3_1'] = [[414, 520, 621]]
        self['RED_CD3_2'] = [[476, 580, 684]]
        self['RED_CD3_3'] = [[500, 600, 705]]
        self['RED_CD$_1'] = [[660, 860, 1060]]

class UVESSet():
    def __init__(self, parent, name=None):
        self.parent = parent
        self.name = name
        print(self.name)
        self.gobject = None
        self.setData()
        self.color = (44, 160, 44)

    def setData(self):
        x, y = [], []
        for d in self.parent.UVESSetups[self.name]:
            xi = np.linspace(d[0]*10, d[2]*10, 100)
            x = np.append(x, xi)
            y = np.append(y, 3/4*(1 - ((2*xi - 10*d[0] - 10*d[2])/(10*d[2] - 10*d[0]))**2))

        print(x, y)
        self.data = gline(x=x, y=y)
        self.ymax_pos = np.argmax(y)

    def update(self, level):
        self.gobject.setData(x=self.data.x, y=level * self.data.y)

    def set_gobject(self, level):
        self.gobject = pg.PlotCurveItem(x=self.data.x, y=level * self.data.y, pen=pg.mkPen(color=self.color, width=2),
                                        fillLevel=0, brush=pg.mkBrush(self.color + (50,)))
        self.label = pg.TextItem(text=self.name , anchor=(0, 1.2), color=self.color)
        self.label.setFont(QFont("SansSerif", 16))
        self.label.setPos(self.data.x[self.ymax_pos], level * self.data.y[self.ymax_pos])

class obsobject():
    def __init__(self, name='J0000', ra=0.0, dec=0.0):
        self.name = name
        self.ra = ra
        self.dec = dec

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.__str__())

def observability(cand=[], site='VLT', time=['2017-09-01T00:00:00.00', '2018-03-01T00:00:00.00'], airmass=1.3):
    """
    cand is class object with parameters: name, ra, dec
    """
    # set observation site
    if site == 'VLT':
        if 0:
            longitude = '-70d24m12.000s'
            latitude = '-24d37m34.000s'
            elevation = 2635 * u.m
            vlt = EarthLocation.from_geodetic(longitude, latitude, elevation)
            observer = astroplan.Observer(name='VLT',
                                          location=vlt,
                                          pressure=0.750 * u.bar,
                                          relative_humidity=0.11,
                                          temperature=0 * u.deg_C,
                                          timezone=timezone('America/Santiago'),
                                          description="Very Large Telescope, Cerro Paranal")
            eso = astroplan.Observer.at_site('eso')
        else:
            observer = Observer.at_site('Cerro Paranal')

    if site == 'MagE':
        observer = Observer.at_site('las campanas observatory')
    if site == 'keck':
        observer = Observer.at_site('keck')

    print(observer)
    # set time range constrains

    if isinstance(time, str):
        # all year
        if len(time) == 4:
            timerange = 'period'
            time_range = Time(time+"-01-01T00:00:00.00", time+"-12-31T23:59:00.00")
        else:
            timerange = 'onenight'
            time = Time(time)

    elif isinstance(time, list):
        if len(time) == 2:
            timerange = 'period'
            print(Time(['2017-01-03']), time)
            time_range = Time(time)
        else:
            timerange = 'onenight'
            time = Time(time[0])

    if timerange == 'onenight':
        # calculate sunset and sunrise
        sunset = observer.sun_set_time(time, which='nearest')
        print('Sunset at ', sunset.iso)
        sunrise = observer.sun_rise_time(time, which='nearest')
        print('Sunrise at ', sunrise.iso)
        time_range = Time([sunset, sunrise])

        # set time array during the night
        time = time_range[0] + (time_range[1] - time_range[0]) * np.linspace(0, 1, 55)

    print(time)
    # set visibility constrains
    # constraints = [AirmassConstraint(1.5), AtNightConstraint.twilight_civil()]
    print(airmass)
    constraints = [AirmassConstraint(airmass), AtNightConstraint.twilight_civil()]

    # set parameters of calculations
    read_vis = 0
    if read_vis == 0:
        f_vis = open('DR12_cand_vis_temp.dat', 'w')
    month_detalied = 1
    show_moon = 0
    airmass_plot = 0
    sky_plot = 0
    if airmass_plot == 1:
        f, ax_air = plt.subplots()
    if sky_plot == 1:
        f, ax_sky = plt.subplots()

    targets = []

    if show_moon == 1:
        print(observer.moon_altaz(time).alt)
        print(observer.moon_altaz(time).az)
        # moon = SkyCoord(alt = observer.moon_altaz(time).alt, az = observer.moon_altaz(time).az, obstime = time, frame = 'altaz', location = observer.location)
        # print(moon.icrs)

    for i, can in enumerate(cand):

        print(can.name)
        # calculate target coordinates
        coordinates = SkyCoord(float(can.ra) * u.deg, float(can.dec) * u.deg, frame='icrs')
        #print(can.ra, can.dec)
        #print(coordinates.to_string('hmsdms'))
        target = FixedTarget(name=can.name, coord=coordinates)
        targets.append(target)

        # print(observer.target_is_up(time, targets[i]))
        # calculate airmass
        if timerange == 'onenight':
            if sky_plot == 1:
                plot_sky(target, observer, time)

            airmass = observer.altaz(time, target).secz
            if airmass_plot == 1:
                plot_airmass(target, observer, time, ax=ax)

            air_min = 1000
            k_min = -1
            for k, a in enumerate(airmass):
                if 0 < a < air_min:
                    air_min = a
                    k_min = k
            print(air_min, time[k_min].iso)

            if k_min > -1 and show_moon == 1:
                moon = SkyCoord(alt=observer.moon_altaz(time[k_min]).alt, az=observer.moon_altaz(time[k_min]).az,
                                obstime=time[k_min], frame='altaz', location=observer.location)
                can.moon_sep = Angle(moon.separation(target.coord)).to_string(fields=1)
                print(can.moon_sep)

            can.airmass = air_min
            can.time = time[k_min].iso

        # ever_observable = astroplan.is_observable(constraints, observer, targets, time_range=time_range)
        # print(ever_observable)

        if month_detalied == 1:
            tim = []
            months = ['2017-10-01', '2017-11-01', '2017-12-01', '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01']
            #for l in range(int(str(time_range[0])[5:7]), int(str(time_range[1])[5:7]) + 1):
            for l in range(len(months)-1):
                if 0:
                    start = "2017-" + "{0:0>2}".format(l) + "-01T00:00"
                    end = "2017-" + "{0:0>2}".format(l+1) + "-01T00:00"
                    if l == 12:
                        end = "2018-01-01T00:00"
                else:
                    start = months[l]
                    end = months[l+1]

                time_range_temp = Time([start, end])
                table = astroplan.observability_table(constraints, observer, [target],
                                                      time_range=time_range_temp)
                tim.append(table[0][3])

            # print(tim, max(tim), tim.index(max(tim)))
            print(tim)
            can.time = max(tim)

            if max(tim) != 0:
                if 0:
                    can.month = str(calendar.month_name[tim.index(max(tim)) + 1])[:3]
                else:
                    can.month = tim.index(max(tim))
                can.up = 'True'
            else:
                can.up = 'False'
                can.month = '---'

            print(can.up, can.month, can.time)

    if month_detalied == 0:
        table = astroplan.observability_table(constraints, observer, targets, time_range=time_range)
        print(table)
        for i, can in enumerate(cand):
            can.up = table[i][1]
            can.time = table[i][3]

    # print(table[k][0], table[k][1], table[k][2], table[k][3])
    #table.write('DR12_candidates_obs.dat', format='ascii')
    # f_out.write(table)

    if sky_plot == 1:
        plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
        plt.show()

    # sort candidates array

