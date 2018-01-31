#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

# calculates autocorrelation function of a time series
# uses a top hot function as the input time series
# and plots the autocorrelation function which will be
# a triangle function
#
# https://en.wikipedia.org/wiki/Autocorrelation#Estimation

def correlate(s1, s2=None, ax=None):
    """
    parameters:
        - s1     :  signal 1
        - s2     :  signal 2, if None than calculate autocorrelation of s1
        - ax     :  where to plot
    
    return:
        - corr   :  correlation array
    """
    # subtract the mean of the time series
    y1 = s1 - s1.mean()

    if s2 is not None:
        y2 = s2 - s2.mean()

    # correlate
    if s2 is None:
        corr = np.correlate(y1, y1, mode="full")
    else:
        corr = np.correlate(y1, y2, mode="full")

    # take only the second half of the time series because
    # correlate returns the autocorrelation from a negative time
    # and not zero
    corr = corr[corr.size / 2:]

    # normalize by the variance
    corr /= (s1.var() * np.arange(s1.size, 0, -1))

    if ax is not None:
        ax.plot(corr)

    return corr

# create signal
n = 300
x = np.arange(n)
s2 = np.random.normal(0, 1, size=n)
s1 = np.sin(x * np.pi / 10) + np.sin(x * np.pi / 23) + s2


corr1 = correlate(s1)
corr2 = correlate(s2)

# get sample times for plotting
t = np.arange(n/2)

# plot
fig, ax = plt.subplots()
ax.plot(t, corr1[n/2:])
ax.plot(t, corr2[n/2:])
ax.set_ylabel("Autocorrelation Function")
ax.set_xlabel("Time (s)")
plt.show()