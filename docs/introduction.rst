Introduction
============

.. _introduction:


What is spectro package?
------------------------

Spectro package is a package collecting the scripts for the analysis of the astrophysical observations. It is main module "sviewer" provided powerful tool for the interactive spectral analysis. For the moment the majority of the tool is for the absorption line analysis. The graphical interface is based on PyQt and pyqtgraph packages, that are high-performance graphical packages. The base of the code is written in ``Python`` and use the ability to use extrenal high-levels packages provided by astronomical community, however the code also allow to use ``Julia`` language to improve the heavy calculations of the absorption line profiles models and parameter estimation.   


What it can do?
---------------

The main features of spectro package is 

* Complex absorption line modeling
* Levenberg-Marquard and MCMC parameters estimations of line profiles
* 1d spectrum reduction routines
* 2d to 1d spectral reduction
* synchronization with external databases, such as SDSS, NIST, ...
* calculation of popultaion ratio for atomic and molecular levels
* ISM neutral phase diagram calculation
* ...  


