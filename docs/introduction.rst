Introduction
============

.. _introduction:


What is spectro package?
------------------------

Spectro package is a package collects the scripts for the analysis of the astrophysical observations. It is main module ``sviewer`` provided powerful tool for the interactive spectral analysis. For the moment the majority of the tool is for the absorption line analysis. The graphical interface is based on PyQt and pyqtgraph packages, that are high-performance graphical packages. The base of the code is written in ``Python`` with use of the external high-level packages provided by astronomical community, however the code also allows one to use ``Julia`` language to improve extesive calculations of the absorption line profiles models and estimation of their parameters.   


What it can do?
---------------

The main features of spectro package is 

* Powerful :ref:`gui` for spetra analysis 
* Comprehensive visually guided absorption line modeling
* Parameter estimations of line profiles using Levenberg-Marquard and MCMC methods
* 1d spectrum reduction routines
* 2d to 1d spectral reduction
* synchronization with external databases, such as SDSS, NIST, ...
* calculation of popultaion ratio for atomic and molecular levels
* ISM neutral phase diagram calculation
* ...  

This documentation
------------------

The main part of this documentaion describes the ``sviewer`` - :ref:`gui` for the absorption line analysis, that is located in the ``sviewer``, including some tutorials for the basic introductions in absorption line analysis with ``sviewer``.
