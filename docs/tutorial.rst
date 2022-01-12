.. _tutorial:

Tutorial
========

Here we provide a brief tutorial to start fitting with the ``sviewer`` GUI.

Loading the spectrum
--------------------

The spectrum can be loaded in GUI by several ways:

* Drag and drop method. You can load several files in one drop.

* Main menu: File/Import spectrum...

* Main menu: File/Open... In that case it load the spectra specified in .spv file, whih contains saved progress of data analysis.

* Main menu: File/Import list...: import a list of spectra from the file with list contains of path to spectra.

* Main menu: File/Import folder...: import all spectra from the specified folder. 

The basic ``sviewer`` spectral format is plain ``FITS`` file, which stored spectrum in column-like representation with the following order: <wavelength> <flux> <uncertainty>. The spectrum do not necessary need to have uncertainties column, as well as can read the continuum provided in the forth column. The header of ascii file should be commented out by **#** 

Additionally there is possibility to load the spectrum in ``FITS`` format. The program can automatically recognize several FITS format produced by standard reduction routines, such as UVES popler, SDSS, etc.

.. _constructing-continuum:
Constructing continuum
----------------------


Making fit model
----------------

Fitting
-------

Viewing results
---------------