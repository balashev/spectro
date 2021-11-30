Graphical User Interface
========================

**Spectro** package contains **sviewer** - the graphical user interface for the interactive spectral data analysis. It is located in the separate folder "sviewer" and 
should be loaded by running ``sviewer/__main__.py``, for example 

    $ cd sviewer
    $ python __main__.py


Main window
-----------
The main window of **sviewer** are typically looks like

.. image:: ./images/main.png

where the numbers highlights the following parts of the GUI:

1. Main menu
#. Residuals panel  
#. Spectrum panel    
#. Main control panel
#. Console
#. Status bar

Main menu
---------
For the whole descriptions see 

Residuals panel
--------------
It shows the residuals between spectrum and the fit model. It has shared x-axis with the Spectrum panel.  **Residuals panel** can be activated/hide by pressing ``F4`` or ``view/Residuals`` in the Main Menu. The blue area and green line in the left of this panel show the kde of the residual distribution, that is calculated using pixels from the whole spectrum and from the view window only, respectively. The red line shows the gaussian function with unit dispersion, that should be approaching in case of good fit. Note: that consistency between blue and red lines are not necessary means relaible fit, since you also control the structure in residuals. 

Spectrum panel
--------------
Tha main interactive window to work with the spectra. It shows the spectrum and the fit model and different graphical objects suitable for the spectral analysis


Main controls panel
-------------------

Console
-------
It allows to input commands moslty concerned with GUI management. For the whole descriptions see :ref:`console <console>`

Status bar
----------
Status bar shows some messages and indicate some useful numbers that are can be instructive during the fitting process, e.g. 
