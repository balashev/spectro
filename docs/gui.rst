Graphical User Interface
========================

**Spectro** package contains **sviewer** - the graphical user interface for the interactive spectral data analysis. It is located in the separate folder "sviewer" and 
should be loaded by running ``sviewer/__main__.py``, for example 

    $ cd sviewer
    $ python __main__.py


Main vindow
-----------
The main window of **sviewer** are typically looks like

.. image:: ./images/main.png

where the numbers highlights the following parts of the GUI:

1. The main menu. For the whole descriptions see file
#. The residuals panel.  
#. The 1d spectrum window. Show the spectrum and the fit model.    

Residuals panel
--------------
It show the residuals between spectrum and the fit model. It has shared x-axis with the Spectrum panel.  **Residuals panel** can be activated/hide by pressing ``F4`` or ``view/Residuals`` in the Main Menu. The blue area and green line in the left of this panel show the kde of the residual distribution, that is calculated using the pixels from the whole spectrum and view window only. The red line shows the gaussian function with unit dispersion, that should be approaching in case of good fit. Note: that consistency between blue and red lines are not necessary means relaible fit, since you also control the structure in residuals. 

Spectrum panel
--------------

