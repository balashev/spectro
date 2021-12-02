.. _console:

Console
=======

Console command
---------------

The console is a powerful text commands based instrument.

Here we provide the list of the commands:

* **<species>**: print the avaliable lines of the species (e.g. H I, Si IV, H2j0 etc) in spectro database.
* **show <species>**: add the line marks of avaliable lines of the species in the spectrum panel. The line marks will attach to the spectrum.
* **high <species>**: highlight the line marks corresponding to the species in the spectrum panel.
* **hide <species>**: hide all the line marks corresponding to the species in the spectrum panel.
* **z <float>**: set redshift of the line marks to be <float>.
* **load <filename>**: load the file with <filename> from the temp folder .
* **fit <options>**: start fit of the model. <options> can be:

    * **comp**: fit only selected component.
    * **mcmmc**: load MCMC window to run MCMC fitting procedure.
    * **grid**: run profile likelihood estimation on the grid of the paramaters. Be careful, it is too long for number of paramaters more than 3.
    * **cont**: fit continuum.

* **logN <species>**: calculate the total column densities along the components of the fit model.
* **me <species> <logNHI>**: calculate the metallicity of the <species> (based on the total column density along components) assuming the total HI column density, <logNHI>.
* **depletion <species1> <species2>**: calculate the depletion of the <species1> relative to <species2>.
* **abundance <species> <logNHI> <me>**: calculate expeceted column density of the <species> assuming the total HI column density, <logNHI>, and metallicity, <me>.
* **x <num1> <num2>**: adjust the x-axis scale to <num1>..<num2>.
* **y <num1> <num2>**: adjust the y-axis scale to <num1>..<num2>.
* **rescale <arg> <float>**: rescale the spectrum/continuum/error by <float> factor. It applies with a window region. The <arg> specifed what is rescaled:

    * : rescale both flux and pixel uncertainties.
    * **err** or **unc**: rescale only pixel uncertainties.
    * **spline** or **cont**: rescale B-continuum.

* **shift <arg> <float>**: shift spectrum by <float>, l = l + <float>. If <arg>=v than the shift is in velocity space, i.e. l = l * (1 + <float>/c).
* **set <arg1> <arg2> <float>**: set flux/unc/cont (set by <arg1>) to be equal <float> value. <arg2> can be:

    * **all**: apply for a whole range.
    * **screen** or **window** or **view** or **disp**: apply only within window range.
    * **region** or **regions** or **reg**: apply only for a region.

* **cont <float>**: set continuum to be <float>.
* **divide <spec1> <spec2>**: divide <spec1> on <spec2>.
* **substract <spec1> <spec2>**: substract <spec2> from <spec1>.
* **apply <arg>**: apply transormation of x-axis. <arg> can be:

    * **vac**: correction to vacuum.
    * **helio <float>**: correction to baryocentric wavelenghts, specified by <float> velocity.
    * **restframe**: convert wavlnghts for restframe, using redshift from z panel.

* **ston**: print signal to noize of the spectrum at the region items.
* **stats**: print signal to noize, dispersion of the pixels in the spectrum at the region items.
* **level**: print level of the spectrum in the regions (optimize to measure the line flux residuals).
* **ew**: print equivalent widths of the lines in line regions.
