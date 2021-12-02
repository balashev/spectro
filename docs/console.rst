.. _console:

Console
=======

Console command
---------------

The console is a powerful text commands based instrument

Here we provide the list of the commands:

* **<species>**: print the avaliable lines of the species (e.g. H I, Si IV, H2j0 etc) in spectro database.
* **show <species>**: add the line marks of avaliable lines of the species in the spectrum panel. The line marks will attach to the spectrum.
* **high <species>**: highlight the line marks corresponding to the species in the spectrum panel. 
* **hide <species>**: hide all the line marks corresponding to the species in the spectrum panel. 
* **z <some number>**: set redshift of the line marks
* **load <filename>**: load the file with <filename> from the temp folder 
* **fit <options>**: start fit of the model. options can be:

    * **comp**: fit only selected component
    * **mcmmc**: load MCMC window to run MCMC fitting procedure
    * **grid**: run profile likelihood estimation on the grid of the paramaters. Be careful, it is too long for number of paramaters more than 3.
    * **cont**: fit continuum

* **logN <species>**: calculate the total column densities along the components of the fit model
* **Me <species> <logNHI>**: calculate the metallicity of the <species> (based on the total column density along components) assuming the total HI column density, <logNHI>
* **depletion <species1> <species2>**: calculate the depletion of the <species1> relative to <species2>
* **abundance <species> <logNHI> <me>**: calculate expeceted column density of the <species> assuming the total HI column density, <logNHI>, and metallicity, <me>
* **x <num1> <num2>**: adjust the x-axis scale to <num1>..<num2>
* **y <num1> <num2>**: adjust the y-axis scale to <num1>..<num2>
* **rescale <arg> <float>**: rescale the spectrum/continuum/error by <float> factor. It applies with a window region. The <arg> specifed what is rescaled:

    * ** **: rescale both flux and pixel uncertainties
    * **err** or **unc**: rescale only pixel uncertainties
    * **spline** or **cont**: rescale B-continuum
    