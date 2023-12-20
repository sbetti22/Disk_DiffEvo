# Disk_DiffEvo

Differential Evolution code for disk fitting with both JWST/NIRCam position dependent PSFs and regular PSFs
author: Sarah Betti

based off of disk modeling codes by: Johan Mazoyer (https://github.com/johanmazoyer/debrisdisk_mcmc_fit_and_plot), differential evolution technique by: Kellen Lawson (https://ui.adsabs.harvard.edu/abs/2020AJ....160..163L/abstract), and the pyKLIP (J. Wang; https://ui.adsabs.harvard.edu/abs/2015ascl.soft06001W/abstract) DiskFM tools (J. Mazoyer: https://ui.adsabs.harvard.edu/abs/2020SPIE11447E..59M/abstract) 

---
The main module is DiskDiffEvo.py with all free variables listed in the de_parameter_file.yaml. The differential evolution is performed using the scipy.optimize.differential_evolution() function.  

In order to create the initial files necessary to run DiskDiffEvo (inc. a First convolved model and the klbasis for diskFM, a mask for calculating chi2), run through the make_initialize_files.ipynb.

The initial files should be:
	1. starting MCFOST .para file
	2. an original PSF convolved model
	3. the klbasis .h5 file from diskFM
 	4. a mask .FITS file with 0 indicating disk pixels (i.e. where you want to calculate chi2) and 1 indicating background pixels to ignore.  These will be set to np.nan.  

NOTE: as of 12/20/2023, convolving with a non-NIRCam PSF does not rebin into the correct pixel size.  

Within DiskDiffEvo.py, the make_model() function will call the MCFOST generator from: https://github.com/mperrin/mcfost-python.  All you need is the utils.py and paramfile.py.  The folder called mcfost with the paramfile.py module MUST be in the same folder as this code to run! 

If you get an error "Don't know how to set a parameter named XX", then go to paramfile/set_parameter(), and add a new elif statement to call that parameter

for example 
        
	elif paramname == 'inclinations':
            self.RT_imin = value
            self.RT_imax = value

Each parameter set run through the differential evolution code will be saved to its own .csv file.  This should be relatively small (10s of KB).  

The original data and noise must be in MJy/steradian.  If they are not, go into DiskDiffEvo.py and change astropy units.  The noise should be the standard deviation within an annulus of the noise map (see make_initialize_files.ipynb).

---
The disk_convolution.py script is only needed if you are doing JWST/NIRCam PSF convolution.  If so, you need:	
 	- webbpsf_ext (https://github.com/JarronL/webbpsf_ext)
  	- webpsf (https://github.com/spacetelescope/webbpsf)
   
