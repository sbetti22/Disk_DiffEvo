# Disk_DiffEvo
diskFM MCFOST differential evolution disk modeling


Differential Evolution code for disk fitting with both JWST/NIRCam position dependent PSFs and regular PSFs
author: Sarah Betti

based off of disk modeling codes by: Johan Mazoyer (https://github.com/johanmazoyer/debrisdisk_mcmc_fit_and_plot) 
and Kellen Lawson ()

The make_model() function will call the MCFOST generator which is from: https://github.com/mperrin/mcfost-python.  All you need is the utils.py and paramfile.py.  The folder called mcfost MUST be in the same folder as this code to run! 

If you get an error "Don't know how to set a parameter named XX", then go to paramfile/set_parameter(), and add a new elif statement to call that parameter

for example 
        
	elif paramname == 'inclinations':
            self.RT_imin = value
            self.RT_imax = value


Each parameter tested will be saved to its own .csv file.  This should be relatively small (10s of KB).  

All free parameters are in the de_parameter_file.yaml

You also need an initialize_files folder with:
	1. starting MCFOST .para file
	2. an original PSF convolved model
	3. the klbasis .h5 file from diskFM
