"""
Differential Evolution code for disk fitting with both JWST/NIRCam position dependent PSFs and regular PSFs
author: Sarah Betti
formalized on: 12/20/2023 (originally written in 2020)

based off of disk modeling codes by: Johan Mazoyer (https://github.com/johanmazoyer/debrisdisk_mcmc_fit_and_plot), differential evolution technique by: Kellen Lawson (https://ui.adsabs.harvard.edu/abs/2020AJ....160..163L/abstract), and the pyKLIP (J. Wang; https://ui.adsabs.harvard.edu/abs/2015ascl.soft06001W/abstract) DiskFM tools (J. Mazoyer: https://ui.adsabs.harvard.edu/abs/2020SPIE11447E..59M/abstract) 

"""
########################################################
# CHANGE THIS TO THE PARAMETER FILE YOU WANT TO USE!!
yaml_paramater_file = 'de_parameter_file.yaml'
########################################################

import os
import glob
import yaml
import logging
logging = logging.getLogger('diffevo')

import multiprocessing
multiprocessing.set_start_method("fork")
# with python >3.8, mac multiprocessing switched to spawn so global variables do not work.  switch to fork instead.  

import math as mt
import numpy as np  
import pandas as pd 

import numpy.ma as ma

import matplotlib
import matplotlib.pyplot as plt

import astropy.io.fits as fits
from astropy.convolution import convolve
from astropy.wcs import FITSFixedWarning
from astropy.convolution import convolve
from astropy import constants as const
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.visualization import ZScaleInterval
interval = ZScaleInterval()

from pyklip.fmlib.diskfm import DiskFM

import time
from datetime import datetime

from scipy.optimize import differential_evolution

import shutil
import subprocess

import mcfost # mcfost-master analysis code from https://github.com/mperrin/mcfost-python, The folder called mcfost MUST be in the same folder as this code to run! 
import disk_convolution as dc
import DE_plot

########################################################

def make_model(x):
    '''
    make MCFOST parameter file and scattered light image at specific wavelength

    this function will call the MCFOST generator which is from: https://github.com/mperrin/mcfost-python.  All you need is the utils.py and paramfile.py.  The folder called mcfost MUST be in the same folder as this code to run! 
    If you get an error "Don't know how to set a parameter named XX", then go to paramfile/set_parameter(), and add a new elif statement to call that parameter

    for example: 
        elif paramname == 'inclinations':
            self.RT_imin = value
            self.RT_imax = value

    Parameters:
        all free parameters that the user wants!  
    '''
    # convert into dictionary for mcfost.grid_generator.  the key MUST match the name in mcfost.grid_generator code! 
    paramsdict = {}
    for i, key in enumerate(gridgenerator_names):
        if key == 'dust_settling':
            param = int(round(x[i],0))
        else:
            param = x[i]
        paramsdict[key] = [param]
    
    # creates a unique folder name to save the model into
    hash_string = str(hash(np.array2string(np.array(x)))) + str(hash(np.array2string(np.random.rand(len(gridgenerator_names))))) 
    # Hash the values to shrink the size, 2nd part is additional hash values to avoid same names

    paraPath_hash =  SAVE_DIR + '/mcfost_model_' + hash_string + '/'
    
    # remove the folder if it already exists
    if os.path.exists(paraPath_hash):
        shutil.rmtree(paraPath_hash)

    # make the folder and move into it
    os.mkdir(paraPath_hash)
    owd = os.getcwd()
    os.chdir(paraPath_hash) 
    
    # create the params file using the mcfost-master grid_generator()
    logging.info(f'creating grid for mcfost_model_{hash_string}')
    mcfost.grid_generator(INITIAL_PARA,
                          paramsdict, 
                          paraPath_hash,
                          filename_prefix=FILE_PREFIX,
                          start_counter=hash_string) 
                          
    # go into the folder 
    os.chdir(SAVE_DIR + f'/mcfost_model_{hash_string}/')
    mod = f'{FILE_PREFIX}_{hash_string}.para'

    # run mcfost to create scattered light image
    logging.info(f'creating mcfost model for mcfost_model_{hash_string}')
    subprocess.call(f'mcfost {mod} -img {WAVELENGTH} -only_scatt >> {FILE_PREFIX}_imagemcfostout.txt', shell = True)
    
    # add a backup in case it fails
    if not os.path.exists(SAVE_DIR + f'/mcfost_model_{hash_string}/data_{WAVELENGTH}/RT.fits.gz'):
        logging.warn(f'model failed first time...trying again for mcfost_model_{hash_string}')
        subprocess.call(f'mcfost {mod} -img {WAVELENGTH} -only_scatt >> {FILE_PREFIX}_imagemcfostout.txt', shell = True)
    
    # go back to original directory
    os.chdir(owd)

    # return newly created model!
    return SAVE_DIR + f'/mcfost_model_{hash_string}/data_{WAVELENGTH}/RT.fits.gz'
    
########################################################

def fobj(x):
    """ 
    main function to create, convolve, and calculate chi2 for differential evolution

    Parameters:
        x: list of parameters for the differential evolution

    Returns:
        Chi2
    """

    # create model and get filepath
    logging.info(f'making model with {x}')
    model_gz = make_model(x[:-1])

    # get mcfost pixel scale
    modelpixelscale = fits.getheader(model_gz)['CDELT2'] * 3600

    # convolve model with PSF 
    logging.info('convolving model for ' + model_gz.split('/')[-3])
    if INSTRUMENT == 'NIRCam': 
        model_here_convolved = dc.convolve_disk(inst, tel_point, obj_params, hdul_psfs, model_gz, modelpixelscale, WAVELENGTH, DISTANCE_STAR)
        pad = (REDUCED_DATA.shape[0]//2) - (model_here_convolved.shape[0]//2)
        model_here_convolved = np.pad(model_here_convolved, pad)
        if model_here_convolved.shape[0] != REDUCED_DATA.shape[0]:
            raise ValueError('something wrong with padding')
    else: # NOTE: still needs to be regridded to right pixel scale
        model_here = fits.getdata(model_gz)[0,0,0,:,:]
        model_here_hdr = fits.getheader(model_gz)
        freq = const.c.value / (WAVELENGTH * 1e-6)
        disk_model_mJy_as2 = ( (1e26 * model_here  / freq) * 1e3 ) / (modelpixelscale**2.)
        xcen, ycen = model_here_hdr['CRPIX1'], model_here_hdr['CRPIX2']
        disk_model_mJy_as2[ycen-2:ycen+2, xcen-2:xcen+2] = 0
        model_here_convolved = convolve(disk_model_mJy_as2, hdul_psfs, boundary='wrap')    

    # scale
    model_contrast = model_here_convolved * x[-1] 
    
    # do diskFM
    logging.info('performing diskFM for ' + model_gz.split('/')[-3])
    DISKOBJ.update_disk(model_contrast)
    model_fm = DISKOBJ.fm_parallelized()[0]
    model_fits = np.copy(model_fm)
   
    data = np.copy(REDUCED_DATA)
    noise = np.copy(NOISE)
    mod = np.copy(model_fm)

    # mask data so just disk is used in chi2
    for DT in [data, noise, mod]:
        DT[MASK] = np.nan
    
    # calculate chi2
    N = np.count_nonzero(~np.isnan(data))
    M = len(x)
    v = N-M
    chi2 = (1./v) * np.nansum(((data - mod) / noise)**2.)
    logging.info('chi2 for ' + model_gz.split('/')[-3] + f' : {chi2}')
    # save parameters to .csv file.  each model gets its own .csv file b/c I can't figure out how to open/write/save 1 file while multiprocessing...
    file_path = model_gz.split(f'data_{WAVELENGTH}')[0]

    data_dict = {}
    data_dict['model']= [file_path.split('/')[-2]]
    for i, key in enumerate(csv_names):
        data_dict[key] = [x[i]]
    data_dict['CHI2'] = [chi2]

    new_df = pd.DataFrame(data_dict)
    logging.info('saving to ' +  file_path.split('/')[-2] + '.csv')
    new_df.to_csv(file_path[:-1] +  '.csv')

    if SAVE: #if its the best fit model, save it, otherwise, delete the folder with fits files to save space
        hdr = fits.getheader(model_gz)
        for key in data_dict:
            hdr[key] = data_dict[key][0]
        fits.writeto(os.path.dirname(model_gz) + '/diskfm.fits', model_fits,header=hdr, overwrite=True)
    else: # delete fits files and folder  
        logging.info('deleting folder: ' + file_path.split('/')[-2])
        shutil.rmtree(file_path)

    return chi2 

########################################################

def initialize_diskfm():
    '''
    initialize the Diff Evo by prepearing the diskFM object 
        modeled from J. Mayozer debrisdisk_mcmc_fit_and_plot/diskfit_mcmc/initialize_diskfm()
    '''
    numbasis = [3]

    model_here_convolved = fits.getdata(INITIALIZE_DIR + '/' + FILE_PREFIX + '_FirstModelConvolved.fits')
    pad = (REDUCED_DATA.shape[0]//2) - (model_here_convolved.shape[0]//2)
    model_here_convolved = np.pad(model_here_convolved, pad, constant_values=np.nan) 
    logging.info('model opened .. starting DiskFM')
    diskobj = DiskFM(None,
                     numbasis,
                     None,
                     model_here_convolved,
                     basis_filename=INITIALIZE_DIR + '/' + FILE_PREFIX + '_klbasis.h5',
                     load_from_basis=True)
    # test the diskFM object
    diskobj.update_disk(model_here_convolved)
    logging.info('done diskfM')

    return diskobj

########################################################

# open the parameter file
logging.info('getting parameters')
with open(yaml_paramater_file, 'r') as yaml_file:
    params_de_yaml = yaml.safe_load(yaml_file)

# extract all parameters! 
PATH = params_de_yaml['PATH'] # path to data
DATA_DIR = params_de_yaml['DATA_DIR'] # location of raw datasets
DISKFM_DIR = params_de_yaml['DISKFM_DIR']  # main directory to save mcfost models
SAVE_DIR = params_de_yaml['SAVE_DIR']  # actual directory to save mcfost models
FILE_PREFIX = params_de_yaml['FILE_PREFIX']   # name of level 2 data - roll 9
REDUCEDDATA_DIR = params_de_yaml['REDUCEDDATA_DIR'] # path to reduced RDI data
NOISE_DIR  =  params_de_yaml['NOISE_DIR'] # path to noise annuli RDI data
INITIALIZE_DIR = params_de_yaml['INITIALIZE_DIR'] # path to initial files

INITIAL_PARA = INITIALIZE_DIR + '/' + params_de_yaml['INITIAL_PARA'] # initial .para file

WAVELENGTH = params_de_yaml['WAVELENGTH'] # in micron
FILTER = params_de_yaml['FILTER'] # name of filter
INSTRUMENT = params_de_yaml['INSTRUMENT'] # instrument
logging.info(f'performing Differential Evolution on : {INSTRUMENT}/{FILTER}')
    
DISTANCE_STAR = params_de_yaml['DISTANCE_STAR']  # pc
PIXSCALE_INS = params_de_yaml['PIXSCALE_INS']  # arcsecond/pixel
            
# load reduced_data and make it a global variable
pyklip_frame = params_de_yaml['PYKLIP_FRAME'] # # image to use if data has more than one extension
REDUCED_DATA_MJYSR = fits.getdata(REDUCEDDATA_DIR + '/' + params_de_yaml['REDUCED_DATA'])[pyklip_frame]  
# convert to mJy/as2
REDUCED_DATA_MJYSR = REDUCED_DATA_MJYSR * u.MJy/u.steradian
REDUCED_DATA = REDUCED_DATA_MJYSR.to(u.mJy/u.arcsecond**2)
REDUCED_DATA = REDUCED_DATA.value

# # load noise and make it global
NOISE_MJYSR = fits.getdata(NOISE_DIR + '/' + params_de_yaml['NOISE']) # this is the ANNULUS STD NOISE
# convert to mJy/as2
NOISE_MJYSR = NOISE_MJYSR * u.MJy/u.steradian
NOISE = NOISE_MJYSR.to(u.mJy/u.arcsecond**2)
NOISE = NOISE.value

bounds = list(params_de_yaml['BOUNDS'].values())  # initial bounds
csv_names = list(params_de_yaml['BOUNDS'].keys())  # name of bounds to save in csv file
gridgenerator_names = params_de_yaml['GRIDGEN_DICT'] # name of free parameters for gridgenerator in mcfost/paramfile.py
labels = params_de_yaml['LABELS'] # labels for corner plot 

MASK = fits.getdata(INITIALIZE_DIR + '/' + FILE_PREFIX + '_MASK.fits') # mask for calculating chi2
nu = np.count_nonzero(MASK)-len(bounds) # nu for chi2 calculation
MASK = ma.make_mask(MASK)

# make save directory if it does not exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

logging.info(f'bounds {bounds}')

# delete exisiting .csv and fits files in save_dir if they exist
csv_fils  = glob.glob(SAVE_DIR + '/*.csv')
logging.info(f'deleting {len(csv_fils)} .csv files')
for mod in csv_fils:
    os.remove(mod)

mcfost_fils  = glob.glob(SAVE_DIR + '/*')
logging.info(f'deleting {len(mcfost_fils)} mcfost files')
for mod in mcfost_fils:
    shutil.rmtree(mod)

logging.info('starting diskobj')   
DISKOBJ = initialize_diskfm()

logging.info('starting PSF creation')
obsdate=params_de_yaml['obsdate'] 
grid_shape=params_de_yaml['grid_shape']
ROLL_REF_ANGLE  = params_de_yaml['ROLL_REF_ANGLE']
OBJ_INFO =params_de_yaml['OBJ_INFO']

inst, tel_point, obj_params = dc.make_psfs(ROLL_REF_ANGLE, OBJ_INFO, FILTER, obsdate=obsdate)
hdul_psfs = dc.make_psfgrid(inst, tel_point, grid_shape=grid_shape)

########################################################
    
logging.info('start DE')      
# perform the differential evolution search
start_time = datetime.now()
logging.info(f'STARTING TIME: {start_time}')
t1 = time.time()
SAVE=False 
## SINGLE - only for a very very small and fast grid
# result = differential_evolution(fobj, bounds, popsize=10, recombination=0.7, mutation=(0.5, 1.0), seed=5, polish=True)

## PARALLEL - highly recommended unless it is a very very small and fast grid
result = differential_evolution(fobj, bounds, popsize=10, recombination=0.7, mutation=(0.5, 1.0), seed=5, polish=True, updating='deferred',workers=-1)

# summarize the result
logging.info('Status : %s' % result['message'])
logging.info('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
SAVE=True
evaluation = fobj(solution)
logging.info('Solution: f(%s) = %.5f' % (solution, evaluation))

t2 = time.time()
end_time = datetime.now()
logging.info('started at: ', start_time)
logging.info('ended at: ', end_time)
logging.info('total time: ', t2-t1)

########################################################
     
# do plotting and main .csv filing
logging.info('starting plotting')
# extract all parameters from .csv file and put them into 1 file.  if clean_up=True, the original .csv files will be deleted leaving only the combined file. 
clean_up = params_de_yaml['CLEAN_UP']
FIL = glob.glob(SAVE_DIR + '/*.csv')
data, chi2, truths, MCFOST_DIR = DE_plot.pull_out_csvparams(FIL, all_csv=True, savefig = os.path.dirname(SAVE_DIR) + f'/FINAL_DEchi2_{FILE_PREFIX}.csv',
                                                            clean_up=clean_up)

# plot a corner plot of the parameter space
min_val, max_val = 0.03,0.99
n = 6
orig_cmap = plt.cm.bone_r
colors = orig_cmap(np.linspace(min_val, max_val, n))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)
DE_plot.plot_parameterspace(len(truths), data, truths, chi2, bounds, labels, nu,
     vmin=0.26, vmax=.7, get_stats=False, cmap=cmap, figsize=(20,20), 
     title = FILE_PREFIX.replace('_', ' '),
     savefig=os.path.dirname(SAVE_DIR) + f'/{FILE_PREFIX}_params_space.png')

# plot the best fit image and residuals
BESTMOD_DIR = glob.glob(f'{SAVE_DIR}/*/data_{WAVELENGTH}/')[0]
DE_plot.make_final_images(os.path.dirname(SAVE_DIR), BESTMOD_DIR, REDUCED_DATA, NOISE, WAVELENGTH, FILE_PREFIX, MASK=MASK, 
                          title = FILE_PREFIX.replace('_', ' ') + ' ' + 'Best Fit model', 
                          savefig=os.path.dirname(SAVE_DIR) + f'/{FILE_PREFIX}_DEmodel.png',
                          make_files=False)

logging.info("FINISHED!")