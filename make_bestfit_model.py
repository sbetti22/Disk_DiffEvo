# DISK FM DE functions
import os
import glob

import math as mt
import numpy as np  
import pandas as pd 

import astropy.io.fits as fits
from astropy.convolution import convolve
from astropy.wcs import FITSFixedWarning
from astropy.convolution import convolve
from astropy import constants as const
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.visualization import ZScaleInterval
interval = ZScaleInterval()

import pyklip.parallelized as parallelized
from pyklip.fmlib.diskfm import DiskFM
import pyklip.fm as fm
import pyklip.rdi as rdi

import time
from datetime import datetime
import subprocess
import mcfost

from scipy.optimize import rosen, differential_evolution
# from scipy.optimize import NonlinearConstraint, Bounds
import shutil
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
import disk_convolution as dc

def make_mcfost_model(SAVE_DIR, WAVELENGTH, INITIAL_PARAM_FILE, FILE_PREFIX, paramsdict):
    '''
    '''
    if 'dust_settling' in paramsdict.keys():
        paramsdict['dust_settling'] = int(round(paramsdict['dust_settling'],0))
    # if 'dust_mass' in  paramsdict.keys():
    #     paramsdict['dust_mass'] = [10**paramsdict['dust_mass'][0]]
    if 'dust_porosity' in  paramsdict.keys():
        porosity = paramsdict['dust_porosity'][0]
        ## from Brunngräber+2017  to calculate amin with dependence on porosity #### 
        amin = 0.414 * ((1-porosity)**(-0.508)) * (16**(0.685 * ((1-porosity)**(-0.168))))
        paramsdict['dust_amin'] = [amin]
    if 'surface_density' in  paramsdict.keys():
        paramsdict['gamma_exp'] = paramsdict['surface_density']

    print(paramsdict)
    paraPath_hash =  SAVE_DIR + '/mcfost_models/bestfit_model/'

    if os.path.exists(paraPath_hash):
        shutil.rmtree(paraPath_hash)

    os.mkdir(paraPath_hash)
    owd = os.getcwd()
    os.chdir(paraPath_hash) 
    
    mcfost.grid_generator(INITIAL_PARAM_FILE,
                          paramsdict, 
                          paraPath_hash,
                          filename_prefix=FILE_PREFIX,
                          start_counter='best_model') # , 
                          
    os.chdir(SAVE_DIR + '/mcfost_models/bestfit_model/')
    mod = f'{FILE_PREFIX}_best_model.para'

    print('running model')
    subprocess.call(f'mcfost {mod} -img {WAVELENGTH} -only_scatt >> {FILE_PREFIX}_imagemcfostout.txt', shell = True)
    
    # add a backup
    if not os.path.exists(SAVE_DIR + f'/mcfost_models/bestfit_model/data_{WAVELENGTH}/RT.fits.gz'):
        subprocess.call(f'mcfost {mod} -img {WAVELENGTH} -only_scatt >> {FILE_PREFIX}_imagemcfostout.txt', shell = True)
    
    os.chdir(owd)

    return SAVE_DIR + f'/mcfost_models/bestfit_model/data_{WAVELENGTH}/RT.fits.gz'

    

########################################################
def make_bestfit_model(SAVE_DIR, WAVELENGTH, DISKOBJ, REDUCED_DATA, NOISE, INITIAL_PARAM_FILE, FILE_PREFIX, INSTRUMENT, DISTANCE_STAR, MASK, ROLL_REF_ANGLE, OBJ_INFO, FILTER,
         obsdate, grid_shape, bestparams_dict, amplitude):
    """ 
    Parameters:
        x: list of parameters for the differential evolution

    Returns:
        Chi2
    """

    inst, tel_point, obj_params = dc.make_psfs(ROLL_REF_ANGLE, OBJ_INFO, FILTER, obsdate=obsdate)
    hdul_psfs = dc.make_psfgrid(inst, tel_point, grid_shape=grid_shape)

    print('making model!')

    wv = WAVELENGTH

    model_gz = make_mcfost_model(SAVE_DIR, wv, INITIAL_PARAM_FILE, FILE_PREFIX, bestparams_dict)
    modelpixelscale = fits.getheader(model_gz)['CDELT2'] * 3600
    # convolve model with PSF 
    if INSTRUMENT == 'NIRCam': 
        model_here_convolved = dc.convolve_disk(inst, tel_point, obj_params, hdul_psfs, model_gz, modelpixelscale, wv, DISTANCE_STAR)
        pad = (REDUCED_DATA.shape[0]//2) - (model_here_convolved.shape[0]//2)
        model_here_convolved = np.pad(model_here_convolved, pad) 
        if model_here_convolved.shape[0] != REDUCED_DATA.shape[0]:
            raise ValueError('something wrong with padding')
    else: # still needs to be regridded to right pixel scale
        model_here = fits.getdata(model_gz)[0,0,0,:,:]
        model_here_hdr = fits.getheader(model_gz)
        freq = const.c.value / (wv * 1e-6)
        disk_model_mJy_as2 = ( (1e26 * model_here  / freq) * 1e3 ) / (modelpixelscale**2.)
        xcen, ycen = model_here_hdr['CRPIX1'], model_here_hdr['CRPIX2']
        disk_model_mJy_as2[ycen-2:ycen+2, xcen-2:xcen+2] = 0
        model_here_convolved = convolve(disk_model_mJy_as2, hdul_psfs, boundary='wrap')    

    model_contrast = model_here_convolved * amplitude 
    
    DISKOBJ.update_disk(model_contrast)
    model_fm = DISKOBJ.fm_parallelized()[0]
    model_fits = np.copy(model_fm)
   
    XX = model_fm.shape[1]
    YY = model_fm.shape[0]
    Y,X = np.mgrid[:YY,:XX]

    data = np.copy(REDUCED_DATA)
    noise = np.copy(NOISE)
    mod = np.copy(model_fm)

    for DT in [data, noise, mod]:
        DT[MASK] = np.nan
    
    N = np.count_nonzero(~np.isnan(data))
    M = len(bestparams_dict.keys())+1
    v = N-M
    # reduced data have already been naned outside of the minimization
    # zone, so we don't need to do it also for model_fm

    chi2 = (1./v) * np.nansum(((data - mod) / noise)**2.)
    hdr = fits.getheader(model_gz)
    for key in bestparams_dict:
        hdr[key] = bestparams_dict[key][0]
    hdr['AMP'] = amplitude
    hdr['CHI2'] = chi2
    fits.writeto(os.path.dirname(model_gz) + '/diskfm.fits', model_fits,header=hdr, overwrite=True)

    return os.path.dirname(model_gz) + '/'

########################################################
def initialize_diskfm(KLBASIS, INITIALIZE_DIR, FILE_PREFIX, REDUCED_DATA):
    '''
    initialize the Diff Evo by prepearing the diskFM object 
        modeled from J. Mayozer debrisdisk_mcmc_fit_and_plot/diskfit_mcmc/initialize_diskfm()
    '''
    numbasis = [1]

    model_here_convolved = fits.getdata(INITIALIZE_DIR + '/' + FILE_PREFIX + '_FirstModelConvolved.fits')
    pad = (REDUCED_DATA.shape[0]//2) - (model_here_convolved.shape[0]//2)
    model_here_convolved = np.pad(model_here_convolved, pad, constant_values=np.nan) 
    print('model opened .. starting DiskFM')
    print(KLBASIS)
    diskobj = DiskFM(None,
                     numbasis,
                     None,
                     model_here_convolved,
                     basis_filename=KLBASIS,
                     load_from_basis=True)
    # test the diskFM object
    diskobj.update_disk(model_here_convolved)
    print('done diskfM')

    return diskobj

########################################################

