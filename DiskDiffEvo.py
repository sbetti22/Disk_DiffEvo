"""
Differential Evolution code for disk fitting with both JWST/NIRCam position dependent PSFs and regular PSFs
author: Sarah Betti

based off of disk modeling codes by: Johan Mazoyer (https://github.com/johanmazoyer/debrisdisk_mcmc_fit_and_plot) 
and Kellen Lawson ()

"""
import os
import glob
import yaml

import multiprocessing
from multiprocessing import Process, Queue, Lock
# multiprocessing.set_start_method("fork")
# with python >3.8, mac multiprocessing switched to spawn so global variables do not work.  switch to fork instead.  

import math as mt
import numpy as np  
import pandas as pd 
import matplotlib

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
import mcfost # mcfost-master analysis code from https://github.com/mperrin/mcfost-python, The folder called mcfost MUST be in the same folder as this code to run! 

from scipy.optimize import differential_evolution
import shutil
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from numpy.random import rand

import disk_convolution as dc
import DE_plot


class DiskDiffEvo_creator:    
    def __init__(self, params_file, initialize_DE=True):
        # open the parameter file
        with open(params_file, 'r') as yaml_file:
            self.params_de_yaml = yaml.safe_load(yaml_file)
    
        # L = Lock()
        self.PATH = self.params_de_yaml['PATH'] # path to data
        self.DATA_DIR = self.params_de_yaml['DATA_DIR'] # location of raw datasets
        self.DISKFM_DIR = self.params_de_yaml['DISKFM_DIR']  # main directory to save mcfost models
        self.SAVE_DIR = self.params_de_yaml['SAVE_DIR']  # actual directory to save mcfost models
        self.FILE_PREFIX = self.params_de_yaml['FILE_PREFIX']   # name of level 2 data - roll 9
        self.REDUCEDDATA_DIR = self.params_de_yaml['REDUCEDDATA_DIR'] # path to reduced RDI data
        self.NOISE_DIR  =  self.params_de_yaml['NOISE_DIR'] 

        self.INITIAL_PARA = self.params_de_yaml['INITIAL_PARA'] #  

        self.WAVELENGTH = self.params_de_yaml['WAVELENGTH'] # \m
        self.FILTER = self.params_de_yaml['FILTER'] # 
        self.INSTRUMENT = self.params_de_yaml['INSTRUMENT'] # 
        print('wavelength:', self.WAVELENGTH)
            
        # load DISTANCE_STAR & PIXSCALE_INS and make them global
        self.DISTANCE_STAR = self.params_de_yaml['DISTANCE_STAR'] #59
        self.PIXSCALE_INS = self.params_de_yaml['PIXSCALE_INS'] #0.031217475
                    
        # load reduced_data and make it a global variable
        REDUCED_DATA_MJYSR = fits.getdata(self.REDUCEDDATA_DIR + '/' + self.params_de_yaml['REDUCED_DATA'])[2]  ### we take only the third KL mode
        # convert to mJy/as2
        REDUCED_DATA_MJYSR = REDUCED_DATA_MJYSR * u.MJy/u.steradian
        REDUCED_DATA = REDUCED_DATA_MJYSR.to(u.mJy/u.arcsecond**2)
        self.REDUCED_DATA = REDUCED_DATA.value

        # # load noise and make it global
        NOISE_MJYSR = fits.getdata(self.NOISE_DIR + '/' + self.params_de_yaml['NOISE']) # this is the ANNULUS STD NOISE
        # convert to mJy/as2
        NOISE_MJYSR = NOISE_MJYSR * u.MJy/u.steradian
        NOISE = NOISE_MJYSR.to(u.mJy/u.arcsecond**2)
        self.NOISE = NOISE.value

        self.bounds = list(self.params_de_yaml['BOUNDS'].values()) 
        self.csv_names = list(self.params_de_yaml['BOUNDS'].keys()) 
        self.gridgenerator_names = self.params_de_yaml['GRIDGEN_DICT']
        self.labels = self.params_de_yaml['LABELS']

        self.MASK = self.create_mask()
        self.nu = np.count_nonzero(self.MASK)-len(self.bounds)

        
        self.initialize_DE = initialize_DE

        if self.initialize_DE:
            if not os.path.exists(self.SAVE_DIR):
                os.makedirs(self.SAVE_DIR)

            print('starting PSF creation')
            print(self.bounds)
            print('starting diskobj')   
            self.DISKOBJ = self.initialize_diskfm()

            self.csv_fils  = glob.glob(self.SAVE_DIR + '/*.csv')
            print(f'deleting {len(self.csv_fils)} .csv files')
            for mod in self.csv_fils:
                os.remove(mod)

            self.mcfost_fils  = glob.glob(self.SAVE_DIR + '/*')
            print(f'deleting {len(self.mcfost_fils)} mcfost files')
            for mod in self.mcfost_fils:
                shutil.rmtree(mod)

            self.obsdate=self.params_de_yaml['obsdate'] #'2023-08-24T22:49:38.762'
            self.grid_shape=self.params_de_yaml['grid_shape']#'circle'
            self.ROLL_REF_ANGLE  = self.params_de_yaml['ROLL_REF_ANGLE']#261.370
            self.OBJ_INFO =self.params_de_yaml['OBJ_INFO']

            self.inst, self.tel_point, self.obj_params = dc.make_psfs(self.ROLL_REF_ANGLE, self.OBJ_INFO, self.FILTER, obsdate=self.obsdate)
            self.hdul_psfs = dc.make_psfgrid(self.inst, self.tel_point, grid_shape=self.grid_shape)

    def make_model(self, x):
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
        for i, key in enumerate(self.gridgenerator_names):
            if key == 'dust_settling':
                param = int(round(x[i],0))
            else:
                param = x[i]
            paramsdict[key] = [param]
        print(paramsdict)
        
        # creates a unique folder name to save the model into
        hash_string = str(hash(np.array2string(np.array(x)))) + str(hash(np.array2string(np.random.rand(len(self.gridgenerator_names))))) 
        # Hash the values to shrink the size, 2nd part is additional hash values to avoid same names

        paraPath_hash =  self.SAVE_DIR + '/mcfost_model_' + hash_string + '/'
        
        # remove the folder if it already exists
        if os.path.exists(paraPath_hash):
            shutil.rmtree(paraPath_hash)

        # make the folder and move into it
        os.mkdir(paraPath_hash)
        owd = os.getcwd()
        os.chdir(paraPath_hash) 
        
        # create the params file using the mcfost-master grid_generator()
        mcfost.grid_generator(self.INITIAL_PARA,
                            paramsdict, 
                            paraPath_hash,
                            filename_prefix=self.FILE_PREFIX,
                            start_counter=hash_string) 
                            
        # go into the folder 
        os.chdir(self.SAVE_DIR + f'/mcfost_model_{hash_string}/')
        mod = f'{self.FILE_PREFIX}_{hash_string}.para'

        # run mcfost to create scattered light image
        subprocess.call(f'mcfost {mod} -img {self.WAVELENGTH} -only_scatt >> {self.FILE_PREFIX}_imagemcfostout.txt', shell = True)
        
        # add a backup in case it fails
        if not os.path.exists(self.SAVE_DIR + f'/mcfost_model_{hash_string}/data_{self.WAVELENGTH}/RT.fits.gz'):
            subprocess.call(f'mcfost {mod} -img {self.WAVELENGTH} -only_scatt >> {self.FILE_PREFIX}_imagemcfostout.txt', shell = True)
        
        # go back to original directory
        os.chdir(owd)

        # return newly created model!
        return self.SAVE_DIR + f'/mcfost_model_{hash_string}/data_{self.WAVELENGTH}/RT.fits.gz'
    

    ########################################################
    def __call__(self, x, *args):
        """ 
        Parameters:
            x: list of parameters for the differential evolution

        Returns:
            Chi2
        """
        if not args:
            SAVE=False

        # create model and get filepath
        model_gz = self.make_model(x[:-1])

        # get mcfost pixel scale
        modelpixelscale = fits.getheader(model_gz)['CDELT2'] * 3600

        # convolve model with PSF 
        if self.INSTRUMENT == 'NIRCam': 
            model_here_convolved = dc.convolve_disk(self.inst, self.tel_point, self.obj_params, self.hdul_psfs, model_gz, modelpixelscale, self.WAVELENGTH, self.DISTANCE_STAR)
            pad = (self.REDUCED_DATA.shape[0]//2) - (model_here_convolved.shape[0]//2)
            model_here_convolved = np.pad(model_here_convolved, pad) #160 mJy/as2
            if model_here_convolved.shape[0] != self.REDUCED_DATA.shape[0]:
                raise ValueError('something wrong with padding')
        else: # still needs to be regridded to right pixel scale
            model_here = fits.getdata(model_gz)[0,0,0,:,:]
            model_here_hdr = fits.getheader(model_gz)
            freq = const.c.value / (self.WAVELENGTH * 1e-6)
            disk_model_mJy_as2 = ( (1e26 * model_here  / freq) * 1e3 ) / (modelpixelscale**2.)
            xcen, ycen = model_here_hdr['CRPIX1'], model_here_hdr['CRPIX2']
            disk_model_mJy_as2[ycen-2:ycen+2, xcen-2:xcen+2] = 0
            model_here_convolved = convolve(disk_model_mJy_as2, self.hdul_psfs, boundary='wrap')    

        # scale
        model_contrast = model_here_convolved * x[-1] 
        
        # do diskFM
        self.DISKOBJ.update_disk(model_contrast)
        model_fm = self.DISKOBJ.fm_parallelized()[0]
        model_fits = np.copy(model_fm)
    
        data = np.copy(self.REDUCED_DATA)
        noise = np.copy(self.NOISE)
        mod = np.copy(model_fm)

        # mask data so just disk is used in chi2
        for DT in [data, noise, mod]:
            DT[self.MASK] = np.nan
        
        # calculate chi2
        N = np.count_nonzero(~np.isnan(data))
        M = len(x)
        v = N-M
        chi2 = (1./v) * np.nansum(((data - mod) / noise)**2.)

        # save parameters to .csv file.  each model gets its own .csv file b/c I can't figure out how to open/write/save 1 file while multiprocessing...
        file_path = model_gz.split(f'data_{self.WAVELENGTH}')[0]

        data_dict = {}
        data_dict['model']= [file_path.split('/')[-2]]
        for i, key in enumerate(self.csv_names):
            data_dict[key] = [x[i]]
        data_dict['CHI2'] = [chi2]

        new_df = pd.DataFrame(data_dict)
        new_df.to_csv(file_path[:-1] +  '.csv')
        print('chi2: ', chi2)
        print(data_dict)

        # L.aquire()
        # DFIN = pd.read_csv(os.path.dirname(self.SAVE_DIR) + '/params.csv')
        # DFIN = pd.concat([DFIN, new_df])
        # DFIN.to_csv(os.path.dirname(self.SAVE_DIR) + '/params.csv',index=False)
        # L.release()

        if SAVE:
            hdr = fits.getheader(model_gz)
            for key in data_dict:
                hdr[key] = data_dict[key]
            fits.writeto(os.path.dirname(model_gz) + '/diskfm.fits', model_fits,header=hdr, overwrite=True)
        else: # delete fits files and folder  
            print('deleting fits: ', file_path.split('/')[-2])
            shutil.rmtree(file_path)

        return chi2 

       
    ########################################################
    def initialize_diskfm(self):
        '''
        initialize the Diff Evo by prepearing the diskFM object 
            modeled from J. Mayozer debrisdisk_mcmc_fit_and_plot/diskfit_mcmc/initialize_diskfm()
        '''
        numbasis = [3]

        model_here_convolved = fits.getdata(self.DISKFM_DIR + '/initialize_files/' + self.FILE_PREFIX + '_FirstModelConvolved.fits')
        pad = (self.REDUCED_DATA.shape[0]//2) - (model_here_convolved.shape[0]//2)
        model_here_convolved = np.pad(model_here_convolved, pad, constant_values=np.nan) #160
        print('model opened .. starting DiskFM')
        diskobj = DiskFM(None,
                        numbasis,
                        None,
                        model_here_convolved,
                        basis_filename=self.DISKFM_DIR + '/initialize_files/' + self.FILE_PREFIX + '_klbasis.h5',
                        load_from_basis=True)
        print('staring update_disk')
        # test the diskFM object
        diskobj.update_disk(model_here_convolved)
        print('done diskfM')

        # print('initialize .csv to save parameters')
        # data = {}
        # data['model'] = []
        # for key in self.csv_names:
        #     data[key] = []
        # data['CHI2'] = []
        # # data = {'model':[],'I':[], 'H0':[], 'MDUST':[], 'BETA':[], 'F':[], 'AMIN':[], 'AEXP':[], 'POROSITY':[], 'DUSTSET':[], 'ALPH_VIS':[], 'AMP':[], 'CHI2':[]}
        # df = pd.DataFrame(data)
        # df.to_csv(os.path.dirname(self.SAVE_DIR) + '/params.csv',index=False)

        return diskobj

    ########################################################
    def create_mask(self, sigma=4, levels=[0.005]):
        # make mask image for chi2 calculation
        result = gaussian_filter(self.REDUCED_DATA, sigma=sigma)  # Round 4 used sigma=4
        plt.figure()
        cs = plt.contour(result, levels=levels, colors='white', zorder=100, linewidths=1)
        verts = cs.collections[0].get_paths()[0]
        mask = verts.contains_points(list(np.ndindex(result.shape)))
        mask = mask.reshape(result.shape).T
        MASK = np.logical_not(mask)
        plt.close()
        return MASK


def run_DE(params_file, initialize_DE=True):
    DDE = DiskDiffEvo_creator(params_file, initialize_DE=initialize_DE)

    print('start DE')      

    # perform the differential evolution search
    start_time = datetime.now()
    t1 = time.time()
    SAVE = False

    result = differential_evolution(DDE, DDE.bounds, popsize=10, recombination=0.7, mutation=(0.5, 1.0), seed=5, polish=True, updating='deferred',workers=-1)
    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    # SAVE = True
    evaluation = DDE(solution, True)
    print('Solution: f(%s) = %.5f' % (solution, evaluation))

    t2 = time.time()
    end_time = datetime.now()
    print('started at: ', start_time)
    print('ended at: ', end_time)
    print('total time: ', t2-t1)

def make_plots(params_file, parameterspace_plot=True, final_image=True):
    DDE = DiskDiffEvo_creator(params_file, initialize_DE=False)
    # # do plotting and main .csv filing
    FIL = glob.glob(DDE.SAVE_DIR + '/*.csv')
    data, chi2, truths, MCFOST_DIR = DE_plot.pull_out_csvparams(FIL, all_csv=True, savefig = os.path.dirname(DDE.SAVE_DIR) + f'/FINAL_DEchi2_{DDE.FILE_PREFIX}.csv')

    min_val, max_val = 0.03,0.99
    n = 6
    orig_cmap = plt.cm.bone_r
    colors = orig_cmap(np.linspace(min_val, max_val, n))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)
    DE_plot.plot_parameterspace(len(truths), data, truths, chi2, DDE.bounds, DDE.labels, DDE.nu,
        vmin=0.26, vmax=.7, get_stats=False, cmap=cmap, figsize=(20,20), 
        title = DDE.FILE_PREFIX.replace('_', ' '),
        savefig=os.path.dirname(DDE.SAVE_DIR) + f'/{DDE.FILE_PREFIX}_params_space.png')

    BESTMOD_DIR = glob.glob(f'{DDE.SAVE_DIR}/*/data_2.0/')[0]
    DE_plot.make_final_images(os.path.dirname(DDE.SAVE_DIR), BESTMOD_DIR, DDE.REDUCED_DATA, DDE.NOISE, DDE.WAVELENGTH, DDE.FILE_PREFIX, 
                            title = DDE.FILE_PREFIX.replace('_', ' ') + ' ' + 'Best Fit model', 
                            savefig=os.path.dirname(DDE.SAVE_DIR) + f'/{DDE.FILE_PREFIX}_DEmodel.png')