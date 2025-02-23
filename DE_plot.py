"""
Disk plotting code for DiskDiffEvo solution
author: Sarah Betti
formalized on: 12/20/2023 (originally written in 2020)

based off of PSF modeling by: Kellen Lawson (https://ui.adsabs.harvard.edu/abs/2023AJ....166..150L/graphics) and implemented by Sarah Betti (https://ui.adsabs.harvard.edu/abs/2022AJ....163..145B/abstract)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.convolution import convolve
from astropy import constants as const
from astropy import units as u
from astropy.visualization import simple_norm, ZScaleInterval
from astropy.table import QTable
interval = ZScaleInterval()

import os
import copy
import glob
import yaml

from scipy import stats
import scipy.ndimage as ndimage

import matplotlib
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

import numpy.ma as ma
import make_bestfit_model as mbm
from pyklip.fmlib.diskfm import DiskFM

def pull_out_csvparams(FIL, **kwargs): 
    '''pull out all parameters from csvs
    Parameter:
        FIL : list - list of .csv files from DiskDiffEvo.py 
                OR
        FIL : str - name of combined .csv file file after running through pull_out_csvparams() already

        optional:
            all_csv : boolean - determine if you are going though all parameters or opening the combined file default: True
            savefig : str - name of final combined .csv file to save the dataframe to
            clean_up : boolean - determine if you should remove all intermediary .csv files used to create the final combined file. default: True
    '''
    if kwargs.get('all_csv'):
        print('going through .csv files')
        d = {}
        for i in FIL:
            hdr = pd.read_csv(i)
            for name in hdr.columns:
                if name not in d:
                    d[name] = []
                d[name].append(hdr[name].values[0])
        df= pd.DataFrame(d)
        if 'savefig' in kwargs: 
            if 'Unnamed: 0' in df.columns.to_list():
                df = df.drop('Unnamed: 0', axis=1)
            if 'model' in df.columns.to_list():
                df = df.drop('model', axis=1)
            print('  ==> saving as' + kwargs['savefig'])
            df.to_csv(kwargs['savefig'], index=False)
        
        if kwargs.get('clean_up'):
            for f in FIL:
                os.remove(f)

    else:
        print('opening FINAL csv file')
        df = pd.read_csv(FIL, index_col=False)
        if 'Unnamed: 0' in df.columns.to_list():
            df = df.drop('Unnamed: 0', axis=1)
    
    # get best fit solution 
    truths = df.iloc[df['CHI2'].argmin()]
    truths = truths.drop('CHI2').to_list()
    print(truths)

    # drop the chi2 value from best fit solution list and put it in its own list
    data = df.drop('CHI2', axis=1)
    chi2 = df['CHI2']
    
    # get path to mcfost folder of best fit model
    IND = np.where(chi2.values == np.nanmin(chi2.values))
    if kwargs.get('all_csv'):
        MCFOST_DIR = os.path.dirname(np.array(FIL)[IND][0])
    else:
        MCFOST_DIR = None
    
    # return all parameters, all chi2, the best fit parameters, and best fit folder
    return data, chi2, truths, MCFOST_DIR

def pull_out_params(FIL, params, **kwargs): 
    '''OLD!  do not use !! 
    pull out all parameters from .FITS or .csv 
    Parameter:
        FIL : list - list of .fits or .csv files from DiskDiffEvo.py 
        params : list - list of free parameters to extract from header to .csv file
        savefig : str (optional) - name of final combined .csv file to save the dataframe to
        '''
    if '.csv' in FIL:
        print('opening csv file')
        df = pd.read_csv(FIL, index_col=False)
        if 'Unnamed: 0' in df.columns.to_list():
            df = df.drop('Unnamed: 0', axis=1)
        df = df[params]

    else:  
        print('going through .FITS files')
        d = {}

        for i in FIL:
            hdr = fits.getheader(i)
            for name in params:
                if name not in d:
                    d[name] = []
                d[name].append(hdr[name])
        df= pd.DataFrame(d)

        if 'savefig' in kwargs: 
            print('  ==> saving as' + kwargs['savefig'])
            df.to_csv(kwargs['savefig'], index=False)


    truths = df.iloc[df['chi2'].argmin()]
    truths = truths.drop('chi2').to_list()
    print(truths)

    data = df.drop('chi2', axis=1)
    chi2 = df['chi2']
    
    IND = np.where(chi2.values == np.nanmin(chi2.values))
    if '.csv' not in FIL:
        MCFOST_DIR = os.path.dirname(np.array(FIL)[IND][0])
    else:
        MCFOST_DIR = None
    
    return data, chi2, truths, MCFOST_DIR


def plot_parameterspace(M, data, truths, chi2, bounds, colnames, nu, **kwargs):
    '''
    plot the parameter space explored as a corner plot

    Parameter:
        M : int - number of best fit parameters
        data : dataframe - dataframe of all model parameters
        truths : list - list of best fit model parameters
        chi2 : dataframe -  dataframe of all chi2 values
        bounds : list - list of lists of bounds used in DiskDiffEvo.py
        colnames : list - labels to name the parameters
        nu : degrees of freedom from chi2
               
        optional:
            figsize : tuple - size of figure to plot
            cmap : str - cmap color to use
            vmin, vmax : float - vmin and vmax for Normalization of colormap
            get_stats : boolean - determine if you should print the minimum and maximum values 
            title : str - time of plot
            savefig : str - outfile name to save corner plot
    '''
    figsize = kwargs.get('figsize', (12,12))
    fig, axs = plt.subplots(figsize=figsize,ncols=M, nrows=M)
    
    for M1 in np.arange(M):
        for M2 in np.arange(M):
            if M2 > M1:
                ax = axs[M1,M2]
                ax.remove()

    label = colnames #['$i$ (deg)','B', 'h$_{100}$ (AU)', 'A']
    rang=bounds 
    datacolnames = data.columns.to_list()
    chi2 = chi2.values

    cmap = kwargs['cmap']
    norm = Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax'])

    bins = 6
    name, minx, maxy, minchi2, maxchi2, horline = [],[],[],[],[],[]
    for i in range(M):
        x = data[datacolnames[i]].to_list()
        ax = axs[i,i]
        name.append(datacolnames[i])
        minx.append(round(np.min(x),2))
        maxy.append(round(np.max(x),2))
        minchi2.append(round(np.min(chi2),2))
        maxchi2.append(round(np.max(chi2),2))
        bin_means, bin_edges, binnumber = stats.binned_statistic(x,chi2, 'min', bins=bins)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        ax.bar(bin_centers,bin_means, bin_width, color=cmap(norm(bin_means)), edgecolor='k')
        ax.axvline(truths[i], color='r', linewidth=2.5)
        if i != M-1:
            ax.set_xticklabels([])
        ax.set_ylabel('$\chi_r^2$', fontsize=14)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=True, labelsize=14)
        ax.minorticks_on()

        if 'e' in str(truths[i]):
            truth_val = '{:0.2e}'.format(truths[i])
        else:
            truth_val = str(round(truths[i],2))

        if '(deg)' in label[i]:
            lab = label[i].split(' (deg)')[0].replace(r'\n', '\n')
            ax.set_title(lab + ' = ' + truth_val + '$^\circ$', fontsize=14)
        elif '(AU)' in label[i]:
            lab = label[i].split(' (AU)')[0].replace(r'\n', '\n')
            ax.set_title(lab + ' = ' + truth_val + ' AU', fontsize=14)
        elif 'dust settling' in label[i]:
            lab = label[i].replace(r'\n', '\n')
            ax.set_title(lab + ' = ' + str(round(float(truth_val),0)), fontsize=14)
        else:
            lab = label[i].replace(r'\n', '\n')

            ax.set_title(lab + ' = ' + truth_val, fontsize=14)
        ax.set_xlim(rang[i])
        hline = bin_means.min() + np.sqrt(2/nu)
        horline.append(round(hline,2))
        ax.axhline(hline, color='k', linestyle='--')
        ylim_min = kwargs.get('ylim_min', np.nanmin(bin_means)+0.1)
        ylim_max = kwargs.get('ylim_max', np.nanmax(bin_means)+0.05)
        ax.set_ylim(ylim_min, ylim_max)
    t = QTable([name, minx, maxy, minchi2, maxchi2, horline],
               names = ('name', 'min(x)', 'max(x)', 'min(chi2)', 'max(chi2)', 'horline'))
    print(t)

    for j in range(M):
        for i in range(j):
            x = data[datacolnames[j]].to_list()
            y = data[datacolnames[i]].to_list()
            ax = axs[j,i]
   
            ax.axvline(truths[i], color='r', linewidth=2.5)
            ax.axhline(truths[j], color='r', linewidth=2.5)
            ax.plot(truths[i], truths[j], 'sr', markersize=10)

            ret = stats.binned_statistic_2d(y,x,chi2, 'min', bins=bins)
            val = ret.statistic.T
            if kwargs.get('get_stats'):
                print(val.min(), val.max())
            im = ax.pcolormesh(ret.x_edge, ret.y_edge, ret.statistic.T,
                cmap=cmap, vmax=kwargs['vmax'], vmin=kwargs['vmin'])
                  
            ax.set_xlim(rang[i])
            ax.set_ylim(rang[j])
            if j != M-1:
                ax.set_xticklabels([])
            if i != 0:
                ax.set_yticklabels([])

            if i == 0:
                ax.set_ylabel(label[j].replace(r'\n', '\n'), fontsize=14)
            if j == M-1:
                ax.set_xlabel(label[i].replace(r'\n', '\n'), fontsize=14)
            ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=14)
            ax.minorticks_on()
            
            xleft, xright = ax.get_xlim()
            ybottom, ytop = ax.get_ylim()

            ax.set_aspect(abs((xright-xleft)/(ybottom-ytop)))
        
    title = kwargs.get('title', 'model parameters')
    plt.suptitle(title, fontsize=24)
    if 'savefig' in kwargs:
        plt.savefig(kwargs['savefig'], dpi=150)
    plt.show()


def make_final_images(MCFOST_DIR, BESTMOD_DIR, REDUCED_DATA, NOISE, FILTER, FILE_PREFIX,
                      MASK=None, 
                      yaml_paramater_file = None, make_files=False,
                      bestparams_dict=None, amplitude=None, **kwargs):
    '''
    plot the best fit model found in DiskDiffEvo.  

    Parameter:
        MCFOST_DIR : str - path to where mcfost code is ran. 
        BESTMOD_DIR : str - path to best fit model
        REDUCED_DATA : array - original reduced data
        NOISE : array - original std annulus noise data
        FILTER : str - instrument filter name
        FILE_PREFIX : str - prefix used for initial files 
        MASK : array - fits file for mask

        JUST IF YOU NEED TO CREATE THE FILES FROM SCATCH
            yaml_paramater_file : str - yaml parameter file to create models
            make_files : boolean - determine if you need to make the best fit model (default=False)
            bestparams_dict : dictionary - dictionary of params for making best fit model
            amplitude : float - best fit amplitude to scale model
               
        optional:
            title : str - time of plot
            savefig : str - outfile name to save corner plot
    '''

    if make_files: 
        with open(yaml_paramater_file, 'r') as yaml_file:
            params_de_yaml = yaml.safe_load(yaml_file)
        INITIALIZE_DIR = params_de_yaml['INITIALIZE_DIR']
        INITIAL_PARAMS = INITIALIZE_DIR + '/' + params_de_yaml['INITIAL_PARA'] # initial .para file
        INSTRUMENT = params_de_yaml['INSTRUMENT']
        DISTANCE_STAR = params_de_yaml['DISTANCE_STAR']
        obsdate = params_de_yaml['obsdate']
        grid_shape = params_de_yaml['grid_shape']
        ROLL_REF_ANGLE = params_de_yaml['ROLL_REF_ANGLE']
        OBJ_INFO = params_de_yaml['OBJ_INFO']
        WAVELENGTH = params_de_yaml['WAVELENGTH']
        KLBASIS = params_de_yaml['KLBASIS']
        DISKOBJ = mbm.initialize_diskfm(KLBASIS, INITIALIZE_DIR, FILE_PREFIX, REDUCED_DATA)
        BESTMOD_DIR = mbm.make_bestfit_model(MCFOST_DIR, WAVELENGTH, DISKOBJ, REDUCED_DATA, NOISE, INITIAL_PARAMS, FILE_PREFIX, INSTRUMENT, DISTANCE_STAR, MASK, 
                                             ROLL_REF_ANGLE, OBJ_INFO, FILTER, obsdate, grid_shape, bestparams_dict, amplitude)

    else:
        print('   ==> using existing fits files')
    if FILTER=='F200W':
        fold = 'F200W'
        conv_mod =  'RT_model_NRCA2_MASK335R_F200W_mJyas2.fits'
    else:
        conv_mod = 'RT_model_NRCA5_MASK335R_F444W_mJyas2.fits'
        fold = 'F444W'

    hdr = fits.getheader(os.path.join(BESTMOD_DIR + 'diskfm.fits'))
    disk_ml_FM = fits.getdata(os.path.join(BESTMOD_DIR + 'diskfm.fits'))
    
    ####
    bkg = np.mean(disk_ml_FM[200:240, 200:240])
    disk_ml_FM=disk_ml_FM-bkg
    ####

    disk_model_mJy_px = fits.getdata(os.path.join(BESTMOD_DIR + 'RT_mJypx.fits'))
    modhdr = fits.getheader(os.path.join(BESTMOD_DIR + 'RT_mJypx.fits'))
    if 'CDELT2' not in modhdr:
        modelpixelscale = 1.722528E-05
    else:
        modelpixelscale = modhdr['CDELT2'] * 3600 
    disk_model_mJy_as2 = disk_model_mJy_px / modelpixelscale**2. 

    pad = (REDUCED_DATA.shape[0]//2) - (fits.getdata(BESTMOD_DIR + conv_mod).shape[0]//2)
    disk_ml_convolved = np.pad(fits.getdata(BESTMOD_DIR + conv_mod), pad) * hdr['AMP'] 
    data_pixscale = fits.getheader(BESTMOD_DIR + conv_mod)['PIXELSCL']

    #Measure the residuals
    modeldisk = np.copy(disk_ml_FM)
    convovleddisk = np.copy(disk_ml_convolved)
    origdisk = np.copy(disk_model_mJy_as2)

    residuals = REDUCED_DATA - (modeldisk)
    snr_residuals = (REDUCED_DATA - (modeldisk)) / NOISE
    
    if MASK is not None:
        residuals[MASK]=np.nan
        snr_residuals[MASK]=np.nan

    cmap = copy.copy(cm.get_cmap("seismic"))
    cmap.set_bad(color='white')

    #Set the fontsize
    caracsize = 12
    pix_scale = data_pixscale
    halfsize = np.asarray(modeldisk.shape[-2:]) / 2 * pix_scale
    extent = [halfsize[0], -halfsize[0], -halfsize[1], halfsize[1]]

    stretch = kwargs.get('stretch', 'sqrt')
    vmin = kwargs.get('vmin', 0)
    vmax = kwargs.get('vmax', 0.5)
    origvmin = kwargs.get('origvmin', vmin)
    origvmax = kwargs.get('origvmax', vmax)
    resvmin = kwargs.get('resvmin', -vmax)
    resvmax = kwargs.get('resvmax', vmax)

    fig, ((ax1, ax2, ax3),(ax4,ax5, ax6)) = plt.subplots(figsize=(8,5), nrows=2, ncols=3, dpi=150)
    
    # FIG 1 The model
    img = disk_model_mJy_as2 
    norm = simple_norm(img, 'sqrt', min_cut=origvmin, max_cut=origvmax)
    print(np.nanmax(img))
    cax = ax1.imshow(img, origin='lower', extent=extent, cmap='inferno', norm=norm)
    ax1.set_title("Best Model", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax , ax=ax1, fraction=0.046, pad=0.04)
    cbar.minorticks_off()
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4., length=3)
    ax1.axis('off')
    
    #FIG 2 The FM convolved model
    norm = simple_norm(modeldisk, stretch, min_cut=vmin, max_cut=vmax, invalid=0)
    cax = ax2.imshow(modeldisk, origin='lower', extent=extent, norm=norm, cmap='magma')
    ax2.set_title("Model Convolved + FM", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax,ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4., length=3)
    cbar.minorticks_off()
    ax2.axis('off')
    
    #FIG 3 The residuals
    cax = ax3.imshow(residuals, origin='lower', extent=extent, vmin=resvmin, vmax=resvmax, cmap=cmap)
    ax3.set_title("Residuals", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax,ax=ax3, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4., length=3)

    #FIG 4 convolved model
    norm = simple_norm(convovleddisk, stretch, min_cut=vmin, max_cut=vmax, invalid=0)
    cax = ax4.imshow(convovleddisk, origin='lower',extent=extent, cmap='magma', norm=norm)
    ax4.set_title("Model Convolved", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, ax=ax4, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4., length=3)
    cbar.minorticks_off()
    ax4.axis('off')
    
    #FIG 5 The data
    norm = simple_norm(REDUCED_DATA, stretch, min_cut=vmin, max_cut=vmax, invalid=0)
    cax = ax5.imshow(REDUCED_DATA, origin='lower', extent=extent, cmap='magma', norm=norm)
    ax5.set_title("KLIP reduced data", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax,ax=ax5, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4., length=3)
    cbar.minorticks_off()
    ax5.axis('off')

    #FIG 6 The SNR of the residuals
    cax = ax6.imshow(snr_residuals**2., origin='lower', extent=extent, vmin=0, vmax=1, cmap='magma')
    ax6.set_title("$\chi^2$ Residuals", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, ax=ax6, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4., length=3)
    cbar.minorticks_off()

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlim(5,-5)
        ax.set_ylim(-5,5)
        cbar.minorticks_off()
        c = plt.Circle((0,0), .8, color='k')
        # ax.add_artist(c)

    for ax in [ax3,ax6]:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
     
    title = kwargs.get('title', str(FILTER))
    fig.suptitle(title)
    plt.subplots_adjust(top=0.8)
    fig.tight_layout()
    
    if 'savefig' in kwargs:
        plt.savefig(kwargs['savefig'], dpi=150)
    plt.show()


def run_doplot(yaml_paramater_file, make_files=False, paramplot_kw=None, modelplot_kw=None, amplitude=1, **kwargs, ):
    #######################
    with open(yaml_paramater_file, 'r') as yaml_file:
        params_de_yaml = yaml.safe_load(yaml_file)
    SAVE_DIR = kwargs.get('SAVE_DIR', params_de_yaml['SAVE_DIR'])
    FILE_PREFIX = params_de_yaml['FILE_PREFIX']   # name of level 2 data - roll 9
    REDUCEDDATA_DIR = params_de_yaml['REDUCEDDATA_DIR']
    NOISE_DIR = params_de_yaml['NOISE_DIR']
    WAVELENGTH = params_de_yaml['WAVELENGTH'] # number
    FILTER = params_de_yaml['FILTER'] # string ie. F200W
    bounds = params_de_yaml['BOUNDS']
    labels = list(params_de_yaml['LABELS'].values())
    INITIALIZE_DIR = params_de_yaml['INITIALIZE_DIR']
    MASK = fits.getdata(INITIALIZE_DIR + '/' + FILE_PREFIX + '_MASK.fits') # mask for calculating chi2
    nu = np.count_nonzero(MASK==0)-len(bounds)
    MASK = ma.make_mask(MASK)
    clean_up = params_de_yaml['CLEAN_UP']

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
    #######################

    param_plot = kwargs.get('param_plot', True)
    model_plot =  kwargs.get('model_plot', True)
    if kwargs.get('do_all'):
        param_plot = True
        model_plot = True

    # do plotting and main .csv filing
    print('starting plotting')
    # extract all parameters from .csv file and put them into 1 file.  if clean_up=True, the original .csv files will be deleted leaving only the combined file. 

    if 'manual_params' in kwargs:
        print('manual params')
        truths = kwargs['manual_params']
    else:
        savefil = os.path.dirname(SAVE_DIR) + f'/FINAL_DEchi2_{FILE_PREFIX}.csv'
        if os.path.isfile(savefil):
            FIL = savefil
            all_csv = False
        else:
            FIL = glob.glob(SAVE_DIR + '/*.csv')
            all_csv = True
        data, chi2, truths, MCFOST_DIR = pull_out_csvparams(FIL, all_csv=all_csv, savefig = savefil, clean_up=clean_up)

    #######################
    if param_plot:
        # plot a corner plot of the parameter space
        min_val, max_val = 0.03,0.99
        n = 6
        orig_cmap = plt.cm.bone_r
        colors = orig_cmap(np.linspace(min_val, max_val, n))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)

        print('min/max chi2: ', np.nanmin(chi2), '/', np.nanmax(chi2))
        plot_parameterspace(len(truths), data, truths, chi2, list(bounds.values()), labels, nu,
            get_stats=False, cmap=cmap, figsize=(20,20), 
            title = FILE_PREFIX.replace('_', ' '),
            savefig=os.path.dirname(SAVE_DIR) + f'/{FILE_PREFIX}_params_space.png', **paramplot_kw
            )

    #######################
    if model_plot:

        if make_files:
            BESTMOD_DIR = SAVE_DIR
            GRIDGEN_DICT = params_de_yaml['GRIDGEN_DICT']
            bestparams_dict = {}
            print(truths)
            for i, v in enumerate(GRIDGEN_DICT):
                if 'dust_settling' in v:
                    truth_val = int(round(truths[i],0))
                elif 'dust_mass' in v:
                    truth_val = 10**truths[i]
                else:
                    truth_val = truths[i]
                bestparams_dict[v] = [truth_val]
            amplitude = truths[-1]
            print(amplitude)
            make_final_images(os.path.dirname(SAVE_DIR), BESTMOD_DIR, 
                            REDUCED_DATA, NOISE,
                            FILTER, FILE_PREFIX, MASK, 
                            yaml_paramater_file=yaml_paramater_file, make_files=make_files,
                            bestparams_dict=bestparams_dict, amplitude=amplitude, 
                            title = FILE_PREFIX.replace('_', ' ') + ' ' + 'Best Fit model', savefig=os.path.dirname(SAVE_DIR) + f'/{FILE_PREFIX}_DEmodel.png', **modelplot_kw
                            )
        else:
            # plot the best fit image and residuals
            BESTMOD_DIR = glob.glob(f'{SAVE_DIR}/*/data_{WAVELENGTH}/')[0]
            make_final_images(os.path.dirname(SAVE_DIR), BESTMOD_DIR, REDUCED_DATA, NOISE, FILTER, FILE_PREFIX, MASK=MASK,
                        title = FILE_PREFIX.replace('_', ' ') + ' ' + 'Best Fit model', 
                                savefig=os.path.dirname(SAVE_DIR) + f'/{FILE_PREFIX}_DEmodel.png', 
                                make_files=make_files, **modelplot_kw)


    print("FINISHED!")