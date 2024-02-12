#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
#import emcee
import os
import subprocess 
#import models
import acor
#import utils
from mcfost.utils import generate_image_covariance_matrix
from mcfost.utils import generate_sed_covariance_matrix
from mcfost.chisqr import image_chisqr
from mcfost.chisqr import sed_chisqr
from mcfost.paramfiles import Paramfile
from mcfost.models import ModelResults
from mcfost.models import Observations
from mcfost.chisqr import image_likelihood
from mcfost.chisqr import sed_likelihood

import scipy.interpolate

from emcee import PTSampler

import logging
_log = logging.getLogger('mcfost')


#Set a few Parameters
ndim = 5
imwavelength = 0.8
parameters = ['inclination','scale_height','disk_mass','alpha','beta']


# Read in the observations to save computation time below:
#maindir = '/astro/4/mperrin/eods/models_esoha569/ESOha569_EMCEE/'
maindir = '/home/swolff/EMCEE/mcmcrun_083016/'
obs = Observations(maindir+'data')

# Read in the auto-correlation array to be used to generate the covariance matrix
autocorrfn = maindir+'mean_autocorr.npy'
mean_autocorr = np.load(autocorrfn)
autocorrelation = np.zeros((100))
# Choose 1D slice of autocorrelation map
autocorrelation[0:50] = mean_autocorr[49,49:]


# Generate the covariance matrix for the image (plus the det and inverse).
covariance =  generate_image_covariance_matrix(obs, autocorrelation, analytical=True, wavelength=imwavelength)
#cholesky = np.linalg.cholesky(np.matrix(covariance))
#covariance_inv = np.dot(np.linalg.inv(cholesky),np.linalg.inv(np.transpose(cholesky)))
#covariance_det = np.linalg.det(covariance)
#covariance_det = (np.linalg.det(cholesky))**2.0
covariance_inv = np.linalg.inv(covariance)
(sign, logdet) = np.linalg.slogdet(covariance)
# Stack the covariance, inverse, and determinite along the third dimension
image_covariance = np.array([covariance,covariance_inv,logdet,sign])

# Generate the covariance matrix for the SED (plus det and inv).
covariance =  generate_sed_covariance_matrix(obs)
covariance_inv = np.linalg.inv(covariance)
(sign, logdet) = np.linalg.slogdet(covariance)
# Stack the covariance, inverse, and determinite along the third dimension
sed_covariance = np.array([covariance,covariance_inv,logdet,sign])


# Pass obs, image_covariance, and sed_covariance as additional arguments to emcee sampler


# Define the log likelihood 
def lnprobab(theta):

    """
    PARAMETERS
    ----------
    Theta:
    Parameters to vary. 
         theta[0] = inclination
         theta[1] = scale_height
         theta[2] = disk_mass
         theta[3] = alpha
         theta[4] = beta
         

    USAGE
    -----
    Computes and returns the log of the likelihood 
    distribution for a given model.

    """

    imageuncert, imagechi, seduncert, sedchi = mcmcwrapper(theta)


    lnpimage = -0.5*np.log(2*np.pi)*imageuncert.size-0.5*imagechi-np.sum(-np.log(imageuncert))

    lnpsed = -0.5*np.log(2*np.pi)*seduncert.size-0.5*sedchi-np.sum(-np.log(seduncert))

    return theta[6]*lnpimage + (1.0-theta[6])*lnpsed


# Alternative method to define the log likelihood function using a covariance matrix.
def lnprobabmatrix(theta,obs,image_covariance,sed_covariance):

    """
    PARAMETERS
    ----------
    Theta:
    Parameters to vary. 
         theta[0] = inclination
         theta[1] = scale_height
         theta[2] = disk_mass
         theta[3] = alpha
         theta[4] = beta
         

    USAGE
    -----
    Computes and returns the log of the likelihood 
    distribution for a given model.

    """

    sedloglikelihood, imageloglikelihood = mcmcwrapper(theta,obs,image_covariance,sed_covariance,usematrix=True)


    return sedloglikelihood + imageloglikelihood




# Define the log Priors 
def lnprior(theta):
    
    inc    = theta[0]
    ho     = theta[1]
    mass   = theta[2]
    alpha  = theta[3]
    beta   = theta[4]
    
    # include priors here
    if (inc < 65.0 or inc > 90.0):
        return -np.inf    
    
    if (ho < 5.0 or ho > 35.0):
        return -np.inf    

    if (np.log10(mass) < -6.0 or np.log10(mass) > -3.0 or mass < 0.0):
        return -np.inf

    #if (np.log10(amax) < 2.0 or np.log10(amax) > 4.0 or amax < 0.03 or amax > 3000.0):
    #    return -np.inf

    if (beta < 1.0 or beta > 1.6):
        return -np.inf
    
    if (alpha < -2.0 or alpha > 0.0):
        return -np.inf

    #if (weight < 0.3 or weight > 0.7):
    #    return -np.inf

    # otherwise ...
    return 0.0


def mcmcwrapper(theta, obs, image_covariance, sed_covariance, usematrix=False):

    """
    PARAMETERS
    ----------
    Params - numpy array
         Parameters to be used in this call
         to emcee.

         For the purposes of ESO Halpha 569,
         these are inclination, scale height, 
         dust mass, amax, beta, alpha, and 
         rstar. 

    USAGE
    -----
    Takes a parameter file, the variable parameters
    and a directory. Creates the model image and 
    SED, computes the chisqr, reads in the 
    observables, and returns the uncertainties and 
    Chi^2 values. This is called by the function
    that calculates the likelihood function. 

    Step 1: Get parameters for this step for emcee.
    - Right a standalone function that will take in the list of parameters to vary and write the parameter files. This is unique to each object.
    Step 2: Write new parameter files with these values.
    Step 3: Run mcfost and store images and SEDs for this file. (Switch for running seds or images separately. )
    Step 4: Calculate the chi2 values for the given model and SED.
    Step 5: Pass the values to the log probability function. 
    """
    # Add a couple parameters that differentiate between the matrix method and chisqrd method
    if usematrix:
        fitstring = 'likelihood.txt'
    else:
        fitstring = 'chisqrd.txt'

    # STEP 1: This is passed via theta 
    # STEP 2:
    olddir=os.path.abspath(os.curdir)
    maindir = '/astro/mperrin/eods/models_esoha569/ESOha569_EMCEE/mcmcrun_083016/'
    ndim = 5
    par = Paramfile(maindir+'data/esoha569.para')
    par.RT_imax = theta[0]
    par.RT_imin = theta[0]
    par.RT_n_incl = 1
    par.set_parameter('scale_height',theta[1])
    par.set_parameter('dust_mass',theta[2])
    #par.set_parameter('dust_amax',theta[3])
    par.set_parameter('flaring',theta[4])
    par.set_parameter('surface_density',theta[3])
    gammaexp = -1.0*theta[3]
    par.set_parameter('gamma_exp',gammaexp)
    # Do I need to write these, or just do this in memory?
    # write the parameter file in the default directory
    fnstring = "{0:0.4g}".format(theta[0])
    for i in np.arange(ndim-1):
        fnstring += '_'+"{0:0.4g}".format(theta[i+1])

    par.writeto(fnstring+'.para')
    modeldir = olddir+'/'+fnstring

    try:
        os.mkdir(modeldir)
    except:
        #subprocess.call('rm -r '+modeldir,shell=True)
        #os.mkdir(modeldir)
        print 'Model already exists.'
        subprocess.call('rm '+fnstring+'.para',shell=True)
        os.chdir(modeldir)
        
        try:
            f = open(fitstring,'r')
            chisqrds = f.readlines()
            f.close()
            chistr = chisqrds[0].split(' ')
            sedchi = float(chistr[1])
            imchi = float(chistr[4])

            seduncertainty = obs.sed.nu_fnu_uncert
            imageuncertainty = obs.images[0.8].uncertainty
            os.chdir(olddir)
            if usematrix:
                return sedchi, imchi
            else:
                return imageuncertainty, imchi, seduncertainty.value, sedchi

        except:
             print 'Model Failed'
             os.chdir(olddir)
             dummyval = 1.0e15
             if usematrix:
                 return dummyval,dummyval
             else:
                 return np.asarray([dummyval,dummyval]), dummyval, np.asarray([dummyval,dummyval]), dummyval



    subprocess.call('chmod -R g+w '+modeldir,shell=True)
    subprocess.call('mv '+fnstring+'.para '+modeldir,shell=True)
    os.chdir(modeldir)

    #modeldir 3:
    # run mcfost in the given directory
    subprocess.call('mcfost '+fnstring+'.para -rt >> sedmcfostout.txt',shell=True)
    subprocess.call('mcfost '+fnstring+'.para -img 0.8 -rt >> imagemcfostout.txt',shell=True)

    #STEP 4: 
    try:
        model = ModelResults(maindir+fnstring)
    except IOError:
        print 'Model Failed'
        os.chdir(olddir)
        dummyval = 1.0e15
        if usematrix:
            return dummyval,dummyval
        else:
            return np.asarray([dummyval,dummyval]), dummyval, np.asarray([dummyval,dummyval]), dummyval
        
    try:
        obs
    except NameError:
        print "well, it WASN'T defined after all!"
        obs = Observations(maindir+'data')
        
    try:
        if usematrix:
            imagechi = image_likelihood(model,obs,image_covariance,wavelength=0.8)
            sedchi = sed_likelihood(model, obs, sed_covariance, dof=1, write=True, save=False, vary_distance=False,vary_AV=True, AV_range=[0.0,10.0])
        else:
            imagechi = image_chisqr(model,obs,wavelength=0.8)
            sedchi = sed_chisqr(model, obs, dof=1, write=True, save=False, vary_distance=False,vary_AV=True, AV_range=[0.0,10.0])
    except:
        print 'Model Failed'
        dummyval = 1.0e15
        os.chdir(olddir)
        if usematrix:
            return dummyval,dummyval
        else:
            return np.asarray([dummyval,dummyval]), dummyval, np.asarray([dummyval,dummyval]), dummyval


    sedchi=sedchi[0]
    imagechi=imagechi[0]

    sedstring = 'SED {0}'.format(sedchi)+'  '
    imagestring = 'Image {0}'.format(imagechi)
    
    f = open(fitstring,'w')
    f.write(sedstring+imagestring)
    f.close()
    
    seduncertainty = obs.sed.nu_fnu_uncert
    imageuncertainty = obs.images[0.8].uncertainty
    
    subprocess.call('pwd',shell=True)
    os.chdir(olddir)


    # force close all model images:
    model.images.closeimage()  

    #STEP 5:
    if usematrix:
        return sedchi,imagechi
    else:
        return imageuncertainty, imagechi, seduncertainty.value, sedchi


def generate_image_covariance_matrix(observations, autocorrelation, matern=False, analytical=True, wavelength=None):

    # Get Observed Data
    im = observations.images
    mask = im[wavelength].mask
    image = im[wavelength].image
    noise = im[wavelength].uncertainty
    
    # Generate x and y position matrices for the covariance matrix
    xij = np.zeros(image.shape)
    yij = np.zeros(image.shape)
    for ind in np.arange(image.shape[0]):
        yij[:,ind] = np.arange(image.shape[0])
        xij[ind,:] = np.arange(image.shape[1])

    # Convert all relevant arrays to 1D matrix
    matrix_obs = image[mask != 0]
    matrix_noise = noise[mask != 0]
    xij_matrix = xij[mask != 0]
    yij_matrix = yij[mask != 0]

    # Set some parameters needed for the Matern Kernel
    l = 5.0
    ro = 5.0*l
    b = 1.0
    ag = 1.0


    # Covariance Matrix has the functional form:
    # C_ij = b * \delta_ij noise^2_i + K_ij(phi_G)
    # Construct  K_ij:
    nx = mask[mask != 0].shape[0]
    ny = nx
    xx = np.arange(nx)
    yy = np.arange(ny)
    rij = np.zeros((nx,ny))
    wij = np.zeros((nx,ny))
    sigma_diag = np.zeros((nx,ny))
    global_covariance = np.zeros((nx,ny))
    #rij = sqrt((xi-xj)**2+(yi-yj)**2)
    #cij = sigmai sigmaj wij (1+(np.sqrt(3)*rij/l))*np.exp(-np.sqrt(3)*rij/l) 
    interpolator = scipy.interpolate.interp1d(np.arange(autocorrelation.shape[0]), autocorrelation, kind='linear', copy=False)


    for i in xx: 
        for j in yy:
            if i == j:
                sigma_diag[j,i] = b*matrix_noise[i]*matrix_noise[i]
            rij[j,i] = np.sqrt((xij_matrix[i]-xij_matrix[j])**2.0+(yij_matrix[i]-yij_matrix[j])**2.0)
            if rij[j,i] <= ro:
                wij[j,i] = 0.5+0.5*np.cos(np.pi*rij[j,i]/ro)
            else:
                wij[j,i] = 0.0
            global_covariance[j,i]=wij[j,i]*interpolator(rij[j,i])    
            

    K_global =  ag * (1+(np.sqrt(3)*rij/l))*np.exp(-np.sqrt(3)*rij/l)*wij
    
    if matern:
        covariance = sigma_diag + K_global
    if analytical:
        covariance = sigma_diag + global_covariance

    return covariance

def generate_sed_covariance_matrix(observations):

    # Get the sed and uncertainties
    obs_wavelengths = observations.sed.wavelength
    obs_nufnu = observations.sed.nu_fnu
    obs_nufnu_uncert = observations.sed.nu_fnu_uncert
    noise = obs_nufnu_uncert.value


    # Covariance Matrix has the functional form:
    # C_ij = b * \delta_ij noise^2_i + K_ij(phi_G)
    # Construct  K_ij:
    nx = noise.shape[0]
    ny = nx
    xx = np.arange(nx)
    yy = np.arange(ny)
    sigma_diag = np.zeros((nx,ny))
    b = 1.0

    for k in xx: 
        for j in yy:
            if k == j:
                sigma_diag[j,k] = b*noise[k]*noise[k]
               
    covariance = sigma_diag

    return covariance

######################################################## 

ntemps = 2
nwalkers = 50
ndim = 5

sampler=PTSampler(ntemps, nwalkers, ndim, lnprobabmatrix, lnprior,loglkwargs={'obs':obs,'image_covariance':image_covariance,'sed_covariance':sed_covariance})
#sampler=EnsembleSampler(nwalkers,ndim,lnprobabmatrix,lnprior,args)
# Use the MPIPool instead
"""
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
"""
#sampler = PTSampler(ntemps, nwalkers, ndim, lnprobab, lnprior, threads=15)

#############################################################
# Initialize the walkers. The best technique seems to be
# to start in a small ball around the a priori preferred position.
# Dont worry, the walkers quickly branch out and explore the
# rest of the space.

# inclination: w0
# scale height: w1
# dust mass: w2
# alpha: w3
# beta: w4

w0 = np.random.uniform(73.0,80.0,size=(ntemps,nwalkers))
w1 = np.random.uniform(10,20,size=(ntemps,nwalkers))
w2 = np.random.uniform(0.0001,0.001,size=(ntemps,nwalkers))
#w3 = np.random.uniform(100.0,3000.0,size=(ntemps,nwalkers))
w3 = np.random.uniform(-2.0,0.0,size=(ntemps,nwalkers))
w4 = np.random.uniform(1.2,1.4,size=(ntemps,nwalkers))
#w6 = np.random.uniform(0.4,0.6,size=(ntemps,nwalkers))

p0 = np.dstack((w0,w1,w2,w3,w4))
#print p0.shape
niter = 100
nburn = np.int(0.025*niter)

diskname='esoha569'


######################################################
##################  BURN IN STAGE ####################
######################################################
"""
burnfn = "/Users/swolff/EMCEE/mcmcrun_083016/burnchain.dat"
f = open(burnfn,"w")
f.close()

for p, lnprob, lnlike in sampler.sample(p0, iterations=nburn):
    position = p[0]
    f = open(burnfn,"a")
    for k in range(position.shape[0]):
        f.write('{0:4d} {1:s}\n'.format(k, " ".join(str(position[k]))))
    f.close()
    
position = p
np.save('p_burnin',position)
    
print 'Burn in complete'

#pool.close()

samples = sampler.chain[:,:,:].reshape((-1,ndim))
np.save('run_sampler_burnin',samples)
"""
sampler.reset()



#############################################################                                                                                                  
##################### Restart from File #####################                                                                                                  
############################################################# 

# Load samples from the burnin phase:
#p = np.load('/Users/swolff/EMCEE/mcmcrun_083016/p_burnin.npy')

# Load in samples from previous mcmc chain, and run any models
# that weren't generated.
p = np.load(maindir+'p_frommcmc.npy')

##################################################################
##################### Continue After Burn-in #####################
##################################################################

chainfn = maindir+"chain.dat"
tempsfn = maindir+"temps.dat"
f = open(chainfn,"w")
f.close()
g = open(tempsfn,"w")
g.close()

for p, lnprob, lnlike in sampler.sample(p, iterations=niter,storechain=True):
    position = p[0]
    position_temp = p[1]
    f = open(chainfn,"a")
    g = open(tempsfn,"a")
    for k in range(position.shape[0]):
        f.write("{0:4d} {1:s}\n".format(k, " ".join(str(position[k]))))
        g.write("{0:4d} {1:s}\n".format(k, " ".join(str(position_temp[k]))))
    f.close()
    g.close()
    
position = p
np.save('p_mcmc',position)
    
print "MCMC Complete"
samples = sampler.chain
np.save('run_sampler',samples)


inclres = np.ndarray.flatten(sampler.chain[:,:,:,0])
hores = np.ndarray.flatten(sampler.chain[:,:,:,1])
massres = np.ndarray.flatten(sampler.chain[:,:,:,2])
alphares = np.ndarray.flatten(sampler.chain[:,:,:,3])
betares = np.ndarray.flatten(sampler.chain[:,:,:,4])

arrays = [inclres, hores, massres, alphares, betares]

for i in np.arange(ndim):
    filename = diskname + '_'+parameters[i]+'.txt'
    np.savetxt(filename, arrays[i])

acceptance_fraction = sampler.acceptance_fraction
print 'acceptance_fraction: ',np.mean(sampler.acceptance_fraction)
np.save('acceptance_fraction',acceptance_fraction)



import pdb #@@@
pdb.set_trace() #@@@
print 'stop here' #@@@

