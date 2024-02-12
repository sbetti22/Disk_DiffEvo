
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.interpolate
import scipy.ndimage as sciim
import astropy.units as units
import image_registration

import models
from . import utils
import logging
_log = logging.getLogger('mcfost')

def sed_chisqr(modelresults, observations, dof=1,
    write=False, logfit=False,
    plot=False, save=False,
    vary_distance=False, distance_range=None,
    vary_AV=False, AV_range=[0,10],
    **kwargs):

    """ Compute chi^2 for a given model

    Not written yet - this is just a placeholder

    Parameters
    -----------
    dof : int
        Number of degrees of freedom to use in computing the
        reduced chi^2. Will be incremented appropriately if
        vary_distance or vary_AV are set.
    vary_distance : bool
        Allow the distance to the target to vary
    distance_range : 2 element iterable
        Min and max allowable distance, in pc
    vary_AV : bool
        Allow optical extinction (A_V) to vary
    AV_range : 2 element iterable
        Min and max allowable A_V
    RV : float
        Reddening law color parameter for extinction
    write : bool
        Write output to disk as chi^2 results FITS file in the model directory?
        Default is True
    save : bool
        Save results to disk as FITS bintable? Default is True.
    plot : bool
        Display plot of chi^2 fit?
    logfit: bool
        Perform chi^2 fit in log space. Default is False

    """

#    if not isinstance(modelresults, models.ModelResults):
#        print type(modelresults)
#        raise ValueError("First argument to sed_chisqr must be a ModelResults object")
#
#    if not isinstance(observations, models.Observations):
#        raise ValueError("Second argument to sed_chisqr must be an Observations object")
    my_dof = dof
    if vary_distance:
        my_dof += 1
    if vary_AV:
        #_log.info("Computing chi^2 while allowing A_V to vary between {0} and {1} with R_V={2}".format(AV_range[0], AV_range[1], RV) )
        my_dof += 1



    # observed wavelengths and fluxes
    obs_wavelengths = observations.sed.wavelength
    obs_nufnu = observations.sed.nu_fnu
    obs_nufnu_uncert = observations.sed.nu_fnu_uncert
    # model wavelengths and fluxes. Fluxes will be a 2D array: [ninclinations, nwavelengths]
    mod_wavelengths = modelresults.sed.wavelength
    mod_nufnu= modelresults.sed.nu_fnu
    mod_inclinations = modelresults.parameters.inclinations

   
    if logfit:
        ln_observed_sed_nuFnu = obs_nufnu.value
        ln_err_obs = obs_nufnu_uncert.value
        subset = obs_nufnu != 0.0
        ln_observed_sed_nuFnu[subset] = np.log(obs_nufnu[subset].value)
        ln_err_obs[subset] = np.log(obs_nufnu_uncert[subset].value)

    if plot:
        observations.sed.plot(**kwargs)

    ninc = mod_inclinations.size
    chi2s = np.zeros(ninc)
    avs = np.zeros(ninc)
    rvs = np.zeros(ninc)
    distances = np.zeros(ninc) + modelresults.parameters.distance
    # iterate over inclinations, computing estimated observations and fitting
    # allow reddening to vary
    for i in range(ninc):

        interpolator = scipy.interpolate.interp1d(mod_wavelengths, mod_nufnu[i], kind='linear', copy=False)
        est_mod_nufnu = interpolator(obs_wavelengths)

        if plot:
            modelresults.sed.plot(inclination=mod_inclinations[i], overplot=True)
            ax = plt.gca()
            color = ax.lines[-1].get_color()
            ax.plot(obs_wavelengths, est_mod_nufnu, color=color, marker='s', linestyle='none')

        if vary_distance or vary_AV:
            residuals, best_distance, best_av, best_rv,chi2 =  fit_dist_extinct(obs_wavelengths, obs_nufnu.value, est_mod_nufnu, obs_nufnu_uncert.value, modeldist=distances[i], additional_free=my_dof, distance_range=distance_range, vary_av=vary_AV, vary_distance=vary_distance, av_range=AV_range)
        else:
           if logfit:
               chi2 = ((ln_observed_sed_nuFnu - np.log(est_mod_nufnu))**2 / ln_err_obs**2).sum()
           else:
               chi2 = ((obs_nufnu.value - est_mod_nufnu)**2 / obs_nufnu_uncert.value**2).sum()

        _log.info( "inclination {0} : {1:4.1f} deg has chi2 = {2:5g}".format(i, mod_inclinations[i], chi2))


        chi2s[i] = chi2



    if save:
        import astropy.table
        import os
        tabledict = {'INCLINATION': mod_inclinations, 'SED_CHISQ': chi2s, 'AV':avs, 'RV':rvs, 'DISTANCE':distances}
        meta = {'SOURCE': 'Python mcfost.chisqr.sed_chisqr()',
                'VARY_AV': vary_AV,
                'VARYDIST': vary_distance,
                'AVRANGE': str(AV_range),
                'DISTRNGE': str(distance_range) }
        tab = astropy.table.Table(tabledict, meta=meta)

        tab.write(os.path.join(modelresults.directory, 'observables', 'py_sed_chisq.fits'), overwrite=True)


        # Save the results:
        """
        results = {inclination: param.grid.inclinations, $
				sed_chisq: chisqs, $
				distance: best_distances, $
				AV: best_AVs, Rv: Rv }

	if ~dir_exist(directory+"/observables") then file_mkdir, directory+"/observables"

	mwrfits, results, directory+"/observables/sed_chisq.fits", /create
	message,/info, "Results stored to "+ directory+"/observables/sed_chisq.fits"

        """

    return chi2s




def fit_dist_extinct(wavelength, observed_sed_nuFnu, model,
        error_observed_sed_nuFnu = None, Rv=3.1, modeldist=1.0,
        additional_free=None, logfit=False,
        distance_range=[0.0,1000.0],model_noise_frac=0.1,
        vary_av=True, vary_distance=True, av_range=[0.0,10.0],
        rv_range=[2.0,20.0],vary_rv=False, **kwargs):

    """
    Adapted from the fit_dist_extinct3.pro MCRE file designed to allow
    distance and extinction to vary when computing the SED chsqr. For
    a given inclination, this function returns the best fit distance,
    extinction, Rv and chisqrd value.

    Parameters
    -----------
    wavelength : Float Array
        The wavelengths .... in microns
    observed_sed_nuFnu: Quanity object in units W/m2
        The nuFnu for the observed SED.
    error_observed_sed_nuFnu : Float array
        Error for the observed SED
    model_noise_frac : Float
        Noise fraction for the model SEDs
    modeldist : Float
        Distance to model disk in pc.
    additional_free : Int
        Number of additionaly free parameters to
        include in the weighted chisqrd.
    vary_distance : bool
        Allow the distance to the target to vary?
        Default is False
    distance_range : 2 element iterable
        Min and max allowable distance, in pc
    vary_av : bool
        Allow optical extinction (A_V) to vary?
        Default is False
    av_range : 2 element iterable
        Min and max allowable A_V
    vary_rv : bool
        Allow rv to vary?
        Default is False
    Rv : float
        Reddening law color parameter for extinction.
        Default is 3.1
    logfit : bool
        Fit the SED on a logarithmic scale?
        Default is False

    """

    #wave_ang = [x*0.0001 for x in wavelength] #Convert from microns to angstroms
    wave_ang = wavelength.to(units.Angstrom)
    wave_ang = wave_ang.value
    observed_sed_nuFnu = np.asarray(observed_sed_nuFnu)
    wave_mu = wavelength.value

    # Scale model to 1pc
    model_sed_1pc = np.asarray(model * modeldist**2)
    if vary_distance:
        diststeps = (distance_range[1] - distance_range[0])/50.0
        a_distance = np.arange(distance_range[0],distance_range[1],diststeps)
    else:
        a_distance = np.asarray([modeldist])



    if vary_av:
        avsteps = (av_range[1] - av_range[0])/100.0
        a_av = np.arange(av_range[0],av_range[1],avsteps)
    else:
        a_av = np.asarray([0])

    if vary_rv:
        a_rv = np.asarray([10] + rv_range)
    else:
        a_rv = np.asarray([Rv])
        
    try:
        err_obs = np.asarray(error_observed_sed_nuFnu)
    except:
        print 'assigning 10% errors.'
        err_obs = np.asarray(0.1*observed_sed_nuFnu)
  
    

    if logfit:

        ln_observed_sed_nuFnu = observed_sed_nuFnu
        ln_err_obs = err_obs
        subset = observed_sed_nuFnu != 0.0
        ln_observed_sed_nuFnu[subset] = np.log(observed_sed_nuFnu[subset])
        ln_err_obs[subset] = err_obs[subset]/observed_sed_nuFnu[subset]

    # How many degrees of freedom?
    dof = len(observed_sed_nuFnu)


    chisqs = np.zeros(([max(len(a_distance),1),max(len(a_av),1),max(len(a_rv),1)]))

    for i_r in np.arange(len(a_rv)):
        ext = np.asarray(utils.ccm_extinction(a_rv[i_r],wave_mu)) # Use wavelength in Angstroms
        for i_d in np.arange(len(a_distance)):

            for i_a in np.arange(len(a_av)):
                extinction = 10.0**((ext*a_av[i_a])/(-2.5))
                vout = (np.multiply(model_sed_1pc,extinction))/(a_distance[i_d])**2

                if logfit:
                    ln_vout = vout
                    ln_vout[subset] = np.log(vout[subset])
                    chicomb = (ln_vout-ln_observed_sed_nuFnu)**2/(ln_err_obs**2)# + (ln_vout*np.log(model_noise_frac))**2)
                else:
                    chicomb = (vout-observed_sed_nuFnu)**2/(err_obs**2)# + (vout*model_noise_frac)**2)
                chisqs[i_d, i_a, i_r] = np.asarray(chicomb).sum()#/(dof+additional_free) #Normalize to Reduced chi square.

    wmin = np.where(chisqs == np.nanmin(chisqs))
    sed_chisqr = np.nanmin(chisqs)
    best_distance = a_distance[wmin[0]]
    best_av = a_av[wmin[1]]
    best_rv = a_rv[wmin[2]]
    #print 'distance: ',best_distance, ' av: ',best_av,' rv: ',best_rv

    # Now regenerate the model SED to pass to the goodness of fit calculator
    ext = np.asarray(utils.ccm_extinction(best_rv,wave_mu)) # Use wavelength in Angstroms
    extinction = 10.0**((ext*best_av)/(-2.5))
    best_model = (np.multiply(model_sed_1pc,extinction))/(best_distance)**2
    residuals = observed_sed_nuFnu - best_model

    return residuals, best_distance, best_av, best_rv, sed_chisqr


def image_chisqr(modelresults, observations, wavelength=None, write=True,
                 normalization='total', registration='sub_pixel',
                 inclinationflag=True, convolvepsf=True, background=0.0):
    """
    Not written yet - this is just a placeholder

    Parameters
    ------------
    wavelength : float
        Wavelength of the image to compute the chi squared of.
    write : bool
        If set, write output to a file, in addition to displaying
        on screen.

    """

    if inclinationflag:
        mod_inclinations = modelresults.parameters.inclinations
    else:
        mod_inclinations = ['0.0']

    im = observations.images
    mask = im[wavelength].mask
    image = im[wavelength].image
    noise = im[wavelength].uncertainty
    if convolvepsf:
        psf = im[wavelength].psf
    model = modelresults.images[wavelength].data
    
    #mask[:,:]=1
    sz = len(mod_inclinations)
    chisqr = np.zeros(sz)

    for n in np.arange(sz):
        if inclinationflag:
            model_n = np.asarray(model[0,0,n,:,:])
        else:
            model_n = np.asarray(model)

        # Convolve the model image with the appropriate psf
        if convolvepsf:
            model_n = np.asarray(image_registration.fft_tools.convolve_nd.convolvend(model_n,psf))
        # Determine the shift between model image and observations via fft cross correlation

        # Normalize model to observed image and calculate chisqrd
        background=np.min(noise)
        background=0.0
        model_n+=background
        if normalization == 'total':
            weightgd=image.sum()/model_n.sum()
        elif normalization == 'peak':
            weightgd=image.max()/model_n.max()
        else:
            weightgd = 1.0
        model_n*=weightgd
        subgd=image-model_n
        print 'subgd  ',np.sum(np.square(subgd))
        print 'normalization = ',weightgd
       

        #model_n=np.multiply(model_n,mask)
        #image=np.multiply(image,mask)
        dy,dx,xerr,yerr = image_registration.chi2_shift(model_n,image)

        if registration == 'integer_pixel':
            dx = np.round(dx)
            dy = np.round(dy)
        #if registration == 'sub_pixel':
            #print dx, dy
        # Shift the model image to the same location as observations
        model_n = sciim.interpolation.shift(model_n,np.asarray((dx,dy)))
        print 'subgd  ',np.max(image-model_n)

        chisquared=(image-model_n)**2.0/noise**2.0
        chisqr[n]=chisquared[mask !=0].sum()#/2500.0
        if dx == 0 or dy == 0:
            chisqr[n]=chisqr[n-1]+1.0

#        modelresults.images.closeimage
        _log.info( "inclination {0} : {1:4.1f} deg has chi2 = {2:5g}".format(n, mod_inclinations[n], chisqr[n]))

    return chisqr


def image_likelihood(modelresults, observations, image_covariance, wavelength=None, write=True,
        normalization='total', registration='sub_pixel'):
    """
    Not written yet - this is just a placeholder

    Parameters
    ------------
    wavelength : float
        Wavelength of the image to compute the log likelihood of a model image.
    write : bool
        If set, write output to a file, in addition to displaying
        on screen.

    """

    mod_inclinations = modelresults.parameters.inclinations

    im = observations.images
    mask = im[wavelength].mask
    image = im[wavelength].image
    noise = im[wavelength].uncertainty
    psf = im[wavelength].psf
    model = modelresults.images[wavelength].data


    #mask[:,:]=1
    sz = len(mod_inclinations)
    chisqr = np.zeros(sz)
    loglikelihood = np.zeros(sz)


    # Unpack covariance structure
    covariance = image_covariance[0]
    covariance_inv = image_covariance[1]
    logdet = image_covariance[2]
    sign = image_covariance[3]



    for n in np.arange(sz):
        model_n = np.asarray(model[0,0,n,:,:])

        # Convolve the model image with the appropriate psf
        model_n = np.asarray(image_registration.fft_tools.convolve_nd.convolvend(model_n,psf))
        # Determine the shift between model image and observations via fft cross correlation

        # Normalize model to observed image and calculate chisqrd
        weightgd=image.sum()/model_n.sum()
        model_n*=weightgd



        #model_n=np.multiply(model_n,mask)
        #image=np.multiply(image,mask)
        dy,dx,xerr,yerr = image_registration.chi2_shift_iterzoom(model_n,image)

        if registration == 'integer_pixel':
            dx = np.round(dx)
            dy = np.round(dy)
        #if registration == 'sub_pixel':
            #print dx, dy
        # Shift the model image to the same location as observations
        model_n = scipy.ndimage.interpolation.shift(model_n,np.asarray((dx,dy)))

        matrix_model = model_n[mask != 0]
        matrix_obs = image[mask != 0]
        residual = matrix_obs - matrix_model
        nx = residual.shape[0]

        residual = residual[:,None] # Changing this to a column vector
        #print residual.min()
        #print residual.max()
        #print np.transpose(residual).shape
        matrix_product = np.dot(np.dot(np.transpose(residual),covariance_inv),residual)
        loglikelihood[n] = -0.5*(matrix_product[0][0] + sign*logdet + nx*np.log(2.0*np.pi))
        #print 'matrix_product',matrix_product[0][0]
        #print 'logdet',logdet

        chisquared=(image-model_n)**2.0/noise**2.0
        chisqr[n]=chisquared[mask !=0].sum()#/(mask[mask != 0].shape[0])
        chisqr = np.asarray(chisqr)

        _log.info( "inclination {0} : {1:4.1f} deg has chi2 = {2:7f}".format(n, mod_inclinations[n], chisqr[n]))
        _log.info( "inclination {0} : {1:4.1f} deg has loglike = {2:7f}".format(n, mod_inclinations[n], loglikelihood[n]))

    return loglikelihood



def sed_likelihood(modelresults, observations, sed_covariance,dof=1,
    write=True,
    plot=False, save=True,
    vary_distance=False, distance_range=None,
    vary_AV=False, AV_range=[0,10],
    **kwargs):

    """ Compute chi^2 for a given model

    Not written yet - this is just a placeholder

    Parameters
    -----------
    dof : int
        Number of degrees of freedom to use in computing the
        reduced chi^2. Will be incremented appropriately if
        vary_distance or vary_AV are set.
    vary_distance : bool
        Allow the distance to the target to vary
    distance_range : 2 element iterable
        Min and max allowable distance, in pc
    vary_AV : bool
        Allow optical extinction (A_V) to vary
    AV_range : 2 element iterable
        Min and max allowable A_V
    RV : float
        Reddening law color parameter for extinction
    write : bool
        Write output to disk as chi^2 results FITS file in the model directory?
        Default is True
    save : bool
        Save results to disk as FITS bintable? Default is True.
    plot : bool
        Display plot of chi^2 fit?

    """

#    if not isinstance(modelresults, models.ModelResults):
#        print type(modelresults)
#        raise ValueError("First argument to sed_chisqr must be a ModelResults object")
#
#    if not isinstance(observations, models.Observations):
#        raise ValueError("Second argument to sed_chisqr must be an Observations object")
    my_dof = dof
    if vary_distance:
        #raise NotImplementedError("Varying distance not yet implemented")
        my_dof += 1
    if vary_AV:
        #_log.info("Computing chi^2 while allowing A_V to vary between {0} and {1} with R_V={2}".format(AV_range[0], AV_range[1], RV) )
        my_dof += 1


    # Unpack covariance structure
    covariance = sed_covariance[0]
    covariance_inv = sed_covariance[1]
    logdet = sed_covariance[2]
    sign = sed_covariance[3]
    #print 'logdet det: ',sign*logdet
    #print 'det: ',np.log(np.linalg.det(covariance))

    # observed wavelengths and fluxes
    obs_wavelengths = observations.sed.wavelength
    obs_nufnu = observations.sed.nu_fnu
    obs_nufnu_uncert = observations.sed.nu_fnu_uncert
    #print obs_nufnu_uncert.value
    # model wavelengths and fluxes. Fluxes will be a 2D array: [ninclinations, nwavelengths]
    mod_wavelengths = modelresults.sed.wavelength
    mod_nufnu= modelresults.sed.nu_fnu
    mod_inclinations = modelresults.parameters.inclinations

    if plot:
        observations.sed.plot(**kwargs)

    ninc = mod_inclinations.size
    chi2s = np.zeros(ninc)
    loglikelihood = np.zeros(ninc)
    #avs = np.zeros(ninc)
    #rvs = np.zeros(ninc)
    #distances = np.zeros(ninc) + modelresults.parameters.distance
    # iterate over inclinations, computing estimated observations and fitting
    # allow reddening to vary
    for i in range(ninc):

        interpolator = scipy.interpolate.interp1d(mod_wavelengths, mod_nufnu[i], kind='linear', copy=False)
        est_mod_nufnu = interpolator(obs_wavelengths)

        if plot:
            modelresults.sed.plot(inclination=mod_inclinations[i], overplot=True)
            ax = plt.gca()
            color = ax.lines[-1].get_color()
            ax.plot(obs_wavelengths, est_mod_nufnu, color=color, marker='s', linestyle='none')

        if vary_distance or vary_AV:
            residuals, best_distance, best_av, best_rv,chi2 =  fit_dist_extinct(obs_wavelengths, obs_nufnu.value, est_mod_nufnu, obs_nufnu_uncert.value, additional_free=my_dof, distance_range=distance_range, vary_av=vary_AV, vary_distance=vary_distance, av_range=AV_range)
        else:
            best_model = est_mod_nufnu
            residuals = np.asarray(obs_nufnu.value - best_model) #.to(units.erg/units.cm**2)
            chi2 = ((obs_nufnu.value - est_mod_nufnu)**2 / obs_nufnu_uncert.value**2).sum()


        residuals = residuals[:,None] # Changing this to a column vector

        nx = residuals.shape[0]

        # Now compute the log likelihood. Functional Form:
        # ln p(D|M) = -1/2 (R^TC^-1R + ln detC + Npix ln2pi)
        matrix_producta = np.dot(np.transpose(residuals),covariance_inv)
        matrix_product = np.dot(matrix_producta,residuals)
        loglikelihood[i] = -0.5*(matrix_product[0] + logdet + nx*np.log(2.0*np.pi))


        _log.info( "inclination {0} : {1:4.1f} deg has loglike = {2:15f}".format(i, mod_inclinations[i], loglikelihood[i]))
        _log.info( "inclination {0} : {1:4.1f} deg has chi2 = {2:15f}".format(i, mod_inclinations[i], chi2))
        #print "obs = ", obs_nufnu.value
        #print "mod = ", est_mod_nufnu


        chi2s[i] = chi2



    if save:
        import astropy.table
        import os
        tabledict = {'INCLINATION': mod_inclinations, 'SED_CHISQ': chi2s, 'AV':avs, 'RV':rvs, 'DISTANCE':distances}
        meta = {'SOURCE': 'Python mcfost.chisqr.sed_chisqr()',
                'VARY_AV': vary_AV,
                'VARYDIST': vary_distance,
                'AVRANGE': str(AV_range),
                'DISTRNGE': str(distance_range) }
        tab = astropy.table.Table(tabledict, meta=meta)

        tab.write(os.path.join(modelresults.directory, 'observables', 'py_sed_chisq.fits'), overwrite=True)


    #print loglikelihood
    return loglikelihood
