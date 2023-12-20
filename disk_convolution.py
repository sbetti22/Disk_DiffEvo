"""
Disk convolution code for JWST/NIRCam position dependent PSF convolution
author: Sarah Betti
formalized on: 12/20/2023 

based off of PSF modeling by: Kellen Lawson (https://ui.adsabs.harvard.edu/abs/2023AJ....166..150L/graphics) utilizing webbpsf (M. Perrin; https://github.com/spacetelescope/webbpsf) and webbpsf_ext (J. Leisenring; https://github.com/JarronL/webbpsf_ext).

The webbpsf PSF convolution is based off of the webbpsf_ext example > https://github.com/JarronL/webbpsf_ext/blob/main/notebooks/NIRCam_MASK430R_F356W_Vega.ipynb

"""

import numpy as np
import os

from astropy.io import fits
from astropy import constants as sc
from scipy import ndimage

# Progress bar
from tqdm.auto import trange, tqdm

import webbpsf_ext, pysiaf
from webbpsf_ext import image_manip, setup_logging, spectra, coords
from webbpsf_ext.coords import jwst_point
from webbpsf_ext import miri_filter, nircam_filter, bp_2mass
from webbpsf_ext.image_manip import pad_or_cut_to_size
from webbpsf_ext import stellar_spectrum

def make_model_mJypx(model_path, wavelength):
    '''
    convert MCFOST model from W/m2 to mJy/pixel
    '''
    disk_model_Wm2 = (fits.open(model_path)[0].data[0,0,0,:,:])
    freq = sc.c.value / (wavelength * 1e-6)

    disk_model_mJy_px = ( (1e26 * disk_model_Wm2  / freq) * 1e3 ) 
    mod_nx, mod_ny = disk_model_Wm2.shape[1], disk_model_Wm2.shape[0]
    mod_cx, mod_cy = mod_nx//2, mod_ny//2

    # mask star following Johan
    h, w = disk_model_Wm2.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - mod_cx)**2 + (Y-mod_cy)**2)
    mask = dist_from_center <= 2
    disk_model_mJy_px[mask] = 0

    path = model_path.split('.fits')[0]

    fits.writeto(path + f'_mJypx.fits', disk_model_mJy_px, header = fits.getheader(model_path), overwrite=True)

    return path+f'_mJypx.fits'   


def make_spec(name=None, sptype=None, flux=None, flux_units=None, bp_ref=None, **kwargs):
    """
    Create pysynphot stellar spectrum from input dictionary properties.
    """
    # Renormalization arguments
    renorm_args = (flux, flux_units, bp_ref)
    
    # Create spectrum
    sp = stellar_spectrum(sptype, *renorm_args, **kwargs)
    if name is not None:
        sp.name = name
    
    return sp

def make_psfs(ROLL_REF_ANGLE, obj_params, filt, obsdate='2023-08-24T22:49:38.762', **kwargs):
    '''
    make position dependent PSFs
    #
        # # Information necessary to create pysynphot spectrum of star
    # obj_params = {
    #     'name': '49 Ceti', 
    #     'sptype': 'A0V', 
    #     'Teff': 9817, 'log_g': 4.0, 'metallicity': -0.5, 
    #     'dist': 59,
    #     'flux': 5.458, 'flux_units': 'vegamag', 'bp_ref': bp_2mass('k'),
    #     'RA_obj'  : +36.48744,  # RA (decimal deg) of source
    #     'Dec_obj' :  -12.29054,  # Dec (decimal deg) of source
    # }
    '''
    print('  --> starting make_psfs')
    if isinstance(obj_params['bp_ref'], str):
        obj_params['bp_ref'] = eval(obj_params['bp_ref'])

    # Mask information
    mask = kwargs.get('mask','MASK335R')
    pupil = kwargs.get('pupil', 'MASKRND')

    # Initiate instrument class with selected filters, pupil mask, and image mask
    inst = webbpsf_ext.NIRCam_ext(filter=filt, pupil_mask=pupil, image_mask=mask)

    # Set desired PSF size and oversampling
    inst.fov_pix = kwargs.get('fov',320)
    inst.oversample = kwargs.get('oversample', 2)

    if filt == 'F200W':
        detector = kwargs.get('detector', 'NRCA2')
        inst.detector=detector
        inst.aperturename=f'{detector}_{mask}'
        # Observed and reference apertures
        ap_obs = f'{detector}_{mask}'
        ap_ref = f'{detector}_{mask}'
    elif filt == 'F444W':
        detector = kwargs.get('detector', 'NRCA5')
        inst.detector=detector
        inst.aperturename=f'{detector}_{mask}'
        # Observed and reference apertures
        ap_obs = f'{detector}_{mask}'
        ap_ref = f'{detector}_{mask}'
    else:
        raise ValueError('filter does not exist in code.  need to add it')

    # Calculate PSF coefficients
    inst.gen_psf_coeff()

    # Calculate position-dependent PSFs due to FQPM
    # Equivalent to generating a giant library to interpolate over
    inst.gen_wfemask_coeff()

    inst.load_wss_opd_by_date(obsdate, plot=False)
    
    # Define the RA/Dec of reference aperture and telescope position angle
    # Position angle is angle of V3 axis rotated towards East
    ra_ref, dec_ref = (obj_params['RA_obj'], obj_params['Dec_obj']) 
    pos_ang = ROLL_REF_ANGLE 

    # Set any baseline pointing offsets (e.g., specified in APT's Special Requirements)
    base_offset=(0,0)
    # Define a list of nominal dither offsets
    dith_offsets = kwargs.get('offsets', [(0,0)]) 

    # Telescope pointing information
    tel_point = jwst_point(ap_obs, ap_ref, ra_ref, dec_ref, pos_ang=pos_ang,
                        base_offset=base_offset, dith_offsets=dith_offsets)

    # Create stellar spectrum and add to dictionary
    sp_star = make_spec(**obj_params)
    obj_params['sp'] = sp_star

    # # Get sci position shifts from center in units of detector pixels
    # siaf_ap = tel_point.siaf_ap_obs
    return inst, tel_point, obj_params

def make_psfgrid(inst, tel_point, grid_shape='circle'):
    print('  --> starting make_psfgrid')
    siaf_ap = tel_point.siaf_ap_obs
    if grid_shape == 'circle':
        thvals = np.linspace(0, 360, 9, endpoint=False)
    else:
        thvals = np.linspace(0, 360, 4, endpoint=False)
    field_rot = 0 if inst._rotation is None else inst._rotation
    rvals = 10**(np.linspace(-2,1,7))
    rvals_all = [0]
    thvals_all = [0]
    for r in rvals:
        for th in thvals:
            rvals_all.append(r)
            thvals_all.append(th)
    rvals_all = np.array(rvals_all)
    thvals_all = np.array(thvals_all)

    xgrid_off, ygrid_off = coords.rtheta_to_xy(rvals_all, thvals_all)

    # Science positions in detector pixels
    xoff_sci_asec, yoff_sci_asec = coords.xy_rot(-1*xgrid_off, -1*ygrid_off, -1*field_rot)
    xsci = xoff_sci_asec / siaf_ap.XSciScale + siaf_ap.XSciRef
    ysci = yoff_sci_asec / siaf_ap.YSciScale + siaf_ap.YSciRef

    xtel, ytel = siaf_ap.convert(xsci, ysci, 'sci', 'tel')

    hdul_psfs = inst.calc_psf_from_coeff(coord_vals=(xtel, ytel), coord_frame='tel', return_oversample=True)
    return hdul_psfs

def convolve_disk(inst, tel_point, obj_params, hdul_psfs, disk_model_path, modelpixelscale, wavelength, distance, verbose=False ):
    model_mJypx_path = make_model_mJypx(disk_model_path, wavelength)
    siaf_ap = tel_point.siaf_ap_obs

    # Disk model information
    modelpixelscale = fits.getheader(disk_model_path)['CDELT2'] * 3600 
    disk_params = {
        'file': model_mJypx_path,
        'pixscale': modelpixelscale,
        'wavelength': wavelength,
        'units': 'mJy/pixel',
        'dist' : distance,
        'cen_star' : False,
    }

    # Open model and rebin to PSF sampling
    # Scale to instrument wavelength assuming grey scattering function
    # Converts to phot/sec/lambda
    # Crop to twice the FoV
    npix = int(2*siaf_ap.YSciSize*inst.oversample)
    hdul_disk_model = image_manip.make_disk_image(inst, disk_params, sp_star=obj_params['sp'], shape_out=npix)

    # Rotation necessary to go from sky coordinates to 'idl' frame
    rotate_to_idl = -1*(tel_point.siaf_ap_obs.V3IdlYAngle + tel_point.pos_ang)

    # Select the first dither location offset
    delx, dely = tel_point.position_offsets_act[0]
    hdul_out = image_manip.rotate_shift_image(hdul_disk_model, angle=rotate_to_idl,
                                            delx_asec=delx, dely_asec=dely)

    sci_cen = (siaf_ap.XSciRef, siaf_ap.YSciRef)

    # Distort image on 'sci' coordinate grid
    im_sci, xsci_im, ysci_im = image_manip.distort_image(hdul_out, ext=0, to_frame='sci', return_coords=True, 
                                                        aper=siaf_ap, sci_cen=sci_cen)

    # If the image is too large, then this process will eat up much of your computer's RAM
    # So, crop image to more reasonable size (20% oversized)
    osamp = inst.oversample
    xysize = int(1.2 * np.max([siaf_ap.XSciSize,siaf_ap.YSciSize]) * osamp)
    xy_add = osamp - np.mod(xysize, osamp)
    xysize += xy_add

    im_sci_crop = pad_or_cut_to_size(im_sci, xysize)

    # Multiply times ND mask
    if inst.name=='NIRCam':
        im_mask = inst.gen_mask_image(npix=xysize)
        rarr, tharr = coords.dist_image(im_mask, pixscale=inst.pixelscale, return_theta=True)
        xarr, yarr = coords.rtheta_to_xy(rarr, tharr)
        ind_clear = np.abs(yarr)<4
        im_mask[ind_clear] = 1
        im_sci_crop = im_sci_crop * im_mask

    hdul_out[0].header['CFRAME'] = 'sci'
    hdul_disk_model_sci = fits.HDUList(fits.PrimaryHDU(data=im_sci_crop, header=hdul_out[0].header))

    # Convolve image
    im_conv = image_manip.convolve_image(hdul_disk_model_sci, hdul_psfs)

    # Add cropped image to final oversampled image
    im_conv = pad_or_cut_to_size(im_conv, (640,640))

    # # Rebin science data to detector pixels
    im_sci = image_manip.frebin(im_conv, scale=1/osamp)

    # De-rotate to sky orientation
    imrot = image_manip.rotate_offset(im_sci, rotate_to_idl, reshape=False, cval=np.nan)

    #  convert to mJy/as2
    imrot = imrot / (inst.pixelscale**2.)

    # Save image to FITS file
    hdu_diff = fits.PrimaryHDU(imrot)

    copy_keys = [
        'PIXELSCL', 'DISTANCE', 
        'INSTRUME', 'FILTER', 'PUPIL', 'CORONMSK',
        'APERNAME', 'MODULE', 'CHANNEL',
        'DET_NAME', 'DET_X', 'DET_Y', 'DET_V2', 'DET_V3'
    ]

    hdr = hdu_diff.header
    for head_temp in (inst.psf_coeff_header, hdul_out[0].header):
        for key in copy_keys:
            try:
                hdr[key] = (head_temp[key], head_temp.comments[key])
            except (AttributeError, KeyError):
                pass

    hdr['PIXELSCL'] = inst.pixelscale

    path = disk_model_path.split('.fits')[0]
    outfile = path+ f'_model_{inst.aperturename}_{inst.filter}_mJyas2.fits'
    if verbose: print('writing convolved image to: ', outfile)
    hdu_diff.writeto(outfile, overwrite=True)

    return imrot