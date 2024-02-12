# Various utility functions
import matplotlib
import numpy as np
import logging
#import image_registration
#import astropy.io.fits as fits
import scipy.interpolate
#import astropy.units as units



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


def imshow_with_mouseover(ax, *args, **kwargs):
    """ Wrapper for pyplot.imshow that sets up a custom mouseover display formatter
    so that mouse motions over the image are labeled in the status bar with
    pixel numerical value as well as X and Y coords.

    Why this behavior isn't the matplotlib default, I have no idea...
    """
    myax = ax.imshow(*args, **kwargs)
    aximage = ax.images[0].properties()['array']
    # need to account for half pixel offset of array coordinates for mouseover relative to pixel center,
    # so that the whole pixel from e.g. ( 1.5, 1.5) to (2.5, 2.5) is labeled with the coordinates of pixel (2,2)
    report_pixel = lambda x, y : "(%5.1f, %5.1f)   %g" % \
        (x,y, aximage[np.floor(y+0.5),np.floor(x+0.5)])
    ax.format_coord = report_pixel

    return ax


# purely cosmetic: print e.g. '100' instead of '10^2' for axis labels with small exponents
class NicerLogFormatter(matplotlib.ticker.LogFormatter):
    """ A LogFormatter subclass to print nicer axis labels
        e.g. '100' instead of '10^2'

        Parameters                                                                                                   ----------
        threshhold : int
            Absolute value of log base 10 above which values are printed in exponential notation.
            By default this is 3, which means you'll get labels like                                                     10^-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10^4 ...
                                                                                                                     usage:
          ax = gca()
          ax.set_yscale("log")
          ax.set_xscale("log")
          ax.xaxis.set_major_formatter(NiceLogFormatter())
    """
    def __init__(self, threshhold=3):
        self.threshhold = threshhold
    def __call__(self,val,pos=None):
        if abs(np.log10(val)) > self.threshhold:
            return "$10^{%d}$" % np.log10(val)
        elif val >= 1:
            return "%d"%val
        else:
            return "%.1f"%val



def setup_logging(level='INFO',  filename=None, verbose=False):
    """ Simple wrapper function to set up convenient log defaults, for
    users not familiar with Python's logging system.

    """
    import logging
    _log = logging.getLogger('mcfost')

    lognames=['mcfost']

    if level.upper() =='NONE':
        # disable logging
        lev = logging.CRITICAL  # we don't generate any CRITICAL flagged log items, so
                                # setting the level to this is effectively the same as ignoring
                                # all log events. FIXME there's likely a cleaner way to do this.
        if verbose: print("No log messages will be shown.")
    else:
        lev = logging.__dict__[level.upper()] # obtain one of the DEBUG, INFO, WARN, or ERROR constants
        if verbose: print("Log messages of level {0} and above will be shown.".format(level))

    for name in lognames:
        logging.getLogger(name).setLevel(lev)
        _log.debug("Set log level of {name} to {lev}".format(name=name, lev=level.upper()))

    # set up screen logging
    logging.basicConfig(level=logging.INFO,format='%(name)-10s: %(levelname)-8s %(message)s')
    if verbose: print("Setup_logging is adjusting Python logging settings.")


    if str(filename).strip().lower() != 'none':
        hdlr = logging.FileHandler(filename)

        formatter = logging.Formatter('%(asctime)s %(name)-10s: %(levelname)-8s %(message)s')
        hdlr.setFormatter(formatter)

        for name in lognames:
            logging.getLogger(name).addHandler(hdlr)

        if verbose: print("Log outputs will also be saved to file "+filename)
        _log.debug("Log outputs will also be saved to file "+filename)

def find_closest(list_, item):
    """ return the index for the entry closest to a desired value """
    minelt = min(list_,key=lambda x:abs(x-item))
    return np.where(np.asarray(list_) == minelt)[0][0]

def ccm_extinction(Rv, lambda_ang):

    """
    Python implementation of the idl_lib extinction correction function
    to be called by the SED chisqrd fitting method in the python version of
    MCRE. Accepts Rv, the reddening index at V (Default = 3.1) and the
    wavelength in Angstroms. Extinction curve A(lambda)/A(V) is returned.
    NOTE: lambda_ang is the wavelength in microns
    """
    
    lambda_ang = np.asarray(lambda_ang)
    #lambda_ang = lambda_ang*0.0001 
    inv_lam = 1.0/lambda_ang
    
    s = len(lambda_ang)
    a = np.zeros((s))
    b = np.zeros((s)) # confirm proper syntax

    # Range that CCM restrict it to.
    ir = inv_lam<=1.1
    #print 'ir',ir
    c_ir = len(ir)
    #flags = choose(greater(inv_lam,1.1),(-1,1))

    a[ir] = 0.574*inv_lam[ir]**1.61
    b[ir] = -0.527*inv_lam[ir]**1.61

    #opt = where inv_lam > 1.1 and inv_lam <= 3.3 then c_opt
    opt = ((inv_lam > 1.1) & (inv_lam <= 3.3))
    c_opt = len(opt)
    y = np.asarray(inv_lam[opt] - 1.82)
    a[opt] = 1+ 0.17699*y-0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5-0.77530*y**6 + 0.32999*y**7
    b[opt] = 1.41338*y + 2.28306*y**2 +1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

    #uv = where inv_lam > 3.3 and inv_lam <= 8.0 then c_uv
    uv = ((inv_lam > 3.3) & (inv_lam <= 8.0))
    c_uv = len(uv)
    x = inv_lam[uv]
    xm = x - 5.9
    fa = -0.04473*(xm)**2 - 0.009779*(xm)**3
    fb = 0.2130*(xm)**2 + 0.1207*(xm)**3
    nulls = xm <= 0
    fa[nulls] = 0.0
    fb[nulls] = 0.0

    a[uv] = 1.752 - 0.316*x - 0.104/( (x-4.67)**2 +0.341) + fa
    b[uv] = -3.090 + 1.825*x + 1.206/( (x-4.62)**2 + 0.263) + fb

    # Compute the extintion at each wavelength and return

    A_lambda = np.asarray((a+b)/Rv)

    return A_lambda


def sed_from_vizier(vizier_fn,from_file=False,radius=2.0,refine=False, variable=0.0):

    """
    Generate spectral energy distribution file in the format expected 
    for an observed SED from a Vizier generated SED file. Output file
    includes wavelength in microns, flux in janskys, uncertainty in 
    janskys, and a string giving the source of the photometry.
    
    Parameters
    ----------
    vizier_fn: string
        name of object
        alternatively, if from_file is true, this is the
            filename of the votable.
    radius: float
        position matching in arseconds.
    from_file: boolean
        set to True if using previously generated votable
    refine: boolean
        Set to True to get rid of duplicates, assign missing 
        uncertainties, sorting 
    variable: float
        Minimum percentage value allowed for uncertainty
        e.g. 0.1 would require 10% uncertainties on all measurements.
     
    Returns
    -------
    table: astropy.Table
        VO table returned by the Vizier service.
         
    """

    from astroquery.vizier import Vizier
    import astropy.units as u
    from StringIO import StringIO as BytesIO
    from httplib import HTTPConnection
 
    from astropy.table import Table
    
    from astropy.coordinates import SkyCoord
    import numpy as np
    
    if from_file:
        from astropy.io.votable import parse_single_table
        table = parse_single_table(vizier_fn)


        sed_freq = table.array['sed_freq']# frequency in Ghz
        sed_flux = table.array['sed_flux']# flux in Jy
        sed_eflux = table.array['sed_eflux']# flux error in Jy
        sed_filter = table.array['sed_filter']# filter for photometry
    else:
    
        try:
            coords = SkyCoord.from_name(vizier_fn)
        except:
            print('Object name was not resolved by Simbad. Play again.')
            return False
        pos = np.fromstring(coords.to_string(), dtype=float, sep=' ')  
        ra, dec = pos
        target = "{0:f},{1:f}".format(ra, dec)

        # Queue Vizier directly without having to use the web interface
        # Specify the columns:
        #v = Vizier(columns=['_RAJ2000', '_DEJ2000','_sed_freq', '_sed_flux', '_sed_eflux','_sed_filter'])
        #result = v.query_region(vizier_fn, radius=2.0*u.arcsec)
        #print result

        url = "http:///viz-bin/sed?-c={target:s}&-c.rs={radius:f}"
        host = "vizier.u-strasbg.fr"
        port = 80
        path = "/viz-bin/sed?-c={target:s}&-c.rs={radius:f}".format(target=target, radius=radius)
        connection = HTTPConnection(host, port)
        connection.request("GET", path)
        response = connection.getresponse()

        table = Table.read(BytesIO(response.read()), format="votable")

        sed_freq = table['sed_freq'].quantity.value
        sed_flux = table['sed_flux'].quantity.value
        sed_eflux = table['sed_eflux'].quantity.value
        sz = sed_flux.shape[0]
        filters = []
        for i in np.arange(sz):
            filters.append(table['sed_filter'][i])
        sed_filter = np.asarray(filters)

    

    wavelength = 2.99e14/(sed_freq*1.0e9)# wavelength in microns
    flux = sed_flux # flux in Jy
    uncertainty = sed_eflux # uncertainty in Jy
    source = sed_filter # string of source of photometry
    
    sz = np.shape(wavelength)[0]

    
    if refine:
        
        # if uncertainty values don't exist, generate
        # new uncertainty for 3*flux.
        for j in np.arange(sz):
            if np.isnan(uncertainty[j]):
                uncertainty[j] = flux[j]*3
                
        # Remove duplicate entries based on filter names.
        # Choose entry with smallest error bar for duplicates.
        filter_set = list(set(source))
        wls = []
        jys = []
        ejys = []
        for filt in filter_set:
            inds = np.where(source == filt)
            filt_eflux = uncertainty[inds]
            picinds = np.where(filt_eflux == np.min(filt_eflux))
            wls.append(wavelength[inds][picinds][0])
            jys.append(flux[inds][picinds][0])
            ejys.append(uncertainty[inds][picinds][0])

        # Sort by increasing wavelength
        sortind = np.argsort(np.asarray(wls))
        wavelength = np.asarray(wls)[sortind]
        flux = np.asarray(jys)[sortind]
        uncertainty = np.asarray(ejys)[sortind]
        source = np.asarray(filter_set)[sortind]
        
        sz = np.shape(wavelength)[0]
     
    
    # Update uncertainties for value of variable
    for j in np.arange(sz):
        if uncertainty[j] < flux[j]*variable:
            uncertainty[j] = flux[j]*variable

            
    txtfn = 'observed_sed'+vizier_fn.replace(" ", "")+'.txt'
    f = open(txtfn,'w')
    f.write('#  Wavelength (microns)'+"\t"+'Flux (Jy)'+"\t"+'Uncertainty (Jy)'+"\t"+'Source'+"\n")

    for j in np.arange(sz):

        f.write(str(wavelength[j])+"\t"+str(flux[j])+"\t"+str(uncertainty[j])+"\t"+str(source[j])+"\n")

    f.close()
        
    return table
