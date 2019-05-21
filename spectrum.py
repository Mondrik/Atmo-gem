import numpy as np
import os
from utils import linefitter as lf
from utils import utils
from utils import proc_image as pi
import astroscrappy as scrap
from specutils import Spectrum1D,SpectralRegion
from specutils.analysis import equivalent_width
import astropy.io.fits as pft
import astropy.units as u
from astropy.time import Time
from scipy.interpolate import UnivariateSpline
import scipy.stats
from multiprocessing import Pool
import copy
import pickle

#This funtion needs to be outside of the Spectrum class to allow
#for parallelization of the column fitting
def _fit_cols(args):
    #unpack args here
    col_num,col_data,xgrid,x,avoid,fit_type,fit_func,extraction_radius = args
    results_dict = {}
    results_dict['col_num'] = col_num #for convenience later
    results_dict['skipped'] = False
    if any([(col_num >= low) and (col_num <= high) for low,high in avoid]):
        results_dict['skipped'] = True
        return results_dict
    try:
        results_dict['my_fit'],results_dict['chi_sq'] = lf.fit_indiv_col(col_data,fit_type=fit_type)
        results_dict['fit'] = fit_func(xgrid,*results_dict['my_fit'])
        results_dict['model_col'] = fit_func(x,*results_dict['my_fit'])
        ##optimal counts: approx
        # -1 to lower extraction bound b/c of python 0 indexing
        low = np.argmax(col_data) - (extraction_radius - 1)
        high = np.argmax(col_data) + extraction_radius
        #fit func w/ no offset should be PSF
        temp = copy.deepcopy(results_dict['my_fit'])
        temp[-1] = 0
        norm = np.trapz(fit_func(xgrid,*temp),x=xgrid)
        denom = np.sum((fit_func(x[low:high],*temp)/norm)**2.).astype(np.float)
        num = np.sum(fit_func(x[low:high],*temp)/norm * col_data[low:high])
        results_dict['optimal_counts'] = num/denom
        results_dict['sum_counts'] = np.sum(col_data[low:high])
        results_dict['failed_fit'] = False
    except Exception as e:
        print('Failed to extract column {}'.format(col_num),e)
        results_dict['optimal_counts'] = np.nan
        results_dict['sum_counts'] = np.nan
        results_dict['chi_sq'] = np.nan
        results_dict['failed_fit'] = True
    return results_dict


class Spectrum():
    def __init__(self,filename,remove_cosmics=True,remove_sky=True,remove_avoidance=True,bootstrap=False,verbose=True):
        #get bias-subtracted and mosaiced image:
        self.filename = filename
        self.fits_file,self.data = pi.proc_gmoss_image(self.filename,verbose=verbose)
        basename = os.path.split(self.filename)
        if bootstrap:
            #bootstrapped image is poission resampling of the gain-corrected, bias-removed image
            self.data = scipy.stats.poisson.rvs(np.abs(self.data))
        self.avoid = utils.get_avoidance_regions()
        if remove_cosmics:
            self.mask,self.working_img = self.remove_cosmics()
        else:
            self.mask = np.ones_like(self.data,dtype=bool)
            self.working_img = self.data
        if remove_avoidance:
            self.remove_avoidance_regions()
        self.generate_uncert_array()
        if remove_sky:
            self.sky_arr = self.estimate_sky()
            self.sky_uncert = np.sqrt(np.median(self.sky_arr,axis=0) / self.sky_arr.shape[0])
            self.working_img = self.working_img - np.median(self.sky_arr,axis=0)
        self.pixels = np.arange(self.working_img.shape[1])

    def remove_cosmics(self,verbose=False):
        mask,clean_img = scrap.detect_cosmics(self.data,sigclip=8,objlim=2.5, \
                fsmode='convolve',psfmodel='gauss',psffwhm=5.0,verbose=verbose,niter=8)
        return mask,clean_img

    def remove_avoidance_regions(self):
        for low,high in self.avoid:
            self.working_img[:,low:high] = np.nan
        return None

    def generate_uncert_array(self):
        self.uncert_array = np.sqrt(np.abs(self.working_img)) + 2. #used to be 10.


    def estimate_sky(self,sky_start=40,region_size=30):
        sky_arr = np.zeros((region_size*2,self.working_img.shape[1]))
        #assume sky starts sky_start=30pix away from spectrum peak and sample region_size=30pix
        region_size -= 1 #need to remove 1 from region_size bc of python 0 indexing
        #ie, region size = 30 gives us 31 pixels on each side
        #search for argmax only in "center" 100 rows
        arg_max = 200+np.argmax(self.working_img[200:300,:],axis=0)
        #vector ennumerating rows of ccd
        row_vec = np.arange(self.working_img.shape[0])
        for i,x in enumerate(arg_max):
            if any([(i > low) and (i < high) for low,high in self.avoid]):
                continue
            region = np.logical_and(np.abs(row_vec-x)>=sky_start,\
                    np.abs(row_vec-x)<=sky_start+region_size)
            try:
                sky_arr[:,i] = self.working_img[region,i]
            except ValueError as e:
                print(self.working_img[:,i])
                print('Problem in estimate_sky: {} col: {}'.format(e,i))
                raise
        return sky_arr

    def _build_col_fit_list(self,fit_type,fit_func,extraction_radius):
        #col_num,col_data,xgrid,x,avoid,fit_type,fit_func,extraction_radius = args
        cols = np.arange(self.working_img.shape[1])
        x = np.arange(self.working_img.shape[0])
        xgrid = np.linspace(np.min(x),np.max(x),10000)
        col_fit_list = []

        for c in cols:
            col_fit_list.append([c,self.working_img[:,c],xgrid,x,self.avoid,fit_type,fit_func,extraction_radius])

        return col_fit_list

    def parallel_extract(self,fit_type='moffat',extraction_radius=30,n_procs=5):
        """
        Given a fit_type, construct an integrated spectrum, and also a model image following results of the fit.  Also construct an array of the best fit parameters.
        INPUTS:
            fit_type            ::  Type of analytic function to fit LSF at each column. Options = ['voigt','moffat','gaussian'] (default: moffat)
            extraction_bounds   ::  Width of spectral extraction region.  Larger extraction bounds include more pixels in extraction.  Defined as extent from peak pixel (ie, 2xextraction_bounds pixels are included in the integration).  Default = 30.
            n_procs             ::  Number of processors to use
        OUTPUTS:
            NONE
        No outputs, but allows access to self.optimal_counts (pseudo-optimal extraction over extraction_bounds as above)
        Pseudo-optimal because we are using an analytic PSF rather than an empirical/exact PSF.
        """
        self.optimal_counts = np.zeros(self.working_img.shape[1])
        self.sum_counts = np.zeros(self.working_img.shape[1])
        self.chi_sq = np.zeros(self.working_img.shape[1])
        self.model_img = np.zeros_like(self.working_img)
        self.fit_func_name = fit_type
        self.fit_func = lf.get_fit_func(fit_type)
        #initialize all best fit params to nan, then replace if fit converges
        self.model_bf_params = np.ones((self.working_img.shape[1],lf.get_num_params(fit_type)))*np.nan
        self.failed_fit_cols = []
        col_fit_list = self._build_col_fit_list(fit_type,self.fit_func,extraction_radius)

        with Pool(n_procs) as mypool:
            results_list = mypool.map(_fit_cols,col_fit_list)
        self.results_list = results_list

        for res in self.results_list:
            if res['skipped']:
                continue
            if res['failed_fit']:
                self.failed_fit_cols.append(res['col_num'])
                continue
            self.optimal_counts[res['col_num']] = res['optimal_counts']
            self.sum_counts[res['col_num']] = res['sum_counts']
            self.chi_sq[res['col_num']] = res['chi_sq']
            self.model_img[:,res['col_num']] = res['model_col']
            self.model_bf_params[res['col_num'],:] = res['my_fit']


        self.failed_fit_cols = np.asarray(self.failed_fit_cols)
        self.optimal_counts[self.optimal_counts==0] = np.nan

    def extract(self,fit_type='moffat',extraction_radius=30):
        """
        Given a fit_type, construct an integrated spectrum, and also a model image following results of the fit.  Also construct an array of the best fit parameters.
        INPUTS:
            fit_type            ::  Type of analytic function to fit LSF at each column. Options = ['voigt','moffat','gaussian'] (default: voigt)
            extraction_bounds   ::  Width of spectral extraction region.  Larger extraction bounds include more pixels in extraction.  Defined as extent from peak pixel (ie, 2xextraction_bounds pixels are included in the integration).  Default = 30.
        OUTPUTS:
            NONE
        No outputs, but allows access to self.optimal_counts (pseudo-optimal extraction over extraction_bounds as above)
        Pseudo-optimal because we are using an analytic LSF rather than an empirical/exact LSF.
        """
        self.optimal_counts = np.zeros(self.working_img.shape[1])
        self.sum_counts = np.zeros(self.working_img.shape[1])
        self.chi_sq = np.zeros(self.working_img.shape[1])
        cols = np.arange(self.working_img.shape[1])
        x = np.arange(self.working_img.shape[0])
        xgrid = np.linspace(np.min(x),np.max(x),10000)
        self.model_img = np.zeros_like(self.working_img)
        self.fit_func_name = fit_type
        self.fit_func = lf.get_fit_func(fit_type)
        #initialize all best fit params to nan, then replace if fit converges
        self.model_bf_params = np.ones((self.working_img.shape[1],lf.get_num_params(fit_type)))*np.nan
        self.failed_fit_cols = []

        for i in cols:
            if i % 250 == 0:
                print('starting {}'.format(i))
            #check to make sure we're not in bad region
            if any([(i >= low) and (i <= high) for low,high in self.avoid]):
                continue
            try:
                my_fit,chi = lf.fit_indiv_col(self.working_img[:,i],fit_type=fit_type)
                self.chi_sq[i] = chi
                self.model_bf_params[i] = my_fit
                fit = self.fit_func(xgrid,*my_fit)
                self.model_img[:,i] = self.fit_func(x,*my_fit)

                ##optimal counts: (approx)
                # -1 to lower extraction bound b/c of python 0 indexing
                low = np.argmax(self.working_img[:,i]) - (extraction_radius - 1)
                high = np.argmax(self.working_img[:,i]) + (extraction_radius)
                #fit_func w/o offset should be PSF
                my_fit[-1] = 0. #hack to disable offsets
                norm = np.trapz(self.fit_func(xgrid,*my_fit),x=xgrid)
                denom = np.sum((self.fit_func(x[low:high],*my_fit)/norm)**2.).astype(float)
                num = np.sum(self.fit_func(x[low:high],*my_fit)/norm*self.working_img[low:high,i])
                self.optimal_counts[i] = num/denom
                self.sum_counts[i] = np.sum(self.working_img[low:high,i])
            #runtime error should be failure to converge in fitting
            except RuntimeError as e:
                self.optimal_counts[i] = np.nan
                self.sum_counts[i] = np.nan
                self.chi_sq[i] = np.nan
                self.failed_fit_cols.append(i)
                print(e)
        self.failed_fit_cols = np.asarray(self.failed_fit_cols)
        self.optimal_counts[self.optimal_counts==0.] = np.nan

    def _get_indicies_in_wavelength_region(self,region):
        #Given a region (=[[lower,upper],[lower,upper]...]), find the indicies (pixel numbers) corresponding to the wavelength values in that region
        indicies = []
        for l,u in region:
            r = np.where(np.logical_and(self.wavelengths>l,self.wavelengths<u))[0]
            for i in r:
                indicies.append(i)
        return np.asarray(indicies)

    def generate_continuum_spectrum(self,spline_x_points=None,spline_smoothing_factor=5e8):
        """
        Create a continuum spectrum using points defined in the utils module (or user provided x points, which must be defined as pixel numbers corresponding to self.optimal_counts).
        """
        if spline_x_points is None:
            spline_x_points = np.array(utils.get_continuum_points())
        spline_y_values = self.optimal_counts[spline_x_points]

        not_nan = ~np.isnan(spline_y_values)
        self.continuum_spline = self._fit_spline(\
                spline_x_points[not_nan],spline_y_values[not_nan],spline_smoothing_factor)
        self.continuum_x = np.arange(np.min(spline_x_points),np.max(spline_x_points)+1)
        self.continuum_spectrum = self.continuum_spline(self.continuum_x)

    def _fit_spline(self,spline_x_points,spline_y_values,spline_smoothing_factor):
        #spline breaks if nans are present
        if any(np.isnan(spline_x_points)) or any(np.isnan(spline_y_values)):
            raise ValueError('spectrum.Spectrum._fit_spline: Found NaN in input data.')
        spline = UnivariateSpline(spline_x_points,spline_y_values,s=spline_smoothing_factor)
        return spline

    def generate_wavelength_solution(self,order=3):
        """
        Instantiates a polynomial that provides the pixel-wavelength map.  The polynomial is available from Spectrum.pix2wave(), and the coefficients are available in Spectrum.wavelength_fit_coeffs.
        Inputs:
            Order (int) :: Order of the polynomial to fit
        Returns:
            None
        """
        regions = utils.get_line_regions()
        self.line_centers = np.zeros(len(regions))

        norm_spec = self.optimal_counts[self.continuum_x] / self.continuum_spectrum
        for i,region in enumerate(regions):
            self.line_centers[i] = lf.find_line_center(region,self.continuum_x,norm_spec)
        wavelengths = utils.get_line_wavelengths()
        c = ~np.isnan(wavelengths)
        self.wavelength_fit_coeffs = np.polyfit(self.line_centers[c],wavelengths[c],deg=order)
        self.pix2wave = np.poly1d(self.wavelength_fit_coeffs)
        self.wavelengths = self.pix2wave(self.pixels)

    def _get_local_continuum(self,continuum_regions,order=1):
        i = self._get_indicies_in_wavelength_region(continuum_regions)
        x = self.wavelengths[i]
        y = self.optimal_counts[i]
        w = np.sqrt(y)
        line_region = np.arange(np.min(i),np.max(i))
        continuum_fit = np.polyfit(x,y,deg=order,w=w)
        continuum_p = np.poly1d(continuum_fit)
        continuum_wavelengths = self.wavelengths[line_region]
        continuum_flux = continuum_p(continuum_wavelengths)
        return continuum_wavelengths,continuum_flux

    def calc_Halpha_EW(self):
        continuum_regions = utils.get_Halpha_continuum_regions()
        i = self._get_indicies_in_wavelength_region(continuum_regions)
        Ha_indicies = np.arange(np.min(i),np.max(i))
        self.Ha_region,self.Ha_continuum = self._get_local_continuum(continuum_regions)
        Ha_EW = self._calc_EW(self.Ha_region,self.optimal_counts[Ha_indicies],self.Ha_continuum)
        return Ha_EW

    def calc_Hbeta_EW(self):
        continuum_regions = utils.get_Hbeta_continuum_regions()
        i = self._get_indicies_in_wavelength_region(continuum_regions)
        Hb_indicies = np.arange(np.min(i),np.max(i))
        self.Hb_region,self.Hb_continuum = self._get_local_continuum(continuum_regions)
        Hb_EW = self._calc_EW(self.Hb_region,self.optimal_counts[Hb_indicies],self.Hb_continuum)
        return Hb_EW


    def _calc_EW(self,wavelengths,flux,continuum):
        """
        Generic function to calculate (using SpecUtils) the Equivalent width of a line
        requires: Wavelengths (assumed to be in nm)
                  Flux (assumed to be in photons)
                  Continuum (assumed to be in photons)
        The equivalent width will be measured on the normalized spectrum (flux/continuum)
        output: Equivalent_Width (Quantity)
        """
        norm_spec = (flux / continuum)*u.photon
        norm_spec_wave = wavelengths*u.nanometer
        not_nan = ~np.isnan(norm_spec)
        spec = Spectrum1D(spectral_axis=norm_spec_wave[not_nan][::-1],flux=norm_spec[not_nan][::-1])
        spec_region = SpectralRegion(np.min(wavelengths)*u.nm,np.max(wavelengths)*u.nm)
        return equivalent_width(spec,regions=spec_region)

    def get_save_dict(self):
        save_dict = {}
        save_dict['filename'] = self.filename
        save_dict['obs_time'] = utils.get_obs_time(self.fits_file)
        save_dict['parallactic_angle'] = utils.calc_parallactic_angle(self.fits_file[0].header)
        save_dict['airmass'] = np.float(self.fits_file[0].header['AIRMASS'])
        save_dict['instrument_alignment_angle'] = np.float(self.fits_file[0].header['IAA'])
        save_dict['position_angle'] = np.float(self.fits_file[0].header['PA'])
        save_dict['optimal_counts'] = self.optimal_counts
        save_dict['wavelength'] = self.wavelengths
        save_dict['wlfit_line_centers'] = self.line_centers
        save_dict['wavelength_fit_coeffs'] = self.wavelength_fit_coeffs
        save_dict['working_img'] = self.working_img
        save_dict['fit_func'] = self.fit_func_name
        save_dict['model_bf_params'] = self.model_bf_params
        save_dict['fwhm'] = 2.*self.model_bf_params[:,2]*np.sqrt(2.**(1./self.model_bf_params[:,3])-1)
        save_dict['Halpha_EW'] = self.calc_Halpha_EW()
        save_dict['Hbeta_EW'] = self.calc_Hbeta_EW()
        save_dict['sky_array'] = self.sky_arr
        return save_dict
