import numpy as np
from utils import linefitter as lf
from utils import utils
from utils import proc_image as pi
import astroscrappy as scrap
from specutils import Spectrum1D,SpectralRegion
from specutils.analysis import equivalent_width
import astropy.io.fits as pft
import astropy.units as u
from scipy.interpolate import UnivariateSpline
import scipy.stats

class Spectrum():
    def __init__(self,filename,remove_cosmics=True,remove_sky=True,remove_avoidance=True,bootstrap=False,correct_working_img_qe=True):
        #get bias-subtracted and mosaiced image:
        #TODO: MASTER BIAS RESIDUAL
        self.fits_file,self.data = pi.proc_gmoss_image(filename)
        if bootstrap:
            self.data = scipy.stats.poisson.rvs(np.abs(self.data))
        self.avoid = utils.get_avoidance_regions()
        if remove_cosmics:
            self.mask,self.working_img = self.remove_cosmics()
        else:
            self.mask = np.ones_like(self.data,dtype=bool)
            self.working_img = self.data
        if remove_avoidance:
            self.remove_avoidance_regions()
        if remove_sky:
            self.sky_arr = self.estimate_sky()
            self.working_img = self.working_img - np.median(self.sky_arr,axis=0)
        if correct_working_img_qe:
            self.working_img[:,:2100] *= 1.2123
            self.qe_correct = True
        else:
            self.qe_correct = False

    def remove_cosmics(self):
        mask,clean_img = scrap.detect_cosmics(self.data,sigclip=8,objlim=2.5, \
                fsmode='convolve',psfmodel='gauss',psffwhm=5.0,verbose=True,niter=8)
        return mask,clean_img

    def remove_avoidance_regions(self):
        for low,high in self.avoid:
            self.working_img[:,low:high] = np.nan
        return None

    def estimate_sky(self,sky_start=30,region_size=30):
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

    def generate_profile_img(self,profile_half_width=30,order=10):
        #I THINK THIS IS GOING TO HAVE TO BE DONE ON A CCD-BY-CCD BASIS
        #(SEE PLOTS OF PROF_IMG)
        #width of the profile fit by the algorithm is phw*2+1, since it extends one half-width
        #in each direction from the center (and includes the center pixel)
        #generate P_xlambda following Horne 1986
        col_sum = np.sum(self.working_img[200:300,:],axis=0)
        self.prof_img = self.working_img / col_sum
        self.prof_img[self.profile_img<0] = 0.
        self.fit_array = np.ones((profile_half_width*2+1,order+1))*np.nan
        #best guess for center of spectrum is where nanmedian is highest...
        center_spec = np.argmax(np.nanmedian(self.working_img,axis=1))
        pixels = np.arange(0,self.working_img.shape[1],1)
        self.profile_region = np.arange(center_spec-profile_half_width,\
                center_spec+profile_half_width+1)
        for i in self.profile_region:
            not_nan = ~np.isnan(self.profile_img[i,:])
            fit = np.polyfit(pixels[not_nan],self.profile_img[i,:][not_nan],deg=order)
            self.fit_array[i-np.min(self.profile_region),:] = fit


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
        if not self.qe_correct:
            self.optimal_counts[cols<2100] = self.optimal_counts[cols<2100] * 1.2123

    def generate_continuum_spectrum(self,spline_x_points=None):
        """
        Create a continuum spectrum using points defined in the utils module (or user provided x points, which must be defined as pixel numbers corresponding to self.optimal_counts).
        """
        if spline_x_points is None:
            spline_x_points = np.array(utils.get_continuum_points())
        #spline breaks if nans are present
        spline_y_points = self.optimal_counts[spline_x_points]
        not_nan = ~np.isnan(spline_y_points)
        self.continuum_spline = UnivariateSpline(spline_x_points[not_nan],spline_y_points[not_nan])
        self.continuum_x = np.arange(np.min(spline_x_points),np.max(spline_x_points)+1)
        self.continuum_spectrum = self.continuum_spline(self.continuum_x)

    def generate_wavelength_solution(self,order=3):
        """
        Instantiates a polynomial that provides the pixel-wavelength map
        """
        regions = utils.get_line_regions()
        self.line_centers = np.zeros(len(regions))

        norm_spec = self.optimal_counts[self.continuum_x] / self.continuum_spectrum
        for i,region in enumerate(regions):
            self.line_centers[i] = lf.find_line_center(region,self.continuum_x,norm_spec)
        wavelengths = utils.get_line_wavelengths()
        c = ~np.isnan(wavelengths)
        fit = np.polyfit(self.line_centers[c],wavelengths[c],deg=order)
        self.p = np.poly1d(fit)

    def calc_equiv_width(self,region):
        norm_spec = (self.optimal_counts[self.continuum_x] / self.continuum_spectrum)*u.photon
        norm_spec_wave = self.p(self.continuum_x)*u.nanometer
        not_nan = ~np.isnan(norm_spec)
        spec = Spectrum1D(spectral_axis=norm_spec_wave[not_nan][::-1],flux=norm_spec[not_nan][::-1])
        spec_region = SpectralRegion(region[0]*u.nm,region[1]*u.nm)
        return equivalent_width(spec,regions=spec_region)
