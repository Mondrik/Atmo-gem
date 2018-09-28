import numpy as np
from utils import linefitter as lf
from utils import utils
from utils import proc_image as pi
import astroscrappy as scrap
import astropy.io.fits as pft
from scipy.interpolate import UnivariateSpline

class Spectrum():
    def __init__(self,filename,remove_cosmics=True,remove_sky=True):
        #get bias-subtracted and mosaic'd image:
        #TODO: MASTER BIAS RESIDUAL
        self.fits_file,self.data = pi.proc_gmoss_image(filename)
        self.avoid = utils.get_avoidance_regions()
        if remove_cosmics:
            self.mask,self.working_img = self.remove_cosmics()
        else:
            self.mask = np.ones_like(self.data,dtype=bool)
            self.working_img = self.data
        if remove_sky:
            self.sky_arr = self.estimate_sky()
            self.working_img = self.working_img - np.mean(self.sky_arr,axis=0)

    def remove_cosmics(self):
        mask,clean_img = scrap.detect_cosmics(self.data,sigclip=8,objlim=2.5, \
                fsmode='convolve',psfmodel='gauss',psffwhm=5.0,verbose=True,niter=8)
        return mask,clean_img

    def estimate_sky(self,sky_start=30,region_size=30):
        sky_arr = np.zeros((region_size*2,self.working_img.shape[1]))
        #assume sky starts sky_start=10pix away from spectrum peak and sample region_size=30pix
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

    def generate_profile_img(self):
        #generate P_xlambda following Horne 1986
        col_sum = np.sum(self.working_img,axis=0)
        self.prof_img = self.working_img / col_sum

    def extract(self,fit_type='voigt',extraction_bounds=30,apply_ccd1_corr=True):
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
        cols = np.arange(self.working_img.shape[1])
        x = np.arange(self.working_img.shape[0])
        xgrid = np.linspace(np.min(x),np.max(x),10000)
        self.model_img = np.zeros_like(self.working_img)
        self.fit_func = lf.get_fit_func(fit_type)
        #initialize all best fit params to nan, then replace if fit converges
        self.model_bf_params = np.ones((self.working_img.shape[1],lf.get_num_params(fit_type)))*np.nan

        for i in cols:
            if i % 250 == 0:
                print('starting {}'.format(i))
            #check to make sure we're not in bad regions
            if any([(i >= low) and (i <= high) for low,high in self.avoid]):
                continue
            try:
                my_fit = lf.fit_indiv_col(self.working_img[:,i],fit_type=fit_type)
                self.model_bf_params[i] = my_fit
                fit = self.fit_func(xgrid,*my_fit)
                self.model_img[:,i] = self.fit_func(x,*my_fit)

                ##optimal counts: (approx)
                # -1 to extraction bounds b/c of python 0 indexing
                low = np.argmax(self.working_img[:,i]) - (extraction_bounds - 1)
                high = np.argmax(self.working_img[:,i]) + (extraction_bounds - 1)
                #fit_func w/o offset should be LSF
                my_fit[-1] = 0. #hack to disable offsets
                norm = np.trapz(self.fit_func(xgrid,*my_fit),x=xgrid)
                denom = np.sum((self.fit_func(x[low:high],*my_fit)/norm)**2.).astype(float)
                num = np.sum(self.fit_func(x[low:high],*my_fit)/norm*self.working_img[low:high,i])
                self.optimal_counts[i] = num/denom
            except RuntimeError as e:
                self.optimal_counts[i] = np.nan
                print(e)
        self.optimal_counts[self.optimal_counts==0.] = np.nan
        if apply_ccd1_corr:
            self.optimal_counts[cols<2100] = self.optimal_counts[cols<2100] * 1.2123

    def generate_continuum_spectrum(self,spline_x_points=None):
        if spline_x_points is None:
            spline_x_points = utils.get_continuum_points()
        self.continuum_spline = UnivariateSpline(spline_x_points,self.optimal_counts[spline_x_points])
        self.continuum_x = np.arange(np.min(spline_x_points),np.max(spline_x_points)+1)
        self.continuum_spectrum = self.continuum_spline(self.continuum_x)
