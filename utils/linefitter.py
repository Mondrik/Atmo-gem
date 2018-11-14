import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pft
import utils
import emcee
from scipy.optimize import curve_fit
from astropy.modeling.models import Voigt1D, Moffat1D
from astropy.convolution import Gaussian1DKernel,convolve

def voigt(x,x0,a_l,fwhm_l,fwhm_g,offset):
    v = Voigt1D(x_0=x0,amplitude_L=a_l,fwhm_L=fwhm_l,fwhm_G=fwhm_g)
    return v(x) + offset

def gaussian(x,A,mu,sigma,offset):
    return A*np.exp(-0.5*((x-mu)/sigma)**2.) + offset

def moffat(x,x0,A,gamma,alpha,offset):
    m = Moffat1D(x_0=x0,amplitude=A,gamma=gamma,alpha=alpha)
    return m(x) + offset

def get_num_params(fit_type):
    if fit_type=='voigt':
        return 5
    elif fit_type=='moffat':
        return 5
    elif fit_type=='gaussian':
        return 4

def get_fit_func(fit_type):
    if fit_type=='voigt':
        return voigt
    if fit_type=='moffat':
        return moffat
    if fit_type=='gaussian':
        return gaussian
    else:
        print('Fit type {} not recognized...'.format(fit_type))

def fit_indiv_col(data,x=None,p0=None,sigma=None,fit_type='voigt'):
    if x is None:
        x = np.arange(len(data))
    if sigma is None:
        sigma = np.sqrt(np.abs(data))+10
    if fit_type == 'gaussian':
        func = gaussian
        if p0 is None:
            mu0 = x[np.argmax(data)]
            A0 = np.max(data)
            sigma0 = 4.
            offset0 = np.median(data)
            p0 = [A0,mu0,sigma0,offset0]
    if fit_type == 'voigt':
        func = voigt
        if p0 is None:
            x0 = x[np.argmax(data)]
            a_l = np.max(data)
            fwhm_l = 2.
            fwhm_g = 2.
            offset = np.median(data)
            p0 = [x0,a_l,fwhm_l,fwhm_g,offset]
    if fit_type == 'moffat':
        func = moffat
        if p0 is None:
            x0 = x[np.argmax(data)]
            A = np.max(data)
            gamma = 4.
            alpha = 1.
            offset = np.median(data)
            p0 = [x0,A,gamma,alpha,offset]
    popt,pcov = curve_fit(func,x,data,p0=p0,sigma=sigma)
    chisq = np.sum(((data-func(data,*popt))/sigma)**2.)
    return popt,chisq

def find_line_center(x_region,x,flux,convolve_flux=True):
    """
        Given a n x_region [x_low,x_high], find the center of the largest absorption line
        in that region
    """
    low,high = x_region
    i = np.where((x>low) & (x<high))[0]
    #fit quadratic to small region around center and find center
    xx = x[i]
    if convolve_flux:
        g = Gaussian1DKernel(stddev=2)
        yy = convolve(flux,g)[i]
    else:
        yy = flux[i]
    xmin = np.argmin(yy)

    fit = np.polyfit(xx[xmin-5:xmin+6],yy[xmin-5:xmin+6],deg=2)
    return -fit[1]/(2.*fit[0])
