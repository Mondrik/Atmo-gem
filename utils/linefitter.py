import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pft
import utils
import emcee
from scipy.optimize import curve_fit
from astropy.modeling.models import Voigt1D, Moffat1D

def voigt(x,x0,a_l,fwhm_l,fwhm_g,offset):
    v = Voigt1D(x_0=x0,amplitude_L=a_l,fwhm_L=fwhm_l,fwhm_G=fwhm_g)
    return v(x) + offset

def gaussian(x,A,mu,sigma,offset):
    return A*np.exp(-0.5*((x-mu)/sigma)**2.) + offset

def moffat(x,x0,A,gamma,alpha,offset):
    m = Moffat1D(x_0=x0,amplitude=A,gamma=gamma,alpha=alpha)
    return m(x) + offset

def get_fit_func(fit_type):
    if fit_type=='voigt':
        return voigt
    if fit_type=='moffat':
        return moffat
    if fit_type=='gaussian':
        return gaussian
    else:
        print('Fit type {} not recognized...'.format(fit_type))

def fit_indiv_col(data,x=None,p0=None,fit_type='voigt'):
    if x is None:
        x = np.arange(len(data))
    if fit_type == 'gaussian':
        if p0 is None:
            mu0 = x[np.argmax(data)]
            A0 = np.max(data)
            sigma0 = 4.
            offset0 = np.median(data)
            p0 = [A0,mu0,sigma0,offset0]
        popt,pcov = curve_fit(gaussian,x,data,p0=p0,sigma=np.sqrt(np.abs(data))+10)
    if fit_type == 'voigt':
        if p0 is None:
            x0 = x[np.argmax(data)]
            a_l = np.max(data)
            fwhm_l = 2.
            fwhm_g = 2.
            offset = np.median(data)
            p0 = [x0,a_l,fwhm_l,fwhm_g,offset]
        popt,pcov = curve_fit(voigt,x,data,p0=p0,sigma=np.sqrt(np.abs(data))+10)
    if fit_type == 'moffat':
        if p0 is None:
            x0 = x[np.argmax(data)]
            A = np.max(data)
            gamma = 4.
            alpha = 1.
            offset = np.median(data)
            p0 = [x0,A,gamma,alpha,offset]
        popt,pcov = curve_fit(moffat,x,data,p0=p0,sigma=np.sqrt(np.abs(data))+10.)
    return popt

def fit_spectrum():
    pass
