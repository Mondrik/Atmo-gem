import numpy as np
import os
from utils import linefitter as lf
import matplotlib.pyplot as plt
import astropy.io.fits as pft

def get_test_sci_files(data_path='/home/mondrik/Gemini/chunks/',raw=True):
    good_nights = ['2017-01-08', \
                   '2016-09-04']
    paths = [os.path.join(data_path,n) for n in good_nights]
    sci_files = []
    for p in paths:
        sci_list = np.loadtxt(os.path.join(p,'sciFiles.txt'),dtype=np.str)
        if len(sci_list) == 0:
            raise ValueError('No science files found for {}'.format(p))
        for f in sci_list:
            if raw:
                sci_path = os.path.join(p,'raw',f)
            else:
                sci_path = os.path.join(p,'gs'+f)
            sci_files.append(sci_path)
    return sci_files

def get_avoidance_regions():
    red_ccd_start = [0,30]
    blue_ccd_end = [6230,6265]
    chip_gap1 = [2030,2130]
    chip_gap2 = [4140,4230]
    bad_col1 = [1475,1488]
    bad_col2 = [2344,2355]
    bad_col3 = [5645,5652]
    return [red_ccd_start,blue_ccd_end,chip_gap1,chip_gap2,bad_col1,bad_col2,bad_col3]

def get_continuum_points():
    points = [110, 709, 1082, 2581, 4596, 5056, 5392, 5770, 5928, 6152]
    return points

def get_line_regions():
    Halpha = [330,450] #656.45nm
    Hbeta = [3650,3800] #486.14nm
    Hgamma = [4650,4850] #434.04nm
    Hdelta = [5220,5320] #410.17nm
    Heps = [5500,5580] #397.01nm
    unknown1 = [5588,5630]
    Hzeta = [5660,5740] #388.90nm
    return [Halpha,Hbeta,Hgamma,Hdelta,Heps,unknown1,Hzeta]


def examine_linefitter_col_results(examine_raw=True,fit_type='voigt'):
    d = pft.open(get_test_sci_files()[0])
    data = d[2].data
    x = np.arange(data.shape[0])
    fit_array = np.zeros_like(data)
    fit_func = lf.get_fit_func(fit_type)
#    if fit_type=='voigt':
#        fit_func = lf.voigt
#    if fit_type=='moffat':
#        fit_func = lf.moffat
#    if fit_type=='gaussian':
#        fit_func = lf.gaussian

    for col in data[:,::500].T:
        #plt.plot(x,col)
        myfit = lf.fit_indiv_col(col,fit_type=fit_type)
        xv = np.linspace(np.min(x),np.max(x),10000)
        v = fit_func(xv,*myfit)
        if examine_raw:
            plt.plot(col,'-b')
            plt.plot(xv,v,'--k')
            plt.ylabel('Counts')
        else:
            plt.plot(x,np.abs(col-fit_func(x,*myfit))/col)
            plt.axhline(0.1,color='r')
            plt.axhline(0.01,color='r')
            plt.ylim(0,0.5)
            plt.ylabel('Data-Model / Data')
        plt.xlabel('Pixel')
        plt.show()


def examine_extract_results(data,fit_type='voigt'):
    counts,model_img = spectrum.extract(data,fit_type='voigt')
    diff = data - model_img
    qlow = np.percentile(diff.flatten(),10)
    qhigh = np.percentile(diff.flatten(),90)

    plt.imshow(diff,origin='lower',vmin=qlow,vmax=qhigh)
    plt.show()
