import numpy as np
import os
from utils import linefitter as lf
import matplotlib.pyplot as plt
import astropy.io.fits as pft
import glob
import astropy.units as u

#Collection of useful data + functions that
#help with GMOS reductions

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

def get_all_sci_files(data_path='/home/mondrik/Gemini/raw/',mask_name=None):
    file_list = glob.glob(os.path.join(data_path,'*.fits'))
    sci_list = []
    for f in file_list:
        file_name = os.path.join(data_path,f)
        d = pft.open(file_name)
        if mask_name is not None:
            if d[0].header['MASKNAME']!=mask_name:
                continue
        if d[0].header['OBSCLASS'] == 'science':
            sci_list.append(file_name)
    return sci_list

def get_avoidance_regions():
    red_ccd_start = [0,30]
    blue_ccd_end = [6230,6265]
    chip_gap1 = [2030,2130]
    chip_gap2 = [4140,4230]
    bad_col1 = [1475,1488]
    bad_col2 = [2344,2355]
    bad_col3 = [5645,5652]
    bad_col4 = [3917,3924]
    return [red_ccd_start,blue_ccd_end,chip_gap1,chip_gap2,bad_col1,bad_col2,bad_col3,bad_col4]

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

def get_line_wavelengths():
    return np.asarray([656.45, 486.14, 434.04, 410.17, 397.01, np.nan, 388.9])

def calc_parallactic_angle(hour_angle,obj_dec,obs_lat=-30.*u.degree):
    #following eqn 10 from Fillipenko 1982
    h = hour_angle.to(u.radian)
    phi = obs_lat.to(u.radian)
    obj_dec = obj_dec.to(u.radian)
    temp1 = np.sin(h)*np.cos(phi)
    temp2 = np.sqrt(1. - (np.sin(phi)*np.sin(obj_dec) + np.cos(phi)*np.cos(obj_dec)*np.cos(h))**2.)
    return np.arcsin(temp1/temp2)
