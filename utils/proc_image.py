import numpy as np
import astropy.io.fits as pft
import os

def proc_gmoss_image(filename,verbose=False,apply_master_bias=True,master_bias_path='/home/mondrik/Gemini/analysis/biases/',apply_gain_correction=True):
    #assebles a GMOS-S fits file into a science array by subtracting overscan and possibly a master bias, applying gain, and aranging amplifiers (w/ chip gaps) into an array).
    #also returns an estimate of the bias standard deviation as estimated from the overscan region
    gain_file = np.loadtxt('/home/mondrik/Gemini/analysis/gmosamps_gains.txt')
    N_amps = 12

    fits_file = pft.open(filename)
    n_rows = fits_file[1].data.shape[0]
    science_array = np.empty((n_rows,0))
    gain_array = np.empty((n_rows,0))
    #chip gaps are 61 pixels (unbinned in x dir)
    chip_gap = np.zeros((n_rows,61),dtype=np.float)

    for i in range(N_amps):
        bias_sec = fits_file[i+1].header['BIASSEC']
        #make list with [col_start,col_end,row_start,row_end]
        bias_sec = bias_sec.strip('[]').replace(',',':').split(':')
        bias_col_start = np.int(bias_sec[0]) - 1 #indexing btwn IRAF & Python is diff
        bias_col_end = np.int(bias_sec[1]) #don't need to adjust upper though

        #extract bias vector and clone it into a 512x512 array
        temp = fits_file[i+1].data[:,bias_col_start:bias_col_end]
        bias_vec = np.median(temp,axis=1) #median combined estimate of bias from overscan
        bias_array = np.repeat(bias_vec[:,np.newaxis],512,axis=1)

        #extract 512x512 science image and put into temp array
        #then concatenate into science array
        if apply_gain_correction:
            gain = gain_file[i]
        else:
            gain = 1.
        sci_sec = fits_file[i+1].header['DATASEC']
        sci_sec = sci_sec.strip('[]').replace(',',':').split(':')
        sci_col_start = np.int(sci_sec[0]) - 1 #indexing is diff
        sci_col_end = np.int(sci_sec[1])
        sci_temp = (fits_file[i+1].data[:,sci_col_start:sci_col_end] - bias_array)
        gain_temp = np.ones_like(sci_temp)*gain
        science_array = np.hstack((science_array,sci_temp))
        gain_array = np.hstack((gain_array,gain_temp))
        if i in [3,7]:
        #put in chip gap at end of the CCD
            science_array = np.hstack((science_array,chip_gap))
            gain_array = np.hstack((gain_array,chip_gap))

        if verbose:
            print('AMPLIFIER {amp:d} BIAS: {bias:7.1f} BIAS_STD: {biasstd:5.2f}'.format(amp=i,bias=np.median(bias_vec),biasstd=np.std(bias_vec)))

    if apply_master_bias:
        img_name = os.path.split(filename)[-1]
        master_name = os.path.join(master_bias_path,'master_bias_'+img_name)
        mb_hdu = pft.open(master_name)
        science_array = science_array - mb_hdu[0].data

    #Need to multiply by gain array to actually correct data (gain array will be all 1's (0 in chip gap) if gain correction is set to False
    science_array = science_array * gain_array

    return fits_file,science_array
