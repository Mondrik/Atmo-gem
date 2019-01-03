import numpy as np
import astropy.io.fits as pft

def proc_gmoss_image(filename,verbose=False):
    gain_file = np.loadtxt('/home/mondrik/Gemini/analysis/gmosamps_gains.txt')
    N_amps = 12

    fits_file = pft.open(filename)
    science_array = np.empty((512,0))
    #chip gaps are 62 pixels (unbinned in x dir)
    chip_gap = np.zeros((512,61),dtype=np.float)

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
        gain = gain_file[i]
        sci_sec = fits_file[i+1].header['DATASEC']
        sci_sec = sci_sec.strip('[]').replace(',',':').split(':')
        sci_col_start = np.int(sci_sec[0]) - 1 #indexing is diff
        sci_col_end = np.int(sci_sec[1])
        sci_temp = (fits_file[i+1].data[:,sci_col_start:sci_col_end] - bias_array)*gain
        science_array = np.hstack((science_array,sci_temp))
        if i in [3,7]:
        #put in chip gap at end of the CCD
            science_array = np.hstack((science_array,chip_gap))

        if verbose:
            print('AMPLIFIER {amp:d} BIAS: {bias:7.1f}'.format(amp=i,bias=np.median(bias_vec)))

    return fits_file,science_array
