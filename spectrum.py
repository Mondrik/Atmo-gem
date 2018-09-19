import numpy as np
from utils import linefitter as lf
import astroscrappy as scrap

def get_avoidance_regions():
    chip_gap1 = [2040,2120]
    chip_gap2 = [4150,4220]
    bad_col1 = [1475,1488]
    bad_col2 = [2344,2355]
    return [chip_gap1,chip_gap2,bad_col1,bad_col2]

def remove_cosmics(data):
    mask,clean_img = scrap.detect_cosmics(data,sigclip=8,objlim=2.5,fsmode='convolve',psfmodel='gaussy',psffwhm=5.0)
    return mask, clean_img

def extract(data,fit_type='voigt'):
    mask, clean_img = remove_cosmics(data)
    counts = np.zeros(data.shape[1])
    x = np.arange(data.shape[0])
    xgrid = np.linspace(np.min(x),np.max(x),10000)
    avoid = get_avoidance_regions()
    model_img = np.zeros_like(data)
    fit_func = lf.get_fit_func(fit_type)

    for i in range(len(counts)):
        print('starting {}'.format(i))
    #check to make sure we're not in bad regions
        if any([(i > low) and (i < high) for low,high in avoid]):
            continue
        try:
            my_fit = lf.fit_indiv_col(clean_img[:,i],fit_type=fit_type)
            fit = fit_func(xgrid,*my_fit)
            counts[i] = np.trapz(fit,x=xgrid)
            model_img[:,i] = fit_func(x,*my_fit)
        except RuntimeError as e:
            counts[i] = np.nan
            print(e)
    return counts, model_img
