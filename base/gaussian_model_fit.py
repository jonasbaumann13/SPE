# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2024 Jonas Baumann, Steffen Staeck, Christopher Schlesiger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED BY Jonas Baumann, Steffen Staeck, Christopher Schlesiger "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import numpy as np
from .spe_tools import contains_box


def pixel_integrated_2d_gaussian(volume, z_0, center_x, center_y, sigma_x, sigma_y, area_shape, pixel_x=1, pixel_y=1):
    """
    calculates the integral of a 2d gaussian in a mesh grid (e.g. pixels in ccd).

    Parameters
    ----------
    volume: float
            total volume of the 2d gaussian
    z_0:  float
             offset value of the data
    center_x: float
            center of gaussian in x direction
    center_y: float
            center of gaussian in y direction
    sigma_x: float
             sigma width of gaussian in x direction
    sigma_y: float
             sigma width of gaussian in y direction
    area_shape: tuple
             size of the xy area used for calculation
    pixel_x: float
             relative pixel size in x direction (in units of pixel sizes)
    pixel_y: float
             relative pixel size in y direction (in units of pixel sizes)


    Returns
    -------
    [X,Y,Z]: array of numpy.2darrays
           X meshgrid coordinates in x direction
           Y meshgrid coordinates in y direction
           Z integrated values for every pixel

    Author
    ------
    Jonas Baumann
    """
    from numpy import sqrt, meshgrid
    from scipy.special import erf

    #   just to simplify formula
    quotienth = sqrt(2) * sigma_x
    quotientv = sqrt(2) * sigma_y
    #   determine indices of subimage (convolution of whole image is too slow)
    x = range(int(area_shape[0]))
    y = range(int(area_shape[1]))
    pixelarrayh, pixelarrayv = meshgrid(x, y)

    #   create subimage with distributed counts (convolution of 2D gaussian distribution with rectangular function of pixel size; this equals the part of the whole volume being in each pixel)
    image_temp = z_0 + volume * 0.25 * (
            (erf((pixel_x / 2. + pixelarrayh - center_x) / quotienth) + erf(
                (pixel_x / 2. - pixelarrayh + center_x) / quotienth)) *
            (erf((pixel_y / 2. + pixelarrayv - center_y) / quotientv) + erf(
                (pixel_y / 2. - pixelarrayv + center_y) / quotientv)))

    return pixelarrayh, pixelarrayv, image_temp


def fit_pixel_integrated_2d_gaussian(data, p0=None, fix=None, bounds=None, full_output=False, **kwargs):
    """
    Fits a pixel_integrated_2d_gaussian to a two-dimensional array using
    scipy.optimize.curve_fit. This is the shape assumed to be created by a
    charge cloud on a ccd chip.

    Parameters
    ----------
    data : 2d-array
        data array, where a Gaussian-shaped peak is expected
    p0 : list, optional
        starting values for the fit [Volume, offset, center_x, center_y,
        sigma_x, sigma_y]. Use None for any of the parameters, if it should be
        estimated by the fitting routine. If all starting parameters are to be
        estimated, use p0=None
    fix : list of bools, optional
        list of form [bool,bool,bool,bool,bool,bool] to decide via True and False which
        of the starting parameters of p0 [Volume, offset, center_x, center_y,
        sigma_x, sigma_y] should be kept fixed. If none should be fixed, set fix=None.
    bounds : 2-tuple of array_like, optional
        (lower, upper), with lower and upper being the respective bounds for the
        fit. Use np.inf with an appropriate sign to disable bounds on all or some
        parameters. The order is [Volume, offset, center_x, center_y, sigma_x, sigma_y]
    full_output : boolean, optional
        if full_output is set to 'true', a second output is returned with an
        dictionary containing further information about the fit with the
        key entries as in 'Returns'.
    kwargs : Keyword arguments
        Keyword arguments passed to scipy.optimize.curve_fit

    Returns
    -----
    returns the parameters of the 2d gaussian function (which is the basis of
    the integrals in 2d) in an array:
    [Amplitude, offset, center_x, center_y, sigma_x, sigma_y]

    returned dictionary keys if full_output=True:

    'cov' :
        covariance matrix of the fit, errors for the parameters can be accessed
        by np.sqrt(cov[i,i]) if the fit model is accurate.
    'fit' :
        two-dimensional array of fit values
    'res' :
        two-dimensional residual array, where in every entry R = 1-data/fit is
        calculated

    Author
    ------
    Jonas Baumann
    """

    from scipy.optimize import curve_fit
    from numpy import sqrt, indices, ravel, asarray, inf

    def moments(data):
        epsilon = 1e-14
        total = data.sum()
        X, Y = indices(data.shape)
        x = (X * data).sum() / (total+epsilon)
        y = (Y * data).sum() / (total+epsilon)
        col = data[:, int(y)]
        width_x = sqrt(abs((range(col.size) - y) ** 2 * col).sum() / (col.sum()+epsilon))
        row = data[int(x), :]
        width_y = sqrt(abs((range(row.size) - x) ** 2 * row).sum() / (row.sum()+epsilon))
        z_0 = (data[0][0])
        height = (data.max() - z_0)
        volume = 2 * 3.14 * height * width_x * width_y  # volumen eines 2d gauss, grob, da nur startwert
        return volume, z_0, x, y, width_x, width_y

    def fit_pixel_integrated_2d_gaussian(area_shape, volume, z_0, center_x, center_y, width_x, width_y):
        ig = pixel_integrated_2d_gaussian(volume, z_0, center_x, center_y, width_x, width_y, area_shape)[2]
        return ravel(ig)

    data = asarray(data)
    p0_fit = moments(data)
    if p0:
        for i in range(len(p0)):
            if p0[i] == None:   p0[i] = p0_fit[i]
    else:
        p0 = p0_fit
        fix = None

    if not bounds:
        bounds = ([-inf] * len(p0), [inf] * len(p0))

    if fix:
        for i in range(len(fix)):
            if fix[i]:
                bounds[0][i] = p0[i] - 1e-12
                bounds[1][i] = p0[i] + 1e-12

    param, cov = curve_fit(fit_pixel_integrated_2d_gaussian, data.shape, ravel(data), p0=p0, bounds=bounds, **kwargs)
    if full_output:
        [volume, z_0, center_x, center_y, width_x, width_y] = param
        fit = pixel_integrated_2d_gaussian(volume, z_0, center_x, center_y, width_x, width_y, data.shape)[2]
        res = 1 - data / fit
        return param, {'cov': cov, 'fit': fit, 'res': res}
    else:
        return param
def _moments_gmf(data):
    x = np.shape(data)[0]/2.
    y = np.shape(data)[1]/2.
    width_x = np.shape(data)[0]/2.
    width_y = np.shape(data)[1]/2.
    z_0 = data.min()
    volume = np.sum(data-z_0)
    return volume, z_0, x, y, width_x, width_y


def get_photon_events(image, sig_fac_1, noise_rms, squ_a, p0=None, fix=None, bounds=None, full_output=False, **kwargs):
    """
    Evaluate single phton events on a 2d array (e.g. ccd, cmos) using the
    gaussian_model_fit method. Here, in a 3x3 pixel box around each significant
    pixel (intensity > sig_fac_1xnoise_rms) a fit of the charge distribution is
    performed. The function for the fit is from axp_tools.two_d.pixel_integrated_2d_gaussian
    by calling axp_tools.two_d.fit_pixel_integrated_2d_gaussian

    Parameters
    ----------
    image : 2d-array
        image to be evaluated
    sig_fac_1 : float
        first noise threshold factor. determines which pixels are (probably)
        used as starting pixels for the algorithm
    noise_rms : float or 2d-array
        expected standard deviation of the noise in the image, if 2d array is
        given, then the standard deviation is used pixelwise
    squ_a : int
        defining the edge length of the box of the subarray used for fitting in
        pixel sizes.
    p0 : list, optional
        starting values for the fit [Volume, offset, center_x, center_y, 
        sigma_x, sigma_y]. Use None for any of the parameters, if it should be 
        estimated by the fitting routine. If all starting parameters are to be 
        estimated, use p0=None        
    fix : list of bools, optional
        list of form [bool,bool,bool,bool,bool,bool] to decide via True and False which
        of the starting parameters of p0 [Volume, offset, center_x, center_y, 
        sigma_x, sigma_y] should be kept fixed. If none should be fixed, set fix=None.
    bounds : 2-tuple of array_like, optional
        (lower, upper), with lower and upper being the respective bounds for the
        fit. Use np.inf with an appropriate sign to disable bounds on all or some 
        parameters. The order is [Volume, offset, center_x, center_y, sigma_x, sigma_y]
    full_output : bool, optional
        if True, the output is a list [photon_events, residuum_image, fails].
        See 'Returns' for details.
    kwargs : Keyword arguments
        Keyword arguments passed to scipy.optimize.curve_fit


        
    Returns
    -------
    returns a list of photon_events with each entry containing                  [tmp_event_sum,tmp_event_n,px_list_int,px_list_pos]
    pos_x : float
        center of gravity position in x-direction round up to 1 decimal place.
        Unit is pixel sizes.
    pos_y : float
        center of gravity position in y-direction round up to 1 decimal place.
        Unit is pixel sizes.
    volume : float
        total event intensity (proportional to energy of photon)
    sig_x : float
        noise_rms width of the gaussian function in x-direction
    sig_y : float
        noise_rms width of the gaussian function in y-direction
    err_list : list
        list of square root of diagonal elements of estimated covariance matrix
    mean_sq_err : float
        mean squared error normalized to pixel number
    
    if full_output = True, then additionally the residuum image and the number
    of fails of the fit is returned. The residuum image shows the residuum of
    the fit in the fitted areas only. The remaining area is set to 0.

    Author
    ------
    Jonas Baumann
    """
    fails = 0
    photon_events = []
    image_copy = np.array(image)
    residuum_image = np.zeros(np.shape(image))
    squ_ah = int(squ_a/2.0)

    if not hasattr(noise_rms,"__len__"):    
        noise_rms_image = np.zeros(np.shape(image))
        noise_rms_image.fill(noise_rms)
    else:   
        noise_rms_image = np.array(noise_rms)
        noise_rms = np.mean(noise_rms_image)
        
    noise_rms_diff = image/np.array(noise_rms_image,dtype='float')              # to ensure no rounding
    significant_pixels = np.where(noise_rms_diff>=sig_fac_1)
    
    significant_pixels = np.where(image>=(sig_fac_1*noise_rms))
    significant_x_values= significant_pixels[0]
    significant_y_values= significant_pixels[1]
    
    for i in range(0,len(significant_x_values)):

        x = significant_x_values[i]
        y = significant_y_values[i]    
        if contains_box(x,y,np.shape(image)[0],np.shape(image)[1],squ_a+2):

            s = np.array(image_copy[x-squ_ah:x+squ_ah+1,y-squ_ah:y+squ_ah+1])

            if image[x][y] == np.max(s):
                
                ### setting starting values for the fit ###
                p0_fit = _moments_gmf(s)
                if p0:
                    for i in range(len(p0)):
                        if p0[i] == None:   p0[i] = p0_fit[i]
                else:
                    p0 = p0_fit
                    fix = None
                    
                if not bounds:
                    bounds = ([-np.inf]*len(p0),[np.inf]*len(p0))
                
                if fix:
                    for i in range(len(fix)):
                        if fix[i]:
                            bounds[0][i] = p0[i]-1e-12
                            bounds[1][i] = p0[i]+1e-12
                
                ### try the fit ###
                try:
                    param, dict_fit = fit_pixel_integrated_2d_gaussian(s, p0, fix, bounds, True, **kwargs)
                    volume = param[0]
                except:
                    volume = np.nan
                    fails += 1
                    
                if not np.isnan(volume):
                    pos_x, pos_y = x-squ_ah+param[2], y-squ_ah+param[3]
                    sig_x, sig_y = param[4], param[5]
                
                    err_list = np.sqrt([dict_fit['cov'][2][2],dict_fit['cov'][3][3],dict_fit['cov'][0][0],dict_fit['cov'][4][4],dict_fit['cov'][5][5]])      # center_x, center_y, volume, width_x, width_y
                    residuum_image[x-squ_ah:x+squ_ah+1,y-squ_ah:y+squ_ah+1] = dict_fit['res']               # set the nxn area in the residuum image to the residuum
                    image_copy[x-squ_ah:x+squ_ah+1,y-squ_ah:y+squ_ah+1] = dict_fit['fit']               # set the nxn area in the residuum image to the residuum
                    mean_sq_err = np.sum((s-dict_fit['fit'])**2)/sum(len(row) for row in s)
                    photon_events.append([round(pos_x,1),round(pos_y,1),volume,sig_x,sig_y,err_list,mean_sq_err])
                    

    if full_output:
        return [photon_events,image_copy,residuum_image, fails]

    else:
        return photon_events