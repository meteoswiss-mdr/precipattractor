#!/usr/bin/env python
'''
Module to perform various statistics on the data
(anisotropy estimation, spectral slope computation, WAR statistics, etc).

Documentation convention from https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

26.09.2016
Loris Foresti
'''
from __future__ import division
from __future__ import print_function

import sys
import time
import numpy as np
import math
import pywt

import matplotlib.pyplot as plt

from scipy import stats
import scipy.ndimage as ndimage
from skimage import measure

import pandas as pd
import statsmodels.formula.api as sm
from statsmodels.nonparametric.api import KernelReg

def compute_war(rainfield, rainThreshold, noData):
    idxRain = rainfield >= rainThreshold
    idxRadarDomain = rainfield > noData + 1
    
    if (len(idxRain) >= 0) and (len(idxRain) < sys.maxsize) and \
    (len(idxRadarDomain) >= 0) and (len(idxRadarDomain) < sys.maxsize) \
    and (np.sum(idxRain) <= np.sum(idxRadarDomain)) and (np.sum(idxRadarDomain) > 0):
        war = 100.0*np.sum(idxRain)/np.sum(idxRadarDomain)
    else:
        print("Problem in the computation of WAR. idxRain = ", idxRain, " and idxRadarDomain = ", idxRadarDomain, " are not valid values.")
        print("WAR set to -1")
        war = -1
    return war
    
def compute_beta(logScale, logPower):
    beta, intercept, r_beta, p_value, std_err = stats.linregress(logScale, logPower)
    return(beta, intercept, r_beta)
    
def compute_beta_weighted(logScale, logPower, weights):
        # normalize sum of weights to 1
        weights = weights/float(np.sum(weights))
          
        degree = 1
        coefficients = np.polynomial.polynomial.polyfit(logScale, logPower, degree, w=weights)
        intercept = coefficients[0]
        beta = coefficients[1]
        
        # Get coefficient of correlation (it should also be adapted to weight more the sparser sets of points...)
        yhat = coefficients[0] + coefficients[1]*logScale   # Prediction model
        #ybar = np.sum(logPower)/len(logPower)          # Unweighted average of the predictand
        ybar = np.sum(weights*logPower)/np.sum(weights) # Weighted average of the predictand
        ssreg = np.sum(weights*(yhat-ybar)**2)           # Regression sum of squares
        sstot = np.sum(weights*(logPower - ybar)**2)      # Total sum of squares
        r_betaSq = ssreg/sstot
        
        if beta >= 0:
            r_beta = np.sqrt(r_betaSq)
        else:
            r_beta = -np.sqrt(r_betaSq)
            
        return(beta, intercept, r_beta)

def compute_beta_sm(logScale, logPower, weights = None):
    x_list = logScale.tolist()
    y_list = logPower.tolist()
    
    ws = pd.DataFrame({
    'x': x_list,
    'y': y_list
    })

    # Compute weighted or unweighted OLS
    if weights is not None:
        weightsPD = pd.Series(weights)
        results = sm.wls('y ~ x', data=ws, weights=weightsPD).fit()
    else:
        results = sm.ols('y ~ x', data=ws).fit()
    
    # Get results
    r_betaSq = results.rsquared
    beta = results.params.x
    intercept = results.params.Intercept

    if beta >= 0:
        r_beta = np.sqrt(r_betaSq)
    else:
        r_beta = -np.sqrt(r_betaSq)
    return(beta, intercept, r_beta)
    
def GaussianKernel(v1, v2, sigma):
    return exp(-norm(v1-v2, 2)**2/(2.*sigma**2))
    
def compute_fft_anisotropy(psd2d, fftSizeSub = -1, percentileZero = -1, rotation = True, radius = -1, sigma = -1, verbose = 0):
    ''' 
    Function to compute the anisotropy from a 2d power spectrum or autocorrelation function.
    
    Parameters
    ----------
    psd2d : numpyarray(float)
        Input 2d array with the autocorrelation function or the power spectrum
    fftSizeSub : int
        Half-size of the sub-domain to extract to zoom in (maximum = psd2d.size[0]/2)
    percentileZero : int
        Percentile to use to shift the autocorrelation/spectrum to 0. Values below the percentile will be set to 0.
    rotation : bool
        Whether to rotate the spectrum (Fourier spectrum needs a 90 degrees rotation w.r.t autocorrelation)
    radius : int
        Radius in pixels from the center to mask the data (not needed if using the percentile criterion)
    sigma : float
        Bandwidth of the Gaussian kernel used to smooth the 2d spectrum (not needed for the autocorrelation function, already smooth)
    verbose: int
        Verbosity level to use (0: nothing is printed)
    
    Returns
    -------
    psd2dsub : numpyarray(float)
        Output 2d array with the autocorrelation/spectrum selected on the subdomain (and rotated if asked)
    eccentricity : float
        Eccentricity of the anisotropy (sqrt(1-eigval_max/eigval_min)) in range 0-1
    orientation : float
        Orientation of the anisotropy (degrees using trigonometrical convention, -90 degrees -> South, 90 degrees -> North, 0 degrees -> East)
    xbar : float
        X-coordinate of the center of the intertial axis of anisotropy (pixels)
    ybar : float
        Y-coordinate of the center of the intertial axis of anisotropy (pixels)
    eigvals : numpyarray(float)
        Eigenvalues obtained after decomposition of covariance matrix using selected values of spectrum/autocorrelation
    eigvecs : numpyarray(float)
        Eigenvectors obtained after decomposition of covariance matrix using selected values of spectrum/autocorrelation
    percZero : float
        Value of the autocorrelation/spectrum corresponding to the asked percentile (percentileZero)
    psd2dsubSmooth: numpyarray(float)
        Output 2d array with the smoothed autocorrelation/spectrum selected on the subdomain (and rotated if asked)
    '''
    
    # Get dimensions of the large and subdomain
    if fftSizeSub == -1:
        fftSizeSub = psd2d.shape[0]/2
    
    fftSize = psd2d.shape
    
    if ((fftSize[0] % 2) != 0) or ((fftSize[1] % 2) != 0):
        print("Error in compute_fft_anisotropy: please provide an even sized 2d FFT spectrum.")
        sys.exit(1)
    fftMiddleX = fftSize[1]/2
    fftMiddleY = fftSize[0]/2
    
    # Select subset of autocorrelation/spectrum
    psd2dsub = psd2d[fftMiddleY-fftSizeSub:fftMiddleY+fftSizeSub,fftMiddleX-fftSizeSub:fftMiddleX+fftSizeSub]

    ############### CIRCULAR MASK
    # Apply circular mask from the center as mask (not advised as it will often yield a circular anisotropy, in particular if the radisu is small)
    if radius != -1:
        # Create circular mask
        y,x = np.ogrid[-fftSizeSub:fftSizeSub, -fftSizeSub:fftSizeSub]
        mask = x**2+y**2 <= radius**2
        
        # Apply mask to 2d spectrum
        psd2dsub[~mask] = 0.0
    
    ############### ROTATION
    # Rotate FFT spectrum by 90 degrees
    if rotation:
        psd2dsub = np.rot90(psd2dsub)

    ############### SMOOTHING
    # Smooth spectrum field if too noisy (to help the anisotropy estimation)
    if sigma > 0:
        psd2dsubSmooth = ndimage.gaussian_filter(psd2dsub, sigma=sigma)
    else:
        psd2dsubSmooth = psd2dsub.copy() # just to give a return value...
    
    ############### SHIFT ACCORDING TO PERCENTILE
    # Compute conditional percentile on smoothed spectrum/autocorrelation
    minThresholdCondition = 0.01 # Threshold to compute to conditional percentile (only values greater than this)
    if percentileZero > 0:
        percZero = np.nanpercentile(psd2dsubSmooth[psd2dsubSmooth > minThresholdCondition], percentileZero)
    else:
        percZero = np.min(psd2dsubSmooth)
    
    if percZero == np.nan:
        percZero= 0.0
        
    # Shift spectrum/autocorrelation to start from 0 (zeros will be automatically neglected in the computation of covariance)
    psd2dsubSmoothShifted = psd2dsubSmooth - percZero
    psd2dsubSmoothShifted[psd2dsubSmoothShifted < 0] = 0.0
    
    ############### IMAGE SEGMENTATION
    # Image segmentation to remove high autocorrelations/spectrum values at far ranges/high frequencies
    psd2dsubSmoothShifted_bin = np.uint8(psd2dsubSmoothShifted > minThresholdCondition)
    
    # Compute image segmentation
    labelsImage = measure.label(psd2dsubSmoothShifted_bin, background = 0)
    
    # Get label of center of autocorrelation function (corr = 1.0)
    labelCenter = labelsImage[labelsImage.shape[0]/2,labelsImage.shape[1]/2]
    
    # Compute mask to keep only central polygon
    mask = (labelsImage == labelCenter).astype(int)
    
    nrNonZeroPixels = np.sum(mask)
    if verbose == 1:
        print("Nr. central pixels used for anisotropy estimation: ", nrNonZeroPixels)
    
    ############### COVARIANCE DECOMPOSITION
    # Find inertial axis and covariance matrix
    xbar, ybar, cov = _intertial_axis(psd2dsubSmoothShifted*mask)
    
    # Decompose covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    ############### ECCENTRICITY/ORIENTATION
    # Compute eccentricity and orientation of anisotropy
    idxMax = np.argmin(eigvals)
    #eccentricity = np.max(np.sqrt(eigvals))/np.min(np.sqrt(eigvals))
    eccentricity = math.sqrt(1-np.min(eigvals)/np.max(eigvals))
    orientation = np.degrees(math.atan(eigvecs[0,idxMax]/eigvecs[1,idxMax]))
        
    return psd2dsub, eccentricity, orientation, xbar, ybar, eigvals, eigvecs, percZero, psd2dsubSmooth

def compute_autocorrelation_fft(imageArray, FFTmod = 'NUMPY'):
    '''
    This function computes the autocorrelation of an image using the FFT.
    It exploits the Wiener-Khinchin theorem, which states that the Fourier transform of the auto-correlation function   
    is equal to the Fourier transform of the signal. Thus, the autocorrelation function can be obtained as the inverse transform of
    the power spectrum.
    It is very important to know that the auto-correlation function, as it is referred to as in the literature, is in fact the noncentred
    autocovariance. In order to obtain values of correlation between -1 and 1, one must center the signal by removing the mean before
    computing the FFT and then divide the obtained auto-correlation (after inverse transform) by the variance of the signal.
    '''
    
    tic = time.clock()
    
    # Compute field mean and variance
    field_mean = np.mean(imageArray)
    field_var = np.var(imageArray)
    field_dim = imageArray.shape
    
    # Compute FFT
    if FFTmod == 'NUMPY':
        fourier = np.fft.fft2(imageArray - field_mean) # Numpy implementation
    if FFTmod == 'FFTW':
        fourier = pyfftw.interfaces.numpy_fft.fft2(imageArray - field_mean) # FFTW implementation
        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()
    
    # Compute power spectrum
    powerSpectrum = np.abs(fourier)**2/(field_dim[0]*field_dim[1])
    
    # Compute inverse FFT of spectrum
    if FFTmod == 'NUMPY':
        autocovariance = np.fft.ifft2(powerSpectrum) # Numpy implementation
    if FFTmod == 'FFTW':
        autocovariance = pyfftw.interfaces.numpy_fft.ifft2(powerSpectrum) # FFTW implementation
        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()
    
    # Compute auto-correlation from auto-covariance
    autocorrelation = autocovariance.real/field_var
    
    # Shift autocorrelation function and spectrum to have 0 lag/frequency in the center
    autocorrelation_shifted = np.fft.fftshift(autocorrelation)
    powerSpectrum_shifted = np.fft.fftshift(powerSpectrum) # Add back mean to spectrum??
    
    toc = time.clock()
    #print("Elapsed time for ACF using FFT: ", toc-tic, " seconds.")
    return(autocorrelation_shifted, powerSpectrum_shifted)
    
def percentiles(array, percentiles):
    '''
    Function to compute a set of quantiles from an array
    '''
    nrPerc = len(percentiles)
    percentilesArray = []
    for p in range(0,nrPerc):
        perc = np.percentile(array,percentiles[p])
        percentilesArray.append(perc)
    percentilesArray = np.array(percentilesArray)
    return(percentilesArray)
    
def smooth_extrapolate_velocity_field(u, v, prvs, next, sigma):
    '''
    In development...
    '''
    nrRows = u.shape[0]
    nrCols = u.shape[1]
    nrGridPts = nrRows*nrCols
    
    mask = (prvs > 0) & (next > 0) 
    
    # Generate all grid coordinates
    idx = np.arange(0,nrRows)
    idxMat = np.tile(idx, [nrRows,1])
    idxMatMask = idxMat.copy()
    idxMatMask[mask != 1] = -999 
    
    idy = np.arange(0,nrCols)
    idyMat = np.tile(idy.T, [nrCols,1]).T
    idyMatMask = idyMat.copy()
    idyMatMask[mask != 1] = -999 
    
    allCoords = np.array([idxMat.ravel(),idyMat.ravel()]).T
    
    # Inputs
    trainingX = idxMatMask.ravel()
    trainingX = trainingX[trainingX == -999]
    trainingY = idyMatMask.ravel()
    trainingY = trainingY[trainingY == -999]
    
    # Outputs
    trainingU = u.ravel()
    trainingU = trainingU[trainingY == -999]
    trainingV = v.ravel()
    trainingV = trainingV[trainingV == -999]
    
    from scipy.interpolate import Rbf
    print(trainingX.shape)
    rbfi = Rbf(trainingX, trainingY, trainingU, epsilon = 10)
    uvec = rbfi(allCoords[:,0], allCoords[:,1])
    
    rbfi = Rbf(trainingX, trainingY, trainingV, epsilon = 10)
    vvec = rbfi(allCoords[:,0], allCoords[:,1])
    
    ugrid = uvec.reshape(nrRows,nrCols)
    vgrid = vvec.reshape(nrRows,nrCols)
    
    flow = np.dstack((ugrid,vgrid))
    print(flow.shape)

#### Methods to compute the anisotropy ####
def generate_data():
    data = np.zeros((200, 200), dtype=np.float)
    cov = np.array([[200, 100], [100, 200]])
    ij = np.random.multivariate_normal((100,100), cov, int(1e5))
    for i,j in ij:
        data[int(i), int(j)] += 1
    return data 

def _raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()

def _intertial_axis(data):
    """Calculate the x-mean, y-mean, and cov matrix of an image."""
    data_sum = data.sum()
    m10 = _raw_moment(data, 1, 0)
    m01 = _raw_moment(data, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum
    u11 = (_raw_moment(data, 1, 1) - x_bar * m01) / data_sum
    u20 = (_raw_moment(data, 2, 0) - x_bar * m10) / data_sum
    u02 = (_raw_moment(data, 0, 2) - y_bar * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return x_bar, y_bar, cov

def _make_lines(eigvals, eigvecs, mean, i):
        """Make lines a length of 2 stddev."""
        std = np.sqrt(eigvals[i])
        vec = 2 * std * eigvecs[:,i] / np.hypot(*eigvecs[:,i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
        
def decompose_cov_plot_bars(x_bar, y_bar, cov, ax):
    """Plot bars with a length of 2 stddev along the principal axes."""
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    ax.plot(*_make_lines(eigvals, eigvecs, mean, 0), marker='o', color='white')
    ax.plot(*_make_lines(eigvals, eigvecs, mean, -1), marker='o', color='white')
    ax.axis('image')
    return(eigvals,eigvecs)

def plot_bars(x_bar, y_bar, eigvals, eigvecs, ax, colour='white'):
    """Plot bars with a length of 2 stddev along the principal axes."""
    mean = np.array([x_bar, y_bar])
    ax.plot(*_make_lines(eigvals, eigvecs, mean, 0), marker='o', color=colour)
    ax.plot(*_make_lines(eigvals, eigvecs, mean, -1), marker='o', color=colour)
    ax.axis('image')
########################

def update_mean(data, newSample):
    '''
    Algorithm to compute the online mean.
    '''
    oldMean = np.nanmean(data)
    n = np.sum(~np.isnan(data))
    
    n += 1
    # Contribution of the new sample to the old mean
    delta = newSample - oldMean 
    # Update of the old mean
    newMean += delta/n 

    if n < 2:
        return float('nan')
    else:
        return newMean
        
def generate_wavelet_noise(rainfield, wavelet='db4', nrLevels=6, level2perturb='all', nrMembers=1):
    '''
    First naive attempt to generate stochastic noise using wavelets
    '''
    fieldSize = rainfield.shape
    
    # Decompose rainfall field
    coeffsRain = pywt.wavedec2(rainfield, wavelet, level=nrLevels)
    
    stochasticEnsemble = []
    for member in range(0,nrMembers):
        # Generate and decompose noise field
        noisefield = np.random.randn(fieldSize[0],fieldSize[1])
        coeffsNoise = pywt.wavedec2(noisefield, wavelet, level=nrLevels)
        
        if level2perturb == 'all':
            levels2perturbList = np.arange(1,nrLevels).tolist()
        else:
            if type(level2perturb) == int:
                levels2perturbList = [level2perturb]
            elif type(level2perturb) == np.ndarray:
                levels2perturbList = level2perturb.to_list()
            elif type(level2perturb) == list:
                levels2perturbList = level2perturb
            else:
                print('List of elvels to perturb in generate_wavelet_noise is not in the right format.')
                sys.exit(0)
        
        # Multiply the wavelet coefficients of rainfall and noise fields at each level
        for level in levels2perturbList:
            # Get index of the level since data are organized in reversed order
            levelReversed = nrLevels - level
            
            # Get coefficients of noise field at given level
            coeffsNoise[levelReversed] = list(coeffsNoise[levelReversed])
            
            # Get coefficients of rain field at given level
            coeffsRain[levelReversed] = list(coeffsRain[levelReversed])
            
            # Perturb rain coefficients with noise coefficients
            rainCoeffLevel = np.array(coeffsRain[levelReversed][:])
            noiseCoeffLevel = np.array(coeffsNoise[levelReversed][:])
            
            for direction in range(0,2):
                # Compute z-scores
                rainCoeffLevel_zscores,mean,stdev = to_zscores(rainCoeffLevel[direction])
                noiseCoeffLevel_zscores,mean,stdev = to_zscores(noiseCoeffLevel[direction])
                
                #rainCoeffLevel_zscores = rainCoeffLevel[direction]
                #noiseCoeffLevel_zscores = noiseCoeffLevel[direction]
                
                #print(rainCoeffLevel_zscores,noiseCoeffLevel_zscores)
                coeffsRain[levelReversed][direction] = rainCoeffLevel[direction]*noiseCoeffLevel[direction] #rainCoeffLevel_zscores#*noiseCoeffLevel_zscores
            
            # print(coeffsRain[levelReversed])
            # sys.exit()
            # Replace the rain coefficients with the perturbed coefficients
            coeffsRain[levelReversed] = tuple(coeffsRain[levelReversed])
        
        # Reconstruct perturbed rain field
        stochasticRain = pywt.waverec2(coeffsRain, wavelet)
        
        # Append ensemble members
        stochasticEnsemble.append(stochasticRain)
    
    return stochasticEnsemble
    
def to_zscores(data):
    mean = np.nanmean(data)
    stdev = np.nanstd(data)
    
    zscores = (data - mean)/stdev
    
    return zscores, mean, stdev
    
def from_zscores(data, mean, stdev):
    data = zscores*stdev + mean
    return data
    
def nanscatter(data, axis=0, minQ=16, maxQ=84):
    '''
    Function to compute the scatter score of Germann (simplified version without weighting).
    For a Gaussian distribution, the difference from the 84-16 quantiles is equal to +/- one standard deviation
    '''
    scatter = np.nanpercentile(data, maxQ, axis=axis) - np.nanpercentile(data, minQ, axis=axis)
    return scatter