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

from scipy import stats, fftpack
import scipy.signal as ss
import scipy.ndimage as ndimage
from skimage import measure
import datetime as datetime

import pandas as pd
import statsmodels.formula.api as sm
from statsmodels.nonparametric.api import KernelReg

import scipy as sp
import scipy.spatial.distance as dist

import radialprofile
import time_tools_attractor as ti

fmt2 = "%.2f"

def compute_imf(rainfield, rainThreshold=-1, noData=-999.0):
    idxRain = (rainfield > rainThreshold) & (rainfield != noData)
    
    if len(idxRain) != 0:
        imf = np.nanmean(rainfield[idxRain])
    else:
        imf = noData
    return(imf)

def compute_imf_array(rainfieldArray, rainThreshold=-1, noData=-999.0):
    imfArray = []
    for i in range(0,len(rainfieldArray)):
        imf = compute_imf(rainfieldArray[i], rainThreshold, noData)
        imfArray.append(imf)
    imfArray = np.array(imfArray)
    return(imfArray)
    
def compute_war(rainfield, rainThreshold, noData=-999.0):
    idxRain = rainfield > rainThreshold
    idxRadarDomain = rainfield > noData + 1
    
    if (len(idxRain) >= 0) and (len(idxRain) < sys.maxsize) and \
    (len(idxRadarDomain) >= 0) and (len(idxRadarDomain) < sys.maxsize) \
    and (np.sum(idxRain) <= np.sum(idxRadarDomain)) and (np.sum(idxRadarDomain) > 0):
        war = 100.0*np.sum(idxRain)/np.sum(idxRadarDomain)
    else:
        print("Problem in the computation of WAR. idxRain = ", idxRain, " and idxRadarDomain = ", idxRadarDomain, " are not valid values.")
        print("WAR set to -1")
        war = noData
    return war

def compute_war_array(rainfieldArray, rainThreshold, noData=-999.0):
    warArray = []
    for i in range(0,len(rainfieldArray)):
        war = compute_war(rainfieldArray[i], rainThreshold, noData)
        warArray.append(war)
    warArray = np.array(warArray)
    return(warArray)
    
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

def compute_2d_spectrum(rainfallImage, resolution=1, window=None, FFTmod='NUMPY'):
    '''
    Function to compute the 2D FFT power spectrum.
    
    Parameters
    ----------
    rainfallImage : numpyarray(float)
        Input 2d array with the rainfall field (or any kind of image)
    resolution : float
        Resolution of the image grid (e.g. in km) to compute the Fourier frequencies
    '''
    
    fieldSize = rainfallImage.shape
    minFieldSize = np.min(fieldSize)
    
    # Generate a window function
    if window == 'blackman':
        w1d = ss.blackman(minFieldSize)
        w = np.outer(w1d,w1d)
    elif window == 'hanning':
        w1d = np.hanning(minFieldSize)
        w = np.outer(w1d,w1d)
    elif window == 'flat-hanning':
        T = minFieldSize/4
        W = minFieldSize/2
        B = np.linspace(-W,W,2*W)
        R = np.abs(B)-T
        R[R<0]=0.
        A = 0.5*(1.0 + np.cos(np.pi*R/T))
        A[np.abs(B)>(2*T)]=0.0
        w1d = A
        w = np.outer(w1d,w1d)
    else:
        w = np.ones((fieldSize[0],fieldSize[1]))    
    
    # Compute FFT
    if FFTmod == 'NUMPY':
        fprecipNoShift = np.fft.fft2(rainfallImage*w) # Numpy implementation
    if FFTmod == 'FFTW':
        fprecipNoShift = pyfftw.interfaces.numpy_fft.fft2(rainfallImage*window) # FFTW implementation
        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()
    
    # Shift 2D spectrum
    fprecip = np.fft.fftshift(fprecipNoShift)
    
    # Compute 2D power spectrum
    psd2d = np.abs(fprecip)**2/(fieldSize[0]*fieldSize[1])    
    
    # Compute frequencies
    freqNoShift = fftpack.fftfreq(minFieldSize, d=float(resolution))
    freq = np.fft.fftshift(freqNoShift)
    
    return(psd2d, freq)

def compute_dft_1d_spectrum(rainfallImage, resolution=1, window='flat-hanning'):
    '''
    Function to compute the 1D Discrete Fourier Transform power spectrum.
    
    Parameters
    ----------
    rainfallImage : numpyarray(float)
        Input 2d array with the rainfall field (or any kind of image)
    resolution : float
        Resolution of the image grid (e.g. in km) to compute the Fourier frequencies
    '''
    
    fieldSize = rainfallImage.shape
    

    # Compute 2D power spectrum
    
    psd2d,_ = compute_2d_spectrum(rainfallImage,resolution, window)

    # Radial average
    psd1d, freq, wavelength = compute_radialAverage_spectrum(psd2d, resolution=1)

    return psd1d, freq, wavelength    
    
def compute_dct_1d_spectrum(rainfallImage, resolution=1):
    '''
    Function to compute the 1D Discrete Cosine Transform power spectrum.
    
    Parameters
    ----------
    rainfallImage : numpyarray(float)
        Input 2d array with the rainfall field (or any kind of image)
    resolution : float
        Resolution of the image grid (e.g. in km) to compute the Fourier frequencies
    '''
    
    fieldSize = rainfallImage.shape
    
    # Compute DCT
    fprecip = fftpack.dct(fftpack.dct(rainfallImage.T, norm='ortho').T, norm='ortho') 
    
    # Compute 2D power spectrum
    psd2d = np.abs(fprecip)**2/(fieldSize[0]*fieldSize[1])     

    # Variance binning (Denis et al., 2002)
    ix = range(fieldSize[0])
    jx = range(fieldSize[1])
    I,J = np.meshgrid(ix,jx)
    alphas = np.sqrt(I**2/fieldSize[0]**2 + J**2/fieldSize[1]**2)
    # plt.imshow(alphas)
    # plt.show()
    
    kmax = np.minimum(fieldSize[0],fieldSize[1])
    psd1d = np.zeros(kmax-1)
    wavelength = np.zeros(kmax-1)
    # for k in xrange(2,kmax):
    for k in xrange(1,kmax):
        alpha_k_low = k/kmax
        alpha_k_up = (k+1)/kmax
        wavelength[k-1] = 2*resolution/alpha_k_low
        # print(k,wavelength[k-1])
        idx00 = np.logical_or(I>0,J>0)
        idx = np.logical_and(alphas >= alpha_k_low,alphas < alpha_k_up, idx00)
        psd1d[k-1] += psd2d[idx].sum()
    
    freq = resolution*1.0/wavelength

    return psd1d, freq, wavelength
    
def compute_radialAverage_spectrum(psd2d, resolution=1):
    '''
    Function to compute the 1D radially averaged spectrum from the 2D spectrum.
    
    Parameters
    ----------
    psd2d : numpyarray(float)
        Input 2d array with the power spectrum.
    resolution : float
        Resolution of the image grid (e.g. in km) to compute the Fourier frequencies
    '''
    
    fieldSize = psd2d.shape
    minFieldSize = np.min(fieldSize)
    
    bin_size = 1
    nr_pixels, bin_centers, psd1d = radialprofile.azimuthalAverage(psd2d, binsize=bin_size, return_nr=True)
    
    # Extract subset of spectrum
    validBins = (bin_centers < minFieldSize/2) # takes the minimum dimension of the image and divide it by two
    psd1d = psd1d[validBins]
    
    # Compute frequencies
    freqNoShift = fftpack.fftfreq(minFieldSize, d=float(resolution))
    freqAll = np.fft.fftshift(freqNoShift)
    
    # Select only positive frequencies
    freq = freqAll[len(psd1d):]
    
    # Compute wavelength [km]
    with np.errstate(divide='ignore'):
        wavelength = resolution*(1.0/freq)
    # Replace 0 frequency with NaN
    freq[freq==0] = np.nan
    
    return(psd1d, freq, wavelength)

def create_xticks_1d_spectrum(maxScaleKM, minScaleKm):
    '''
    Use output as follows:
    ax.set_xticks(ticks_loc)
    ax.set_xticklabels(ticks)
    or
    plt.xticks(ticks_loc, ticks)
    '''
    # Create ticks in km
    ticksList = []
    tickLocal = maxScaleKM
    for i in range(0,20):
        ticksList.append(tickLocal)
        tickLocal = tickLocal/2
        if tickLocal < minScaleKm:
            break
    ticks = np.array(ticksList, dtype=int)
    ticks_loc = 10.0*np.log10(1.0/ticks)
    
    return(ticks_loc, ticks)

    
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

    if isinstance(fftSizeSub,float):
        fftSizeSub = int(fftSizeSub)
    
    fftSize = psd2d.shape
    
    if ((fftSize[0] % 2) != 0) or ((fftSize[1] % 2) != 0):
        print("Error in compute_fft_anisotropy: please provide an even sized 2d FFT spectrum.")
        sys.exit(1)
    
    fftMiddleX = int(fftSize[1]/2)
    fftMiddleY = int(fftSize[0]/2)
    
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
        percZero = 0.0
    
    # Treat cases where percentile is not a good choice and take a minimum correlation value (does not work with 2d spectrum)
    autocorrThreshold = 0.2
    if (percZero > 0) and (percZero < autocorrThreshold):
        percZero = autocorrThreshold
    
    # Shift spectrum/autocorrelation to start from 0 (zeros will be automatically neglected in the computation of covariance)
    psd2dsubSmoothShifted = psd2dsubSmooth - percZero
    psd2dsubSmoothShifted[psd2dsubSmoothShifted < 0] = 0.0
    
    ############### IMAGE SEGMENTATION
    # Image segmentation to remove high autocorrelations/spectrum values at far ranges/high frequencies
    psd2dsubSmoothShifted_bin = np.uint8(psd2dsubSmoothShifted > minThresholdCondition)
    
    # Compute image segmentation
    labelsImage = measure.label(psd2dsubSmoothShifted_bin, background = 0)
    
    # Get label of center of autocorrelation function (corr = 1.0)
    labelCenter = labelsImage[int(labelsImage.shape[0]/2),int(labelsImage.shape[1]/2)]
    
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
    idxMax = np.argmax(eigvals)
    #eccentricity = np.max(np.sqrt(eigvals))/np.min(np.sqrt(eigvals))
    eccentricity = math.sqrt(1-np.min(eigvals)/np.max(eigvals))
    orientation = np.degrees(math.atan2(eigvecs[1,idxMax],eigvecs[0,idxMax])) # atan or atan2?
        
    return psd2dsub, eccentricity, orientation, xbar, ybar, eigvals, eigvecs, percZero, psd2dsubSmoothShifted*mask

def compute_autocorrelation_fft2(imageArray, resolution=1, FFTmod = 'NUMPY'):
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
    
    # Compute frequencies
    freq_noshift = fftpack.fftfreq(np.min(field_dim), d=float(resolution))
    freq_shifted = np.fft.fftshift(freq_noshift)
    
    # Compute lags
    lag_shifted = np.arange(-np.min(field_dim)/2, (np.max(field_dim)/2)+1)*resolution
    
    toc = time.clock()
    #print("Elapsed time for ACF using FFT: ", toc-tic, " seconds.")
    return(autocorrelation_shifted, lag_shifted, powerSpectrum_shifted, freq_shifted)

def compute_autocorrelation_fft(timeSeries, FFTmod = 'NUMPY'):
    '''
    This function computes the autocorrelation of a time series using the FFT.
    It exploits the Wiener-Khinchin theorem, which states that the Fourier transform of the auto-correlation function   
    is equal to the Fourier transform of the signal. Thus, the autocorrelation function can be obtained as the inverse transform of
    the power spectrum.
    It is very important to know that the auto-correlation function, as it is referred to as in the literature, is in fact the noncentred
    autocovariance. In order to obtain values of correlation between -1 and 1, one must center the signal by removing the mean before
    computing the FFT and then divide the obtained auto-correlation (after inverse transform) by the variance of the signal.
    '''
    
    tic = time.clock()
    
    # Compute field mean and variance
    field_mean = np.mean(timeSeries)
    field_var = np.var(timeSeries)
    
    nr_samples = len(timeSeries)
    
    # Compute FFT
    if FFTmod == 'NUMPY':
        fourier = np.fft.fft(timeSeries - field_mean) # Numpy implementation
    if FFTmod == 'FFTW':
        fourier = pyfftw.interfaces.numpy_fft.fft(timeSeries - field_mean) # FFTW implementation
        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()
    
    # Compute power spectrum
    powerSpectrum = np.abs(fourier)**2/nr_samples
    
    # Compute inverse FFT of spectrum
    if FFTmod == 'NUMPY':
        autocovariance = np.fft.ifft(powerSpectrum) # Numpy implementation
    if FFTmod == 'FFTW':
        autocovariance = pyfftw.interfaces.numpy_fft.ifft(powerSpectrum) # FFTW implementation
        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()
    
    # Compute auto-correlation from auto-covariance
    autocorrelation = autocovariance.real/field_var
    
    # Take only first half (the autocorrelation and spectrum are symmetric)
    autocorrelation = autocorrelation[0:int(nr_samples/2)]
    powerSpectrum = powerSpectrum[0:int(nr_samples/2)]
    
    if (nr_samples % 2) != 0:
        print('Beware that it is better to pass an even number of samples for FFT to compute_autocorrelation_fft.')
        
    toc = time.clock()
    #print("Elapsed time for ACF using FFT: ", toc-tic, " seconds.")
    return(autocorrelation, powerSpectrum)

def fourier_low_pass2d(imageArray, cutoff_scale_km, resolution_km=1, FFTmod='NUMPY'):
    field_mean = np.mean(imageArray)
    field_var = np.var(imageArray)
    field_dim = imageArray.shape
    
    # Compute FFT
    if FFTmod == 'NUMPY':
        fourier = np.fft.fft2(imageArray) # Numpy implementation
    if FFTmod == 'FFTW':
        fourier = pyfftw.interfaces.numpy_fft.fft2(imageArray) # FFTW implementation
        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()

    fourier_shifted = np.fft.fftshift(fourier)
    
    # Compute frequencies
    freq_noshift_x = fftpack.fftfreq(field_dim[1], d=float(resolution_km))
    freq_noshift_y = fftpack.fftfreq(field_dim[0], d=float(resolution_km))
    freq_shifted_x = np.fft.fftshift(freq_noshift_x)
    freq_shifted_y = np.fft.fftshift(freq_noshift_y)
    
    freq_shifted_x, freq_shifted_y = np.meshgrid(freq_shifted_x, freq_shifted_y)
    
    # Get filter
    cutoff_frequency = resolution_km/cutoff_scale_km
    mask = freq_shifted_x**2 + freq_shifted_y**2 <= cutoff_frequency**2
    mask = mask.astype(int)
    
    # Apply mask
    fourier_shifted = fourier_shifted*mask
    fourier = np.fft.fftshift(fourier_shifted)
    
    # Plotting masked spectrum to check everything is right
    # nr_samples = field_dim[0]*field_dim[1]
    # plt.imshow(np.log10(np.abs(fourier_shifted)**2/nr_samples))
    # # plt.imshow(mask)
    # plt.show()
    
    # Tranform back
    if FFTmod == 'NUMPY':
        imageArray_bandpassed = np.fft.ifft2(fourier) # Numpy implementation
    if FFTmod == 'FFTW':
        imageArray_bandpassed = pyfftw.interfaces.numpy_fft.ifft2(fourier) # FFTW implementation
        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()

    # Get real part
    imageArray_bandpassed = imageArray_bandpassed.real

    return(imageArray_bandpassed)
    
def time_delay_embedding(timeSeries, nrSteps=1, stepSize=1, noData=np.nan):
    '''
    This function takes an input time series and gives an ndarray of delayed vectors.
    nrSteps=1 will simple give back the same time series.
    nrSteps=2 will give back the time series and as second column the delayed time series by stepsSize
    '''
    
    timeSeries = np.array(timeSeries)
    if nrSteps == 1:
        timeSeries = timeSeries.reshape(len(timeSeries),1)
        return(timeSeries)
    
    nrSamples = len(timeSeries)
    delayedArray = np.ones((nrSamples,nrSteps))*noData
    
    # Put lag0 time series in the first column
    delayedArray[:,0] = timeSeries
    
    for i in range(0,nrSteps):
        # Generate nodata to append to the delayed time series
        if i*stepSize <= nrSamples:
            timeSeriesNoData = noData*np.ones(i*stepSize)
        else:
            timeSeriesNoData = noData*np.ones(nrSamples)
        
        # Get the delayed time series segment
        timeSeriesSegment = timeSeries[i*stepSize:]
        timeSeriesSegment = np.hstack((timeSeriesSegment, timeSeriesNoData)).tolist()
        
        delayedArray[:,i] = timeSeriesSegment
    
    return(delayedArray)
        
def correlation_dimension(dataArray, nrSteps=100, Lnorm=2, plot=False):
    '''
    Function to estimate the correlation dimension (Grassberger-Procaccia algorithm)
    '''
    
    nr_samples = dataArray.shape[0]
    nr_dimensions = dataArray.shape[1]
    
    if nr_samples < 10:
        print('Not enough samples to estimate fractal dimension.')
        radii = []
        Cr = []
        fractalDim = np.nan
        intercept = np.nan
        return(radii, Cr, fractalDim, intercept)
    
    # Compute the L_p norm between all pairs of points in the high dimensional space
    # Correlation dimension requires the computation of the L1 norm (p=1), i.e. |Xi-Xj|
    lp_distances = dist.squareform(dist.pdist(dataArray, p=Lnorm))
    #lp_distances = dist.pdist(dataArray, p=1) # Which one appropriate? It gives different fractal dims...
    
    # Normalize distances by their st. dev.?
    sd_dist = np.std(lp_distances)
    #lp_distances = lp_distances/sd_dist
    
    # Define range of radii for which to evaluate the correlation sum Cr
    strategyRadii = 'log'# 'log' or 'linear'
    
    if strategyRadii == 'linear':
        r_min = np.min(lp_distances)
        r_max = np.max(lp_distances)
        radii = np.linspace(r_min, r_max, nrSteps)
    if strategyRadii == 'log':
        r_min = np.percentile(lp_distances[lp_distances != 0],0.01)
        r_max = np.max(lp_distances)
        radiiLog = np.linspace(np.log10(r_min), np.log10(r_max), nrSteps)
        radii = 10**radiiLog
    
    Cr = []
    for r in radii:
        s = 1.0 / (nr_samples * (nr_samples-1)) * np.sum(lp_distances <= r) # fraction
        #s = np.sum(lp_distances < r)/2 # count
        Cr.append(s)
    Cr = np.array(Cr)
    
    # Filter zeros from Cr
    nonzero = np.where(Cr != 0)
    radii = radii[nonzero]
    Cr = Cr[nonzero]
    
    # Put r and Cr in log units
    logRadii = np.log10(radii)
    logCr = np.log10(Cr)
    
    fittingStrategy = 2
    
    ### Strategy 1 for fitting the slope
    if fittingStrategy == 1:
        # Define a subrange for which the log(Cr)-log(r) curve is linear and good for fitting
        r_min_fit = np.percentile(lp_distances,5)
        r_max_fit = np.percentile(lp_distances,50)
        subsetIdxFitting = (radii >= r_min_fit) & (radii <= r_max_fit)
        
        # Compute correlation dimension as the linear slope in loglog plot
        reg = sp.polyfit(logRadii[subsetIdxFitting], logCr[subsetIdxFitting], 1)
        slope = reg[0]
        fractalDim = slope
        intercept = reg[1]
    
    ### Strategy 2 for fitting the slope
    if fittingStrategy == 2:
        nrPointsFitting = 20
        startIdx = 0
        maxSlope = 0.0
        maxIntercept = -9999
        while startIdx < (len(radii) - nrPointsFitting):
            subsetIdxFitting = np.arange(startIdx, startIdx+nrPointsFitting)
            reg = sp.polyfit(logRadii[subsetIdxFitting], logCr[subsetIdxFitting], 1)
            slope = reg[0]
            intercept = reg[1]
            if slope > maxSlope:
                maxSlope = slope
                maxIntercept = intercept
            startIdx = startIdx + 2
        # Get highest slope (largest fractal dimension estimation)
        slope = maxSlope
        fractalDim = slope
        intercept = maxIntercept
    
    ######## Plot fitting of correlation dimension
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(logRadii, logCr, 'b', linewidth=2)
        regFit = intercept + slope*logRadii
        plt.plot(logRadii, regFit, 'r', linewidth=2)
        
        plt.title('Correlation dimension estimation', fontsize=24)
        plt.xlabel('log(r)', fontsize=20)
        plt.ylabel('log(C(r))', fontsize=20)
        
        plt.text(0.05,0.95,'Sample size   = ' + str(nr_samples), transform=ax.transAxes, fontsize=16)
        plt.text(0.05,0.90,'Embedding dim = ' + str(nr_dimensions), transform=ax.transAxes, fontsize=16)
        plt.text(0.05,0.85,'Fractal dim   = ' + str(fmt2 % slope), transform=ax.transAxes, fontsize=16)
        plt.show()
        
        # plt.imshow(lp_distances)
        # plt.show()
    
    return(radii, Cr, fractalDim, intercept)

def logarithmic_r(min_n, max_n, factor):
	"""
	Creates a list of values by successively multiplying a minimum value min_n by
	a factor > 1 until a maximum value max_n is reached.

	Args:
		min_n (float): minimum value (must be < max_n)
		max_n (float): maximum value (must be > min_n)
		factor (float): factor used to increase min_n (must be > 1)

	Returns:
		list of floats: min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
	"""
	assert max_n > min_n
	assert factor > 1
	max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
    
	return [min_n * (factor ** i) for i in range(max_i+1)]
    
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
    rbfi = Rbf(trainingX, trainingY, trainingU, epsilon = 10)
    uvec = rbfi(allCoords[:,0], allCoords[:,1])
    
    rbfi = Rbf(trainingX, trainingY, trainingV, epsilon = 10)
    vvec = rbfi(allCoords[:,0], allCoords[:,1])
    
    ugrid = uvec.reshape(nrRows,nrCols)
    vgrid = vvec.reshape(nrRows,nrCols)
    
    flow = np.dstack((ugrid,vgrid))

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
    #ax.axis('image') # may give a weird displacement of axes...
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

def wavelet_decomposition_2d(rainfield, wavelet = 'haar', nrLevels = None):
    nrRows = rainfield.shape[0]
    nrCols = rainfield.shape[1]

    if nrLevels == None:
        minDim = np.min([nrRows,nrRows])
        nrLevels = int(np.log2(minDim))
    # Perform wavelet decomposition
    w = pywt.Wavelet(wavelet)
    
    wavelet_coeff = []
    for level in range(0,nrLevels):
        # Decompose rainfield with wavelet
        cA, (cH, cV, cD) = pywt.dwt2(rainfield, wavelet)
        # Next rainfield to decompose is equal to the wavelet approximation
        rainfield = cA/2.0
        wavelet_coeff.append(rainfield)
    
    return(wavelet_coeff)

def generate_wavelet_coordinates(wavelet_coeff, originalImageShape, Xmin, Xmax, Ymin, Ymax, gridSpacing):
    
    nrScales = len(wavelet_coeff)
    # Generate coordinates of centers of wavelet coefficients
    xvecs = []
    yvecs = []
    for scale in range(0,nrScales):
        wc_fieldsize = np.array(wavelet_coeff[scale].shape)
        wc_boxsize = np.array(originalImageShape)/wc_fieldsize*gridSpacing
        gridX = np.arange(Xmin + wc_boxsize[1]/2,Xmax,wc_boxsize[1])
        gridY = np.flipud(np.arange(Ymin + wc_boxsize[0]/2,Ymax,wc_boxsize[0]))
        # print(wc_fieldsize, wc_boxsize)
        # print(Xmin, Xmax, gridX, gridY)
        xvecs.append(gridX)
        yvecs.append(gridY)
    
    return(xvecs, yvecs)
    
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

def get_level_from_scale(resKM, scaleKM):
    if resKM == scaleKM:
        print('scaleKM should be larger than resKM in st.get_level_from_scale')
        sys.exit()
    elif isPower(scaleKM, resKM*2) == False:
        print('scaleKM should be a power of 2 in st.get_level_from_scale')
        sys.exit()
        
    for t in range(0,50):
        resKM = resKM*2
        if resKM == scaleKM:
            level = t
    return(level)

def isPower(n, base):
    return base**int(math.log(n, base)+.5)==n

    
def to_zscores(data, axis=None):

    if axis is None:
        mean = np.nanmean(data)
        stdev = np.nanstd(data)    
    else:
        mean = np.nanmean(data, axis=axis)
        stdev = np.nanstd(data, axis=axis)
    
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
    
def spherical_model(h, nugget, sill, range):
    c0 = nugget
    c1 = sill
    a = range
    
    spherical = np.where(h > a, c0 + c1, c0 + c1*(1.5*(h/a) - 0.5*(h/a)**3))
    return spherical

def exponential_model(h, nugget, sill, range):
    c0 = nugget
    c1 = sill
    a = range
    exponential = c0 + c1*(1-np.exp(-3*h/a))
    return exponential
    
def box_cox_transform(datain,Lambda):
    dataout = datain.copy()
    if Lambda==0:
        dataout = np.log(dataout)
    else:
        dataout = (dataout**Lambda - 1)/Lambda
    return dataout
    
def box_cox_transform_test_lambdas(datain,lambdas=[]):
    if len(lambdas)==0:
        lambdas = np.linspace(-1,1,11)
    data = []
    labels=[]
    sk=[]
    for l in lambdas:
        data_transf = box_cox_transform(datain,l)
        data_transf = (data_transf - np.mean(data_transf))/np.std(data_transf)
        data.append(data_transf)
        labels.append('{0:.1f}'.format(l))
        sk.append(stats.skew(data_transf)) # skewness
    
    bp = plt.boxplot(data,labels=labels)
    
    ylims = np.percentile(data,0.99)
    plt.title('Box-Cox transform')
    plt.xlabel('Lambdas')
    
    ymax = np.zeros(len(data))
    for i in range(len(data)):
        y = sk[i]
        x = i+1
        plt.plot(x, y,'ok',ms=5, markeredgecolor ='k')
        fliers = bp['fliers'][i].get_ydata()
        if len(fliers>0):
            ymax[i] = np.max(fliers)
    ylims = np.percentile(ymax,60)
    plt.ylim((-1*ylims,ylims))
    plt.show()
    
def ortho_rotation(lam, method='varimax',gamma=None, 
                    eps=1e-6, itermax=100): 
    """ 
    Return orthogal rotation matrix 

    TODO: - other types beyond  
    """ 
    if gamma == None: 
        if (method == 'varimax'): 
            gamma = 1.0 
        if (method == 'quartimax'): 
            gamma = 0.0 

    nrow, ncol = lam.shape
    R = np.eye(ncol) 
    var = 0 

    for i in range(itermax): 
        lam_rot = np.dot(lam, R) 
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma 
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp))) 
        R = np.dot(u, v) 
        var_new = np.sum(s) 
        if var_new < var * (1 + eps): 
            break 
        var = var_new 

    return R 


from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    '''
    Function to compute the varimax rotation.
    Adapted from http://stackoverflow.com/questions/17628589/perform-varimax-rotation-in-python-using-numpy
    
    Parameters
    ----------
    Phi : numpyarray(float)
        Input matrix with the loadings (eigenvectors)
    gamma : float
        gamma = 1 (varimax), gamma = 0 (quartimax)
    q : int
        Maximum number of iterations
    tol : float
        Tolerance criterion to stop the iterations
    
    Returns
    ----------
    Phi_rot: numpyarray(float)
        Output matrix with the rotated loadings (eigenvectors)
    R: numpyarray(float)
        Output rotation matrix (it can be used to re-project the PC scores)
    '''
    
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    
    Phi_rot = dot(Phi, R)
    return(Phi_rot, R) 

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.abs(np.diff(data)) != stepsize)[0]+1)
   
def retrieve_analogues(timeStampDt, timeStampDtArray, phaseSpaceArray, N=10, indepTimeHours=6):
    '''
    Function to retrieve the analogues from a dataset.
    '''
    
    dataDim = phaseSpaceArray.shape
    if len(timeStampDtArray) != phaseSpaceArray.shape[0]:
        print(len(timeStampDtArray), 'vs', phaseSpaceArray.shape[0])
        print('timeStampDtArray, phaseSpaceArray should have the same number of elements.')
        sys.exit(1)
    
    if type(timeStampDt) == int:
        timeStampDt = str(timeStampDt)
    
    # Find the asked time stamp index
    targetIdx = np.where(ti.datetime2absolutetime(timeStampDt) == ti.datetime2absolutetime(timeStampDtArray))[0]
    
    if len(targetIdx) == 0:
        print('The asked target timestamp is not in the archive.')
        analogueIndices = []
        analogueDateTimes = []
        targetIdx = []
        distances = []
        return(analogueIndices, analogueDateTimes, targetIdx, distances)
    
    # Get values of phase space dimensions for that time stamp
    targetFeatures = phaseSpaceArray[targetIdx,:]
    
    # Compute M-Dimensional Euclidean distances between target feature vector and all the archive
    distancesArray = dist.cdist(targetFeatures, phaseSpaceArray, p=2)
    distancesArray = distancesArray.flatten()
    
    # Collect the time stamps iteratively to respect the temporal independence criterion (e.g. 6 hours)
    analogueIndices, analogueDateTimes = select_independent_times(timeStampDtArray, distancesArray, N=N, indepTimeHours=indepTimeHours, keepFirst=False)
    
    return(analogueIndices, analogueDateTimes, targetIdx, distancesArray)
    
def select_independent_times(timeStampDtArray, distancesArray, N=5, indepTimeHours=6, keepFirst=False):
    
    if keepFirst == True:
        N = N-1
    nrSamples = len(timeStampDtArray)
    
    # Sort Euclidean distances
    sortedIdx = np.argsort(distancesArray)
        
    # Collect analogues iteratively to respect the independence criterion
    indepTimeSecs = indepTimeHours*60*60
    
    nrAnalogues = 0
    indepIndices = [sortedIdx[0]]
    indepDateTimes = [timeStampDtArray[sortedIdx[0]]] # start with the sample with minimum distance (zero)
    for a in range(0,nrSamples): 
        analogueIdx = sortedIdx[a]
        tmpDt = timeStampDtArray[analogueIdx]
        
        ## Check whether the new time stamp is far enough from the ones already collected
        timeDiffArray = []
        for t in range(0,len(indepDateTimes)):
            timeDiff = np.abs((indepDateTimes[t] - tmpDt).total_seconds())
            timeDiffArray.append(timeDiff)
        timeDiffArray = np.array(timeDiffArray)
        
        ## If yes, collect it
        if np.min(timeDiffArray) >= indepTimeSecs:
            indepIndices.append(analogueIdx)
            indepDateTimes.append(timeStampDtArray[analogueIdx])
            
            # If collected enough analogues break the loop
            if len(indepIndices) > N:
                break
    
    # Remove first element (analogue with itself)
    if keepFirst == False:
        indepIndices = indepIndices[1:]
        indepDateTimes = indepDateTimes[1:]
    
    return(indepIndices, indepDateTimes)
    
def scores_det_cat_fcst(pred,y):
    
    ##############################
    
    ## purpose:
    # calculate scores (simple + skill) for deterministic categorical forecasts
    
    ## input:
    # pred: 1d array with the predictions
    # y: 1d array with the true values

    ## output:
    # ss: calculated simple + skill scores
    # ss_names: namelist of the calculated scores
    
    ##############################


    ##############################        
    ## calculate hits, misses, false positives, correct rejects   
    ##############################
    
    H_idx = np.logical_and(pred==1,y==1) # correctly predicted precip
    F_idx = np.logical_and(pred==1,y==0) # predicted precip even though none there
    M_idx = np.logical_and(pred==0,y==1) # predicted no precip even though there was
    R_idx = np.logical_and(pred==0,y==0) # correctly predicted no precip

    H = sum(H_idx).astype(float)
    M = sum(M_idx).astype(float)
    F = sum(F_idx).astype(float)
    R = sum(R_idx).astype(float)
    tot = H+M+F+R


    print('H:',H/tot*100,'M:',M/tot*100,'F:',F/tot*100,'R:',R/tot*100)

    
    ##############################        
    ## calculate simple scores 
    ##############################
    
    POD = H/(H+M) # probability of detection
    FAR = F/(H+F) # false alarm ratio
    FA = F/(F+R) # false alarm rate = prob of false detection
    s = (H+M)/(H+M+F+R) # base rate = freq of observed events

    #POR = R/(R+F) # probability of rejection
    #FRR = M/(M+R) # false rejection ratio
    #FB = (H+F)/(H+M) # frequency bias: systematic error only

    ACC = (H+R)/(H+M+F+R) #accuracy (fraction correct) <- attention: not really suitable measure for rare events:
            # large values for conservative fcst there (because events / non-events treated symmetrically)
    CSI = H/(H+M+F) # critical success index: fraction of all fcsted or observed events that were correct
            # (asymmetric between events / non-events)


    ##############################        
    ## calculate skill scores 
    ##############################
    HSS = 2*(H*R-F*M)/((H+M)*(M+R)+(H+F)*(F+R)) # Heidke Skill Score (-1 < HSS < 1) < 0 implies no skill
    # CSI2 = POD/(1+FA*(1-s)/s) # just entered for cross reference test -> ok :)
    # HSS2 = 2*s*(1-s)*(POD-FA)/(s+s*(1-2*s)*POD+(1-s)*(1-2*s)*FA) # just entered for cross reference test -> ok :)

    HK = POD-FA #Hanssen-Kuipers Discriminant
    GSS = (POD-FA)/((1-s*POD)/(1-s)+FA*(1-s)/s) # Gilbert Skill Score
    #aref = (H+M)*(H+F)/(H+F+M+R)
    #GSS2 = (H-aref)/(H-aref+F+M) # jep, also GSS is calculated correctly

    #SEDI = (np.log(FA)-np.log(POD)+np.log(1-POD)-np.log(1-FA))/(np.log(FA)+np.log(POD)+np.log(1-POD)+np.log(1-FA))
        # Symmetric extremal dependence index: not a form of a SS, specifically designed for rare events

    ss = [POD,FA,FAR,ACC,CSI,HSS,HK,GSS] # all explained in Lecture Frei except POR & FRR (-> paper
                # Roebling&Holleman2009)
   
    ss_names = ['POD','FA','FAR','ACC','CSI','HSS','HK','GSS']
    
    print('HK:',HK, 'HSS:', HSS)
    del H_idx, F_idx, M_idx, R_idx, H, M, F, R, tot
    
    return ss,ss_names

def elements_in_list(small_list, large_list):
    '''
    Finds whether any of the items in small_list is contained in large_list
    '''
    b = any(item in large_list for item in small_list)        
    return(b)
    
from scipy.stats import spearmanr, pearsonr
def scores_det_cont_fcst(pred, o, 
scores_list=['ME_add','RMSE_add','RV_add','corr_s','corr_p','beta','ME_mult','RMSE_mult','RV_mult'], offset=0.01):
    '''
    ##############################
    
    Purpose:
    calculate scores (simple + skill) for deterministic continuous forecasts
    
    Input:
    pred: 1d array with the predictions
    o: 1d array with the true values
    scores_list: list of scores to compute

    Output:
    ss: calculated simple + skill scores
    ss_names: namelist of the calculated scores
    
    '''
    ##############################
    isNaN = np.isnan(pred) | np.isnan(o)
    pred = pred[~isNaN]
    o = o[~isNaN]
    
    N = o.shape[0]
    s_o = np.sqrt(1.0/N*sum((o-o.mean())**2))
    s_pred = np.sqrt(1.0/N*sum((pred-pred.mean())**2)) # sample standard deviation of prediction
    
    # Compute additive and multiplicative residuals
    add_res = pred-o # additive residuals
    b = elements_in_list(['ME_mult', 'RMSE_mult', 'RV_mult'], scores_list)
    if b:
        mult_res = 10.0*np.log10((pred + offset)/(o + offset))# multiplicative residuals
        if (np.sum(pred < 0) > 0) or (np.sum(o < 0) > 0):
            print('Beware that pred and o should not contain negative values to compute the multiplicative residuals.')
    
    scores = []
    scores_list_sorted = []
    
    # mean error (stm called bias... but somehow doesn't add up with multiplicative bias from Christoph Frei's lecture)
    if 'ME_add' in scores_list:
        ME_add = np.mean(add_res)
        scores.append(ME_add)
        scores_list_sorted.append('ME_add')
    
    if 'ME_mult' in scores_list:
        ME_mult = np.mean(mult_res)
        scores.append(ME_mult)
        scores_list_sorted.append('ME_mult')
    
    # root mean squared errors
    if 'RMSE_add' in scores_list:
        RMSE_add = np.sqrt(1.0/N*sum((add_res)**2))
        scores.append(RMSE_add)
        scores_list_sorted.append('RMSE_add')
    
    if 'RMSE_mult' in scores_list:
        RMSE_mult = np.sqrt(1.0/N*sum((mult_res)**2))
        scores.append(RMSE_mult)
        scores_list_sorted.append('RMSE_mult')
    
    # reduction of variance scores (not sure whether even makes sense in multiplicative space)
    if 'RV_add' in scores_list:
        RV_add = 1.0 - 1.0/N*sum((add_res)**2)/s_o**2
        scores.append(RV_add)
        scores_list_sorted.append('RV_add')
    
    if 'RV_mult' in scores_list:
        dBo = 10*np.log10(o+offset)
        s_dBo = np.sqrt(1.0/N*sum((dBo-dBo.mean())**2))
        RV_mult = 1.0-1.0/N*sum((mult_res)**2)/s_dBo**2
        scores.append(RV_mult)
        scores_list_sorted.append('RV_mult')
    
    # spearman corr (rank correlation)
    if 'corr_s' in scores_list:
        corr_s = spearmanr(pred,o)[0]
        scores.append(corr_s)
        scores_list_sorted.append('corr_s')
    
    # pearson corr
    if 'corr_p' in scores_list:
        corr_p = pearsonr(pred,o)[0]
        scores.append(corr_p)
        scores_list_sorted.append('corr_p')
        
    # beta (linear regression slope)
    if 'beta' in scores_list:
        beta = s_o/s_pred*corr_p
        scores.append(beta)
        scores_list_sorted.append('beta')
    
    #scores_dict = dict(zip(scores_list, scores))
    
    return scores, scores_list_sorted
 
def plot_2dhistogram(x_array, y_array, step_x=None, step_y=None, xlims=None, ylims=None, cmap='jet', add_regress=True):
    '''
    Function to plot a 2D histogram, e.g. to visualize a scatterplot of observed vs predicted values.
    The function returns the axes so that you can set title, xlabel and ylabel, e.g.
    ax.set_xlabel('Observed growth and decay [dB]')
    ax.set_ylabel('Predicted growth and decay [dB]')
    ax.set_title('MLP predictions against observations')
    '''
    
    # X and Y lims
    if xlims == None:
        xlims = [np.nanmin(x_array), np.nanmax(x_array)]
    if ylims == None:
        ylims = [np.nanmin(y_array), np.nanmax(y_array)]
    
    # Step size for 2D hist
    if step_x == None:
        bins_x = np.linspace(xlims[0], xlims[1], num=40)
    else:
        bins_x = np.arange(xlims[0], xlims[1]+step_x, step_x)
    
    if step_y == None:
        bins_y = np.linspace(ylims[0], ylims[1], num=40)
    else:
        bins_y = np.arange(ylims[0], ylims[1]+step_y, step_y)    
        
    ########
    # Compute 2D histogram
    H, xedges, yedges = np.histogram2d(x_array, y_array, bins=(bins_x, bins_y))
    
    # Mask histogram for zeros
    import numpy.ma as ma
    H[H==0] = np.nan
    H = ma.masked_where(np.isnan(H),H)
    X, Y = np.meshgrid(xedges, yedges)
    
    # Plot 2D histogram
    plt.figure()
    ax = plt.subplot(111)
    
    import matplotlib as mpl
    pc = ax.pcolormesh(X, Y, H.T, norm=mpl.colors.LogNorm(), cmap=plt.get_cmap(cmap))
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.colorbar(pc)
    
    if add_regress:
        add_regression_line_scores(x_array, y_array, ax)
    
    return(ax)

def add_regression_line_scores(x_array, y_array, ax):
    '''
    Function to add a regression line to a scatterplot / 2D histogram + a legend with scores
    '''
    ########
    fmt2 = "%.2f"
    
    xmin, xmax = ax.get_ylim()
    xlims = [xmin, xmax]
    ymin, ymax = ax.get_ylim()
    ylims = [ymin, ymax]
     
    # Plot line perfect regression
    ax.plot(xlims, ylims, 'k--', linewidth=0.5)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot regression line
    beta, intercept, rho, = compute_beta(x_array, y_array)
    ax.plot(xlims, intercept+ beta*np.array(xlims), 'k--', linewidth=0.5)
    
    ################################
    # Compute statistics and errors
    scores_list = ['corr_s', 'RMSE_add'] #, 'RV_add']
    scores, scores_list = scores_det_cont_fcst(x_array, y_array, scores_list)
    
    text_legend = ''
    scores_names = scores_list_names(scores_list)
    for i in range(len(scores_names)):
        text_legend += scores_names[i] + ' = ' + (fmt2 % (scores[i])) + '\n'
        
    # Add legend with scores
    t = ax.text(0.95, 0.15, text_legend, transform=ax.transAxes, fontsize=10, horizontalalignment='right')   
    t.set_bbox((dict(facecolor='white', alpha=0.7, edgecolor='white')))

    return(ax)

def scores_list_names(scores_list_in):
    '''
    Replaces the scores_list with netter names for plotting.
    '''
    scores_list = ['ME_add','RMSE_add','RV_add','corr_s','corr_p','beta','ME_mult','RMSE_mult','RV_mult']
    scores_names = ['ME','RMSE','RV','SCORR','PCORR','beta','ME_mult','RMSE_mult','RV_mult']
    
    scores_names_out = []
    for i in range(len(scores_list_in)):
        for j in range(len(scores_list)):
            if scores_list[j] == scores_list_in[i]:
                scores_names_out.append(scores_names[j])
        
    return(scores_names_out)    