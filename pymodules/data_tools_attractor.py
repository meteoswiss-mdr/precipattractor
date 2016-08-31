#!/usr/bin/env python
'''
Module to perform various data operations and statistics
(anisotropy estimation, rainfall-reflectivity conversion, spectral slope computation, WAR statistics, etc).

Documentation convention from https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

30.08.2016
Loris Foresti
'''

from __future__ import division
from __future__ import print_function

import sys
import math 

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from statsmodels.nonparametric.api import KernelReg
import cv2

from scipy import stats
import scipy.ndimage as ndimage
from scipy.signal import blackman
from skimage import measure

import matplotlib.colors as colors
import matplotlib.pyplot as plt

def to_dB(array, offset=0.01):
    '''
    Transform array to dB
    '''
    isList = isinstance(array, list)
    if isList == True:
        array = np.array(array)
    if offset != -1:
        dBarray = 10.0*np.log10(array + offset)
    else:
        dBarray = 10.0*np.log10(array)
        
    if isList == True:
        dBarray = dBarray.tolist()
    return(dBarray)

def get_column_list(list2D, columnNr):
    listColumn = [item[columnNr] for item in list2D]
    return(listColumn)

def get_variable_indices(subsetVariableNames, listVariableNames):
    nrVarTot = len(listVariableNames)
    nrVarSubset = len(subsetVariableNames)

    indices = []
    for item in range(0,nrVarSubset):
        var = subsetVariableNames[item]
        index = listVariableNames.index(var)
        indices.append(index)
    return(indices)

def get_reduced_extent(width, height, domainSizeX, domainSizeY):
    borderSizeX = (width - domainSizeX)/2
    borderSizeY = (height - domainSizeY)/2
    extent = (borderSizeX, borderSizeY, width-borderSizeX, height-borderSizeY) # left, upper, right, lower
    return(extent)
    
def extract_middle_domain_img(demImg, domainSizeX, domainSizeY):
    width, height = demImg.size
    extent = get_reduced_extent(width, height, domainSizeX, domainSizeY)
    demImgCropped = demImg.crop(extent)
    
    return(demImgCropped)

def extract_middle_domain(rainfield, domainSizeX, domainSizeY):
    borderSizeX = int((rainfield.shape[1] - domainSizeX)/2)
    borderSizeY = int((rainfield.shape[0] - domainSizeY)/2)
    mask = np.ones((rainfield.shape))
    mask[0:borderSizeY,:] = 0
    mask[borderSizeY+domainSizeY:,:] = 0
    mask[:,0:borderSizeX] = 0
    mask[:,borderSizeX+domainSizeX:] = 0
    
    rainfieldDomain = rainfield[mask==1].reshape(domainSizeY,domainSizeX)
    
    return(rainfieldDomain)

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
    
def rainrate2reflectivity(rainrate, A, b):
    zerosIdx = rainrate == 0
    rainIdx = rainrate > 0
    
    # Compute or set minimum reflectivity
    nrRainPixels = np.sum(rainIdx)
    if nrRainPixels >= 10:
        minRainRate = np.min(rainrate[rainIdx])
    else:
        minRainRate = 0.012 # 0.0115537519713
    minDBZ = 10.0*np.log10(A*minRainRate+b)
    
    # Compute reflectivity
    dBZ = -999.0*np.ones(rainrate.shape)
    dBZ[rainIdx] = 10.0*np.log10(A*rainrate[rainIdx]+b)
    
    # Replace zero rainrate by the minimum observed reflectivity
    dBZ[zerosIdx] = minDBZ
    
    return dBZ
    
def get_rainfall_lookuptable(noData):
    precipIdxFactor=71.5
    lut = np.zeros(256)
    for i in range(0,256):
        if (i < 2) or (i > 250 and i < 255):
            lut[i] = 0.0
        elif (i == 255):
            lut[i] = noData
        else:
            lut[i] = (10.**((i-(precipIdxFactor))/20.0)/316.0)**(0.6666667)
    
    return lut

def get_colorlist(type):
    if type == 'STEPS':
        color_list = ['cyan','deepskyblue','dodgerblue','blue','chartreuse','limegreen','green','darkgreen','yellow','gold','orange','red','magenta','darkmagenta']
        clevs = [0.1,0.25,0.4,0.63,1,1.6,2.5,4,6.3,10,16,25,40,63,100]
    if type == 'MeteoSwiss':
        pinkHex = '#%02x%02x%02x' % (232, 215, 242)
        redgreyHex = '#%02x%02x%02x' % (156, 126, 148)
        color_list = [pinkHex, redgreyHex, "#640064","#AF00AF","#DC00DC","#3232C8","#0064FF","#009696","#00C832",
        "#64FF00","#96FF00","#C8FF00","#FFFF00","#FFC800","#FFA000","#FF7D00","#E11900"] # light gray "#D3D3D3"
        clevs= [0,0.08,0.16,0.25,0.40,0.63,1,1.6,2.5,4,6.3,10,16,25,40,63,100,160]
        
    return(color_list, clevs)

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
    
def create_sparse_grid(gridSpacing, nrRows, nrCols):
    xSub = []
    ySub = []
    for i in range(0,nrRows):
        for j in range(0,nrCols):
            if ((i % gridSpacing) == 0) & ((j % gridSpacing) == 0):
                xSub.append(j)
                ySub.append(i)
                
    xSub = np.asarray(xSub)
    ySub = np.asarray(ySub)
    
    return(xSub, ySub)
    
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
    
def generate_data():
    data = np.zeros((200, 200), dtype=np.float)
    cov = np.array([[200, 100], [100, 200]])
    ij = np.random.multivariate_normal((100,100), cov, int(1e5))
    for i,j in ij:
        data[int(i), int(j)] += 1
    return data 

def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()

def intertial_axis(data):
    """Calculate the x-mean, y-mean, and cov matrix of an image."""
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_bar * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_bar * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_bar * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return x_bar, y_bar, cov

def make_lines(eigvals, eigvecs, mean, i):
        """Make lines a length of 2 stddev."""
        std = np.sqrt(eigvals[i])
        vec = 2 * std * eigvecs[:,i] / np.hypot(*eigvecs[:,i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
        
def decompose_cov_plot_bars(x_bar, y_bar, cov, ax):
    """Plot bars with a length of 2 stddev along the principal axes."""
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    ax.plot(*make_lines(eigvals, eigvecs, mean, 0), marker='o', color='white')
    ax.plot(*make_lines(eigvals, eigvecs, mean, -1), marker='o', color='white')
    ax.axis('image')
    return(eigvals,eigvecs)

def plot_bars(x_bar, y_bar, eigvals, eigvecs, ax, colour='white'):
    """Plot bars with a length of 2 stddev along the principal axes."""
    mean = np.array([x_bar, y_bar])
    ax.plot(*make_lines(eigvals, eigvecs, mean, 0), marker='o', color=colour)
    ax.plot(*make_lines(eigvals, eigvecs, mean, -1), marker='o', color=colour)
    ax.axis('image')

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def GaussianKernel(v1, v2, sigma):
    return exp(-norm(v1-v2, 2)**2/(2.*sigma**2))
    
def compute_fft_anisotropy(psd2d, fftSizeSub = -1, percentileZero = -1, rotation = True, radius = -1, sigma = -1):
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
    
    Returns
    -------
    psd2dsub : numpyarray(float)
        Output 2d array with the autocorrelation/spectrum selected on the subdomain (and rotated if asked)
    eccentricity : float
        Eccentricity of the anisotropy (major/minor axis)
    orientation : float
        Orientation of the anisotropy (degrees using trigonometrical convention, 0 degrees -> East, 90 degrees -> North, 180 degrees -> West)
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
        fftSizeSub = psd2d.shape[0]
    
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
        psd2dsubSmooth = psd2dsub # just to give a return value...
    
    ############### SHIFT ACCORDING TO PERCENTILE
    # Compute conditional percentile on smoothed spectrum/autocorrelation
    minThresholdCondition = 0.01 # Threshold to compute to conditional percentile (only values greater than this)
    if percentileZero > 0:
        percZero = np.percentile(psd2dsubSmooth[psd2dsubSmooth > minThresholdCondition], percentileZero)
    else:
        percZero = np.min(psd2dsubSmooth)
    
    # Shift spectrum/autocorrelation to start from 0 (zeros will be automatocally neglected in the computation of covariance)
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
    print("Nr. central pixels used for anisotropy estimation: ", nrNonZeroPixels)
    
    ############### COVARIANCE DECOMPOSITION
    # Find inertial axis and covariance matrix
    xbar, ybar, cov = intertial_axis(psd2dsubSmoothShifted*mask)
    
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