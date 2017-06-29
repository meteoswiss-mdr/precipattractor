#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from PIL import Image

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab

import numpy as np
import shutil
import datetime
import time
import warnings
from collections import OrderedDict

import pyfftw
from scipy import stats

import scipy.ndimage as ndimage
import pywt
from pyearth import Earth
import cv2

import getpass
usrName = getpass.getuser()

#### Import personal libraries
import time_tools_attractor as ti
import io_tools_attractor as io
import data_tools_attractor as dt
import stat_tools_attractor as st
import optical_flow as of
import maple_ree

import gis_base as gis
    
################
np.set_printoptions(precision=2)

noData = -999.0
fmt1 = "%.1f"
fmt2 = "%.2f"
fmt3 = "%.3f"
fmt4 = "%.4f"
fmt5 = "%.5f"

########SET DEFAULT ARGUMENTS##########
timeAccumMin = 5
resKm = 1 # To compute FFT frequency
inBaseDir = '/scratch/' + usrName + '/data/' # directory to read from
outBaseDir = '/store/msrad/radar/precip_attractor/data/'
fourierVar = 'dbz' # field on which to perform the fourier analysis ('rainrate' or 'dbz')
scalingBreakArray_KM = [12] #np.arange(6, 42, 2) # [15]
maxBeta1rangeKM = 512
minBeta2rangeKM = 4
fftDomainSize = 512
FFTmod = 'NUMPY' # 'FFTW' or 'NUMPY'
windowFunction = 'none' #'blackman' or 'none'

########GET ARGUMENTS FROM CMD LINE####
parser = argparse.ArgumentParser(description='Compute radar rainfall field statistics.')
parser.add_argument('-start', default='201601310600', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-end', default='201601310600', type=str,help='Ending date YYYYMMDDHHmmSS.')
parser.add_argument('-product', default='AQC', type=str,help='Which radar rainfall product to use (AQC, CPC, etc).')
parser.add_argument('-plot', default=0, type=int,help='Whether to plot the rainfall fields and the power spectra.')
parser.add_argument('-analysis', nargs='+', default=['autocorr', 'of'], type=str,help='Type of analysis to do (1d, 2d, of, autocorr, wavelets, 1dnoise, 2dnoise).')
parser.add_argument('-wols', default=0, type=int,help='Whether to use the weighted ordinary leas squares or not in the fitting of the power spectrum.')
parser.add_argument('-minR', default=0.08, type=float,help='Minimum rainfall rate for computation of WAR and various statistics.')
parser.add_argument('-format', default="netcdf", type=str,help='File format for output statistics (netcdf or csv).')
parser.add_argument('-accum', default=5, type=int,help='Accumulation time of the product [minutes].')
parser.add_argument('-temp', default=5, type=int,help='Temporal sampling of the products [minutes].')

args = parser.parse_args()

timeStartStr = args.start
timeEndStr = args.end
boolPlotting = args.plot
product = args.product
weightedOLS = args.wols
timeAccumMin = args.accum
analysis = args.analysis

if set(analysis).issubset(['1d', '2d', 'of', 'autocorr', '2d+autocorr', '1d+2d+autocorr', 'wavelets', '1dnoise', '2dnoise']) == False:
   print('You have to ask for a valid analysis [1d, 2d, of, autocorr, 2d+autocorr, 1d+2d+autocorr, wavelets, 1dnoise, 2dnoise]')
   sys.exit(1)

if type(scalingBreakArray_KM) != list and type(scalingBreakArray_KM) != np.ndarray:
    scalingBreakArray_KM = [scalingBreakArray_KM]

if len(scalingBreakArray_KM) > 1:
    variableBreak = 1
else:
    variableBreak = 0

if (timeAccumMin == 60) | (timeAccumMin == 60*24):
    timeSampMin = timeAccumMin
else:
    timeSampMin = args.temp
 
if args.format == 'netcdf':
    strFileFormat = '.nc'
elif args.format == 'csv':
    strFileFormat = '.csv'
else:
    print('File -format', args.format, ' not valid')
    sys.exit(1)
    
if (int(args.start) > int(args.end)):
    print('Time end should be after time start')
    sys.exit(1)

if (int(args.start) < 198001010000) or (int(args.start) > 203001010000):
    print('Invalid -start or -end time arguments.')
    sys.exit(1)
else:
    timeStartStr = args.start
    timeEndStr = args.end

if (product == 'AQC') or (product == 'CPC'):
    print('Computing statistics on ', args.product)
else:
    print('Invalid -product argument.')
    sys.exit(1)
    
if fourierVar == 'rainrate':
        unitsSpectrum = r"Rainfall field power $\left[ 10\mathrm{log}_{10}\left(\frac{(mm/hr)^2}{km^2}\right)\right]$"
elif fourierVar == 'dbz':
        unitsSpectrum = r"Reflectivity field power $\left[ 10\mathrm{log}_{10}\left(\frac{dBZ^2}{km^2}\right)\right]$"
                        
###################################
# Get dattime from timestamp
timeStart = ti.timestring2datetime(timeStartStr)
timeEnd = ti.timestring2datetime(timeEndStr)

timeAccumMinStr = '%05i' % timeAccumMin
timeAccum24hStr = '%05i' % (24*60)

## COLORMAPS
color_list, clevs, clevsStr = dt.get_colorlist('MeteoSwiss') #'STEPS' or 'MeteoSwiss'
cmap = colors.ListedColormap(color_list)
norm = colors.BoundaryNorm(clevs, cmap.N)
cmap.set_over('black',1)

cmapMask = colors.ListedColormap(['black'])

# Load background DEM image
dirDEM = '/users/' + usrName + '/scripts/shapefiles'
fileNameDEM = dirDEM + '/ccs4.png'
isFile = os.path.isfile(fileNameDEM)
if (isFile == False):
    print('File: ', fileNameDEM, ' not found.')
else:
    print('Reading: ', fileNameDEM)
demImg = Image.open(fileNameDEM)
demImg = dt.extract_middle_domain_img(demImg, fftDomainSize, fftDomainSize)
demImg = demImg.convert('P')

# Limits of CCS4 domain
Xmin = 255000
Xmax = 965000
Ymin = -160000
Ymax = 480000
allXcoords = np.arange(Xmin,Xmax+resKm*1000,resKm*1000)
allYcoords = np.arange(Ymin,Ymax+resKm*1000,resKm*1000)

# Set shapefile filename
fileNameShapefile = dirDEM + '/CHE_adm0.shp'
proj4stringWGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84"
proj4stringCH = "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 \
+k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs" 

#proj4stringCH = "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 \
#+k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs"

# Array containing the statistics for one single day
nrFilesDay = 24*(60/timeAccumMin)

##### LOOP OVER FILES ##########################################################

# Rainfall stack
nrValidFields = 0
stackSize = 12
rainfallStack = np.zeros((stackSize,fftDomainSize,fftDomainSize))
waveletStack = [None] * stackSize

# Flow stack
zStack = []
tStack = []
rowStack = []
colStack = []
uStack = []
vStack = []

## Daily arrays to write out
dailyStats = []
dailyU = []
dailyV = []
dailyTimesUV = []

dailyWavelets = []
dailyTimesWavelets = []

tic = time.clock()

timeLocal = timeStart
while timeLocal <= timeEnd:
    ticOneImg = time.clock()
    
    # Read in radar image into object
    timeLocalStr = ti.datetime2timestring(timeLocal)
    r = io.read_gif_image(timeLocalStr, product='AQC', minR = args.minR, fftDomainSize = 512, \
    resKm = 1, timeAccumMin = 5, inBaseDir = '/scratch/lforesti/data/', noData = -999.0, cmaptype = 'MeteoSwiss', domain = 'CCS4')
    
    hourminStr = ti.get_HHmm_str(timeLocal.hour, timeLocal.minute) # Used to write out data also when there is no valid radar file
    minWAR = 0.1
    if r.war >= minWAR:
        
        Xmin = r.extent[0]
        Xmax = r.extent[1]
        Ymin = r.extent[2]
        Ymax = r.extent[3]
        
        # Move older rainfall fields down the stack
        for s in range(0, rainfallStack.shape[0]-1):
            rainfallStack[s+1,:] = rainfallStack[s,:]
        # Add last rainfall field on top
        rainfallStack[0,:] = r.dBZFourier
        
        # Increment nr of consecutive valid rainfall fields (war >= 0.01)
        nrValidFields += 1
        
        ########### Compute velocity field ##############
        # It will be used to estimate the Lagrangian auto-correlation
        
        if (nrValidFields >= 2) and ('of' in analysis):
            print('\t')
            ticOF = time.clock()
            # extract consecutive images
            prvs = rainfallStack[1].copy()
            next = rainfallStack[0].copy()
            
            prvs *= 255.0/np.max(prvs)
            next *= 255.0/np.max(next)

            # 8-bit int
            prvs = np.ndarray.astype(prvs,'uint8')
            next = np.ndarray.astype(next,'uint8')
            
            # plt.figure()
            # plt.imshow(prvs)
            # plt.colorbar()
            # plt.show()
            
            # remove small noise with a morphological operator (opening)
            prvs = of.morphological_opening(prvs, thr=r.zerosDBZ, n=5)
            next = of.morphological_opening(next, thr=r.zerosDBZ, n=5)
            
            #+++++++++++ Optical flow parameters
            maxCornersST = 500 # Number of asked corners for Shi-Tomasi
            qualityLevelST = 0.05
            minDistanceST = 5 # Minimum distance between the detected corners
            blockSizeST = 15
            
            winsizeLK = 100 # Small windows (e.g. 10) lead to unrealistic high speeds
            nrLevelsLK = 0 # Not very sensitive parameter
            
            kernelBandwidth = 100 # Bandwidth of kernel interpolation of vectors
            
            maxSpeedKMHR = 100 # Maximum allowed speed
            nrIQRoutlier = 3 # Nr of IQR above median to consider the vector as outlier (if < 100 km/hr)
            #++++++++++++++++++++++++++++++++++++
            
            # (1b) Shi-Tomasi good features to track
            p0, nCorners = of.ShiTomasi_features_to_track(prvs, maxCornersST, qualityLevel=qualityLevelST, minDistance=minDistanceST, blockSize=blockSizeST)   
            print("Nr of points OF ShiTomasi          =", len(p0))
            
            # (2) Lucas-Kanade tracking
            col, row, u, v, err = of.LucasKanade_features_tracking(prvs, next, p0, winSize=(winsizeLK,winsizeLK), maxLevel=nrLevelsLK)
            
            # (3) exclude outliers   
            speed = np.sqrt(u**2 + v**2)
            q1, q2, q3 = np.percentile(speed, [25,50,75])
            maxspeed = np.min((maxSpeedKMHR/12, q2 + nrIQRoutlier*(q3 - q1)))
            minspeed = np.max((0,q2 - 2*(q3 - q1)))
            keep = (speed <= maxspeed) # & (speed >= minspeed)
            
            print('Max speed       =',np.max(speed)*12)
            print('Median speed    =',np.percentile(speed,50)*12)
            print('Speed threshold =',maxspeed*12)
            
            # Plot histogram of speeds
            # plt.close()
            # plt.hist(speed*12, bins=30)
            # plt.title('min = %1.1f, max = %1.1f' % (minspeed*12,maxspeed*12))
            # plt.axvline(x=maxspeed*12)
            # plt.xlabel('Speed [km/hr]')
            # plt.show()
            
            u = u[keep].reshape(np.sum(keep),1)
            v = v[keep].reshape(np.sum(keep),1)
            row = row[keep].reshape(np.sum(keep),1)
            col = col[keep].reshape(np.sum(keep),1)
            
            # (4) stack vectors within time window
            rowStack.append(row)
            colStack.append(col)
            uStack.append(u)
            vStack.append(v)
        
            # convert lists of arrays into single arrays
            row = np.vstack(rowStack)
            col = np.vstack(colStack) 
            u = np.vstack(uStack)
            v = np.vstack(vStack)
            
            if (nrValidFields >= 4):
                colStack.pop(0)
                rowStack.pop(0)
                uStack.pop(0)
                vStack.pop(0)
            
            # (1) decluster sparse motion vectors
            col, row, u, v = of.declustering(col, row, u, v, R = 20, minN = 3)
            print("Nr of points OF after declustering =", len(row))
            
            # (2) kernel interpolation
            domainSize = [fftDomainSize, fftDomainSize]
            colgrid, rowgrid, U, V, b = of.interpolate_sparse_vectors_kernel(col, row, u, v, domainSize, b = kernelBandwidth)
            print('Kernel bandwith =',b)
            
            # Add U,V fields to daily collection
            dailyU.append(U)
            dailyV.append(-V) # Reverse V orientation (South -> North)
            dailyTimesUV.append(timeLocalStr)
            
            # Compute advection
            # resize motion fields by factor f (for advection)
            f = 0.5
            if (f<1):
                Ures = cv2.resize(U, (0,0), fx=f, fy=f)
                Vres = cv2.resize(V, (0,0), fx=f, fy=f) 
            else:
                Ures = U
                Vres = V
            
            tocOF = time.clock()
            
            # Call MAPLE routine for advection
            net = 1
            rainfield_lag1 = maple_ree.ree_epol_slio(rainfallStack[1], Vres, Ures, net)
            
            # Call MAPLE routine for advection over several time stamps
            # net = np.min([12, nrValidFields])
            # for lag in range(2,net):
            # rainfield_advected = maple_ree.ree_epol_slio(rainfallStack[2], Vres, Ures, net)
            
            # plt.close()
            # plt.subplot(121)
            # plt.imshow(rainfallStack[1], vmin=8, vmax=55)
            # plt.subplot(122)
            # plt.imshow(rainfield_lag1[:,:,-1], vmin=8, vmax=55)
            # plt.show()
            # sys.exit()
            
            # Resize vector fields for plotting
            xs, ys, Us, Vs = of.reduce_field_density_for_plotting(colgrid, rowgrid, U, V, 25)
            
            # Plot vectors to check if correct
            # plt.quiver(xs, ys, Us, Vs)
            # plt.show()
                
            print('Elapsed time OF: ', tocOF - ticOF, ' seconds.')
            print('\t')
            
        ########### Compute Wavelet transform ###########
        if 'wavelets' in analysis:
            wavelet = 'haar'
            w = pywt.Wavelet(wavelet)
            #print(w)
            
            # Upscale field in rainrate
            wavelet_coeff = st.wavelet_decomposition_2d(r.rainrate, wavelet, nrLevels = None)
            
            # Transform into dBZ
            for level in range(0,len(wavelet_coeff)):
                wavelet_coeff[level],_,_ = dt.rainrate2reflectivity(wavelet_coeff[level])
            
            # Generate coordinates of centers of wavelet coefficients
            xvecs, yvecs = st.generate_wavelet_coordinates(wavelet_coeff, r.dBZFourier.shape, Xmin, Xmax, Ymin, Ymax, resKm*1000)
            
            # Append a given wavelet scale to write out into daily netCDF files
            scaleKm_asked = 8
            scale2keep = st.get_level_from_scale(resKm, scaleKm_asked)

            scaleKm = xvecs[scale2keep][1] - xvecs[scale2keep][0]
            scaleKm = int(scaleKm/1000)
            if scaleKm_asked != scaleKm:
                print('Asked and returned wavelet scales not matching.', scaleKm_asked, 'vs', scaleKm)
                sys.exit()
            else:
                print('Wavelet scale = ', scaleKm, 'km')
            
            dailyWavelets.append(wavelet_coeff[scale2keep])
            dailyTimesWavelets.append(timeLocalStr)
            
            # # Write out wavelet coefficients to netCDF file
            # # Keep only large scales (smaller file size) 
            # wavelet_coeff_image = wavelet_coeff[1:]
            
            # analysisType = 'WAVELET'
            # fileNameWavelet,_,_ = io.get_filename_stats(inBaseDir, analysisType, timeLocal, product, \
            # timeAccumMin=timeAccumMin, quality=0, format='netcdf')
            
            # io.write_netcdf_waveletcoeffs(fileNameWavelet, timeLocalStr, \
            # xvecs, yvecs, wavelet_coeff_image, waveletType = wavelet)
            # print('Saved:', fileNameWavelet)
            
            ## Add wavelet coeffs to the stack
            for s in range(0, len(waveletStack)-1):
                waveletStack[s+1] = waveletStack[s]
            waveletStack[0] = wavelet_coeff
            
            # # Full wavelet decomposition to get also the HDV residual components
            waveletHVD = False
            nrLevels = 6
            if waveletHVD:
                coeffs = pywt.wavedec2(r.dBZFourier, w, level=nrLevels)
                #cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
                cA2 = coeffs[0]
            
            # ###### Use wavelets to generate a field of correlated noise
            waveletNoise = False
            level2perturb = [3,4,5]
            nrMembers = 3
            if waveletNoise:
                # Generate white noise at a given level
                stochasticEnsemble = st.generate_wavelet_noise(r.dBZFourier, w, nrLevels, level2perturb, nrMembers)
        
        ########### Compute Fourier power spectrum ###########
        ticFFT = time.clock()
        minFieldSize = np.min(fftDomainSize)
        
        # Replace zeros with the lowest rainfall threshold (to obtain better beta2 estimations)
        if fourierVar == 'rainrate':
            rainfieldFourier = r.rainrate
            rainfieldFourier[rainfieldFourier < args.minR] = args.minR
        if fourierVar == 'dbz':
            rainfieldFourier = r.dBZFourier
            zerosDBZ,_,_ = dt.rainrate2reflectivity(args.minR)
            
            # Method 1: Set the zeros to the dBZ threshold
            # rainfieldFourier[rainfieldFourier < zerosDBZ] = zerosDBZ
            # Method 2: Remove the dBZ threshold to all data
            rainfieldFourier = rainfieldFourier - zerosDBZ
            
            # plt.imshow(rainfieldFourier)
            # plt.colorbar()
            # plt.show()
            
        # Compute 2D power spectrum
        psd2d, freqAll = st.compute_2d_spectrum(rainfieldFourier, resolution=resKm, window=None, FFTmod='NUMPY')
        
        # Compute autocorrelation using inverse FFT of spectrum
        if ('autocorr' in analysis) or ('1d' in analysis) or ('2d+autocorr' in analysis) or ('1d+2d+autocorr' in analysis) or ('wavelets' in analysis):
            # Compute autocorrelation
            autocorr,_,_,_ = st.compute_autocorrelation_fft2(rainfieldFourier, FFTmod = 'NUMPY')
            
            # Compute anisotropy from autocorrelation function
            autocorrSizeSub = 255
            percentileZero = 90
            autocorrSub, eccentricity_autocorr, orientation_autocorr, xbar_autocorr, ybar_autocorr, eigvals_autocorr, eigvecs_autocorr, percZero_autocorr,_ = st.compute_fft_anisotropy(autocorr, autocorrSizeSub, percentileZero, rotation=False)

        if ('2d' in analysis) or ('2d+autocorr' in analysis) or ('1d+2d+autocorr' in analysis) or ('wavelets' in analysis):
            cov2logPS = True # Whether to compute the anisotropy on the log of the 2d PS
            # Extract central region of 2d power spectrum and compute covariance
            if cov2logPS:
                psd2d_anis = 10.0*np.log10(psd2d)
            else:
                psd2d_anis = np.copy(psd2d)
            
            # Compute anisotropy from FFT spectrum
            fftSizeSub = 40#255
            percentileZero = 90
            smoothing_sigma = 3
            psd2dsub, eccentricity_ps, orientation_ps, xbar_ps, ybar_ps, eigvals_ps, eigvecs_ps, percZero_ps, psd2dsubSmooth = st.compute_fft_anisotropy(psd2d_anis, fftSizeSub, percentileZero, sigma = smoothing_sigma)
        
            print(percentileZero,'- percentile = ', percZero_ps)
        
        # Compute 1D radially averaged power spectrum
        psd1d, freq, wavelengthKm = st.compute_radialAverage_spectrum(psd2d, resolution=resKm)
        
        ############ Compute spectral slopes Beta
        r_beta1_best = 0
        r_beta2_best = 0
        for s in range(0,len(scalingBreakArray_KM)):
            scalingBreak_KM = scalingBreakArray_KM[s]
            largeScalesLims = np.array([maxBeta1rangeKM, scalingBreak_KM])
            smallScalesLims = np.array([scalingBreak_KM, minBeta2rangeKM])
            idxBeta1 = (wavelengthKm <= largeScalesLims[0]) & (wavelengthKm > largeScalesLims[1]) # large scales
            idxBeta2 = (wavelengthKm <= smallScalesLims[0]) & (wavelengthKm > smallScalesLims[1]) # small scales
            idxBetaBoth = (wavelengthKm <= largeScalesLims[0]) & (wavelengthKm > smallScalesLims[1]) # all scales
            
            #print('Nr points beta1 = ', np.sum(idxBeta1))
            #print('Nr points beta2 = ', np.sum(idxBeta2))
            #io.write_csv('/users/' + usrName + '/results/ps_marco.csv', ['freq','psd'], np.asarray([freq,psd1d]).T.tolist())
            
            # Compute betas using OLS
            if weightedOLS == 0:
                beta1, intercept_beta1, r_beta1 = st.compute_beta_sm(10*np.log10(freq[idxBeta1]),10*np.log10(psd1d[idxBeta1]))          
                beta2, intercept_beta2, r_beta2  = st.compute_beta_sm(10*np.log10(freq[idxBeta2]), 10*np.log10(psd1d[idxBeta2]))
            elif weightedOLS == 1:
                # Compute betas using weighted OLS
                linWeights = len(freq[idxBeta1]) - np.arange(len(freq[idxBeta1]))
                #logWeights = 10*np.log10(linWeights)
                logWeights = linWeights
                beta1, intercept_beta1,r_beta1  = st.compute_beta_sm(10*np.log10(freq[idxBeta1]), 10*np.log10(psd1d[idxBeta1]), logWeights)
                
                linWeights = len(freq[idxBeta2]) - np.arange(len(freq[idxBeta2]))
                #logWeights = 10*np.log10(linWeights)
                logWeights = linWeights
                beta2, intercept_beta2, r_beta2  = st.compute_beta_sm(10*np.log10(freq[idxBeta2]), 10*np.log10(psd1d[idxBeta2]), logWeights)
            else:
                print("Please set weightedOLS either to 0 or 1")
                sys.exit(1)
            
            # Select best fit based on scaling break                   
            if np.abs(r_beta1 + r_beta2) > np.abs(r_beta1_best + r_beta2_best):
                r_beta1_best = r_beta1
                r_beta2_best = r_beta2
                beta1_best = beta1
                intercept_beta1_best = intercept_beta1
                beta2_best = beta2
                intercept_beta2_best = intercept_beta2
                scalingBreak_best = scalingBreak_KM
                smallScalesLims_best = smallScalesLims
                largeScalesLims_best = largeScalesLims
                scalingBreak_Idx = idxBeta2[0]
                
        r_beta1 = r_beta1_best
        r_beta2 = r_beta2_best
        beta1 = beta1_best
        beta2 = beta2_best
        intercept_beta1 = intercept_beta1_best
        intercept_beta2 = intercept_beta2_best
        smallScalesLims = smallScalesLims_best
        largeScalesLims = largeScalesLims_best
        
        if variableBreak == 1:
            print("Best scaling break corr. = ", scalingBreak_best, ' km')
        else:
            print("Fixed scaling break = ", scalingBreak_best, ' km')
            
        #### Fitting spectral slopes with MARS (Multivariate Adaptive Regression Splines)
        useMARS = False
        if useMARS:
            model = Earth(max_degree = 1, max_terms = 2)
            model.fit(dt.to_dB(freq[idxBetaBoth]), dt.to_dB(psd1d[idxBetaBoth]))
            mars_fit = model.predict(dt.to_dB(freq[idxBetaBoth]))
            
            # plt.scatter(dt.to_dB(freq),dt.to_dB(psd1d))
            # plt.plot(dt.to_dB(freq[idxBetaBoth]), mars_fit)
            # plt.show()
            
            # print(model.trace())
            # print(model.summary())
            # print(model.basis_)
            # print(model.coef_[0])
            #y_prime_hat = model.predict_deriv(dt.to_dB(freq[idxBetaBoth]), 'x6')
            scalingBreak_MARS = str(model.basis_[2])[2:7]
            scalingBreak_MARS_KM = 1.0/dt.from_dB(float(scalingBreak_MARS))
            print("Best scaling break MARS = ", scalingBreak_MARS_KM, ' km')
        
        tocFFT = time.clock()
        #print('FFT time: ', tocFFT-ticFFT, ' seconds.')
        
        ##################### COMPUTE SUMMARY STATS #####################################
        # Compute field statistics in rainfall units
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            rainmean = np.nanmean(r.rainrate.ravel())
            rainstd = np.nanstd(r.rainrate.ravel())
            raincondmean = np.nanmean(r.rainrateNans.ravel())
            raincondstd = np.nanstd(r.rainrateNans.ravel())
            
            # Compute field statistics in dBZ units
            dBZmean = np.nanmean(r.dBZ.ravel())
            dBZstd = np.nanstd(r.dBZ.ravel())
            dBZcondmean = np.nanmean(r.dBZNans.ravel())
            dBZcondstd = np.nanstd(r.dBZNans.ravel())
        
        # Compute Eulerian Auto-correlation 
        if (nrValidFields >= 2) and ('of' in analysis):
            corr_eul_lag1 = np.corrcoef(rainfallStack[0,:].flatten(), rainfallStack[1,:].flatten())
            corr_eul_lag1 = corr_eul_lag1[0,1]
            print("Eulerian correlation       =", fmt3 % corr_eul_lag1)
            
            # Compute Eulerian correlation at each wavelet coeff level
            # corr_eul_wavelet_levels = []
            # for level in range(0,len(wavelet_coeff)):
                # corr_eul_level = np.corrcoef(np.array(waveletStack[0][level]).flatten(), np.array(waveletStack[1][level]).flatten())
                # corr_eul_level = corr_eul_level[0,1]
                # corr_eul_wavelet_levels.append(corr_eul_level)
            # print(corr_eul_wavelet_levels)
            # plt.figure()
            # plt.scatter(rainfallStack[0,:], rainfallStack[1,:])
            # plt.show()
        else:
            corr_eul_lag1 = np.nan
        
        # Compute Lagrangian auto-correlation
        if (nrValidFields >= 2) and ('of' in analysis):
            corr_lagr_lag1 = np.corrcoef(rainfield_lag1.flatten(), rainfallStack[0,:].flatten())
            corr_lagr_lag1 = corr_lagr_lag1[0,1]
            print("Lagrangian correlation     =", fmt3 % corr_lagr_lag1)
            print("Diff. Lagr-Eul correlation =", fmt3 % (corr_lagr_lag1 - corr_eul_lag1))
            # plt.figure()
            # plt.scatter(rainfallStack[0,:], rainfallStack[1,:])
            # plt.show()
            corr_lagr_lags = []
            for lag in range(1,net):
                corr_lagr = np.corrcoef(rainfield_advected[lag].flatten(), rainfallStack[0,:].flatten())
                corr_lagr_lags.append(corr_lagr[0,1])
            print('Lagrangian correlation lags =', corr_lagr_lags)
        else:
            corr_lagr_lag1 = np.nan
        
        ################### COLLECT DAILY STATS 
        timeStampStr = ti.datetime2timestring(timeLocal)
        
        # Headers
        headers = ['time', 'alb', 'doe', 'mle', 'ppm', 'wei', 'war', 'r_mean', 'r_std', 'r_cmean', 'r_cstd',
        'dBZ_mean', 'dBZ_std', 'dBZ_cmean', 'dBZ_cstd', 
        'beta1', 'corr_beta1', 'beta2', 'corr_beta2' , 'scaling_break', 'eccentricity', 'orientation',
        'corr_eul_lag1', 'corr_lagr_lag1']
        
        if '2d' in analysis:
            eccentricity = eccentricity_ps
            orientation = orientation_ps
        else:
            eccentricity = eccentricity_autocorr
            orientation = orientation_autocorr
            
        # Data
        instantStats = [timeStampStr,
        str(r.alb), 
        str(r.dol), 
        str(r.lem),
        str(r.ppm),
        str(r.wei),             
        fmt4 % r.war,
        fmt5 % rainmean, 
        fmt5 % rainstd,
        fmt5 % raincondmean, 
        fmt5 % raincondstd,        
        fmt4 % dBZmean, 
        fmt4 % dBZstd,        
        fmt4 % dBZcondmean, 
        fmt4 % dBZcondstd,
        fmt4 % beta1,
        fmt4 % r_beta1,
        fmt4 % beta2,
        fmt4 % r_beta2,
        int(scalingBreak_best),
        fmt4 % eccentricity,
        fmt4 % orientation,
        fmt4 % corr_eul_lag1,
        fmt4 % corr_lagr_lag1
        ]
        print('+++++++ Radar statistics +++++++')
        outputPrint = OrderedDict(zip(headers, instantStats))
        print(outputPrint)
        print('++++++++++++++++++++++++++++++++')
        
        # Append statistics to daily array
        dailyStats.append(instantStats)
        
        ######################## PLOT WAVELETS ######################
        if 'wavelets' in analysis and boolPlotting:
            
            if waveletNoise:
                nrRows,nrCols = dt.optimal_size_subplot(nrMembers+1)
                # Adjust figure parameters
                ratioFig = nrCols/nrRows
                figWidth = 14
                colorbar = 'off'
                fig = plt.figure(figsize=(ratioFig*figWidth,figWidth))
                padding = 0.01
                plt.subplots_adjust(hspace=0.05, wspace=0.01)
                mpl.rcParams['image.interpolation'] = 'nearest'

                # Plot rainfield
                plt.subplot(nrRows, nrCols, 1)
                PC = plt.imshow(r.dBZFourier, vmin=15, vmax=45)
                plt.title('Rainfield [dBZ]',fontsize=15)
                plt.axis('off')
                
                # Plot stochastic ensemble
                for member in range(0, nrMembers):
                    plt.subplot(nrRows, nrCols, member+2)
                    plt.imshow(stochasticEnsemble[member],vmin=15, vmax=45)
                    plt.title('Member '+ str(member+1), fontsize=15)
                    plt.axis('off')
                plt.suptitle('Stochastic ensemble based on wavelet type: ' + wavelet + '\n by perturbing levels ' + str(level2perturb), fontsize=20)
                
                stringFigName = '/users/lforesti/results/' + product + r.yearStr + r.julianDayStr + r.hourminStr + '-' + wavelet + '-waveletEnsemble_' + timeAccumMinStr + '.png'
                plt.savefig(stringFigName, dpi=300)
                print(stringFigName, ' saved.')
                plt.close()
        
            if waveletHVD:
                # Plot of all the horizontal, diagonal and vertical components of the wavelet transform
                pltWavelets = ['H','V','D']
                nrPlots = (len(coeffs)-1)*len(pltWavelets)+2
                mpl.rcParams['image.interpolation'] = 'none'
                
                nrRows,nrCols = dt.optimal_size_subplot(nrPlots)
                print('Nr. plots = ' + str(nrPlots), ' in ', str(nrRows), 'x', str(nrCols))
                
                # Adjust figure parameters
                ratioFig = nrCols/nrRows
                figWidth = 14
                colorbar = 'off'
                fig = plt.figure(figsize=(ratioFig*figWidth,figWidth))
                padding = 0.01
                plt.subplots_adjust(hspace=0.05, wspace=0.01)
                ###
                
                # Plot rainfield
                ax1 = plt.subplot(nrRows, nrCols, 1)
                PC = plt.imshow(r.dBZFourier, vmin=15, vmax=45)
                    
                plt.title('Rainfield [dBZ]')
                plt.axis('off')
                
                # Colorbar
                if colorbar == 'on':
                    divider = make_axes_locatable(ax1)
                    cax1 = divider.append_axes("right", size="5%", pad=padding)
                    cbar = plt.colorbar(PC, cax = cax1)
                
                nplot = 2
                for level in range(1,nrLevels+1):   
                    for p in range(0,len(pltWavelets)):
                        waveletLevel = nrLevels+1 - level
                        
                        # Plot wavelet coefficients for horizontal/vertical/diagonal components
                        var = coeffs[waveletLevel][p]
                        minimum = np.percentile(var, 1)
                        maximum = np.percentile(var, 99)
                        
                        ax1 = plt.subplot(nrRows, nrCols, nplot)
                        PC = plt.imshow(var, vmin=minimum, vmax=maximum, aspect=var.shape[1]/var.shape[0])

                        if p == 0:
                            titleStr = 'Level ' + str(level) + ' - horizontal'
                        if p == 1:
                            titleStr = 'Level ' + str(level) + ' - vertical'
                        if p == 2:
                            titleStr = 'Level ' + str(level) + ' - diagonal'
                        plt.title(titleStr)
                        plt.axis('off')
                        
                        # Colorbar
                        if colorbar == 'on':
                            divider = make_axes_locatable(ax1)
                            cax1 = divider.append_axes("right", size="5%", pad=padding)
                            cbar = plt.colorbar(PC, cax = cax1)

                        nplot = nplot + 1
                
                # Plot approximation at last scale
                minimum = np.percentile(cA2, 1)
                maximum = np.percentile(cA2, 99)

                ax1 = plt.subplot(nrRows, nrCols, nplot)
                PC = plt.imshow(cA2, aspect=cA2.shape[1]/cA2.shape[0])

                plt.title('Approximation')
                plt.axis('off')
                
                # Colorbar
                if colorbar == 'on':
                    divider = make_axes_locatable(ax1)
                    cax1 = divider.append_axes("right", size="5%", pad=padding)
                    cbar = plt.colorbar(PC, cax = cax1)

                plt.suptitle('Wavelet type: ' + wavelet, fontsize=20)
                #plt.show()
                waveletDirs = "".join(pltWavelets)
                stringFigName = '/users/lforesti/results/' + product + r.yearStr + r.julianDayStr \
                + r.hourminStr + '-' + wavelet + '-wavelet_' + waveletDirs + '_' + timeAccumMinStr + '.png'
                plt.savefig(stringFigName, dpi=300)
                print(stringFigName, ' saved.')
            
            ###### Plots of the wavelet approximation at each scale
            nrPlots = len(wavelet_coeff)
            nrRows,nrCols = dt.optimal_size_subplot(nrPlots)
            fig = plt.figure()
            ax = fig.add_axes()
            ax = fig.add_subplot(111)

            for scale in range(1,nrPlots+1):
                plt.subplot(nrRows, nrCols, scale)
                im = plt.imshow(wavelet_coeff[scale-1], vmin=r.dbzThreshold, vmax=50, interpolation='nearest')

                if scale == nrPlots:
                    scaleKm_l = (xvecs[scale-2][1] - xvecs[scale-2][0])*2
                else:
                    scaleKm_l = xvecs[scale-1][1] - xvecs[scale-1][0]
                scaleKm_l = int(scaleKm/1000)
                titleStr = 'Scale = ' + str(scaleKm_l) + ' km'
                plt.title(titleStr, fontsize=12)
                plt.axis('off')
            
            fig.tight_layout()
            fig.subplots_adjust(top=0.92, right=0.8)
            cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            plt.suptitle('Low pass wavelet decomposition', fontsize=15)
            
            stringFigName = '/users/lforesti/results/' + product + r.yearStr + r.julianDayStr \
            + r.hourminStr + '-' + wavelet + '-waveletApprox_' + timeAccumMinStr + '.png'
            plt.savefig(stringFigName, dpi=300)
            print(stringFigName, ' saved.')
            
        ################ PLOTTING RAINFIELD #################################
        # ++++++++++++
        if boolPlotting:
            titlesSize = 20
            labelsSize = 18
            ticksSize = 16
            unitsSize=14
            colorbarTicksSize=14
            mpl.rcParams['xtick.labelsize'] = ticksSize 
            mpl.rcParams['ytick.labelsize'] = ticksSize 
            
            plt.close("all")

            analysisFFT = []
            for i in range(0,len(analysis)):
                if (analysis[i] == '1d') or (analysis[i] == '2d') or (analysis[i] == 'autocorr') or (analysis[i] == '1d+2d+autocorr') or (analysis[i] == '2dnoise') or (analysis[i] == '2d+autocorr'):
                    analysisFFT.append(analysis[i])
            
            # Loop over different analyses (1d, 2d autocorr)               
            for an in analysisFFT:
                if an == '1d+2d+autocorr':
                    fig = plt.figure(figsize=(18,18))
                elif an == '2d+autocorr':
                    fig = plt.figure(figsize=(8.3,20))
                else:
                    fig = plt.figure(figsize=(16,7.5))
                
                ax = fig.add_axes()
                ax = fig.add_subplot(111)
                
                if an == '1d+2d+autocorr':
                    rainAx = plt.subplot(221)
                elif an == '2d+autocorr':
                    rainAx = plt.subplot(311)
                else:
                    rainAx = plt.subplot(121)
                    
                # Draw DEM
                rainAx.imshow(demImg, extent = r.extent, vmin=100, vmax=3000, cmap = plt.get_cmap('gray'))
                
                # Draw rainfield
                rainIm = rainAx.imshow(r.rainrateNans, extent = r.extent, cmap=cmap, norm=norm, interpolation='nearest')
                
                # Draw shapefile
                gis.read_plot_shapefile(fileNameShapefile, proj4stringWGS84, proj4stringCH,  ax=rainAx, linewidth = 0.75)
                
                if (nrValidFields >= 2) and ('of' in analysis):
                    ycoord_flipped = fftDomainSize-1-ys
                    plt.quiver(Xmin+xs*1000, Ymin+ycoord_flipped*1000, Us, -Vs, angles = 'xy', scale_units='xy')
                    #plt.quiver(Xmin+x*1000, Ymin+ycoord_flipped*1000, u, -v, angles = 'xy', scale_units='xy')
                # Colorbar
                cbar = plt.colorbar(rainIm, ticks=clevs, spacing='uniform', norm=norm, extend='max', fraction=0.04)
                cbar.ax.tick_params(labelsize=colorbarTicksSize)
                cbar.set_ticklabels(clevsStr, update_ticks=True)
                if (timeAccumMin == 1440):
                    cbar.ax.set_title("   mm/day",fontsize=unitsSize)
                elif (timeAccumMin == 60):
                    cbar.ax.set_title("   mm/h",fontsize=unitsSize)    
                elif (timeAccumMin == 5):
                    if an == '2d+autocorr':
                        cbar.set_label(r"mm h$^{-1}$",fontsize=unitsSize)
                    else:
                        cbar.ax.set_title(r"   mm hr$^{-1}$",fontsize=unitsSize)
                else:
                    print('Accum. units not defined.')
                #cbar.ax.xaxis.set_label_position('top')

                
                # # Set ticks for dBZ on the other side
                # ax2 =plt.twinx(ax=cbar.ax)
                # dBZlimits,_,_ = dt.rainrate2reflectivity(clevs,A,b)
                # dBZlimits = np.round(dBZlimits)
                # ax2.set_ylim(-10, 10)
                # ax2.set_yticklabels(dBZlimits)
                
                titleStr = timeLocal.strftime("%Y.%m.%d %H:%M") + ', ' + product + ' rainfall field, Q' + str(r.dataQuality)
                titleStr = 'Radar rainfall field on ' + timeLocal.strftime("%Y.%m.%d %H:%M")
                plt.title(titleStr, fontsize=titlesSize)
                
                # Draw radar composite mask
                rainAx.imshow(r.mask, cmap=r.cmapMask, extent = r.extent, alpha = 0.5)
                
                # Add product quality within image
                dataQualityTxt = "Quality = " + str(r.dataQuality)
                
                if (an == 'of'):
                    plt.text(-0.15,-0.12, "Eulerian      correlation = " + fmt3 % corr_eul_lag1, transform=rainAx.transAxes)
                    plt.text(-0.15,-0.15, "Lagrangian correlation = " + fmt3 % corr_lagr_lag1, transform=rainAx.transAxes)
                    diffPercEulLagr = (corr_lagr_lag1 - corr_eul_lag1)*100
                    plt.text(-0.15,-0.18, "Difference Lagr/Eul      = " + fmt2 % diffPercEulLagr + ' %', transform=rainAx.transAxes)
                
                # Set X and Y ticks for coordinates
                xticks = np.arange(400, 900, 100)
                yticks = np.arange(0, 500 ,100)
                plt.xticks(xticks*1000, xticks)
                plt.yticks(yticks*1000, yticks)
                plt.xlabel('Swiss Easting [km]', fontsize=labelsSize)
                plt.ylabel('Swiss Northing [km]', fontsize=labelsSize)
                
                #################### PLOT SPECTRA ###########################################################
                #++++++++++++ Draw 2d power spectrum
                if (an == '2d') | (an == '2dnoise') | (an == '2d+autocorr') | (an == '1d+2d+autocorr'):
                    if an == '1d+2d+autocorr':
                        psAx2 = plt.subplot(222)
                    elif an == '2d+autocorr':
                        psAx2 = plt.subplot(312)
                    else:
                        psAx2 = plt.subplot(122)

                    if fourierVar == 'rainrate':
                        psLims =[-50,40]
                    if fourierVar == 'dbz':
                        psLims = [-20,70]
                        
                    extentFFT = (-minFieldSize/2,minFieldSize/2,-minFieldSize/2,minFieldSize/2)
                    if (an == '2d') | (an == '2d+autocorr') | (an == '1d+2d+autocorr'):
                        # Smooth 2d PS for plotting contours
                        if cov2logPS == False:
                            psd2dsubSmooth = 10.0*np.log10(psd2dsubSmooth)

                        # Plot image of 2d PS
                        #psAx2.invert_yaxis()
                        clevsPS = np.arange(-5,70,5)
                        cmapPS = plt.get_cmap('nipy_spectral', clevsPS.shape[0]) #nipy_spectral, gist_ncar
                        normPS = colors.BoundaryNorm(clevsPS, cmapPS.N-1)
                        cmapPS.set_over('white',1)
                        
                        # Compute alpha transparency vector
                        #cmapPS._init()
                        #cmapPS._lut[clevsPS <= percZero,-1] = 0.5
                        
                        if cov2logPS:
                            imPS = psAx2.imshow(psd2dsub, interpolation='nearest', cmap=cmapPS, norm=normPS)
                        else:
                            imPS = psAx2.imshow(10.0*np.log10(psd2dsub), interpolation='nearest', cmap=cmapPS, norm=normPS)
                        
                        # Plot smooth contour of 2d PS
                        # percentiles = [70,80,90,95,98,99,99.5]
                        # levelsPS = np.array(st.percentiles(psd2dsubSmooth, percentiles))
                        # print("Contour levels quantiles: ",percentiles)
                        # print("Contour levels 2d PS    : ", levelsPS)
                        # if np.sum(levelsPS) != 0:
                            # im1 = psAx2.contour(psd2dsubSmooth, levelsPS, colors='black', alpha=0.25)
                            # im1 = psAx2.contour(psd2dsubSmooth, [percZero], colors='black', linestyles='dashed')
                        
                        # Plot major and minor axis of anisotropy
                        #st.plot_bars(xbar_ps, ybar_ps, eigvals_ps, eigvecs_ps, psAx2, 'red')
                        
                        #plt.text(0.05, 0.95, 'eccentricity = ' + str(fmt2 % eccentricity_ps), transform=psAx2.transAxes, backgroundcolor = 'w', fontsize=14)
                        #plt.text(0.05, 0.90, 'orientation = ' + str(fmt2 % orientation_ps) + '$^\circ$', transform=psAx2.transAxes,backgroundcolor = 'w', fontsize=14)
                        
                        # Create ticks in km
                        ticks_loc = np.arange(0,2*fftSizeSub,1)
                        
                        # List of ticks for X and Y (reference from top)
                        ticksListX = np.hstack((np.flipud(-resKm/freq[1:fftSizeSub+1]),0,resKm/freq[1:fftSizeSub])).astype(int)
                        ticksListY = np.flipud(ticksListX)
                        
                        # List of indices where to display the ticks
                        if fftSizeSub <= 20:
                            idxTicksX = np.hstack((np.arange(0,fftSizeSub-1,2),fftSizeSub-1,fftSizeSub+1,np.arange(fftSizeSub+2,2*fftSizeSub,2))).astype(int)
                            idxTicksY = np.hstack((np.arange(1,fftSizeSub-2,2),fftSizeSub-2,fftSizeSub,np.arange(fftSizeSub+1,2*fftSizeSub,2))).astype(int)
                        else:
                            idxTicksX = np.hstack((np.arange(1,fftSizeSub-2,4),fftSizeSub-1,fftSizeSub+1,np.arange(fftSizeSub+3,2*fftSizeSub,4))).astype(int)
                            idxTicksY = np.hstack((np.arange(0,fftSizeSub-3,4),fftSizeSub-2,fftSizeSub,np.arange(fftSizeSub+2,2*fftSizeSub,4))).astype(int)
                        
                        plt.xticks(rotation=90)
                        psAx2.set_xticks(ticks_loc[idxTicksX])
                        psAx2.set_xticklabels(ticksListX[idxTicksX], fontsize=13)
                        psAx2.set_yticks(ticks_loc[idxTicksY])
                        psAx2.set_yticklabels(ticksListY[idxTicksY], fontsize=13)

                        plt.xlabel('Wavelength [km]', fontsize=labelsSize)
                        plt.ylabel('Wavelength [km]', fontsize=labelsSize)
                        
                        #plt.gca().invert_yaxis()
                    else:
                        #plt.contourf(10*np.log10(psd2dnoise), 20, vmin=-15, vmax=0)
                        
                        imPS = plt.imshow(10*np.log10(psd2dnoise), extent=(extentFFT[0], extentFFT[1], extentFFT[2], extentFFT[3]), vmin=-15, vmax=0)
                        plt.gca().invert_yaxis()
                    cbar = plt.colorbar(imPS, ticks=clevsPS, spacing='uniform', norm=normPS, extend='max', fraction=0.04)
                    cbar.ax.tick_params(labelsize=colorbarTicksSize)
                    cbar.set_label(unitsSpectrum, fontsize=unitsSize)
                    #cbar.ax.set_title(unitsSpectrum, fontsize=unitsSize)
                    titleStr = '2D power spectrum (rotated by 90$^\circ$)'
                    plt.title(titleStr, fontsize=titlesSize)
                
                #++++++++++++ Draw autocorrelation function
                if (an == 'autocorr') | (an == '2d+autocorr') | (an == '1d+2d+autocorr'):
                    if an == '1d+2d+autocorr':
                        autocorrAx = plt.subplot(223)
                    elif an == '2d+autocorr':
                        autocorrAx = plt.subplot(313)
                    else:
                        autocorrAx = plt.subplot(122)
                    
                    maxAutocov = np.max(autocorrSub)
                    if maxAutocov > 50:
                        clevsPS = np.arange(0,200,10)
                    elif maxAutocov > 10:
                        clevsPS = np.arange(0,50,5)
                    else:
                        clevsPS = np.arange(-0.05,1.05,0.05)
                        clevsPSticks = np.arange(-0.1,1.1,0.1)
                    cmapPS = plt.get_cmap('nipy_spectral', clevsPS.shape[0]) #nipy_spectral, gist_ncar
                    normPS = colors.BoundaryNorm(clevsPS, cmapPS.N)
                    cmaplist = [cmapPS(i) for i in range(cmapPS.N)]
                    # force the first color entry to be white
                    #cmaplist[0] = (1,1,1,1.0)
                    
                    # Create the new map
                    cmapPS = cmapPS.from_list('Custom cmap', cmaplist, cmapPS.N)
                    cmapPS.set_under('white',1)
                    
                    ext = (-autocorrSizeSub, autocorrSizeSub, -autocorrSizeSub, autocorrSizeSub)
                    imAC = autocorrAx.imshow(autocorrSub, cmap=cmapPS, norm=normPS, extent = ext)
                    #cbar = plt.colorbar(imAC, ticks=clevsPS, spacing='uniform', norm=normPS, extend='max', fraction=0.03)
                    cbar = plt.colorbar(imAC, ticks=clevsPSticks, spacing='uniform', extend='min', norm=normPS,fraction=0.04)
                    cbar.ax.tick_params(labelsize=colorbarTicksSize)
                    cbar.set_label('correlation coefficient', fontsize=unitsSize)
                    
                    im1 = autocorrAx.contour(np.flipud(autocorrSub), clevsPS, colors='black', alpha = 0.25, extent = ext)
                    im1 = autocorrAx.contour(np.flipud(autocorrSub), [percZero_autocorr], colors='black', linestyles='dashed', extent = ext) 

                    # Plot major and minor axis of anisotropy
                    xbar_autocorr = xbar_autocorr - autocorrSizeSub
                    ybar_autocorr = ybar_autocorr - autocorrSizeSub
                    
                    # Reverse sign of second dimension for plotting
                    eigvecs_autocorr[1,:] = -eigvecs_autocorr[1,:]
                    st.plot_bars(xbar_autocorr, ybar_autocorr, eigvals_autocorr, eigvecs_autocorr, autocorrAx, 'red')
                    # autocorrAx.invert_yaxis()
                    # autocorrAx.axis('image')
                    
                    if an == '2d+autocorr':
                        xoffset = 0.05
                        yoffset = 0.93
                        yspace = 0.04
                        eccFontSize = 12
                    else:
                        xoffset = 0.05
                        yoffset = 0.95
                        yspace = 0.05
                        eccFontSize = 14                        
                    
                    plt.text(xoffset, yoffset, 'eccentricity = ' + str(fmt2 % eccentricity_autocorr), transform=autocorrAx.transAxes, backgroundcolor = 'w', fontsize=eccFontSize)
                    plt.text(xoffset, yoffset-yspace, 'orientation = ' + str(fmt2 % orientation_autocorr) + '$^\circ$', transform=autocorrAx.transAxes,backgroundcolor = 'w', fontsize=eccFontSize)
                    
                    plt.xticks(rotation=90) 
                    autocorrAx.set_xlabel('Spatial lag [km]', fontsize=labelsSize)
                    autocorrAx.set_ylabel('Spatial lag [km]', fontsize=labelsSize)
                    
                    titleStr = str(timeLocal) + ', 2D autocorrelation function (ifft(spectrum))'
                    titleStr = '2D autocorrelation function'
                    autocorrAx.set_title(titleStr, fontsize=titlesSize)
                
                #++++++++++++ Draw 1D power spectrum
                if (an == '1d') | (an == '1dnoise') | (an == '1d+2d+autocorr'):
                    if an == '1d+2d+autocorr':
                        psAx = plt.subplot(224)
                    else:
                        psAx = plt.subplot(122)
                    
                    freqLimBeta1 = np.array([resKm/float(largeScalesLims[0]),resKm/float(largeScalesLims[1])])
                    psdLimBeta1 = intercept_beta1+beta1*10*np.log10(freqLimBeta1)
                    plt.plot(10*np.log10(freqLimBeta1), psdLimBeta1,'b--')
                    
                    freqLimBeta2 = np.array([resKm/float(smallScalesLims[0]),resKm/float(smallScalesLims[1])])
                    psdLimBeta2 = intercept_beta2+beta2*10*np.log10(freqLimBeta2)
                    plt.plot(10*np.log10(freqLimBeta2), psdLimBeta2,'r--')
                    
                    # Draw turning point
                    plt.vlines(x=10*np.log10(1.0/scalingBreak_best), ymin=psdLimBeta2[0]-5, ymax = psdLimBeta2[0]+5, linewidth=0.5, color='grey')
                    
                    # Write betas and correlations
                    startX = 0.67
                    startY = 0.95
                    offsetY = 0.04
                    
                    if weightedOLS == 0:
                        txt = "Ordinary least squares"
                    if weightedOLS == 1:
                        txt = "Weighted ordinary least squares"

                    # psAx.text(startX,startY, txt, color='k', transform=psAx.transAxes)
                    
                    txt = r'$\beta_1$ = ' + (fmt2 % beta1) + ",   r = " + (fmt3 % r_beta1)
                    psAx.text(startX,startY-offsetY, txt, color='b', transform=psAx.transAxes)
                    
                    txt = r'$\beta_2$ = ' + (fmt2 % beta2) + ",   r = " + (fmt3 % r_beta2)
                    psAx.text(startX,startY-2*offsetY, txt, color='r', transform=psAx.transAxes)
                    
                    txt = 'WAR = ' + (fmt1 % r.war) + ' %'
                    psAx.text(startX,startY-3*offsetY, txt, transform=psAx.transAxes)
                    
                    txt = 'MM = ' + (fmt3 %raincondmean) + ' mm/hr'
                    psAx.text(startX,startY-4*offsetY, txt, transform=psAx.transAxes)
                    
                    # if (args.minR < 0.01): 
                        # txt = 'Rmin = ' + (fmt3 % args.minR) + ' mm/hr'
                    # else:
                        # txt = 'Rmin = ' + (fmt2 % args.minR) + ' mm/hr'
                    # psAx.text(startX,startY-5*offsetY, txt, transform=psAx.transAxes)
                    
                    # txt = 'Scaling break = ' + str(scalingBreak_best) + ' km'
                    # psAx.text(startX,startY-6*offsetY, txt, transform=psAx.transAxes)
                    
                    # txt = 'Zeros = ' + (fmt1 % zerosDBZ) + ' dBZ - ' + (fmt2 % args.minR) + ' mm/hr'
                    # psAx.text(startX,startY-7*offsetY, txt, transform=psAx.transAxes, fontsize=10)
                    
                    if an == '1dnoise':
                        # Draw 1d noise spectrum
                        plt.plot(10*np.log10(freq),10*np.log10(psd1dnoise),'k')
                    else:
                        # Draw Power spectrum
                        #print(10*np.log10(freq))
                        plt.plot(10*np.log10(freq),10*np.log10(psd1d),'k')
                        
                    titleStr = 'Radially averaged power spectrum'
                    plt.title(titleStr, fontsize=titlesSize)
                    plt.xlabel("Wavelength [km]", fontsize=15)
                    
                    plt.ylabel(unitsSpectrum, fontsize= 15)
                    
                    if fourierVar == 'rainrate':
                        plt.ylim([-50.0,40.0])
                    if fourierVar == 'dbz':
                        plt.ylim([-20.0,70.0])
                    
                    # Create ticks in km
                    ticksList = []
                    tickLocal = minFieldSize
                    for i in range(0,20):
                        ticksList.append(tickLocal)
                        tickLocal = tickLocal/2
                        if tickLocal < resKm:
                            break
                    ticks = np.array(ticksList, dtype=int)
                    ticks_loc = 10.0*np.log10(1.0/ticks)
                    psAx.set_xticks(ticks_loc)
                    psAx.set_xticklabels(ticks)
                    
                    # if (an == '1d+2d+autocorr'):
                        # psAx.set_aspect('equal')
                #plt.gcf().subplots_adjust(bottom=0.15, left=0.20)
                
                if (an == '1d+2d+autocorr'):
                    plt.subplots_adjust(hspace=0.2, wspace=0.35)
                else:
                    fig.tight_layout()
                
                ########### SAVE AND COPY PLOTS
                # Save plot in scratch
                analysisType = an + 'PS'
                stringFigName, inDir,_ = io.get_filename_stats(inBaseDir, analysisType, timeLocal,\
                product, timeAccumMin=timeAccumMin, quality=0, minR=args.minR, wols=weightedOLS, format='png')
                
                with warnings.catch_warnings():  
                    warnings.simplefilter("ignore") 
                    plt.savefig(stringFigName,dpi=300)
                print(stringFigName, ' saved.')
                
                # Copy plot to /store
                stringFigNameOut, outDir,_  = io.get_filename_stats(outBaseDir, analysisType, timeLocal, product, timeAccumMin=timeAccumMin, \
                quality=0, minR=args.minR, wols=weightedOLS, format='png')

                cmd = 'mkdir -p ' + outDir
                os.system(cmd)
                shutil.copy(stringFigName, stringFigNameOut)
                print('Copied: ', stringFigName, ' to ', stringFigNameOut)
    else:
        nrValidFields = 0 # Reset to 0 the number of valid fields with consecutive rainfall
        print('Not enough rain to compute statistics')
        
    ############ WRITE OUT DAILY STATS ###########################
    print('------------------')
    print('Nr valid samples during day: ', len(dailyStats)) 
    minNrDailySamples = 2
    try:
        conditionForWriting = (len(dailyStats) >= minNrDailySamples) and ((hourminStr == '0000') or (timeLocal == timeEnd))
    except:
        print(dir(r))
        sys.exit(1)
        
    if conditionForWriting: 
        # List to numpy array 
        dailyStats = np.array(dailyStats) 
        
        # Write stats in the directory of previous day if last time stamp (midnight of next day) 
        timePreviousDay = timeLocal - datetime.timedelta(days = 1) 
                  
        # Generate filenames 
        analysisType = 'STATS' 
        if hourminStr == '0000': 
            fileNameStats,_,_ = io.get_filename_stats(inBaseDir, analysisType, timePreviousDay, product, timeAccumMin=timeAccumMin,\
            quality=0, minR=args.minR, wols=weightedOLS, variableBreak = variableBreak, format=args.format) 
        else: 
            fileNameStats,_,_ = io.get_filename_stats(inBaseDir, analysisType, timeLocal, product, timeAccumMin=timeAccumMin,\
            quality=0, minR=args.minR, wols=weightedOLS, variableBreak = variableBreak, format=args.format) 
        
        # Write out files 
        spectralSlopeLims = [largeScalesLims_best[0], largeScalesLims_best[1], smallScalesLims_best[1]]
        if (boolPlotting == False): 
            if args.format == 'csv': 
                # Write out CSV file 
                io.write_csv_globalstats(fileNameStats, headers, dailyStats) 
            elif args.format == 'netcdf': 
                # Write out NETCDF file 
                io.write_netcdf_globalstats(fileNameStats, headers, dailyStats, str(args.minR), str(weightedOLS), spectralSlopeLims) 
        
        print(fileNameStats, ' saved.') 
        
        #### Print out some average daily stats
        eulerian_corr_vector = np.array(dt.get_column_list(dailyStats,22)).astype(float)
        lagrangian_corr_vector = np.array(dt.get_column_list(dailyStats,23)).astype(float)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            print('Daily average Eulerian correlation    =',np.nanmean(eulerian_corr_vector))
            print('Daily average Lagrangian correlation  =',np.nanmean(lagrangian_corr_vector))
            print('Daily difference Eul-Lagr correlation =',100*(np.nanmean(lagrangian_corr_vector) - np.nanmean(eulerian_corr_vector)),'%')
        
        #### Write out wavelet decomposed rainfall arrays
        if 'wavelets' in analysis:
            # Write out wavelet coefficients to netCDF file
            analysisType = 'WAVELET'
            
            if hourminStr == '0000': 
                fileNameWavelet,_,_ = io.get_filename_wavelets(inBaseDir, analysisType, timePreviousDay, product, \
                timeAccumMin=timeAccumMin, scaleKM=scaleKm, format='netcdf')
            else:
                fileNameWavelet,_,_ = io.get_filename_wavelets(inBaseDir, analysisType, timeLocal, product, \
                timeAccumMin=timeAccumMin, scaleKM=scaleKm, format='netcdf')

            #timePreviousDayStr = ti.datetime2timestring(timePreviousDay)
            # Write out netCDF file
            io.write_netcdf_waveletscale(fileNameWavelet, dailyTimesWavelets, \
            xvecs[scale2keep], yvecs[scale2keep], dailyWavelets, scaleKm, waveletType = wavelet)
            print('Saved:', fileNameWavelet)
        
            # Copy wavelet netCDFs to /store
            outFileNameWavelet,outDir,_ = io.get_filename_wavelets(outBaseDir, analysisType, timePreviousDay, product, \
            timeAccumMin=timeAccumMin, scaleKM=scaleKm, format='netcdf')

            cmd = 'mkdir -p ' + outDir
            os.system(cmd)
            shutil.copy(fileNameWavelet, outFileNameWavelet)
            print('Copied: ', fileNameWavelet, ' to ', outFileNameWavelet)
        
        #### Reset dailyStats array 
        dailyStats = []
        dailyWavelets = []
        dailyTimesWavelets = []

    ############ WRITE OUT DAILY VELOCITY FIELDS ###########################
    if conditionForWriting and ('of' in analysis):
        analysisType = 'VELOCITY'
        fileNameFlow,_,_ = io.get_filename_stats(inBaseDir, analysisType, timeLocal, product, \
        timeAccumMin=timeAccumMin, quality=0, format='netcdf')
        
        xvec = Xmin + colgrid*1000
        yvec = Ymax - rowgrid*1000 # turn Y vector to start from highest value on top
        io.write_netcdf_flow(fileNameFlow, dailyTimesUV, xvec, yvec, dailyU, dailyV)
        print(fileNameFlow, 'saved.')
        
        #### Reset daily U,V arrays 
        dailyU = []
        dailyV = []
        dailyTimesUV = []        
    
    ####### UPDATE TIME STAMPS    
    # Add 5 minutes (or one hour if working with longer accumulations)
    timeLocal = timeLocal + datetime.timedelta(minutes = timeSampMin)
    tocOneImg = time.clock()
    #print('Elapsed time: ', tocOneImg - ticOneImg)

toc = time.clock()
print('Total archive elapsed time: ', toc-tic, ' seconds.')


