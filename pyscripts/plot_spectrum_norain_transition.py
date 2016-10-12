#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from PIL import Image

import matplotlib as mpl
#mpl.use('Agg')
mpl.rcParams['image.interpolation'] = 'nearest'
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab

import numpy as np
import shutil
import datetime
import time
import warnings

import pyfftw
from scipy import fftpack,stats
import scipy.signal as ss
import scipy.ndimage as ndimage
import pywt

import getpass
usrName = getpass.getuser()

#### Import personal libraries
import time_tools_attractor as ti
import io_tools_attractor as io
import data_tools_attractor as dt
import stat_tools_attractor as st

import radialprofile
import gis_base as gis
    
################
np.set_printoptions(precision=2)

noData = -999.0
fmt1 = "%.1f"
fmt2 = "%.2f"
fmt3 = "%.3f"
fmt5 = "%.5f"

########SET DEFAULT ARGUMENTS##########
timeAccumMin = 5
resKm = 1 # To compute FFT frequency
inBaseDir = '/scratch/' + usrName + '/data/' # directory to read from
outBaseDir = '/store/msrad/radar/precip_attractor/data/'
fourierVar = 'dbz' # field on which to perform the fourier analysis ('rainrate' or 'dbz')
scalingBreakArray_KM = np.arange(5, 42, 1) # [15]
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
parser.add_argument('-plt', default='1d', type=str,help='Type of plot on the side of the precipitation field (1d, 2d, 1dnoise, 2dnoise or noise field).')
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
plotSpectrum = args.plt

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

###################################
# Get dattime from timestamp
timeStart = ti.timestring2datetime(timeStartStr)
timeEnd = ti.timestring2datetime(timeEndStr)

timeAccumMinStr = '%05i' % timeAccumMin
timeAccum24hStr = '%05i' % (24*60)

## COLORMAPS
color_list, clevs = dt.get_colorlist('MeteoSwiss') #'STEPS' or 'MeteoSwiss'
clevsStr = []
for i in range(0,len(clevs)):
    if (clevs[i] < 10) and (clevs[i] >= 1):
        clevsStr.append(str('%.1f' % clevs[i]))
    elif (clevs[i] < 1):
        clevsStr.append(str('%.2f' % clevs[i]))
    else:
        clevsStr.append(str('%i' % clevs[i]))

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
timeLocal = timeStart
dailyStats = []
tic = time.clock()
jobs = []
nrValidFields = 0
rainfallStack = np.zeros((2,fftDomainSize,fftDomainSize))

while timeLocal <= timeEnd:
    ticOneImg = time.clock()
    
    year, yearStr, julianDay, julianDayStr = ti.parse_datetime(timeLocal)
    hour = timeLocal.hour
    minute = timeLocal.minute

    # Create filename for input
    hourminStr = ('%02i' % hour) + ('%02i' % minute)
    radarOperWildCard = '?'

    subDir = str(year) + '/' + yearStr + julianDayStr + '/'
    inDir = inBaseDir + subDir
    fileNameWildCard = inDir + product + yearStr + julianDayStr + hourminStr + radarOperWildCard + '_' + timeAccumMinStr + '*.gif'
    
    # Get filename matching regular expression
    fileName = io.get_filename_matching_regexpr(fileNameWildCard)
    # Get data quality from fileName
    dataQuality = io.get_quality_fromfilename(fileName)
    
    # Check if file exists
    isFile = os.path.isfile(fileName)
    if (isFile == False):
        print('File: ', fileNameWildCard, ' not found.')
    else:
        # Reading GIF file
        print('Reading: ', fileName)
        try:
            # Open GIF image
            rain8bit, nrRows, nrCols = io.open_gif_image(fileName)
            
            # Get GIF image metadata
            alb, doe, mle, ppm, wei = io.get_gif_radar_operation(fileName)
            
            # If metadata are not written in gif file derive them from the quality number in the filename
            if (alb == -1) & (doe == -1) & (mle == -1) & (ppm == -1) & (wei == -1):
                alb, doe, mle = io.get_radaroperation_from_quality(dataQuality)
                
            # Generate lookup table
            lut = dt.get_rainfall_lookuptable(noData)

            # Replace 8bit values with rain rates 
            rainrate = lut[rain8bit]

            if (product == 'AQC') & (timeAccumMin == 5): # AQC is given in millimiters!!!
                rainrate[rainrate != noData] = rainrate[rainrate != noData]*(60/5)
            
            #print('Max rainrate: ', np.max(np.max(rainrate))
            
            # Get coordinates of reduced domain
            extent = dt.get_reduced_extent(rainrate.shape[1], rainrate.shape[0], fftDomainSize, fftDomainSize)
            Xmin = allXcoords[extent[0]]
            Ymin = allYcoords[extent[1]]
            Xmax = allXcoords[extent[2]]
            Ymax = allYcoords[extent[3]]
            
            subXcoords = np.arange(Xmin,Xmax,resKm*1000)
            subYcoords = np.arange(Ymin,Ymax,resKm*1000)
            
            # Select 512x512 domain in the middle
            rainrate = dt.extract_middle_domain(rainrate, fftDomainSize, fftDomainSize)
            rain8bit = dt.extract_middle_domain(rain8bit, fftDomainSize, fftDomainSize)
            
            # Create mask radar composite
            mask = np.ones(rainrate.shape)
            mask[rainrate != noData] = np.nan
            mask[rainrate == noData] = 1
            
            # Set lowest rain thresholds
            if (args.minR > 0.0) and (args.minR < 500.0):
                rainThresholdWAR = args.minR
                rainThresholdPlot = args.minR
                rainThresholdStats = args.minR
            else: # default minimum rainfall rate
                rainThresholdWAR = 0.08
                rainThresholdPlot = 0.08
                rainThresholdStats = 0.08
            
            # Compute WAR
            war = st.compute_war(rainrate,rainThresholdWAR, noData)

            # Set all the non-rainy pixels to NaN (for plotting)
            rainratePlot = np.copy(rainrate)
            condition = rainratePlot < rainThresholdPlot
            rainratePlot[condition] = np.nan
            
            # Set all the data below a rainfall threshold to NaN (for conditional statistics)
            rainrateC = np.copy(rainrate)
            condition = rainrateC < rainThresholdStats
            rainrateC[condition] = np.nan
            
            # Set all the -999 to NaN (for unconditional statistics)
            condition = rainrate < 0
            rainrate[condition] = np.nan
            condition = (rainrate < rainThresholdStats) & (rainrate > 0.0)
            rainrate[condition] = 0.0
        except IOError:
            print('File ', fileName, ' not readable')
            war = -1
        if war >= 0.01:
            # Compute corresponding reflectivity
            A = 316.0
            b = 1.5
            
            ########### Set rain / no-rain parameters
            zerosDBZ,_,_ = dt.rainrate2reflectivity(rainThresholdWAR, A, b)
            zerosDBZlist = [0.0, zerosDBZ]
            filterSizes = [0,2]
            filter = 'gaussian' # 'uniform' or 'gaussian'
            transitionType = 'two-sided' # 'two-sided' or 'one-sided'
            ###########
            
            # Start figures
            nrRows, nrCols = dt.optimal_size_subplot(len(zerosDBZlist)*len(filterSizes))
            fig_spec = plt.figure()
            ax1 = fig_spec.add_subplot(111)
            fig_noise = plt.figure()
            
            colorLine = (0,0,0)
            colors = []
            p = 1
            
            for zerosDBZ in zerosDBZlist:
                print('%%%%%%%%%%%%%%%%%%')
                print('FFT analysis using as zero:', zerosDBZ, 'dBZ')
                dBZ, minDBZ, minRainRate = dt.rainrate2reflectivity(rainrate, A, b, zerosDBZ)
                
                # Get corresponding rainrate from reflectivity
                zeroRain = dt.reflectivity2rainrate(zerosDBZ, A, b)
                
                condition = rainrateC < rainThresholdStats
                dBZC = np.copy(dBZ)
                dBZC[condition] = np.nan
                
                # Replaze NaNs with zeros for Fourier transform
                print("Fourier analysis on", fourierVar, "field.")
                if (fourierVar == 'rainrate'):
                    rainfieldZeros = rainrate.copy()
                    rainfieldZeros[np.isnan(rainfieldZeros)] = 0.0
                elif (fourierVar == 'dbz'):
                    rainfieldZeros = dBZ.copy()
                    
                    # Very delicate choice on which dBZ value to give to the zeros...
                    rainfieldZeros[np.isnan(rainfieldZeros)] = zerosDBZ
                    
                    print('Minimum dBZ: ', minDBZ)
                    print('Zeros dBZ:', zerosDBZ)
                else:
                    print('Invalid variable string for Fourier transform')
                    sys.exit()
                
                f = 0
                for filterSize in filterSizes:
                    # Rainfield mask
                    if (fourierVar == 'rainrate'):
                        rainMask = rainfieldZeros > 0.0
                    elif (fourierVar == 'dbz'):
                        rainMask = rainfieldZeros > zerosDBZ
                    if filterSize > 0:
                        ###### Try smoothing the edges of the rain areas #####
                        # to reduce the impact of rain / no-rain transition

                                
                        # Smoothing of the mask and of the rainfield
                        if filter == 'uniform':
                            rainfieldSmoothed = ndimage.uniform_filter(rainfieldZeros, filterSize)
                            maskSmoothed = ndimage.uniform_filter(rainMask.astype(float), filterSize)
                        if filter == 'gaussian':
                            rainfieldSmoothed = ndimage.gaussian_filter(rainfieldZeros, sigma=(filterSize, filterSize))
                            maskSmoothed = ndimage.gaussian_filter(rainMask.astype(float), sigma=(filterSize, filterSize))
                        
                        # Replacement of values at the edges
                        maskTransition = ((maskSmoothed > 0.01) & (maskSmoothed < 0.99)).astype(bool)
                        
                        if transitionType == 'two-sided':
                            rainfieldSmoothed[~maskTransition] = rainfieldZeros[~maskTransition]
                        if transitionType == 'one-sided':
                            rainfieldSmoothed[rainMask] = rainfieldZeros[rainMask]
                    else:
                        rainfieldSmoothed = np.copy(rainfieldZeros)
                        
                    # Plot field just to check it is ok for FFT/Wavelets
                    # fig_rain = plt.figure()
                    
                    # if filterSize != 0:
                        # ax3 = fig_rain.add_subplot(1,2,1)
                        # ax3.imshow(maskTransition, interpolation='nearest')#, vmin=0.99, vmax=1)
                        # ax3 = fig_rain.add_subplot(1,2,2)
                    # else:
                        # ax3 = fig_rain.add_subplot(1,1,1)
                    # im_rain = ax3.imshow(rainfieldSmoothed, interpolation='nearest')#, vmin=0.99, vmax=1)
                    # fig_rain.subplots_adjust(right=0.8)
                    # cbar_ax = fig_rain.add_axes([0.85, 0.15, 0.05, 0.7])
                    # fig_rain.colorbar(im_rain, cax=cbar_ax)
                    
                    # ax3.set_title('minDBZ = ' + fmt2 % minDBZ + ', filterSize = ' + str(filterSize))
                    # plt.show()
                    # sys.exit()
                    
                    ########### Compute Fourier power spectrum ###########
                    ticFFT = time.clock()
                    
                    # Generate a window function
                    if windowFunction == 'blackman':
                        w = ss.blackman(fftDomainSize)
                        window = np.outer(w,w)
                    else:
                        window = np.ones((fftDomainSize,fftDomainSize))

                    # Compute FFT
                    if FFTmod == 'NUMPY':
                        fprecipNoShift = np.fft.fft2(rainfieldSmoothed*window) # Numpy implementation
                    if FFTmod == 'FFTW':
                        fprecipNoShift = pyfftw.interfaces.numpy_fft.fft2(rainfieldSmoothed*window) # FFTW implementation
                        # Turn on the cache for optimum performance
                        pyfftw.interfaces.cache.enable()
                    
                    # Shift frequencies
                    fprecip = np.fft.fftshift(fprecipNoShift)
                    
                    # Compute 2D power spectrum
                    psd2d = np.abs(fprecip)**2/(fftDomainSize*fftDomainSize)
                    psd2dNoShift = np.abs(fprecipNoShift)**2/(fftDomainSize*fftDomainSize)
                    
                    # Compute 1D radially averaged power spectrum
                    bin_size = 1
                    nr_pixels, bin_centers, psd1d = radialprofile.azimuthalAverage(psd2d, binsize=bin_size, return_nr=True)
                    fieldSize = rainrate.shape
                    minFieldSize = np.min(fieldSize)
                    
                    # Extract subset of spectrum
                    validBins = (bin_centers < minFieldSize/2) # takes the minimum dimension of the image and divide it by two
                    psd1d = psd1d[validBins]
                    
                    # Compute frequencies
                    freq = fftpack.fftfreq(minFieldSize, d=float(resKm))
                    freqAll = np.fft.fftshift(freq)
                    
                    # Select only positive frequencies
                    freq = freqAll[len(psd1d):] 
                    
                    # Compute wavelength [km]
                    with np.errstate(divide='ignore'):
                        wavelengthKm = resKm*(1.0/freq)
                    # Replace 0 frequency with NaN
                    freq[freq==0] = np.nan
                    
                    ############ Compute spectral slopes Beta
                    r_beta1_best = 0
                    r_beta2_best = 0
                    for s in range(0,len(scalingBreakArray_KM)):
                        scalingBreak_KM = scalingBreakArray_KM[s]
                        largeScalesLims = np.array([maxBeta1rangeKM,scalingBreak_KM])
                        smallScalesLims = np.array([scalingBreak_KM,minBeta2rangeKM])
                        idxBeta1 = (wavelengthKm <= largeScalesLims[0]) & (wavelengthKm > largeScalesLims[1]) # large scales
                        idxBeta2 = (wavelengthKm <= smallScalesLims[0]) & (wavelengthKm > smallScalesLims[1]) # small scales
                        idxBetaBoth = (wavelengthKm <= largeScalesLims[0]) & (wavelengthKm > smallScalesLims[1]) # large scales
                        
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
                        print("Best scaling break = ", scalingBreak_best, ' km')
                    else:
                        print("Fixed scaling break = ", scalingBreak_best, ' km')
                    
                    raincondmean = np.nanmean(rainrateC.ravel())
                    
                    ##### Generate stochastic field with given spectrum
                    # Generate a field of white noise
                    #np.random.seed(5)
                    randValues = np.random.randn(fftDomainSize,fftDomainSize)
                    #randValues = np.random.rand(fftDomainSize,fftDomainSize)
                    #randValues[rainMask != 0] = 0.0
                    
                    # plt.imshow(randValues)
                    # plt.colorbar()
                    # plt.show()
                    
                    # randValues = np.fft.fft2(randValues)
                    
                    #noiseSpectrum = np.abs(np.fft.fftshift(randValues))**2/fftDomainSize**2
                    
                    # plt.imshow(dt.to_dB(noiseSpectrum))
                    # plt.colorbar()
                    # plt.show()
                    # Multiply the FFT of the precip field with the the field of white noise
                    fcorrNoise = randValues*fprecipNoShift
                    
                    # Do the inverse FFT
                    corrNoise = np.fft.ifft2(fcorrNoise)
                    # Get the real part
                    corrNoiseReal = np.array(corrNoise.real)
                    corrNoiseReal,_,_ = st.to_zscores(corrNoiseReal)
                    
                    #### PLOT STOCHASTIC FIELDS

                    # Create title text
                    if filterSize == 0:
                            txtFilterSize = 'None'
                    else:
                            txtFilterSize = str(filterSize) + 'x' + str(filterSize)
                    txtTitle = 'Zeros = ' + (fmt1 % minDBZ) + ' dBZ, filter = ' + txtFilterSize + '\n'\
                    + r'$\beta_1$ = ' + (fmt2 % beta1) + r', $\beta_2$ = ' + (fmt2 % beta2) + '\n'\
                    + 'Scaling break at ' + str(scalingBreak_best) + ' km'
                        
                    ax2 = fig_noise.add_subplot(nrRows, nrCols, p)
                    im_noise = ax2.imshow(corrNoiseReal, vmin=-3,vmax=3)
                    ax2.set_title(txtTitle, fontsize=8)
                    ax2.set_axis_off()
                    
                    p += 1
                    
                    ################PLOTTING  
                    titlesSize = 16      
                    
                    # Draw 1D power spectrum
                    if (plotSpectrum == '1d'):
                        # Get data for beta1 and beta2 ranges
                        freqLimBeta1 = np.array([resKm/float(largeScalesLims[0]),resKm/float(largeScalesLims[1])])
                        psdLimBeta1 = intercept_beta1+beta1*10*np.log10(freqLimBeta1)
                        freqLimBeta2 = np.array([resKm/float(smallScalesLims[0]),resKm/float(smallScalesLims[1])])
                        psdLimBeta2 = intercept_beta2+beta2*10*np.log10(freqLimBeta2)
                        # Draw turning point
                        ax1.vlines(x=10*np.log10(1.0/scalingBreak_best), ymin=psdLimBeta2[0]-5, ymax = psdLimBeta2[0]+5, linewidth=0.5, color='grey')
                        ax1.vlines(x=10*np.log10(1.0/smallScalesLims[1]), ymin=psdLimBeta2[1]-2, ymax = psdLimBeta2[1]+2, linewidth=0.5, color='grey')
                        
                        # Draw fitted slopes
                        ax1.plot(10*np.log10(freqLimBeta1), psdLimBeta1,'b-', alpha=0.25)
                        ax1.plot(10*np.log10(freqLimBeta2), psdLimBeta2,'r-', alpha=0.25)
                        
                        if f == 0:
                            lineStyle = '-'
                        elif f == 1:
                            lineStyle = '--'
                        else:
                            lineStyle = ':'
                        f += 1
                        
                        # Create legend text
                        if filterSize == 0:
                            txtFilterSize = 'No. '
                        elif filterSize > 0 and filterSize < 10:
                            txtFilterSize = str(filterSize) + '  . '
                        else:
                            txtFilterSize = str(filterSize) + '. '
                        
                        txtLegend = 'Zeros = ' + (fmt1 % zerosDBZ) + ' dBZ, filter = ' +  txtFilterSize \
                        + r'$\beta_1$ = ' + (fmt2 % beta1) + r', $\beta_2$ = ' + (fmt2 % beta2) \
                        + ', break at ' + str(scalingBreak_best) + ' km'
                    
                        # Draw spectrum
                        ax1.plot(10*np.log10(freq),10*np.log10(psd1d), color=colorLine, linestyle = lineStyle, label=txtLegend)
                    else:
                        print('You can only plot the 1d spectrum with this script...')
                        sys.exit(1)
                    colors.append(colorLine)
                colorLine = np.array(colorLine) + 1/len(zerosDBZlist)
                
        # Set colorbar stochastic realizations
        fig_noise.subplots_adjust(right=0.8)
        cbar_ax = fig_noise.add_axes([0.85, 0.15, 0.05, 0.7])
        fig_noise.colorbar(im_noise, cax=cbar_ax)
        
        # Generate legend for spectra
        legend = ax1.legend(loc='lower left', fontsize=9, labelspacing=0.1)
        # for text in legend.get_texts():
            # text.set_color("red")
        for color,text in zip(colors,legend.get_texts()):
            text.set_color(color)
    
        # Write betas and correlations
        startX = 0.7
        startY = 0.95
        offsetY = 0.04
        
        if weightedOLS == 0:
            txt = "Ordinary least squares"
        if weightedOLS == 1:
            txt = "Weighted ordinary least squares"
        
        # ax1.text(startX,startY, txt, color='k', transform=ax1.transAxes)
        
        # txt = r'$\beta_1$ = ' + (fmt2 % beta1) + ",   r = " + (fmt3 % r_beta1)
        # ax1.text(startX,startY-offsetY, txt, color='b', transform=ax1.transAxes)
        
        # txt = r'$\beta_2$ = ' + (fmt2 % beta2) + ",   r = " + (fmt3 % r_beta2)
        # ax1.text(startX,startY-2*offsetY, txt, color='r', transform=ax1.transAxes)
        
        txt = 'WAR = ' + (fmt1 % war) + ' %'
        ax1.text(startX,startY-1*offsetY, txt, transform=ax1.transAxes)
        
        txt = 'MM = ' + (fmt3 %raincondmean) + ' mm/hr'
        ax1.text(startX,startY-2*offsetY, txt, transform=ax1.transAxes)
        
        # if (rainThresholdWAR < 0.01): 
            # txt = 'Rmin = ' + (fmt3 % rainThresholdWAR) + ' mm/hr'
        # else:
            # txt = 'Rmin = ' + (fmt2 % rainThresholdWAR) + ' mm/hr'
        # ax1.text(startX,startY-3*offsetY, txt, transform=ax1.transAxes)
        
        # if len(scalingBreakArray_KM) == 1:
            # txt = 'Scaling break = ' + str(scalingBreak_best) + ' km'
        # else:
            # txt = 'Variable scaling break'
        # ax1.text(startX,startY-4*offsetY, txt, transform=ax1.transAxes)
        
        # txt = 'Zeros = ' + (fmt1 % zerosDBZ) + ' dBZ - ' + (fmt2 % zeroRain) + ' mm/hr'
        # plt.text(startX,startY-7*offsetY, txt, transform=ax1.transAxes, fontsize=10)
            
        titleStr = '1D power spectrum for ' + str(timeLocal)
        ax1.set_title(titleStr, fontsize=titlesSize)
        ax1.set_xlabel("Wavelenght [km]", fontsize=15)
        
        if fourierVar == 'rainrate':
            unitsSpectrum = r"Rainfall field power $\left[ 10\mathrm{log}_{10}\left(\frac{(mm/hr)^2}{km}\right)\right]$"
        elif fourierVar == 'dbz':
            unitsSpectrum = r"Reflectivity field power $\left[ 10\mathrm{log}_{10}\left(\frac{dBZ^2}{km}\right)\right]$"
        ax1.set_ylabel(unitsSpectrum, fontsize= 15)
        
        if fourierVar == 'rainrate':
            ax1.set_ylim([-50.0,40.0])
        if fourierVar == 'dbz':
            ax1.set_ylim([-20.0,70.0])
        
        # Create ticks in km
        ticksList = []
        tickLocal = minFieldSize
        for i in range(0,20):
            ticksList.append(tickLocal)
            tickLocal = tickLocal/2
            if tickLocal < resKm:
                break
        ticks = np.array(ticksList)
        ticks_loc = 10.0*np.log10(1.0/ticks)
        ax1.set_xticks(ticks_loc)
        ax1.set_xticklabels(ticks)
                
        #plt.gcf().subplots_adjust(bottom=0.15, left=0.20)
        fig_spec.tight_layout()
        
        # Save plot in scratch
        analysisType = plotSpectrum + 'PS'
        #stringFigName, inDir,_ = io.get_filename_stats(inBaseDir, analysisType, timeLocal, product, timeAccumMin=timeAccumMin, quality=0, minR=rainThresholdWAR, wols=weightedOLS, format='png')
        stringFigName = '/users/lforesti/results/' + yearStr + julianDayStr + hourminStr + '_1dspectra_norain.png'
        stringFigNameNoise = '/users/lforesti/results/' + yearStr + julianDayStr + hourminStr + '_stochasticNoise_norain.png'
        with warnings.catch_warnings():  
            warnings.simplefilter("ignore") 
            fig_spec.savefig(stringFigName, dpi=300)
            fig_noise.savefig(stringFigNameNoise, dpi=300)
        print(stringFigName, ' saved.')
        print(stringFigNameNoise, ' saved.')
        
    # Add 5 minutes (or one hour if working with longer accumulations)
    timeLocal = timeLocal + datetime.timedelta(minutes = timeSampMin)
    tocOneImg = time.clock()
    #print('Elapsed time: ', tocOneImg - ticOneImg)

toc = time.clock()
print('Total archive elapsed time: ', toc-tic, ' seconds.')