#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import os
import sys
import fnmatch
import argparse
from PIL import Image

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import pylab

import math
import numpy as np
import csv
import shutil
import datetime
import time
import warnings
import cv2

import pyfftw
from scipy import fftpack,stats
import scipy.signal as ss
import scipy.ndimage as ndimage

import getpass
usrName = getpass.getuser()

#### Import personal libraries
import time_tools_attractor as ti
import io_tools_attractor as io
import data_tools_attractor as dt

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

resKm = 1 # To compute FFT frequency
inBaseDir = '/scratch/lforesti/data/' # directory to read from
outBaseDir = '/scratch/' + usrName + '/tmp/'
fourierVar = 'dbz' # field on which to perform the fourier analysis ('rainrate' or 'dbz')
domainSize = 512
weightedOLS = 1
FFTmod = 'NUMPY' # 'FFTW' or 'NUMPY'
windowFunction = 'none' #'blackman' or 'none'

########GET ARGUMENTS FROM CMD LINE####
parser = argparse.ArgumentParser(description='Compute radar rainfall field statistics.')
parser.add_argument('-start', default='201505151600', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-end', default='201601310600', type=str,help='Ending date YYYYMMDDHHmmSS.')
parser.add_argument('-product', default='AQC', type=str,help='Which radar rainfall product to use (AQC, CPC, etc).')
parser.add_argument('-minR', default=0.08, type=float,help='Minimum rainfall rate for computation of WAR and various statistics.')
parser.add_argument('-accum', default=5, type=int,help='Accumulation time of the product [minutes].')
parser.add_argument('-temp', default=10, type=int,help='Temporal sampling of the products [minutes].')
parser.add_argument('-dpi', default=150, type=int,help='Image resolution.')
parser.add_argument('-delay', default=10, type=int,help='Display the next image after pausing.')
parser.add_argument('-domainSizeLat', default=640, type=int,help='[km]')
parser.add_argument('-domainSizeLon', default=640, type=int,help='[km]')

args = parser.parse_args()

product = args.product
timeAccumMin = args.accum
dpi = args.dpi
delay = args.delay
domainSize = (args.domainSizeLat,args.domainSizeLon)

if (timeAccumMin == 60) | (timeAccumMin == 60*24):
    timeSampMin = timeAccumMin
else:
    timeSampMin = args.temp

if (int(args.start) < 198001010000) or (int(args.start) > 203001010000):
    print('Invalid -start or -end time arguments.')
    sys.exit(1)
    
if (int(args.start) > int(args.end) ):
    print('Invalid -start or -end time arguments.')
    sys.exit(1)
    
else:
    timeStartStr = args.start
    timeEndStr = args.end

if (product == 'AQC') or (product == 'CPC'):
    print('Using the ', product, ' product.')
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
dirDEM = '/users/' + usrName + '/pyscripts/shapefiles'
fileNameDEM = dirDEM + '/ccs4.png'
isFile = os.path.isfile(fileNameDEM)
if (isFile == False):
    print('File: ', fileNameDEM, ' not found.')
demImg = Image.open(fileNameDEM)
demImg = dt.extract_middle_domain_img(demImg, domainSize[0], domainSize[1])
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

# create temp folder
outDir = outBaseDir + '_frames' + timeStartStr + '/'
cmd = 'mkdir -p ' + outDir
os.system(cmd)

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
        # print('Reading: ', fileName)
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
            
            # Get coordinates of reduced domain
            extent = dt.get_reduced_extent(rainrate.shape[1], rainrate.shape[0], domainSize[0], domainSize[1])
            Xmin = allXcoords[extent[0]]
            Ymin = allYcoords[extent[1]]
            Xmax = allXcoords[extent[2]]
            Ymax = allYcoords[extent[3]]
            
            subXcoords = np.arange(Xmin,Xmax,resKm*1000)
            subYcoords = np.arange(Ymin,Ymax,resKm*1000)
            
            # Select 512x512 domain in the middle
            rainrate = dt.extract_middle_domain(rainrate, domainSize[0], domainSize[1])
            rain8bit = dt.extract_middle_domain(rain8bit, domainSize[0], domainSize[1])
            
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
            war = dt.compute_war(rainrate,rainThresholdWAR, noData)
            
            # Set all the non-rainy pixels to NaN (for plotting)
            rainratePlot = np.copy(rainrate)
            condition = rainratePlot <= rainThresholdPlot
            rainratePlot[condition] = np.nan
            
            # Set all the data below a rainfall threshold to NaN (for conditional statistics)
            rainrateC = np.copy(rainrate)
            condition = rainrateC <= rainThresholdStats
            rainrateC[condition] = np.nan
            
            # Set all the -999 to NaN (for unconditional statistics)
            condition = rainrate < rainThresholdStats
            rainrate[condition] = np.nan
        except IOError:
            print('File ', fileName, ' not readable')
            war = -1
            
        if war >= 0:
            # Compute corresponding reflectivity
            A = 316.0
            b = 1.5
            dBZ = dt.rainrate2reflectivity(rainrate,A,b)
            
            condition = rainrateC <= rainThresholdStats
            dBZC = np.copy(dBZ)
            dBZC[condition] = np.nan
            dBZ[condition] = np.nan
            
            # Replaze NaNs with zeros for Fourier and optical flow
            if (fourierVar == 'rainrate'):
                rainfieldZeros = rainrate.copy()
            elif (fourierVar == 'dbz'):
                rainfieldZeros = dBZ.copy()
            else:
                print('Invalid variable string for Fourier transform')
                sys.exit()
            
            rainfieldZeros[rainfieldZeros == noData] = 0.0 # set 0 dBZ for zeros???
            
            
            ############# PLOTTING RAINFIELD AND SPECTRUM #################################
            plt.close("all")
            fig = plt.figure(figsize=(9,7.5))
            
            ax = fig.add_axes()
            
            rainAx = plt.subplot(111)
            
            # Draw DEM
            rainAx.imshow(demImg, extent = (Xmin, Xmax, Ymin, Ymax), vmin=100, vmax=3000, cmap = plt.get_cmap('gray'))
           
            # Draw rainfield
            rainIm = rainAx.imshow(rainratePlot, extent = (Xmin, Xmax, Ymin, Ymax), cmap=cmap, norm=norm, interpolation='nearest')
            
            # Draw shapefile
            gis.read_plot_shapefile(fileNameShapefile, proj4stringWGS84, proj4stringCH,  ax = rainAx, linewidth = 0.75)
            
                        # Colorbar
            cbar = plt.colorbar(rainIm, ticks=clevs, spacing='uniform', norm=norm, extend='max', fraction=0.03)
            cbar.set_ticklabels(clevsStr, update_ticks=True)
            if (timeAccumMin == 1440):
                cbar.set_label("mm/day")
            elif (timeAccumMin == 60):
                cbar.set_label("mm/hr")    
            elif (timeAccumMin == 5) and (product == 'AQC'):
                cbar.set_label("mm/hr")
            elif (timeAccumMin == 5):
                cbar.set_label("mm/hr equiv.")
            else:
                print('Accum. units not defined.')
                
            titleStr = timeLocal.strftime("%Y.%m.%d %H:%M") + ', ' + product + ' rainfall field, Q' + str(dataQuality)
            # plt.title(titleStr, fontsize=15)
            
            # Draw radar composite mask
            rainAx.imshow(mask, cmap=cmapMask, extent = (Xmin, Xmax, Ymin, Ymax), alpha = 0.5)
            
            # Add product quality within image
            dataQualityTxt = "Quality = " + str(dataQuality)
            
            # Set X and Y ticks for coordinates
            xticks = np.arange(400, 900, 100)
            yticks = np.arange(0, 500 ,100)
            plt.xticks(xticks*1000, xticks)
            plt.yticks(yticks*1000, yticks)
            plt.xlabel('Swiss easting [km]')
            plt.ylabel('Swiss northing [km]')
            
            txt = str(timeLocal.strftime('%Y-%b-%d %H:%M'))
            rainAx.text(0.74,0.96,txt,backgroundcolor='white', fontsize=12,transform=rainAx.transAxes)   
            
            fig.tight_layout()
            
            # plt.show()
            
            # Save plot in scratch
            analysisType = 'plotRadar'
            # stringFigName, outDir,_ = io.get_filename_stats(outBaseDir, analysisType, timeLocal, product, timeAccumMin=timeAccumMin, quality=0, minR=rainThresholdWAR, wols=0, format='png')
            
            stringFigName = outDir + timeLocal.strftime("%Y%m%d%H%M") + '.png'
            
            with warnings.catch_warnings():  
                warnings.simplefilter("ignore") 
                plt.savefig(stringFigName,dpi=dpi)
            print(stringFigName, ' saved.')
            
    # Add 5 minutes (or one hour if working with longer accumulations)
    timeLocal = timeLocal + datetime.timedelta(minutes = timeSampMin)
    tocOneImg = time.clock()            

# generate gif
print('Generating the animation...')
stringGifName = timeStartStr + '-' + timeEndStr + '_' + product + '_dpi' + str(dpi) + '_step' + str(timeSampMin) + 'min_delay' + str(delay) + '.gif'
cmd = 'convert -delay ' + str(delay) + ' -loop 0 ' + outDir + '/*.png ' + stringGifName
os.system(cmd)
print(stringGifName, ' saved.')

# delete tmp folder
cmd = 'rm -rf ' + outDir
os.system(cmd)
# print(cmd) 
print(outDir, ' removed.')

toc = time.clock()
print('Total elapsed time: ', toc-tic, ' seconds.')