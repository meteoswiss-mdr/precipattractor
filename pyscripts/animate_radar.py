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

resKm = 1 # To compute FFT frequency
inBaseDir = '/scratch/' + usrName + '/data/' # directory to read from
outBaseDir = '/scratch/' + usrName + '/tmp/'
fourierVar = 'dbz' # field on which to perform the fourier analysis ('rainrate' or 'dbz')
domainSize = 512
weightedOLS = 1
FFTmod = 'NUMPY' # 'FFTW' or 'NUMPY'
windowFunction = 'none' #'blackman' or 'none'

########GET ARGUMENTS FROM CMD LINE####
parser = argparse.ArgumentParser(description='Compute radar rainfall field statistics.')
parser.add_argument('-start', default='201505151600', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-end', default='201505152355', type=str,help='Ending date YYYYMMDDHHmmSS.')
parser.add_argument('-product', default='AQC', type=str,help='Which radar rainfall product to use (AQC, CPC, etc).')
parser.add_argument('-minR', default=0.08, type=float,help='Minimum rainfall rate for computation of WAR and various statistics.')
parser.add_argument('-accum', default=5, type=int,help='Accumulation time of the product [minutes].')
parser.add_argument('-temp', default=10, type=int,help='Temporal sampling of the products [minutes].')
parser.add_argument('-dpi', default=150, type=int,help='Image resolution.')
parser.add_argument('-delay', default=20, type=int,help='Display the next image after pausing.')
parser.add_argument('-domainSizeLat', default=512, type=float,help='[km]')
parser.add_argument('-domainSizeLon', default=512, type=float,help='[km]')
parser.add_argument('-figsize', default=9, type=float,help='')
parser.add_argument('-fontsize', default=15, type=float,help='')


args = parser.parse_args()

product = args.product
timeAccumMin = args.accum
dpi = args.dpi
delay = args.delay
domainSize = (args.domainSizeLat,args.domainSizeLon)
figsize = args.figsize
figsize = (figsize, figsize/1.3)
fsize = args.fontsize

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
color_list, clevs, clevsStr = dt.get_colorlist('MeteoSwiss') #'STEPS' or 'MeteoSwiss'
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

radar_observations_10min,_,_ = lf.produce_radar_observation_with_accumulation(startForecastStr, endForecastStr, newAccumulationMin=10)

while timeLocal <= timeEnd:
    ticOneImg = time.clock()
    
    nextTimeStampStr = ti.datetime2timestring(timeLocal)
    if product=='RZC':
        r = io.read_bin_image(nextTimeStampStr,fftDomainSize=domainSize[0])
    else:
        r = io.read_gif_image(nextTimeStampStr,fftDomainSize=domainSize[0],product=product)
            
        if r.war > -1:
                  
            ############# PLOTTING #################################
            plt.close("all")
            fig = plt.figure(figsize=(figsize[0],figsize[1]))
            
            ax = fig.add_axes()
            
            rainAx = plt.subplot(111)
            
            # Draw DEM
            rainAx.imshow(demImg, extent = r.extent, vmin=100, vmax=3000, cmap = plt.get_cmap('gray'))
           
            # Draw rainfield
            rainIm = rainAx.imshow(r.rainrateNans, extent = r.extent, cmap=r.cmap, norm=r.norm, interpolation='nearest')
            
            # Draw shapefile
            gis.read_plot_shapefile(fileNameShapefile, proj4stringWGS84, proj4stringCH,  ax = rainAx, linewidth = 0.75)
            
                        # Colorbar
            cbar = plt.colorbar(rainIm, ticks=r.clevs, spacing='uniform', norm=r.norm, extend='max', fraction=0.03)
            cbar.set_ticklabels(r.clevsStr, update_ticks=True)
            cbar.set_label(r"mm h$^{-1}$",fontsize=fsize)
            
            # Draw radar composite mask
            rainAx.imshow(r.mask, cmap=r.cmapMask, extent = r.extent, alpha = 0.5)
            
            
            # Set X and Y ticks for coordinates
            xticks = np.arange(400, 900, 100)
            yticks = np.arange(0, 500 ,100)
            plt.xticks(xticks*1000, xticks)
            plt.yticks(yticks*1000, yticks)
            plt.xlabel('Swiss easting [km]',fontsize=fsize)
            plt.ylabel('Swiss northing [km]',fontsize=fsize)
            
            txt = str(timeLocal.strftime('%Y-%b-%d %H:%M'))
            rainAx.text(0.98,0.98,txt,backgroundcolor='white', fontsize=fsize,transform=rainAx.transAxes,ha='right',va='top')
            
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