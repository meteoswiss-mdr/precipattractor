#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

# General libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import argparse
import datetime
import getpass
import os
import time

# OpenCV
import cv2

# Precip Attractor libraries
import time_tools_attractor as ti
import io_tools_attractor as io
import data_tools_attractor as dt

# optical flow libraries
import optical_flow as of

# advection libraries
import adv2d
print(adv2d.__doc__)
import maple_ree
print(maple_ree.__doc__)


####################################
###### RADAR EXTRAPOLATION IN PYTHON
####################################

######## Default parameters

noData = -999.0
timeAccumMin = 5
domainSize = 512
resKm = 1 
rainThreshold = 0.08

######## Folder paths

usrName = getpass.getuser()
usrName = "lforesti"
inBaseDir = '/scratch/' + usrName + '/data/' # directory to read from
outBaseDir = '/store/msrad/radar/precip_attractor_' + usrName + '/data/'

######## Parse arguments from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('-start', default='201505151600', type=str,help='Start date of forecast YYYYMMDDHHmmSS.')
parser.add_argument('-leadtime', default=60, type=int,help='')
parser.add_argument('-stack', default=15, type=int,help='')
parser.add_argument('-product', default='AQC', type=str,help='Which radar rainfall product to use (AQC, CPC, etc).')
parser.add_argument('-frameRate', default=0.5, type=float,help='')
parser.add_argument('-adv', default='maple', type=str,help='')

args = parser.parse_args()

advectionScheme = args.adv
frameRate = args.frameRate
product = args.product
leadtime = args.leadtime
timewindow = np.max((5,args.stack))

if (int(args.start) < 198001010000) or (int(args.start) > 203001010000):
    print('Invalid -start or -end time arguments.')
    sys.exit(1)
else:
    timeStartStr = args.start

######## Get dattime from timestamp
timeStart = ti.timestring2datetime(timeStartStr)
timeAccumMinStr = '%05i' % timeAccumMin
timeAccum24hStr = '%05i' % (24*60)    

######## GIS stuff
# Limits of CCS4 domain
Xmin = 255000
Xmax = 965000
Ymin = -160000
Ymax = 480000
allXcoords = np.arange(Xmin,Xmax+resKm*1000,resKm*1000)
allYcoords = np.arange(Ymin,Ymax+resKm*1000,resKm*1000)

# Shapefile filename
fileNameShapefile = "/users/" + usrName + "/pyscripts/shapefiles/CHE_adm0.shp"
proj4stringWGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84"
proj4stringCH = "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 \
+k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs" 

######## Colormaps
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

######## Loop over files to get two consecutive images
nrValidFields = 0
rainfallStack = np.zeros((2,domainSize,domainSize))
nStacks = np.max((1,np.round(timewindow/timeAccumMin))).astype(int)
zStack = []
tStack = []
xStack = []
yStack = []
uStack = []
vStack = []

tic = time.clock()
for i in range(nStacks,-1,-1):
    ######## Load radar images
    timeLocal = timeStart - datetime.timedelta(seconds=i*60*timeAccumMin)
    print(timeLocal)
    
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
        try:
            # Open GIF image
            rain8bit, nrRows, nrCols = io.open_gif_image(fileName)
            
            # Get GIF image metadata
            alb, doe, mle, ppm, wei = io.get_gif_radar_operation(fileName)
            
            # Generate lookup table
            lut = dt.get_rainfall_lookuptable(noData)

            # Replace 8bit values with rain rates 
            rainrate = lut[rain8bit]
            
            if product == 'AQC': # AQC is given in millimiters!!!
                rainrate[rainrate != noData] = rainrate[rainrate != noData]*(60/5)
                
            # Get coordinates of reduced domain
            extent = dt.get_reduced_extent(rainrate.shape[1], rainrate.shape[0], domainSize, domainSize)
            Xmin = allXcoords[extent[0]]
            Ymin = allYcoords[extent[1]]
            Xmax = allXcoords[extent[2]]
            Ymax = allYcoords[extent[3]]
            
            subXcoords = np.arange(Xmin,Xmax,resKm*1000)
            subYcoords = np.arange(Ymin,Ymax,resKm*1000)

            # Select 512x512 domain in the middle
            rainrate = dt.extract_middle_domain(rainrate, domainSize, domainSize)
            rain8bit = dt.extract_middle_domain(rain8bit, domainSize, domainSize)
            
            # Create mask radar composite
            mask = np.ones(rainrate.shape)
            mask[rainrate != noData] = np.nan
            mask[rainrate == noData] = 1         

            # Compute WAR
            war = dt.compute_war(rainrate,rainThreshold, noData)
            
        except IOError:
            print('File ', fileName, ' not readable')
            war = -1
                      
        if war >= 0.01:
            # Compute corresponding reflectivity
            A = 316.0
            b = 1.5
            dBZ = dt.rainrate2reflectivity(rainrate,A,b)
            
            condition = rainrate <= rainThreshold
            rainrate[condition] = np.nan 
            dBZ[condition] = np.nan  
            # Replace NaNs with zeros
            rainfieldZeros = dBZ.copy()
            rainfieldZeros[np.isnan(rainfieldZeros)] = 0.0 
            
            # remove small noise with a morphological operator (opening)
            rainfieldZeros = of.morphological_opening(rainfieldZeros, thr=rainThreshold, n=3)
            
            # Move rainfall field down the stack
            nrValidFields = nrValidFields + 1
            rainfallStack[1,:,:] = rainfallStack[0,:]
            rainfallStack[0,:,:] = rainfieldZeros
            
            # Stack image for plotting
            zStack.append(rainrate)
            tStack.append(timeLocal)
            
            ########### Compute optical flow on these two images
            if nrValidFields >= 2:
            
                # extract consecutive images
                prvs = rainfallStack[1,:,:]
                next = rainfallStack[0,:,:]
                
                # 8-bit int
                prvs = np.ndarray.astype(prvs,'uint8')
                next = np.ndarray.astype(next,'uint8')    
                
                # (1a) features to track by threshold (cells)
                maxCorners = 10
                p0a, nCorners = of.threshold_features_to_track(prvs, maxCorners, blockSize = 20)
                
                # (1b) Shi-Tomasi good features to track
                maxCorners = 200
                p0b, nCorners = of.ShiTomasi_features_to_track(prvs, maxCorners, qualityLevel=0.1, minDistance=10, blockSize=3)   
                
                # merge both
                p0 = np.vstack((p0a,p0b))

                # (2) Lucas-Kande tracking
                x, y, u, v, err = of.LucasKanade_features_tracking(prvs, next, p0, winSize=(15,15), maxLevel=0)
                
                # (3) stack vectors within time window
                xStack.insert(0,x)
                yStack.insert(0,y)
                uStack.insert(0,u)
                vStack.insert(0,v)
                
 
########### Compute optical flow on these two images

# convert lists of arrays into single arrays
x = np.vstack(xStack)
y = np.vstack(yStack) 
u = np.vstack(uStack)
v = np.vstack(vStack)
t = np.hstack(tStack)
zplot = np.dstack(zStack)
# use rollaxis to get the shape to be tx512x512:
zplot = np.rollaxis(zplot,-1)

# (1) decluster sparse motion vectors
if (nStacks > 1):
    x, y, u, v = of.declustering(x, y, u, v, R = 15, minN = 3)

# (2) kernel interpolation
xgrid, ygrid, U, V = of.interpolate_sparse_vectors_kernel(x, y, u, v, \
                        domainSize, b = 30)
    
toc = time.clock()
print('OF time: ',str(toc-tic),' seconds.')    
            
# plot field
# U = np.ones((domainSize,domainSize))
# V = np.ones((domainSize,domainSize))
# V[:,350:512] = np.ones((512,162))*-1
xs, ys, Us, Vs = of.reduce_field_density_for_plotting(xgrid, ygrid, U, V, 20)
# plt.quiver(xs,ys,Us,-Vs)
# plt.show()

########### Advect most recent rainrate field using the computed optical flow

# number of leadtimes
net = np.round(leadtime/timeAccumMin).astype(int)
# leadtimes + number of observations
nt = net + nStacks + 1

# MAPLE advection scheme
if (advectionScheme=='maple'):
    # MAPLE computes all leadtimes in one run
    print("Running",str(net),"leadtimes with MAPLE's advection scheme ...")
    
    # resize motion fields by half
    Ures = cv2.resize(U, (0,0), fx=0.5, fy=0.5) 
    Vres = cv2.resize(V, (0,0), fx=0.5, fy=0.5) 

    # extract last radar image to advect
    z = zplot[nStacks,:,:]
    z[np.isnan(z)]=0
    
    # call routine
    tic = time.clock()
    zmaple = maple_ree.ree_epol_slio(z, Vres, Ures, net)
    toc = time.clock()
    print('AD time: ',str((toc-tic)/net),' seconds per time-step.')

# Michael's advection scheme
if (advectionScheme=='ethz'):
    print("Running",str(net),"leadtimes with Michael's advection scheme ...")
    
    # extract last radar image to advect
    z = zplot[nStacks,:,:]
    z[np.isnan(z)]=0
    
    tic = time.clock()
    # loop for n leadtimes
    ztmp = np.zeros((domainSize,domainSize,net+1))
    ztmp[:,:,0] = z
    for nt in range(net):
        ztmp[:,:,nt+1] = adv2d.advxy(V,U,ztmp[:,:,nt],0)
        ztmp[ztmp<0.001]=0
    zethz = ztmp[:,:,1:nt+1]    
    toc = time.clock()
    print('AD time: ',str((toc-tic)/net),' seconds per time-step.')
    
# animation
for it in range(nt):
    plt.clf() 
    
    if (it <= nStacks): # observation mode
        timeLocal = t[it]
        z = zplot[it,:,:]
        z[z<=0]=np.nan
        titleStr = timeLocal.strftime("%Y.%m.%d %H:%M") + ', ' + product + ' rainfall field' 
        if (it>0):
            x = xStack[it-1]
            y = yStack[it-1]
            u = uStack[it-1]
            v = vStack[it-1]
            plt.quiver(x,y,u,v,angles = 'xy', scale_units='xy', color='darkred')
    
    else: # extrapolation mode
        if (advectionScheme=='maple'):
            z = np.squeeze(zmaple[:,:,it - nStacks - 1])
        elif (advectionScheme=='ethz'):
            z = np.squeeze(zethz[:,:,it - nStacks - 1])
        z[z<=0]=np.nan
        titleStr = timeLocal.strftime("%Y.%m.%d %H:%M") + ' + ' + str((it-nStacks)*5) + ' min, ' + product + ' rainfall field' 

    rainIm = plt.imshow(z, cmap=cmap, norm=norm, interpolation='nearest')
    cbar = plt.colorbar(rainIm, ticks=clevs, spacing='uniform', norm=norm, extend='max', fraction=0.03)
    cbar.set_ticklabels(clevsStr, update_ticks=True)
    cbar.set_label("mm/hr equiv.")   
    plt.quiver(xs,ys,Us,Vs,angles = 'xy', scale_units='xy')
    plt.title(titleStr)
    # asd = np.nansum(z>0)/(100*100)*100
    # plt.title('{:.2f}%'.format(asd))
    plt.pause(frameRate)
    