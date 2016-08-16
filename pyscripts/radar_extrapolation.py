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
domainSize = [512,512] #512
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
    print('Invalid -start time arguments.')
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
rainfallStack = np.zeros((2,domainSize[0],domainSize[1]))
nStacks = np.max((1,np.round(timewindow/timeAccumMin))).astype(int) + 1 # includes present obs
# number of leadtimes
net = np.round(leadtime/timeAccumMin).astype(int)
# leadtimes + number of observations
nt = net + nStacks

# initialise variables
zStack = []
tStack = []
xStack = []
yStack = []
uStack = []
vStack = []

tic = time.clock()
for i in range(nStacks-1,-1*net-1,-1):
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
            extent = dt.get_reduced_extent(rainrate.shape[1], rainrate.shape[0], domainSize[1], domainSize[0])
            Xmin = allXcoords[extent[0]]
            Ymin = allYcoords[extent[1]]
            Xmax = allXcoords[extent[2]]
            Ymax = allYcoords[extent[3]]
            
            subXcoords = np.arange(Xmin,Xmax,resKm*1000)
            subYcoords = np.arange(Ymin,Ymax,resKm*1000)

            # Select 512x512 domain in the middle
            rainrate = dt.extract_middle_domain(rainrate, domainSize[1], domainSize[0])
            rain8bit = dt.extract_middle_domain(rain8bit, domainSize[1], domainSize[0])
            # rainrate = rainrate[150:350,50:250]
            # rain8bit = rain8bit[150:350,50:250]
            # Create mask radar composite
            mask = np.ones(rainrate.shape)
            mask[rainrate != noData] = np.nan
            mask[rainrate == noData] = 1         

            # Compute WAR
            war = dt.compute_war(rainrate,rainThreshold, noData)
            
        except IOError:
            print('File ', fileName, ' not readable')
            war = -1
                      
        if (war >= 0.01 or i < 0):
        
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
            rainfieldZeros = of.morphological_opening(rainfieldZeros, thr=rainThreshold, n=5)
            
            # Move rainfall field down the stack
            nrValidFields = nrValidFields + 1
            rainfallStack[1,:,:] = rainfallStack[0,:]
            rainfallStack[0,:,:] = rainfieldZeros
            
            # Stack image for plotting
            zStack.append(rainrate)
            tStack.append(timeLocal)
            
            ########### Compute optical flow on these two images
            if (nrValidFields >= 2 and i >= 0):
            
                # extract consecutive images
                prvs = rainfallStack[1,:,:]
                next = rainfallStack[0,:,:]
                
                # scale values between 0 and 255
                prvs *= 255.0/prvs.max()   
                next *= 255.0/next.max()   
                
                # 8-bit int
                prvs = np.ndarray.astype(prvs,'uint8')
                next = np.ndarray.astype(next,'uint8') 
                
                # (1a) features to track by threshold (cells)
                maxCorners = 200
                p0, nCorners = of.threshold_features_to_track(prvs, maxCorners, minThr = rainThreshold, blockSize = 35)
                
                # (1b) Shi-Tomasi good features to track
                # maxCorners = 100
                # p0b, nCorners = of.ShiTomasi_features_to_track(prvs, maxCorners, qualityLevel=0.2, minDistance=15, blockSize=5)   
                
                # use both
                # p0 = np.vstack((p0a,p0b))

                # (2) Lucas-Kande tracking
                x, y, u, v, err = of.LucasKanade_features_tracking(prvs, next, p0, winSize=(30,30), maxLevel=3)
                
                # (3) exclude some unrealistic vectors 
                maxspeed = 100/12 # km/5min
                speed = np.sqrt(u**2 + v**2)
                keep = speed < maxspeed
                u = u[keep].reshape(np.sum(keep),1)
                v = v[keep].reshape(np.sum(keep),1)
                x = x[keep].reshape(np.sum(keep),1)
                y = y[keep].reshape(np.sum(keep),1)
                
                # (4) stack vectors within time window
                xStack.append(x)
                yStack.append(y)
                uStack.append(u)
                vStack.append(v)
                
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
    x, y, u, v = of.declustering(x, y, u, v, R = 20, minN = 3)

# (2) kernel interpolation
xgrid, ygrid, U, V = of.interpolate_sparse_vectors_kernel(x, y, u, v, \
                        domainSize, b = [])

# or linear interpolation                        
# xgrid, ygrid, U, V = of.interpolate_sparse_vectors_linear(x, y, u, v, \
                        # domainSize)
                       
toc = time.clock()
print('OF time: ',str(toc-tic),' seconds.')    
            
# resize vector fields for plotting
xs, ys, Us, Vs = of.reduce_field_density_for_plotting(xgrid, ygrid, U, V, 25)

########### Advect most recent rainrate field using the computed optical flow


# MAPLE advection scheme
if (advectionScheme=='maple'):
    # MAPLE computes all leadtimes in one run
    print("Running",str(net),"leadtimes with MAPLE's advection scheme ...")
    
    # resize motion fields by factor f
    f = 0.5
    if (f<1):
        Ures = cv2.resize(U, (0,0), fx=f, fy=f)
        Vres = cv2.resize(V, (0,0), fx=f, fy=f) 
    else:
        Ures = U
        Vres = V

    # extract last radar image to advect
    z = zplot[nStacks-1,:,:]
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
    z = zplot[nStacks-1,:,:]
    z[np.isnan(z)]=0
    
    # U,V per minute
    Ured = U/timeAccumMin
    Vred = V/timeAccumMin
    
    tic = time.clock()
    # loop for n leadtimes
    ztmp = np.zeros((domainSize[0],domainSize[1],net*timeAccumMin+1))
    ztmp[:,:,0] = z
    for it in range(net*timeAccumMin):
        ztmp[:,:,it+1] = adv2d.advxy(Vred,Ured,ztmp[:,:,it],0)
        ztmp[ztmp<0.001]=0
    zethz = ztmp[:,:,range(1,(net*timeAccumMin)+1,timeAccumMin)]    
    toc = time.clock()
    print('AD time: ',str((toc-tic)/net),' seconds per time-step.')
    
# extract edges from observations
edgedStack = []
for it in range(zplot.shape[0]):
    zobs = zplot[it,:,:]
    zobs[np.isnan(zobs)] = 0.0  
    zobs = np.ndarray.astype(zobs>rainThreshold,'uint8')
    # print(np.unique(zobs))
    # zobs = cv2.bilateralFilter(zobs, 5, 17, 17)
    edged = cv2.Canny(zobs, rainThreshold, rainThreshold)
    edged = np.array(edged)
    edged = edged.astype(float)
    edged[edged<=0] = np.nan
    edgedStack.append(edged)
       
# animation
try:
    while True:   
        for it in range(nt):
            plt.clf() 
            if (it <= nStacks-1): # observation mode
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
                    z = np.squeeze(zmaple[:,:,it - nStacks])
                elif (advectionScheme=='ethz'):
                    z = np.squeeze(zethz[:,:,it - nStacks])
                z[z<=0]=np.nan
                titleStr = timeLocal.strftime("%Y.%m.%d %H:%M") + ' + ' + str((it-nStacks+1)*5) + ' min, ' + product + ' rainfall field' 
                
            rainIm = plt.imshow(z, cmap=cmap, norm=norm, interpolation='nearest')
            cbar = plt.colorbar(rainIm, ticks=clevs, spacing='uniform', norm=norm, extend='max', fraction=0.03)
            cbar.set_ticklabels(clevsStr, update_ticks=True)
            cbar.set_label("mm/hr equiv.")   
            if (it > nStacks-1):
                plt.quiver(xs,ys,Us,Vs,angles = 'xy', scale_units='xy')
            plt.imshow(edgedStack[it],cmap='Greys_r',interpolation='nearest')
            plt.grid()
            plt.title(titleStr)
            plt.pause(frameRate)    
            
except KeyboardInterrupt:
    pass