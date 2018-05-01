#!/usr/bin/env python

"""


"""

######## Load libraries
from __future__ import division
from __future__ import print_function
from sys import stdout
from PIL import Image

import netCDF4
import argparse
import cv2
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sys
import time
import warnings
import os
import gis_base as gis
import pywt

from scipy.stats import spearmanr, pearsonr

import stat_tools_attractor as st
import data_tools_attractor as dt
import io_tools_attractor as io
import maple_ree
import optical_flow as of
import time_tools_attractor as ti
import ssft
import run_fieldextra_c1 as rf

def get_radar_observations(timeStartStr, leadTimeMin, domainSize = 512, product = 'RZC', rainThreshold = 0.08):

    # Get datetime format
    startTime = ti.timestring2datetime(timeStartStr)
    endTime = startTime + datetime.timedelta(minutes=leadTimeMin)
    
    # Number of consecutive radar images to retrieve
    dt = 5 # minutes
    nimages = int((endTime - startTime).total_seconds()/60/dt) + 1
    
    # Retrieve images
    radarStack = get_n_next_radar_image(timeStartStr, nimages, domainSize, product, rainThreshold )

    # Store in Numpy array
    radar_observations = np.zeros((domainSize,domainSize,nimages))
    radar_mask = np.zeros((domainSize,domainSize,nimages))
    timestamps = []
    currentTime = startTime
    for t in xrange(len(radarStack)):
        # radar_observations[:,:,t] = radarStack[t].rainrateNans.copy()
        if radarStack[t].war>-1:
            radar_observations[:,:,t] = radarStack[t].rainrate.copy()
            radar_mask[:,:,t] = radarStack[t].mask.copy()
            timestamps.append(radarStack[t].datetime)
        else:
            radar_observations[:,:,t] = np.zeros((domainSize,domainSize))*np.nan
            radar_mask[:,:,t] = np.ones((domainSize,domainSize))
            timestamps.append(currentTime)
        
        currentTime = currentTime + datetime.timedelta(minutes=5)

    robject = radarStack[0]
    
    return radar_observations, radar_mask, timestamps, robject
    
def radar_extrapolation(timeStartStr, leadTimeMin, domainSize = 640, finalDomainSize = 512, product = 'RZC', rainThreshold = 0.08):
    ######## preamble
    ticTotal = time.time()

    ######## parameters and settings
    resKm = 1
    timeAccumMin = 5
    
    NumberLeadtimes = int(leadTimeMin/timeAccumMin)
    nrImagesOpticalFlow = 3
    nrLastRadarObservations = nrImagesOpticalFlow
    
    # (1) load last observations 
    # last element in stack is most recent observation
    print('Reading radar files...')
    tic = time.time()
    radarStack = get_n_last_radar_image(timeStartStr, nrLastRadarObservations, domainSize, product, rainThreshold)

    # (2) extract dBZ fields and buffer them
    dbzStack = extract_dBZ_and_buffer(radarStack, domainSize, buffer = 0)
    cascadeShape = dbzStack[0].shape
    nrOfFields = len(dbzStack)
    lastField = dbzStack[-1].copy() + radarStack[-1].dbzThreshold
    toc = time.time()
    print('\t Elapsed time: ', toc - tic, ' seconds.')

    # (3) compute the motion field
    print('Computing the motion field...')
    tic = time.time()
    U,V = get_motion_field(dbzStack, doplot=0, verbose=0)
    toc = time.time()
    print('\t Elapsed time: ', toc - tic, ' seconds.')

    # (4) perform the deterministic forecast
    # advect last radar image
    deterministicForecast = compute_advection(lastField,U,V,net=NumberLeadtimes)
    # convert dBZ to rainrates and apply radar mask, get timestamps
    timestamps=[]
    deterministicForecastFinal=np.zeros((finalDomainSize,finalDomainSize,NumberLeadtimes))
    for t in range(NumberLeadtimes):
        deterministicForecastFinal[:,:,t] = dt.extract_middle_domain(dt.reflectivity2rainrate(deterministicForecast[:,:,t]),finalDomainSize,finalDomainSize)
        timestamps.append(radarStack[-1].datetime + datetime.timedelta(minutes=(t+1)*5))
    
    # Include NaNs
    deterministicForecastFinal[deterministicForecastFinal <= radarStack[0].rainThreshold] = 0
    
    # final runnning time
    tocTotal = time.time()
    print('Total elapsed time: ', tocTotal - ticTotal, ' seconds.')
    
    return deterministicForecastFinal, timestamps
    
def probabilistic_radar_extrapolation(timeStartStr, leadTimeMin, domainSize = 640, finalDomainSize = 512, product = 'RZC', \
        NumberMembers = 2, NumberLevels = 8, dbzStack=[], timeAccumMin = 5, buffer = 1, rainThreshold = 0.08, local_level = 0, seed = 42):
        
    ######## preamble
    np.random.seed(seed)
    ticTotal = time.time()

    ######## parameters and settings

    resKm = 1
    fftDomainSize = domainSize
    
    NumberLeadtimes = int(leadTimeMin/timeAccumMin)
    Width = 2.0 # width of the gaussian filter
    ARorder = 2
    nrImagesOpticalFlow = 3
    nrLastRadarObservations = np.max((nrImagesOpticalFlow,ARorder+1))
    
    # (1) load last observations 
    # last element in stack is most recent observation
    tic = time.time()
    print('Reading radar files...')
    with np.errstate(divide='ignore',invalid='ignore'): 
        radarStack = get_n_last_radar_image(timeStartStr, nrLastRadarObservations, fftDomainSize, product, rainThreshold)

    # (2) extract dBZ fields and buffer them
    if not dbzStack:
        dbzStack = extract_dBZ_and_buffer(radarStack, fftDomainSize, buffer = buffer)
    cascadeShape = dbzStack[0].shape
    nrOfFields = len(dbzStack)
    target = dbzStack[-1].copy() + radarStack[-1].dbzThreshold
    radarMask = dt.extract_middle_domain(radarStack[-1].mask, fftDomainSize, fftDomainSize)
    radarMask_buffered = np.zeros((fftDomainSize+2*buffer,fftDomainSize+2*buffer))
    radarMask_buffered[buffer:buffer+fftDomainSize,buffer:buffer+fftDomainSize] = radarMask
    radarMask = radarMask_buffered
    radarMask = np.array(np.isnan(radarMask),dtype=int) # to do: add buffer to radarMask as well
    toc = time.time()
    print('\t Elapsed time: ', toc - tic, ' seconds.')

    # (3) compute the motion field
    print('Computing the motion field...')
    tic = time.time()
    U,V = get_motion_field(dbzStack, doplot=0, verbose=0)
    toc = time.time()
    print('\t Elapsed time: ', toc - tic, ' seconds.')
    # prepare perturbations for motion field
    perturbations_mf = np.random.normal(loc=1.0, scale=0.1, size=NumberMembers)

    # (4) advect all observations to t0
    dbzStack_at_t0 = []
    for n in np.arange(0,nrOfFields-1):
        tmp = compute_advection(dbzStack[n].copy(),U,V,net=nrOfFields-n-1)
        dbzStack_at_t0.append(tmp[:,:,-1])
    dbzStack_at_t0.append(dbzStack[-1].copy())    

    # (5) cascade decomposition of the last radar observations
    print('Computing the cascade levels...')
    tic = time.time()
    # first compute the bandpass filter
    BandpassFilter2D,CentreWaveLengths = calculate_bandpass_filter(cascadeShape,NumberLevels, Width = Width, doplot = 0)
    cascadeStack, cascadeMeanStack, cascadeStdStack = get_cascade_from_stack(dbzStack_at_t0, NumberLevels, BandpassFilter2D, CentreWaveLengths, \
                                verbose=0, doplot=0)
    toc = time.time()
    print('\t Elapsed time: ', toc - tic, ' seconds.')

    CascadeSum = np.zeros(dbzStack_at_t0[-1].shape)
    for LevelA in np.arange(0,NumberLevels):
        CascadeSum += cascadeStack[-1][:,:,LevelA]*cascadeStdStack[-1][LevelA] + cascadeMeanStack[-1][LevelA]

    # (6) estimation of the AR(n) parameters
    print('Estimating the AR(%i) parameters for all levels...' % ARorder)
    tic = time.time()
    phi,r = autoregressive_parameters(cascadeStack, cascadeMeanStack, cascadeStdStack, ARorder)
    phin = phi.copy()
    print('Phi:')
    print('\n'.join('{}: {}'.format(*k) for k in enumerate(phi)))

    toc = time.time()
    print('\t Elapsed time: ', toc - tic, ' seconds.')

    # (7) generation of all the perturbation fields
    print('Generating the perturbation fields...')
    tic = time.time()
    noiseStack = get_perturbation_fields(dbzStack[-1], NumberMembers, NumberLeadtimes, \
                local_level=local_level, seed=seed)
    toc = time.time()
    print('\t Elapsed time: ', toc - tic, ' seconds.')
    
    # r0 = io.read_gif_image('201601010000')
    # for iii in xrange(10):
        # plt.imshow(noiseStack[iii],interpolation='nearest',cmap=r0.cmap,vmin=-3,vmax=3)
        # plt.show()
        
    # (8) cascade decomposition of all the perturbation fields
    print('Cascade decomposition of the perturbation fields...')
    tic = time.time()
    noiseCascadeStack, noiseCascadeMeanStack, noiseCascadeStdStack = get_cascade_from_stack(noiseStack, NumberLevels, BandpassFilter2D, CentreWaveLengths, \
                verbose=0, doplot=0)
    toc = time.time()
    print('\t Elapsed time: ', toc - tic, ' seconds.')

    # (9) perform the stochastic forecast
    stochasticForecast = np.zeros((finalDomainSize,finalDomainSize,NumberLeadtimes,NumberMembers))
    deterministicForecast = np.zeros((finalDomainSize,finalDomainSize,NumberLeadtimes))
    target = compute_advection(target,U,V,net=NumberLeadtimes);print(target.shape)
    radarMask = compute_advection(radarMask,U,V,net=NumberLeadtimes)
    radarMask = np.array(radarMask>0,dtype=int)
    radarMask_final = np.zeros((finalDomainSize,finalDomainSize,NumberLeadtimes))
    
    Forecast = []
    countnoise = -1
    for m in range(NumberMembers):  
        print('member %i' % m)
        countnoise+=2
        
        # noise cascade
        noiseCascadeLag1 = noiseCascadeStack[countnoise].copy()
        noiseCascadeLag2 = noiseCascadeStack[countnoise-1].copy()
        
        # radar cascade
        extrapolationCascadeLag1 = cascadeStack[-1].copy()
        extrapolationCascadeMeanLag1 = cascadeMeanStack[-1].copy()
        extrapolationCascadeStdLag1 = cascadeStdStack[-1].copy()
        extrapolationCascadeLag2 = cascadeStack[-2].copy()
        extrapolationCascadeMeanLag2 = cascadeMeanStack[-2].copy()
        extrapolationCascadeStdLag2 = cascadeStdStack[-2].copy()
        
        noiseCascade = np.zeros_like(extrapolationCascadeLag1)
        forecastCascade = np.zeros_like(extrapolationCascadeLag1)
        timestamps=[]
        for t in range(NumberLeadtimes):
            print('\t +%i min' % int((t+1)*timeAccumMin))
            timestamps.append(radarStack[-1].datetime + datetime.timedelta(minutes=(t+1)*5))
            
            if m==0:
                # target[:,:,t] = ssft.quantile_transformation(target[:,:,t],dbzStack[-1].copy() + radarStack[-1].dbzThreshold)
                target_sub = dt.extract_middle_domain(target[:,:,t].copy(),finalDomainSize,finalDomainSize)
                target_sub[target_sub<=radarStack[-1].dbzThreshold] = np.nan 
                deterministicForecast[:,:,t] = dt.reflectivity2rainrate(target_sub.copy())
            countnoise +=1
            memberForecast = np.zeros(cascadeShape)
            for l in range(NumberLevels):
            
                noiseShockTerm = noiseCascadeStack[countnoise][:,:,l].copy()
                noiseVariance = ( (1 + phi[l,1]) * (1 + phi[l,0] - phi[l,1])*(1 - phi[l,0] - phi[l,1]) ) / ( 1 - phi[l,1])
                if (t==0) and (m==0):
                    print('v_n = %.3f' % np.sqrt(noiseVariance))
                
                # advect noise cascade levels
                #noiseCascadeLag1[:,:,l] = np.squeeze(compute_advection(noiseCascadeLag1[:,:,l],U,V,net=1))
                #noiseCascadeLag2[:,:,l] = np.squeeze(compute_advection(noiseCascadeLag2[:,:,l],U,V,net=1))
                noiseShockTerm = noiseCascadeStack[countnoise][:,:,l].copy()
         
                # AR() for noise cascade
                #noiseCascade[:,:,l] =  phin[l,0] * noiseCascadeLag1[:,:,l] \
                #                       + phin[l,1] * noiseCascadeLag2[:,:,l] \
                #                       + np.sqrt(noiseVariance)*noiseShockTerm # the shock term
                # renormalize the level N(0,1)
                #noiseCascade[:,:,l] = (noiseCascade[:,:,l] - noiseCascade[:,:,l].mean())/noiseCascade[:,:,l].std()                  
                
                # advect radar extrapolation cascade           
                extrapolationCascadeLag1[:,:,l] = ( compute_advection(extrapolationCascadeLag1[:,:,l],perturbations_mf[m]*U,perturbations_mf[m]*V,net=1) ).squeeze()
                extrapolationCascadeLag2[:,:,l] = ( compute_advection(extrapolationCascadeLag2[:,:,l],perturbations_mf[m]*U,perturbations_mf[m]*V,net=1) ).squeeze()
                
                # AR() process
                forecastCascade[:,:,l] =  phi[l,0] * extrapolationCascadeLag1[:,:,l] \
                                               + phi[l,1] * extrapolationCascadeLag2[:,:,l] \
                                               + np.sqrt(noiseVariance)*noiseShockTerm#noiseCascade[:,:,l]

                #forecastCascade[:,:,l] = (forecastCascade[:,:,l] - forecastCascade[:,:,l].mean())/forecastCascade[:,:,l].std()   
                
                # recompose the cascade 
                memberForecast += forecastCascade[:,:,l] * cascadeStdStack[-1][l] + cascadeMeanStack[-1][l]
                
            # update the stacks
            # noiseCascadeLag2 = noiseCascadeLag1.copy()
            # noiseCascadeLag1 = noiseCascade.copy()
            extrapolationCascadeLag2 = extrapolationCascadeLag1.copy()
            extrapolationCascadeLag1 = forecastCascade.copy()
                    
            # add back the dBZ threshold 
            memberForecast += radarStack[-1].dbzThreshold  
                
            # probability matching
            # memberForecast = ssft.quantile_transformation(memberForecast,dbzStack[-1].copy() + radarStack[-1].dbzThreshold)    
            memberForecast = ssft.quantile_transformation(memberForecast,target[:,:,t])    
            
            # Apply the zeros and convert to rainrates
            memberForecast[memberForecast<=radarStack[-1].dbzThreshold] = 0
            memberForecast = dt.reflectivity2rainrate(memberForecast)
            # memberForecast[memberForecast<=radarStack[-1].rainThreshold] = np.nan 
            
            # Apply radar mask
            # memberForecast *= radarMask[:,:,t]       
            # memberForecast[radarMask[:,:,t]==0] = np.nan   

            # extract middle domain and renormalize the field        
            memberForecast = dt.extract_middle_domain(memberForecast, finalDomainSize, finalDomainSize) 
            
            # plt.clf()
            # plt.imshow(memberForecast,interpolation='none')
            # plt.colorbar()
            # plt.pause(.1) 
            
            stochasticForecast[:,:,t,m] = memberForecast.copy()
    
            if m==0:
                radarMask_final[:,:,t] = dt.extract_middle_domain(radarMask[:,:,t], finalDomainSize, finalDomainSize) 
    
    # final runnning time
    tocTotal = time.time()
    print('Total elapsed time: ', tocTotal - ticTotal, ' seconds.')
    
    return stochasticForecast,timestamps,radarMask_final
    
def move_field1_to_field2(field1,field2):

    # to dBZ
    field1dBZ, _, _ = dt.rainrate2reflectivity(field1)
    field2dBZ, _, _ = dt.rainrate2reflectivity(field2)
    

    # create stack
    fieldStack = []
    fieldStack.append(field1dBZ)
    fieldStack.append(field2dBZ)
    
    # generate vectors field1 -> field2
    U,V = get_motion_field(fieldStack,verbose=0)
    
    # translate field1 -> field2
    field1moved = compute_advection(field1,U,V)[:,:,0]
    # field1moved = np.array(field1moved)

    return field1moved,U,V
    
# retrieve n last radar images 
def get_n_last_radar_image(timeStampStr, nimages, domainSize, product = 'RZC', rainThreshold = 0.08):
    timeStamp = ti.timestring2datetime(timeStampStr)
    radarStack = []
    nextTimeStamp = timeStamp
    for n in xrange(nimages):
        nextTimeStampStr = ti.datetime2timestring(nextTimeStamp)
        if product == 'RZC':
            r = io.read_bin_image(nextTimeStampStr,fftDomainSize=domainSize,product=product, minR = rainThreshold) # get full extent of the radar image
        else:
            r = io.read_gif_image(nextTimeStampStr,fftDomainSize=domainSize,product=product, minR = rainThreshold) # get full extent of the radar image
        radarStack.insert(0,r)
        nextTimeStamp = nextTimeStamp - datetime.timedelta(minutes=5)
    return radarStack

# retrieve n next radar images 
def get_n_next_radar_image(timeStampStr, nimages, domainSize, product = 'RZC', rainThreshold = 0.08):
    timeStamp = ti.timestring2datetime(timeStampStr)
    radarStack = []
    nextTimeStamp = timeStamp
    for n in xrange(nimages):
        nextTimeStampStr = ti.datetime2timestring(nextTimeStamp)
        if product=='RZC':
            r = io.read_bin_image(nextTimeStampStr,fftDomainSize=domainSize,product=product, minR = rainThreshold)
        else:
            r = io.read_gif_image(nextTimeStampStr,fftDomainSize=domainSize,product=product, minR = rainThreshold)
        radarStack.append(r)
        nextTimeStamp = nextTimeStamp + datetime.timedelta(minutes=5)
    return radarStack
    
# crop image at the largest possible square size and fill with zeros to achieve desired buffer of zeros
def extract_dBZ_and_buffer(radarStack, domainSize, buffer):
    dbzStack = []
    buffer = int(buffer)
    for n in xrange(len(radarStack)):
        # extract full dBZ field
        dbzTmp = radarStack[n].dBZFourier.copy()
        # remove the dbZ threshold
        dbzTmp = dbzTmp - radarStack[n].dbzThreshold
        # crop
        if n==0:
            min_size = np.min(dbzTmp.shape)
            buffer = int( buffer - (min_size - domainSize)/2 )
        dbzTmp_reduced = dt.extract_middle_domain(dbzTmp, min_size, min_size)
        # and buffer
        dbzTmp_buffered = np.zeros((min_size+2*buffer,min_size+2*buffer))
        dbzTmp_buffered[buffer:buffer+min_size,buffer:buffer+min_size] = dbzTmp_reduced
        # add to the stack
        dbzStack.append(dbzTmp_buffered)
    return dbzStack

# compute the motion field using all available images    
def get_motion_field(dbzStack, verbose=1, doplot=0, resKm=1, resMin = 5):

    if verbose:
        print('________start of OF routine___________')

    #+++++++++++ Optical flow parameters
    maxCornersST = 1000 # Number of asked corners for Shi-Tomasi
    qualityLevelST = 0.05
    minDistanceST = 5 # Minimum distance between the detected corners
    blockSizeST = 15
    
    winsizeLK = 50 # Small windows (e.g. 10) lead to unrealistic high speeds
    nrLevelsLK = 10 # Not very sensitive parameter
    
    kernelBandwidth = []  # Bandwidth of kernel interpolation of vectors [km]
    
    maxSpeedKMHR = 120 # Maximum allowed speed [km/hr]
    nrIQRoutlier = 3 # Nr of IQR above median to consider the vector as outlier (if < 100 km/hr)
    
    #+++++++++++ 
    
    nrOfFields = len(dbzStack)
    rowStack=[]
    colStack=[]
    uStack=[]
    vStack=[]

    for n in np.arange(0,nrOfFields-1):
    
        if verbose:
            print('(%i) -------------' % n)
        
        # extract consecutive images
        prvs = dbzStack[n].copy()
        next = dbzStack[n+1].copy()
        
        # scale between 0 and 255
        prvs = ( prvs - prvs.min() )/ ( prvs.max() - prvs.min() ) * 255
        next = ( next - next.min() )/ ( next.max() - next.min() ) * 255
        
        # 8-bit int
        prvs = np.ndarray.astype(prvs,'uint8')
        next = np.ndarray.astype(next,'uint8')
        
        # remove small noise with a morphological operator (opening)
        prvs = of.morphological_opening(prvs, thr=0, n=3)
        next = of.morphological_opening(next, thr=0, n=3)
        
        # (1) Shi-Tomasi good features to track
        p0, nCorners = of.ShiTomasi_features_to_track(prvs, maxCornersST, qualityLevel=qualityLevelST, minDistance=minDistanceST, blockSize=blockSizeST)   
        if verbose:
            print("Nr of points OF ShiTomasi          =", len(p0))
        
        # (2) Lucas-Kanade tracking
        col, row, u, v, err = of.LucasKanade_features_tracking(prvs, next, p0, winSize=(winsizeLK,winsizeLK), maxLevel=nrLevelsLK)

        # (3) exclude outliers   
        speed = np.sqrt((u*resKm)**2 + (v*resKm)**2) # km/resMin
        q1, q2, q3 = np.percentile(speed, [25,50,75]) # km/resMin
        maxspeed = np.min((maxSpeedKMHR/(60/resMin), q2 + nrIQRoutlier*(q3 - q1))) # km/resMin
        minspeed = np.max((0,q2 - 2*(q3 - q1)))
        keep = (speed <= maxspeed) & (speed >= minspeed)

        if verbose:
            print('Max speed       =',np.max(speed)*(60/resMin))
            print('Median speed    =',np.percentile(speed,50)*(60/resMin))
            print('Speed max threshold =',maxspeed*(60/resMin))
            print('Speed min threshold =',minspeed*(60/resMin))
            print('Units           = Km/h')
        
        u = u[keep].reshape(np.sum(keep),1)
        v = v[keep].reshape(np.sum(keep),1)
        row = row[keep].reshape(np.sum(keep),1)
        col = col[keep].reshape(np.sum(keep),1)
        
        # (4) stack vectors within time window
        rowStack.append(row)
        colStack.append(col)
        uStack.append(u)
        vStack.append(v)
    if verbose: 
            print('======================')    
    # (5) convert lists of arrays into single arrays
    row = np.vstack(rowStack)
    col = np.vstack(colStack) 
    u = np.vstack(uStack)
    v = np.vstack(vStack)
     
    # (6) decluster sparse motion vectors
    Rsize = 10
    col, row, u, v = of.declustering(col, row, u, v, R = Rsize, minN = 2)
    if verbose:
        print("Nr of points OF after declustering (R=%i) = %i" % (Rsize,len(row)))
        
    # (7) kernel interpolation
    domainSize = [next.shape[0], next.shape[1]]
    colgrid, rowgrid, U, V, b = of.interpolate_sparse_vectors_kernel(col, row, u, v, domainSize)#, b = kernelBandwidth/resKm)
    if verbose:
        print('Kernel bandwith = %.2f' % b)
        print('Mean U = %.2f [dx/dt], mean V = %.2f [dy,dt]' % (U.mean(),V.mean()))
        print('________end of OF routine____________')
    
    
    if doplot:
        # Resize vector fields for plotting
        xs, ys, Us, Vs = of.reduce_field_density_for_plotting(colgrid, rowgrid, U, V, 30)

        # Plot vectors to check if correct
        plt.imshow(next, interpolation='none')
        plt.quiver(xs, ys, Us, Vs,angles = 'xy', scale_units='xy')
        plt.show()
        
    return U,V

def compute_advection(field,U,V,net=1):
    # resize motion fields by factor f (for advection)
    f = 0.3
    if (f<1):
        Ures = cv2.resize(U, (0,0), fx=f, fy=f)
        Vres = cv2.resize(V, (0,0), fx=f, fy=f) 
    else:
        Ures = U
        Vres = V

    # Call MAPLE routine for advection
    field_lag = maple_ree.ree_epol_slio(field, Vres, Ures, net)
    # field_lag = np.squeeze(field_lag)
    return field_lag
    
def advect_radar_to_t0(dbzStack,U,V):
    nrOfFields = len(dbzStack)
    
    dbzStack_at_t0 = []
    for n in np.arange(0,nrOfFields-1):
        tmp = compute_advection(dbzStack[n].copy(),U,V,net=nrOfFields-n-1)
        dbzStack_at_t0.append(tmp[:,:,-1])
        
    dbzStack_at_t0.append(dbzStack[-1].copy())    
    
def calculate_bandpass_filter(FFTShape, NumberLevels, Width = 2.0, resKm=1, doplot=False):
    """
    Method to construct the band pass filters needed for each level in the cascade.
    
    """
    if NumberLevels>1:
        FFTSize = FFTShape[0]
        NumberRows, NumberCols = FFTShape
        CascadeSize = np.max((NumberRows, NumberCols))
        ScaleRatio = ( 2.0/CascadeSize )**( 1.0/(NumberLevels - 1) )
        # while ( ScaleRatio < 0.42 ):
                # NumberLevels+=1
                # ScaleRatio = ( 2.0/CascadeSize )**( 1.0/(NumberLevels - 1) )
        # ScaleRatio = 0.5
        
        Nyquest = int(FFTSize/2)
        FFTStride = Nyquest + 1
        CentreWaveLength = CascadeSize
        
        # Initialiaze variables
        BandpassFilter = np.zeros((NumberLevels,Nyquest**2))
        BandpassFilter1D = np.zeros(NumberLevels*Nyquest)
        BandpassFilter1D[0] = 1.0
        FilterSum = np.zeros(Nyquest) # Sum of the filters
        NormFactor = np.zeros(Nyquest) # Normalisation to sum  = 1
        
        # Start the loop over the levels in the cascade
        CentreWaveLengths = np.zeros(NumberLevels)
        for Level in xrange(NumberLevels):
            CentreFreq = 1.0/CentreWaveLength
            CentreWaveLengths[Level] = CentreWaveLength.copy()
            for WaveNumber in xrange(1,Nyquest):
                Freq = WaveNumber/FFTSize
                if ( Freq > CentreFreq ):
                    RelFrequency = Freq/CentreFreq 
                else:
                    RelFrequency = CentreFreq/Freq
                Filter = np.exp(-Width*RelFrequency)
                BandpassFilter1D[WaveNumber+Level*Nyquest] = Filter
                FilterSum[WaveNumber] += Filter 
                
            CentreWaveLength *= ScaleRatio
        
        # loop over the wave numbers and calculate the normalisation factor
        for WaveNumber in xrange(1,Nyquest):
            NormFactor[WaveNumber] = 1.0/FilterSum[WaveNumber] 
        
        # Normalise the filters so that each wave number sums to one    
        for Level in xrange(NumberLevels):
            for WaveNumber in xrange(1,Nyquest):
                Filter =  BandpassFilter1D[WaveNumber+Level*Nyquest]*NormFactor[WaveNumber]
                if ( Filter < 0.001 ):
                    Filter = 0.0
                BandpassFilter1D[WaveNumber+Level*Nyquest] = Filter 
        
        # Okay, we've generated a 1D filter, now predetermine the 2D filter radarMask for each level
        BandpassFilter2D = np.zeros((NumberLevels,FFTShape[0],FFTShape[1]))
        for Level in xrange(NumberLevels):
            FilterOffset = Level * Nyquest
            for Row in np.arange(0,Nyquest):
                RowIdx = Row * Nyquest
                RowSqr = Row * Row
                
                for Col in xrange(Nyquest):
                    CurrentWaveNumber = np.sqrt(RowSqr + Col*Col)
                    if (CurrentWaveNumber <= Nyquest):
                        BandpassFilter[Level,RowIdx + Col] += BandpassFilter1D[int(FilterOffset + CurrentWaveNumber)]
            # Finally, construct the full 2D filter
            count=0
            idxi = np.zeros(2,dtype=int)
            idxj = np.zeros(2,dtype=int)
            for r in xrange(2):
                for c in xrange(2):
                    subFilter2d = BandpassFilter[Level,:].copy().reshape((Nyquest,Nyquest))
                    subFilter2d = np.rot90(subFilter2d[0:Nyquest,0:Nyquest],-1*count)
                    if r>0:
                        subFilter2d = np.fliplr(subFilter2d)
                    idxi[0] = r*Nyquest
                    idxi[1] = ((r+1)*Nyquest)
                    idxj[0] = c*Nyquest
                    idxj[1] = ((c+1)*Nyquest)
                    BandpassFilter2D[Level,idxi[0]:idxi[1],idxj[0]:idxj[1]] += subFilter2d
                    count+=1
             
        # shift the filter 
        FilterSum = np.sum(BandpassFilter2D,0)    
        lastFilter = BandpassFilter2D[-1,:,:]
        lastFilter[FilterSum==0] = 1
        BandpassFilter2D[-1,:,:] = lastFilter
        
        if doplot:
            plt.close()
            print('ScaleRatio = ', ScaleRatio,', n levels = ',NumberLevels)
            # Plot the filters
            # create ticks in km
            ticksList = []
            tickLocal = FFTSize*resKm
            for i in xrange(0,20):
                ticksList.append(tickLocal)
                tickLocal = tickLocal/2
                if tickLocal < resKm:
                    break
            ticks = np.array(ticksList)
            ticks_loc = 10.0*np.log10(1/ticks)
            ticksStr = ["%1.0f" % tick for tick in ticks]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_prop_cycle('color',plt.cm.tab10(np.linspace(0,1,NumberLevels)))
            for Level in xrange(NumberLevels):
                filter2d = BandpassFilter2D[Level,:,:]
                filter1d = filter2d[0,0:FFTStride+1]
                freq = np.linspace(1/(FFTSize*resKm),1/(2*resKm),filter1d.size)
                ax.plot(10*np.log10(freq),filter1d)
                ax.set_aspect('auto')
                ax.set_ylim([0,1.1])
                ax.set_xticks(ticks_loc)
                ax.set_xticklabels(ticksStr)
                ha='center'
                if Level==0 :
                    ha='left'
                elif Level==NumberLevels-1:
                    ha='right'   
                xpeak = freq[np.argmax(filter1d)]       
                ax.text(10*np.log10(xpeak),1.02,'Level %i' % Level,va='bottom',ha=ha,fontsize=14)
            plt.xlim([10*np.log10(freq[1/freq==512]),10*np.log10(freq[1/freq==resKm*2])])
            # plt.title('Band-pass filters for cascade levels')
            plt.ylabel('Filter value []',fontsize=14)
            plt.xlabel('Wavelength [km]',fontsize=14)
            # plt.show()
            plt.tight_layout()
            plt.savefig('fig_bandpass_filter_1d_%ilevels.pdf' % NumberLevels)
            print('Saved: fig_bandpass_filter_1d_%ilevels.pdf' % NumberLevels)
    else:
        BandpassFilter2D = np.ones((1,FFTShape[0],FFTShape[1]))
        CentreWaveLengths = np.zeros(NumberLevels)
        
    return BandpassFilter2D,CentreWaveLengths*resKm  
 
def get_cascade_from_stack(dbzStack, NumberLevels, BandpassFilter2D, CentreWaveLengths, zerothr = None, zeroPadding = 0, verbose=0, doplot=0):
    
    nrOfFields = len(dbzStack)
    cascadeStack = []
    cascadeMeanStack = []
    cascadeStdStack = []
    for n in xrange(nrOfFields):
    
        InputMap = dbzStack[n].copy()

        FFTSize = InputMap.shape[0]
        Nyquest = FFTSize/2
        
        Cascade = np.zeros((FFTSize,FFTSize,NumberLevels))
        CascadeMean = np.zeros(NumberLevels)
        CascadeStd = np.zeros(NumberLevels)
        
        # Zero padding
        if zeroPadding > 0:
            InputMap = cv2.copyMakeBorder(InputMap,zeroPadding,zeroPadding,zeroPadding,zeroPadding,cv2.BORDER_CONSTANT,0)
                
        # Calculate the FFT
        fftNoShift = np.fft.fft2(InputMap)
        
        # For each level, filter the transform and place the data into the cascade
        if (verbose==1) and (n==0):
            print('Applying the filters',end="")
            stdout.flush()
        
        for Level in xrange(NumberLevels):
            if verbose==1:
                print('.',end="")
                stdout.flush()
                       
            FFTPassedOut = np.zeros_like(fftNoShift)
            
            # Extract the filter for the given level
            Filter = BandpassFilter2D[Level,:,:].copy()

            # Apply the filter        
            FFTPassedOut =  fftNoShift * Filter
                    
            # Calculate the inverse ff
            new_image = np.real(np.fft.ifft2(FFTPassedOut)) 
            
            # Crop the zero edges
            if zeroPadding > 0:
                new_image = new_image[zeroPadding:-zeroPadding,zeroPadding:-zeroPadding]
            
            if zerothr is None:
                CascadeMean[Level] = new_image.mean()
                CascadeStd[Level] = new_image.std()
                Cascade[:,:,Level] = (new_image - CascadeMean[Level]) / CascadeStd[Level]
            else:
                wet_pixels = new_image > zerothr
                
                CascadeMean[Level] = new_image[wet_pixels].mean()
                CascadeStd[Level] = new_image[wet_pixels].std()
                new_image = (new_image - CascadeMean[Level]) / CascadeStd[Level]
                # new_image[~wet_pixels] = new_image[wet_pixels].min()
                Cascade[:,:,Level] = new_image
        
        if zeroPadding > 0:
                InputMap = InputMap[zeroPadding:-zeroPadding,zeroPadding:-zeroPadding]
        
        # The k level is the residuals
        k = NumberLevels-1
        CascadeSum = np.zeros((FFTSize,FFTSize))
        for LevelA in xrange(NumberLevels):
            if LevelA!=k:
                CascadeSum += Cascade[:,:,LevelA]*CascadeStd[LevelA] + CascadeMean[LevelA]
        Residual = InputMap - CascadeSum
        CascadeMean[k] = Residual.mean()
        CascadeStd[k] = Residual.std()
        Cascade[:,:,k] = (Residual - CascadeMean[k]) / CascadeStd[k]
        
        if (doplot==1) and (n==(nrOfFields-1)):
        
            fsize = 14
            
            if NumberLevels<3:
                ncols=2
            elif NumberLevels<7:
                ncols=3
            else:
                ncols=4
            
            nrows = np.ceil(NumberLevels/ncols) #+ 1
            
            plt.close()
            plt.figure(figsize=(5*ncols, 4.9*nrows))
            
            # plt.subplot(nrows,4,1)
            # plt.title('Original image')
            # plt.imshow(InputMap,interpolation='none',vmin=0,vmax=45)
            # cbar = plt.colorbar()
            # cbar.set_label('dBZ')
            # plt.axis('off')
            
            # CascadeSum = np.zeros(InputMap.shape)
            # for LevelA in xrange(NumberLevels):
                # if LevelA!=k:
                    # CascadeSum += Cascade[:,:,LevelA]*CascadeStd[LevelA] + CascadeMean[LevelA]
            # plt.subplot(nrows,4,2)
            # plt.title('Sum of levels 0 to %i' % (NumberLevels-2))
            # plt.imshow(CascadeSum,interpolation='none',vmin=0,vmax=45)
            # cbar=plt.colorbar()
            # cbar.set_label('dBZ')
            # plt.axis('off')

            recon_image = np.zeros(InputMap.shape)
            for nl in xrange(NumberLevels):
                plt.subplot(nrows,ncols,1+nl)
                plt.title('(%s) Level %i (%i km)' % (chr(97+nl),nl, CentreWaveLengths[nl]),fontsize=fsize)
                if nl==NumberLevels-1:
                    plt.title('(%s) Level %i (%i km + residuals)' % (chr(97+nl),nl, CentreWaveLengths[nl]),fontsize=fsize)
                nlevel = Cascade[:,:,nl].copy()*CascadeStd[nl] + CascadeMean[nl] 
                recon_image += nlevel
                vmax = np.percentile(nlevel,99.0)
                vmin = np.percentile(nlevel,1.0)
                ax = plt.gca()
                im = ax.imshow(nlevel,vmin=vmin,vmax=vmax,interpolation='none')
                plt.axis('off')  
                # plt.imshow(Cascade[:,:,nl],vmin=-5,vmax=5,interpolation='none')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label('dBZ',fontsize=fsize)
                
            
            # plt.subplot(nrows,4,3)
            # plt.title('Reconstructed image + residuals')
            # plt.imshow(recon_image,interpolation='nearest',vmin=0,vmax=45)
            # cbar = plt.colorbar()
            # cbar.set_label('dBZ')
            # plt.axis('off')

            # plt.show()
            plt.tight_layout()
            plt.savefig('fig_cascade_decomposition_%ilevels.pdf' % NumberLevels)
            print('Saved: fig_cascade_decomposition_%ilevels.pdf' % NumberLevels)
            
        # add to the stack
        cascadeStack.append(Cascade)    
        cascadeMeanStack.append(CascadeMean)
        cascadeStdStack.append(CascadeStd)
        
    if verbose==1:    
        print(' DONE!') 
    
    return cascadeStack, cascadeMeanStack, cascadeStdStack

def get_cascade_from_array(InputMap, NumberLevels, BandpassFilter2D, CentreWaveLengths, zerothr = None, zeroPadding = 0, squeeze=True, verbose=0, doplot=0):
    
    if InputMap.ndim == 2:
        InputMap = InputMap[None,:,:]
    
    FFTSize = InputMap.shape[1]
    Nyquest = FFTSize/2
    nrOfFields = InputMap.shape[0]
    
    Cascade = np.zeros((nrOfFields,FFTSize,FFTSize,NumberLevels))
    CascadeMean = np.zeros((nrOfFields,NumberLevels))
    CascadeStd = np.zeros((nrOfFields,NumberLevels))
    
    for i in xrange(nrOfFields):
    
        thisInputMap = InputMap[i,:,:].copy()
    
        # Zero padding
        if zeroPadding > 0:
            thisInputMap = cv2.copyMakeBorder(thisInputMap,zeroPadding,zeroPadding,zeroPadding,zeroPadding,cv2.BORDER_CONSTANT,0)
        
        # Calculate the FFT
        fftNoShift = np.fft.fft2(thisInputMap)
        
        # For each level, filter the transform and place the data into the cascade
        if (verbose==1) and (n==0):
            print('Applying the filters',end="")
            stdout.flush()
        
        for Level in xrange(NumberLevels):
            if verbose==1:
                print('.',end="")
                stdout.flush()
                       
            FFTPassedOut = np.zeros_like(fftNoShift)
            
            # Extract the filter for the given level
            Filter = BandpassFilter2D[Level,:,:].copy()

            # Apply the filter        
            FFTPassedOut =  fftNoShift * Filter
                    
            # Calculate the inverse ff
            new_image = np.real(np.fft.ifft2(FFTPassedOut)) 
            
            # Crop the zero edges
            if zeroPadding > 0:
                new_image = new_image[zeroPadding:-zeroPadding,zeroPadding:-zeroPadding]
            
            if zerothr is None:
                CascadeMean[i,Level] = new_image.mean()
                CascadeStd[i,Level] = new_image.std()
                Cascade[i,:,:,Level] = (new_image - CascadeMean[i,Level]) / CascadeStd[i,Level]
            else:
                wet_pixels = new_image > zerothr   
                CascadeMean[i,Level] = new_image[wet_pixels].mean()
                CascadeStd[i,Level] = new_image[wet_pixels].std()
                new_image = (new_image - CascadeMean[i,Level]) / CascadeStd[i,Level]
                # new_image[~wet_pixels] = new_image[wet_pixels].min()
                Cascade[i,:,:,Level] = new_image

        # The k level is the residuals
        k = NumberLevels-1
        CascadeSum = np.zeros(InputMap[i,:,:].shape)
        for LevelA in xrange(NumberLevels):
            if LevelA!=k:
                CascadeSum += Cascade[i,:,:,LevelA]*CascadeStd[i,LevelA] + CascadeMean[i,LevelA]
        Residual = InputMap[i,:,:] - CascadeSum
        CascadeMean[i,k] = Residual.mean()
        CascadeStd[i,k] = Residual.std()
        Cascade[i,:,:,k] = (Residual - CascadeMean[i,k]) / CascadeStd[i,k]
    
    if (doplot==1):
        vmaxorig = InputMap.max()
    
        nrows = np.ceil(NumberLevels/4) + 1

        plt.subplot(nrows,4,1)
        plt.title('Original image')
        plt.imshow(InputMap[0,:,:],interpolation='none',vmin=0,vmax=vmaxorig)
        cbar = plt.colorbar()
        # cbar.set_label('dBZ')
        plt.axis('off')
        
        CascadeSum = np.zeros(InputMap[0,:,:].shape)
        for LevelA in xrange(NumberLevels):
            if LevelA!=k:
                CascadeSum += Cascade[0,:,:,LevelA]*CascadeStd[0,LevelA] + CascadeMean[0,LevelA]
        
        plt.subplot(nrows,4,2)
        plt.title('Sum of levels 0 to %i' % (NumberLevels-2))
        plt.imshow(CascadeSum,interpolation='none',vmin=0,vmax=vmaxorig)
        cbar=plt.colorbar()
        # cbar.set_label('dBZ')
        plt.axis('off')

        recon_image = np.zeros(InputMap[0,:,:].shape)
        for nl in xrange(NumberLevels):
            plt.subplot(nrows,4,5+nl)
            plt.title('Level %i (%i km)' % (nl, CentreWaveLengths[nl]))
            if nl==NumberLevels-1:
                plt.title('Level %i (%i km + residuals)' % (nl, CentreWaveLengths[nl]))
            nlevel = Cascade[0,:,:,nl].copy()*CascadeStd[0,nl] + CascadeMean[0,nl] 
            recon_image += nlevel
            vmax = np.percentile(nlevel,99.0)
            vmin = np.percentile(nlevel,1.0)
            plt.imshow(nlevel,vmin=vmin,vmax=vmax,interpolation='none')
            # plt.imshow(Cascade[:,:,nl],vmin=-5,vmax=5,interpolation='none')
            cbar = plt.colorbar()
            # cbar.set_label('dBZ')
            plt.axis('off')
        
        plt.subplot(nrows,4,3)
        plt.title('Reconstructed image + residuals')
        plt.imshow(recon_image,interpolation='nearest',vmin=0,vmax=vmaxorig)
        cbar = plt.colorbar()
        # cbar.set_label('dBZ')
        plt.axis('off')

        # plt.show()
        plt.savefig('cascade_decomposition.pdf')
        print('Saved: cascade_decomposition.pdf')
    
    if nrOfFields == 1 and squeeze:
        Cascade = Cascade[0,:,:,:]
        CascadeMean = CascadeMean[0,:]
        CascadeStd = CascadeStd[0,:]
    
    return Cascade,CascadeMean,CascadeStd

def get_cascade_with_wavelets(rainfield, nrLevels=6, wavelet = 'db4', doplot=0):
    
    rainfieldSize = rainfield.shape
    
    # Decompose rainfall field
    coeffsRain = pywt.wavedec2(rainfield, wavelet, level=nrLevels)
    
    
    if (doplot==1):
        vmaxorig = rainfield.max()
        ncols = 3
        nrows = np.ceil(nrLevels/ncols) + 1

        plt.subplot(nrows,ncols,1)
        plt.title('Original image')
        plt.imshow(rainfield,interpolation='none',vmin=0,vmax=vmaxorig)
        cbar = plt.colorbar()
        # cbar.set_label('dBZ')
        plt.axis('off')
        
        recomposedCascade = pywt.waverec2(coeffsRain, wavelet)
        
        plt.subplot(nrows,ncols,2)
        plt.title('Reconstructed field')
        plt.imshow(recomposedCascade,interpolation='none',vmin=0,vmax=vmaxorig)
        cbar=plt.colorbar()
        plt.axis('off')

        for nl in xrange(nrLevels):
            plt.subplot(nrows,ncols,ncols+1+nl)
            # plt.title('Level %i (%i km)' % (nl, CentreWaveLengths[nl]))
            nlevel = coeffsRain[nl][0].copy()
            print(nlevel.shape)
            vmax = np.percentile(nlevel,99.0)
            vmin = np.percentile(nlevel,1.0)
            plt.imshow(nlevel,vmin=vmin,vmax=vmax,interpolation='none')
            cbar = plt.colorbar()
            plt.axis('off')

        plt.show()

    return Cascade 
    
def autoregressive_parameters(cascadeStack, cascadeMeanStack, cascadeStdStack, order):
    NumberLevels = cascadeStack[0].shape[2]
    NumberLags = len(cascadeStack) - 1
    phi = np.zeros((NumberLevels,2))
    # lag-n autocorrelation
    r = np.zeros((NumberLevels,NumberLags))
    for n in xrange(NumberLags):
        for l in xrange(NumberLevels):

            array1 = cascadeStack[NumberLags][:,:,l].flatten()*cascadeStdStack[NumberLags][l] + cascadeMeanStack[NumberLags][l] 
            array2 = cascadeStack[NumberLags-1-n][:,:,l].flatten()*cascadeStdStack[NumberLags-1-n][l] + cascadeMeanStack[NumberLags-1-n][l] 
            # conditional correlation coefficient
            # thr = np.max((array1.min(),array2.min()))
            thr = -999
            idx1 = (array1>thr) 
            idx2 = (array2>thr)
            idx3 = ~np.logical_or(np.isnan(array1),np.isnan(array2))
            idx = idx1 * idx2 * idx3
            # print('%i' % ((array1.size - np.sum(idx))/array1.size*100))
            if any(idx):
                # r[l,n] = np.min(np.corrcoef(array1[idx],array2[idx])) # pearson
                r[l,n] = pearsonr(array1[idx],array2[idx])[0] # pearson
                # r[l,n] = spearmanr(array1[idx],array2[idx])[0] # spearman 
            else:
                r[l,n] = 0
    r[np.isnan(r)] = 0.0
    
    # correct correlation coefficients
    if NumberLevels>1:
        # cf = 1/(1.01 - 0.0005*np.arange(NumberLevels)**3.5)
        cf = 1
        for n in xrange(NumberLags):
            r[:,n] *= cf
        # r[0,:] *= 0.99    
        print('Corrected correlation coefficients:')
        print('\n'.join('{}: {}'.format(*k) for k in enumerate(r)))
    else:
        print('Correlation coefficients:')
        print('\n'.join('{}: {}'.format(*k) for k in enumerate(r)))
    
    if order==1:    # AR(1)
        phi[:,0] = r[:,0]
        
    elif order==2:  # AR(2) (Wilks pag. 416)
    
        # Yule-Walker equations
        phi[:,0] =  r[:,0]*(1 - r[:,1])/(1 - r[:,0]**2) # phi1 
        phi[:,1] = (r[:,1] - r[:,0]**2)/(1 - r[:,0]**2) # phi2 
        
        # Criteria to make sure the AR(2) process is stationary
        criteria1 = (phi[:,1] + phi[:,0])<1
        criteria2 = (phi[:,1] - phi[:,0])<1
        criteria3 = (phi[:,1])>-1
        criteria4 = (phi[:,1])<1
        criteria = criteria1*criteria2*criteria3*criteria4
        # print(criteria)
        
        # 
        phi[~criteria,0] = r[~criteria,0]
        phi[~criteria,1] = 0.0
    else:
        print('Error: AR(%i) not implemented yet!' % order);sys.exit()
        

    return phi,r
    
def get_perturbation_fields(radarImage, NumberMembers, NumberLeadtimes, winsize = [], local_level = 0, seed = 42):
    totfields = int(NumberMembers*(NumberLeadtimes + 2))
    if not winsize:
        perturbationFields = ssft.nested_fft2(radarImage, nr_frames = totfields, max_level = local_level, seed = seed)
    else:
        perturbationFields,_,_ = ssft.corrNoise(radarImage, winsize = winsize, nmembers = totfields, verbose = 0, fillgaps = 1, seed = seed)
    
    noiseStack = []
    for n  in xrange(totfields):
        noiseStack.append(perturbationFields[:,:,n])
    return noiseStack
    
def aggregate_in_time(dataArray,timeAccumMin,type='sum'):
    
    # flatten 2d fields (rows with time, columns with space)
    origShape = dataArray.shape
    dataArrayFlat = np.zeros((origShape[2],origShape[0]*origShape[1]))
    for t in xrange(origShape[2]):
        dataArrayFlat[t,:] = dataArray[:,:,t].flatten()
    
    accumFactor = np.int(timeAccumMin/5.0)
    if type=='sum':
        dataArrayFlatAcc =  dataArrayFlat.reshape(int(dataArrayFlat.shape[0]/accumFactor), accumFactor, dataArrayFlat.shape[1]).sum(axis=1)
        # test = np.allclose(dataArrayFlatAcc[0,1000:2000].sum(), dataArrayFlat[0:2,1000:2000].sum())
    elif type=='mean':
        dataArrayFlatAcc =  dataArrayFlat.reshape(int(dataArrayFlat.shape[0]/accumFactor), accumFactor, dataArrayFlat.shape[1]).mean(axis=1)
    elif type=='nansum':
        dataArrayFlatAcc =  np.nansum(dataArrayFlat.reshape(int(dataArrayFlat.shape[0]/accumFactor), accumFactor, dataArrayFlat.shape[1]), axis=1)
    

    # reshape as original field
    newDataArray = np.zeros((origShape[0],origShape[1],int(origShape[2]/accumFactor)))
    for t in xrange(int(origShape[2]/accumFactor)):
        newDataArray[:,:,t] = dataArrayFlatAcc[t,:].reshape(origShape[0],origShape[1])
    
    return newDataArray
 
def top_flat_hanning(winsize):
    T = winsize/4
    W = winsize/2
    B=np.linspace(-W,W,2*W)
    R = np.abs(B)-T
    R[R<0]=0.
    A = 0.5*(1.0 + np.cos(np.pi*R/T))
    A[np.abs(B)>(2*T)]=0.0
    w1d = A   
    wind = np.sqrt(np.outer(w1d,w1d))
    return wind
    
def to_dBR(R, rainThr = 0.08):
    R[R<=0] = rainThr
    return np.log10(R)
    
def from_dBR(dBR, rainThr = 0.08):
    R = 10**dBR
    R[R<=rainThr] = 0
    return R
    
def add_nans(A, rainThr = 0.08):
    A[A<=rainThr]=np.nan
    return A