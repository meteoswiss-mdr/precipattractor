#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import numpy as np
import sys

import cv2

from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt
###############################################
############################################### 
    
def ShiTomasi_features_to_track(image, maxCorners=100, qualityLevel=0.1, minDistance=10, blockSize=3):
    '''
    Function to call the Shi-Tomasi corner detection algorithm
    '''
    
    # ShiTomasi corner detection parameters 
    ShiTomas_params = dict( maxCorners = maxCorners, qualityLevel = qualityLevel, \
        minDistance = minDistance, blockSize = blockSize ) 
    
    # Detect corners
    p0 = cv2.goodFeaturesToTrack(image, mask = None, **ShiTomas_params)
    nCorners = p0.shape[0] 
    
    return(p0, nCorners)
      
def threshold_features_to_track(image, maxCorners, minThr = 0.08, blockSize = 30):
    '''
    Local maxima detection
    '''
    
    image = image.astype(float)
    idxWetPixels = image > minThr
    wetPixels = image[idxWetPixels]
    nWetPixels = np.sum(idxWetPixels)
    q = (maxCorners/nWetPixels)*100
    thr = np.floor(np.percentile(wetPixels,q))
    
    # normalize 0-1
    thr = thr*1.0/image.max()
    image *= 1.0/image.max()
    

    nCorners = maxCorners+1
    iter = 0
    miter = 500
    while (nCorners > maxCorners) & (iter < miter):
        iter = iter+1
        imageWet = image.copy()
        imageWet[imageWet <= thr] = 0
        image_max = filters.maximum_filter(imageWet,size=blockSize)
        maxima = (imageWet == image_max)
        image_min = filters.minimum_filter(imageWet,size=blockSize)
        notminima = (imageWet != image_min)
        maxima = maxima*notminima
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy,dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2    
            y.append(y_center)
        nCorners = len(x)
        thr=thr*1.05

    p0 = np.vstack((x,y)).T
    p0 = p0.reshape((nCorners,1,2))
    p0 = np.float32(p0)

    return(p0, nCorners)
    
    
def LucasKanade_features_tracking(prvs, next, p0, winSize=(35,35), maxLevel=10):
    '''
    Function to call the Lucas-Kanade features tracking algorithm
    '''
    
    # LK parameters
    lk_params = dict( winSize  = winSize, maxLevel = maxLevel, \
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # call Lucas-Kande function
    p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)

    # keep only features that have been found
    st = st[:,0]==1
    p1 = p1[st,:,:]
    p0 = p0[st,:,:]
    err = err[st,:]    
    
    # extract vectors                  
    x0 = p0[:,:,0]
    y0 = p0[:,:,1]       
    u = (p1-p0)[:,:,0]
    v = (p1-p0)[:,:,1]    
    
    return(x0, y0, u, v, err)    
    
    
def declustering(x, y, u, v, R, minN=3):
    '''
    Function to filter out outliers and get more representative data points.
    It assigns data points to a (RxR) declustering grid and then take the median of all values within one cell.
    '''
    
    # make sure these are vertical arrays
    x = x.reshape(x.size,1)
    y = y.reshape(y.size,1)
    u = u.reshape(u.size,1)
    v = v.reshape(v.size,1)
    
    # trasform coordinates into units of R
    xT = x/R
    yT = y/R
       
    # round coordinates to nearest integer 
    xT = np.floor(xT)
    yT = np.floor(yT)

    # keep only unique combinations of coordinates
    xy = np.hstack((xT,yT))
    unique_xy = unique_rows(xy)
    
    # now loop through these unique values and average vectors which belong to the same unit
    xN=[]; yN=[]; uN=[]; vN=[]
    for i in range(0,unique_xy.shape[0]):
        idx = (xT==unique_xy[i,0])&(yT==unique_xy[i,1])
        npoints = np.sum(idx)
        if npoints >= minN:
            xtemp = x[idx]
            ytemp = y[idx]
            utemp = u[idx]
            vtemp = v[idx]
            
            xN.append(np.median(xtemp))
            yN.append(np.median(ytemp))
            uN.append(np.median(utemp))
            vN.append(np.median(vtemp))
    
    # extract declustered values
    xP = np.array(xN)
    yP = np.array(yN) 
    uP = np.array(uN)
    vP = np.array(vN) 

    
    return(xP, yP, uP, vP)
    
def morphological_opening(image, thr=0.08, n=3):
    '''
    Function to apply a binary morphological opening to filter small isolated echoes
    
    Parameters
    ----------


    Returns
    -------

    '''
    
    # convert to binary image (rain/no rain)
    fieldBin = np.ndarray.astype(image > thr,'uint8')
    
    # build a structuring element of size (nx)
    kernel = np.ones((n,n),np.uint8)    
    
    # apply morphological opening (i.e. erosion then dilation)
    fieldBinOut = cv2.morphologyEx(fieldBin, cv2.MORPH_OPEN, kernel)
    
    # build mask to be applied on the original image
    mask = fieldBin - fieldBinOut
    mask = mask > 0
    image[mask] = 0
    
    return(image)

def gaussian_kernel(distance,bandwidth):
    '''
    Gaussian kernel
    '''
    
    out = 1/np.sqrt(2*np.pi)*np.exp(-0.5*(distance/float(bandwidth))**2)
    
    return(out)
    
    
def silverman(sigma,n):
    '''
    Silverman's rule of thumb for the estimation of the bandwidth 
    '''
    
    bandwidth = sigma*(4/3./float(n))**(1/5.)
    
    return(bandwidth)

    
def interpolate_sparse_vectors_kernel(x, y, u, v, domainSize, b = []):
    '''
    Gaussian kernel interpolation to obtain a dense field of motion vectors
    '''

    # make sure these are vertical arrays
    x = x.reshape(x.size,1)
    y = y.reshape(y.size,1)
    u = u.reshape(u.size,1)
    v = v.reshape(v.size,1)
    
    if len(domainSize)==1:
        domainSize=[domainSize,domainSize]
        
    # generate the grid
    xgrid = np.arange(domainSize[1])
    ygrid = np.arange(domainSize[0])
    X, Y = np.meshgrid(xgrid,ygrid)
    grid = np.column_stack((X.flatten(),Y.flatten()))
    
    # compute  distances
    points = np.column_stack((x,y))
    D = cdist(points,grid, 'euclidean')
    
    # get bandwidth if empty argument
    if not b:
        n = points.shape[0]
        sigma = np.std(D[:])
        b = silverman(sigma,n)
    
    # compute kernel weights
    weights = gaussian_kernel(D,b)
   
    uR = np.repeat(u,weights.shape[1]).reshape(weights.shape)
    vR = np.repeat(v,weights.shape[1]).reshape(weights.shape)
    
    # perform weighted average on the grid
    U = np.sum(weights*u,axis=0)/np.sum(weights,axis=0)
    V = np.sum(weights*v,axis=0)/np.sum(weights,axis=0)
    
    # reshape back to domain size
    U = U.reshape(domainSize[0],domainSize[1])
    V = V.reshape(domainSize[0],domainSize[1])
    
    return(xgrid, ygrid, U, V)
    
    
def interpolate_sparse_vectors_linear(x, y, u, v, domainSize):
    '''
    Linear interpolation to obtain a dense field of motion vectors.
    Extrapolation is performed using a nearest-neighbours approach.
    '''
    
    if len(domainSize)==1:
        domainSize=[domainSize,domainSize]
    
    # make sure these are vertical arrays
    x = x.reshape(x.size,1)
    y = y.reshape(y.size,1)
    u = u.reshape(u.size,1)
    v = v.reshape(v.size,1)
    
    # generate the grid
    xgrid = np.arange(domainSize[1])
    ygrid = np.arange(domainSize[0])
    X, Y = np.meshgrid(xgrid,ygrid)
   
    # first linear interpolation
    grid = np.column_stack((X.flatten(),Y.flatten()))
    points = np.column_stack((x,y))
    method = 'linear'
    U = griddata(points, u[:], grid, method = method)
    V = griddata(points, v[:], grid, method = method)
    # reshape back to domain size
    U = U.reshape(domainSize[0],domainSize[1])
    V = V.reshape(domainSize[0],domainSize[1])
    
    # then extrapolation by nearest neighbour
    idWithin = np.isfinite(U)
    points = np.column_stack((X[idWithin].flatten(),Y[idWithin].flatten()))
    method = 'nearest'
    U = griddata(points, U[idWithin], grid, method = method)
    V = griddata(points, V[idWithin], grid, method = method) 
    U = U.reshape(domainSize[0],domainSize[1])
    V = V.reshape(domainSize[0],domainSize[1])
    
    return(xgrid, ygrid, U, V)
    
    
def reduce_field_density_for_plotting(x, y, u, v, gridSpacing):
    '''
    Reduce density of arrows for plotting.
    '''
    domainSize = u.shape
    uSub = []; vSub = []; xSub = []; ySub = []
    for i in range(0,domainSize[0]):
        for j in range(0,domainSize[1]):
            if ((i % gridSpacing) == 0) & ((j % gridSpacing) == 0):
                uSub.append(u[i,j])
                vSub.append(v[i,j])
                xSub.append(x[j])
                ySub.append(y[i])
    xSub = np.asarray(xSub)
    ySub = np.asarray(ySub)  
    uSub = np.asarray(uSub)
    vSub = np.asarray(vSub)
    
    return(xSub, ySub, uSub, vSub)
    
   
def unique_rows(a):
    '''
    Returns unique combinations of rows in the matrix a.
    '''
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    
    return(a[idx])
    