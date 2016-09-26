#!/usr/bin/env python
'''
Module to perform various data operations.

Documentation convention from https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

30.08.2016
Loris Foresti
'''

from __future__ import division
from __future__ import print_function

import sys
import math 
import time 

import numpy as np

#### Functions to converty reflectivity to rainfall and vice-versa
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

def from_dB(array):
    '''
    Transform array from dB back to linear units
    '''
    isList = isinstance(array, list)
    if isList == True:
        array = np.array(array)
    
    linearArray = 10.0**(array/10.0)
        
    if isList == True:
        linearArray = linearArray.tolist()
    return(linearArray)
    
def rainrate2reflectivity(rainrate, A, b, zerosDBZ='auto'):

    isList = isinstance(rainrate, list)
    if isList == True:
        rainrate = np.array(rainrate)
        
    zerosIdx = rainrate == 0
    rainIdx = rainrate > 0
    
    # Compute or set minimum reflectivity
    nrRainPixels = np.sum(rainIdx)
    if nrRainPixels >= 10:
        minRainRate = np.min(rainrate[rainIdx])
    else:
        minRainRate = 0.012 # 0.0115537519713
    minDBZ = 10.0*np.log10(A*minRainRate**b)

    # Compute reflectivity
    dBZ = rainrate.copy()
    dBZ[rainIdx] = 10.0*np.log10(A*rainrate[rainIdx]**b)
    
    # Replace zero rainrate by the minimum observed reflectivity or set it by hand to a fixed value
    if zerosDBZ == 'auto':
        dBZ[zerosIdx] = minDBZ
    else:
        dBZ[zerosIdx] = zerosDBZ
    
    return dBZ, minDBZ, minRainRate
    
def reflectivity2rainrate(reflectivityDBZ, A, b):
    rainrate = (10.0**(reflectivityDBZ/10.0)/A)**(1.0/b)
    return(rainrate)
  
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
#########
    
def get_column_list(list2D, columnNr):
    listColumn = [item[columnNr] for item in list2D]
    return(listColumn)

def get_variable_indices(subsetVariableNames, listVariableNames):
    '''
    Function to return the linear indices of the subset of variables in a longer list of variables
    '''

    if type(subsetVariableNames) is not list:
        if len(subsetVariableNames) == 1:
            subsetVariableNames = [subsetVariableNames]
        else:
            subsetVariableNames = subsetVariableNames.tolist()
        
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
   
def myLogFormat(y,pos):
    '''
    Function to format the ticks labels of a loglog plot
    from 10^-1,10^0, 10^1, 10^2 to 0.1, 1, 10, 100
    Use as:
    axSP.xaxis.set_major_formatter(ticker.FuncFormatter(dt.myLogFormat))
    axSP.yaxis.set_major_formatter(ticker.FuncFormatter(dt.myLogFormat))
    '''
    
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)
    
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

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def unique(array):
    ''' 
    Overload of numpy unique function to obtain the values in the same order they appear in the array (no sorting)
    '''
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]
    
def update_progress(progress,processName = "Progress"):
    '''
    update_progress() : Displays or updates a console progress bar
    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%
    Source: https://stackoverflow.com/questions/3160699/python-progress-bar/15860757#15860757
    '''
    
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt!\r\n"
    if progress >= 1:
        progress = 1
        status = "Done!\r\n"
    block = int(round(barLength*progress))
    text = "\r{0}: [{1}] {2}% {3}".format(processName, "#"*block + "-"*(barLength-block), int(progress*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()
