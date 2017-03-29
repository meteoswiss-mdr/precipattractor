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
import datetime as datetime
import pandas as pd

import numpy as np

#### Functions to converty reflectivity to rainfall and vice-versa
def to_dB(array, offset=-1):
    '''
    Transform array to dB
    '''
    isList = isinstance(array, list)
    if isList == True:
        array = np.array(array)
    if offset != -1:
        print('ciaoooo')
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
    
def rainrate2reflectivity(rainrate, A=316.0, b=1.5, zerosDBZ='auto'):

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
    
    if type(rainrate) == np.ndarray and len(rainrate) > 1:
        # Compute reflectivity
        dBZ = rainrate.copy()
        dBZ[rainIdx] = 10.0*np.log10(A*rainrate[rainIdx]**b)
        
        # Replace zero rainrate by the minimum observed reflectivity or set it by hand to a fixed value
        if zerosDBZ == 'auto':
            dBZ[zerosIdx] = minDBZ
        else:
            dBZ[zerosIdx] = zerosDBZ
    else:
        if rainrate != 0.0:
            dBZ = 10.0*np.log10(A*rainrate**b)
        else:
            if zerosDBZ == 'auto':
                dBZ = minDBZ
            else:
                dBZ = zerosDBZ
    
    return dBZ, minDBZ, minRainRate
    
def reflectivity2rainrate(reflectivityDBZ, zeroDBZ=0.0, A=316.0, b=1.5):
    rainrate = (10.0**(reflectivityDBZ/10.0)/A)**(1.0/b)
    
    # Replace zero rain rate with 0.0
    zerorainrate = (10.0**(zeroDBZ/10.0)/A)**(1.0/b)
    rainrate[rainrate <= zerorainrate] = 0.0
    
    return(rainrate)
  
def get_rainfall_lookuptable(noData, A=316.0, b=1.5):
    precipIdxFactor=71.5
    lut = np.zeros(256)
    for i in range(0,256):
        if (i < 2) or (i > 250 and i < 255):
            lut[i] = 0.0
        elif (i == 255):
            lut[i] = noData
        else:
            lut[i] = (10.**((i-(precipIdxFactor))/20.0)/A)**(1.0/b)
    
    return lut
#########
    
def get_column_list(list2D, columnNr):
    listColumn = [item[columnNr] for item in list2D]
    return(listColumn)

def get_variable_indices(subsetVariableNames, listVariableNames):
    '''
    Function to return the linear indices of the subset of variables in a longer list of variables
    '''
    
    # Transforms input variables to lists if necessary
    if type(subsetVariableNames) is not list:
        if type(subsetVariableNames) is np.ndarray:
            subsetVariableNames = subsetVariableNames.tolist()
        else:
            subsetVariableNames = [subsetVariableNames]
            
    if type(listVariableNames) is not list:
        if type(listVariableNames) is np.ndarray:
            listVariableNames = listVariableNames.tolist()
        else:
            listVariableNames = [listVariableNames]
    
    nrVarTot = len(listVariableNames)
    nrVarSubset = len(subsetVariableNames)

    indices = []
    for item in range(0,nrVarSubset):
        var = subsetVariableNames[item]
        index = listVariableNames.index(var)
        indices.append(index)
    return(indices)

def get_reduced_extent(width, height, domainSizeX, domainSizeY):
    '''
        Function to get the indices of a central reduced extent of size (domainSizeX, domainSizeY) within a larger 2D array of size (width, height)
    '''

    if ((width - domainSizeX) % 2) == 0:
        borderSizeX = (width - domainSizeX)/2
    else:
        print('Problem in get_reduced_extent. Non-even border size in X dimension.')
        sys.exit(1)
    if ((height - domainSizeY) % 2) == 0:
        borderSizeY = (height - domainSizeY)/2
    else:
        print('Problem in get_reduced_extent. Non-even border size in Y dimension.')
        sys.exit(1)
        
    # extent
    extent = (int(borderSizeX), int(borderSizeY), int(width-borderSizeX), int(height-borderSizeY)) # left, upper, right, lower
    
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

def get_colorlist(type='MeteoSwiss', units='R'):
    if type == 'STEPS':
        color_list = ['cyan','deepskyblue','dodgerblue','blue','chartreuse','limegreen','green','darkgreen','yellow','gold','orange','red','magenta','darkmagenta']
        if units == 'R':
            clevs = [0.1,0.25,0.4,0.63,1,1.6,2.5,4,6.3,10,16,25,40,63,100]
        elif units == 'dBZ':
            clevs = np.arange(-10,70,5)
        else:
            print('Wrong units in get_colorlist')
            sys.exit(1)
    if type == 'MeteoSwiss':
        pinkHex = '#%02x%02x%02x' % (232, 215, 242)
        redgreyHex = '#%02x%02x%02x' % (156, 126, 148)
        color_list = [pinkHex, redgreyHex, "#640064","#AF00AF","#DC00DC","#3232C8","#0064FF","#009696","#00C832",
        "#64FF00","#96FF00","#C8FF00","#FFFF00","#FFC800","#FFA000","#FF7D00","#E11900"] # light gray "#D3D3D3"
        if units == 'R':
            clevs= [0,0.08,0.16,0.25,0.40,0.63,1,1.6,2.5,4,6.3,10,16,25,40,63,100,160]
        elif units == 'dBZ':
            clevs = np.arange(-10,70,5)
        else:
            print('Wrong units in get_colorlist')
            sys.exit(1)
    
    # Color level strings    
    clevsStr = []
    for i in range(0,len(clevs)):
        if (clevs[i] < 10) and (clevs[i] >= 1):
            clevsStr.append(str('%.1f' % clevs[i]))
        elif (clevs[i] < 1):
            clevsStr.append(str('%.2f' % clevs[i]))
        else:
            clevsStr.append(str('%i' % clevs[i]))
        
    return(color_list, clevs, clevsStr)

def dynamic_formatting_floats(floatArray):
    if type(floatArray) == list:
        floatArray = np.array(floatArray)
        
    labels = []
    for label in floatArray:
        if label >= 0.1 and label < 1:
            formatting = ',.1f'
        elif label >= 0.01 and label < 0.1:
            formatting = ',.2f'
        elif label >= 0.001 and label < 0.01:
            formatting = ',.3f'
        elif label >= 0.0001 and label < 0.001:
            formatting = ',.4f'
        elif label >= 1 and label.is_integer():
            formatting = 'i'
        else:
            formatting = ',.1f'
            
        if formatting != 'i':
            labels.append(format(label, formatting))
        else:
            labels.append(str(int(label)))
        
    return labels
    
def update_xlabels(ax):
    '''
    Does not work yet
    '''
    xlabels = []
    for label in ax.get_xticks():
        if label >= 0.1 and label < 1:
            formatting = ',.1f'
        elif label >= 0.01 and label < 0.1:
            formatting = ',.2f'
        elif label >= 0.001 and label < 0.01:
            formatting = ',.3f'
        elif label >= 0.0001 and label < 0.001:
            formatting = ',.4f'
        else:
            formatting = 'i'
        
        print(label, formatting)
        if formatting != 'i':
            xlabels.append(format(label, formatting))
        else:
            xlabels.append(int(label))
    
    ax.set_xticklabels(xlabels)

def myDecimalFormat(y,pos):
    '''
    Does not work yet

    '''
    # Find the number of decimal places required
    if y >= 0.1 and y < 1:
        decimalplaces = 1
    elif y >= 0.01 and y < 0.1:
        decimalplaces = 2
    elif y >= 0.001 and y < 0.01:
        decimalplaces = 3
    elif y >= 0.0001 and y < 0.001:
        decimalplaces = 4
    else:
        decimalplaces = 0
    
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)
    
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
    
def update_progress(progress, processName = "Progress"):
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
    
def optimal_size_subplot(nrPlots):
    # Get divisors of number of plots
    div = divisors(nrPlots)
    
    if nrPlots == 1:
        nrRows = 1
        nrCols = 1
    elif nrPlots == 2:
        nrRows = 1
        nrCols = 2
    elif nrPlots == 3:
        nrRows = 1
        nrCols = 3
    elif nrPlots == 4:
        nrRows = 2
        nrCols = 2
    elif nrPlots == 5 or nrPlots == 6:
        nrRows = 2
        nrCols = 3
    elif nrPlots == 7:
        nrRows = 2
        nrCols = 4
    elif nrPlots == 8 or nrPlots == 9:
        nrRows = 3
        nrCols = 3
    elif nrPlots >= 10 and nrPlots <= 12:
        nrRows = 3
        nrCols = 4
    elif nrPlots >= 13 and nrPlots <= 15:
        nrRows = 3
        nrCols = 5
    elif nrPlots >= 16 and nrPlots <= 20:
        nrRows = 4
        nrCols = 5
    elif np.sqrt(nrPlots).is_integer():
        nrRows = int(np.sqrt(nrPlots))
        nrCols = int(np.sqrt(nrPlots))
    elif len(div) > 1:
        nrRows = int(np.sqrt(nrPlots))
        idxCol = np.where(div > nrRows)[0]
        
        nrCols = div[idxCol[0]]

        # remove unnecessary additional columns
        nrColsMax = nrCols.copy()
        for i in range(1,20):
            if nrRows*(nrColsMax-i) >= nrPlots:
                nrCols = nrColsMax - i
            else:
                break;
        
    else:
        nrRows = int(np.sqrt(nrPlots))
        nrCols = int(np.sqrt(nrPlots)+1)
        
        nrColsMax = nrCols
        # increase columns if necessary
        for i in range(0,20):
            if nrPlots >= nrRows*(nrColsMax+i):
                nrCols = nrColsMax + i + 1
            else:
                break;
    
    if nrRows*nrCols < nrPlots:
        print('Not enough rows and columns to draw all the subplots.')
        print('Consider updating the function optimal_size_subplot.')
        sys.exit(1)
    
    return(nrRows, nrCols)

def divisors(number):
    n = 1
    div = []
    while(n<number):
        if(number%n==0):
            div.append(n)
        else:
            pass
        n += 1
    div = np.array(div)
    return(div)

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index. Last index is not included (not need to add +1)
    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx
    
def fill_attractor_array_nan(arrayStats, timeStamps_datetime, timeSampMin = 5):
    
    isStatsArrayNumpy = False
    isTimesArrayNumpy = False
    if type(timeStamps_datetime) == np.ndarray:
        timeStamps_datetime = timeStamps_datetime.tolist()
        isTimesArrayNumpy = True
        
    if type(arrayStats) == np.ndarray:
        arrayStats = arrayStats.tolist()
        isStatsArrayNumpy = True
    
    if len(timeStamps_datetime) == len(arrayStats):
        nrSamples = len(timeStamps_datetime)
    else:
        print("arrayStats, timeStamps_datetime in fill_attractor_array_nan should have the same number of rows.")
        sys.exit(1)
    
    # Prepare list of NaNs
    emptyListNaN = np.empty((len(arrayStats[0])-1)) # without the first column
    emptyListNaN[:] = np.nan
    emptyListNaN = emptyListNaN.tolist()

    tend = timeStamps_datetime[nrSamples-1]
    tstart = timeStamps_datetime[0]
    t = 0
    while tstart < tend:
        diffTimeSecs = (timeStamps_datetime[t+1] - timeStamps_datetime[t]).total_seconds()
        if diffTimeSecs > timeSampMin*60:
            missingDateTime = timeStamps_datetime[t] + datetime.timedelta(minutes = timeSampMin)
            # Insert missing date time
            timeStamps_datetime.insert(t+1, missingDateTime)
            # Insert line of NaNs in arrayStats (except date)
            missingDate = int(missingDateTime.strftime("%Y%m%d%H%M%S"))
            arrayStats.insert(t+1,[missingDate] + emptyListNaN)
        t = t+1
        tstart = tstart + datetime.timedelta(minutes = timeSampMin)
    
    if isStatsArrayNumpy:
        arrayStats = np.asarray(arrayStats)
    if isTimesArrayNumpy:
        timeStamps_datetime = np.asarray(timeStamps_datetime)
    
    return(arrayStats, timeStamps_datetime)
    
def fill_attractor_array_nan2(arrayStats, timeStamps_datetime, timeSampMin = 5):
    '''
    Attempt to make the previous function faster. Very expensive to convert list of lists to np.ndarray.
    np.insert even slower...
    '''
    if len(timeStamps_datetime) == len(arrayStats):
        nrSamples = len(timeStamps_datetime)
    else:
        print("arrayStats, timeStamps_datetime in fill_attractor_array_nan should have the same number of rows.")
        sys.exit(1)
    
    # Prepare list of NaNs
    emptyListNaN = np.empty((len(arrayStats[0]))) # without the first column
    emptyListNaN[:] = np.nan
    
    # df = pd.DataFrame(timeStamps_datetime)
    # df = pd.to_datetime(df)
    # df = df.resample('D').fillna(0)

    # print(df)
    # df = df.resample('5T').sum()
    # print(df)
    # # df = df.fillna(np.nan)
    # print(df)
    # sys.exit()
    
    tend = timeStamps_datetime[nrSamples-1]
    tstart = timeStamps_datetime[0]
    t = 0
    while tstart < tend:
        diffTimeSecs = (timeStamps_datetime[t+1] - timeStamps_datetime[t]).total_seconds()
        if diffTimeSecs > timeSampMin*60:
            missingDateTime = timeStamps_datetime[t] + datetime.timedelta(minutes = timeSampMin)
            # Insert missing date time
            np.insert(timeStamps_datetime, t+1, missingDateTime, axis=0)
            # Insert line of NaNs in arrayStats (except date)
            missingDate = int(missingDateTime.strftime("%Y%m%d%H%M%S"))
            np.insert(arrayStats, t+1, emptyListNaN, axis=0)

        t = t+1
        tstart = tstart + datetime.timedelta(minutes = timeSampMin)
    
    return(arrayStats, timeStamps_datetime)
    
def print_list_vertical(letters):
    for s1,s2 in zip(letters[:len(letters)//2], letters[len(letters)//2:]): #len(letters)/2 will work with every paired length list
       print(s1,s2)

def unique_rows(array):
    new_array = [tuple(row) for row in array]
    uniques = np.unique(new_array)
    return(uniques)
    