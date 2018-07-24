#!/usr/bin/env python
'''
Module to perform various data operations.

Documentation convention from https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

30.08.2016
Loris Foresti
'''

from __future__ import division
from __future__ import print_function

import os
import sys
import math 
import time 
import datetime as datetime
import pandas as pd

from osgeo import gdal, osr, ogr
import geo
from scipy import ndimage
from scipy.interpolate import griddata, Rbf, LinearNDInterpolator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import stat_tools_attractor as st

def linear_rescaling(value, oldmin, oldmax, newmin, newmax):
    newvalue= (newmax-newmin)/(oldmax-oldmin)*(value-oldmax)+newmax
    return(newvalue)

#### Functions to converty reflectivity to rainfall and vice-versa
def to_dB(array, offset=-1):
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
    
    if type(rainrate) == np.ndarray and not np.isscalar(rainrate):
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
        print('Problem in data_tools_attractor::get_reduced_extent. Non-even border size in X dimension.')
        print('Dims [w,h,dsizex,dsizey]:', width, height, domainSizeX, domainSizeY)
        # sys.exit(1)
    if ((height - domainSizeY) % 2) == 0:
        borderSizeY = (height - domainSizeY)/2
    else:
        print('Problem in data_tools_attractor::get_reduced_extent. Non-even border size in Y dimension.')
        print('Dims [w,h,dsizex,dsizey]:', width, height, domainSizeX, domainSizeY)
        # sys.exit(1)
        
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
    
def shift2(arr,num):
    arr=np.roll(arr,num)
    if num<0:
         np.put(arr,range(len(arr)+num,len(arr)),np.nan)
    elif num > 0:
         np.put(arr,range(num),np.nan)
    return arr
    
def plot_polar_axes_wind(ax2):
    ax2.set_xticklabels([])
    ax = plt.gca()
    gridX,gridY = 10.0,15.0
    ax.text(0.5,1.025,'N',transform=ax.transAxes,horizontalalignment='center',verticalalignment='bottom',size=25)
    
    directionsDegrees = np.array([45, 90, 135, 180, 225, 270, 315])
    directionsText = np.array(['NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    
    for para in np.arange(gridY,360,gridY):
        x_txt = (1.1*0.5*np.sin(np.deg2rad(para)))+0.5
        y_txt = (1.1*0.5*np.cos(np.deg2rad(para)))+0.5
        if para in directionsDegrees:
            idx = np.where(para == directionsDegrees)[0]
            if (directionsText[idx[0]] == 'S') | (directionsText[idx[0]] == 'SE') | (directionsText[idx[0]] == 'SW'):
                y_txt = y_txt - 0.03
            ax.text(x_txt,y_txt,directionsText[idx[0]],transform=ax.transAxes,horizontalalignment='center',verticalalignment='bottom',size=25)
        else:
            ax.text(x_txt,y_txt,u'%i\N{DEGREE SIGN}'%para,transform=ax.transAxes,horizontalalignment='center',verticalalignment='center')

def deg2degN(degrees):
    '''
    Conversion from trigonometrical degrees (0->E) to degrees north (wind direction convention, 0->N)
    [N=90, E=0, S=270, W=180] -> [N=0, E=90, S=180, W=270]
    '''
    degreesN = 90-degrees
    
    # Make them all positive (0-360)
    signIdx = (np.sign(degreesN) == -1)
    if type(degreesN) == list:
        degreesN = np.array(degreesN)
        degreesN[signIdx] = 360 + degreesN[signIdx]
        degreesN = degreesN.tolist()
    elif type(degrees) == np.ndarray: 
        degreesN[signIdx] = 360 + degreesN[signIdx]
    else:
        if signIdx == -1:
            degreesN = 360 + degreesN
        
    return degreesN

def degN2deg(degreesN):
    '''
    Conversion from degrees north (wind direction convention, 0->N) to trigonometrical degrees (0->E)
    [N=0, E=90, S=180, W=270] -> [N=90, E=0, S=270, W=180]
    '''
    degrees = 360 - degreesN + 90
    
    if type(degrees) == list: 
        degrees = np.array(degrees)
        degrees[degrees>=360] = degrees[degrees>=360] - 360
        degrees = degrees.tolist()
    elif type(degrees) == np.ndarray: 
        degrees[degrees>=360] = degrees[degrees>=360] - 360
    else:
        if degrees >= 360:
            degrees - 360
    
    return degrees
    
def deg2compass(arrayDegrees, stringType='short'):
    
    if stringType == 'short':
        arr=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    elif stringType == 'long':
        arr=["North","North-northeast","Northeast","East-northeast","East","East-southeast", "Southeast",\
        "South-southeast","South","South-southwest","Southwest","West-southwest","West","West-northwest","Northwest","North-northwest"]
    
    arrayCompass = []
    for i in range(0,len(arrayDegrees)):
        val=int((arrayDegrees[i]/22.5)+.5)
        arrayCompass.append(arr[(val % 16)])
    
    if type(arrayDegrees) == np.ndarray:
        arrayDegrees = np.array(arrayDegrees)
    return arrayCompass

from matplotlib.colors import from_levels_and_colors 
from pylab import get_cmap
def smart_colormap(clevs, name='jet', extend='both', minval=0.0, maxval=1.0):
    '''
    Automatically grabs the colors to extend the colorbar from the colormap.
    '''
    
    # Define number of colors
    if extend == 'both':
        nrColors = len(clevs)+1
    elif (extend == 'min') | (extend == 'max'):
        nrColors = len(clevs)
    elif (extend == 'neither'):
        nrColors = len(clevs)-1
    else:
        nrColors = len(clevs)-1
        extend = 'neither'
    
    # Get colormap
    cmap = get_cmap(name, nrColors)
    
    # Truncate colormap if asked
    if (minval != 0.0) or (maxval != 1.0):
        cmap = truncate_colormap(cmap, minval=minval, maxval=maxval, n=nrColors/2)
    
    # Get the list of colors
    colors = []
    for i in range(0, nrColors):
        colors.append(cmap(i/(nrColors-1)))
    
    # Use utility function to get cmap and norm at the same time
    cmap, norm = from_levels_and_colors(clevs, colors, extend=extend)

    return(cmap, norm)

def colormap_meteoswiss(clevs):
    colorsMS,_,_ = get_colorlist(type='MeteoSwiss')
    cmap = colors.LinearSegmentedColormap.from_list("cmap", colorsMS, len(clevs)-1)
    cmap.set_over('darkred',1)
    norm = colors.BoundaryNorm(clevs, cmap.N)    
    
    return(cmap, norm)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    
def create_smart_clevels(minCountLevels=0, maxCountLevels=100):
    diffMaxMin = maxCountLevels - minCountLevels
    
    if (diffMaxMin >= 50000) & (diffMaxMin < 500000):
        maxCountLevels = int(maxCountLevels/10000)*10000
        clevs = np.arange(minCountLevels,maxCountLevels,10000)
    if (diffMaxMin >= 10000) & (diffMaxMin < 50000):
        maxCountLevels = int(maxCountLevels/1000)*1000
        clevs = np.arange(minCountLevels,maxCountLevels,1000)
    elif (diffMaxMin >= 5000) & (diffMaxMin < 10000):
        maxCountLevels = int(maxCountLevels/100)*100
        clevs = np.arange(minCountLevels,maxCountLevels,500)
    elif (diffMaxMin >= 2000) & (diffMaxMin < 5000):
        maxCountLevels = int(maxCountLevels/100)*100
        clevs = np.arange(minCountLevels,maxCountLevels,250)
    elif (diffMaxMin >= 1000) & (diffMaxMin < 2000):
        maxCountLevels = int(maxCountLevels/100)*100
        clevs = np.arange(minCountLevels,maxCountLevels,100)
    elif (diffMaxMin >= 500) & (diffMaxMin < 1000):
        maxCountLevels = int(maxCountLevels/10)*10
        clevs = np.arange(minCountLevels,maxCountLevels,50)
    elif (diffMaxMin >= 100) & (diffMaxMin < 500):
        maxCountLevels = int(maxCountLevels/10)*10
        clevs = np.arange(minCountLevels,maxCountLevels,25)                
    else:
        maxCountLevels = int(maxCountLevels)
        clevs = np.arange(minCountLevels,maxCountLevels,10)  
    
    return(clevs)

# def create_smart_clevels_new(minCountLevels=0, maxCountLevels=100):
    # diffMaxMin = maxCountLevels - minCountLevels
    
    # array_classSep = np.array([1, 2, 2.5, 5])
    
    # clevs = np.arange(minCountLevels, maxCountLevels,10)  
    
    # return(clevs)
    
import matplotlib as mpl
def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r

def get_coordinates_swiss_locations(locations='radars', proj4stringCH=None):
    proj4stringWGS84 = "+proj=longlat +ellps=WGS84 +nadgrids=@null +no_defs"
    
    if proj4stringCH== None:
        proj4stringCH = "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 \
        +k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs" 

    s_srs = geo.get_proj4_projection(proj4stringWGS84)
    t_srs = geo.get_proj4_projection(proj4stringCH)
    ct = osr.CoordinateTransformation(s_srs,t_srs)
    #p1 = pyproj.Proj(proj4stringWGS84)
    #p2 = pyproj.Proj(t_srs)
    
    if locations == 'radars':
        coordinates = [
        ('LEM', 46.04, 8.83),
        ('DOL', 46.43, 6.10),
        ('ALB', 47.29, 8.51),
        ('PPM', 46.370833, 7.486944),
        ('WEI', 46.835258, 9.794108)
        ]
    elif locations == 'cities':
        coordinates = [
        # ('Lugano', 46.003678, 8.951052),
        ('Milan', 45.458626,9.181872999999996),
        ('Zurich', 47.3768866,8.541694),
        ('Bern', 46.9479739,7.447446799999966),
        ('Basel', 47.55959860000001,7.588576099999955),
        ('Geneva',46.2043907,6.143157699999961),
        # ('Lucerne', 47.050354, 8.304885),
        # ('Aosta', 45.736374, 7.315577),
        # ('Gotthard', 46.559302, 8.561800),
        # ('Grenoble', 45.189712, 5.723413),
        # ('Chambery', 45.573669, 5.927933),
        # ('Lyon', 45.751703, 4.838691),
        # ('Constanz',47.670897, 9.173335),
        ('Besancon', 47.238605, 6.024600),
        ('Dijon', 47.322436, 5.038421),
        # ('Domodossola', 46.111880, 8.298555),
        ('Locarno', 46.169088, 8.786945),
        # ('Varese', 45.820100, 8.824963),
        ('Bergamo', 45.699911, 9.677642),
        # ('Brescia',45.540414, 10.218764),
        ('Lausanne', 46.519610, 6.628720),
        # ('Ivrea',45.466663, 7.881943),
        ('St Moritz',46.490273, 9.833650),
        # ('Verbania', 45.928402, 8.555404),
        ('Turin', 45.068369, 7.678724),
        ('Genoa',44.406567, 8.947670),
        ('Lyon',45.765961, 4.832295),
        ('Strasbourg', 48.576825, 7.751060),
        ('Stuttgart', 48.778949, 9.182309)
        ]
    elif locations == 'regions':
        # 4th column is rotation
        coordinates = [
        ('Berner Prealps', 46.644444, 7.641923),
        ('Glarner Prealps', 47.027443, 9.066961),
        ('Ticino', 46.330798, 8.783747),
        ('Plateau', 47.150606, 7.985627),
        ('Berner \n Jura', 47.188765, 7.163854),
        ('Jura \n Vaudois', 46.633969, 6.315448),
        ('Central \n Alps', 46.631409, 8.383590),
        ('Vosges', 48.012662, 6.885191),
        ('Black \n Forest', 47.877341, 8.023283),
        ('Savoy', 45.515406, 6.419615),
        ('Upper \n Savoy', 46.058678, 6.386667),
        ('Simplon', 46.251999, 8.030979),
        ('L. Magg.', 45.980404, 8.653914),
        ('L. Constance',47.636015, 9.387289),
        ('L. Geneva', 46.406273, 6.553283),
        ('Orobie Alps', 45.993991, 9.734690),
        ('Aosta \n valley', 45.740531, 7.360523),
        ('Po Plain', 45.470641, 8.593407),
        ('Grisons', 46.697617, 9.633758),
        ('Valais', 46.193337, 7.466644),
        #('Rhine \n valley', 48.309796, 7.598972),
        ('Mont \n Blanc', 45.832736, 6.865442),
        ('Monte \n Rosa', 45.935122, 7.865444),
        ]
    elif locations == 'countries':    
        coordinates = [
        ('FRANCE', 47.001804, 5.424870),
        ('ITALY', 45.639623, 8.504649),
        ('AUSTRIA', 47.162365, 10.014188, 9.923111),
        ('GERMANY',48.008381, 9.363776)
        ]
    else:
        print('Possible values for locations=radars,regions,cities.')
        sys.exit()
    
    # Collect and project locations
    labels = []
    locx = []
    locy = []
    for p in range(0,len(coordinates)):
        lat = coordinates[p][1]
        lon = coordinates[p][2]
        #x,y = pyproj.transform(p1, p2, lon, lat)
        temp = np.zeros((1,3))
        temp[0,0] = lon
        temp[0,1] = lat
        temp2 = ct.TransformPoints(temp)
        locx.append(temp2[0][0])
        locy.append(temp2[0][1])
        labels.append(coordinates[p][0])
	
    return locx, locy, labels 

def draw_radars(ax, which=['LEM','DOL','ALB','PPM','WEI'], fontsize=10, marker='^', markersize=15, markercolor='r', bboxcolor='', only_location=False):
    # Draw location of radars
    loc_x, loc_y, loc_l = get_coordinates_swiss_locations(locations='radars')
    
    for label,x,y in zip(loc_l, loc_x, loc_y):
        if label in which:
            ax.scatter(x, y, c=markercolor, marker=marker, s=markersize, edgecolors='k') # radars
    
    if only_location == False:
        for label, x, y in zip(loc_l, loc_x, loc_y):
            if label in which:
                if len(bboxcolor) > 0:
                    ax.annotate(label, xy=(x,y), xytext=(7,-2), textcoords = 'offset points', fontsize=fontsize, bbox=dict(boxstyle="square", fc=bboxcolor),)
                else:
                    ax.annotate(label, xy=(x,y), xytext=(7,-2), textcoords = 'offset points', fontsize=fontsize)
            
def draw_cities(ax, fontsize=10, marker='o', markersize=5, markercolor='k'):
    # Draw location of major cities
    loc_x, loc_y, loc_l = get_coordinates_swiss_locations(locations='cities')
    
    ax.scatter(loc_x, loc_y, c=markercolor, marker=marker, s=markersize)
    for label, x, y in zip(loc_l, loc_x, loc_y):
        ax.annotate(label, xy = (x, y), xytext = (7, -2), textcoords = 'offset points',fontsize=fontsize)

def draw_regions(ax, fontsize=11, color='k'):        
    # Draw location of regions
    loc_x, loc_y, loc_l = get_coordinates_swiss_locations(locations='regions')
    
    for label, x, y in zip(loc_l, loc_x, loc_y):
        if (label == 'Berner Prealps') or (label == 'Glarner Prealps') or (label == 'Plateau') or (label == 'Berner \n Jura') or (label == 'Jura \n Vaudois'):
            rotation = 30
        elif (label == 'L. Magg.'):
            rotation = 60
        elif (label == 'L. Constance'):
            rotation = -30
        elif (label == 'L. Geneva'):
            rotation = 15
        else:
            rotation = 0
        ax.annotate(label, xy = (x, y), xytext = (0, 0), style='italic', color=color, rotation=rotation, ha='center', va='center', textcoords = 'offset points',fontsize=fontsize)

def draw_countries(ax, fontsize=11, color='k'):        
    # Draw location of regions
    loc_x, loc_y, loc_l = get_coordinates_swiss_locations(locations='countries')
    
    for label, x, y in zip(loc_l, loc_x, loc_y):
        rotation = 0
        ax.annotate(label, xy = (x, y), xytext = (0, 0), style='italic', color=color, rotation=rotation, ha='center', va='center', textcoords = 'offset points',fontsize=fontsize)
        
def extract_field_values_at_coords(field_flat, all_field_coordinates, location_coordinates, verbose=1):      
    '''
    Extract the values of a field (1d flattened array) at given coordinates.
    all_field_coordinates is an Nx2 array with all the coordinate combinations of the field [[x0,y0],[x1,y0],[x2,y0],...,[x0,y1],[x1,y1],...]
    location_coordinates is the same but with the subset of coordinates to extract
    
    
    Returns
    ----------
    location_values_array: list(float)
        Nx3 array with x,y coordinates of the box and the HZT value
       
    '''
    nrCoordinates = len(location_coordinates)
    
    xcoord_all = all_field_coordinates[:,0]
    ycoord_all = all_field_coordinates[:,1]

    location_values_array = []
    for box in range(0, nrCoordinates):
        xcoord = location_coordinates[box,0]
        ycoord = location_coordinates[box,1]
        
        # Find corresponding index on HZT field
        idx = np.where((xcoord_all == xcoord) & (ycoord_all == ycoord))[0]
        
        if len(idx) == 1:
            hzt_value = field_flat[idx][0]
            rowBoxes = [xcoord, ycoord, hzt_value]
            location_values_array.append(rowBoxes)
        elif len(idx) == 0:
            print('Index of field not found', len(idx))
        else:
            print('Too many indices on field found', len(idx))
        
        # Print out advancement
        if verbose == 1:
            update_progress((box+1)/nrCoordinates, processName = "Progress")
        
    return(location_values_array)

def centers(bins):
    return np.vstack([bins[:-1], bins[1:]]).mean(axis=0)
    
def transformSwissToStandard(xc,yc):
    x=xc-255
    y=640-(yc+160)
    return(x,y)

def transformStandardToSwiss(x,y):                                                                                         
    xc=x+255
    yc=640-(y+160)
    return(xc,yc)
    
def add_plus_array(clevs, fmt="%.1f"):
    score = lambda i: ("+" if i > 0 else "") + fmt % i
    clevsTicks = []
    for i in range(0,len(clevs)):
        clevsTicks.append(score(clevs[i]))
    return(clevsTicks)

def elements_in_list(small_list, large_list):
    '''
    Finds whether any of the items in small_list is contained in large_list
    '''
    b = any(item in large_list for item in small_list)        
    return(b)
    
def set_yaxis_color(ax, color):
    '''
    Function to set all ticks, ticklabels and axis labels to a certain color.
    '''
    ax.yaxis.label.set_color(color)
    [t.set_color(color) for t in ax.yaxis.get_ticklines()]
    [t.set_color(color) for t in ax.yaxis.get_ticklabels()]

def interp_FFTfilter_HZT(fieldHZT, missingData_metranet=6362.5, missingData_netcdf=-999, minPercMissing=30, cutoff_scale_km=50, resolution_km=1):
    
    # Interpolate missing HZT values
    maxPercMissing=30
    field_raw_interp, percMissing, boolMissing = interp_missing_HZT(fieldHZT, missingData_metranet, missingData_netcdf, maxPercMissing=maxPercMissing)
    
    # Apply Fourier filter
    if percMissing < maxPercMissing:
        print('Applying Fourier low pass filter...')
        field_fft = st.fourier_low_pass2d(field_raw_interp, cutoff_scale_km, resolution_km=1)
    else:
        print('Not enough data for Fourier filter.')
        field_fft = field_raw_interp
        
    return(field_fft, field_raw_interp, percMissing, maxPercMissing, boolMissing)
    
def interp_missing_HZT(fieldHZT, missingData_metranet=6362.5, missingData_netcdf=-999, maxPercMissing=30):
    '''
    Function to re-interpolate the missing HZT values
    '''
    #### PARAMETERS ####
    maxHZT = 5500
    
    print('++++++++++++')
    ####################
    # Limits of CCS4 domain (from extent)
    Xmin = 255000
    Xmax = 965000
    Ymin = -160000
    Ymax = 480000
    extent_CCS4 = [Xmin, Xmax, Ymin, Ymax]
        
    # Compute coordinates of CCS4 domain
    resKM = 1
    xcoords_ccs4 = np.arange(extent_CCS4[0]/1000, extent_CCS4[1]/1000, resKM)
    ycoords_ccs4 = np.flipud(np.arange(extent_CCS4[2]/1000, extent_CCS4[3]/1000, resKM))

    xcoords_ccs4_grid, ycoords_ccs4_grid = np.meshgrid(xcoords_ccs4, ycoords_ccs4)
    xcoords_ccs4_grid_flat = xcoords_ccs4_grid.flatten()
    ycoords_ccs4_grid_flat = ycoords_ccs4_grid.flatten()

    coordinates_unique_ccs4 = np.column_stack((xcoords_ccs4_grid_flat, ycoords_ccs4_grid_flat))
    
    ####################
    field_raw = np.array(fieldHZT)
    field_raw_flat = field_raw.flatten()
    
    # Check for missing or bad data
    boolMissing = (field_raw == missingData_netcdf) | (field_raw == missingData_metranet) | (field_raw > maxHZT) | (field_raw == 0.0) | (field_raw == np.nan)
    boolMissing_flat = boolMissing.flatten()
    nrMissing = np.sum(boolMissing)
    percMissing = 100*nrMissing/(field_raw.shape[0]*field_raw.shape[1])
    print(nrMissing, 'missing or bad pixels found.', int(percMissing), '%')
    
    # What to do when too many missing data
    valueMissing = 'nan' # 'nan' or 'field_mean'
    if (percMissing > 0) & (percMissing < maxPercMissing):
        
        #### DILATE REGIONS
        # Dilate the region of missing data to account for close aberrant values
        print('Dilating region of missing data...')
        struct = ndimage.generate_binary_structure(2,2)
        boolMissing_dilated = ndimage.binary_dilation(boolMissing, structure=struct)
        boolMissing_dilated_flat = boolMissing_dilated.flatten()
        
        # Dilate more to define buffer of pixels for interpolation (faster for griddata)
        struct = ndimage.generate_binary_structure(2,2)
        boolMissing_dilated_more = ndimage.binary_dilation(boolMissing_dilated, structure=struct)
        boolMissing_dilated_more = ndimage.binary_dilation(boolMissing_dilated_more, structure=struct)
        boolMissing_dilated_more = ndimage.binary_dilation(boolMissing_dilated_more, structure=struct)
        boolMissing_dilated_flat_more = boolMissing_dilated_more.flatten()
        
        boolValid_buffer = (boolMissing_dilated == False) & (boolMissing_dilated_more == True)
        boolValid_buffer_flat = boolValid_buffer.flatten()
        
        ##### INTERPOLATE MISSING DATA
        interp_function = 'griddata' #'regular_grid'
        print('Interpolating missing data...')
        coords_valid = coordinates_unique_ccs4[boolValid_buffer_flat]
        values = field_raw_flat[boolValid_buffer_flat]
        coords_missing = coordinates_unique_ccs4[boolMissing_dilated_flat]
        
        if interp_function == 'griddata':
            # Interpolate missing data
            interpType = 'linear' #'linear' ('cubic' may lead to unrealistically high values)
            interpolated_missing = griddata(coords_valid, values, (coords_missing), method=interpType)
            # Replace missing values with interpolated ones
            field_raw_flat[boolMissing_dilated_flat] = interpolated_missing
        elif interp_function == 'regular_grid':
            # Interpolate missing data
            my_interp_func = LinearNDInterpolator((coords_valid), values, fill_value=np.nan, rescale=False)
            interpolated_missing = my_interp_func((coords_missing[:,0], coords_missing[:,1]))
            # Replace missing values with interpolated ones
            field_raw_flat[boolMissing_dilated_flat] = interpolated_missing
        else:
            print('Invalid interpolation function.')
            sys.exit()
        
        # Interpolate remaining NaNs (scipy bug with linear interpolation)
        isnan_data = np.isnan(field_raw_flat)
        print(np.sum(isnan_data), 'nans left.')
        if np.sum(isnan_data) > 0:
            print('Interpolate remaining nans...')
            coords_valid = coordinates_unique_ccs4[~isnan_data]
            values = field_raw_flat[~isnan_data]
            coords_missing = coordinates_unique_ccs4[isnan_data]
            interpolated_nans = griddata(coords_valid, values, (coords_missing), method='nearest')
            field_raw_flat[isnan_data] = interpolated_nans
        
        field_raw_interp = field_raw_flat.reshape(field_raw.shape)
    elif (percMissing >= maxPercMissing):
        # Not enough HZT data... take mean HZT over the whole field or set NaN everywhere
        print('Not enough data for interpolation.')
        if valueMissing == 'field_mean':
            meanHZT = np.nanmean(field_raw)
        elif valueMissing == 'nan':
            meanHZT = missingData_netcdf # np.nan
        else:
            print('valueMissing should be either nan or field_mean')
            sys.exit(1)
        field_raw_interp = meanHZT*np.ones((field_raw.shape))
    else:
        print('No missing data found.')
        field_raw_interp = field_raw
        
    return(field_raw_interp, percMissing, boolMissing)
    
