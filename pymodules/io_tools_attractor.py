#!/usr/bin/env python
'''
Module to perform various input/output operations on gif radar files and netCDF/CSV files containing the statistics of the attractor.
The module also provide functionality to generate filenames, etc
Documentation convention from https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

07.07.2016
Loris Foresti
'''

from __future__ import division
from __future__ import print_function

import numpy as np
import os
import subprocess
import sys

import fnmatch
import pandas as pd

import csv
from PIL import Image
from netCDF4 import Dataset

import datetime
from operator import itemgetter

import time_tools_attractor as ti

def get_filename_stats(inBaseDir, analysisType, timeDate, product='AQC', timeAccumMin=5, quality=0, minR=0.08,  wols=0, format='netcdf'):
    if format == 'netcdf':
        extension = '.nc'
    elif format == 'csv':
        extension = '.csv'
    elif format == 'png':
        extension = '.png'
    elif format == 'gif':
        extension = '.gif'
    else:
        print('Wrong file format in get_filename_stats')
        sys.exit(1)

    # Create time timestamp strings
    timeAccumMinStr = '%05i' % (timeAccumMin)
    year, yearStr, julianDay, julianDayStr = ti.parse_datetime(timeDate)
    hourminStr = ti.get_HHmm_str(timeDate.hour, timeDate.minute)
    subDir = ti.get_subdir(timeDate.year, julianDay)
        
    inDir = inBaseDir + subDir
    
    ### Define filename of statistics
    fullName = inDir + product + '_' + analysisType + '_' + yearStr + julianDayStr + hourminStr + str(quality) + \
        '_Rgt' + str(minR) + '_WOLS' + str(wols) + '_' + timeAccumMinStr + extension
    
    # Get directory name and base filename
    dirName = inDir
    fileName = os.path.basename(fullName)

    return(fullName, dirName, fileName)
    
def write_csv(fileName, headers, dataArray):
    f = open(fileName, 'w')
    csvOut = csv.writer(f)

    csvOut.writerow(headers)
    csvOut.writerows(dataArray)
    f.close()

# Read-in list of CSV or NETCDF files containing radar rainfall statistics
def csv_list2array(timeStart, timeEnd, inBaseDir, analysisType='STATS', product='AQC', quality=0, timeAccumMin=5, minR=0.08, wols=0):
    timeAccumMinStr = '%05i' % (timeAccumMin)
    
    listStats = []
    variableNames = []
    timeLocal = timeStart
    while timeLocal <= timeEnd:
        # Create filename
        fileName,_,_ = get_filename_stats(inBaseDir, analysisType, timeLocal, product, timeAccumMin, quality, minR,  wols, format='csv')
        
        print('Reading: ', fileName)
        try:
            if len(variableNames) > 0:
                df = pd.read_csv(fileName, sep=',')
            else:
                df = pd.read_csv(fileName, sep=',')
                variableNames = list(df.columns.values)
            
            # to np array
            arrayStats = df.as_matrix() 
            # to list
            listStatsLocal = arrayStats.tolist()
            # Concatenate lists
            listStats = listStats + listStatsLocal 
        except:
            print(fileName, ' empty.')
            
        # Update time (one file per day)
        timeLocal = timeLocal + datetime.timedelta(hours = 24)
    
    # Check if found data
    if (len(listStats) == 0):
        listStats = []
        print('No data stored in array.')
        return(listStats, variableNames)
    
    # Sort list of lists by first variable (time) 
    listStats.sort(key=itemgetter(0))
    
    # Remove duplicates
    df = pd.DataFrame(listStats)
    df = df.drop_duplicates(0)
    listStats = df.values.tolist()
    
    return(listStats, variableNames)

    # Read-in list of CSV or NETCDF files containing radar rainfall statistics
def netcdf_list2array(timeStart,timeEnd, inBaseDir, variableNames = [], analysisType='STATS', product='AQC', quality=0, timeAccumMin=5, minR=0.08, wols=0):
    timeAccumMinStr = '%05i' % (timeAccumMin)
    '''
    
    '''
    listStats = []
    timeLocal = timeStart
    while timeLocal <= timeEnd:
        # Create filename
        fileName,_,_ = get_filename_stats(inBaseDir, analysisType, timeLocal, product, timeAccumMin, quality, minR,  wols, format='netcdf')
        
        print('Reading: ', fileName)
        try:
            # Read netcdf
            if len(variableNames) > 0:
                arrayStats,_ = read_netcdf(fileName, variableNames)
            else:
                arrayStats, variableNames = read_netcdf(fileName)

            # Concatenate lists
            listStats = listStats + arrayStats 
        except:
            print(fileName, ' empty.')
            
        # Update time (one file per day)
        timeLocal = timeLocal + datetime.timedelta(hours = 24)
    
    # Check if found data
    if (len(listStats) == 0):
        listStats = []
        print('No data stored in array.')
        return(listStats, variableNames)
    
    # Sort list of lists by first variable (time) 
    listStats.sort(key=itemgetter(0))
    
    # Remove duplicates
    df = pd.DataFrame(listStats)
    df = df.drop_duplicates(0)
    listStats = df.values.tolist()

    return(listStats, variableNames)
    
def write_netcdf(fileName, headers, dataArray, lowRainThreshold, boolWOLS):
    nrSamples = dataArray.shape[0]
    if boolWOLS == 1:
        strWeightedOLS = "Weighted Ordinary Least Squares"
    else:
        strWeightedOLS = "Ordinary Least Squares"

    # Create netCDF Dataset
    nc_fid = Dataset(fileName, 'w', format='NETCDF4')
    nc_fid.description = "Statistics computed for radar rainfall >= " + str(lowRainThreshold) + " mm/hr"
    nc_fid.comment = "FFT spectrum fitted by " + strWeightedOLS
    nc_fid.source = "MeteoSwiss"
    
    # Create and fill data into variables
    nc_fid.createDimension('time', nrSamples) # Much larger file if putting 'None' (unlimited size) 
    nc_time = nc_fid.createVariable('time', 'i8', dimensions=('time'))
    nc_time.description = "Timestamp (UTC)"
    nc_time.units = "%YYYY%MM%DD%HH%mm%SS"
    nc_time[:] = dataArray[:,0]
    
    nrVariables = dataArray.shape[1]-1
    for var in range(0,nrVariables):
        varName = headers[var+1]
        
        # Create variable
        if (varName == 'alb') | (varName == 'doe') | (varName == 'mle') |(varName == 'ppm') | (varName == 'wei'):
            nc_var = nc_fid.createVariable(varName, 'i1', dimensions=('time',))
        else:
            nc_var = nc_fid.createVariable(varName, 'f4', dimensions=('time',))
        
        # Put data into variable
        nc_var[:] = dataArray[:,var+1]
        
        # Write radar attributes
        if varName == 'alb':
            nc_var.description = "Number of valid fields from Albis radar (-1: not active, 0: not in operation, 1: ok, 12: correct hourly accumulation)."
        if varName == 'doe':
            nc_var.description = "Number of valid fields from Dole radar"
        if varName == 'mle':
            nc_var.description = "Number of valid fields from Lema radar"
        if varName == 'ppm':
            nc_var.description = "Number of valid fields from Plaine Morte radar"
        if varName == 'wei':
            nc_var.description = "Number of valid fields from Weissfluhjoch radar"
        # Rainfall stats
        if varName == 'war':
            nc_var.description = "Wet area ratio (WAR). Fraction of rainy pixels."
            nc_var.units = "Percentage [%]"
        if varName == 'r_mean':
            nc_var.description = "Unconditional mean precipitation (including zeros). A.k.a. image mean flux (IMF)"
            nc_var.units = "mm/hr"
        if varName == 'r_std':
            nc_var.description = "Unconditional st. dev. of precipitation (including zeros)."
            nc_var.units = "mm/hr"
        if varName == 'r_cmean':
            nc_var.description = "Conditional mean precipitation >= " + str(lowRainThreshold) + " mm/hr"
            nc_var.units = "mm/hr"
        if varName == 'r_cstd':
            nc_var.description = "Conditional st. dev. of precipitation >= " + str(lowRainThreshold) + " mm/hr"
            nc_var.units = "mm/hr"
        # dBZ stats
        if varName == 'dBZ_mean':
            nc_var.description = "Unconditional mean precipitation in dBZ units (including zeros)"
            nc_var.units = "dB"
        if varName == 'dBZ_std':
            nc_var.description = "Unconditional st. dev. of precipitation in dBZ units (including zeros)."
            nc_var.units = "dB"
        if varName == 'dBZ_cmean':
            nc_var.description = "Conditional mean precipitation in dBZ units  >= " + str(lowRainThreshold) + " mm/hr"
            nc_var.units = "dB"
        if varName == 'dBZ_cstd':
            nc_var.description = "Conditional st. dev. of precipitation in dBZ units >= " + str(lowRainThreshold) + " mm/hr"
            nc_var.units = "dB"
        # Fourier stats
        if varName == 'beta1':
            nc_var.description = "Slope of the Fourier power spectrum for large spatial wavelengths [20-512 km]"
        if varName == 'corr_beta1':
            nc_var.description = "Correlation coeff. of the linear fit for beta1"    
        if varName == 'beta2':
            nc_var.description = "Slope of the Fourier power spectrum for small spatial wavelengths [3-20 km]"
        if varName == 'corr_beta2':
            nc_var.description = "Correlation coeff. of the linear fit for beta2"
            
    nc_fid.close()

def read_netcdf(fileName, variableNames = None):
    # Open data set
    nc_fid = Dataset(fileName, 'r', format='NETCDF4')
    
    # Get and read whole list of variables if None is passed
    if variableNames == None:
        variableNames = [str(var) for var in nc_fid.variables]
    
    # Read-in variables one by one
    nrVariables = len(variableNames)
    dataArray = []
    for var in range(0,nrVariables):
        varName = variableNames[var]
        varData = nc_fid.variables[varName][:]
        dataArray.append(varData)
    
    nc_fid.close()

    # Transpose the list of lists
    dataArray = zip(*dataArray)
    
    # How many ways to transpose a list of lists...
    #dataArray = map(list,map(None,*dataArray))
    #dataArray = np.asarray(dataArray).T.tolist()
    # Return floating numpy array
    # dataArray = np.array(dataArray).T 
    return(dataArray, variableNames)

def get_file_matching_expr(inDir, fileNameWildCard):
    listFilesDir = os.listdir(inDir)
    boolFound = False
    for file in listFilesDir:
            if fnmatch.fnmatch(file, fileNameWildCard):
                fileName = file
                boolFound = True
    if boolFound == False:
        fileName = None
    return(fileName)

def get_files_period(timeStart, timeEnd, inBaseDir, fileNameExpr, tempResMin = 5):
    '''
    Function to generalize further
    '''
    timeLocal = timeStart
    fileList = []
    while timeLocal <= timeEnd:
        year, yearStr, julianDay, julianDayStr = ti.parse_datetime(timeLocal)
        timeLocalStr = ti.datetime2juliantimestring(timeLocal)
        # Get directory
        subDir = str(year) + '/' + yearStr + julianDayStr + '/'
        imgDir = inBaseDir + subDir
        
        # Get list of filenames in a given directory matching expression
        fileNameExprTime = fileNameExpr + '*' + timeLocalStr + '*.png'
        fileName = get_file_matching_expr(imgDir, fileNameExprTime)
        # Append filelist
        if fileName != None:
            fileList.append(imgDir + fileName)
        
        # Update time
        timeLocal = timeLocal + datetime.timedelta(minutes = tempResMin)
    return(fileList)
    
def read_csv_array(fileName):
    listStats = []
    with open(fileName, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            rowFmt = np.array(row)
            rowFmt = rowFmt.reshape(1,rowFmt.shape[0])
            #print(rowFmt,rowFmt.shape)
            listStats.append(rowFmt)
    arrayStats = np.array(listStats)
    print(arrayStats[0])
    
    return(arrayStats)

def get_filename_matching_regexpr(fileNameWildCard):
    inDir = os.path.dirname(fileNameWildCard)
    baseNameWildCard = os.path.basename(fileNameWildCard)

    # Check if directory exists
    if os.path.isdir(inDir) == False:
        print('Directory: ' + inDir + ' does not exists.')
        fileName = 'none'
        return(fileName)
    
    # If it does, check for files matching regular expression
    listFiles = os.listdir(inDir)
    if len(listFiles) > 0:
        for file in listFiles:
            if fnmatch.fnmatch(file, baseNameWildCard) == True:
                fileNameRel = file
                # Get absolute filename
                if inDir[len(inDir)-1] == '/':
                    fileName = inDir + fileNameRel
                else:
                    fileName = inDir + '/' + fileNameRel
                break
            else:
                fileName = 'none'
    else:
        fileName = 'none'
        
    return(fileName)
        
def open_gif_image(fileName):
    '''
    Function to read the radar rainfall field from a gif file.
    
    Parameters
    ----------
    fileName : str
    
    Returns
    -------
    rain8bit: int
        2d numpy array containing the radar rainfall field values using 8-bit coding
    nrRows: int
        Number of rows of the radar field
    nrCols: int
        Number of cols of the radar field
    '''
    
    rainImg = Image.open(fileName)
    nrCols = rainImg.size[0]
    nrRows = rainImg.size[1]
    
    rain8bit = np.array(rainImg,dtype=int)
    
    del rainImg
    
    return(rain8bit, nrRows, nrCols)
   
def get_gif_radar_operation(fileName):
    '''
    Function to read the metadata of a radar composite gif file and get the quality of the different radars.
    
    Parameters
    ----------
    fileName : str
    
    Returns
    -------
    alb: int
        Quality of the Albis radar data (number of valid fields in the accumulation)
    doe: int
        Quality of the Dole radar data
    mle: int
        Quality of the Lema radar data
    ppm: int
        Quality of the Plaine Morte radar data
    wei: int
        Quality of the Weissfluhgipfel radar data
    
    '''

    # Default values for period before radar installation
    alb = -1
    doe = -1
    mle = -1
    ppm = -1
    wei = -1
    
    try:
        # Use ImageMagick identify command to grep the line with the radar operations
        cmd = 'identify -format "%c" ' + fileName + ' | grep ALB'
        outString = subprocess.check_output(cmd, shell=True)
        
        # Parse output string
        outStringArray = outString.split(' ')
        
        # Get data quality integer for each radar
        for radar in range(0,len(outStringArray)):
            radarString = outStringArray[radar].split('=')
            if radarString[0] == 'ALB':
                alb = int(radarString[1])
            if (radarString[0] == 'DOE') | (radarString[0] == 'DOL'):
                doe = int(radarString[1])
            if (radarString[0] == 'MLE') | (radarString[0] == 'LEM'):
                mle = int(radarString[1])
            if radarString[0] == 'PPM':
                ppm = int(radarString[1])
            if radarString[0] == 'WEI':
                wei = int(radarString[1])
    except:
        print('ALB activity not readable from ', fileName)
        print('Use the data quality from file name instead')
        
        # Default values if nothing is found in gif file
        alb = -1
        doe = -1
        mle = -1
        ppm = -1
        wei = -1

    return(alb, doe, mle, ppm, wei)

def get_quality_fromfilename(fileName):
    '''
    Function to parse a filname to get the one digit number or letter describing the quality of the radar composite.
    
    Parameters
    ----------
    fileName : str
    
    Returns
    -------
    dataQuality: int, str
        Quality of the radar composite (1 or 0)
    '''
    baseName = os.path.basename(fileName)
    dataQualityIdx = baseName.find('_')-1
    dataQuality = baseName[dataQualityIdx]
    return(dataQuality)
    
def get_radaroperation_from_quality(quality):
    '''
    Function to convert a one digit quality number into the activity of the three 3rd gen. radars.
    
    Parameters
    ----------
    quality : str, int
        One digit quality number (extracted from the filename, e.g. 7)
    
    Returns
    -------
    alb: int
        Quality of the Albis radar data (1 or 0)
    doe: int
        Quality of the Dole radar data (1 or 0)
    mle: int
        Quality of the Lema radar data (1 or 0)
    '''
    quality = int(quality)
    
    # binary codes for radars
    alb_bin = '1'
    doe_bin = '10'
    mle_bin = '100'
    
    # decimal codes for radars
    alb_dec = int(alb_bin,2)
    doe_dec = int(doe_bin,2)
    mle_dec = int(mle_bin,2)
    
    # quality of each individual radar
    alb = -1
    doe = -1
    mle = -1

    if alb_dec == quality:
        alb = 1
    elif doe_dec == quality:
        doe = 1
    elif mle_dec == quality:
        mle = 1
    elif (alb_dec + doe_dec) == quality:
        alb = 1
        doe = 1
    elif (alb_dec + mle_dec) == quality:
        alb = 1
        mle = 1
    elif (doe_dec + mle_dec) == quality:
        doe = 1
        mle = 1
    elif (alb_dec + doe_dec + mle_dec) == quality:
        alb = 1
        doe = 1
        mle = 1
    
    print(quality, '->', 'ALB=', alb, 'DOE=', doe, 'MLE=', mle)
    return(alb,doe,mle)