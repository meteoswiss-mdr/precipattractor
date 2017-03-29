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

import matplotlib.colors as colors

import datetime
from operator import itemgetter

import time_tools_attractor as ti
import data_tools_attractor as dt
import stat_tools_attractor as st

# Radar structure
class Radar_object(object):
    
    # Radar stats
    war = -1

def read_gif_image(timeStartStr, product='AQC', minR = 0.08, fftDomainSize = 512, resKm = 1, timeAccumMin = 5,\
    inBaseDir = '/scratch/lforesti/data/', noData = -999.0, cmaptype = 'MeteoSwiss', domain = 'CCS4'):
    
    # time parameters
    timeAccumMinStr = '%05i' % timeAccumMin
    timeAccum24hStr = '%05i' % (24*60)
    
    # Limits of spatial domain
    if domain == 'CCS4':
        Xmin = 255000
        Xmax = 965000
        Ymin = -160000
        Ymax = 480000
    else:
        print('Domain not found.')
        sys.exit(1)
    allXcoords = np.arange(Xmin,Xmax+resKm*1000,resKm*1000)
    allYcoords = np.arange(Ymin,Ymax+resKm*1000,resKm*1000)
    
    # colormap
    color_list, clevs, clevsStr = dt.get_colorlist(cmaptype) 

    cmap = colors.ListedColormap(color_list)
    norm = colors.BoundaryNorm(clevs, cmap.N)
    cmap.set_over('black',1)
    cmapMask = colors.ListedColormap(['black'])
    
    # timestamp
    timeStart = ti.timestring2datetime(timeStartStr)
    timeLocal = timeStart
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
    fileName = get_filename_matching_regexpr(fileNameWildCard)
    
    # Get data quality from fileName
    dataQuality = get_quality_fromfilename(fileName)
    
    # Check if file exists
    isFile = os.path.isfile(fileName)
    if (isFile == False):
        print('File: ', fileNameWildCard, ' not found.')
        radar_object = Radar_object()  
    else:
        # Reading GIF file
        print('Reading: ', fileName)
        try:
            # Open GIF image
            rain8bit, nrRows, nrCols = open_gif_image(fileName)
            
            # Get GIF image metadata
            alb, doe, mle, ppm, wei = get_gif_radar_operation(fileName)
            
            # If metadata are not written in gif file derive them from the quality number in the filename
            if (alb == -1) & (doe == -1) & (mle == -1) & (ppm == -1) & (wei == -1):
                alb, doe, mle = get_radaroperation_from_quality(dataQuality)

            # Generate lookup table
            lut = dt.get_rainfall_lookuptable(noData)

            # Replace 8bit values with rain rates 
            rainrate = lut[rain8bit]
            
            if (product == 'AQC'): # AQC is given in millimiters!!!
                rainrate[rainrate != noData] = rainrate[rainrate != noData]*(60/5)

            # Get coordinates of reduced domain
            if fftDomainSize>0:
                extent = dt.get_reduced_extent(rainrate.shape[1], rainrate.shape[0], fftDomainSize, fftDomainSize)
                Xmin = allXcoords[extent[0]]
                Ymin = allYcoords[extent[1]]
                Xmax = allXcoords[extent[2]]
                Ymax = allYcoords[extent[3]]

            subXcoords = np.arange(Xmin,Xmax,resKm*1000)
            subYcoords = np.arange(Ymin,Ymax,resKm*1000)
            
            # Select 512x512 domain in the middle
            if fftDomainSize>0:
                rainrate = dt.extract_middle_domain(rainrate, fftDomainSize, fftDomainSize)
                rain8bit = dt.extract_middle_domain(rain8bit, fftDomainSize, fftDomainSize)
          
            # Create mask radar composite
            mask = np.ones(rainrate.shape)
            mask[rainrate != noData] = np.nan
            mask[rainrate == noData] = 1

            # Set lowest rain thresholds
            if (minR > 0.0) and (minR < 500.0):
                rainThreshold = minR
            else: # default minimum rainfall rate
                rainThreshold = 0.08
                
            # Compute WAR
            war = st.compute_war(rainrate,rainThreshold, noData)
            
            # fills no-rain with nans (for conditional statistics)
            rainrateNans = np.copy(rainrate)
            condition = rainrateNans < rainThreshold
            rainrateNans[condition] = np.nan
            
            # fills no-rain with zeros and missing data with nans (for unconditional statistics)
            condition = rainrate < 0
            rainrate[condition] = np.nan
            condition = (rainrate < rainThreshold) & (rainrate > 0.0)
            rainrate[condition] = 0.0
            
            # Compute corresponding reflectivity
            A = 316.0
            b = 1.5
            
            # Take reflectivity value corresponding to minimum rainfall threshold as zero(0.08 mm/hr)
            dbzThreshold,_,_ = dt.rainrate2reflectivity(rainThreshold, A, b)
            
            # Convert rainrate to reflectivity, no-rain are set to zero (for unconditional statistics)
            dBZ, minDBZ, minRainRate = dt.rainrate2reflectivity(rainrate, A, b, 0.0)
           
            # fills nans with dbzThreshold for Fourier analysis
            condition1 = np.isnan(dBZ) 
            condition2 = dBZ < dbzThreshold
            dBZFourier = np.copy(dBZ)
            dBZFourier[(condition1 == True) | (condition2 == True)] = dbzThreshold
            
            # fills no-rain and missing data with nans (for conditional statistics)
            condition = rainrateNans < rainThreshold
            dBZNans = np.copy(dBZ)
            dBZNans[condition] = np.nan
            
            ## Creates radar object
            radar_object = Radar_object()

            # fields
            radar_object.dBZ = dBZ
            radar_object.dBZFourier = dBZFourier
            radar_object.dBZNans = dBZNans
            radar_object.rain8bit = rain8bit
            radar_object.rainrate = rainrate
            radar_object.rainrateNans = rainrateNans
            radar_object.mask = mask
            
            # statistics
            radar_object.war = war
            
            # time stamps
            radar_object.datetime = timeStart
            radar_object.datetimeStr = timeStartStr
            radar_object.hourminStr = hourminStr
            radar_object.yearStr = yearStr
            radar_object.julianDayStr = julianDayStr
            
            # metadata
            radar_object.fileName = fileName
            radar_object.dbzThreshold = dbzThreshold
            radar_object.rainThreshold = rainThreshold
            radar_object.alb = alb
            radar_object.doe = doe
            radar_object.mle = mle
            radar_object.ppm = ppm
            radar_object.wei = wei
            radar_object.dataQuality = dataQuality
            
            # Location
            radar_object.extent = (Xmin, Xmax, Ymin, Ymax)
            radar_object.subXcoords = subXcoords
            radar_object.subYcoords = subYcoords
            if dBZ.shape[0] == dBZ.shape[1]:
                radar_object.fftDomainSize = dBZ.shape[0]
            else:
                radar_object.fftDomainSize = dBZ.shape
            
            # colormaps
            radar_object.cmap = cmap
            radar_object.norm = norm
            radar_object.clevs = clevs
            radar_object.clevsStr = clevsStr
            radar_object.cmapMask = cmapMask

        except IOError:
            print('File ', fileName, ' not readable')
            radar_object = Radar_object()
            
    return(radar_object)    
    
def get_filename_wavelets(inBaseDir, analysisType, timeDate, product='AQC', timeAccumMin=5, scaleKM=None, minR=0.08, format='netcdf'):
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
    
    if scaleKM is None:
        print('You have to input the spatial scale of the wavelet decomposition in KM')
        sys.exit(1)
        
    # Create time timestamp strings
    timeAccumMinStr = '%05i' % (timeAccumMin)
    year, yearStr, julianDay, julianDayStr = ti.parse_datetime(timeDate)
    hourminStr = ti.get_HHmm_str(timeDate.hour, timeDate.minute)
    subDir = ti.get_subdir(timeDate.year, julianDay)
        
    inDir = inBaseDir + subDir
    
    ### Define filename of statistics
    fullName = inDir + product + '_' + analysisType + '_' + str(scaleKM) + 'km_' + yearStr + julianDayStr + hourminStr + \
        '_Rgt' + str(minR) + '_' + timeAccumMinStr + extension
    
    # Get directory name and base filename
    dirName = inDir
    fileName = os.path.basename(fullName)

    return(fullName, dirName, fileName)

def get_filename_stats(inBaseDir, analysisType, timeDate, product='AQC', timeAccumMin=5, quality=0, minR=0.08,  wols=0, variableBreak = 0, format='netcdf'):
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
        '_Rgt' + str(minR) + '_WOLS' + str(wols) + '_varBreak' + str(variableBreak) + '_' + timeAccumMinStr + extension
    
    # Get directory name and base filename
    dirName = inDir
    fileName = os.path.basename(fullName)

    return(fullName, dirName, fileName)
    
def get_filename_velocity(inBaseDir, analysisType, timeDate, product='AQC', timeAccumMin=5, quality=0, format='netcdf'):
    if format == 'netcdf':
        extension = '.nc'
    elif format == 'csv':
        extension = '.csv'
    elif format == 'png':
        extension = '.png'
    elif format == 'gif':
        extension = '.gif'
    else:
        print('Wrong file format in get_filename_velocity')
        sys.exit(1)

    # Create time timestamp strings
    timeAccumMinStr = '%05i' % (timeAccumMin)
    year, yearStr, julianDay, julianDayStr = ti.parse_datetime(timeDate)
    hourminStr = ti.get_HHmm_str(timeDate.hour, timeDate.minute)
    subDir = ti.get_subdir(timeDate.year, julianDay)
        
    inDir = inBaseDir + subDir
    
    ### Define filename of statistics
    fullName = inDir + product + '_' + analysisType + '_' + yearStr + julianDayStr + hourminStr + str(quality) + \
    '_' + timeAccumMinStr + extension
    
    # Get directory name and base filename
    dirName = inDir
    fileName = os.path.basename(fullName)

    return(fullName, dirName, fileName)
    
def get_filename(inBaseDir, analysisType, timeDate, varNames, varValues, product='AQC', timeAccumMin=5, quality=0, format='netcdf', sep='_'):
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
        
    if len(varNames) != len(varValues):
        print('Number of elements in varNames and varValues must be the same.')
        sys.exit(1)

    # Create time timestamp strings
    timeAccumMinStr = '%05i' % (timeAccumMin)
    year, yearStr, julianDay, julianDayStr = ti.parse_datetime(timeDate)
    hourminStr = ti.get_HHmm_str(timeDate.hour, timeDate.minute)
    subDir = ti.get_subdir(timeDate.year, julianDay)
        
    inDir = inBaseDir + subDir
    
    ### Define filename
    nrVars = len(varNames)
    fullName = inDir + product + sep + analysisType + sep + yearStr + julianDayStr + hourminStr + str(quality)
    for i in range(nrVars):
        fullName = fullName + sep + str(varNames[i]) + str(varValues[i])
    fullName = fullName + sep + timeAccumMinStr + extension
        
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

def write_csv_matrix(fileName, dataArray):
    f = open(fileName, 'w')
    csvOut = csv.writer(f)
    
    csvOut.writerows(dataArray)
    f.close()

# Read-in list of CSV or NETCDF files containing radar rainfall statistics
def csv_list2array(timeStart, timeEnd, inBaseDir, analysisType='STATS', product='AQC', quality=0, timeAccumMin=5, minR=0.08, wols=0, variableBreak=0):
    timeAccumMinStr = '%05i' % (timeAccumMin)
    
    listStats = []
    variableNames = []
    timeLocal = timeStart
    while timeLocal <= timeEnd:
        # Create filename
        fileName,_,_ = get_filename_stats(inBaseDir, analysisType, timeLocal, product, timeAccumMin, quality, minR,  wols, variableBreak, format='csv')
        
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
def netcdf_list2array(timeStart,timeEnd, inBaseDir, variableNames = [], analysisType='STATS', product='AQC', quality=0, timeAccumMin=5, minR=0.08, wols=0, variableBreak=0):
    timeAccumMinStr = '%05i' % (timeAccumMin)
    '''
    
    '''
    listStats = []
    timeLocal = timeStart
    while timeLocal <= timeEnd:
        # Create filename
        fileName,_,_ = get_filename_stats(inBaseDir, analysisType, timeLocal, product, timeAccumMin, quality, minR, wols, variableBreak, format='netcdf')
        
        print('Reading: ', fileName)
        try:
            # Read netcdf
            if len(variableNames) > 0:
                arrayStats,_ = read_netcdf_globalstats(fileName, variableNames)
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
    
def write_netcdf_globalstats(fileName, headers, dataArray, lowRainThreshold, boolWOLS, spectralSlopeLims):
    nrSamples = dataArray.shape[0]
    if boolWOLS == 1:
        strWeightedOLS = "Weighted Ordinary Least Squares"
    else:
        strWeightedOLS = "Ordinary Least Squares"

    # Create netCDF Dataset
    nc_fid = Dataset(fileName, 'w', format='NETCDF4')
    nc_fid.description = "Statistics computed for radar rainfall >= " + str(lowRainThreshold) + " mm/hr"
    nc_fid.comment = "FFT spectrum fitted by " + strWeightedOLS
    nc_fid.comment = "Spectral_slope_lims [km] = " + str(spectralSlopeLims)
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
        if varName == 'scaling_break':
            nc_var.description = "Best scaling break of the 1d power spectrum"
            nc_var.units = "km"
        if varName == 'eccentricity':
            nc_var.description = "Eccentricity of the anisotropy [0-1]"
        if varName == 'orientation':
            nc_var.description = "Orientation of the anisotropy [-90 to 90 degrees]"
            
    nc_fid.close()
    
def read_netcdf_globalstats(fileName, variableNames = None):
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
 
def write_netcdf_flow(fileName, timeStamps, xvec, yvec, Ufields, Vfields, noData=-999.0):
    '''
    Function to write out one or several flow fields to netCDF file
    '''
    if type(timeStamps) is not list:
        if type(timeStamps) is np.ndarray:
            timeStamps = timeStamps.tolist()
        else:
            timeStamps = [timeStamps]
    
    # Create netCDF Dataset
    nc_fid = Dataset(fileName, 'w', format='NETCDF4')
    nc_fid.title = 'Apparent radar velocity field'
    nc_fid.institution = 'MeteoSwiss, Locarno-Monti'
    nc_fid.description ="Motion vectors computed using the Lucas-Kanade tracking algorithm and gridded by Kernel interpolation"
    nc_fid.comment = 'File generated the ' + str(datetime.datetime.now()) + '.'
    nc_fid.noData = noData
    
    # Dimensions
    nrSamples = len(timeStamps)
    nc_fid.createDimension('time', nrSamples) # Much larger file if putting 'None' (unlimited size) 
    nc_time = nc_fid.createVariable('time', 'i8', dimensions=('time'))
    nc_time.description = "Timestamp (UTC)"
    nc_time.units = "%YYYY%MM%DD%HH%mm%SS"
    nc_time[:] = timeStamps
    
    dimNames = ['x','y']
    dimensions = [int(xvec.shape[0]),
                  int(yvec.shape[0])]
    for i in range(len(dimensions)):
        nc_fid.createDimension(dimNames[i],dimensions[i])
    
    # Variables
    w_nc_x = nc_fid.createVariable('x', 'f4', dimensions='x')
    w_nc_x.description = "Swiss easting"
    w_nc_x.units = "km"
    w_nc_x[:] = xvec/1000
    
    w_nc_y = nc_fid.createVariable('y', 'f4', dimensions='y')
    w_nc_y.description = "Swiss northing"
    w_nc_y.units = "km"
    w_nc_y[:] = yvec/1000
    
    w_nc_u = nc_fid.createVariable('U', 'f4', dimensions=('time', 'y', 'x'), zlib=True)
    w_nc_u.description = "Optical flow - zonal component (West -> East)"
    w_nc_u.units = "km/5min"
    w_nc_u[:] = Ufields
    
    w_nc_v = nc_fid.createVariable('V', 'f4', dimensions=('time', 'y', 'x'), zlib=True)
    w_nc_v.description = "Optical flow - meridional component (South -> North))"
    w_nc_v.units = "km/5min"  
    w_nc_v[:] = Vfields
    
    nc_fid.close()                   

def write_netcdf_waveletcoeffs(fileName, timeStamps, \
    xvecs, yvecs, waveletCoeffs, waveletType = 'none', noData=-999.0):
    '''
    Function to write out one field of wavelet coefficients to netCDF file
    '''
    if type(timeStamps) is not list:
        if type(timeStamps) is np.ndarray:
            timeStamps = timeStamps.tolist()
        else:
            timeStamps = [timeStamps]
    
    # Create netCDF Dataset
    nc_fid = Dataset(fileName, 'w', format='NETCDF4')
    nc_fid.title = 'Wavelet coefficients of rainfall field'
    nc_fid.institution = 'MeteoSwiss, Locarno-Monti'
    nc_fid.description = waveletType + " wavelet"
    nc_fid.comment = 'File generated the ' + str(datetime.datetime.now()) + '.'
    nc_fid.noData = noData
    
    # Time dimension
    nrSamples = len(timeStamps)
    nc_fid.createDimension('time', nrSamples) # Much larger file if putting 'None' (unlimited size) 
    nc_time = nc_fid.createVariable('time', 'i8', dimensions=('time'))
    nc_time.description = "Timestamp (UTC)"
    nc_time.units = "%YYYY%MM%DD%HH%mm%SS"
    nc_time[:] = timeStamps
    
    # Generate groups to store the wavelet coefficients at different scales (x,y,wc)
    for scale in range(0,len(waveletCoeffs)):
        # Get data at particular scale
        scalegrp = nc_fid.createGroup("wc_scale" + str(scale))
        xvec = xvecs[scale]
        yvec = yvecs[scale]
        waveletCoeffScale = np.array(waveletCoeffs[scale])
        # Spatial dimension
        dimNames = ['x','y']
        dimensions = [int(xvec.shape[0]),int(yvec.shape[0])]
        for i in range(len(dimensions)):
            scalegrp.createDimension(dimNames[i],dimensions[i])
        
        # Write out coordinates
        w_nc_x = scalegrp.createVariable('x', 'f4', dimensions='x')
        w_nc_x.description = "Swiss easting"
        w_nc_x.units = "km"
        w_nc_x[:] = xvec/1000
        
        w_nc_y = scalegrp.createVariable('y', 'f4', dimensions='y')
        w_nc_y.description = "Swiss northing"
        w_nc_y.units = "km"
        w_nc_y[:] = yvec/1000
        
        # Write out wavelet coefficients
        varName = 'wc' + str(scale)
        if scale != len(waveletCoeffs)-1:
            scaleKm = int((xvec[1] - xvec[0])/1000)
        else:
            previousScaleKm = xvecs[scale-1][1] - xvecs[scale-1][0]
            scaleKm = int(previousScaleKm*2/1000)
        
        w_nc_u = scalegrp.createVariable(varName, 'f4', dimensions=('time', 'y', 'x'), zlib=True)
        w_nc_u.description = "Wavelet coefficients at scale " + str(scaleKm) + ' km'
        w_nc_u.units = "amplitude"
        w_nc_u[:] = waveletCoeffScale
    
    nc_fid.close() 
    
#@profile
def netcdf_list2wavelet_array(timeStart, timeEnd, inBaseDir, analysisType='WAVELET', \
    product='AQC', timeAccumMin=5, scaleKM=None):
    '''
    Fucntion to open a series of netCDF files containing "upscaled" rainfall fields with wavelets
    '''  
    
    timeAccumMinStr = '%05i' % (timeAccumMin)
    
    listWaveletScale = []
    listTimeStamps = []
    fieldSizeDone = False
    
    timeLocal = timeStart
    while timeLocal <= timeEnd:
        # Create filename
        fileName,_,_ = get_filename_wavelets(inBaseDir, analysisType, timeLocal, product, \
        timeAccumMin=timeAccumMin, scaleKM=scaleKM, format='netcdf')

        try:
            # Read netcdf
            arrayWaveletScale, arrayTimes, extent = read_netcdf_waveletscale(fileName)
            
            if (arrayWaveletScale[0].shape > 0) & (fieldSizeDone == False):
                fieldSize = arrayWaveletScale[0].shape
                fieldSizeDone = True
            
            # Flatten 2D arrays of wavelet coeffs
            arrayWaveletScaleFlat = []
            for t in range(0,len(arrayWaveletScale)):
                arrayWaveletScaleFlat.append(arrayWaveletScale[t].ravel())
                
            # Concatenate lists
            listWaveletScale = listWaveletScale + arrayWaveletScaleFlat
            listTimeStamps = listTimeStamps + arrayTimes.tolist()
            print(fileName, 'read successfully.')
        except:
            print(fileName, 'empty.')
            
        # Update time (one file per day)
        timeLocal = timeLocal + datetime.timedelta(hours = 24)
    
    # Check if found data
    if (len(listWaveletScale) == 0):
        listWaveletScale = []
        print('No data stored in array.')
        return(listWaveletScale)
    
    # Convert to numpy arrays
    arrayTimeStamps = np.asarray(listTimeStamps)
    arrayWaveletScale = np.asarray(listWaveletScale)
    
    # Remove duplicates
    uniqueTimeStamps, idxUnique = np.unique(arrayTimeStamps, return_index=True)
    arrayTimeStamps = arrayTimeStamps[idxUnique]
    arrayWaveletScale = arrayWaveletScale[idxUnique,:]

    # Sort list of lists by first variable (time) 
    # dataArray = np.column_stack((arrayTimeStamps, arrayWaveletScale))
    # dataArray[dataArray[:,0].argsort()]
    # arrayTimeStamps = dataArray[:,0]
    # arrayWaveletScale = dataArray[:,1:]

    return(arrayWaveletScale, arrayTimeStamps, fieldSize, extent)
    
def read_netcdf_waveletscale(fileName):
    # Open data set
    nc_fid = Dataset(fileName, 'r', format='NETCDF4')
    
    # Read-in the array of timestamps
    timeArray = nc_fid.variables['time'][:]
    
    # Read-in the array of wavelet coefficients
    waveletArray = nc_fid.variables['wc'][:]
    
    # Red-in the extent of the domain
    xcoords = nc_fid.variables['x'][:]
    ycoords = nc_fid.variables['y'][:]
    
    resX = np.abs(xcoords[1] - xcoords[0])
    resY = np.abs(ycoords[1] - ycoords[0])
    
    extent = [np.min(xcoords), np.max(xcoords), np.min(ycoords), np.max(ycoords)]
    extent = [np.min(xcoords)-resX/2, np.max(xcoords)+resX/2, np.min(ycoords)-resY/2, np.max(ycoords)+resY/2]
    
    nc_fid.close()

    # Transpose the list of lists
    # dataArray = zip(*dataArray)
    return(waveletArray, timeArray, extent)
    
def write_netcdf_waveletscale(fileName, timeStampsArray, \
    xvec, yvec, waveletCoeffsArray, scaleKM, waveletType = 'none', noData=-999.0):
    '''
    Function to write out multiple fields of wavelet coefficients at ONE SELECTED SCALE to netCDF file
    '''
    
    if len(timeStampsArray) != len(waveletCoeffsArray):
        print('timeStampsArray and waveletCoeffsArray should have the same number of elements in write_netcdf_waveletscale')
        print(len(timeStampsArray), 'vs', len(waveletCoeffsArray))
        sys.exit(1)
    
    if type(timeStampsArray) is not list:
        if type(timeStampsArray) is np.ndarray:
            timeStampsArray = timeStampsArray.tolist()
        else:
            timeStampsArray = [timeStampsArray]
    
    try:
        # Create netCDF Dataset
        nc_fid = Dataset(fileName, 'w', format='NETCDF4')
        nc_fid.title = 'Wavelet coefficients of rainfall field'
        nc_fid.institution = 'MeteoSwiss, Locarno-Monti'
        nc_fid.description = waveletType + " wavelet"
        nc_fid.comment = 'File generated the ' + str(datetime.datetime.now()) + '.'
        nc_fid.noData = noData
        
        # Time dimension
        nrSamples = len(timeStampsArray)
        nc_fid.createDimension('time', nrSamples) # Much larger file if putting 'None' (unlimited size) 
        nc_time = nc_fid.createVariable('time', 'i8', dimensions=('time'))
        nc_time.description = "Timestamp (UTC)"
        nc_time.units = "%YYYY%MM%DD%HH%mm%SS"
        nc_time[:] = timeStampsArray

        # Spatial dimension
        dimNames = ['x','y']
        dimensions = [int(len(xvec)),int(len(yvec))]
        for i in range(len(dimensions)):
            nc_fid.createDimension(dimNames[i],dimensions[i])
        
        # Write out coordinates
        w_nc_x = nc_fid.createVariable('x', 'f4', dimensions='x')
        w_nc_x.description = "Swiss easting"
        w_nc_x.units = "km"
        w_nc_x[:] = xvec/1000
        
        w_nc_y = nc_fid.createVariable('y', 'f4', dimensions='y')
        w_nc_y.description = "Swiss northing"
        w_nc_y.units = "km"
        w_nc_y[:] = yvec/1000
        
        # Write out wavelet coefficients
        varName = 'wc'
        
        w_nc_u = nc_fid.createVariable(varName, 'f4', dimensions=('time', 'y', 'x'), zlib=True)
        w_nc_u.description = "Wavelet coefficients at scale " + str(scaleKM) + ' km'
        w_nc_u.units = "amplitude"
        waveletCoeffsArray = np.array(waveletCoeffsArray)
        w_nc_u[:] = waveletCoeffsArray
        
        nc_fid.close()
    except:
        print('NetCDF writing error in write_netcdf_waveletscale')
        print('xvec:', xvec)
        print('yvec:', yvec)
        print('waveletCoeffsArray:', waveletCoeffsArray)
        print('waveletCoeffsArray.shape:', waveletCoeffsArray.shape)
        sys.exit(1)
    
    
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