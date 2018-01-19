#!/usr/bin/env python
"""
DESCRIPTION

    Utils for extracting COSMO data using fieldexta. 
    Currently supported:
    - analysis for COSMO-2 and COSMO-7 (HZEROCL, TOT_PREC).
    - forecasts for COSMO-1 and COSMO-E (TOT_PREC)

AUTHOR

    Daniele Nerini <daniele.nerini@meteoswiss.ch>

VERSION

    1.1
"""

import datetime
import os
import subprocess
import sys
import glob 
import re
import fnmatch
import numpy as np
import netCDF4

def run_fieldextra_analysis(analysisTimeStr, 
                    gribDir,
                    outBaseDir='',
                    fieldName='HZEROCL',
                    modelName='cosmo-7',
                    overwrite=False):
    '''
    Wrap to the fieldextra program to read the COSMO-2/7 analyses and store them as netcdf.
    
    Parameters
    ----------
    analysisTimeStr : str
        Date of the analysis day (YYYYMMDD). Will extract all hourly analyses of that day.
        Can use <datetimeobject>.strftime(%Y%m%d) to print it correctly.
    gribDir : str
        Full path to the folder containing the untared grib cosmo data.
    fieldName : str ['HZEROCL', 'TOT_PREC']
        Variable to extract. Currently only 'HZEROCL' and 'TOT_PREC' are supported.
    modelName : str ['cosmo-7', 'cosmo-2']
        COSMO model name.
    overwrite : bool 
        Check if data were already extracted.
    '''
                    
    # store current working directory
    cwd = os.getcwd()
    # move to script directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    # get datetime format
    analysisTime = datetime.datetime(int(analysisTimeStr[0:4]), int(analysisTimeStr[4:6]), int(analysisTimeStr[6:8]),0,0)
    
    # build output folder according to analyis time
    year = analysisTime.year
    yearStr =  str(year)[2:4]
    julianDay = analysisTime.timetuple().tm_yday
    julianDayStr = '%03i' % julianDay
    yearJulianStr = yearStr + julianDayStr
    outDir = outBaseDir + analysisTime.strftime('%Y') + '/' + analysisTime.strftime('%y') + julianDayStr + '/'
    cmd = 'mkdir -p ' + outDir
    os.system(cmd)
    
    # test if analysis exists
    referenceFileName = gribDir + 'laf' + analysisTime.strftime('%Y%m%d') + '00'
    if os.path.isfile(referenceFileName) == False:
        print('Error: %s does not exists.' % referenceFileName)
        outFile = referenceFileName
        return(outFile)

    # build full path to output forecast file
    fcstName = modelName + '_' + fieldName + '_' + analysisTime.strftime('%Y%m%d')
    outFile = r'%s' % (outDir + fcstName + '.nc')
    
    # check if file was already extracted
    if os.path.isfile(outFile) and not overwrite:
        print('File already exists: ' + outFile)
        return outFile

    # prepare the filename of the temporary config file for fieldextra
    configFile = 'nl_cosmo_laf'
    fileIn = configFile
    fileOut = configFile + '_' + fcstName
    
    # find the <referenceRun> variable and replace it 
    textToSearch = r'<referenceRun>'
    textToReplace = 'laf' + analysisTime.strftime('%Y%m%d') + '00'
    insert_variable(fileIn,fileOut,textToSearch,textToReplace)
    
    # find the <pathToFcst> variable and replace it with the input grib filename
    textToSearch = r'<pathToFcst>'
    textToReplace = r'%s' % gribDir  
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)

    # find the <pathToOutput> variable and replace it with the output filename
    textToSearch = r'<pathToOutput>'
    textToReplace = outFile
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <loopFile> variable and replace it with the given analysis timestamp
    textToSearch = r'<loopFile>'
    textToReplace = 'laf' + analysisTime.strftime('%Y%m%d') + '<HH>'
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <modelName> variable and replace it with the given cosmo version
    textToSearch = r'<modelName>'
    textToReplace = modelName
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <resKm> variable and replace it with the given cosmo resolution
    if modelName=='cosmo-7':
        resKm = 6.6
    elif (modelName=='cosmo-2') or (modelName=='cosmo-e'):
        resKm = 2.2
    elif modelName=='cosmo-1':
        resKm = 1.1
    else:
        print('Error: unknown model name.')
        sys.exit()
    textToSearch = r'<resKm>'
    textToReplace = resKm
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <fieldName> variable and replace it with the given field name to be retrieved
    textToSearch = r'<fieldName>'
    textToReplace = fieldName
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <tstart> variable and replace it 
    textToSearch = r'<tstart>'
    textToReplace = '0'
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <tend> variable and replace it 
    textToSearch = r'<tstop>'
    textToReplace = '23'
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <tincr> variable and replace it
    textToSearch = r'<tincr>'
    textToReplace = '1'
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # add temporal operator for precip
    textToSearch = r'<toper>'
    if fieldName=='TOT_PREC':
        textToReplace = ', toper="delta,1,hour"'
    else:
        textToReplace = ''
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)

    # call the fieldextra routine
    cmd = '/oprusers/owm/bin/fieldextra ' + str(fileOut)
    print('Run: ' + cmd)
    process = subprocess.Popen('ulimit -s unlimited; export OMP_STACKSIZE=500M; ' + cmd, shell=True)
    process.wait()
    
    # remove the temporary files
    cmd = 'rm ' + str(fileOut)
    os.system(cmd)

    print('saved: ' + outFile)
        
    # move back to previous wd
    os.chdir(cwd)
    
    return outFile

def run_fieldextra_forecast(analysisTimeStr,
                    leadtimeHrs = 12,
                    deltaMin = 60,
                    gribDir='',
                    outBaseDir='',
                    fieldName='TOT_PREC',
                    modelName='cosmo-1',
                    overwrite = False):
    '''
    Wrap to the fieldextra program to read the COSMO-1 forecasts and store them as netcdf.
    
    Parameters
    ----------
    analysisTimeStr : str
        Date of the forecast analysis (YYYYMMDDHH). 
        Can use <datetimeobject>.strftime(%Y%m%d%H%M) to print it correctly.
        Cas use the function find_nearest_forecast() to find the nearest available run given any time.
    leadtimeHrs : int
        Hours of forecast since analysis time to be retrieved.
    deltaMin : int
        Time steps in minutes.
    gribDir : str
        Full path to the folder containing the untared grib cosmo data. If an empty '' string is provided, the default gribfolder for the given model is searched.
    fieldName : str ['TOT_PREC']
        Variable to extract. 
    modelName : str ['cosmo-1','cosmo-e']
        COSMO model name.
    overwrite : bool 
        Check if data were already extracted.
    '''
                    
    # store current working directory
    cwd = os.getcwd()
    # move to script directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    # get datetime format
    analysisTime = datetime.datetime(int(analysisTimeStr[0:4]), int(analysisTimeStr[4:6]), int(analysisTimeStr[6:8]), int(analysisTimeStr[8:10]), int(analysisTimeStr[10:12]))
    
    # build output folder according to forecast analysis time
    year = analysisTime.year
    yearStr =  str(year)[2:4]
    julianDay = analysisTime.timetuple().tm_yday
    julianDayStr = '%03i' % julianDay
    yearJulianStr = yearStr + julianDayStr
    outDir = outBaseDir + analysisTime.strftime('%Y') + '/' + analysisTime.strftime('%y') + julianDayStr + '/'
    cmd = 'mkdir -p ' + outDir
    os.system(cmd)
    
    # if not provided, try to look for the grib folder
    if gribDir=='':
        gribDir = get_forecast_folder(modelName, analysisTime)
    
    # test if analysis exists
    if modelName=='cosmo-1':
        referenceFileName = gribDir + 'c1ffsurf00000000'
    elif modelName=='cosmo-e':
        referenceFileName = gribDir + 'ceffsurf000_000'
    else:
        print('Model name %s is not supported.' % modelName)
        sys.exit()
    if os.path.isfile(referenceFileName) == False:
        print('Error: %s does not exists.' % referenceFileName)
        sys.exit()

    # build full path to output forecast file
    fcstName = modelName + '_' + fieldName + '_' + analysisTime.strftime('%Y%m%d%H')  + '_' + str(leadtimeHrs) + 'hours_' + str(deltaMin) + 'min'
    outFile = r'%s' % (outDir + fcstName + '.nc')
    
    # check if file was already extracted
    if os.path.isfile(outFile) and not overwrite:
        print('File already exists: ' + outFile)
        return outFile

    # prepare the filename of the temporary config file for fieldextra
    configFile = 'nl_cosmo_laf'
    fileIn = configFile
    fileOut = configFile + '_' + fcstName
    
    # find the <prefix> variable and replace it with the input grib filename
    textToSearch = r'<referenceRun>'
    if modelName=='cosmo-1':
        textToReplace = 'c1ffsurf00000000'
    elif modelName=='cosmo-e':
        textToReplace = 'ceffsurf000_000'
    else:
        print('Model name %s is not supported.' % modelName)
        sys.exit()
    insert_variable(fileIn,fileOut,textToSearch,textToReplace)
    
    # find the <pathToFcst> variable and replace it with the input grib filename
    textToSearch = r'<pathToFcst>'
    textToReplace = r'%s' % gribDir  
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)

    # find the <pathToOutput> variable and replace it with the output filename
    textToSearch = r'<pathToOutput>'
    textToReplace = outFile
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <loopFile> variable and replace it with the given forecast timestamp
    textToSearch = r'<loopFile>'
    if modelName=='cosmo-1':
        textToReplace = 'c1ffsurf<DDHHMMSS>' 
    elif modelName=='cosmo-e':
        textToReplace = 'ceffsurf<HHH>_<mmm>'
    else:
        print('Model name %s is not supported.' % modelName)
        sys.exit()
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <modelName> variable and replace it with the given cosmo version
    textToSearch = r'<modelName>'
    textToReplace = modelName
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <resKm> variable and replace it with the given cosmo resolution
    if modelName=='cosmo-1':
        resKm = 1.1
    elif modelName=='cosmo-e':
        resKm = 2.2
    else:
        print('Model name %s is not supported.' % modelName)
        sys.exit()
    textToSearch = r'<resKm>'
    textToReplace = resKm
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <epspars> variable and replace it with the COSMO-E ensemble parameters
    if modelName=='cosmo-1':
        epspars = ''
    elif modelName=='cosmo-e':
        epspars = '\n  out_size_field=21\n  epsstart = 0, epsstop = 20, epsincr=1,'
    else:
        print('Model name %s is not supported.' % modelName)
        sys.exit()
    textToSearch = r'<epspars>'
    textToReplace = epspars
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <fieldName> variable and replace it with the given field name to be retrieved
    textToSearch = r'<fieldName>'
    textToReplace = fieldName
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <tstart> variable and replace it 
    textToSearch = r'<tstart>'
    textToReplace = '0'
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <tend> variable and replace it 
    textToSearch = r'<tstop>'
    if modelName=='cosmo-1':
        textToReplace = str(int(leadtimeHrs*3600))
    elif modelName=='cosmo-e':
        textToReplace = str(int(leadtimeHrs))
    else:
        print('Model name %s is not supported.' % modelName)
        sys.exit()
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <tincr> variable and replace it 
    textToSearch = r'<tincr>'
    if modelName=='cosmo-1':
        textToReplace = str(int(deltaMin*60))
    elif modelName=='cosmo-e':
        textToReplace = str(int(deltaMin/60))
    else:
        print('Model name %s is not supported.' % modelName)
        sys.exit()
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # add temporal operator for precip
    textToSearch = r'<toper>'
    if (fieldName=='TOT_PREC') and (modelName=='cosmo-1'):
        textToReplace = ', toper="delta,%i,minute"' % deltaMin
    elif (fieldName=='TOT_PREC') and (modelName=='cosmo-e'):
        textToReplace = ', toper="delta,%i,hour"' % (deltaMin/60)
    else:
        print('Model name %s is not supported.' % modelName)
        sys.exit()
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)

    # call the fieldextra routine
    cmd = '/oprusers/owm/bin/fieldextra ' + str(fileOut)
    print('Run: ' + cmd)
    process = subprocess.Popen('ulimit -s unlimited; export OMP_STACKSIZE=500M; ' + cmd, shell=True)
    process.wait()
    
    # remove the temporary files
    cmd = 'rm ' + str(fileOut)
    os.system(cmd)

    print('saved: ' + outFile)
        
    # move back to previous wd
    os.chdir(cwd)
    
    return outFile
    
def insert_variable(fileIn,fileOut,textToSearch,textToReplace):
    '''
    Function to fill the fieldextra config file template with relevant parameters.
    
    Parameters
    ----------
    fileIn : str
        Filename of the config template file.
    fileOut : str
        New filenamne.
    textToSearch : str 
        Placeholder to replace.
    textToReplace : str
        String to insert.
    '''
    with open(fileIn) as f_in:
        text = f_in.read()
    with open(fileOut, 'w') as f_out:    
        f_out.write(re.sub(textToSearch, lambda x: '{}'.format(textToReplace), text))

def get_forecast_folder(modelName,analysisTime):

    if modelName=='cosmo-1':
        baseDir = '/store/s83/owm/COSMO-1/'

    elif modelName=='cosmo-e':
        baseDir = '/store/s83/owm/COSMO-E/'

    else:
        print('Model name %s is not supported.' % modelName)
        sys.exit()
        
    fcstYearDir = 'FCST' + analysisTime.strftime('%y')    
    if os.path.isdir(baseDir + fcstYearDir) == False:
        print('Error: %s does not exists.' % (baseDir + fcstYearDir))
        print('Available folders:')
        avail_dir = glob.glob(baseDir + 'FCST*')
        for i in avail_dir:
            print('\t ' + i)
        sys.exit()
        
    # Build full path to input forecast folder 
    cosmoOpenWildCard = '*'
    fcstDirWildCard = baseDir + fcstYearDir + '/' + analysisTime.strftime('%y%m%d%H') + '_' + cosmoOpenWildCard  
    fcstDir = get_filename_matching_regexpr(fcstDirWildCard)
    fcstDirGrib = fcstDir + '/grib/'
        
    return fcstDirGrib

def get_filename_matching_regexpr(fileNameWildCard):
    inDir = os.path.dirname(fileNameWildCard)
    baseNameWildCard = os.path.basename(fileNameWildCard)

    # Check if directory exists
    if os.path.isdir(inDir) == False:
        print('Directory: ' + inDir + ' does not exists.')
        fileName = ''
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
                fileName = ''
    else:
        fileName = ''
        
    return fileName
    
def find_nearest_forecast(timeStartStr, modelName, lat_timeMin = -1, lag = 0):
    
    # Get datetime format
    timeStart = datetime.datetime(int(timeStartStr[0:4]), int(timeStartStr[4:6]), int(timeStartStr[6:8]), int(timeStartStr[8:10]),int(timeStartStr[10:12]))

    if modelName == 'cosmo-1':
        timeCosmo = datetime.datetime(int(timeStartStr[0:4]), int(timeStartStr[4:6]), int(timeStartStr[6:8]),21,0) # the last run of the day  
        
        if lat_timeMin < 0:
            lat_timeMin = 100
    
        nruns=0
        while nruns < 10:
            
            timeDiff = timeStart - timeCosmo
            if timeDiff.total_seconds() >= lat_timeMin*60:
                fcstrun = (timeCosmo - lag*datetime.timedelta(hours=3)).strftime('%Y%m%d%H%M')
                break
            else:
                timeCosmo =  timeCosmo - datetime.timedelta(hours=3)
                nruns += 1
    elif modelName == 'cosmo-e':
        timeCosmo = datetime.datetime(int(timeStartStr[0:4]), int(timeStartStr[4:6]), int(timeStartStr[6:8]),12,0) # the last run of the day 
        
        if lat_timeMin < 0:
            lat_timeMin = 120
            
        nruns=0
        while nruns < 10:
            
            timeDiff = timeStart - timeCosmo
            if (timeDiff.total_seconds() >= lat_timeMin*60):
                fcstrun = (timeCosmo - lag*datetime.timedelta(hours=12)).strftime('%Y%m%d%H%M')
                break
            else:
                timeCosmo =  timeCosmo - datetime.timedelta(hours=12)
                nruns += 1  
        
    return fcstrun
 
def load_3darray_netcdf(filename):
    '''
    Read the netcdf file that is produced by fieldextra.
    The dimension of the output numpy array will be [time,y,x].
    Note that the reference is the lower left corner, so for most applications you will have to flip upside-down the images with np.flipud().
    
    Parameters
    ----------
    filename : str
        Filename of the netcdf file.
    '''
    # read netcdf file
    nc_fid = netCDF4.Dataset(filename, 'r', format='NETCDF4')
    variableNames = [str(var) for var in nc_fid.variables]

    # load time
    time_var = nc_fid.variables['time']
    timestamps = netCDF4.num2date(time_var[:],time_var.units)
    
    # load coordinates 
    x = nc_fid.variables["x_1"][:]
    y = nc_fid.variables["y_1"][:]
    Xcoords = np.array(x).squeeze()
    Ycoords = np.array(y).squeeze() 
         
    # load precip data
    data = nc_fid.variables[variableNames[-1]]
    noData = nc_fid.variables[variableNames[-1]]._FillValue 
    
    # convert to numpy array
    data = np.array(data)
    
    # change noData to Nan
    data[data==noData] = np.nan

    return data, timestamps, Xcoords, Ycoords
 
def load_4darray_netcdf(filename):
    '''
    Read the netcdf file that is produced by fieldextra for COSMO-E.
    The dimension of the output numpy array will be [time,eps,y,x].
    Note that the reference is the lower left corner, so for most applications you will have to flip upside-down the images with np.flipud().
    
    Parameters
    ----------
    filename : str
        Filename of the netcdf file.
    '''

    # read netcdf file
    nc_fid = netCDF4.Dataset(filename, 'r', format='NETCDF4')
    variableNames = [str(var) for var in nc_fid.variables]

    # load time
    time_var = nc_fid.variables['time']
    timestamps = netCDF4.num2date(time_var[:],time_var.units)

    # load precip
    data = nc_fid.variables[variableNames[-1]] 
    noData = data._FillValue 
    
    # convert to numpy array
    data = np.array(data)
    
    # change noData to Nan
    data[data==noData] = np.nan
    
    # load coordinates
    x = nc_fid.variables["x_1"][:]
    y = nc_fid.variables["y_1"][:]
    Xcoords = np.array(x).squeeze()
    Ycoords = np.array(y).squeeze()
    
    # load member numbers
    members_var = nc_fid.variables['epsd_1']
    members = np.array(members_var)

    return data, timestamps, members, Ycoords, Xcoords 
