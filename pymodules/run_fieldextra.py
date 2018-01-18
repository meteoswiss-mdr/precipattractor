#!/usr/bin/env python
"""
DESCRIPTION

    Utils for extracting COSMO data using fieldexta. Currently only analysis for COSMO-2 and COSMO-7 are supported.

AUTHOR

    Daniele Nerini <daniele.nerini@meteoswiss.ch>

VERSION

    1.0
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
                    modelName='cosmo-7'):
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

    # prepare the filename of the temporary config file for fieldextra
    configFile = 'nl_cosmo_laf'
    fileIn = configFile
    fileOut = configFile + '_' + fcstName
    
    # find the in_file variable and replace it with the input grib filename
    textToSearch = r'<pathToFcst>'
    textToReplace = r'%s' % gribDir  
    insert_variable(fileIn,fileOut,textToSearch,textToReplace)

    # find the <pathToOutput> variable and replace it with the output filename
    textToSearch = r'<pathToOutput>'
    textToReplace = outFile
    insert_variable(fileOut,fileOut,textToSearch,textToReplace)
    
    # find the <analysisRun> variable and replace it with the given analysis timestamp
    textToSearch = r'<analysisRun>'
    textToReplace = analysisTime.strftime('%Y%m%d')
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