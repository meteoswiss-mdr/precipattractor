#!/usr/bin/env python

import datetime
import os
import subprocess
import sys
import glob 
import re
import fnmatch
import numpy as np
import netCDF4

'''
NOTE: COSMO-1 forecasts available starting 30.09.2015 12 UTC only. 

To use fieldextra it is necessary to relax system limits imposed on size of
stack, both for sequential mode and for OpenMP mode. Please exedcute these commands before you start fieldextra on kesch:

# bash:
ulimit -s unlimited              # unlimit stack size
export OMP_STACKSIZE=500M        # increase OpenMP stack size

'''

def run_fieldextra_analysis(analysisTimeStr, 
                    gribDir,
                    outBaseDir='',
                    fieldName='HZEROCL',
                    modelName='cosmo-7'):

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
        sys.exit()

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
    with open(fileIn) as f_in:
        text = f_in.read()
    with open(fileOut, 'w') as f_out:    
        f_out.write(re.sub(textToSearch, lambda x: '{}'.format(textToReplace), text))

def load_3darray_netcdf(filename):

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