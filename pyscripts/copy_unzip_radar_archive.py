#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import os
import datetime
import time
import argparse
import sys

import getpass
username = getpass.getuser()

import time_tools_attractor as ti

################################
timeStartStr = '201604010000'
timeEndStr =   '201604270000'
timeAccumMin = 60
product = 'CPCH' #'CPCH'
inBaseDir = "/store/msrad/radar/swiss/data/"
outBaseDir = "/scratch/" + username + "/data/"
tempSampHours = 24 # Temporal sampling for files 

########GET ARGUMENTS FROM CMD LINE####
parser = argparse.ArgumentParser(description='Copy and unzip radar data (from store to scratch).')
parser.add_argument('-start', default='201604010000', type=str,help='Start time of the period: YYYYMMDDHHmmSS')
parser.add_argument('-end', default='201604010000', type=str,help='End time of the period: YYYYMMDDHHmmSS')
parser.add_argument('-product', default='AQC', type=str,help='Which radar rainfall product to use (AQC, CPC, etc).')
parser.add_argument('-accum', default=5, type=int,help='Accumulation time of the product [minutes].')
parser.add_argument('-username', default='lforesti', type=str,help='')

args = parser.parse_args()

if (int(args.start) < 198001010000) or (int(args.start) > 203001010000):
    print('Invalid -start or -end time arguments.')
    sys.exit(1)
else:
    timeStartStr = args.start
    timeEndStr = args.end

product = args.product
timeAccumMin = args.accum
timeAccumMinStr = '%05i' % timeAccumMin
username = args.username
################################

timeStart = ti.timestring2datetime(timeStartStr)
timeEnd = ti.timestring2datetime(timeEndStr)

##### LOOP OVER FILES #############
tic = time.clock()
timeLocal = timeStart
while timeLocal <= timeEnd:
    year = timeLocal.year
    hour = timeLocal.hour
    minute = timeLocal.minute
    # Get Julian day
    julianDay = ti.get_julianday(timeLocal)

    # Create filename
    yearStr =  str(year)[2:4]
    julianDayStr = '%03d' % julianDay
    hourminStr = ('%02i' % hour) + ('%02i' % minute)
    yearDayStr = yearStr + julianDayStr

    subDir = str(year) + "/" + yearDayStr + "/"
    fileName = product + yearDayStr + ".zip"

    inFile = inBaseDir + subDir + fileName
    outDir = outBaseDir + subDir

    # Check if file exists and if not try another one
    if os.path.isfile(inFile) == False:
        fileName = product + yearDayStr + "_" + timeAccumMinStr + ".zip"
        inFile = inBaseDir + subDir + fileName
    
    if os.path.isfile(inFile) == True: 
        # Copy data
        cmd = 'mkdir -p ' + outDir
        os.system(cmd)
        cmd = 'cp -f ' + inFile + ' ' + outDir
        print(cmd)
        os.system(cmd)

        # Unzip data in the output directory. Unzip only specific accumulation
        outFile = outDir + fileName
        if product == 'RZC':
            cmd = 'unzip -o ' + outFile + ' "RZC????????[05]??.801"' + ' -d ' + outDir
        elif product == 'HZT':
            # Unzip only analysis
            cmd = 'unzip -q -o ' + outFile + ' "*.800"' + ' -d ' + outDir
        else:
            cmd = 'unzip -q -o ' + outFile + ' "*_' + timeAccumMinStr + '."*' + ' -d ' + outDir
        print(cmd)
        os.system(cmd)

        # Remove zip file at the output
        cmd = 'rm ' + outFile
        print(cmd)
        os.system(cmd)
        print('----------------------------------------------')
    else:
        print(inFile, 'not found.')

    # Add 5 minutes
    timeLocal = timeLocal + datetime.timedelta(hours=tempSampHours)


