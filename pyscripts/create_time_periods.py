#!/usr/bin/env python

import argparse
import datetime
import numpy as np
import sys

import time_tools_attractor as ti

import getpass
usrName = getpass.getuser()

outDir='/users/' + usrName + '/precipattractor/shscripts'

# Parse input arguments
parser = argparse.ArgumentParser(description='Create time periods to pass to the bash script for parallel radar data processing.')
parser.add_argument('-start', default='201601010000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-end', default='201601310000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-n', default=-1, type=int,help='Number of periods.')
parser.add_argument('-days', default=-1, type=int,help='Periods interval length [days].')
parser.add_argument('-accum', default=5, type=int,help='Accumulation time of products.')

args = parser.parse_args()

if (args.n == -1) & (args.days == -1):
    print('You have to define either the number of periods -p or the interval length in days -days')
    sys.exit(1)

timeStart = ti.timestring2datetime(args.start)
timeEnd = ti.timestring2datetime(args.end)

# Compute total duration of the period
totalDurationSec = (timeEnd - timeStart).total_seconds()

###### Get time stamps based on number of intervals (not tested)
if (args.n != -1) & (args.days == -1):
    # Compute duration of the period intervals
    durationPeriod = int(totalDurationSec/args.n)

    timeStamps = []
    for t in range(0, args.n):
        if t != 0:
            timeStartLocalStr = ti.datetime2timestring(timeStart + datetime.timedelta(minutes = 5))
        else:
            timeStartLocalStr = ti.datetime2timestring(timeStart)
        timeEndLocal = timeStart + datetime.timedelta(seconds = durationPeriod)
        timeEndLocalStr = ti.datetime2timestring(timeEndLocal)
        
        timeStamps.append([timeStartLocalStr, timeEndLocalStr])
        
        # Update time start
        timeStart = timeStart + datetime.timedelta(seconds = durationPeriod)

    timeStampsArray = np.array(timeStamps)
    print(timeStampsArray)

###### Get time stamps based on duration of intervals
if (args.days != -1) & (args.n == -1):
    # Compute duration of the period intervals
    durationPeriodSec = datetime.timedelta(days = args.days).total_seconds()
    numberPeriods = int(totalDurationSec/durationPeriodSec)
    
    timeStamps = []
    for t in range(0, numberPeriods + 1):
        if t != 0 and t != (numberPeriods + 1):
            timeStartLocalStr = ti.datetime2timestring(timeStart + datetime.timedelta(minutes = args.accum))
        else:
            timeStartLocalStr = ti.datetime2timestring(timeStart)
        if  t != numberPeriods:
            timeEndLocal = timeStart + datetime.timedelta(seconds = durationPeriodSec)
        else:
            timeEndLocal = timeEnd
        timeEndLocalStr = ti.datetime2timestring(timeEndLocal)
        
        timeStamps.append([timeStartLocalStr[0:12], timeEndLocalStr[0:12]])
        
        # Update time start
        timeStart = timeStart + datetime.timedelta(seconds = durationPeriodSec)

    timeStampsArray = np.array(timeStamps)
    print(timeStampsArray)

print("Number of periods: ", timeStampsArray.shape[0])
##### Save file with time stamps of periods
fileName = outDir + '/timePeriods.txt'
np.savetxt(fileName, timeStampsArray, fmt='%s', delimiter='\t', newline='\n')
print(fileName, ' saved.')