#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

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
parser.add_argument('-start0', default=1, type=int,help='Whether to start the periods at 0 time or args.accum later.')

args = parser.parse_args()

if (args.n == -1) & (args.days == -1):
    print('You have to define either the number of periods -p or the interval length in days -days')
    sys.exit(1)

timeStart = ti.timestring2datetime(args.start)
timeEnd = ti.timestring2datetime(args.end)

# Compute total duration of the period
totalDurationSec = (timeEnd - timeStart).total_seconds()
timeDeltaAccum = datetime.timedelta(minutes = args.accum)

###### Get time stamps based on number of intervals (not tested)
if (args.n != -1) & (args.days == -1):
    # Compute duration of the period intervals
    durationPeriodSec = int(totalDurationSec/args.n)
    if durationPeriodSec >= totalDurationSec:
        numberPeriods = 1
        timeStampsArray = np.array([args.start,args.end])
    else:   
        timeStamps = []
        for t in range(0, args.n):
            # Time start
            timeStartRounded = timeStart.replace(hour=0, minute=0, second=0)
            if (t == 0) or (args.start0 == 1):
                timeStartLocalStr = ti.datetime2timestring(timeStartRounded)
            else:
                timeStartLocalStr = ti.datetime2timestring(timeStartRounded + timeDeltaAccum)
            
            # Time end
            if (args.start0 == 1):
                timeEndLocal = timeStartRounded + datetime.timedelta(seconds = durationPeriodSec) - timeDeltaAccum
            else:
                timeEndLocal = timeStartRounded + datetime.timedelta(seconds = durationPeriodSec)
                
            timeEndLocalStr = ti.datetime2timestring(timeEndLocal)
            
            # Collect times
            timeStamps.append([timeStartLocalStr[0:12], timeEndLocalStr[0:12]])
            
            # Update time start
            timeStart = timeStart + datetime.timedelta(seconds = durationPeriodSec)

        timeStampsArray = np.array(timeStamps)
        numberPeriods = len(timeStampsArray)

###### Get time stamps based on duration of intervals
if (args.days != -1) & (args.n == -1):
    # Compute duration of the period intervals
    durationPeriodSec = datetime.timedelta(days = args.days).total_seconds()
    numberPeriods = int(totalDurationSec/durationPeriodSec)
    
    if durationPeriodSec >= totalDurationSec:
        numberPeriods = 1
        timeStampsArray = np.array([args.start,args.end])
    else:
        timeStamps = []
        for t in range(0, numberPeriods + 1):
            # Time start
            if (t == 0) or (args.start0 == 1) or (t == (numberPeriods + 1)):
                timeStartLocalStr = ti.datetime2timestring(timeStart)
            else:
                timeStartLocalStr = ti.datetime2timestring(timeStart + timeDeltaAccum)
            
            # Time end
            if (args.start0 == 1) or (t == numberPeriods):
                timeEndLocal = timeStart + datetime.timedelta(seconds = durationPeriodSec) - timeDeltaAccum
            else:
                timeEndLocal = timeStart + datetime.timedelta(seconds = durationPeriodSec)
            if timeEndLocal > timeEnd:
                timeEndLocal = timeEnd
            timeEndLocalStr = ti.datetime2timestring(timeEndLocal)
            
            # Collect times
            timeStamps.append([timeStartLocalStr[0:12], timeEndLocalStr[0:12]])
            
            # Update time start
            timeStart = timeStart + datetime.timedelta(seconds = durationPeriodSec)

        timeStampsArray = np.array(timeStamps)
        numberPeriods = len(timeStampsArray)

################################################        
print("Number of periods: ", numberPeriods)
print(timeStampsArray)
##### Save file with time stamps of periods
fileName = outDir + '/timePeriods.txt'
np.savetxt(fileName, timeStampsArray, fmt='%s', delimiter='\t', newline='\n')
print(fileName, ' saved.')