#!/usr/bin/env python
'''
Module to perform various time operations.

Documentation convention from https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

07.07.2016
Loris Foresti
'''

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import time
import sys

fmt1 = "%.1f"
fmt2 = "%.2f"

def timestring2datetime(timestring):
    '''
    Function to convert a time stamp string YYYYmmDDHHMMSS to a datetime object.
    
    Parameters
    ----------
    timestring : str
        Time string YYYYmmDDHHMMSS
    
    Returns
    -------
    timeDate: datetime
        Datetime object
    '''
    #timeDate = datetime.datetime.strptime(timestring,'%Y%m%d%H%M%S')
    timeDate = datetime.datetime(int(timestring[0:4]), int(timestring[4:6]), int(timestring[6:8]), int(timestring[8:10]),int(timestring[10:12]))
    return(timeDate)
    
def datetime2timestring(timeDate):
    '''
    Function to convert datetime object to a time stamp string YYYYmmDDHHMMSS.
    
    Parameters
    ----------
    timeDate : datetime
        Datetime object
    
    Returns
    -------
    timeString: str
        Time string YYYYmmDDHHMMSS
    '''
    timeString = timeDate.strftime("%Y%m%d%H%M%S")
    return(timeString)
    
def datetime2juliantimestring(timeDate, format='YYYYJJJHHMM'):
    '''
    Function to convert datetime object to a Julian time stamp string YYYYJJJHHMM.
    
    Parameters
    ----------
    timeDate : datetime
        Datetime object
    
    Returns
    -------
    timeString: str
        Time string YYYYJJJHHMM
    '''
    year, yearStr, julianDay, julianDayStr = parse_datetime(timeDate)
    hour = timeDate.hour
    minute = timeDate.minute
    hourminStr = ('%02i' % hour) + ('%02i' % minute)
    if format == 'YYYYJJJHHMM':
        timeString = yearStr + julianDayStr + hourminStr
    if format == 'YYJJJHHMM':
        timeString = yearStr[2:4] + julianDayStr + hourminStr
        
    return(timeString)
    
def juliantimestring2datetime(timeString, format='YYJJJHHMM'): 
    '''
    Function to convert Julian time stamp string to a datetime object.
    
    Parameters
    ----------
    timeString: str
        Time string YYYYJJJHHMMSS
    
    Returns
    -------
    timeDate : datetime
        Datetime object
        
    Note: julian day starts at 001 (i.e. January 1st)
    '''
    if format=='YYYYJJJHHMMSS':
        if not len(timeString) == 13:
            print("Not the right string length.")
            sys.exit(1)
        year = int(timeString[0:4])
        day = int(timeString[4:7]) - 1
        hour = int(timeString[7:9])
        min = int(timeString[9:11])
        sec = int(timeString[11:13])
        
        totaldeltaDays = day + hour/24 + min/60/24 + sec/60/60/24
        timeDate = datetime.datetime(year, 1, 1) + datetime.timedelta(days=totaldeltaDays) 

    elif format=='YYJJJHHMM':
        if not len(timeString) == 9:
            print("Not the right string length.")
            sys.exit(1)
        year = int(timeString[0:2])
        if year > 80:
            year = 1900 + year
        else:
            year = 2000 + year
        day = int(timeString[2:5]) - 1
        hour = int(timeString[5:7])
        min = int(timeString[7:9])
    
        totaldeltaDays = day + hour/24 + min/60/24
        timeDate = datetime.datetime(year, 1, 1) + datetime.timedelta(days=totaldeltaDays) 
        
    else:
        print("Julian time stamp string format not supported.")
        sys.exit(1)

    return(timeDate)   

def juliantimestring2datetime_array(timeStampJulianArray, format='YYJJJHHMM', timeString=True):
    '''
    Same as above but for a list or array of time stamps.
    '''
    nrSamples = len(timeStampJulianArray)
    
    # If not many samples...
    if nrSamples < 1000000:
        timeStampJulianArrayStr = np.array(map(lambda n: "%0.9i"%n, timeStampJulianArray))
        timeStampJulianArrayDt = map(juliantimestring2datetime, timeStampJulianArrayStr)
        if timeString == True:
            timeStampArrayStr = map(datetime2timestring, timeStampJulianArrayDt)
        else:
            timeStampArrayStr = []
        return(timeStampJulianArrayDt, timeStampArrayStr)
    
    else:
        # If a lot of samples
        timeStampJulianSet = np.unique(timeStampJulianArray)
        
        nrUniqueSamples = len(timeStampJulianSet)
        print(nrSamples, nrUniqueSamples)
        
        timeStampDt = np.empty((nrSamples,), dtype='datetime64[m]')
        timeStampStr = np.empty((nrSamples,), dtype='S12')
        
        # Do the operations over the unique time stamps
        for i in range(0,nrUniqueSamples):
            timeStampJulianStr = "%0.9i"% timeStampJulianSet[i]
            dt = juliantimestring2datetime(timeStampJulianStr, format=format)
            
            bool = (timeStampJulianArray == timeStampJulianSet[i])
            
            # Set values in array
            timeStampDt[bool] = dt
            if timeString == True:
                dtStr = datetime2timestring(dt)
                timeStampStr[bool] = dtStr

            # Print out advancement (for large arrays)    
            if ((i % 100) == 0):
                print(fmt1 % (i/nrUniqueSamples*100),"%")
            
        return(timeStampDt, timeStampStr)   
    
def get_julianday(timeDate):
    '''
    Get Julian day from datetime object.
    
    Parameters
    ----------
    timeDate : datetime
        Datetime object
    
    Returns
    -------
    julianDay: int
        Julian day
    '''
    julianDay = timeDate.timetuple().tm_yday
    return(julianDay)
    
def parse_datetime(timeDate):
    '''
    Function to parse a datetime object and return the year and Julian day in integer and string formats.
    
    Parameters
    ----------
    timeDate : datetime
        Datetime object
    
    Returns
    -------
    year: int
        Year
    yearStr: str
        Year string in YY
    julianDay: int
        Julian day
    julianDayStr: str 
        Julian day string JJJ
    '''
    year = timeDate.year
    yearStr =  str(year)[2:4]
    julianDay = get_julianday(timeDate)
    julianDayStr = '%03i' % julianDay
    yearJulianStr = yearStr + julianDayStr
    return(year, yearStr, julianDay, julianDayStr)
    
def timestring_array2datetime_array(arrayTimeStampsStr):
    '''
    Function to convert a list array of time strings YYYYmmDDHHMMSS
    into a list of datetime objects
    
    Parameters
    ----------
    arrayTimeStampsStr : list(str)
        List of time strings YYYYmmDDHHMMSS
    
    Returns
    -------
    arrayTimeStampsDt: list(datetime)
        List of datetime objects
    '''
    
    timeStamps = np.array(arrayTimeStampsStr,dtype=int)
    timeStampsStr = np.array(list(map(str,timeStamps)))
    
    arrayTimeStampsDt = []
    for t in range(0,len(arrayTimeStampsStr)):
        timeDate = timestring2datetime(str(timeStampsStr[t]))
        arrayTimeStampsDt.append(timeDate)
    
    return(arrayTimeStampsDt)
    
def get_HHmm_str(hour, minute):
    '''
    Function to concatenate hours and minutes into a 4-digit string.
    
    Parameters
    ----------
    hour : int
    minute: int
    
    Returns
    -------
    hourminStr: str
        4-digit hour and minute string (HHMM)
    '''
    hourminStr = ('%02i' % hour) + ('%02i' % minute)
    return(hourminStr)

def get_subdir(year, julianDay):
    '''
    Function to create the subdirectory string from the year and Julian day.
    
    Parameters
    ----------
    year : int
    julianDay: int
    
    Returns
    -------
    subDir: str
        Sub-directory string YYYY/YYJJJ/
    '''
    yearStr = str(year)[2:4]
    julianDayStr = '%03i' % julianDay
    subDir = str(year) + '/' + yearStr + julianDayStr + '/'
    return(subDir)
    
def datetime2absolutetime(timeDate):
    '''
    Function to convert a datetime object into an epoch (absolute time in seconds since 01/01/1970).
    
    Parameters
    ----------
    timeDate : datetime
    
    Returns
    -------
    absTime: int
        Number of seconds since 01/01/1970
    '''
    
    # Convert list or numpy array of values
    if type(timeDate) == list or type(timeDate) == np.ndarray:
        absTime = []
        for t in range(0,len(timeDate)):
            absTime.append(datetime2absolutetime(timeDate[t]))
    else:
        # Convert single value
        absTime = int((timeDate-datetime.datetime(1970,1,1)).total_seconds())
    
    # Convert list to numpy array if necessary
    if type(timeDate) == np.ndarray:
        absTime = np.array(absTime)
        
    return(absTime)

def absolutetime2datetime(absTime):
    timeDate = datetime.datetime(1970,1,1) + datetime.timedelta(seconds = absTime)
    return(timeDate)

def tic():
    global _start_time 
    _start_time = time.time()

def toc(appendText):
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Time passed: {}h:{}m:{}s'.format(t_hour,t_min,t_sec), appendText)
    
def sample_independent_times(timeStampsDt, indepTimeHours=6, method='start'):
    '''
    This function is not optimal as it selects the first time stamp respecting the condition (usually at the beginning of the event)
    '''
    if len(timeStampsDt) <= 1:
        return(timeStampsDt,[0])
    
    sortedIdx = np.argsort(timeStampsDt)
    timeStampsDt = np.sort(timeStampsDt)
    
    timeDiffs = np.diff(datetime2absolutetime(timeStampsDt))
    timeDiffs = np.hstack((timeDiffs[0], timeDiffs))
    
    indepTimeSecs = indepTimeHours*60*60
    #print(timeStampsDt.shape, timeDiffs)
    
    if method == 'start':
        timeDiffsAccum = 0
        indepIndices = []
        indepTimeStampsDt = []
        for i in range(0,len(timeStampsDt)):
            if (i == 0) | (timeDiffs[i] >= indepTimeSecs) | (timeDiffsAccum >= indepTimeSecs):
                indepIndices.append(sortedIdx[i])
                indepTimeStampsDt.append(sortedIdx[i])
                timeDiffsAccum = 0
            else:
                # Increment the accumulated time difference to avoid excluding the next sample 
                # if closer than X hours from the previous (if not included), 
                # but further than X hours than the one before the previous
                timeDiffsAccum = timeDiffsAccum + timeDiffs[i]

    indepIndices = np.array(indepIndices) 
    indepTimeStampsDt = np.array(indepTimeStampsDt)
    
    return(indepTimeStampsDt, indepIndices)

def generate_datetime_list(startDt, endDt, stepMin=5):
    '''
    Generate a list of datetimes from start to end (included).
    '''
    localTime = startDt
    listDt = []
    while localTime <= endDt:
        listDt.append(localTime)
        localTime = localTime + datetime.timedelta(minutes=stepMin)
        
    return(listDt)
        