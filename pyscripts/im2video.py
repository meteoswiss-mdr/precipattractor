#!/usr/bin/env python

import argparse
import sys
from moviepy.editor import *
import glob
import datetime
import fnmatch

import time_tools_attractor as ti
import io_tools_attractor as io

#######################
product = 'CPC'
inBaseDir = '/scratch/lforesti/data/' # directory to read from
plotSpectrum = '2d'
outBaseDir = '/users/lforesti/results/'
framesPerSecond = 3
timeAccumMin = 5

#######################
parser = argparse.ArgumentParser(description='Plot radar rainfall field statistics.')
parser.add_argument('-start', default='201601310000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-end', default='201601310000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-product', default='AQC', type=str,help='Which radar rainfall product to use (AQC, CPC, etc).')
args = parser.parse_args()

product = args.product
if (int(args.start) > int(args.end)):
    print('Time end should be after time start')
    sys.exit(1)

if (int(args.start) < 198001010000) or (int(args.start) > 203001010000):
    print('Invalid -start or -end time arguments.')
    sys.exit(1)
else:
    timeStartStr = args.start
    timeEndStr = args.end

### Get list of filenames
timeStart = ti.timestring2datetime(timeStartStr)
timeEnd = ti.timestring2datetime(timeEndStr)

fileNameExpr = product + '_' + plotSpectrum + 'PS*'
fileList = io.get_files_period(timeStart, timeEnd, inBaseDir, fileNameExpr, tempResMin = 5)

# Check if there are file found
if len(fileList) == 0:
    print('No files found to generate video. Check your input filestamps.')
    sys.exit()

######## Generate clip
print('Foudn files for video: ', fileList)
clip = ImageSequenceClip(fileList, fps=framesPerSecond)
# Write out clip
outputFileName = outBaseDir + product + timeStartStr + '-' + timeEndStr + '_movieDBZ_' + plotSpectrum + 'PS.mp4'
clip.write_videofile(outputFileName)
