#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys
from moviepy.editor import *
import glob
import datetime
import fnmatch
import numpy as np
import os

import time_tools_attractor as ti
import io_tools_attractor as io

#######################
inBaseDir = '/scratch/lforesti/data/' # directory to read from
outBaseDir = '/users/lforesti/results/'
framesPerSecond = 3
timeAccumMin = 5

#######################
parser = argparse.ArgumentParser(description='Plot radar rainfall field statistics.')
parser.add_argument('-start', default='201601310000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-end', default='201601310000', type=str,help='Starting date YYYYMMDDHHmmSS.')
parser.add_argument('-product', default='AQC', type=str,help='Which radar rainfall product to use (AQC, CPC, etc).')
parser.add_argument('-spec', default='1d', type=str,help='Spectrum type (1d, 2d or autocorr).')
parser.add_argument('-format', default='mp4', type=str,help='Video format (mp4 or avi).')
parser.add_argument('-resize', default=100, type=int, help='Resize percentage (only for gif).')

args = parser.parse_args()

plotSpectrum = args.spec
extension = '.' + args.format
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

fileNameExpr = product + '_' + plotSpectrum + 'PS_*'
fileList = io.get_files_period(timeStart, timeEnd, inBaseDir, fileNameExpr, tempResMin = 5)

# Check if there are file found
if len(fileList) == 0:
    print('No files found to generate video. Check your input filestamps.')
    sys.exit()

######## Generate clip

print("Nr. files: ", len(fileList))
print('Files found for video: ', *fileList, sep='\n')
outputFileName = outBaseDir + product + timeStartStr + '-' + timeEndStr + '_movieDBZ_' + plotSpectrum + 'PS' + extension

if args.format == 'gif':
    cmd = 'convert -verbose -resize ' + str(args.resize) + '% -delay ' + str(100/framesPerSecond) + ' -loop 0 ' + ' '.join(fileList) + ' '+ outputFileName
    print(cmd)
    os.system(cmd)
elif args.format == 'avi':
    clip = ImageSequenceClip(fileList, fps=framesPerSecond)
    # Write out clip
    clip.write_videofile(outputFileName, codec='png')
elif args.format == 'mp4':
    clip = ImageSequenceClip(fileList, fps=framesPerSecond)
    # Write out clip
    clip.write_videofile(outputFileName, codec='libx264')
else:
    print('Format should be gif, avi or mp4')
    sys.exit(1)
