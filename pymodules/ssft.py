#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import pickle

import radialprofile
import cv2

import scipy as sp  
from scipy import fftpack
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter, median_filter
#import pyfftw
from scipy.interpolate import griddata
from skimage import measure
from scipy.optimize import leastsq
import matplotlib.colors as colors

import data_tools_attractor as dt
import stat_tools_attractor as st
import optical_flow as of

###########################################################
########## Short-space Fourier transform  (ssft) ##########
###########################################################

def corrNoise(rainfield_dBZin, randValues=[], winsize=128, wintype='flat-hanning', overlap=0.5, nmembers = 1, \
                    doshiftandscale = 0, fillgaps = 0, fillmethod = 'nearest', warThr = 0.03, verbose = 1, seed = 42): 
    
    
    # start the clock
    tic = time.clock()
    
    # rain/no rain threshold
    rainfield_dBZ = rainfield_dBZin.copy()
    norain = np.nanmin(rainfield_dBZ)

    # get rid of nans
    rainfield_dBZ[np.isnan(rainfield_dBZ)] = norain
    
    rainfield_dBZ[rainfield_dBZ<=norain] = norain
    # rainfield_dBZ = rainfield_dBZ - norain
    
    if verbose==1:
        print(winsize, wintype, overlap, warThr, norain)
    
    # zero padding if global approach
    if winsize == rainfield_dBZ.shape[0]:
        globalApproach = True
        overlap = 0
        # originalSize = rainfield_dBZ.shape
        # nb = int(winsize/2)
        # rainfield_dBZ = cv2.copyMakeBorder(rainfield_dBZ,nb,nb,nb,nb,cv2.BORDER_CONSTANT,0)
        # winsize = rainfield_dBZ.shape[0]
        # nmembers = randValues.shape[2]
        # randValues = []
        # np.random.seed(42)
    else:
        globalApproach = False
        
    # remove small noise with a morphological operator (opening)
    # rainfield_dBZ = of.morphological_opening(rainfield_dBZ, thr=norain, n=3)  

    # define the shift of the window based on the overlap parameter       
    delta = int(winsize*(1 - overlap))
    delta = np.max((delta,1))
    
    # if overlap == 0 and wintype == 'hanning' and winsize < rainfield_dBZ.shape[0]:
        # winsize = int(winsize + 4)
        # delta = int(winsize - 2)
    
    # Generate a field of noise
    np.random.seed(seed)
    if len(randValues)==0:
        randValues = np.random.randn(rainfield_dBZ.shape[0],rainfield_dBZ.shape[1],nmembers)
    else:
        if len(randValues.shape) == 2:
            randValues = randValues[:,:,np.newaxis]
        nmembers = randValues.shape[2]
        
    # Turn on the fftw cache for improved performance
    pyfftw.interfaces.cache.enable()
        
    # Compute FFT for all noise fields
    fnoise=[]
    for n in xrange(nmembers):
        randValues[:,:,n] = _zscores(randValues[:,:,n])
        fnoise.append(pyfftw.interfaces.numpy_fft.fft2(randValues[:,:,n]))  
        
    ## Compute the windowed fft and store the spectra
         
    # Initialise variables
    count=-1
    fprecipNoShift=[]
    wpi=[];wpj=[];wpim=[];wpjm=[]
    fftyes=[]
    idxi = np.zeros((2,1),dtype=int); idxj = np.zeros((2,1),dtype=int)
    
    # loop through rows       
    for i in xrange(0,rainfield_dBZ.shape[0],delta):
        
        # loop through columns
        for j in xrange(0,rainfield_dBZ.shape[1],delta):
            
            idxi[0] = i
            idxi[1] = np.min((i + winsize, rainfield_dBZ.shape[0]))
            idxj[0] = j
            idxj[1] = np.min((j + winsize, rainfield_dBZ.shape[1])) 

            # build window
            wind = build2dWindow(((idxi[1]-idxi[0]),(idxj[1]-idxj[0])),wintype)
            
            # at least half of the window within the frame
            if (wind.shape[0]*wind.shape[1])/winsize**2 > 0.5: 
                count += 1
                mask = np.zeros(rainfield_dBZ.shape) 
                mask[int(idxi[0]):int(idxi[1]),int(idxj[0]):int(idxj[1])] = wind
                
                # apply window to rainfall field
                rmask = mask*rainfield_dBZ
                
                # save window coordinates
                wpi.append(idxi)
                wpj.append(idxj)
                wpim.append(np.mean(idxi))
                wpjm.append(np.mean(idxj))

                # continue only if any precip within window
                if  (np.sum(rmask>norain)/wind.size > warThr):
                    
                    # fft of the windowed rainfall field
                    fftw = pyfftw.interfaces.numpy_fft.fft2(rmask)
                    # normalize the spectrum
                    fftw.imag = _zscores(fftw.imag)
                    fftw.real = _zscores(fftw.real)
                    # fftw.imag = fftw.imag - np.mean(fftw.imag)
                    # fftw.real = fftw.real - np.mean(fftw.real)
                    # keep only the amplitude
                    fftw = np.abs(fftw)
                    # store it
                    fprecipNoShift.append(fftw)
                    fftyes.append(1)    
                    
                    doplot=0
                    if doplot==1:
                        psAx = plt.subplot(111)
                        resKm=1
                        
                        # Plot image of 2d PS
                        # rmask = np.random.randn(rainfield_dBZ.shape[0],rainfield_dBZ.shape[1])
                        fftw = pyfftw.interfaces.numpy_fft.fft2(rmask)
                        fftw = np.fft.fftshift(fftw)
                        psd2d = np.abs(fftw)**2/(fftw.shape[0]*fftw.shape[1])    
                        # Compute frequencies
                        freq = fftpack.fftfreq(fftw.shape[0], d=float(1))
                        freq = np.fft.fftshift(freq)
                        
                        fftSizeSub=64#fftw.shape[0]/2
                        fftSize = psd2d.shape
                        fftMiddleX = fftSize[1]/2
                        fftMiddleY = fftSize[0]/2
                        
                        # Select subset of autocorrelation/spectrum
                        psd2dsub = psd2d[fftMiddleY-fftSizeSub:fftMiddleY+fftSizeSub,fftMiddleX-fftSizeSub:fftMiddleX+fftSizeSub]
                        freq = freq[fftMiddleY-fftSizeSub:fftMiddleY+fftSizeSub]
                        freq = np.fft.fftshift(freq)
                        
                        clevsPS = np.arange(-5,70,5)
                        cmapPS = plt.get_cmap('nipy_spectral', clevsPS.shape[0])  #nipy_spectral, gist_ncar
                        normPS = colors.BoundaryNorm(clevsPS, cmapPS.N-1)
                        cmapPS.set_over('white',1)
                        imPS = psAx.imshow(10.0*np.log10(np.fliplr(psd2dsub)), interpolation='none', cmap=cmapPS)#, norm=normPS)
                        
                        # Create ticks in km
                        
                        ticks_loc = np.arange(0,2*fftSizeSub,1)
                        
                        # List of ticks for X and Y (reference from top)
                        ticksListX = np.hstack((np.flipud(-resKm/freq[1:fftSizeSub+1]),0,resKm/freq[1:fftSizeSub])).astype(int)
                        ticksListY = np.flipud(ticksListX)
                        
                        # List of indices where to display the ticks
                        delta = 4
                        idxTicksX = np.hstack((np.arange(1,fftSizeSub-2,delta),fftSizeSub-1,fftSizeSub+1,np.arange(fftSizeSub+3,2*fftSizeSub,delta))).astype(int)
                        idxTicksY = np.hstack((np.arange(0,fftSizeSub-3,delta),fftSizeSub-2,fftSizeSub,np.arange(fftSizeSub+2,2*fftSizeSub,delta))).astype(int)
                        plt.xticks(rotation=90)
                        psAx.set_xticks(ticks_loc[idxTicksX])
                        psAx.set_xticklabels(ticksListX[idxTicksX], fontsize=13)
                        psAx.set_yticks(ticks_loc[idxTicksY])
                        psAx.set_yticklabels(ticksListY[idxTicksY], fontsize=13)

                        plt.xlabel('Wavelength [km]', fontsize=15)
                        plt.ylabel('Wavelength [km]', fontsize=15)
                        
                        plt.title(r'2D power spectrum (rotated by 90$^{\circ}$)')
                        plt.show()
                        
                else:
                    fprecipNoShift.append(np.nan)
                    fftyes.append(0)
    
    
    if (~globalApproach and fillgaps == 1  and fillmethod == 'global'):
        wind = build2dWindow(rainfield_dBZ.shape,wintype)
        fglobal = pyfftw.interfaces.numpy_fft.fft2(rainfield_dBZ*wind)
        fglobal.imag = _zscores(fglobal.imag)
        fglobal.real = _zscores(fglobal.real)
        fglobal = np.abs(fglobal)
        fprecipNoShift.append(fglobal)
    
    ## Filter the white noise with the local spectra, take the nearest available if missing

    # Initialise variables
    wpim = np.array(wpim)
    wpjm = np.array(wpjm)
    fftyes = np.array(fftyes)
    fcorrNoiseTotal = np.zeros(randValues.shape)
    idxi = np.zeros((2,1),dtype=int); idxj = np.zeros((2,1),dtype=int)
    maskSum = np.zeros(randValues.shape) 
    
    # loop members
    for n in xrange(nmembers):
        count=-1
        
        if (nmembers > 1) and (verbose==1):
            dt.update_progress((n+1)/nmembers)
        # loop through rows       
        for i in xrange(0,rainfield_dBZ.shape[0],delta):
            
            # loop through columns
            for j in xrange(0,rainfield_dBZ.shape[1],delta):
                # print(i,j,winsize)
                idxi[0] = i
                idxi[1] = np.min((i + winsize, rainfield_dBZ.shape[0])) 
                idxj[0] = j
                idxj[1] = np.min((j + winsize, rainfield_dBZ.shape[1])) 
                
                # build window
                wind = build2dWindow(((idxi[1]-idxi[0]),(idxj[1]-idxj[0])),wintype)
                
                # stopping criteria
                if (wind.shape[0]*wind.shape[1])/winsize**2 > 0.5: 
                    count += 1
                    mask = np.zeros(rainfield_dBZ.shape) 
                    mask[int(idxi[0]):int(idxi[1]),int(idxj[0]):int(idxj[1])] = wind + 1e-6
                    
                    # the local spectrum exists
                    if fftyes[count]==1:
                        idxSpectrum = count
                    # or find the nearest available spectrum
                    elif (fftyes[count]==0) and (fillgaps == 1) and (fillmethod == 'nearest'):
                        dx = (wpim[count] - wpim)**2 + (wpjm[count] - wpjm)**2
                        dx[fftyes == 0] = np.inf
                        idxSpectrum = np.argmin(dx)
                    # or use global spectrum to fill the gaps
                    elif (fftyes[count]==0) and (fillgaps == 1) and (fillmethod == 'global'):   
                        idxSpectrum = -1
                    # or do nothing
                    else:
                        idxSpectrum = -2

                    if idxSpectrum >= -1:
                        # Build local white noise Fourier
                        # windN = build2dWindow(((idxi[1]-idxi[0]),(idxj[1]-idxj[0])),'rectangular')
                        # maskN = np.zeros(rainfield_dBZ.shape) 
                        # maskN[idxi[0]:idxi[1],idxj[0]:idxj[1]] = windN
                        # fnoise_l =  pyfftw.interfaces.numpy_fft.fft2(maskN*randValues[:,:,n])
                        
                        # Build the filter based on rain analysis
                        fcorrNoise = fprecipNoShift[idxSpectrum].copy()
                       
                        # apply the filter to the noise spectrum
                        fcorrNoise = fnoise[n]*fcorrNoise
                        # fcorrNoise = fnoise_l*fcorrNoise

                        # Do the inverse FFT
                        corrNoise = pyfftw.interfaces.numpy_fft.ifft2(fcorrNoise)
                        corrNoiseReal = np.array(corrNoise.real)
                        
                        # Merge 
                        fcorrNoiseTotal[:,:,n] = fcorrNoiseTotal[:,:,n] + corrNoiseReal.copy()*mask

                        # Update sum of weights
                        maskSum[:,:,n] = maskSum[:,:,n] + mask
                        
                        
                        # plt.clf()
                        # plt.imshow(corrNoiseReal)
                        # plt.autoscale(False)
                        # plt.plot([idxj[0],idxj[1],idxj[1],idxj[0],idxj[0]],[idxi[0],idxi[0],idxi[1],idxi[1],idxi[0]],color='red')
                        # plt.title((n,i,j))
                        # plt.pause(2)
                                            
    
    # normalize the sum
    idx = maskSum>0
    fcorrNoiseTotal[idx] = fcorrNoiseTotal[idx]/maskSum[idx]
    
    # pinkHex = '#%02x%02x%02x' % (232, 215, 242)
    # redgreyHex = '#%02x%02x%02x' % (156, 126, 148)
    # cm = [pinkHex, redgreyHex, "#640064","#AF00AF","#DC00DC","#3232C8","#0064FF","#009696","#00C832",
            # "#64FF00","#96FF00","#C8FF00","#FFFF00","#FFC800","#FFA000","#FF7D00","#E11900"] # light gray "#D3D3D3"
    # cmap = colors.ListedColormap(cm)
    # plt.imshow(fcorrNoiseTotal[:,:,0].squeeze(), vmin=-4,vmax=4, cmap=cmap)
    # plt.show()
    
    # apply shift and scale
    if doshiftandscale == 1:
        # repeate m times
        for n in xrange(nmembers):
            nmember = fcorrNoiseTotal[:,:,n]
            nmember = shiftandscale(rainfield_dBZ + norain,nmember)
            # nmember = shiftandscale(rainfield_dBZ,nmember)
            fcorrNoiseTotal[:,:,n] = nmember
    else:
        for n in xrange(nmembers):
            nmember = fcorrNoiseTotal[:,:,n]
            idxZeros = nmember==0
            fmean = nmember[~idxZeros].mean()
            fstd = nmember[~idxZeros].std()
            nmember = (nmember - fmean)/fstd
            nmember[idxZeros] = 0
            fcorrNoiseTotal[:,:,n] = nmember
            
    if nmembers==1:
        fcorrNoiseTotal = np.squeeze(fcorrNoiseTotal)
    
    toc = time.clock()
    totalTime = toc - tic 
    iterations = count
    return fcorrNoiseTotal,totalTime,iterations
    
def corrNoise_singleloop(rainfield_dBZ, randValues=[], winsize=128, wintype='hanning', overlap=0, nmembers = 1, verbose=False): 
    print(wintype,winsize)
    # start the clock
    tic = time.clock()

    # rain/no rain threshold
    norain = np.nanmin(rainfield_dBZ)
    rainfield_dBZ[rainfield_dBZ<=norain] = norain
    
    # get rid of nans
    rainfield_dBZ[np.isnan(rainfield_dBZ)] = norain

    # define the shift of the window based on the overlap parameter
    if winsize == rainfield_dBZ.shape[0]:
        overlap = 0
    delta = int(winsize*(1 - overlap))
    delta = np.max((delta,1))
    
    if overlap == 0 and wintype == 'hanning' and winsize < rainfield_dBZ.shape[0]:
        winsize = winsize + 4
        delta = winsize - 2
        
    # Generate a field of noise
    if len(randValues)==0:
        randValues = np.random.randn(rainfield_dBZ.shape[0],rainfield_dBZ.shape[1],nmembers)
    else:
        if len(randValues.shape) == 2:
            randValues = randValues[:,:,np.newaxis]
        nmembers = randValues.shape[2]
        
    # Turn on the fftw cache for improved performance
    pyfftw.interfaces.cache.enable()
        
    # Compute FFT for all noise fields
    fnoise=[]
    for n in range(nmembers):
        randValues[:,:,n] = (randValues[:,:,n] - randValues[:,:,n].mean())/randValues[:,:,n].std() # force zscores
        fnoise.append(pyfftw.interfaces.numpy_fft.fft2(randValues[:,:,n]))  
         
    # Initialise variables
    count=-1
    wpi=[];wpj=[]
    idxi = np.zeros((2,1),dtype=int); idxj = np.zeros((2,1),dtype=int)
    maskSum = np.zeros(randValues.shape)
    maskNans = np.zeros(randValues.shape)
    fcorrNoiseTotal = np.zeros(randValues.shape)
    
    # loop members
    for n in xrange(nmembers):
        if verbose and nmembers>1:
            dt.update_progress((n)/(nmembers-1))
        # loop through rows  
        for i in xrange(0,rainfield_dBZ.shape[0],delta):
            
            # loop through columns
            for j in xrange(0,rainfield_dBZ.shape[1],delta):
                
                idxi[0] = i
                idxi[1] = np.min((i + winsize, rainfield_dBZ.shape[0]))
                idxj[0] = j
                idxj[1] = np.min((j + winsize, rainfield_dBZ.shape[1])) 
     
                # build window
                wind = build2dWindow(((idxi[1]-idxi[0]),(idxj[1]-idxj[0])),wintype)

                # at least half of the window within the frame
                if (wind.shape[0]*wind.shape[1])/winsize**2 > 0.25: 
                    count += 1
                    mask = np.zeros(rainfield_dBZ.shape) 
                    mask[idxi[0]:idxi[1],idxj[0]:idxj[1]] = wind
                    
                    # apply window to rainfall field
                    rmask = mask*rainfield_dBZ

                    # save centre of the window
                    wpi.append(np.mean(idxi))
                    wpj.append(np.mean(idxj))

                    # continue only if any precip within window
                    if  np.sum(rmask>norain)/wind.size > 0.0:
                        # wind2 = build2dWindow(((idxi[1]-idxi[0]),(idxj[1]-idxj[0])),'rectangular')
                        # mask2 = np.zeros(rainfield_dBZ.shape) 
                        # mask2[idxi[0]:idxi[1],idxj[0]:idxj[1]] = wind2
                        # fnoise_l =  pyfftw.interfaces.numpy_fft.fft2(mask2*randValues[:,:,n])
                        # fft of the windowed rainfall field
                        fftw = pyfftw.interfaces.numpy_fft.fft2(mask*rainfield_dBZ)
                        # normalize the spectrum
                        # fftw.imag = _zscores(fftw.imag)
                        # fftw.real = _zscores(fftw.real)
                        fftw.imag = fftw.imag - np.mean(fftw.imag)
                        fftw.real = fftw.real - np.mean(fftw.real)
                        fftw = np.abs(fftw)
                        fcorrNoise = fnoise[n]*fftw
                        # fcorrNoise = fnoise_l*fftw
                        # fcorrNoise = fftw
                        # phase = np.angle(fnoise[n])
                        # ampl = np.ones(fnoise[n].shape)
                        # fnoise_ones = ampl*np.exp(1j*phase)
                        # fcorrNoise = fnoise_ones*fftw
                        corrNoise = pyfftw.interfaces.numpy_fft.ifft2(fcorrNoise)
                        corrNoiseReal = np.array(corrNoise.real)
                        # plt.clf()
                        # plt.imshow(corrNoiseReal,vmin=-3,vmax=3)
                        # autocorr,_ = st.compute_autocorrelation_fft(corrNoiseReal)
                        # plt.contour(autocorr,levels=[0.5,0.6,0.7,0.8,0.9],colors='k')
                        # plt.colorbar()
                        # plt.pause(1)
                        fcorrNoiseTotal[:,:,n] = fcorrNoiseTotal[:,:,n] + corrNoiseReal.copy()*mask
                        maskSum[:,:,n] = maskSum[:,:,n] + mask

    # normalize the sum
    idx = maskSum>0
    fcorrNoiseTotal[idx] = fcorrNoiseTotal[idx]/maskSum[idx]
    fcorrNoiseTotal[idx] = _zscores(fcorrNoiseTotal[idx])
    fcorrNoiseTotal[~idx] = np.nan

    # if nmembers==1:
        # fcorrNoiseTotal = np.squeeze(fcorrNoiseTotal)
    
    toc = time.clock()
    totalTime = toc - tic 
    iterations = count
    return(fcorrNoiseTotal,totalTime,iterations)

def build2dWindow(winsize,wintype='hanning'):
    try:
        # asymmetric window
        
        # Build 1-D window for rows
        if wintype == "rectangular":
            beta = 0
            w1dr = np.kaiser(winsize[0], beta)
        elif wintype == "bartlett":
            w1dr = np.bartlett(winsize[0]) 
        elif wintype == "hanning":
            w1dr = np.hanning(winsize[0])
        elif wintype == "flat-hanning":
            T = winsize[0]/4
            W = winsize[0]/2
            B = np.linspace(-W,W,2*W)
            R = np.abs(B)-T
            R[R<0]=0.
            A = 0.5*(1.0 + np.cos(np.pi*R/T))
            A[np.abs(B)>(2*T)]=0.0
            w1dr = A
        elif wintype == "hamming":
            w1dr = np.hamming(winsize[0])
        elif wintype == "blackman":
            w1dr = np.blackman(winsize[0]) 
        elif wintype == "kaiser":
            beta = 14
            w1dr = np.kaiser(winsize[0], beta)
        else:
            print("Error: unknown window type.")
               
        # Build 1-D window for columns
        if wintype == "rectangular":
            beta = 0
            w1dc = np.kaiser(winsize[1], beta)
        elif wintype == "bartlett":
            w1dc = np.bartlett(winsize[1]) 
        elif wintype == "hanning":
            w1dc = np.hanning(winsize[1])
        elif wintype == "flat-hanning":
            T = winsize[1]/4
            W = winsize[1]/2
            B=np.linspace(-W,W,2*W)
            R = np.abs(B)-T
            R[R<0]=0.
            A = 0.5*(1.0 + np.cos(np.pi*R/T))
            A[np.abs(B)>(2*T)]=0.0
            w1dc = A   
        elif wintype == "hamming":
            w1dc = np.hamming(winsize[1])
        elif wintype == "blackman":
            w1dc = np.blackman(winsize[1]) 
        elif wintype == "kaiser":
            beta = 14
            w1dc = np.kaiser(winsize[1], beta)
        else:
            print("Error: unknown window type.")
            
        # Expand to 2-D
        wind = np.sqrt(np.outer(w1dr,w1dc))
        
    except TypeError:
        # symmetric window
        # Build 1-D window
        if wintype == "rectangular":
            beta = 0
            w1d = np.kaiser(winsize, beta)
        elif wintype == "bartlett":
            w1d = np.bartlett(winsize) 
        elif wintype == "hanning":
            w1d = np.hanning(winsize)
        elif wintype == "flat-hanning":
            T = winsize/4
            W = winsize/2
            B=np.linspace(-W,W,2*W)
            R = np.abs(B)-T
            R[R<0]=0.
            A = 0.5*(1.0 + np.cos(np.pi*R/T))
            A[np.abs(B)>(2*T)]=0.0
            w1d = A   
        elif wintype == "hamming":
            w1d = np.hamming(winsize)
        elif wintype == "blackman":
            w1d = np.blackman(winsize) 
        elif wintype == "kaiser":
            beta = 14
            w1d = np.kaiser(winsize, beta)
        else:
            print("Error: unknown window type.")

        # Expand to 2-D
        wind = np.sqrt(np.outer(w1d,w1d))
        
    # Set nans to zero
    if np.sum(np.isnan(wind))>0:
        wind[np.isnan(wind)]=np.min(wind[wind>0])

    return(wind)

def localAutocorrelation(rainfield_dBZ,winsize=100,wintype='hanning',percentileZero = 90):  
    #    define the shift of the window based on the overlap parameter
    overlap=0
    delta = int(winsize*(1 - overlap))
    delta = np.max((delta,1))
    
    # rain/no rain threshold
    norain = np.nanmin(rainfield_dBZ)
    rainfield_dBZ[rainfield_dBZ<=norain] = norain
    
    # get rid of nans
    rainfield_dBZ[np.isnan(rainfield_dBZ)] = norain
    
    # 
    if len(rainfield_dBZ.shape) == 2:
        rainfield_dBZ = rainfield_dBZ[:,:,np.newaxis]
    nmembers = rainfield_dBZ.shape[2]
    
    # loop through windows
    idxi = np.zeros((2,1),dtype=int); idxj = np.zeros((2,1),dtype=int)
    imageout = np.zeros((rainfield_dBZ.shape[0],rainfield_dBZ.shape[1]))
    simout = np.zeros((rainfield_dBZ.shape[0],rainfield_dBZ.shape[1]))*np.nan
    
    # loop through rows       

    for i in xrange(0,rainfield_dBZ.shape[0],delta):
        
        # loop through columns
        for j in xrange(0,rainfield_dBZ.shape[1],delta):
        
            idxi[0] = i
            idxi[1] = np.min((i + winsize, rainfield_dBZ.shape[0]))
            idxj[0] = j
            idxj[1] = np.min((j + winsize, rainfield_dBZ.shape[1])) 
            
            # build window
            wind = build2dWindow(((idxi[1]-idxi[0]),(idxj[1]-idxj[0])),wintype)            
    
            # at least half of the window within the frame
            if (wind.shape[0]*wind.shape[1])/winsize**2 > 0.7: 
                count = 0
                autocorr = np.zeros((rainfield_dBZ.shape[0],rainfield_dBZ.shape[1]))
                mask = np.zeros((rainfield_dBZ.shape[0],rainfield_dBZ.shape[1]))
                for n in range(nmembers):
                    # rainSub = rainfield_dBZ[idxi[0]:idxi[1],idxj[0]:idxj[1],n]*wind[0:(idxi[1]-idxi[0]),0:(idxj[1]-idxj[0])]
                    mask[idxi[0]:idxi[1],idxj[0]:idxj[1]] = wind 
                    rmask = rainfield_dBZ*mask
                    
                    # continue only if any precip within window
                    if  np.sum(rmask>norain)/wind.size > 0:
                        count+=1
                        
                        # fft noise
                        np.random.seed(42)
                        randValues = np.random.randn(rmask.shape[0],rmask.shape[1])
                        fftw = pyfftw.interfaces.numpy_fft.fft2(rmask)
                        fnoise = pyfftw.interfaces.numpy_fft.fft2(randValues)
                        fcorrNoise = fnoise*fftw
                        corrNoise = pyfftw.interfaces.numpy_fft.ifft2(fcorrNoise)
                        corrNoiseReal = np.array(corrNoise.real)
                        corrNoiseReal = _zscores(corrNoiseReal)
                        simout[idxi[0]:idxi[1],idxj[0]:idxj[1]] = corrNoiseReal[idxi[0]:idxi[1],idxj[0]:idxj[1],]*wind
                        
                        # autocorrelation
                        tmp,_ = st.compute_autocorrelation_fft(rmask)
                        if percentileZero>0:
                            percZero = np.nanpercentile(tmp[tmp > 0.01], percentileZero)
                            tmpShifted = tmp - percZero
                            tmpShifted[tmpShifted < 0] = 0.0
                            # Image segmentation to remove high autocorrelations/spectrum values at far ranges/high frequencies
                            tmpShifted_bin = np.uint8(tmpShifted > 0.01)
                            # Compute image segmentation
                            labelsImage = measure.label(tmpShifted_bin, background = 0)
                            # Get label of center of autocorrelation function (corr = 1.0)
                            labelCenter = labelsImage[labelsImage.shape[0]/2,labelsImage.shape[1]/2]
                            # Compute mask to keep only central polygon
                            mask = (labelsImage == labelCenter).astype(int)
                            # Apply mask and back shift the autocorrelation values
                            tmp = tmpShifted*mask + percZero
                            tmp[tmp<=percZero] = 0
                            tmp = tmp/percZero
                            
                        autocorr = autocorr + tmp
                if count>0:
                    autocorr = autocorr/count   
                # previm = imageout[idxi[0]:idxi[1],idxj[0]:idxj[1]]
                # previm[autocorr>0] = 0
                # imageout[idxi[0]:idxi[1],idxj[0]:idxj[1]] = previm + autocorr
                imageout[idxi[0]:idxi[1],idxj[0]:idxj[1]] = dt.extract_middle_domain(autocorr,(idxi[1]-idxi[0]),(idxj[1]-idxj[0]))    
    
    return(imageout,simout)
    
# Compute various scores on the precip field locally applying a sliding window 
def localScores(rainfield_dBZ,winsize=65,wintype='hanning',overlap=0.5,rainThreshold=-1,\
                doanim=0,doplot=0,saveplots=0,saveresults=0,outDir=""):
       
    # define the shift of the window based on the overlap parameter
    delta = int(winsize*(1 - overlap))
    delta = np.max((delta,1))
    
    # build window
    wind = build2dWindow(winsize,wintype)
    windsh = wind.shape       

    # filename
    filenameOut = outDir + 'ssft_' + wintype + '_' + str(winsize) + '_' + str(delta) 
    if (saveplots or saveresults) and (outDir != ""):
        cmd = 'mkdir -p ' + outDir
        os.system(cmd)

    # compute the windowed fft 
    count=0
    idxi = np.zeros((2,1),dtype=int); idxj = np.zeros((2,1),dtype=int)
    
    # initialise variables
    x=[];y=[];war=[];imf=[];
    eccentricity=[]; orientation=[];
    major_ax=[];minor_ax=[]
    
    # loop through rows       
    for i in xrange(0,rainfield_dBZ.shape[0],delta):
    
        # loop through columns
        for j in xrange(0,rainfield_dBZ.shape[1],delta):
            count += 1
            
            idxi[0] = i
            idxi[1] = np.min((i + windsh[0], rainfield_dBZ.shape[0]))
            idxj[0] = j
            idxj[1] = np.min((j + windsh[1], rainfield_dBZ.shape[1])) 
            
            if np.sum(wind[0:(idxi[1]-idxi[0]),0:(idxj[1]-idxj[0])])/np.sum(wind) == 1:      
              
                rainSub = rainfield_dBZ[idxi[0]:idxi[1],idxj[0]:idxj[1]]*wind[0:(idxi[1]-idxi[0]),0:(idxj[1]-idxj[0])]
                
                # window's mid point coordinates
                x.append(np.mean(idxj))
                y.append(np.mean(idxi))
                
                # war, imf
                if (rainThreshold>=0):
                    war.append(np.sum(rainSub > rainThreshold)/np.sum(~np.isnan(rainSub)))
                    imf.append(np.mean(rainSub[rainSub > rainThreshold]))
                else:
                    war.append(1)
                    imf.append(np.mean(rainSub))
                    
                if war[-1] > 0.05:
                    ## Anisotropy
                    # Compute autocorrelation
                    
                    autocorr,_ = st.compute_autocorrelation_fft(rainSub, FFTmod = 'NUMPY')
                    # plt.show(rainSub)
                    # Compute anisotropy from autocorrelation function
                    fftSizeSub = -1
                    percentileZero = 85
                    radius = -1
                    _, ecc, ori, _, _, eigvals, eigvecs, _,_ = st.compute_fft_anisotropy(autocorr, fftSizeSub, percentileZero, rotation=False, radius=radius, verbose=1)
                    # plt.title(ori)
                    # plt.pause(1)
                    eccentricity.append(ecc)
                    orientation.append(ori)
                    major_ax.append(np.sqrt(np.max(eigvals)))
                    minor_ax.append(np.sqrt(np.min(eigvals)))
                    
                else:
                    eccentricity.append(np.nan)
                    orientation.append(np.nan)
                    major_ax.append(np.nan)
                    minor_ax.append(np.nan)       
    
    return(x,y,war,imf,eccentricity,orientation,major_ax,minor_ax)
    
# Compute the radially averaged 1D power spectrum
def psd1D(rainfield_dBZ,wintype='flat-hanning'):

    # wind = build2dWindow(rainfield_dBZ.shape,wintype)

    # Compute FFT
    fprecipNoShift = np.fft.fft2(rainfield_dBZ) 

    # Shift frequencies
    fprecip = np.fft.fftshift(fprecipNoShift)

    # Compute 2D power spectrum
    psd2d = np.abs(fprecip)**2/(rainfield_dBZ.shape[0]*rainfield_dBZ.shape[1])
    # print(np.sum(psd2d.flatten()))
    # Compute 1D radially averaged power spectrum
    bin_size = 1
    nr_pixels, bin_centers, psd1d = radialprofile.azimuthalAverage(psd2d, binsize=bin_size, return_nr=True)
    fieldSize = rainfield_dBZ.shape
    minFieldSize = np.min(fieldSize)

    # Extract subset of spectrum
    validBins = (bin_centers < minFieldSize/2) # takes the minimum dimension of the image and divide it by two
    psd1d = psd1d[validBins]

    # Compute frequencies
    resKm = 1
    freq = fftpack.fftfreq(minFieldSize, d=float(resKm))
    freqAll = np.fft.fftshift(freq)

    # Select only positive frequencies
    freq = freqAll[len(psd1d):] 

    # Replace 0 frequency with NaN
    freq[freq==0] = np.nan
    
    return(psd1d,freq)

# Shift and scale
def shiftandscale(refimage,image,wintype='hanning',winsize=64,overlap=0.5, quantileCorrection=False):

    # to dos:
    # - maximum possible dBZ value at pixel level
    
    # rain/no rain threshold
    mindBZ = np.nanmin(refimage)
    norain = mindBZ

    # define the shift of the window based on the overlap parameter
    delta = int(winsize*(1 - overlap))
    delta = np.max((delta,1))
    
    # Initialise variables
    count=-1
    WAR=[]
    IMF=[]
    MM=[]
    SD=[]
    MSD=[]
    
    wpi=[];wpj=[]
    subimagemean=[]
    subimagestd=[]
    AlphaWindow=[]
    
    idxi = np.zeros((2,1),dtype=int); idxj = np.zeros((2,1),dtype=int)
    maskSum = np.zeros(image.shape)
    Alpha = np.zeros(image.shape) # shift threshold
    cmean = np.zeros(image.shape)
    cstd = np.zeros(image.shape) 
    rmean = np.zeros(image.shape)
    rstd = np.zeros(image.shape)
    
    # loop through rows       
    for i in xrange(0,refimage.shape[0],delta):
        
        # loop through columns
        for j in xrange(0,refimage.shape[1],delta):
            
            mask = np.zeros(refimage.shape)
            idxi[0] = i
            idxi[1] = np.min((i + winsize, refimage.shape[0]))
            idxj[0] = j
            idxj[1] = np.min((j + winsize, refimage.shape[1])) 
            
            # build window
            wind = build2dWindow(((idxi[1]-idxi[0]),(idxj[1]-idxj[0])),wintype) + 1e-6
            
            if (wind.shape[0]*wind.shape[1])/winsize**2 > 0.1: 
                count += 1
                mask[idxi[0]:idxi[1],idxj[0]:idxj[1]] = wind
                
                # save location of the window
                wpi.append(np.mean(idxi))
                wpj.append(np.mean(idxj))
                
                subrefimage = refimage[idxi[0]:idxi[1],idxj[0]:idxj[1]]
                subimage = image[idxi[0]:idxi[1],idxj[0]:idxj[1]]
                
                WAR.append(np.sum(subrefimage>norain)/(wind.shape[0]*wind.shape[1]))
                IMF.append(np.mean(subrefimage))
                SD.append(np.std(subrefimage))
                
                # if np.sum(subrefimage>norain)>0:
                if WAR[-1] > 0.01:
                
                    MM.append(np.mean(subrefimage[subrefimage>norain]))
                    MSD.append(np.std(subrefimage[subrefimage>norain]))
                    # iqr = np.subtract(*np.percentile(subrefimage[subrefimage>norain], [75, 25]))
                    # MSD.append(iqr)
                    
                    p = np.percentile(subimage,100*(1-WAR[-1]))     
                    idx = (subimage>=p)
                    subimagemean.append(np.mean(subimage[idx]))
                    subimagestd.append(np.std(subimage[idx]))  

                    zp = (p - subimagemean[-1])/subimagestd[-1]
                    zsubimage = (subimage - subimagemean[-1])/subimagestd[-1]
                    
                    scaledp = zp*MSD[-1] + MM[-1] 
                    scaledsubimage = zsubimage*MSD[-1] + MM[-1] 
                    AlphaWindow.append(scaledp)
                
                else:
                
                    MM.append(-5)
                    MSD.append(0)
                    
                    subimagemean.append(subimage.mean())
                    subimagestd.append(subimage.std())
                    
                    AlphaWindow.append(norain)
                
                #interpolate by window                
                cmean[idxi[0]:idxi[1],idxj[0]:idxj[1]] = cmean[idxi[0]:idxi[1],idxj[0]:idxj[1]] + \
                                                        subimagemean[-1]*wind
                cstd[idxi[0]:idxi[1],idxj[0]:idxj[1]] = cstd[idxi[0]:idxi[1],idxj[0]:idxj[1]] + \
                                                        subimagestd[-1]*wind                                        
                rmean[idxi[0]:idxi[1],idxj[0]:idxj[1]] = rmean[idxi[0]:idxi[1],idxj[0]:idxj[1]] + \
                                                        MM[-1]*wind
                rstd[idxi[0]:idxi[1],idxj[0]:idxj[1]] = rstd[idxi[0]:idxi[1],idxj[0]:idxj[1]] + \
                                                        MSD[-1]*wind
                Alpha[idxi[0]:idxi[1],idxj[0]:idxj[1]] = Alpha[idxi[0]:idxi[1],idxj[0]:idxj[1]] + \
                                                        AlphaWindow[-1]*wind

                maskSum[:,:] = maskSum[:,:] + mask  
    

    #interpolate by window                
    # normalize the sums
    idx = maskSum>0
    Alpha[idx] = Alpha[idx]/maskSum[idx]
    cmean[idx] = cmean[idx]/maskSum[idx]
    cstd[idx] = cstd[idx]/maskSum[idx]
    rmean[idx] = rmean[idx]/maskSum[idx]
    rstd[idx] = rstd[idx]/maskSum[idx]
    
    # smooth these fields
    sgm = 10
    Alpha = gaussian_filter(Alpha, sigma=sgm)
    cmean = gaussian_filter(cmean, sigma=sgm)
    cstd = gaussian_filter(cstd, sigma=sgm)
    rmean = gaussian_filter(rmean, sigma=sgm)
    rstd = gaussian_filter(rstd, sigma=sgm)
        
    # plt.subplot(3,2,1)
    # plt.imshow(Alpha)
    # plt.colorbar()
    # plt.subplot(3,2,2)
    # plt.imshow(cmean)
    # plt.colorbar()
    # plt.subplot(3,2,3)
    # plt.imshow(cstd)
    # plt.colorbar()
    # plt.subplot(3,2,4)
    # plt.imshow(rmean)
    # plt.colorbar()
    # plt.subplot(3,2,5)
    # plt.imshow(rstd)
    # plt.colorbar()
    # plt.show()

    # shift and scale the image
    idx = cstd>0
    imout = np.zeros(image.shape)
    imout[idx] = (image[idx] - cmean[idx])/cstd[idx]*rstd[idx] + rmean[idx]
    imout[~idx] = 0
    imout[imout<=Alpha] = 0 
    
    # plt.clf()
    # plt.subplot(121)
    # plt.imshow(rmean)
    # plt.subplot(122)
    # plt.imshow(rstd)
    # plt.show()
    
    if quantileCorrection==True:
        # imout = of.morphological_opening(imout, thr=norain, n=1)
        imout = quantile_transformation(imout,refimage)
        
    
    # adjust global IMF
    # refimageZeros = refimage.copy()
    # refimageZeros[refimageZeros<=norain] = 0
    # imout = imout*np.mean(refimageZeros)/np.mean(imout)
    # imout[imout<=norain] = norain  
    
    # filter image
    # image = gaussian_filter(image, sigma=1.5)
    
    # adjust global MM
    imout[imout>norain] = imout[imout>norain]*np.mean(refimage[refimage>norain])/np.mean(imout[imout>norain])
    

    return(imout)

# zscores 
def _zscores(nparray,zeros=True):
    if zeros==True:
        nparray = np.array(nparray)
        if len(nparray.shape)==3:
            normarray = np.zeros(nparray.shape)
            for n in xrange(nparray.shape[2]):
                normarray[:,:,n] = (nparray[:,:,n] - np.nanmean(nparray[:,:,n])) / np.nanstd(nparray[:,:,n])
        else:    
            normarray = (nparray - nparray.mean()) / nparray.std()
    else:
        nanarray = np.array(nparray).copy()
        nanarray[nanarray==0] = np.nan
        if len(nparray.shape)==3:
            normarray = np.zeros(nanarray.shape)
            for n in xrange(nanarray.shape[2]):
                normarray[:,:,n] = (nanarray[:,:,n] - np.nanmean(nanarray[:,:,n])) / np.nanstd(nanarray[:,:,n])
        else:    
            normarray = (nanarray - np.nanmean(nanarray)) / np.nanstd(nanarray)
        normarray[np.isnan(normarray)] = 0
    return(normarray)

# normalize btw 0 and 1
def _norm0and1(nparray):
    nparray = np.array(nparray)
    if len(nparray.shape)==3:
        normarray = np.zeros(nparray.shape)
        for n in xrange(nparray.shape[2]):
            normarray[:,:,n] = (nparray[:,:,n] - nparray[:,:,n].min()) / (nparray[:,:,n].max() - nparray[:,:,n].min())   
    else:
        normarray = (nparray - nparray.min()) / (nparray.max() - nparray.min())
    return(normarray)
    
def _exp_filter(domainSize,freq,beta=-2,angle=[]):
    
    fx,fy = np.meshgrid(freq,freq)
    
    freq2d = np.sqrt(fx**2+fy**2)
    freq0 = 1/domainSize
    filter = (freq2d/freq0)**(beta/2)
    filter[np.isinf(filter)] = 1
    
    return(filter)

def generalized_scale_invariance_model(domainSize,c=-0.2,e=-0.2,f=0.2,Is=12,d=1.0,beta=-2.5,doplot=False):
    # Niemi, Kokkonen, Seed (2014)
    # c and f are associated with stratification c,f = [-1, 1], d^2 > c^2 + f^2
    # e with rotation; e = [-1.5, 1.5]
    # d with overall contraction of the system; d = 1
    # fully isotropic: c = f = 0
    
    # fully isotropic field
    if c == 0 and f == 0:
        
        # get frequencies
        res = 1 
        freqNoshift = fftpack.fftfreq(domainSize, d=res)
        freq = np.fft.fftshift(freqNoshift)
        fx,fy = np.meshgrid(freq,freq)

        k_lambda = np.sqrt(fx**2 + fy**2)
    
    # anisotropic field
    else:  
        if d**2 - c**2 - f**2 <= 0:
            print('You must satisfy the condition d^2 > c^2 + f^2')
            sys.exit()
            
        asquared = c**2 + f**2 - e**2
        if asquared >0:
            print('a^2 = %1.3f  > 0 : the stratification of the system is dominant.' % asquared)
        elif asquared <0:
            print('a^2 = %1.3f  < 0 : the rotation of the system is dominant.' % asquared)
        else:
            print('a^2 = %1.3f' % asquared)
            
        # get frequencies for half the field only
        res = 1 
        freqNoshift = fftpack.fftfreq(domainSize, d=res)
        freq = np.fft.fftshift(freqNoshift)
        fx,fy = np.meshgrid(freq[domainSize/2::],freq)
        
        # solve the equation only for half of the field
        # for each pixel
        k_lambda = np.zeros(fx.size)
        for n in xrange(fx.size):
            dt.update_progress((n+1)/fx.size)
            
            # compute all the equation terms
            a = np.sqrt(np.abs(asquared))
            kx = fx.flatten()[n]
            ky = fy.flatten()[n]
            Q = kx**2 + ky**2
            R = ( kx**2*(c**2 + (f - e)**2) + ky**2*(c**2 + (f + e)**2) + 4*kx*ky*c*e )/asquared
            S = ( (kx**2 - ky**2)*c + 2*kx*ky*f )/a
            U1 = np.log(Is)
            
            # non-linear solver
            if n==0:
                guess = 0
            U,_ = leastsq(_F, guess, args = (Q,a,R,S,U1))
            k_lambda[n] = np.exp(U)
            guess = U
        # k_lambda = np.round(k_lambda)
        k_lambda = np.reshape(k_lambda,fx.shape)
            
        # duplicate and flip the second half of k_lambda
        k_lambda = np.column_stack((np.flipud(np.fliplr(k_lambda)),k_lambda))
        
        if doplot:
            plt.imshow(k_lambda)
            plt.colorbar()
            plt.show()
        
    k_lambdaNoshift = np.fft.fftshift(k_lambda)
    
    # compute anisotropic filter
    filter = (k_lambdaNoshift)**(beta/2)
    filter[np.isinf(filter)] = 1
    
    return k_lambdaNoshift,filter
    
def _F(U,Q,a,R,S,U1):
    """ Objective function """
    return np.log( Q*(np.cosh(a*(U - U1)))**2 + R*(np.sinh(a*(U - U1)))**2 - S*np.sinh(2*a*(U - U1)) ) - 2*U
    
def _autocovariance_model(domainSize,range_x, range_y, angle,model='exponential',doplot=False):
   
    h = np.arange(1,domainSize/2+1)
    reversed_array = h[::-1]
    h = np.concatenate((h,reversed_array))
       
    # reduced distance, h1
    hx,hy = np.meshgrid(h,h)
    
    
    if model=='gaussian':
        # gaussian
        h1 = np.sqrt(hx**2/range_x**2 + hy**2/range_y**2)
        autocovariance2D = np.exp(-(h1)**2)
    elif model=='exponential':
        # exponential
        h1 = np.sqrt(hx**2/3**2/range_x**2 + hy**2/3**2/range_y**2)
        autocovariance2D = np.exp(-h1)
    
    
    autocovariance2DShifted = np.fft.fftshift(autocovariance2D)
    if angle > 0: 
        # rotation
        autocovariance2DShiftedRotated = rotate(autocovariance2DShifted,angle,reshape=False)
        
        # shift back
        autocovariance2DRotated = np.fft.fftshift(autocovariance2DShiftedRotated)
        autocovariance2D = autocovariance2DRotated
        
        if doplot:
            plt.subplot(2,2,1)
            plt.imshow(autocovariance2D)
            plt.colorbar()
            plt.subplot(2,2,2)
            plt.imshow(autocovariance2DShifted)
            plt.colorbar()
            plt.subplot(2,2,3)
            plt.imshow(autocovariance2DShiftedRotated)
            plt.colorbar()
            plt.subplot(2,2,4)
            plt.imshow(autocovariance2DRotated)
            plt.colorbar()
            plt.show()
    else:
        if doplot:
            plt.subplot(1,2,1)
            plt.imshow(autocovariance2D)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(autocovariance2DShifted)
            plt.colorbar()
            plt.show()
        
    return(autocovariance2D)

def exponential_anisotropic_model(h, range_x, range_y, angle):
    
    # size of squared field
    domainShape = (np.sqrt(h.size/2),np.sqrt(h.size/2))

    # reduced distance, h1
    hx = h[:,0]
    hy = h[:,1]
    h1 = np.sqrt(hx**2/range_x**2 + hy**2/range_y**2)
    
    exponential = np.exp(-h1)
    exponential = rotate(exponential.reshape(domainShape),angle,reshape=False).flatten()
    
    if angle < 0:
        exponential = exponential*9999

    
    return exponential
    
def levy(alpha,size):
# Author : Auguste GIRES (2013)      (auguste.gires@leesu.enpc.fr)
    from scipy.stats import uniform, expon
    phi=uniform.rvs(loc=-sp.pi/2,scale=sp.pi)
    W=expon.rvs(size=size[0]*size[1]*size[2])
    
    if alpha!=1:
        phi0=(sp.pi/2)*(1-np.abs(1-alpha))/alpha;
        L=sp.sign(1-alpha)*(sp.sin(alpha*(phi-phi0)))*(((sp.cos(phi-alpha*(phi-phi0)))/W)**((1-alpha)/alpha))/((sp.cos(phi))**(1/alpha));
    else:
        print('Error : alpha = '+str(alpha)+' in Levy')
    L = np.reshape(L,(size[0],size[1],size[2])) 
    return L
    
def quantile_transformation(initialarray,targetarray):

    # zeros in initial image
    idxZeros = initialarray == 0

    # flatten the arrays
    arrayshape = initialarray.shape
    target = targetarray.flatten()
    array = initialarray.flatten()
    
    # rank target values
    order = target.argsort()
    ranked = target[order]

    # rank initial values order
    orderin = array.argsort()
    ranks = np.empty(len(array), int)
    ranks[orderin] = np.arange(len(array))

    # get ranked values from target and rearrange with inital order
    outputarray = ranked[ranks]

    # reshape as 2D array
    outputarray = outputarray.reshape(arrayshape)
    
    # reassing original zeros
    outputarray[idxZeros] = 0

    return outputarray
    
def local_quantile_transform(refimage,image,wintype='hanning',winsize=64,overlap=0.9):

    # rain/no rain threshold
    mindBZ = np.nanmin(refimage)
    norain = mindBZ
    
    # define the shift of the window based on the overlap parameter
    delta = int(winsize*(1 - overlap))
    delta = np.max((delta,1))
    
    # Initialise variables
    count=-1
    idxi = np.zeros((2,1),dtype=int); idxj = np.zeros((2,1),dtype=int)
    maskSum = np.zeros(image.shape)
    transf = np.zeros(image.shape) 

    
    # loop through rows       
    for i in xrange(0,refimage.shape[0],delta):
        
        # loop through columns
        for j in xrange(0,refimage.shape[1],delta):
            
            mask = np.zeros(refimage.shape)
            idxi[0] = i
            idxi[1] = np.min((i + winsize, refimage.shape[0]))
            idxj[0] = j
            idxj[1] = np.min((j + winsize, refimage.shape[1])) 
            
            # build window
            wind = build2dWindow(((idxi[1]-idxi[0]),(idxj[1]-idxj[0])),wintype) + 1e-6
            
            if (wind.shape[0]*wind.shape[1])/winsize**2 > 0.1: 
                count += 1
                mask[idxi[0]:idxi[1],idxj[0]:idxj[1]] = wind
                
                subrefimage = refimage[idxi[0]:idxi[1],idxj[0]:idxj[1]]
                subimage = image[idxi[0]:idxi[1],idxj[0]:idxj[1]]
                subtransf = quantile_transformation(subimage,subrefimage)
                
                #interpolate by window                
                transf[idxi[0]:idxi[1],idxj[0]:idxj[1]] = transf[idxi[0]:idxi[1],idxj[0]:idxj[1]] + subtransf*wind
                maskSum[:,:] = maskSum[:,:] + mask  

    # normalize the sums
    idx = maskSum>0
    transf[idx] = transf[idx]/maskSum[idx]
       
    return(transf)

###
def logistic(x,L = 1,k = 1,x0 = 0):
    return L/(1 + np.exp(-k*(x - x0)))  

def psd2dtopsd1d(psd2d):

    # Shift frequencies
    fprecip = np.fft.fftshift(psd2d)
    # Compute 2D power spectrum
    psd2d = np.abs(fprecip)**2/(psd2d.shape[0]*psd2d.shape[1])
    # Compute 1D radially averaged power spectrum
    bin_size = 1
    nr_pixels, bin_centers, psd1d = radialprofile.azimuthalAverage(psd2d, binsize=bin_size, return_nr=True)
    fieldSize = psd2d.shape
    minFieldSize = np.min(fieldSize)
    # Extract subset of spectrum
    validBins = (bin_centers < minFieldSize/2) # takes the minimum dimension of the image and divide it by two
    psd1d = psd1d[validBins]
    # Compute frequencies
    resKm = 1
    freq = fftpack.fftfreq(minFieldSize, d=float(resKm))
    freqAll = np.fft.fftshift(freq)
    # Select only positive frequencies
    freq = freqAll[len(psd1d):] 
    # Replace 0 frequency with NaN
    freq[freq==0] = np.nan
    # transform to dB 
    psd1d_dB = 10*np.log10(psd1d)
    freq_dB = 10*np.log10(freq)
    # center it
    # psd1d_dB = psd1d_dB - psd1d_dB.mean() 
    return psd1d_dB,freq

def fourier_filtering(whiteNoise,filter):
    fnoise = np.fft.fft2(whiteNoise)
    corrNoise = filter*fnoise
    corrNoise = np.fft.ifft2(corrNoise)
    corrNoiseReal = np.array(corrNoise.real)
    return corrNoiseReal

def split_field(Size,Segments):
    Delta = int(Size/Segments)
    Idxi = np.zeros((Segments**2,2)); Idxj = np.zeros((Segments**2,2))    
    count=-1
    for i in xrange(0,Size,Delta):
        for j in xrange(0,Size,Delta):
            count+=1
            if count>=Segments**2:
                break
            Idxi[count,0] = i
            Idxi[count,1] = np.min((i + Delta, Size))
            Idxj[count,0] = j
            Idxj[count,1] = np.min((j + Delta, Size)) 
    # print(np.column_stack((Idxi,Idxj)))
    return Idxi,Idxj  

def get_mask(Size,idxi,idxj):
    winsize = idxi[1] - idxi[0]
    w1d = np.hanning(winsize)
    wind = np.sqrt(np.outer(w1d,w1d))
    wind[np.isnan(wind)]=np.min(wind[wind>0])
    mask = np.zeros((Size,Size)) 
    mask[idxi[0]:idxi[1],idxj[0]:idxj[1]] = wind
    return mask

def recursive_split(idxi,idxj,Segments=2):
    Delta = int((idxi[1] - idxi[0])/Segments)
    Idxi = np.zeros((Segments**2,2),dtype=int); Idxj = np.zeros((Segments**2,2),dtype=int)    
    count=-1
    for i in xrange(int(idxi[0]),int(idxi[1]),Delta):
        for j in xrange(int(idxj[0]),int(idxj[1]),Delta):
            count+=1
            if count>=Segments**2:
                break
            Idxi[count,0] = i
            Idxi[count,1] = np.min((i + Delta, 512))
            Idxj[count,0] = j
            Idxj[count,1] = np.min((j + Delta, 512)) 
            
    # print(np.column_stack((Idxi,Idxj)))
    return Idxi,Idxj

def fourier_analysis(fieldin):
    fftw = np.fft.fft2(fieldin)
    # normalize the spectrum
    fftw.imag = _zscores(fftw.imag)
    # fftw.imag = fftw.imag - np.mean(fftw.imag)
    # fftw.imag = fftw.imag/np.std(fftw.imag)
    fftw.real = _zscores(fftw.real)
    # fftw.real = fftw.real - np.mean(fftw.real)
    # fftw.real = fftw.real/np.std(fftw.real)
    # keep only the amplitude
    fftw = np.abs(fftw)

    return fftw   
    
def nested_fft2(target, nr_frames = 10, max_level = 3, win_type = 'flat-hanning', war_thr = 0.1, overlap = 40, do_set_seed = True, do_plot = False, seed = 42):

	#Produces a 2-dimensional correlated noise
	#Use the last observation as filter
    #Nested implementation to account for non-stationarities
	#Example:
    #Created:
    #ned, October 2017
    
    # make sure non-rainy pixels are set to zero
    min_value = np.min(target)
    orig_target = target
    target -= min_value
    
    # store original field size
    orig_dim = target.shape
    orig_dim_x = orig_dim[1]
    orig_dim_y = orig_dim[0]
    
    
    # apply window to the image to limit spurious edge effects
    orig_window = build2dWindow(orig_dim,win_type)
    target = target*orig_window
    
    # now buffer the field with zeros to get a squared domain       <-- need this at the moment for the nested approach, but I guess we could try to avoid it
    dim_x = np.max(orig_dim) 
    dim_y = dim_x
    dim = (dim_y,dim_x)
    ztmp = np.zeros(dim)
    if(orig_dim[1] > dim_x):
        idx_buffer = round((dim_x - orig_dim_x)/2)
        ztmp[:,idx_buffer:(idx_buffer + orig_dim_x)] = z
        z=ztmp 
    elif(orig_dim[0] > dim_y):
        idx_buffer = round((dim_y - orig_dim_y)/2)
        ztmp[idx_buffer:(idx_buffer + orig_dim_y),:] = z
        z=ztmp 
    # else do nothing
    
    ## Nested algorithm
    
    # prepare indices
    Idxi = np.array([[0,dim_y]])
    Idxj = np.array([[0,dim_x]])
    Idxipsd = np.array([[0,2**max_level]])
    Idxjpsd = np.array([[0,2**max_level]])
    
    # generate the FFT sample frequencies
    res_km = 1 
    freq = fftpack.fftfreq(dim_x, res_km)
    fx,fy = np.meshgrid(freq,freq)
    freq_grid = np.sqrt(fx**2 + fy**2)
    
    # get global fourier filter
    mfilter0 = get_fourier_filter(target)
    # and allocate it to the final grid
    mfilter = np.zeros((2**max_level,2**max_level,mfilter0.shape[0],mfilter0.shape[1]))
    mfilter += mfilter0[np.newaxis,np.newaxis,:,:]
    
    # now loop levels and build composite spectra
    level=0 
    while level < max_level:

        for m in xrange(len(Idxi)):
        
            # the indices of rainfall field
            Idxinext,Idxjnext = split_field(Idxi[m,:],Idxj[m,:],2)
            # the indices of the field of fourier filters
            Idxipsdnext,Idxjpsdnext = split_field(Idxipsd[m,:],Idxjpsd[m,:],2)
            
            for n in xrange(len(Idxinext)):
                mask = get_mask(dim[0],Idxinext[n,:],Idxjnext[n,:],win_type)
                war = np.sum((target*mask)>0)/(Idxinext[n,1]-Idxinext[n,0])**2 
                
                if war>war_thr:
                    # the new filter 
                    newfilter = get_fourier_filter(target*mask)
                    
                    # compute logistic function to define weights as function of frequency
                    # k controls the shape of the weighting function
                    merge_weights = logistic_function(1/freq_grid, k=0.05, x0 = (Idxinext[n,1] - Idxinext[n,0])/2)
                    newfilter = newfilter*(1 - merge_weights)
                    
                    # perform the weighted average of previous and new fourier filters
                    mfilter[Idxipsdnext[n,0]:Idxipsdnext[n,1],Idxjpsdnext[n,0]:Idxjpsdnext[n,1],:,:] *= merge_weights[np.newaxis,np.newaxis,:,:]
                    mfilter[Idxipsdnext[n,0]:Idxipsdnext[n,1],Idxjpsdnext[n,0]:Idxjpsdnext[n,1],:,:] += newfilter[np.newaxis,np.newaxis,:,:] 
                    
        # update indices
        level += 1
        Idxi, Idxj = split_field((0,dim[0]),(0,dim[1]),2**level)
        Idxipsd, Idxjpsd = split_field((0,2**max_level),(0,2**max_level),2**level)
        
    ## Power-filter images

	# produce normal noise array
    if do_set_seed: 
        np.random.seed(seed)
	white_noise = np.random.randn(dim[0],dim[1],nr_frames)
    
    # build composite image of correlated noise
    corr_noise = np.zeros((dim_y,dim_x,nr_frames))
    sum_of_masks = np.zeros((dim_y,dim_x,nr_frames))
    idxi = np.zeros((2,1),dtype=int)
    idxj = np.zeros((2,1),dtype=int)
    winsize = np.round( dim[0]  / 2**max_level )
    
    # loop frames
    for m in xrange(nr_frames):
    
        # get fourier spectrum of white noise field
        white_noise_ft = np.fft.fft2(white_noise[:,:,m])
    
        # loop rows
        for i in xrange(2**max_level):
            # loop columns
            for j in xrange(2**max_level):

                # apply fourier filtering with local filter
                this_filter = mfilter[i,j,:,:]
                this_corr_noise_ft = white_noise_ft * this_filter
                this_corr_noise = np.fft.ifft2(this_corr_noise_ft)
                this_corr_noise = np.array(this_corr_noise.real)
                
                # compute indices of local area
                idxi[0] = np.max( (np.round(i*winsize - overlap/2), 0) )
                idxi[1] = np.min( (np.round(idxi[0] + winsize  + overlap/2), dim[0]) )
                idxj[0] = np.max( (np.round(j*winsize - overlap/2), 0) )
                idxj[1] = np.min( (np.round(idxj[0] + winsize  + overlap/2), dim[1]) )
                
                # build mask and add local noise field to the composite image
                mask = get_mask(dim[0],idxi,idxj,win_type)
                corr_noise[:,:,m] += this_corr_noise*mask
                sum_of_masks[:,:,m] += mask
                
    # normalize the sum
    idx = sum_of_masks > 0
    corr_noise[idx] = corr_noise[idx]/sum_of_masks[idx]
    
    # crop the image back to the original size
    difx = dim_x - orig_dim_x
    dify = dim_y - orig_dim_y
    output = corr_noise[int(dify/2):int(dim_y-dify/2),int(difx/2):int(dim_x-difx/2),:]
    
    # standardize the results to N(0,1)
    for m in xrange(nr_frames):
        output[:,:,m]  -= np.mean(output[:,:,m])
        output[:,:,m]  /= np.std(output[:,:,m])
    
    if do_plot:
        for m in xrange(nr_frames):
            plt.clf()
            plt.subplot(121)
            plt.imshow(target,interpolation='nearest')
            plt.subplot(122)
            plt.imshow(output[:,:,m],interpolation='nearest',vmin=-3.5,vmax=3.5)
            plt.pause(1)
   
    return output    
 
def get_fourier_filter(fieldin, do_norm = True):

    # FFT of the field
    fftw = np.fft.fft2(fieldin)
    
    # Normalize the real and imaginary parts
    if do_norm:
        fftw.imag = ( fftw.imag - np.mean(fftw.imag) ) / np.std(fftw.imag)
        fftw.real = ( fftw.real - np.mean(fftw.real) ) / np.std(fftw.real)
        
    # Extract the amplitude
    fftw = np.abs(fftw)

    return fftw  
    
def split_field(idxi,idxj,Segments):

    sizei = (idxi[1] - idxi[0]) 
    sizej = (idxj[1] - idxj[0]) 
    
    winsizei = np.round( sizei / Segments )
    winsizej = np.round( sizej / Segments )
    
    Idxi = np.zeros((Segments**2,2))
    Idxj = np.zeros((Segments**2,2))
    
    count=-1
    for i in xrange(Segments):
        for j in xrange(Segments):
            count+=1
            Idxi[count,0] = idxi[0] + i*winsizei
            Idxi[count,1] = np.min( (Idxi[count,0] + winsizei, idxi[1]) )
            Idxj[count,0] = idxj[0] + j*winsizej
            Idxj[count,1] = min( (Idxj[count,0] + winsizej, idxj[1]) )

    Idxi = np.array(Idxi).astype(int); Idxj =  np.array(Idxj).astype(int)        
    return Idxi, Idxj
    
def get_mask(Size,idxi,idxj,wintype):
    idxi = np.array(idxi).astype(int); idxj =  np.array(idxj).astype(int)
    winsize = (idxi[1] - idxi[0] , idxj[1] - idxj[0])
    wind = build2dWindow(winsize,wintype)
    mask = np.zeros((Size,Size)) 
    mask[int(idxi[0]):int(idxi[1]),int(idxj[0]):int(idxj[1])] = wind
    return mask
    
def logistic_function(x, L = 1,k = 1,x0 = 0):
    return L/(1 + np.exp(-k*(x - x0)))