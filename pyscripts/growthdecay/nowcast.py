#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import os
import sys

import matplotlib as mpl
mpl.use('Agg')

import numpy as np                       
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from scipy import fftpack
from scipy.linalg import inv,pinv
from scipy.spatial.distance import cdist
from scipy.stats import norm

from numpy import dot, zeros, eye, outer
from numpy.random import multivariate_normal

import time_tools_attractor as ti
import stat_tools_attractor as st
import data_tools_attractor as dt
import io_tools_attractor as io
import ssft
import nowcasting as nw

import cv2

import load_forecasts as lf

class Nowcasting:
    '''
    '''

    def __init__(self, data, N, hx, fx, rx, phi,   \
                 AR_order=2, number_levels=8, transformation='dBR', probability_matching=True, \
                 resolution_km=1, label='EnKF', min_rainrate=0.01, \
                 wet_thr=0.5, zero_padding = 0):    
 
        """ Create a Kalman filter.
        Parameters
        ----------
        x : np.array(dim_y,dim_x)
            state mean
        N : int
            number of sigma points (ensembles). Must be greater than 1.
        hx : function hx(x)
            Measurement function. Converts state x into a measurement. 
        fx : function fx(x)
            State transition function. May be linear or nonlinear. Projects
            state x into the next time period. Returns the projected state x.
        rx : function rx(N)
            Random noise function. 
        phi : float 
            Parameters of the AR process.
        """
        
        # lag-n radar images
        xlag_rainrates_fields = data.n_last_obs(AR_order)[::-1,:,:] # xlag_rainrates_fields[0,:,:] is last image
        xlag_rainrates_fields[xlag_rainrates_fields<min_rainrate] = 0
        
        # dimensions
        self.dim_yx =  xlag_rainrates_fields[0,:,:].shape
        self.dim_n = xlag_rainrates_fields[0,:,:].shape[0]*xlag_rainrates_fields[0,:,:].shape[1]
        self.dim_t = data.obs.shape[0]
        self.N = N
        self.idxMember = np.arange(N)
        self.resolution_km = resolution_km
        self.nlags = AR_order
        
        # various operators
        self.hx = hx
        self.fx = fx
        self.rx = rx
        self.phi = phi
        self.pca = []
        self.wet_thr = wet_thr 
        
        # other parameters
        self.min_rainrate = min_rainrate
        self.minDBZ = 10.0*np.log10(316*min_rainrate**1.5)
        self.minDBR = 10.0*np.log10(min_rainrate)
        self.number_levels = number_levels
        self.transformation = transformation
        self.probability_matching = probability_matching
        self.label = label
        self.startStr = data.startStr
        
        # init counters
        self.countupdates = 0
        self.countpredicts = 0
        
        # initialize radar mask 
        #(True is a valid radar pixel, False is no data)
        mask = data.mask.astype(float).copy() 
        mask[mask>=0.5] = 1; mask[mask<0.5] = 0
        self.mask = np.repeat(mask[None,:,:].astype(bool), self.N, axis=0) # (N, dim_y, dim_x)
        
        # band pass filters
        fftshape = tuple(x+zero_padding*2 for x in self.dim_yx) 
        bandpassFilter2D,centreWaveLengths = nw.calculate_bandpass_filter(fftshape, self.number_levels, Width = 2.0, resKm = self.resolution_km, doplot=True)
        self.bandpassFilter2D = bandpassFilter2D
        self.centreWaveLengths = centreWaveLengths
        self.zero_padding = zero_padding

        # perturbations for motion field
        # self.motion_pert = np.abs(np.random.normal(loc=1.0, scale=0.1, size=self.N)) 
        cov = np.array([[ 0.05  ,  0.049],
                        [ 0.049,   0.05  ]])
        self.motion_pert = np.random.multivariate_normal(np.array([1,1]),cov,self.N)
        self.motion_pert[0] = [1,1] # member 0 is control
		
        # verification scores
        self.spreadskill = np.zeros((self.dim_t+1,2,2,int(self.dim_yx[0]/2)))*np.nan
       
        self.initialize(xlag_rainrates_fields)
            
        # store initial field
        self.x_field_t0 = self.xlag_fields[0,0,:,:].copy()
        
        print('________________________________________________________________________')
        print('Kalman filter initialized:')
        print('label = ', self.label, ', N = ',self.N,', AR(', self.nlags, '), levels = ', self.number_levels, ', transformation = ', self.transformation,', probability_matching = ',self.probability_matching,', minR = ',self.min_rainrate,' minDBZ = ', self.minDBZ)

    def initialize(self, xlag_rainrates_fields):
        """ Initializes the filter with the given state. 
        Parameters
        ----------
            
        """
        assert xlag_rainrates_fields[0,:,:].ndim == 2 # is it a field?
        
        N = self.N
        dim_yx = self.dim_yx
        dim_y = self.dim_yx[0]
        dim_x = self.dim_yx[1]
        dim_n = self.dim_n
        nlags = self.nlags
        
        # advect lags > 0 to same t0 position
        for lag in xrange(1,nlags):
            xlag_rainrates_fields[lag,:,:] = self.fx(xlag_rainrates_fields[lag,:,:],[1,1],lag)                  
               
        # keep only lag0 rainrate fields and replicate N times
        self.x_rainrates_fields = np.zeros((N, dim_y, dim_x))
        self.x_rainrates_matrix = np.zeros((N,dim_n))
        self.x_rainrates_fields = np.repeat(xlag_rainrates_fields[0,:,:][None, :, :].copy(), self.N, axis=0) # (N, dim_y, dim_x)
        self.x_rainrates_matrix = self.x_rainrates_fields.reshape((N,-1)) # (N, dim_n)
        
        # transform 
        if not self.transformation==False:
            xlag_fields = self.transform(xlag_rainrates_fields)
        
        # replicate N times
        self.xlag_fields = np.zeros((nlags, N, dim_y, dim_x))
        self.xlag_matrix = np.zeros((nlags, N, dim_n))
        for lag in xrange(nlags):
            self.xlag_fields[lag,:,:,:] = np.repeat(xlag_fields[lag,:,:][None, :, :], self.N, axis=0) # (N, dim_y, dim_x)
            self.xlag_matrix[lag,:,:] = self.xlag_fields[lag,:,:,:].reshape((N,-1)) # (N, dim_n)
            
    def predict(self, growthdecay=None, net=1):
        ''' Predict next position with an AR process. '''
        
        # update counter
        self.countpredicts += 1
        
        # fetch global variables
        N = self.N
        dim_yx = self.dim_yx
        dim_y = self.dim_yx[0]
        dim_x = self.dim_yx[1]
        nlags = self.nlags
        number_levels = self.number_levels
        bandpassFilter2D = self.bandpassFilter2D 
        centreWaveLengths = self.centreWaveLengths
        zero_padding = self.zero_padding
        phi = self.phi
        min_rainrate = self.min_rainrate
        minDBZ = self.minDBZ
        
        # prepare perturbations
        noiseFields = self.rx(dim_yx,N)
        
        for i in xrange(N):
            
            # extract given field 
            x = self.xlag_fields[:,i,:,:].copy()                        # transformed (e.g. dBZ)
            x_rainrates = self.x_rainrates_fields[i,:,:].copy()         # rain rates
            mask = self.mask[i,:,:].copy().astype(float)                # radar mask
            
            # apply extrapolation(lag0 is most recent image)
            motion_pert = self.motion_pert[i]
            for lag in xrange(nlags):
                x[lag,:,:] = self.fx(x[lag,:,:],motion_pert,net)                   
            x_rainrates = self.fx(x_rainrates,motion_pert,net)
            
            # advect radar mask too
            mask = self.fx(mask,motion_pert,net)
            mask[mask>=0.5] = 1; mask[mask<0.5] = 0
            mask = mask.astype(bool)
            
            # build a precipitation mask
            n=20
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))   
            precipmask = (x_rainrates > 0.3 ).astype('uint8')
            precipmask = cv2.dilate(precipmask,kernel).astype(bool)
            
            # cascade decomposition of rainfall field
            cascade, cascadeMean, cascadeStd = nw.get_cascade_from_array(x - minDBZ, number_levels, bandpassFilter2D, centreWaveLengths, zeroPadding = zero_padding, zerothr = None, squeeze=False)
            
            # cascade decomposition of noise field
            noisecascade, _, _ = nw.get_cascade_from_array(noiseFields[i], number_levels, bandpassFilter2D, centreWaveLengths, zeroPadding = zero_padding)
            
            # add noise level by level
            xn = np.zeros(dim_yx)
            for l in xrange(number_levels):
                
                # fetch the AR(1) parameter for this cascade level
                phi_level = phi[l,:].copy()      
                # derive the noise variance (Yule-Walker eq)
                noise_variance = ( (1 + phi_level[1]) * (1 + phi_level[0] - phi_level[1])*(1 - phi_level[0] - phi_level[1]) ) / ( 1 - phi_level[1])
                
                if number_levels>1:
                    # noise_correction = 1/( .75 + 0.09*l )  # original equation
                    noise_correction = 1
                else:
                    noise_correction = 1  
                    
                # AR(nlags)
                xn_l = np.zeros(dim_yx)
                for lag in xrange(nlags):
                    xn_l += phi_level[lag]*cascade[lag,:,:,l]
                xn += ( xn_l + noise_correction*np.sqrt(noise_variance)*noisecascade[:,:,l] )*cascadeStd[0,l] + cascadeMean[0,l]
                
            # growth and decay
            if (growthdecay is not None and
                growthdecay.shape[0] == self.dim_yx[0] and
                growthdecay.shape[1] == self.dim_yx[1]):   
                    
                xn += growthdecay
                
            xn += minDBZ    
                                   
            # update stack
            for lag in xrange(nlags-1,0,-1):
                x[lag,:,:] = x[lag-1,:,:]
            x[0,:,:] = xn   
            
            # store this version in the global stack
            self.xlag_fields[:,i,:,:] = x.copy()
                
            # then apply the masks
            # x[~mask] = x.min()
            xn[~precipmask] = xn.min()         
                
            # probability matching
            if self.probability_matching and (self.transformation=='dBZ' or self.transformation=='dBR'):
                x_rainrates = self.do_probability_matching(xn.ravel(),x_rainrates.ravel()).reshape(dim_yx)
                
            elif self.probability_matching and not self.transformation:
                x_rainrates = self.do_probability_matching(xn.ravel(),x_rainrates.ravel()).reshape(dim_yx)   
            elif not self.probability_matching and (self.transformation=='dBZ' or self.transformation=='dBR'):
                x_rainrates = self.toRainrates(xn.copy())
            else:
                print('Warning: did not fully consider this combination of options (predict -> probability matching)')
                x_rainrates = xn
               
            # back to global stacks
            x_rainrates[x_rainrates<=min_rainrate] = 0
            self.x_rainrates_fields[i,:,:] = x_rainrates.copy()  
            self.mask[i,:,:] = mask.copy().astype(bool) 
            
        self.xlag_matrix = self.xlag_fields.reshape((nlags,N,-1))    
        self.x_rainrates_matrix = self.x_rainrates_fields.reshape((N,-1))    
        
    def update(self, z_rainrates_fields, tapering=0, pairing=False, doplot=False):
        """
        Add a new measurement (z_rainrates_fields) to the kalman filter. If z_rainrates_fields is None, nothing
        is changed.
        Parameters
        ----------
        z_rainrates_fields : np.array(N,dim_y,dim_y)
            measurement ensemble for this update in rain rates [mm h-1].
        """

        if z_rainrates_fields is None:
            return
            
        assert z_rainrates_fields.ndim == 3 # (N,dim_y,dim_x)
        assert z_rainrates_fields.shape[0] == self.N
        assert z_rainrates_fields.shape[1] == self.dim_yx[0]
        assert z_rainrates_fields.shape[2] == self.dim_yx[1]
        
        # update counter
        self.countupdates += 1
        
        # fetch global variables
        N = self.N
        dim_yx = self.dim_yx
        dim_y = self.dim_yx[0]
        dim_x = self.dim_yx[1]
        dim_n = self.dim_n
        min_rainrate = self.min_rainrate
        minDBZ = self.minDBZ
        
        # extra parameters
        wet_thr = self.wet_thr # [0,1] threshold for proportion of wet members per pixel
        inflation_factor_p = 1.0 # radar
        inflation_factor_r = 1.0 # cosmo
        offset_p = 0
        offset_r = 0
        
        # apply mask to z
        z_rainrates_fields[z_rainrates_fields<min_rainrate] = 0
        mask_fields = self.mask
        # z_rainrates_fields[~mask_fields] = 0
        
        # get a matrix view on z
        z_rainrates = z_rainrates_fields.reshape((N,-1)) # (N, dim_n)
        mask = mask_fields.reshape((N,-1))
        
        # extract state data
        x = self.xlag_matrix[0,:,:].copy() # (N, dim_n) only lag0 is used for the update
        x_ref = x.copy()
        x_rainrates = self.x_rainrates_matrix.copy()
        x_rainrates[x_rainrates<min_rainrate] = 0
        
        # let's have a look at the zero rainfall values in both datasets
        print('zero rainfall value = %.3f (%.3f,%.3f)' % (np.min((np.min(x_rainrates),np.min(z_rainrates))),np.min(x_rainrates),np.min(z_rainrates)))
        
        # let's have a look at the min rainfall values in both datasets
        minRx = np.min(x_rainrates[x_rainrates>0])
        minRz = np.min(z_rainrates[z_rainrates>0])
        print('min rainfall value = %.3f (%.3f,%.3f)' % (np.min((minRx,minRz)),minRx,minRz)) 
        
        # transform z
        if not self.transformation==False:
            z = self.transform(z_rainrates.copy())
        else:
            z = z_rainrates
           
        thrzero = np.min((np.min(x),np.min(z)))
        if not self.transformation==False:
            # let's have a look at the zero transformed rainfall values in both datasets
            print('zero transformed rainfall value = %.3f (%.3f,%.3f)' % (thrzero,np.min(x),np.min(z)))

        # compute proportion of wet members for each pixel
        x_porportion_of_wet = np.sum(x!=minDBZ,axis=0) / N
        z_porportion_of_wet = np.sum(z!=minDBZ,axis=0) / N
        idxWet = np.logical_and(x_porportion_of_wet >= wet_thr, z_porportion_of_wet >= wet_thr) 
        # plt.close()
        # plt.imshow(idxWet.reshape(dim_yx))
        # plt.savefig('tmp/idxwet%i.png'%self.countupdates)
             
        # build data matrix 
        x_all = np.concatenate((x, z), axis=0)

        # transform all points into new space (PCA)
        x_all_h, model = self.hx(x_all)
        x_h = x_all_h[:N,:]
        z_h = x_all_h[N:,:]
        self.pca = model
        n_components = x_h.shape[1]
        
        # only with wet pixels
        if wet_thr>0:
            h_components = model.components_.copy()
            x_all_h_wet = ( x_all[:,idxWet] - model.mean_[idxWet] ).dot( h_components[:,idxWet].T ) 
            if model.whiten:
                x_all_h_wet /= np.sqrt(model.explained_variance_)
            x_h_wet = x_all_h_wet[:N,:]
            z_h_wet = x_all_h_wet[N:,:]
        else:
            x_h_wet = x_h
            z_h_wet = z_h
            
        # trasnform-back to estimate reconstruction residuals
        x_reconstr = model.inverse_transform(x_h)
        z_reconstr = model.inverse_transform(z_h)
        
        # tapering function
        if tapering is None:
            tpring = np.ones((n_components,n_components)) 
        else:
            tpring = np.identity(n_components)
            hanvalues = np.hanning(tapering*2 + 1)[(tapering+1):]
            for d in xrange(tapering):   
                tpring += np.diag(np.ones(n_components-d-1)*hanvalues[d],k=d+1)
                tpring += np.diag(np.ones(n_components-d-1)*hanvalues[d],k=-d-1)            

        # compute error covariance matrices
        kmethod = 0
        if kmethod==0:
        
            P = 0 # radar
            x_h_wet_mean = x_h_wet.mean(axis=0)
            for i in xrange(N):
                x_e = ( x_h_wet[i,:] - x_h_wet_mean ) * inflation_factor_p
                P += outer(x_e, x_e)
            P = ( P / (N-1) + offset_p ) * tpring
            P[P<1e-3] = 0
            
            R = 0 # cosmo
            z_h_wet_mean = z_h_wet.mean(axis=0)
            for i in xrange(N):
                z_e = ( z_h_wet[i,:] - z_h_wet_mean ) * inflation_factor_r
                R += outer(z_e, z_e)
            R = ( R / (N-1) + offset_r ) * tpring
            R[R<1e-3] = 0
            
            # Kalman gain
            K_h = dot(P, pinv(P + R))
            K_h[K_h<1e-3] = 0
            print('K_h shape = ',K_h.shape)
            
        if kmethod==1:
            x_h_wet_mean = x_h_wet.mean(axis=0)
            z_h_wet_mean = z_h_wet.mean(axis=0)
            xz_h_wet_mean = (x_h_wet - z_h_wet).mean(axis=0)
            C1 = 0
            C2 = 0
            for i in xrange(N):
                x_e = x_h_wet[i,:] - x_h_wet_mean
                xz_e = x_h_wet[i,:] - z_h_wet[i,:] - xz_h_wet_mean
                C1 += outer(x_e, xz_e)
                C2 += outer(xz_e, xz_e)
            C1 = C1 / (N - 1)
            C2 = C2 / (N - 1)
            C1 *= tpring
            C2 *= tpring
            
            # Kalman gain
            K_h = dot(C1, pinv(C2))
            K_h[K_h<1e-3] = 0
            print('K_h shape = ',K_h.shape)
        
        # pair members that are close
        if pairing:
            dist = cdist(x_h,z_h)
            idxMember = np.zeros(x_h.shape[0],dtype=int)
            for i in xrange(z_h.shape[0]):
                rmin,cmin = np.unravel_index(np.nanargmin(dist),dist.shape)
                idxMember[rmin] = cmin
                dist[rmin,:] = np.nan
                dist[:,cmin] = np.nan
            self.idxMember = idxMember
            print(idxMember)
            z_h = z_h[idxMember,:] 
        else:
            self.idxMember = np.arange(N)
         
        #
        x_h_prior = x_h.copy()
        prior_rainrates = x_rainrates.copy()
        
        # update   
        for i in range(N):
            x_h[i,:] += dot(K_h, z_h[i,:] - x_h[i,:])
            
        # trasform the analysis back to original (transformed) space
        x = model.inverse_transform(x_h)
        
        # compute a global K
        if self.probability_matching:
            K1 = (x.sum(axis=0) - x_reconstr.sum(axis=0)) 
            K2 = (z_reconstr.sum(axis=0) - x_reconstr.sum(axis=0))
            K = K1 / K2
            K[K>1] = 1
            K[K<0] = 0
            Kglob = np.nanmean(K)
            print( 'K = %.3f' % Kglob)
                  
        # store this version before thresholding
        self.xlag_matrix[0,:,:] = x.copy()
            
        # than apply the radar mask
        # x[~mask] = x.min()    
            
        # probability matching
        if self.probability_matching:
            for i in xrange(N):
                merged_array = self.resample_distributions(x_rainrates[i,:],z_rainrates[self.idxMember[i],:], 1-Kglob)
                x_rainrates[i,:] = self.do_probability_matching(x[i,:],merged_array)
            x_rainrates[x_rainrates<min_rainrate] = 0
        elif not self.probability_matching and (self.transformation=='dBZ' or self.transformation=='dBR'):
            x_rainrates = self.toRainrates(x.copy())
            x_rainrates[x_rainrates<min_rainrate] = 0
        else:
            print('Warning: did not fully consider this combination of options (predict -> probability matching)')
            x_rainrates = x    
            x_rainrates[x_rainrates<min_rainrate] = 0
            
        # back to global stacks
        self.x_rainrates_matrix = x_rainrates.copy()  
        self.xlag_fields[0,:,:,:] = self.xlag_matrix[0,:,:].reshape((N,dim_y,dim_x)) # only needed for lag0 fields
        self.x_rainrates_fields = self.x_rainrates_matrix.reshape((N,dim_y,dim_x))         
            
    def transform(self, arrayin, doplot=False):
        if self.transformation == 'dBZ':
            return self.todBZ(arrayin)
        elif self.transformation == 'dBR':
            return self.todBR(arrayin)
        elif self.transformation == 'gaussian':
            zeros='auto'
            return self.toGaussian(arrayin, zeros=zeros, doplot=doplot)
        return arrayin
      
    def toGaussian(self, arrayin, zeros='auto', doplot=False):
        '''
        Using Gaussian anamorphosis to transform any continously distributed variable into a standard Gaussian distribution.
        Lien et al. 2013
        '''
 
        # flatten the arrays
        arrayshape = arrayin.shape
        array = arrayin.flatten()
        
        # zeros in initial image
        zero_value = np.min(array)
        idxzeros = array == zero_value
        probability_of_zero_precipitation = np.sum(idxzeros)/array.size
        
        if doplot:
            plt.close()
            # best fit of data
            (mu, sigma) = norm.fit(array)
            # the histogram of the data
            plt.subplot(221)
            n, bins, patches = plt.hist(array, 100, normed=1, facecolor='blue', alpha=0.75)
            y = norm.pdf( bins, mu, sigma)
            plt.plot(bins, y, 'r--', linewidth=1)
            plt.xlabel('y [mm/h]')
            plt.ylabel('PDF of y')
            plt.ylim([0,np.max(y)*1.2])
            plt.grid(True)
            plt.title('Original distribution')
            
            plt.subplot(222)
            plt.imshow(array.reshape(arrayshape)[0,0,:,:],interpolation='none')
            plt.xticks([]);plt.yticks([])
            plt.colorbar()

        # target gaussian distribution
        target = np.random.normal(0,1,array.size)

        # rank target values
        order = target.argsort()
        ranked = target[order]

        # rank initial values order
        orderin = array.argsort()
        ranks = np.empty(len(array), int)
        ranks[orderin] = np.arange(len(array))

        # get ranked values from target and rearrange with inital order
        outputarray = ranked[ranks]
        
        # dealing with zeros
        if zeros == 'naive':
            '''Only transform the non-zero part of precipitation data.'''
            outputarray[idxzeros] = zero_value
        elif zeros == 'auto': 
            '''Assign the middle value of zero-precip cumulative probability to F(0).'''
            print('zero-precip cumulative probabilty = %.3f' % probability_of_zero_precipitation)
            outputarray[idxzeros] = np.median(outputarray[idxzeros])
            
        # reshape as original array
        outputarray = outputarray.reshape(arrayshape)
        
        if doplot:
            # best fit of data
            (mu, sigma) = norm.fit(outputarray.flatten())
            # the histogram of the data
            plt.subplot(223)
            n, bins, patches = plt.hist(outputarray.flatten(), 100, normed=1, facecolor='blue', alpha=0.75)
            y = norm.pdf( bins, mu, sigma)
            plt.plot(bins, y, 'r--', linewidth=1)
            plt.xlabel(r'y$_{transf}$ [$\sigma$]')
            plt.ylabel(r'PDF of y$_{transf}$')
            plt.ylim([0,np.max(y)*1.2])
            plt.grid(True)     
            plt.title('Transformed\n(' + zeros_approach + ')')
                
            plt.subplot(224)
            plt.imshow(outputarray[0,0,:,:],interpolation='none')
            plt.xticks([]);plt.yticks([])
            plt.colorbar()
            
            plt.tight_layout()
            figname = 'gaussian_transformation_' + self.transformation + '.png'
            plt.savefig(figname)
            print('Saved: ' + figname)
            
        return outputarray    
         
    def todBZ(self, arrayin):
        zeros = self.minDBZ
        arrayout,minDBZ,_ = dt.rainrate2reflectivity(arrayin, 316.0, 1.5, zeros)
        # arrayout,minDBZ,_ = dt.rainrate2reflectivity(arrayin, 316.0, 1.5, 0.0)
        return arrayout
        
    def todBR(self, arrayin):
        idxwet = arrayin>0
        arrayout = arrayin.copy()
        arrayout[idxwet] = dt.to_dB(arrayin[idxwet])
        arrayout[~idxwet] = self.minDBR
        arrayout -= self.minDBR
        return arrayout    
        
    def toRainrates(self, arrayin):
        if self.transformation == 'dBZ':
            zeros = self.minDBZ
            arrayout = dt.reflectivity2rainrate(arrayin, zeros, 316.0, 1.5)
        elif self.transformation == 'dBR':
            arrayout = dt.from_dB(arrayin + self.minDBR)
            arrayout[arrayout<self.minDBR] = 0
        return arrayout

    def resample_distributions(self,a,b,weight):
        '''merge two distributions'''
        
        assert a.size == b.size
        
        asort = np.sort(a.flatten())[::-1]
        bsort = np.sort(b.flatten())[::-1]
        n = asort.shape[0]
        
        # resample
        # print(weight)
        idxsamples = np.random.binomial(1,weight,n).astype(bool)
        #print('%.3f' % (np.sum(idxsamples)/n))
        csort = bsort.copy()
        csort[idxsamples] = asort[idxsamples]
        # print(np.concatenate((asort[:,None],bsort[:,None],csort[:,None]),axis=1))
        csort = np.sort(csort)[::-1]
        
        return csort 
        
    def do_probability_matching(self,initialarray,targetarray):
        
        assert initialarray.size == targetarray.size
        
        # zeros in initial image
        zvalue = initialarray.min()
        idxzeros = initialarray == zvalue

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

        # reshape as original array
        outputarray = outputarray.reshape(arrayshape)
        
        # readding original zeros
        outputarray[idxzeros] = zvalue

        return outputarray
 
    @property
    def getnowcast(self):
        return np.array(self.x_rainrates_matrix.copy().reshape((self.N,self.dim_yx[0],self.dim_yx[1])))
        
    @property
    def getmask(self):
        mask = np.array(self.mask.copy()).astype(float)
        mask[mask==1] = np.nan
        mask[mask>=0] = 1
        return mask
 
 
class Datamanager:
    """ ... 
    """

    def __init__(self, startStr =  '201706272200', endStr = '201706280400', upscaleKm = 1, min_rainrate = 0.01):
    
        startTime = ti.timestring2datetime(startStr)
        endTime = ti.timestring2datetime(endStr)
        
        self.startStr = startStr
        self.endStr = endStr
        self.startTime = startTime
        self.endTime = endTime
        self.upscaleKm = upscaleKm
        self.min_rainrate = min_rainrate
        self._DATA = {}
        
        self.load_observation()
        
    def load_observation(self,product='AQC'):
        # silence numpy's invalid warnings 
        np.seterr(invalid='ignore') 
        
        # from start time to end time
        observations,_,timestamps = lf.produce_radar_observation_with_accumulation(self.startStr,self.endStr,newAccumulationMin=10,product=product,rainThreshold=self.min_rainrate)
        mask = np.ones((observations.shape[1],observations.shape[2]),int)
        mask[np.all(np.isnan(observations),axis=0)] = 0
        observations[np.isnan(observations)] = 0
        observations = observations[:,np.newaxis,:,:]    

        # upscaling
        observations = self.upscale_wavelets(observations,self.upscaleKm)
        observations[observations<self.min_rainrate] = 0
        mask = self.upscale_wavelets(mask[np.newaxis,np.newaxis,:,:],self.upscaleKm)
        mask[mask<0.5] = 0
        mask[mask>=0.5] = 1

        # also retrieve few time steps before start time (just in case)
        startTime_before = self.startTime - timedelta(hours=1)
        startStr_before = ti.datetime2timestring(startTime_before)
        observations_before,_,timestamps_before = lf.produce_radar_observation_with_accumulation(startStr_before,self.startStr,newAccumulationMin=10,product=product,rainThreshold=self.min_rainrate)
        mask_before = np.ones((observations_before.shape[1],observations_before.shape[2]),int)
        mask_before[np.all(np.isnan(observations_before),axis=0)] = 0
        observations_before[np.isnan(observations_before)] = 0
        observations_before = observations_before[:,np.newaxis,:,:]    
        
        # upscaling
        observations_before = self.upscale_wavelets(observations_before,self.upscaleKm)
        observations_before[observations_before<self.min_rainrate] = 0
        mask_before = self.upscale_wavelets(mask_before[np.newaxis,np.newaxis,:,:],self.upscaleKm)
        mask_before[mask_before<0.5] = 0
        mask_before[mask_before>=0.5] = 1

        
        self._DATA[product] =  observations
        self._DATA[product + '_before'] =  observations_before
        self._observations_name = product
        self._DATA['mask'] = mask.squeeze()
        self._DATA['mask_before'] = mask_before.squeeze()
        self.timestamps = timestamps
        self.timestamps_before = timestamps_before
        
    def load_forecast(self, product, N=0):
        # silence numpy's invalid warnings 
        np.seterr(invalid='ignore') 
        if product=='COSMO':
            # COSMO ensemble
            membersCE = 'all'
            forecasts,_,_,_ = lf.get_lagged_cosmo1(self.startStr,self.endStr,rainThreshold=self.min_rainrate)
            if forecasts.shape[1] < N:
                forecasts1, _, _, _ = lf.get_cosmoE10min(self.startStr,self.endStr,members=membersCE,lag=0,rainThreshold=self.min_rainrate)
                forecasts = np.concatenate((forecasts,forecasts1),axis=1)
                del forecasts1
                if forecasts.shape[1] < N:
                    forecasts1, _, _, _ = lf.get_cosmoE10min(self.startStr,self.endStr, members=membersCE,lag=1,rainThreshold=self.min_rainrate)
                    forecasts = np.concatenate((forecasts,forecasts1),axis=1)
                    del forecasts1
                    if forecasts.shape[1] < N:
                        print('Max number of COSMO member was reached (%i).' % forecasts.shape[1])
                        N = forecasts.shape[1]
            forecasts[np.isnan(forecasts)] = 0
            forecasts = forecasts[:,:N,:,:]
        elif product=='COSMO-E':
            forecasts, _, _, _ = lf.get_cosmoE10min(self.startStr,self.endStr,members='all',lag=0,rainThreshold=self.min_rainrate)
            if forecasts.shape[1] < N:
                forecasts1, _, _, _ = lf.get_cosmoE10min(self.startStr,self.endStr,members='all',lag=1,rainThreshold=self.min_rainrate)
                forecasts = np.concatenate((forecasts,forecasts1),axis=1)
                del forecasts1
                if forecasts.shape[1] < N:
                    print('Max number of COSMO-E member was reached (%i).' % forecasts.shape[1])
                    N = forecasts.shape[1]
            forecasts[np.isnan(forecasts)] = 0
            forecasts = forecasts[:,:N,:,:]
            product = 'COSMO'
        elif product=='lagrangian':
            if N==0:
                forecasts, _, _, _ = lf.get_radar_extrapolation(self.startStr,self.endStr,newAccumulationMin=10,product='AQC',rainThreshold=self.min_rainrate)
                forecasts[np.isnan(forecasts)] = 0
                forecasts = forecasts[:,np.newaxis,:,:]
            else:
                print('still need to include the probabilistic radar extrapolation code here.')
                sys.exit()
        
        forecasts = self.upscale_wavelets(forecasts,self.upscaleKm)
        forecasts[forecasts<self.min_rainrate] = 0
                
        self._DATA[product] =  forecasts    
    
    def upscale_wavelets(self, fieldsin, upscaleKm, resolution_km = 1, wavelet = 'haar'):
        '''
        '''
        
        if upscaleKm==resolution_km:
            return fieldsin

        dim_t = fieldsin.shape[0]
        N = fieldsin.shape[1]
        
        for t in xrange(dim_t):
            for i in xrange(N):

                fieldin = fieldsin[t,i,:,:].copy()
                
                # wavelets decomposition
                wavelet_coeff = st.wavelet_decomposition_2d(fieldin, wavelet, nrLevels = None)

                # Generate coordinates of centers of wavelet coefficients
                extent = (0, fieldin.shape[1], 0, fieldin.shape[0])
                xvecs, yvecs = st.generate_wavelet_coordinates(wavelet_coeff, fieldin.shape, extent[0], extent[1], extent[2], extent[3], resolution_km)

                # Append a given wavelet scale to write out into daily netCDF files
                scale2keep = st.get_level_from_scale(resolution_km, upscaleKm)

                scaleKm = int(xvecs[scale2keep][1] - xvecs[scale2keep][0])
                if upscaleKm != scaleKm:
                    print('Asked and returned wavelet scales not matching.', upscaleKm, 'vs', scaleKm)
                    sys.exit()
                
                fieldout = wavelet_coeff[scale2keep]

                if (i==0) and (t==0):
                    fieldsout = np.zeros((dim_t,N,fieldout.shape[0],fieldout.shape[1]))
                fieldsout[t,i,:,:] = fieldout
                    
        return fieldsout    

    def n_last_obs(self,n):
        # max is 1 hour before
        return np.array(self._DATA[self._observations_name + '_before'][-n:,0,:,:].copy()) 
        
    @property
    def obs(self):  
        return np.array(self._DATA[self._observations_name].copy())
    @property
    def mask(self):  
        return np.array(self._DATA['mask'].copy(),dtype=float) 
    @property    
    def cosmo(self):
        return np.array(self._DATA['COSMO'].copy())  
    @property    
    def lagrangian(self):
        return np.array(self._DATA['lagrangian'].copy())

