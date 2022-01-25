#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 00:49:15 2022

@author: gabrielasalazar

FOR OPTIMIZING NEURON CHANNEL BINARIZER

"""

#%%
import skimage.io
import os
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask, threshold_otsu, threshold_triangle, try_all_threshold, threshold_yen
import numpy as np 

#%%
def try_all_threshold_on_set(dataDir, resultsDir, channel):
    '''runs scikit-image's try_all threshold function on all test images

    -----PARAMETERS-----
    dataDir: filepath to directory containing test data
        3D arrays representing 2D images in the form (pixel_row, pixel_column, channel)
    resultsDir: filepath to directory for storing results. 
    channel: integer indicating channel to be tested
    
    ----RETURNS----
    For each test image, produces a figure showing denoised grayscale image
    and results from global thresholding algorithms. 
    Figures are stored in resultsDir directory.
    '''

    #create results directory
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
    #for each image
    for root, directory, file in os.walk(dataDir):
        for f in file:
            if '.tif' in f:
                ipath=os.path.join(root, f)
                #load image with skimage
                fullimg=skimage.io.imread(ipath, plugin="tifffile")
                #isolate channel
                original=fullimg[:, :, channel]                
                #denoise image
                unsharp_img = unsharp_mask(original, radius=20, amount=2)
                #try global thresholds
                try_all_threshold(unsharp_img, figsize=(4, 6), verbose=False)
                #save results
                imgID = f.replace('.tif', '')
                figpath=os.path.join(resultsDir, imgID + '_THRESHOLD' + '.png')
                plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
                plt.close()
    return ('DONE')

def astrocyte_binarizer_optimization (dataDir, resultsDir, channel):
    '''For troubleshooting astrocyte channel binarization. 
    Selects optimal thresholdingn algorithm from otsu, triangle and yen methods

    ----PARAMETERS----
    dataDir: filepath to directory containing test data
        3D arrays representing 2D images in the form (pixel_row, pixel_column, channel)
    resultsDir: filepath to directory for storing results. 
    channel: integer indicating astrocyte channel

    ----RETURNS----
    For each test image, produces a figure showing the results of each thresholding algorithm
    and indicating which algorithm would be selected as optimal.
    Notes the values for the two metrics used for selecting optimal algorithm:
        proportion of true pixels in triangle subsection
        proportion true pixels in otsu subsection
    Figures are stored in resultsDir directory.
    '''
    #create results directory
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
    for root, directory, file in os.walk(dataDir):
        for f in file[:]:
            if '.tif' in f:
                ipath=os.path.join(root, f)
                #load image with skimage
                fullimg=skimage.io.imread(ipath, plugin="tifffile")
                #select single channel
                original=fullimg[:, :, channel]
                
                #denoise image using unsharp mask 
                #radius = 20, amount = 4 worked best for us
                #can be optimized using Step 1
                unsharp_img = unsharp_mask(original, radius=20, amount=2)

                #apply three thresholds options
                #triangle (I refer to image thresholded by triangle algorithm as triangle image)
                #find threshold
                thresh = threshold_triangle(unsharp_img)
                #apply threshold to denoised image to produce binary image
                triangle = unsharp_img > thresh
                #otsu (I refer to image thresholded by otsu algorithm as otsu image)
                thresh = threshold_otsu(unsharp_img)
                otsu = unsharp_img > thresh
                #yen (I refer to image thresholded by yen algorithm as yen image)
                thresh=threshold_yen(unsharp_img)
                yen = unsharp_img > thresh
                
                #calculate metrics to decide between algorithms
                #to decide if triangle is blown out (overestimating true signal)
                #grab subsection of triangle image
                tri_sub=triangle[900:1200, :]
                #calculate number of true pixels in the subsection
                tri_overblown=np.sum(tri_sub.astype(float))
                tri_overblown=tri_overblown/(np.shape(tri_sub)[0]*np.shape(tri_sub)[1])
            
                #check if otsu is blown out (overestimating true signal)
                #grab subsection that tends to get blown out
                o_sub=otsu[800:1200, :]    
                #calculate number of true pixels in the subsection
                o_overblown=np.sum(o_sub.astype(float))
                o_overblown=o_overblown/(np.shape(o_sub)[0]*np.shape(o_sub)[1])
                
                #choose thresholding algorithm
                #if otsu is not overblown choose otsu
                if o_overblown<0.3:
                        good = otsu
                        gthresh='otsu'
                        bad1 = triangle
                        bthresh1='triangle'
                        bad2 = yen
                        bthresh2 = 'yen'
                #if otsu is overblown
                else:
                    #choose triangle if not overblown
                    if tri_overblown<0.4:
                        good = triangle
                        gthresh='triangle'
                        bad1 = otsu
                        bthresh1='otsu'
                        bad2 = yen
                        bthresh2 = 'yen'
                    #choose yen if triangle is overblown
                    else:
                        good = yen
                        gthresh='yen'
                        bad1 = otsu
                        bthresh1='otsu'
                        bad2 = triangle
                        bthresh2 = 'triangle'            
                #create figure showing result of otsu, yen, triangle
                #and labeling the optimal algorithm
                #create figure
                fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
                ax = axes.ravel()
                plt.gray()
                #original image
                ax[0].imshow(original)
                ax[0].set_title(f + '\n Original')
                #denoised image
                ax[1].imshow(unsharp_img)
                ax[1].set_title('Denoised')
                #original otsu image without despeckling
                ax[2].imshow(otsu)
                ax[2].set_title('Otsu')
                #image thresholded by optimal algorithm
                ax[3].imshow(good)
                #note the percent of true pixels in otsu subsection 
                #note the percent of true pixels in triangle subsection
                ax[3].set_title('Picked: ' + gthresh + '   (' +str(round(o_overblown, 3)) 
                                + ', ' + str(round(tri_overblown, 3)) + ')' ) 
                #image thresholded by suboptimal algorithm #1 
                ax[4].imshow(bad1)
                ax[4].set_title(bthresh1)
                #image thresholded by suboptimal algorithm #2
                ax[5].imshow(bad2)
                ax[5].set_title(bthresh2)
                #take axis off
                for a in ax:
                    a.axis('off')
                #save image in results folder
                imgID = f.replace('.tif', '')
                figpath=os.path.join(resultsDir, imgID + 'AsroTHRESH' + '.png')
                plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
                #suppress output showing every figure
                plt.close()
    return('DONE')

#%%# RUN TRY_ALL_THRESHOLD_ON_SET ON ALL IMAGES IN TEST SET

dataDir='TestImages'
resultsDir = 'AstroThresh'
channel = 0

test = try_all_threshold_on_set (dataDir, resultsDir, channel)
print(test)

#%%# RUN ASTRO_BINARIZER_OPTIMIZATION ON ALL IMAGES IN TEST SET

dataDir='TestImages'
resultsDir = 'AstroBinarizerRes'
channel = 0

test = astrocyte_binarizer_optimization (dataDir, resultsDir, channel)
print(test)