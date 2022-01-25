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
from skimage.filters import unsharp_mask, threshold_otsu, threshold_triangle
import numpy as np 
from skimage.morphology import remove_small_objects

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

def cortexglia_binarizer_optimization (dataDir, resultsDir, channel):
    '''For troubleshooting cortex glia channel binarization. 
    Selects optimal thresholding algorithm from otsu, triangle methods

    ----PARAMETERS----
    dataDir: filepath to directory containing test data
        3D arrays representing 2D images in the form (pixel_row, pixel_column, channel)
    resultsDir: filepath to directory for storing results. 
    channel: integer indicating cortex glia channel

    ----RETURNS----
    For each test image, produces a figure showing the results of each thresholding algorithm
    and indicating which algorithm would be selected as optimal.
    Notes the values for the two metrics used for selecting optimal algorithm:
        proportion of true pixels in triangle subsection
        proportion true pixels in otsu subsection
        proportion of pixels removed from otsu image by despeckling
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
                # select cortex glia channel
                original=fullimg[:, :, channel]
                #denoise image using unsharp mask 
                #radius = 20, amount = 2 worked best for us
                #can be optimized using Step 1
                unsharp_img = unsharp_mask(original, radius=20, amount=2)
            
                #apply two threshold options
                #triangle (I refer to image thresholded by triangle algorithm as triangle image)
                #find threshold
                thresh = threshold_triangle(unsharp_img)
                #apply threshold to denoised image to produce binary image
                triangle = unsharp_img > thresh
                #otsu (I refer to image thresholded by otsu algorithm as otsu image)
                thresh = threshold_otsu(unsharp_img)
                otsu = unsharp_img > thresh
                
                #calculate metrics to decide between triangle and otsu
                #to decide if triangle is blown out (overestimating true signal)
                #grab subsection of triangle image
                tri_sub=triangle[900:1200, 400:800]
                #calculate number of true pixels in the subsection
                tri_overblown=np.sum(tri_sub.astype(float))
                tri_overblown=tri_overblown/(np.shape(tri_sub)[0]*np.shape(tri_sub)[1])
                
                #to decide if otsu is blown out (overestimating true signal)
                #grab subsection of original otsu
                o_sub=otsu[:, 400:800]
                #calculate number of true pixels in subsection
                o_overblown=np.sum(o_sub.astype(float))/(np.shape(o_sub)[0]*np.shape(o_sub)[1])
                ##remove specks (objects < some_area) from binary image
                ##most of these specks are noise
                otsu_desp = remove_small_objects(otsu, 75)
                #grab subsection of despeckled image (where most of the noise is)
                od_sub=otsu_desp[:, 400:800]
                #if despeckling helps "a lot", Otsu is the wrong filter
                #calculate the percent of pixels removed by despeckling in subsection 
                o_overblown2=np.sum(o_sub.astype(float)-od_sub.astype(float))
                o_overblown2=o_overblown2/(np.shape(o_sub)[0]*np.shape(o_sub)[1])
            
                #choose a thresholding algorithm
                #if neither algorithm is clearly overblown (areas < cutoffs)
                if (o_overblown<0.5 and o_overblown2 < 0.05 and tri_overblown<0.5):
                    #choose the one that yields the greatest number of true pixels in subsections
                    if o_overblown>tri_overblown:
                        good = otsu
                        gthresh='otsu'
                        bad1 = triangle
                        bthresh1='triangle'
                    else:
                        good = triangle
                        gthresh='triangle'
                        bad1 = otsu
                        bthresh1='otsu'
                #if triangle above certain threshold
                else:
                    #pick otsu if not overblown
                    if (o_overblown<0.5 and o_overblown2< 0.05):
                        good = otsu
                        gthresh='otsu'
                        bad1 = triangle
                        bthresh1='triangle'
                    #pick triangle if otsu is overblown
                    else:
                        good = triangle
                        gthresh='triangle'
                        bad1 = otsu
                        bthresh1='otsu'           
                #create figure showing result of otsu, triangle
                #and labeling the optimal algorithm
                #create figure
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
                ax = axes.ravel()
                plt.gray()
                #original image
                ax[0].imshow(original)
                ax[0].set_title(f + '\n Original')
                #denoised image
                ax[1].imshow(unsharp_img)
                ax[1].set_title('Denoised')
                #image thresholded by optimal algorithm
                ax[2].imshow(good)
                #note the percent of true pixels in triangle subsection
                #note the percent of true pixels in otsu subsection 
                #note the percent of true pixels in removed by despeckling in otsu subsection 
                ax[2].set_title('Picked: ' + gthresh + '   (' +str(round(tri_overblown, 3)) 
                                + ', ' + str(round(o_overblown, 3)) 
                                + ', ' + str(round(o_overblown2, 3))
                                + ')' ) 
                #image thresholded by suboptimal algorithm #1 
                ax[3].imshow(bad1)
                ax[3].set_title(bthresh1)
                #image thresholded by suboptimal algorithm #2
                #take axis off
                for a in ax:
                    a.axis('off')
                #save image in results folder
                imgID = f.replace('.tif', '')
                figpath=os.path.join(resultsDir, imgID + 'CortexGliaTHRESH' + '.png')
                plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
                #suppress output showing every figure
                plt.close()
    return('DONE')


#%%# RUN TRY_ALL_THRESHOLD_ON_SET ON ALL IMAGES IN TEST SET

dataDir='TestImages'
resultsDir = 'CortexGliaThresh'
channel = 1

test = try_all_threshold_on_set (dataDir, resultsDir, channel)
print(test)

#%%# RUN ASTRO_BINARIZER_OPTIMIZATION ON ALL IMAGES IN TEST SET

dataDir='TestImages'
resultsDir = 'CortexGliaBinarizerRes'
channel = 1

test = cortexglia_binarizer_optimization (dataDir, resultsDir, channel)
print(test)