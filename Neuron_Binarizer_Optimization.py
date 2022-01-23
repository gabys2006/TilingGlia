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
from skimage.filters import unsharp_mask, threshold_otsu, threshold_triangle, threshold_li
from skimage.morphology import remove_small_objects
import numpy as np 

#%%
def neuron_binarizer_optimization (dataDir, resultsDir, channel):
    '''For troubleshooting neuron channel binarization. 
    Selects optimal thresholdingn algorithm from otsu, li, triangle methods

    ----PARAMETERS----
    filepath to directory containing test data
        3D arrays representing 2D images in the form (pixel_row, pixel_column, channel)
    resultsDir: filepath to directory for storing results. 
    channel: integer indicating neuron channel

    ----RETURNS----
    For each test image, produces a figure showing the results of each thresholding algorithm
    and indicating which algorithm would be selected as optimal.
    Notes the values for the two metrics used for selecting optimal algorithm:
        percent of pixels removed from otsu image by despeckling
        percent true pixels in otsu image
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
                unsharp_img = unsharp_mask(original, radius=20, amount=4)

                #apply three thresholds being tested to test image
                #otsu 
                ##I refer to image thresholded by otsu algorithm as otsu image
                #find threshold
                thresh = threshold_otsu(unsharp_img)
                #apply threshold to denoised image to produce binary image
                otsu = unsharp_img > thresh
                #triangle
                ##I refer to image thresholded by triangle algorithm as triangle image
                thresh=threshold_triangle(unsharp_img)
                triangle = unsharp_img > thresh
                #li
                ##I refer to image thresholded by li algorithm as li image
                thresh=threshold_li(unsharp_img)
                li = unsharp_img > thresh
                
                #otsu algorithm works for most neuron images so is the default
                #test for cases where it is not optimal
                ##remove specks (objects < some_area) from binary image
                ##most of these specks are noise
                otsu_desp = remove_small_objects(otsu, 75)
                #check if otsu is blown out (overestimating true signal)
                #grab middle subsection (this is where most of the noise is)
                otsu_sub=otsu[:200, 400:800]
                otsu_d_sub=otsu_desp[:200:, 400:800]
                #if despeckling helps "a lot", Otsu is the wrong filter
                ##calculate the percent of pixels removed by despeckling
                desp_fix=otsu_sub.astype(float)-otsu_d_sub.astype(float)
                desp_fix=np.sum(desp_fix)
                desp_fix=desp_fix/(np.shape(otsu_sub)[0]*np.shape(otsu_sub)[1])
                #make sure otsu is not blacked out (underestimating true signal)
                ##there should be neurons in every image (bc of fly CNS morphology)
                ##calculate the number of true pixels after applying otsu
                blackout=np.sum(otsu.astype(float))
                blackout=blackout/(1200*1200)
                
                #choose thresholding algorithm
                #set cutoffs to decide when to use alternatives to Otsu
                ##the ones below worked for us but can be adjust as necessary
                #if otsu blows out thresholded image, choose triangle
                if desp_fix > 0.07:
                    good = triangle
                    gthresh='triangle'
                    bad1 = otsu_desp
                    bthresh1='Otsu, despeckled'
                    bad2 = li
                    bthresh2 = 'Li'
                #if otsu blacks out thresholded image, choose li
                elif blackout<0.02:
                    good=li
                    gthresh='Li'
                    bad1=otsu_desp
                    bthresh1='Otsu, despeckled'
                    bad2=triangle
                    bthresh2='Triangle'
                #if otsu image is not blown/backed out, choose otsu
                else:
                    good=otsu_desp
                    gthresh='Otsu, despeckled'
                    bad1=triangle
                    bthresh1='triangle'
                    bad2=li
                    bthresh2='Li'
                
                #create figure showing result of otsu, li, triangle
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
                ax[2].set_title('Otsu original')
                #image thresholded by optimal algorithm
                ax[3].imshow(good)
                #note the percent of pixels removed from otsu image by despeckling
                #note the percent of true pixels in otsu image
                ax[3].set_title('Picked: ' + gthresh + '   (' +str(round(desp_fix*100, 3)) 
                                + ', ' + str(round(blackout*100, 3)) + ')' ) 
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
                figpath=os.path.join(resultsDir, imgID + 'NeuTHRESH' + '.png')
                plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
                #suppress output showing every figure
                plt.close()
    return('DONE')

#%%# RUN NEURON_BINARIZER_OPTIMIZATION ON ALL IMAGES IN TEST SET

dataDir='TestImages'
resultsDir = 'NeuronBinarizerRes'
channel = 2

test = neuron_binarizer_optimization (dataDir, resultsDir, channel)
print(test)