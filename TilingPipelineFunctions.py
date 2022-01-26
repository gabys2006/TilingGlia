#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:11:33 2022

@author: gabrielasalazar

TilingPipelineFunctions.py

FUNCTIONS USED IN THE PIPELINE


"""

#%%
import numpy as np
import skimage.io
from skimage.util import img_as_float
from skimage.filters import (unsharp_mask, threshold_otsu, 
                             threshold_triangle, threshold_li, threshold_yen,
                             try_all_threshold)
from skimage.morphology import (remove_small_objects, binary_dilation, binary_erosion,
                                remove_small_holes)
from skimage.measure import perimeter, label, regionprops
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import pandas as pd

#%%DENOSINING OPTIMIZATION 

def test_unsharp_mask(dataDir, resultsDir, channel, radii, amts):
    '''
    For optimizing parameters for the radius and amount parameters. 
    Tests 3  values for the radius and amount paramaters of the unsharp mask algorithm. 
    
    ----PARAMETERS----
    dataDir: filepath to directory containing test data
        3D arrays representing 2D images in the form (pixel_row, pixel_column, channel)
    resultsDir: filepath to directory for storing results. 
    radius: list of 3 values to use for radii parameter.
    amts: list of 3 values to use for the amount paramter.
    
    ----RETURNS----
    For each test image, produces a figure showing results using all parameter sets. 
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
                #select single channel
                original=fullimg[:, :, channel]
                #create list of all parameter sets to test
                parameter_sets = []
                for r in radii:
                    for amt in amts:
                        par_s = (r, amt)
                        parameter_sets.append(par_s)
                #create figure
                fig, axes = plt.subplots(nrows=4, ncols=3,
                sharex=True, sharey=True, figsize=(12, 20), tight_layout=True)
                ax = axes.ravel()
                #add original image on figure
                ax[0].imshow(original, cmap=plt.cm.gray)
                ax[0].set_title('Original image: ' + f)
                ax[0].axis('off')
                #remove 2nd and 3rd axes
                ax[1].remove()
                ax[2].remove()
                #iterate thru parameter sets
                for par_s, a in zip(parameter_sets, ax[3:]):
                    #run unsharp mask
                    r, amt = par_s
                    res = unsharp_mask(original, radius=r, amount=amt)
                    a.imshow(res, cmap=plt.cm.gray)
                    a.set_title ('Enhanced, radius=' + str(r) + ', amount=' + str(amt))
                    a.axis('off')
                #save results
                imgID = f.replace('.tif', '')
                figpath=os.path.join(resultsDir, imgID + '_UnsharpMask' + '.png')
                plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
                plt.close()
        return ('DONE')

#%%THRESHOLD OPTIMIZATION

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
                fig, ax = try_all_threshold(unsharp_img, figsize=(4, 6), verbose=False)
                #save results
                imgID = f.replace('.tif', '')
                figpath=os.path.join(resultsDir, imgID + '_THRESHOLD' + '.png')
                plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
                plt.close()
    return ('DONE')


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
                #note the number of true pixels in otsu image
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
                #remove axis
                ax[2].remove()
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


def binarize_neurons(fullimg, channel=2):
    '''Binarizes (denoises and thresholds) neuron channel.
    Chooses between the otsu, triangle and li thresholding algorithms using a decision tree. 

    -----PARAMETERS-----
    fullimg: 3D array representing 2D image (row, col, channel)
    channel: integer indicating neuron channel
    
    ----RETURNS----
    (neuronBIN, neuthresh)
    neuronBIN: Binary version of the neuron channel image. 
    neuthresh: Algorithm selected for thresholding channel for troubleshooting purposes
    '''

    #select neuron channel
    neurons=fullimg[:, :, channel]
    #denoise image using unsharp mask 
    #radius = 20, amount = 4 worked best for us
    #can be optimized using Step 1
    unsharp_img = unsharp_mask(neurons, radius=20, amount=4)

    #apply three thresholds options
    #otsu (I refer to image thresholded by otsu algorithm as otsu image)
    #find threshold
    thresh = threshold_otsu(unsharp_img)
    #apply threshold to denoised image to produce binary image
    otsu = unsharp_img > thresh
    #triangle (I refer to image thresholded by triangle algorithm as triangle image)
    thresh=threshold_triangle(unsharp_img)
    triangle = unsharp_img > thresh
    #li (I refer to image thresholded by li algorithm as li image)
    thresh=threshold_li(unsharp_img)
    li = unsharp_img > thresh
    
    #otsu algorithm works for most neuron images so is the default
    #test for cases where it is not optimal
    ##remove specks (objects < some_area) from binary image
    ##most of these specks are noise
    desp_thresh = 75
    otsu_desp = remove_small_objects(otsu, desp_thresh)
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
    ##the ones below worked for us but can be adjusted as necessary
    #if otsu blows out thresholded image, choose triangle
    if desp_fix > 0.07:
        neuronBIN = triangle
        #note which thresholder was used
        neuthresh='triangle'
    #if otsu blacks out thresholded image, choose li
    elif blackout<0.02:
        neuronBIN=li
        neuthresh='li'
    #if otsu image is not blown/backed out, choose otsu
    else:
        neuronBIN=otsu_desp
        neuthresh='otsu desp' 

    return((neuronBIN, neuthresh))


def binarize_cortex_glia (fullimg, channel=1):
    '''Binarizes (denoises and thresholds) cortex glia channel.
    Chooses between the otsu and triangle thresholding algorithms using a decision tree. 

    -----PARAMETERS-----
    fullimg: 3D array representing 2D image (row, col, channel)
    channel: integer indicating cortex glia channel
    
    ----RETURNS----
    (cgBIN, cgthresh)
    cgBIN: Binary version of the cortex glia channel image. 
    cgthresh: Algorithm selected for thresholding channel for troubleshooting purposess
    '''

    #select cortex glia channel
    cortex_glia = fullimg[:, :, channel]
    #denoise image using unsharp mask 
    #radius = 20, amount = 2 worked best for us
    #can be optimized using Step 1
    unsharp_img = unsharp_mask(cortex_glia, radius=20, amount=2)

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
            cgBIN=otsu
            cgthresh='otsu'
        else:
            cgBIN=triangle
            cgthresh='triangle'
    #if triangle above certain threshold
    else:
        #pick otsu if not overblown
        if (o_overblown<0.5 and o_overblown2< 0.05):
            cgBIN=otsu
            cgthresh='otsu'
        #pick triangle if otsu is overblown
        else:
            cgBIN=triangle
            cgthresh='triangle'            
    return ((cgBIN, cgthresh))


def binarize_astro (fullimg, channel=0):
    '''Binarizes (denoises and thresholds) astrocyte channel.
    Chooses between the otsu, triangle and yen thresholding algorithms using a decision tree. 

    -----PARAMETERS-----
    fullimg: 3D array representing 2D image (row, col, channel)
    channel: integer indicating astrocyte channel
    
    -----RETURNS----
    (astroBIN, astrothresh)
    astroBIN: Binary version of the astrocyte channel image. 
    astrothresh: Algorithm selected for thresholding channel for troubleshooting purposes
    '''

    #select astrocyte channel
    astro=fullimg[:, :, channel]
    #denoise image using unsharp mask 
    #radius = 20, amount = 2 worked best for us
    #can be optimized using Step 1
    unsharp_img = unsharp_mask(astro, radius=20, amount=2)

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
    o_overblown=np.sum(o_sub.astype(float))
    o_overblown=o_overblown/(np.shape(o_sub)[0]*np.shape(o_sub)[1])
                
    #choose thresholding algorithm
    #if otsu is not overblown choose otsu
    if o_overblown<0.3:
            astroBIN=otsu
            astrothresh='otsu'
    #if otsu is overblown
    else:
        #choose triangle if not overblown
        if tri_overblown<0.4:
            astroBIN=triangle
            astrothresh='triangle'
        #choose yen if triangle is overblown
        else:
            astroBIN=yen
            astrothresh='yen'
    
    return((astroBIN, astrothresh))


#%% QUANTIFICATION
def connect_the_dots (cortex_prelim, n, threshold):
    '''Connects objects that are close in the cortex segmentation 
    to produce a solid region representing cortex.
    Turns any black pixels from False (0) to True (1)
    based on the proportion of its neighbors that are True

    ----PARAMETERS---
    cortex_prelim: boolean array representing preliminary cortex segmentation 
    n: neighborhood radius
    threshold: cutoff above which to turn pixels to True (0)

    ----RETURNS----   
    boolean array representing cortex segmentation
    '''

    #make copy of preliminary cortex
    filled_ctx=np.copy(cortex_prelim)
    
    #iterate through pixels
    for i in range(np.shape(cortex_prelim)[0]):
        for j in range(np.shape(cortex_prelim)[1]):
            #find corners of neighborhood
            if i>=n: 
                istart=i-n
            else:
                istart=0  
            if j>=n:
                jstart=j-n
            else:
                jstart=0
            
            if i+n<=np.shape(cortex_prelim)[0]-1:
                iend=i+n
            else:
                iend=np.shape(cortex_prelim)[0]-1
            if j+n<=np.shape(cortex_prelim)[1]-1:
                jend=j+n
            else:
                jend=np.shape(cortex_prelim)[1]-1
            
            #isolate neighborhood
            neighborhood=cortex_prelim[istart:iend, jstart:jend]
            #calculate size of neighborhood
            n_size=np.shape(neighborhood)[0]*np.shape(neighborhood)[1]
            #if neighborhood area is >0
            if n_size>0:
                #turn pixel from 0 to 1 if above set cutoff
                if np.sum(neighborhood)/n_size>threshold:
                    filled_ctx[i, j]=1
                else:
                    filled_ctx[i,j]=cortex_prelim[i,j]
    #convert to boolean
    filled_ctx=filled_ctx.astype(bool)
    return(filled_ctx)


def segment_cortex (cgBIN, neuronBIN, ndil=3, n=15, threshold=0.1, 
                    ctxholes=5000, ctxer=10):
    '''Segments cortex by combining neuron and cortex glia channels
    Fills holes in preliminary segmentation in 2 steps (1st fill, 2nd fill)
        to produce solid region 
    Adjusts segmentation with an erosion. 

    ---PARAMETERS----
    cgBIN: 2D boolean array representing binary cortex glia image
    neuronBIN: 2D boolean array representing binary cortex glia image
    ndil: number of times to dilate binary neuron image
    n: neighborhood radius for first fill 
    threshold: cutoff for turning pixels from False to True in 1st fill 
    ctxholes: area of holes to fill in 2nd fill 
    ctxer: number of erosions for final adjustment

    ---RETURNS----
    AMI: Automated morphology index, metric for globularity 
    '''

    #dilate neuron channel
    for i in range(ndil):
        neuronBINdil=binary_dilation(neuronBIN)
    
    #add cortex glia & neuron channels 
    ctxBIN=neuronBINdil+cgBIN

    #1st fill 
    ctxBINfilled = connect_the_dots(ctxBIN, n, threshold)
    #2nd fill
    ctxBINfinal = remove_small_holes(ctxBINfilled, ctxholes)
    
    #final erosion
    for i in range(ctxer):
        ctxBINfinal=binary_erosion(ctxBINfinal)
            
    return(ctxBINfinal)

def calc_AMI(cgBIN, ctxArea):
    '''Quantifies globularity.

    ---PARAMETERS----
    cgBIN: 2D boolean array representing cortex glia channel
    ctxArea: area of cortex region 

    ---RETURNS----
    AMI: Automated morphology index, metric for globularity 
    '''

    #measure perimeter
    ctxPer = perimeter(cgBIN, neighbourhood=4)
    #normalize, express as percent
    AMI = ctxPer/ctxArea*100

    return(AMI)

def calc_AIS(cortexBIN, astroBIN, ctxArea, 
             cbod_area1=500, cbod_area2=1500, cbod_area3=4000, cbod_area4=5000,
             solid1=0.6, solid2=0.4, ecc1=0.85, 
             return_troubleshooting_img = False, resultsDir=None, sliceID=None):
    
    '''Quantifies aberrant infiltration.

    ---PARAMETERS----
    cortexBIN: 2D boolean array representing cortex region
    astroBIN: 2D boolean array representing astroctyte channel 
    ctxArea: area of cortex region
    return_troubleshooting_img: boolean
        True will produce an image showing all objects in preliminary infiltration
        objects excluded from final infiltration are bounded by a red box
    resultsDir: filepath indicating where to create directory named 'TS_figs'
        for saving troubleshooting figures

    ---RETURNS----
    AIS: Automated infiltration score, metric for aberrant infiltration 
    '''

    #find overlap between cortex and astrocytes 
    ctx_inf=img_as_float(cortexBIN)
    ncg_inf=img_as_float(astroBIN)
    
    inf=ctx_inf+ncg_inf
    inf=inf-1
    inf=np.clip(inf, 0, 1)
    prelim_inf=inf.astype(int)

    # prelim_inf = cortexBIN==astroBIN
    #turn to floats
    prelim_inf = prelim_inf.astype(float)
    
    #CELL BODY EXCLUSION
    #label all objects
    label_inf=label(prelim_inf)
    #measure properties for each object
    rprops= regionprops(label_inf)
    #create final infiltration array to edit
    finalinf=np.copy(prelim_inf)

    #don't produce troubleshooting image
    if return_troubleshooting_img == False:
        #use object areas, eccentricity, and solidity to exclude cell bodies from 
        #final infiltration count
        for region in rprops:
            if region.area > cbod_area1 and region.area <= cbod_area2:
                if region.eccentricity < ecc1:
                    #remove objects fitting the above conditions
                    #from final infiltration 
                    for c in region.coords:
                        finalinf[c[0], c[1]]=0
            if region.area > cbod_area2 and region.area<=cbod_area3:
                if region.solidity>solid1:
                    for c in region.coords:
                        finalinf[c[0], c[1]]=0
                elif region.solidity>solid2 and region.eccentricity<ecc1:
                    for c in region.coords:
                        finalinf[c[0], c[1]]=0
                #to exclude
            if region.area > cbod_area4:
                minr, minc, maxr, maxc = region.bbox
                for c in region.coords:
                    finalinf[c[0], c[1]]=0

    #produce troubleshooting image
    else:
        #create directory to save all troubleshooting images
        TS_Dir = os.path.join(resultsDir, 'TS_figs')
        if not os.path.exists(TS_Dir):
            os.makedirs(TS_Dir)
        #create image
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        plt.gray()
        #show preliminary infiltration 
        ax.imshow(prelim_inf)
        #chose color for bounding excluded objects
        excludeColor='red'
        #use object areas, eccentricity, and solidity to exclude cell bodies from 
        #final infiltration count
        for region in rprops:
            if region.area > cbod_area1 and region.area <= cbod_area2:
                #to exclude
                if region.eccentricity < ecc1:
                    #create bounding box around excluded objects
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1)   
                    ax.add_patch(rect)
                    #remove objects fitting the above conditions
                    for c in region.coords:
                        finalinf[c[0], c[1]]=0
            if region.area > cbod_area2 and region.area<=cbod_area3:
                #to exclude 
                if region.solidity>solid1:
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1) 
                    ax.add_patch(rect)
                    for c in region.coords:
                        finalinf[c[0], c[1]]=0
                elif region.solidity>solid2 and region.eccentricity<ecc1:
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1) 
                    ax.add_patch(rect)
                    for c in region.coords:
                        finalinf[c[0], c[1]]=0
            #to exclude
            if region.area > cbod_area4:
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1)   
                ax.add_patch(rect)
                for c in region.coords:
                    finalinf[c[0], c[1]]=0    
        ax.set_title(sliceID)
        ax.axis('off')
        #save image in results folder
        imgID = sliceID.replace('.tif', '')
        figpath=os.path.join(TS_Dir, imgID + 'TS' + '.png')
        plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
        #suppress output showing every figure
        plt.close()

    #calculate final infiltration area
    infArea = np.sum(finalinf.astype(float))
    #normalize to cortex area
    AIS = infArea/ctxArea*100
    return (AIS)

def quantify_slice(fullimg, sliceID,
                   neuron_channel=0, cortexglia_channel=1, astrocyte_channel=2,
                   return_troubleshooting_img = False, resultsDir=None):
    '''Quantifies globularity and aberrant infiltration for a 2D three-channel slice.

    -----PARAMETERS----
    fullimg: 3D array representing 2D images in the form (pixel_row, pixel_column, channel)
    sliceID: unique filename of image serves as sliceID
    neuron_channel: integer indicating neuron channel 
    cortexglia_channel: integer indicating cortex glia channel 
    astrocyte_channel: integer indicating astrocyte channel 
    return_troubleshooting_img: boolean
        True will save an image showing all objects in preliminary infiltration
        objects excluded from final infiltration are bounded by a red box

    ----RETURNS----
    (GT, sliceID, AMI, AIS)
    GT: genotype indicates experimental group
        6L: non-RNAi control
        7L: driver 1 knockdown 
        8L: driver 2 knockdown
    AMI: Automated morphology index, metric for globularity 
    AIS: Automated infiltration score, metric for aberrant infiltration 
    '''
    #GT code (6L, 7L, 8L) is included in all image filenames/sliceID
    #find which code is in sliceID to determine GT 
    GTs=['6L', '7L', '8L']
    for gt in GTs:
        if gt in sliceID:
            GT = gt
    
    #binarize neuron channel, turn from boolean to floats
    neuronBIN, neuronTHRESH = binarize_neurons(fullimg, neuron_channel)
    #binarize cortex glia channel, turn from boolean to floats
    cgBIN, CGTHRESH = binarize_cortex_glia(fullimg, cortexglia_channel)
    #binarize astrocyte channel, turn from boolean to floats
    astroBIN, astroThresh = binarize_astro(fullimg, astrocyte_channel)

    #segment cortex
    cortexBIN = segment_cortex(cgBIN, neuronBIN)
    #calculate cortex area for normalization 
    ctxArea = np.sum(cortexBIN.astype(float))

    #calculate AMI
    AMI = calc_AMI(cgBIN, ctxArea)
    
    #calculate AIS
    #no troubleshooting image
    if return_troubleshooting_img == False:
        AIS = calc_AIS(cortexBIN, astroBIN, ctxArea)
    #for saving troubleshoothing image
    else:
        AIS = calc_AIS(cortexBIN, astroBIN, ctxArea, 
                       return_troubleshooting_img=True, resultsDir=resultsDir,
                       sliceID=sliceID)
    return ((GT, sliceID, AMI, AIS))

