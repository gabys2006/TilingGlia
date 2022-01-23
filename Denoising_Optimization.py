#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 01:26:11 2022

@author: gabrielasalazar

FOR OPTIMIZATION OF DENOSING IMAGES USING UNSHARP MASK

"""

#%%
import os
import skimage.io
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask

#%%
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

#%%### RUN UNSHARP MASK WITH VARIOUS PARAMATER SETS ON ALL TEST IMAGE SET ###

#test data directory
dataDir = 'TestImages'
#results directory
resultsDir = 'UnsharpMaskResults'
#channel to be denoised
channel = 0
#list of 3 values for radius parameter
radii = [1, 5, 20]
#list of 3 values for amount parameter
amts = [1, 2, 3]

#run unsharp mask on test images
test = test_unsharp_mask(dataDir, resultsDir, channel, radii, amts)
print (test)