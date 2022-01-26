#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 02:18:09 2021

@author: gabrielasalazar
"""

'''put all new methods together'''

#%%
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola, threshold_isodata,
                             threshold_li, threshold_mean,
                             threshold_local, threshold_minimum,
                             threshold_multiotsu, threshold_triangle,
                             sobel, threshold_yen
                             )
from skimage.morphology import (area_closing, remove_small_objects,
                                remove_small_holes, binary_dilation,
                                closing, convex_hull_object, 
                                binary_closing, diameter_closing, binary_erosion)
import matplotlib.patches as mpatches

from skimage.util import invert, img_as_float
from PIL import Image
from skimage.filters import try_all_threshold
from skimage.color import label2rgb
from skimage.filters import meijering, sato, frangi, hessian, median, unsharp_mask
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon, label, regionprops,  regionprops_table, perimeter

#%%
from matplotlib import gridspec

# from skimage.data import chelsea, hubble_deep_field
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import (calibrate_denoiser,
                                 denoise_wavelet,
                                 denoise_tv_chambolle, denoise_nl_means,
                                 estimate_sigma)
from skimage.util import img_as_float, random_noise, img_as_int
from skimage.color import rgb2gray
from functools import partial
from skimage.restoration.j_invariant import _invariant_denoise
from skimage.segmentation import flood_fill

#%%
def connect_the_dots (img, n, threshold):
    
    original_image=np.copy(img)
    new_img=np.copy(img)
    
    for i in range(np.shape(original_image)[0]):
        for j in range(np.shape(original_image)[1]):
            if i>=n: 
                istart=i-n
            else:
                istart=0  
            if j>=n:
                jstart=j-n
            else:
                jstart=0
            
            if i+n<=np.shape(original_image)[0]-1:
                iend=i+n
            else:
                iend=np.shape(original_image)[0]-1
            if j+n<=np.shape(original_image)[1]-1:
                jend=j+n
            else:
                jend=np.shape(original_image)[1]-1
            
            neighborhood=img[istart:iend, jstart:jend]
            n_size=np.shape(neighborhood)[0]*np.shape(neighborhood)[1]
            
            if n_size>0:
                if np.sum(neighborhood)/n_size>threshold:
                    new_img[i, j]=1
                else:
                    new_img[i,j]=img[i,j]
    return(new_img)

#%%
dirName='NewData'

ncgC=0
cgC=1
nC=2

NCG_type=0 #for astro

recordDir='InfRuns'
newRun = '2021_12_30_8LCheck'

newRunPath=os.path.join(recordDir, newRun)
newRunPathcsv=os.path.join(newRunPath, newRun+'.csv')

#make dir for this run
if not os.path.exists(newRunPath):
    os.makedirs(newRunPath)

#make dir for this run CG BINs
cgBinD='cgBIN'
cgBinDir=os.path.join(newRunPath, cgBinD)
if not os.path.exists(cgBinDir):
    os.makedirs(cgBinDir)

#make dir for this run astro BINs
ncgBinD='ncgBIN'
ncgBinDir=os.path.join(newRunPath, ncgBinD)
if not os.path.exists(ncgBinDir):
    os.makedirs(ncgBinDir)
    
# make dir for this run inf BINs
infBinD='infBIN'
infBinDir=os.path.join(newRunPath, infBinD)
if not os.path.exists(infBinDir):
    os.makedirs(infBinDir)

with open(newRunPathcsv, "w", newline='') as rec:
    writer = csv.writer(rec, delimiter=',')
    writer.writerow(['GT', 'sliceID', 'brainID', 'neuronArea', 'cgArea', 'cgPer', 'astroArea', 'ctxArea', 'infArea', 'globScore', 'infPerc'])
    rec.close()

GTs=['1BL', '2BL', '6L', '7L', '8L', '11L']

errors=[]

progress=0

#pick channel
for root, directory, file in os.walk(dirName):
    for d in directory[:]:
        path=os.path.join(root, d)
        for Root, Directory, File in os.walk(path):  
            print("will quantify", len(File), "images")
            for f in File[:]:
                try:
                    #get genotype
                    for gt in GTs:
                        if gt in f:
                            GT = gt
                    brainID=f.replace('.tif', '')
                    #load image
                    ipath=os.path.join(Root, f)
                    fullimg=skimage.io.imread(ipath, plugin="tifffile")
                    #get each slice
                    zlen=np.shape(fullimg)[0]
                    print('there are ', zlen, 'slices in this image')
                    print('current image progress:')
                    for s in range(zlen):
                        print(s/zlen*100, '%')
                        #get sliceID
                        sliceID=f.replace('.tif', '')
                        sliceID=sliceID+'_'+ str(s)+'.tif'
                        #convert img to floats
                        imgslice=img_as_float(fullimg[s, :, :, :])
                        #BINARIZE NEURONS######################################################
                        neurons=imgslice[:, :, nC]
                        unsharp_neu = unsharp_mask(neurons, radius=20, amount=4)
                    
                        #works for most
                        thresh = threshold_otsu(unsharp_neu)
                        otsu = unsharp_neu > thresh
                        otsu_desp = remove_small_objects(otsu, 75)
                    
                        #check if otsu is blown out
                        #grabsubsection
                        otsu_sub=otsu[:200, 400:800]
                        otsu_d_sub=otsu_desp[:200:, 400:800]
                    
                        #if despeckling helps a lot, it's the wrong filter
                        desp_fix=otsu_sub.astype(float)-otsu_d_sub.astype(float)
                        desp_fix=np.sum(desp_fix)
                        desp_fix=desp_fix/(np.shape(otsu_sub)[0]*np.shape(otsu_sub)[1])
                    
                        #make sure otsu is not too little
                        blackout=np.sum(otsu.astype(float))
                        blackout=blackout/(1200*1200)
                    
                        #use triangle for blownouts
                        thresh=threshold_triangle(unsharp_neu)
                        triangle = unsharp_neu > thresh
                    
                        #use li for blackouts
                        thresh=threshold_li(unsharp_neu)
                        li = unsharp_neu > thresh
                 
                        #tri
                        if desp_fix > 0.07:
                            neuronBIN = triangle
                            neuthresh='triangle'
                        
                        elif blackout<0.02:
                            neuronBIN=li
                            neuthresh='li'
                        
                        else:
                            neuronBIN=otsu_desp
                            neuthresh='otsu desp 75'
                    
                        #save this
                        neuArea = np.sum(neuronBIN.astype(float))
                    
                        #BINARIZE CG###########################################################
                        CG=imgslice[:, :, cgC]
                        unsharp_CG = unsharp_mask(CG, radius=20, amount=2)
            
                        #get triangle threshold
                        thresh = threshold_triangle(unsharp_CG)
                        triangle = unsharp_CG > thresh
                        #to check if blown out
                        #grab subsction
                        tri_sub=triangle[900:1200, 400:800]
                        tri_overblown=np.sum(tri_sub.astype(float))
                        tri_overblown=tri_overblown/(np.shape(tri_sub)[0]*np.shape(tri_sub)[1])
                    
                        #otsu
                        thresh = threshold_otsu(unsharp_CG)
                        otsu = unsharp_CG > thresh
                        #to check if blown out
                        o_sub=otsu[:, 400:800]
                        otsu_desp = remove_small_objects(otsu, 75)
                        od_sub=otsu_desp[:, 400:800]
                        o_overblown2=np.sum(o_sub.astype(float)-od_sub.astype(float))
                        o_overblown2=o_overblown2/(np.shape(o_sub)[0]*np.shape(o_sub)[1])
                        o_overblown=np.sum(o_sub.astype(float))/(np.shape(o_sub)[0]*np.shape(o_sub)[1])
                    
                        if (o_overblown<0.5 and o_overblown2 < 0.05 and tri_overblown<0.5):
                            if o_overblown>tri_overblown:
                                cgBIN=otsu
                                cgthresh='otsu'
                            else:
                                cgBIN=triangle
                                cgthresh='triangle'
                        else:
                            if (o_overblown<0.5 and o_overblown2< 0.05):
                                cgBIN=otsu
                                cgthresh='otsu'
                            else:
                                cgBIN=triangle
                                cgthresh='triangle'
                    
                        #save this
                        cgArea = np.sum(cgBIN.astype(float))
                        
                        #calculate perimeter for glob score
                        #Perimeter
                        #save this
                        cgPer=perimeter(cgBIN, neighbourhood=4)
                        
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
                        plt.gray()
        
                        ax.imshow(cgBIN)
                        ax.set_title(f)
                        ax.axis('off')
                        
                        #save img for troubleshooting
                        figpath=os.path.join(cgBinDir, sliceID + 'cgBIN' + '.png')
                        plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
            
                        plt.show()
                            
                        #CORTEX################################################################
                        #add cg & neu to make ctx
                        #dilate neurons a bit
                        ndil=3
                    
                        for i in range(ndil):
                            neuronBINdil=binary_dilation(neuronBIN)
                    
                        ctxBIN=neuronBINdil+cgBIN
                        n=15
                        thresh=0.1
                        ctxBINfilled = connect_the_dots(ctxBIN, n, thresh)
                        ctxholes=5000
                        ctxBINfinal = remove_small_holes(ctxBINfilled, ctxholes)
                    
                        ctxer=10
                        for i in range(ctxer):
                            ctxBINfinal=binary_erosion(ctxBINfinal)
                        
                        #save this
                        ctxArea = np.sum(ctxBINfinal.astype(float))
                
                        #BINARIZE NCG##########################################################
                        if NCG_type==0:
                            NCG=imgslice[:, :, ncgC]
                            unsharp_NCG = unsharp_mask(NCG, radius=20, amount=2)
                            
                            #get triangle threshold
                            thresh = threshold_triangle(unsharp_NCG)
                            triangle = unsharp_NCG > thresh
                            #to check if blown out
                            #grab subsction
                            tri_sub=triangle[900:1200, :]
                            tri_overblown=np.sum(tri_sub.astype(float))
                            tri_overblown=tri_overblown/(np.shape(tri_sub)[0]*np.shape(tri_sub)[1])
                            
                            #otsu
                            thresh = threshold_otsu(unsharp_NCG)
                            otsu = unsharp_NCG > thresh
                            #to check if blown out
                            o_sub=otsu[800:1200, :]
                            o_overblown=np.sum(o_sub.astype(float))
                            o_overblown=o_overblown/(np.shape(o_sub)[0]*np.shape(o_sub)[1])
                    
                            #yen
                            thresh = threshold_yen(unsharp_NCG)
                            yen = unsharp_NCG > thresh
                    
                            if o_overblown<0.3:
                                 ncgBIN=otsu
                                 ncgthresh='otsu'
                            else:
                                if tri_overblown<0.4:
                                    ncgBIN=triangle
                                    ncgthresh='triangle'
                                else:
                                    ncgBIN=yen
                                    ncgthresh='yen'
                            
                            autoDil=1
                            for i in range(autoDil):
                                ncgBIN=binary_dilation(ncgBIN)
                                
                            #save this
                            ncgArea = np.sum(ncgBIN.astype(float))
                                
                            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
                            plt.gray()
        
                            ax.imshow(ncgBIN)
                            ax.set_title(f)
                            ax.axis('off')
                        
                            #save img for troubleshooting
                            figpath=os.path.join(ncgBinDir, sliceID + 'ncgBIN' + '.png')
                            plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
                            
                            plt.show()
                                
                    
                        #INFILTRATION##########################################################
                        ctx_inf=img_as_float(ctxBINfinal)
                        ncg_inf=img_as_float(ncgBIN)
                        
                        inf=ctx_inf+ncg_inf
                        inf=inf-1
                        inf=np.clip(inf, 0, 1)
                        inf=inf.astype(int)
                    
                        #get rid of cell bodies
                        label_inf=label(inf)
                        rprops= regionprops(label_inf)
                        
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
                        plt.gray()
                    
                        finalinf=np.copy(inf)
                        ax.imshow(inf)

                        for region in rprops:
                            cbod_area1=500
                            cbod_area2=1500
                            if region.area > cbod_area1 and region.area <= cbod_area2:                            
                                #to exclude
                                ecc=0.9
                                if region.eccentricity < ecc:
                                    minr, minc, maxr, maxc = region.bbox
                                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='magenta', linewidth=1)   
                                    ax.add_patch(rect)
                                    for c in region.coords:
                                        finalinf[c[0], c[1]]=0
                                        
                                solid=0.5
                                if region.solidity > solid:
                                    minr, minc, maxr, maxc = region.bbox
                                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='magenta', linewidth=1)   
                                    ax.add_patch(rect)
                                    for c in region.coords:
                                        finalinf[c[0], c[1]]=0

                            cbod_area1=1500
                            cbod_area2=5000
                            if region.area > cbod_area1 and region.area<=cbod_area2:
                                solid1=0.55
                                if region.solidity>solid1:
                                    minr, minc, maxr, maxc = region.bbox
                                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='cyan', linewidth=1) 
                                    ax.add_patch(rect)
                                    for c in region.coords:
                                        finalinf[c[0], c[1]]=0
                            
                                solid1=0.45
                                ecc1 = 0.8
                                if region.solidity>solid1 and region.eccentricity>ecc1:
                                    minr, minc, maxr, maxc = region.bbox
                                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='cyan', linewidth=1) 
                                    ax.add_patch(rect)
                                    for c in region.coords:
                                        finalinf[c[0], c[1]]=0
                            
                            cbod_area1=5000
                            if region.area > cbod_area1:
                                #to exclude 
                                #huge things
                                cbod_area1=6000
                                if region.area > cbod_area1:
                                    minr, minc, maxr, maxc = region.bbox
                                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='orange', linewidth=1)   
                                    ax.add_patch(rect)
                                    for c in region.coords:
                                        finalinf[c[0], c[1]]=0
                                #nonround and solid
                                solid1=0.5
                                ecc1 = 0.9
                                if region.solidity>solid1 and region.eccentricity<ecc1:
                                    minr, minc, maxr, maxc = region.bbox
                                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='orange', linewidth=1)   
                                    ax.add_patch(rect)
                                    for c in region.coords:
                                        finalinf[c[0], c[1]]=0
                        
                        ax.set_title(f)   
                        ax.axis('off')
                        
                        #save this
                        infArea = np.sum(finalinf.astype(float))
                    
                        #save img for troubleshooting
                        figpath=os.path.join(infBinDir, sliceID + 'cgBIN' + '.png')
                        plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
            
                        plt.show()
                        
                        #save measurements
                        # ['GT', 'sliceID', 'neuronArea', 'cgArea', 'cgPer', 'astroArea', 'ctxArea', 'infArea', 'globScore', 'infPerc'])

                        measurements=[GT, sliceID, brainID, neuArea, cgArea, cgPer, ncgArea, ctxArea, infArea, cgPer/ctxArea*100, infArea/ctxArea*100]
                        
                        with open(newRunPathcsv, 'a', newline='') as rec:
                            writer = csv.writer(rec, delimiter=',')
                            writer.writerow(measurements)
                            rec.close()
                except:
                    errors.append(f)
        
                #show progress
                progress = progress + 1
                progperc = progress/len(File[:])*100
                print('Total progress in folder ', d, ' : ', progperc, "%")


if len(errors)>0:
    print('\n there are errors with:')
    for e in errors:
        print(errors)
else:
    print('no errors')




