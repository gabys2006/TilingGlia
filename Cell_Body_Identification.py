#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 02:18:09 2021

@author: gabrielasalazar
"""

'''put all new methods together'''

#%%
import os
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

from skimage.util import invert
from PIL import Image
from skimage.filters import try_all_threshold
from skimage.color import label2rgb
from skimage.filters import meijering, sato, frangi, hessian, median, unsharp_mask
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon, label, regionprops,  regionprops_table

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

#%%make infiltration images and save them for trouble shooting

dirName='ValidationImages'

ncgC=0
cgC=1
nC=2

NCG_type=0 #for astro
exludedParticles=[]

recordDir='InfTweaks'
newRun = 'PrelimInf'

newRunPath=os.path.join(recordDir, newRun)
newRunPathcsv=os.path.join(newRunPath, newRun+'.csv')

if not os.path.exists(newRunPath):
    os.makedirs(newRunPath)
    
figD=newRunPath

#pick channel
for root, directory, file in os.walk(dirName):
    for f in file[:]:
        if 'DS_Store' not in f:
            #binarize neurons
            ipath=os.path.join(root, f)
            fullimg=skimage.io.imread(ipath, plugin="tifffile")
            fullimg=img_as_float(fullimg)
            neurons=fullimg[:, :, nC]
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
                
            #binarize cg
            CG=fullimg[:, :, cgC]
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
            
            #binarize NCG
            if NCG_type==0:
                NCG=fullimg[:, :, ncgC]
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
            
            #find overlap
            ctx_inf=img_as_float(ctxBINfinal)
            ncg_inf=img_as_float(ncgBIN)
            
            inf=ctx_inf+ncg_inf
            inf=inf-1
            inf=np.clip(inf, 0, 1)
            inf=inf.astype(int)
            inf=inf.astype(bool)
            
            saving_path=os.path.join(figD, f + 'prelimInf.tif')
            
            skimage.io.imsave(saving_path, inf, plugin="tifffile")

            

#%%label infiltration to decide what to get rid of
#saving infScores for comparison

dirName='InfTweaks/PrelimInf'

ncgC=0
cgC=1
nC=2

NCG_type=0 #for astro
exludedParticles=[]

recordDir='InfTweaks'
newRun = 'ExcludedInf3'

newRunPath=os.path.join(recordDir, newRun)
newRunPathcsv=os.path.join(newRunPath, newRun+'.csv')

if not os.path.exists(newRunPath):
    os.makedirs(newRunPath)
    
figD=newRunPath

GTs=['1BL', '2BL', '6L', '7L', '8L']

RawInf=[]

#pick channel
for root, directory, file in os.walk(dirName):
    for f in file[:]:
        if 'DS_Store' not in f:
            ipath=os.path.join(root, f)
            prelimInf=skimage.io.imread(ipath, plugin="tifffile")
            prelimInf=img_as_float(prelimInf)
           
            finalinf=np.copy(prelimInf)
            
            #get rid of cell bodies
            label_inf=label(prelimInf)
            rprops= regionprops(label_inf)        
          
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax = axes.ravel()
            plt.gray()
            
            ax[0].imshow(prelimInf)
            ax[0].set_title(f + "\n Preliminary Inf")
            
            excludeColor='red'
                        
            s_x=[]
            s_y=[]
            
            m_x=[]
            m_y=[]
            
            l_x=[]
            l_y=[]
            
            prop_measurementID=1
            px=1250
            py=0
            dy=60
            for region in rprops:
                ePprops=[]
                ePprint=[]
                
                cbod_area1=500
                cbod_area2=1500
                if region.area > cbod_area1 and region.area <= cbod_area2:
                    status='kept'
                    colorCode='magenta'
                    for c in region.coords:
                        s_y.append (c[0])
                        s_x.append (c[1])
                    #to exclude
                    ecc=0.9
                    if region.eccentricity < ecc:
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1)   
                        ax[0].add_patch(rect)
                        status='excluded'
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    solid=0.5
                    if region.solidity > solid:
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1)   
                        ax[0].add_patch(rect)
                        status='excluded'
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    ePprint=[prop_measurementID, status, region.area, str(round(region.solidity, 3)), str(round(region.eccentricity,3))]
                    ePprops=[f, prop_measurementID, colorCode, status, region.area, region.solidity, region.eccentricity]

                cbod_area1=1500
                cbod_area2=5000
                if region.area > cbod_area1 and region.area<=cbod_area2:
                    colorCode='cyan'
                    status='kept'
                    for c in region.coords:
                        m_y.append (c[0])
                        m_x.append (c[1])
                    solid1=0.55
                    if region.solidity>solid1:
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1) 
                        ax[0].add_patch(rect)
                        status='excluded'
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    solid1=0.45
                    ecc1 = 0.8
                    if region.solidity>solid1 and region.eccentricity>ecc1:
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1) 
                        ax[0].add_patch(rect)
                        status='excluded'
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    ePprint=[prop_measurementID, status, region.area, str(round(region.solidity, 3)), str(round(region.eccentricity,3))]
                    ePprops=[f, prop_measurementID, colorCode, status, region.area, region.solidity, region.eccentricity]
                
                cbod_area1=5000
                if region.area > cbod_area1:
                    colorCode='orange'
                    status='kept'
                    for c in region.coords:
                        l_y.append (c[0])
                        l_x.append (c[1])
                    #to exclude 
                    #huge things
                    cbod_area1=6000
                    if region.area > cbod_area1:
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1)   
                        status='excluded'
                        ax[0].add_patch(rect)
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    #nonround and solid
                    solid1=0.5
                    ecc1 = 0.9
                    if region.solidity>solid1 and region.eccentricity>ecc1:
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1)   
                        status='excluded'
                        ax[0].add_patch(rect)
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    ePprint=[prop_measurementID, status, region.area, str(round(region.solidity, 3)), str(round(region.eccentricity,3))]
                    ePprops=[f, prop_measurementID, colorCode, status, region.area, region.solidity, region.eccentricity]
                if len(ePprops)>0:
                    #print numbers on image
                    minr, minc, maxr, maxc = region.bbox
                    ax[0].text(maxc, maxr, prop_measurementID, c=colorCode, fontweight='bold', fontsize=12)
                    #print all particles
                    exludedParticles.append(ePprops)
                    plt.text(px, py+dy*prop_measurementID, ePprint)
                    prop_measurementID=prop_measurementID+1
                                
            #color particles                        
            colorCode='magenta'
            dotsize=0.005
            ax[0].scatter(s_x, s_y, color=colorCode, s=dotsize)
            
            colorCode='c'
            ax[0].scatter(m_x, m_y, color=colorCode, s=dotsize)
            
            colorCode='orange'
            ax[0].scatter(l_x, l_y, color=colorCode, s=dotsize)
        
            ax[1].imshow(finalinf)
            ax[1].set_title('Final Inf')
            
            for a in ax:
                a.axis('off')
            
            picID=f.replace('prelimInf.tif', '')
            figpath=os.path.join(figD, f + 'numbered' + '.png')
            plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
    
            plt.show() 
           
            #get and save score
            #get genotype
            for gt in GTs:
                if gt in picID:
                    GT = gt
            rawIS=np.sum(finalinf)
            RawInf.append([GT, picID, rawIS])
  
            
#%%make file for comparison with manual 
col=['GT', 'sliceID', 'rawInf']
rawInf=pd.DataFrame(RawInf, columns=col)

#old inf to get ctx area
recsDir='InfRuns'
run='20210528'
path=os.path.join(recsDir, run, run + '.csv')
oldInf=pd.read_csv(path, index_col=False)
ctx=oldInf.loc[:, ['GT', 'sliceID', 'ctxArea']]

key=['GT', 'sliceID']
mergedInf=pd.merge(rawInf, ctx, how='inner', on=key)  
normalInf=mergedInf.rawInf/mergedInf.ctxArea*100
mergedInf=mergedInf.assign(infPerc=normalInf)

normInf=mergedInf.loc[:, ['GT', 'sliceID', 'infPerc']]

#new file
recsDir='InfRuns'
run='20210531'

newRunDir=os.path.join(recsDir, run)

if not os.path.exists(newRunDir):
    os.makedirs(newRunDir)
    
newRunPath=os.path.join(recsDir, run, run + '.csv')
normInf.to_csv(newRunPath, index=False)

#%%
col=['PicID', 'RegonID', 'Color_Code', 'Staust', 'Area', 'Solidity', 'Eccentricity']
excludedParticles=pd.DataFrame(exludedParticles, columns=col)

excludedParticles.to_csv('InfTweaks/labeledParts.csv', index=False)
