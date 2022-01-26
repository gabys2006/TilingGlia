#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 02:19:58 2021

@author: gabrielasalazar
"""

'''validate selection of cortex ROI and finalize hyperparameter selection'''


#%%
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr 
import csv
import skimage.io
import skimage
import scipy
from scipy import stats
import sklearn
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import f_oneway, sem
import statsmodels.stats.multicomp
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
from PIL import Image
from skimage.filters import try_all_threshold

from skimage.filters import meijering, sato, frangi, hessian, median, unsharp_mask
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon, perimeter, label, regionprops

#%%
from matplotlib import gridspec

# from skimage.data import chelsea, hubble_deep_field
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util import img_as_float, random_noise
from functools import partial
from skimage.segmentation import flood_fill
import pandas as pd

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
dirName='ValidationImages'
# figD='CtxROIVal/BinaryCtxAuto/2021_05_26'

ncgC=0
cgC=1
nC=2

GTs=['1BL', '2BL', '6L', '7L', '8L']


ctxMeasurements=[]
errors=[]

###########Manual
#pick channel
for root, directory, file in os.walk(dirName):
    for f in file[:]:
        #get GT
        for gt in GTs:
            if gt in f:
                GT = gt
        
        if '.DS_Store' not in f:
                
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
            
            n=10
            thresh=0.06
            ctxBINfilled = connect_the_dots(ctxBIN, n, thresh)
            
            ctxholes=10000
            ctxBINfilled2 = remove_small_holes(ctxBINfilled, ctxholes)
            
            ctxBINfinal=ctxBINfilled2
            
            ctxer=10
            for i in range(ctxer):
                ctxBINfinal=binary_erosion(ctxBINfinal)
            
            #measure area
            auto_area=np.sum(ctxBINfinal.astype(float))
            #measure per
            auto_per=perimeter(ctxBINfinal, neighbourhood=4)
            
            # Find and label perimeter
            labeled_auto=label(ctxBINfinal)
            props = regionprops(labeled_auto)
            
            # manual binary ctx ##################################
            #binary ctx image folder 
            mroot='CtxROIVal/BinaryCtxManual'
            mpath=os.path.join(mroot, f)
            #(there are 4 missing that didn't have a ctx perimeter saved)
            try:
                manCtx=skimage.io.imread(mpath, plugin="tifffile")
                manCtx=img_as_float(manCtx)
            except:
                errors.append(f)
                continue
            
            #measure area
            manual_area=np.sum(manCtx.astype(float))
            #measure per
            man_per=perimeter(manCtx, neighbourhood=4)
            
            ###overlap##############
            overlap=ctxBINfinal+manCtx
            overlap=overlap-1
            overlap=np.clip(overlap, 0, 1)
            overlap=overlap.astype(int)
            manual_overlap=np.sum(overlap)/manual_area*100
            auto_overlap=np.sum(overlap)/auto_area*100
            
            #save measurements
            m=[GT, f, manual_area, auto_area, man_per, auto_per, manual_overlap, auto_overlap]
            ctxMeasurements.append(m)
            
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
            ax = axes.ravel()
            plt.gray()
            
            alf1=1
            alf2=0.5
            
            # ax[0].imshow(fullimg, cmap=plt.cm.gray)
            # ax[0].set_title(f + '\nOriginal')
            
            ax[0].imshow(manCtx, cmap=plt.cm.PuRd, alpha=alf2)
            ax[0].set_title('Manual Ctx')
            
            ax[1].imshow(ctxBINfinal, cmap=plt.cm.Greens, alpha=alf1)
            ax[1].imshow(manCtx, cmap=plt.cm.PuRd, alpha=alf2)
            ax[1].set_title('Overlap:' + str(round(manual_overlap, 2)) + ', ' + str(round(auto_overlap, 2)))
    
            ax[2].imshow(ctxBIN, cmap=plt.cm.gray)
            ax[2].set_title('Neurons + CG')
            
            ax[3].imshow(ctxBINfilled, cmap=plt.cm.gray)
            ax[3].set_title('Connect dots: ' + str(n) + ', ' + str(thresh))
            
            ax[4].imshow(ctxBINfilled2, cmap=plt.cm.gray)
            ax[4].set_title('Remove small holes: ' + str(ctxholes))
            
            ax[5].imshow(ctxBINfinal, cmap=plt.cm.Greens, alpha=alf2)
            ax[5].set_title('Eroded: ' + str(ctxer))
    
            for a in ax:
                a.axis('off')
            
            figpath=os.path.join(figD, f + 'ctx' + '.png')
            plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
    
            plt.show()
            
#%%parameter optimization

#Optimizing 4 parameters
#neighborhood for connect-the-dots
ns=[5, 10, 15, 25]
#threshold
threshes=[0.03, 0.06, 0.1]
#size of small holes
holes=[1000, 5000, 10000, 15000]
#number of final erosions
erosions=[5, 10, 15, 25]

#how many combos?
combos=len(ns)*len(threshes)*len(holes)*len(erosions)
print (combos)

#%%
dirName='ValidationImages'

ncgC=0
cgC=1
nC=2

GTs=['1BL', '2BL', '6L', '7L', '8L']

ctxMeasurements=[]
errors=[]
img=0
for root, directory, file in os.walk(dirName):
    print (len(file), "images to process")
    
    for f in file[:]:
        if '.DS_Store' not in f:  
            #get GT
            for gt in GTs:
                if gt in f:
                    GT = gt
            #for each validation image to threshold
            #progress tracking
            img=img+1
            print("Processing image:", img)
            # manual binary ctx ##################################
            #binary ctx image folder 
            mroot='CtxROIVal/BinaryCtxManual'
            mpath=os.path.join(mroot, f)
            #(there are 4 missing that didn't have a ctx perimeter saved)
            try:
                manCtx=skimage.io.imread(mpath, plugin="tifffile")
                manCtx=img_as_float(manCtx)
            except:
                errors.append(f)
                continue
            
            #measure area
            manual_area=np.sum(manCtx.astype(float))
            #measure per
            man_per=perimeter(manCtx, neighbourhood=4)
            
            # auto binary ctx ########################################
            # load image
            ipath=os.path.join(root, f)
            fullimg=skimage.io.imread(ipath, plugin="tifffile")
            fullimg=img_as_float(fullimg)
            
            #binarize neurons
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
            if desp_fix > 0.1:
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
            
            #now hyperparameter optimization 
            #4 loops
            for nval in ns:
                print ('(image ', img, ')', 'nval:', nval)
                for threshval in threshes:
                    print ('    threshval:', threshval)
                    for holeval in holes:
                        print ('    holeval:', holeval)
                        for eroval in erosions:
                            print ('    eroval:', eroval)
                            n=nval
                            thresh=threshval
                            ctxBINfilled = connect_the_dots(ctxBIN, n, thresh)
                            
                            ctxholes=holeval
                            ctxBINfilled2 = remove_small_holes(ctxBINfilled, ctxholes)
                            
                            ctxBINfinal=ctxBINfilled2
                            
                            ctxer=eroval
                            for i in range(ctxer):
                                ctxBINfinal=binary_erosion(ctxBINfinal)
                            
                            #measure area
                            auto_area=np.sum(ctxBINfinal.astype(float))
                            #measure per
                            auto_per=perimeter(ctxBINfinal, neighbourhood=4)
                            
                            #############get overlap##############
                            overlap=ctxBINfinal+manCtx
                            overlap=overlap-1
                            overlap=np.clip(overlap, 0, 1)
                            overlap=overlap.astype(int)
                            manual_overlap=np.sum(overlap)/manual_area*100
                            auto_overlap=np.sum(overlap)/auto_area*100
                            
                            #save measurements
                            m=[GT, f, nval, threshval, holeval, eroval, manual_area, auto_area, man_per, auto_per, manual_overlap, auto_overlap]
                            ctxMeasurements.append(m)

#%%make and save handy table 
col=['GT', 'picID', 'n', 'thresh', 'hole_size', 'erosions', 'manual_area', 'auto_area', 'man_per', 'auto_per', 'manual_overlap', 'auto_overlap']
ctxMeasurements=pd.DataFrame(ctxMeasurements, columns=col)

print(ctxMeasurements)
print(ctxMeasurements.columns)

# path='CtxROIVal/BinaryCtxAuto/ParamaterOptimization.csv'
# ctxMeasurements.to_csv(path, index=False)

#%%load table 
path='CtxROIVal/BinaryCtxAuto/CtxParamaterOptimization.csv'
ctxMeasurements=pd.read_csv(path, index_col=False)

print (ctxMeasurements.columns)

#%%group by parameter combos
key=['n', 'thresh', 'hole_size', 'erosions']
OLscores=ctxMeasurements[['n', 'thresh', 'hole_size', 'erosions', 'manual_overlap', 'auto_overlap']]
OLscores=OLscores.groupby(key, as_index=False).mean()

#%%Find best combo
bestCombo=OLscores[(OLscores.manual_overlap>88) & (OLscores.auto_overlap>85)]
print(bestCombo)


#%%graph results
tickfont=12
ticklength=4
tickwidth=2

matplotlib.rcParams['pdf.fonttype'] = 42

plt.figure(figsize=(3.2, 3), facecolor='w')

y=OLscores.manual_overlap
x=np.arange(len(y))
# plt.fill_between(x, y-sem(y), y+sem(y), color='gray', alpha=0.3)
plt.plot(x, y, c='gray', label='manual', linewidth=1)

y=OLscores.auto_overlap
# plt.fill_between(x, y-sem(y), y+sem(y), color='skyblue', alpha=0.5)
plt.plot(x, y, c='skyblue', label='auto', linewidth=1)
y1, y2 = plt.ylim()
plt.vlines(133, y1, y2, linewidth=1.5, color='r')


# plt.text(133, bestCombo.manual_overlap, '*', fontsize=20, fontweight='bold', c='r')
# plt.text(133, bestCombo.auto_overlap, '*', fontsize=20, fontweight='bold', c='r')

# plt.legend(fontsize=12, prop={'weight': 'bold'}, frameon=False)

# plt.title('ROI parameter optimization', fontsize=16, fontweight='bold')
plt.xlabel ('Parameter Set ID', fontsize=12, fontweight='bold')
xtx=np.arange(20, 201, 40)
plt.xticks(ticks=xtx, labels=xtx, color='k', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='k', 
                labelsize=12, length=ticklength, width=tickwidth)

plt.ylabel ('% Overlap \n Cortex Segmentation', fontsize=12, fontweight='bold')
plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, width=tickwidth, color='k')  

plt.tight_layout()

plt.savefig("SupFig1/ROI_par_select.pdf", transparent=True)

plt.show()




#%%uses only one set of hyperparameter
#optmized hyperparameters found from above
      # n  thresh  hole_size  erosions  manual_overlap  auto_overlap
#133  15    0.10       5000        10       88.211742     85.462992

dirName='ValidationImages'
figD='CtxROIVal/BinaryCtxAuto/2021_05_26'
ctxMeasurements=[]

###########Manual
#pick channel
for root, directory, file in os.walk(dirName):
    for f in file[:]:
        #get GT
        for gt in GTs:
            if gt in f:
                GT = gt
        
        if '.DS_Store' not in f:
                
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
            thresh=0.10
            ctxBINfilled = connect_the_dots(ctxBIN, n, thresh)
            
            ctxholes=5000
            ctxBINfilled2 = remove_small_holes(ctxBINfilled, ctxholes)
            
            ctxBINfinal=ctxBINfilled2
            
            ctxer=10
            for i in range(ctxer):
                ctxBINfinal=binary_erosion(ctxBINfinal)
            
            #measure area
            auto_area=np.sum(ctxBINfinal.astype(float))
            #measure per
            auto_per=perimeter(ctxBINfinal, neighbourhood=4)
            
            # manual binary ctx ##################################
            #binary ctx image folder 
            mroot='CtxROIVal/BinaryCtxManual'
            mpath=os.path.join(mroot, f)
            #(there are 4 missing that didn't have a ctx perimeter saved)
            try:
                manCtx=skimage.io.imread(mpath, plugin="tifffile")
                manCtx=img_as_float(manCtx)
            except:
                errors.append(f)
                continue
            
            #measure area
            manual_area=np.sum(manCtx.astype(float))
            #measure per
            man_per=perimeter(manCtx, neighbourhood=4)
            
            ###overlap##############
            overlap=ctxBINfinal+manCtx
            overlap=overlap-1
            overlap=np.clip(overlap, 0, 1)
            overlap=overlap.astype(int)
            manual_overlap=np.sum(overlap)/manual_area*100
            auto_overlap=np.sum(overlap)/auto_area*100
            
            #save measurements
            m=[GT, f, manual_area, auto_area, man_per, auto_per, manual_overlap, auto_overlap]
            ctxMeasurements.append(m)
            
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
            ax = axes.ravel()
            plt.gray()
            
            alf1=1
            alf2=0.5
            
            ax[0].imshow(manCtx, cmap=plt.cm.PuRd, alpha=alf2)
            ax[0].set_title('Manual Ctx')
            
            ax[1].imshow(ctxBINfinal, cmap=plt.cm.Greens, alpha=alf1)
            ax[1].imshow(manCtx, cmap=plt.cm.PuRd, alpha=alf2)
            ax[1].set_title('Overlap')
            
            ax[2].imshow(ctxBINfinal, cmap=plt.cm.Greens, alpha=alf2)
            ax[2].set_title('Auto Ctx')
    
            ax[3].imshow(ctxBIN, cmap=plt.cm.gray)
            ax[3].set_title('Neurons + CG')
            
            ax[4].imshow(ctxBINfilled, cmap=plt.cm.gray)
            ax[4].set_title('Filled in')
            
            ax[5].imshow(ctxBINfilled2, cmap=plt.cm.gray)
            ax[5].set_title('Remove small holes')
    
            for a in ax:
                a.axis('off')
            
            figpath=os.path.join(figD, f + 'ctx' + '.png')
            plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
    
            plt.show()
            
col=['GT', 'picID', 'manual_area', 'auto_area', 'man_per', 'auto_per', 'manual_overlap', 'auto_overlap']
ctxMeasurements=pd.DataFrame(ctxMeasurements, columns=col)

mname='overlapMeasurements.csv'
path=os.path.join(figD, mname)
ctxMeasurements.to_csv(path, index=False)
            
#%%analyze results 
#total overlap 
col=['GT', 'picID', 'manual_area', 'auto_area', 'man_per', 'auto_per', 'manual_overlap', 'auto_overlap']
ctxMeasurements=pd.DataFrame(ctxMeasurements, columns=col)

print(ctxMeasurements)
print(ctxMeasurements.columns)

# path='CtxROIVal/BinaryCtxAuto/Ver1/ctxMeasurements.csv'
# ctxMeasurements.to_csv(path, index=False)

#%%
path='CtxROIVal/BinaryCtxAuto/Ver1/ctxMeasurements.csv'

ctxMeasurements=pd.read_csv(path, index_col=False)


#%%means
manOLmean=ctxMeasurements.manual_overlap.mean()
print ('manOLmean', manOLmean)

autoOLmean=ctxMeasurements.auto_overlap.mean()
print ('autoOLmean', autoOLmean)
print()

gt='6L'
mean=ctxMeasurements.manual_overlap[ctxMeasurements.GT==gt].mean()
print(mean)
mean=ctxMeasurements.auto_overlap[ctxMeasurements.GT==gt].mean()
print(mean)
print()

gt='7L'
mean=ctxMeasurements.manual_overlap[ctxMeasurements.GT==gt].mean()
print(mean)
mean=ctxMeasurements.auto_overlap[ctxMeasurements.GT==gt].mean()
print(mean)
print()

gt='8L'
mean=ctxMeasurements.manual_overlap[ctxMeasurements.GT==gt].mean()
print(mean)
mean=ctxMeasurements.auto_overlap[ctxMeasurements.GT==gt].mean()
print(mean)
print()


#%%Overlap bar graphs with dots
cap=5
barlw=2


tickfont=12
ticklength=4
tickwidth=2

matplotlib.rcParams['pdf.fonttype'] = 42

plt.figure(figsize=(3.2, 3), facecolor='w')
manual=ctxMeasurements.manual_overlap
auto=ctxMeasurements.auto_overlap

gt='6L'
dots6Lman=ctxMeasurements.manual_overlap[ctxMeasurements.GT==gt]
dots6Lauto=ctxMeasurements.auto_overlap[ctxMeasurements.GT==gt]

gt='7L'
dots7Lman=ctxMeasurements.manual_overlap[ctxMeasurements.GT==gt]
dots7Lauto=ctxMeasurements.auto_overlap[ctxMeasurements.GT==gt]

gt='8L'
dots8Lman=ctxMeasurements.manual_overlap[ctxMeasurements.GT==gt]
dots8Lauto=ctxMeasurements.auto_overlap[ctxMeasurements.GT==gt]

dotsize=18
dx=0.1
w=0.8
bwidth=2
ew=3
ct=2

fs=12

############MANUAL

xloc=1

y=dots6Lman
dotcolor='gray'
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor, label='WT')

y=dots7Lman
dotcolor='green'
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor, label='KD Driver 1')

y=dots8Lman
dotcolor='b'
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor, label='KD Driver 2')

y=manual
plt.bar(xloc, np.mean(y), yerr=sem(y), width=w, edgecolor='k', fill=False, linewidth=bwidth, capsize=cap, 
        error_kw={'elinewidth': ew, 'capthick': ct})

############Auto
xloc=2

y=dots6Lauto
dotcolor='gray'
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor)

y=dots7Lauto
dotcolor='green'
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor)

y=dots8Lauto
dotcolor='b'
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor)

y=auto
plt.bar(xloc, np.mean(y), yerr=sem(y), width=w, edgecolor='k', fill=False, linewidth=bwidth, capsize=cap, 
        error_kw={'elinewidth': ew, 'capthick': ct})
    
# plt.title('Overlap between Manual and Auto ROIs', fontsize=18, c='k', fontweight='bold')

plt.ylabel ('% Overlap \n Cortex Segmentation', fontsize=13.3, fontweight='bold')
# plt.ylim(top=10)

# plt.xlabel('Normalized to:', fontsize=18, c='k', fontweight='bold')
plt.xticks([1, 2], labels=['OL/Manual', 'OL/Auto'], color='k', fontweight='bold', fontsize=12)
plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='k', 
                labelsize=fs, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=13.3, length=ticklength, 
  width=tickwidth, color='k')  

# plt.legend(frameon=True, fontsize=fs, bbox_to_anchor=(1,1))

plt.tight_layout()

plt.savefig("SupFig1/ROI_best.pdf", transparent=True)

plt.show()
