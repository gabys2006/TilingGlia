#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 02:18:09 2021

@author: gabrielasalazar

OPTIMIZE EXCLUSION OF CELL BODIES

LABEL ALL OBJECTS IN PRELIMINARY INFILTRATION AND MEASURE SOLIDITY, ECCENTRICITY AND SIZE 
TO DETERMINE WHICH TO EXCLUDE FROM FINAL INFILTRATION  COUNT

CREATE CSV INCLUDING:
    IMAGEID, OBJECTID, COLORCODE, STATUS, AREA, SOLIDITY, ECCENTRICITY
    IMAGEID: UNIQUE IMAGE ID (NAME OF IMAGE)
    OBJECT: ID FOR OBJECT/REGION ON IMAGE
    COLORCODE: OBJECTS ARE COLORED BY AREA
    AREA:OBJECT AREA
    SOLIDITY: OBJECT SOLIDITY (# TRUE PIXELS/# PIXELS)
    ECCENTRICITY: OBJECT ECCENTRICITY (MEASURE OF ROUNDNESS)

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

from TilingPipelineFunctions import binarize_neurons, binarize_cortex_glia, binarize_astro, segment_cortex

#%% CREATE PRELIMINARY INFILTRATION FIGURES

dirName='TestImages'

ncgC=0
cgC=1
nC=2

NCG_type=0 #for astro
exludedParticles=[]

recordDir='TestRun'
newRun = 'PrelimInf'

newRunPath=os.path.join(recordDir, newRun)
newRunPathcsv=os.path.join(newRunPath, newRun+'.csv')

if not os.path.exists(newRunPath):
    os.makedirs(newRunPath)
    
figD=newRunPath

#store filenames with issues
errors=[]
for root, directory, file in os.walk(dirName):
    for f in file[:3]:
        if 'DS_Store' not in f:
            try:
                #load image 
                ipath=os.path.join(root, f)
                fullimg=skimage.io.imread(ipath, plugin="tifffile")
                fullimg=img_as_float(fullimg)
                
                #binarize neurons
                neuronBIN, nthresh = binarize_neurons(fullimg)
                #binarize cg
                cgBIN, cgthresh = binarize_cortex_glia(fullimg)
                #binarize astrocytes
                ncgBIN, ncgthresh = binarize_astro (fullimg)
                
                #segment cortex
                ctxBINfinal = segment_cortex (cgBIN, neuronBIN)
                
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
            except:
                errors.append (f)
        
print (errors)
            

#%%LABEL OBJECTS IN PRELIMIINARY INFILTRATION 

#set filepath for prelim inf images
recordDir='TestRun'
newRun = 'PrelimInf'

newRunPath=os.path.join(recordDir, newRun)
dirName=newRunPath

#create dir for new labeled images
recordDir='TestRun'
newRun = 'LabeledInf'
newRunPath=os.path.join(recordDir, newRun)
if not os.path.exists(newRunPath):
    os.makedirs(newRunPath)
figD = newRunPath

#iterate through preliminary infiltration images
for root, directory, file in os.walk(dirName):
    for f in file:
        if 'DS_Store' not in f:
            #load image
            ipath=os.path.join(root, f)
            prelimInf=skimage.io.imread(ipath, plugin="tifffile")
            prelimInf=img_as_float(prelimInf)
            
            #make new array to store final infiltration (w/o cell bodies)
            finalinf=np.copy(prelimInf)
            
            #get rid of cell bodies
            label_inf=label(prelimInf)
            rprops= regionprops(label_inf)        
            
            #create figure to show labeled objects on prelim infiltration 
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax = axes.ravel()
            plt.gray()
            
            #add preliminary infiltration to figure
            ax[0].imshow(prelimInf)
            ax[0].set_title(f + "\n Preliminary Inf")
            
            #objects to be excluded will be bounded in red box
            excludeColor='red'
            
            #color objects by size 
            #"small" objects get labeled one color 
            s_x=[]
            s_y=[]
            
            #"medium" objects get labeled diff color 
            m_x=[]
            m_y=[]
            
            #"large" objects get labeled diff color 
            l_x=[]
            l_y=[]
            
            #set ID for each object starting at 1
            prop_measurementID=1
            
            #will print ID next to obejct on image
            #will scoot the number over by a little 
            px=1250
            py=0
            dy=60
            
            #iterate thru regions/objects in image
            for region in rprops:
                ePprops=[]
                ePprint=[]
                
                #set cutoff for small objects
                cbod_area1=500
                cbod_area2=1500
                if region.area > cbod_area1 and region.area <= cbod_area2:
                    #initialize status as kept
                    status='kept'
                    #set colorcode
                    colorCode='magenta'
                    #add coordinates to colorcode object on figure 
                    for c in region.coords:
                        s_y.append (c[0])
                        s_x.append (c[1])
                    #exclude if fit following cutoffs
                    ecc1=0.9
                    if region.eccentricity < ecc1:
                        #add red box around excluded objects
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1)   
                        ax[0].add_patch(rect)
                        #reset status to excluded since it falls within exclusion criteria
                        status='excluded'
                        #delete for final infiltration 
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    solid1=0.5
                    if region.solidity > solid1:
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1)   
                        ax[0].add_patch(rect)
                        status='excluded'
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    #store data for csv and to print on image
                    ePprint=[prop_measurementID, status, region.area, str(round(region.solidity, 3)), str(round(region.eccentricity,3))]
                    ePprops=[f, prop_measurementID, colorCode, status, region.area, region.solidity, region.eccentricity]
                #medium objects    
                cbod_area1=1500
                cbod_area2=5000
                if region.area > cbod_area1 and region.area<=cbod_area2:
                    colorCode='cyan'
                    status='kept'
                    for c in region.coords:
                        m_y.append (c[0])
                        m_x.append (c[1])
                    solid2=0.55
                    if region.solidity>solid1:
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1) 
                        ax[0].add_patch(rect)
                        status='excluded'
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    solid3=0.45
                    ecc2 = 0.8
                    if region.solidity>solid1 and region.eccentricity>ecc1:
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1) 
                        ax[0].add_patch(rect)
                        status='excluded'
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    ePprint=[prop_measurementID, status, region.area, str(round(region.solidity, 3)), str(round(region.eccentricity,3))]
                    ePprops=[f, prop_measurementID, colorCode, status, region.area, region.solidity, region.eccentricity]
                #large objects
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
                                
            #color objects                        
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
            
            #save image
            picID=f.replace('prelimInf.tif', '')
            figpath=os.path.join(figD, f + 'numbered' + '.png')
            plt.savefig(figpath, format='png', pad_inches=0.3, bbox_inches='tight')
            #suppress output
            plt.close() 

#store data in csv
col=['PicID', 'RegionID', 'Color_Code', 'Status', 'Area', 'Solidity', 'Eccentricity']
excludedParticles=pd.DataFrame(exludedParticles, columns=col)
recordDir='TestRun'
newRun = 'PrelimInf'
newRunPathcsv=os.path.join(newRunPath, newRun+'.csv')
excludedParticles.to_csv(newRunPathcsv, index=False)
    
#%%LABEL OBJECTS IN PRIMARY INFILTRATION 

recordDir='LabeledI'
newRun = 'LabeledInf'

newRunPath=os.path.join(recordDir, newRun)
newRunPathcsv=os.path.join(newRunPath, newRun+'.csv')

if not os.path.exists(newRunPath):
    os.makedirs(newRunPath)
    
figD=newRunPath

GTs=['1BL', '2BL', '6L', '7L', '8L']

#store data for csv 
exludedParticles=[]

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
                    ecc1=0.9
                    if region.eccentricity < ecc1:
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1)   
                        ax[0].add_patch(rect)
                        status='excluded'
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    solid1=0.5
                    if region.solidity > solid1:
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
                    solid2=0.55
                    if region.solidity>solid1:
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor=excludeColor, linewidth=1) 
                        ax[0].add_patch(rect)
                        status='excluded'
                        for c in region.coords:
                            finalinf[c[0], c[1]]=0
                    solid3=0.45
                    ecc2 = 0.8
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
            
#store data in csv
col=['PicID', 'RegionID', 'Color_Code', 'Status', 'Area', 'Solidity', 'Eccentricity']
excludedParticles=pd.DataFrame(exludedParticles, columns=col)

csvpath = 
excludedParticles.to_csv('InfTweaks/labeledParts.csv', index=False)
  
            
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
col=['PicID', 'RegionID', 'Color_Code', 'Status', 'Area', 'Solidity', 'Eccentricity']
excludedParticles=pd.DataFrame(exludedParticles, columns=col)

excludedParticles.to_csv('InfTweaks/labeledParts.csv', index=False)
