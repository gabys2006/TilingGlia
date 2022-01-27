#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 02:19:58 2021

@author: gabrielasalazar

FOR CORTEX SEGMENTATION OPTMIZATION 
TEST SEVERAL PARAMETER SETS
COMPARE WITH MANUAL CORTEX SEGMENTATION 
CALCULATE TWO PERFORMANCE METRICS:
    OVERLAP/AUT0MATED SEGMENTATION (AKIN TO A TRUE POSTIVE RATE)
    OVERLAP/MANUAL SEGMENTATION (AKIN TO A FALSE NEGATIVE RATE)
"""

#%%
import os
import csv
import skimage
from skimage.util import img_as_float
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from TilingPipelineFunctions import segment_cortex, binarize_neurons, binarize_cortex_glia
            
#%%PARAMETER OPTIMIZATION

#Optimizing 4 parameters
#neighborhood for connect-the-dots
ns=[5, 10, 15, 25]
#threshold
threshes=[0.03, 0.06, 0.1]
#size of small holes
holes=[1000, 5000, 10000, 15000]
#number of final erosions
erosions=[5, 10, 15, 25]

#how many combinations?
combos=len(ns)*len(threshes)*len(holes)*len(erosions)
print (combos)

#%%COMPARE AUTOMATED AND MANUAL SEGMENTATIONS

#test image directory 
testData='ValidationImages'

#create results directory
newRun = 'TestRun'
newRunDir=os.path.join(testData, newRun)
if not os.path.exists(newRunDir):
    os.makedirs(newRunDir)

#create csv to write data as it is gathered
newRunPathcsv=os.path.join(newRun, newRun+'.csv')
with open(newRunPathcsv, "w", newline='') as rec:
    writer = csv.writer(rec, delimiter=',')
    col=['GT', 'sliceID', 'n', 'thresh', 'hole_size', 'erosions', 'manual_overlap', 'auto_overlap']
    writer.writerow(col)
    rec.close()
    
#genotype codes
GTs=['6L', '7L', '8L']

#store files with errors 
errors=[]
for root, directory, file in os.walk(testData):    
    for f in file[:]:
        if '.DS_Store' not in f:  
            #get GT
            for gt in GTs:
                if gt in f:
                    GT = gt
                    
            # load manual binary ctx ##################################
            #binary ctx image folder 
            mroot='CtxROIVal/BinaryCtxManual'
            mpath=os.path.join(mroot, f)
            try:
                manCtx=skimage.io.imread(mpath, plugin="tifffile")
                manCtx=img_as_float(manCtx)
                #measure area
                manual_area=np.sum(manCtx.astype(float))
            except:
                errors.append(f)
                #if there is an error skip this one 
                continue
            
            # auto binary ctx ########################################
            # load image
            ipath=os.path.join(root, f)
            fullimg=skimage.io.imread(ipath, plugin="tifffile")
            fullimg=img_as_float(fullimg)
            
            #binarize neuron channel 
            neuronBIN, nthresh = binarize_neurons (fullimg)
            
            #binarize cortex glia 
            cgBIN, cgthresh = binarize_cortex_glia(fullimg)
            
            #now hyperparameter optimization 
            #4 loops
            for nval in ns:
                for threshval in threshes:
                    for holeval in holes:
                        for eroval in erosions:
                            ctxBINfinal = segment_cortex (cgBIN, neuronBIN, 
                                                          n=nval, threshold=threshval, 
                                                          ctxholes=holeval, ctxer=eroval)
                            #measure area
                            auto_area=np.sum(ctxBINfinal.astype(float))
            
                            #############get overlap##############
                            overlap=ctxBINfinal+manCtx
                            overlap=overlap-1
                            overlap=np.clip(overlap, 0, 1)
                            overlap=overlap.astype(int)
                            manual_overlap=np.sum(overlap)/manual_area*100
                            auto_overlap=np.sum(overlap)/auto_area*100
                          
                            #write data on csv
                            measurements=[GT, f, nval, threshval, holeval, eroval, manual_overlap, auto_overlap]
                            with open(newRunPathcsv, 'a', newline='') as rec:
                                writer = csv.writer(rec, delimiter=',')
                                writer.writerow(measurements)
                                rec.close()


#%%LOAD DATA
ctxMeasurements=pd.read_csv(newRunPathcsv, index_col=False)

print (ctxMeasurements.columns)

#%%GROUP BY PARAMETER COMBOS TO GET AVERAGE SCORES PER COMBO
key=['n', 'thresh', 'hole_size', 'erosions']
OLscores=ctxMeasurements[['n', 'thresh', 'hole_size', 'erosions', 'manual_overlap', 'auto_overlap']]
OLscores=OLscores.groupby(key, as_index=False).mean()

#%%GRAPH AVERAGE SCORES
tickfont=12
ticklength=4
tickwidth=2

matplotlib.rcParams['pdf.fonttype'] = 42

plt.figure(figsize=(3.2, 3), facecolor='w')

y=OLscores.manual_overlap
x=np.arange(len(y))
plt.plot(x, y, c='gray', label='manual', linewidth=1)

y=OLscores.auto_overlap
plt.plot(x, y, c='skyblue', label='auto', linewidth=1)
y1, y2 = plt.ylim()
plt.vlines(133, y1, y2, linewidth=1.5, color='r')

plt.legend(fontsize=12, prop={'weight': 'bold'}, frameon=False)

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


#%%DETERMINE THE BEST PARAMETER SET BY CHOOSING THE ONE WITH THE HIGHEST SCORES
#chose these bottom threshold from graphs
bestCombos=OLscores[(OLscores.manual_overlap>85) & (OLscores.auto_overlap>85)]
print(bestCombos)

bestCombo=OLscores[(OLscores.manual_overlap>88) & (OLscores.auto_overlap>85)]
print(bestCombo)

