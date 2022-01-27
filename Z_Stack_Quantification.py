#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 02:18:09 2021

@author: gabrielasalazar


QUANTIFY AMI AND AIS FOR ALL SLICES IN A 3-CHANNEL Z-STACK

"""

#%%
import os
import csv
import skimage
import numpy as np

from TilingPipelineFunctions import quantify_slice

#%%
dirName='DATA'

recordDir='InfRuns'
newRun = 'NewRun'

newRunPath=os.path.join(recordDir, newRun)
newRunPathcsv=os.path.join(newRunPath, newRun+'.csv')

#make dir for this run
if not os.path.exists(newRunPath):
    os.makedirs(newRunPath)

#make csv to store data as it's gathered
with open(newRunPathcsv, "w", newline='') as rec:
    writer = csv.writer(rec, delimiter=',')
    writer.writerow(['GT', 'sliceID', 'brainID', 'AMI', 'AIS'])
    rec.close()

#genotype codes
#present in every z-stack name
#denotes which experimental group zstack belongs to 
GTs=['6L', '7L', '8L', '11L']

#store any files with errors in this list
errors=[]

#iterate through all files
for root, directory, file in os.walk(dirName):
    for d in directory[:]:
        path=os.path.join(root, d)
        for Root, Directory, File in os.walk(path):  
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
                    #for each slice
                    zlen=np.shape(fullimg)[0]
                    for s in range(zlen):
                        #get sliceID
                        sliceID=f.replace('.tif', '')
                        sliceID=sliceID+'_'+ str(s)+'.tif'
                        imgslice=fullimg[s, :, :, :]
                        #quantify AIM and AIS for each slice
                        GT, sliceID, AMI, AIS = quantify_slice(fullimg, sliceID)
                        #save measurements
                        measurements=[GT, sliceID, brainID, AMI, AIS]
                        with open(newRunPathcsv, 'a', newline='') as rec:
                            writer = csv.writer(rec, delimiter=',')
                            writer.writerow(measurements)
                            rec.close()
                except:
                    errors.append(f)

if len(errors)>0:
    print('\n there are errors with:')
    for e in errors:
        print(errors)
else:
    print('no errors')


