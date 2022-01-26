#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 00:49:15 2022

@author: gabrielasalazar

FOR OPTIMIZING CORTEX GLIA CHANNEL BINARIZER

"""
#%%

from TilingPipelineFunctions import try_all_threshold_on_set, cortexglia_binarizer_optimization

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