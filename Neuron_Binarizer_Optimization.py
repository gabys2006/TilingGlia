#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 00:49:15 2022

@author: gabrielasalazar

FOR OPTIMIZING NEURON CHANNEL BINARIZER

"""

from TilingPipelineFunctions import try_all_threshold_on_set, neuron_binarizer_optimization

#%%# RUN TRY_ALL_THRESHOLD_ON_SET ON ALL IMAGES IN TEST SET

dataDir='TestImages'
resultsDir = 'NeuronThresh'
channel = 2

test = try_all_threshold_on_set (dataDir, resultsDir, channel)
print(test)

#%%# RUN NEURON_BINARIZER_OPTIMIZATION ON ALL IMAGES IN TEST SET

dataDir='TestImages'
resultsDir = 'NeuronBinarizerRes'
channel = 2

test = neuron_binarizer_optimization (dataDir, resultsDir, channel)
print(test)