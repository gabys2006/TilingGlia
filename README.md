# TilingGlia

This repo provides Python code for quantifying cortex glia morphology and aberrant astrocyte infiltration into the cortex in confocal images of drosophila larval VNC.

Please see the accompanying paper "Quantifying glial-glial tiling using automated image analysis" for a full description of the pipeline and its validation.

## Abstract:

Not only do glia form close associations with neurons throughout the central nervous system (CNS), but glial cells also interact closely with other glial cells.  As these cells mature, they undergo a phenomenon known as glial tiling, where they grow to abut one another without invading each other’s boundaries.  Glial tiling occurs throughout the animal kingdom, from fruit flies to humans; however, not much is known about the glial-glial interactions that lead to and maintain this tiling. Drosophila provide a strong model to investigate glial-glial tiling, as tiling occurs among individual glial cells of the same subtype, as well as between those of different subtypes.  Furthermore, the spatial segregation of the CNS allows for the unique ability to visualize and manipulate inter-subtype interactions.  Previous work in Drosophila has suggested an interaction between cortex glia and astrocytes, where astrocytes cross the normal neuropil-cortex boundary in response to dysfunctional cortex glia. Here, we further explore this interaction by implementing an automated pipeline to more fully characterize this astrocyte-cortex glial relationship. By quantifying and correlating the extent of cortex glial dysfunction and aberrant astrocyte infiltration using automated analysis, we maximize the size of the quantified dataset to reveal subtle patterns in astrocyte-cortex glial interactions.  We provide a guide for creating and validating a fully-automated image analysis pipeline for exploring these interactions, and describe a significant correlation between cortex glial dysfunction and aberrant astrocyte infiltration, as well as demonstrate variations in their relationship across different regions of the CNS.

## Automated image analysis pipeline for 3-channel confocal Z-stacks.

<img src="https://user-images.githubusercontent.com/57374720/150917981-079885c1-86ce-49e6-9d73-3a3cebb47583.png" width=700>  

Our image analysis pipeline consists of the following steps:

**Step 1:** Denoise individual channel images. 

**Step 2:** Threshold 2D arrays to produce binary images. 

**Step 3A:** Measure cortex glia (CG) perimeter, normalize to total cortex to produce automated morphology index (AMI). 

**Step 3B:** Combine cortex glia and neuronal nuclei channels to define the cortex area (ctx). Calcuate overlap between the cortex and astrocyte (astro) channels. Normalize to total cortex area to produce an automated infiltration score (AIS). 

**Step 4:** Analyze the relationship between AMI (globularity) and AIS (aberrant infiltration). 

## Files in this repo

### TilingPipelineFunctions.py
Python code containing all functions used in the image analysis pipeline. 

### Tutorial.pynb
Juypter notebook providing an explanation for using the automated pipeline. It allows for testing the denoising, thresholding and quantification steps using a small test dataset we have provided (see below for downloading). It also provides a short explanation for performing an analysis of the resulting scores (full code also provided in this repo). 

We performed optimization and/or validation for the denoising, thresholding, and quantifying steps of the pipeline. The troubleshooting sections for the denoising and thresholding steps include code used to optimize these steps and can be tested using the sample test data. The troubleshooting sections for the quantification step include a short explanation of the optimization and validation performed for this step (full code also provided in this repo). 

### Denoising_Optimization.py
Python code for optimizing denoising using the unsharp mask. 

### Neuron_Binarizer_Optimization.py
Python code for optimizing a binarization algorithm for the neuron channel. 

### Astrocyte_Binarizer_Optimization.py
Python code for optimizing a binarization algorithm for the astrocyte channel. 

### CortexGlia_Binarizer_Optimization.py
Python code for optimizing a binarization algorithm for the cortex glia channel. 

### Cortex_Segmentation_Optimization.py
Python code for optimizing the cortex segmentation function by comparing automated to manual segmentation. 

### AMI_Validation.py
Python code for validating globularity quantification by comparing manual globularity scores to AMI. 

### Cell_Body_Identification.py
Python code for labeling and measuring features of objects in preliminary infiltration. 

### AIS_Validation.py
Python code for validating AIS by comparing automated to manual infiltration quantification. 

### Z_Stack_Quantification.py
Python code for quantifying globularity and aberrant infiltration in 3-channel confocal z-stacks. 

### Global_Score_Analysis.py
Python code for analyzing global scores, the average of the scores for all slices in a z-stack. A global score represents the average score of the ventral nerve cord (VNC, part of the CNS) of a single animal. 

### Local_Score_Analysis.py
Python code for analyzing local scores, the average of the scores of a subset of slices in a z-stack. A local score represents the average score of a small region of the VNC of a single animal. 

## Istructions for running code on this repo:

### [Install Python](https://www.python.org/downloads/)
You can find a tutorial [here](https://realpython.com/installing-python/). 

### [Install Anaconda](https://docs.anaconda.com/anaconda/install/index.html)

### To create the Python environment for the tutorial and code on this repo enter the following commands into a terminal:

conda create --name Tiling python=3.7.3  
conda activate Tiling  
conda install jupyter  
conda install numpy  
conda install pandas  
conda install scikit-image  
conda install scipy
conda install -c conda-forge scikit-posthocs
pip install pylr2

### To run the tutorial:
Run a jupyter notebook instance in that environment by entering the following command into a terminal:

jupyter notebook

Download Tutorial.ipynb from this repo, navigate to it in Jupyter notebook, and run it. 

### To run Python scripts:
Download script and run directly from command line or use your favorite IDE. You can find a tutorial [here](https://realpython.com/run-python-scripts/)

## Test Data
Test data for the tutorial can be downloaded at [The Cell Image Library](http://cellimagelibrary.org/groups/54646). Dataset consists of 45 2D 3-channel tifs depicting the _Drosophila_ larval VNC stained for astrocytes, cortex glia and neuronal nuclei. 

## Questions

Please email questions to gabys2006@gmail.com
