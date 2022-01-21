# TilingGlia

The tutorial on this repo guides you through the automated pipeline for quantifying cortex glia morphology and aberrant astrocyte infiltration into the cortex in confocal images of drosophila larval VNC. The tutorial is a Jupyter notebook and the pipeline was created using Python.

Please see the accompanying paper "Quantifying glial-glial tiling using automated image analysis" for a full description of the pipeline.

**Abstract:**


Not only do glia form close associations with neurons throughout the central nervous system (CNS), but glial cells also interact closely with other glial cells.  As these cells mature, they undergo a phenomenon known as glial tiling, where they grow to abut one another without invading each other’s boundaries.  Glial tiling occurs throughout the animal kingdom, from fruit flies to humans; however, not much is known about the glial-glial interactions that lead to and maintain this tiling. Drosophila provide a strong model to investigate glial-glial tiling, as tiling occurs among individual glial cells of the same subtype, as well as between those of different subtypes.  Furthermore, the spatial segregation of the CNS allows for the unique ability to visualize and manipulate inter-subtype interactions.  Previous work in Drosophila has suggested an interaction between cortex glia and astrocytes, where astrocytes cross the normal neuropil-cortex boundary in response to dysfunctional cortex glia. Here, we further explore this interaction by implementing an automated pipeline to more fully characterize this astrocyte-cortex glial relationship. By quantifying and correlating the extent of cortex glial dysfunction and aberrant astrocyte infiltration using automated analysis, we maximize the size of the quantified dataset to reveal subtle patterns in astrocyte-cortex glial interactions.  We provide a guide for creating and validating a fully-automated image analysis pipeline for exploring these interactions, and describe a significant correlation between cortex glial dysfunction and aberrant astrocyte infiltration, as well as demonstrate variations in their relationship across different regions of the CNS.


![Fig2V10](https://user-images.githubusercontent.com/57374720/150566445-bb26d3c3-4974-4be6-a4bc-6ca077f941a3.png width="400" height="200")

## Instructions for setting up tutorial environment

To create the Python environment for this tutorial enter the following commands into a terminal:

conda create --name Tiling python=3.7.3  
conda activate Tiling  
conda install jupyter  
conda install numpy  
conda install pandas  
conda install scikit-image  
conda install pywget  

Run a jupyter notebook instance in that environment:

jupyter notebook

Download Automated_AMI_AIS_Quantification_Tutorial.ipynb from this repo, navigate to it in Jupyter notebook, and run it. 

## Test Data

Test data for the tutorial can be downloaded at The Cell Image Library http://www.cellimagelibrary.org/home or you can run the first cell of the tutorial to download it for you. 
