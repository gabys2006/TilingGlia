#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 23:47:19 2021

@author: gabrielasalazar

DETERMINE LOCAL AIS/AIM SCORES USING A SLIDING WINDOW
ANALYZE LOCAL AIS/AIM AND FIND THEIR RELATIONSHIP

"""

import pandas as pd 
import numpy as np
from scipy.stats import sem, spearmanr
import matplotlib.pyplot as plt


#%%LOAD DATA 
#these data are AMI/AIS scores for full zstacks quantified using Z_Stack_Quantification.py 
#also included is z_norm = slice_number/total slices in stack indicating the z-coordinate

path='20210531withBrainIDswithZ.csv'
measurements=pd.read_csv(path, index_col=False)
col=['sliceID', 'zNorm', 'GT', 'brainID', 'AMI', 'AIS']
measurements.columns=col
print(measurements.columns)   


#%%FIND LCOCAL SCORES BY AVERAGING SUBSET OF SLICES
#isolate scores per per genotype/experimental group
WT=measurements.loc[measurements.GT=='6L']
Spz3=measurements.loc[measurements.GT=='7L']
aSNAP=measurements.loc[measurements.GT=='8L']

#window size
win_size=0.1
z_0=0
z_final=1-win_size

WT_z=[]
Spz3_z=[]
aSNAP_z=[]

#iterate through subsets
for i in np.arange(0, 100-win_size*100+1):
    z_end=round(z_0+win_size, 2)
    #calculate local scores and SEM
    brainWT_z = WT.loc[(WT.zNorm>z_0) & (WT.zNorm<=z_end)]
    brainWT_z = brainWT_z.groupby(['brainID', 'GT'], as_index=False).mean()
    gsWT_z_mean = np.mean(brainWT_z.AMI)
    gsWT_err = sem(brainWT_z.AMI)
    isWT_z_mean = np.mean(brainWT_z.AIS)
    isWT_err = sem(brainWT_z.AIS)
    WT_z.append([z_0, z_end, gsWT_z_mean, gsWT_err, isWT_z_mean, isWT_err])
    
    brainSpz3_z = Spz3.loc[(Spz3.zNorm>z_0) & (Spz3.zNorm<=z_end)]
    brainSpz3_z = brainSpz3_z.groupby(['brainID', 'GT'], as_index=False).mean()
    gsSpz3_z_mean = np.mean(brainSpz3_z.AMI)
    gsSpz3_err = sem(brainSpz3_z.AMI)
    isSpz3_z_mean = np.mean(brainSpz3_z.AIS)
    isSpz3_err = sem(brainSpz3_z.AIS)
    Spz3_z.append([z_0, z_end, gsSpz3_z_mean, gsSpz3_err, isSpz3_z_mean, isSpz3_err])
    
    brainaSNAP_z = aSNAP.loc[(aSNAP.zNorm>z_0) & (aSNAP.zNorm<=z_end)]
    brainaSNAP_z = brainaSNAP_z.groupby(['brainID', 'GT'], as_index=False).mean()
    gsaSNAP_z_mean = np.mean(brainaSNAP_z.AMI)
    gsaSNAP_err = sem(brainaSNAP_z.AMI)
    isaSNAP_z_mean = np.mean(brainaSNAP_z.AIS)
    isaSNAP_err = sem(brainaSNAP_z.AIS)
    aSNAP_z.append([z_0, z_end, gsaSNAP_z_mean, gsaSNAP_err, isaSNAP_z_mean, isaSNAP_err])
    
    z_0=round(z_0+0.01, 2)
    
WT_z=pd.DataFrame(WT_z, columns=['z_0', 'z_end', 'globScoreMean', 'globScoreErr', 'infScoreMean', 'infScoreErr'])
Spz3_z=pd.DataFrame(Spz3_z, columns=['z_0', 'z_end', 'globScoreMean', 'globScoreErr', 'infScoreMean', 'infScoreErr'])
aSNAP_z=pd.DataFrame(aSNAP_z, columns=['z_0', 'z_end', 'globScoreMean', 'globScoreErr', 'infScoreMean', 'infScoreErr'])

#%%GRAPH AMI

ticklength=4
tickwidth=2

plt.figure(figsize=(5, 5), facecolor='w')

x=(WT_z.z_0+win_size/2)*100

a=0.3
ms=3
plt.plot(x, WT_z.globScoreMean, marker='o', label='Control', c='gray', markersize=ms)
plt.fill_between(x, WT_z.globScoreMean-WT_z.globScoreErr, WT_z.globScoreMean+WT_z.globScoreErr, color='gray', alpha=a)

plt.plot(x, Spz3_z.globScoreMean, marker='o', label='KD Driver 1', c='green', markersize=ms)
plt.fill_between(x, Spz3_z.globScoreMean-Spz3_z.globScoreErr, Spz3_z.globScoreMean+Spz3_z.globScoreErr, color='green', alpha=a)

plt.plot(x, aSNAP_z.globScoreMean, marker='o', label='KD Driver 2', c='blue', markersize=ms)
plt.fill_between(x, aSNAP_z.globScoreMean-aSNAP_z.globScoreErr, aSNAP_z.globScoreMean+aSNAP_z.globScoreErr, color='blue', alpha=a)

plt.title('Globularity along Z-axis', fontsize=16, c='k', fontweight='bold')

plt.ylabel('Globularity \n (100* perimeter/cortex area)', fontsize=14, c='k', fontweight='bold')
plt.ylim(top=12)
plt.xlim(left=5, right=95)

plt.xlabel('% total brain depth \n (along z-axis)', fontsize=14, c='k', fontweight='bold')

plt.xticks(np.arange(10, 101, 10), labels=['10 \n ventral', '20', '30', '40', '50', '60', '70', '80', '90', '100 \n dorsal'],
            fontweight='bold', fontsize=14)
plt.xticks(np.arange(10, 100, 10), fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)
plt.tick_params(axis='x', which='both',  labelbottom=True, color='k', labelsize=12, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=12, length=ticklength, width=tickwidth)

plt.legend(frameon=False, prop=dict(weight='bold', size=12))

plt.show()

#%%GRAPH AIS

ticklength=4
tickwidth=2
a=0.3

plt.figure(figsize=(5, 5), facecolor='w')

x=(WT_z.z_0+win_size/2)*100

plt.plot(x, WT_z.infScoreMean, marker='o', label='Control', c='gray', markersize=ms)
plt.fill_between(x, WT_z.infScoreMean-WT_z.infScoreErr, WT_z.infScoreMean+WT_z.infScoreErr, color='gray', alpha=a)

plt.plot(x, Spz3_z.infScoreMean, marker='o', label='KD Driver 1', c='green', markersize=ms)
plt.fill_between(x, Spz3_z.infScoreMean-Spz3_z.infScoreErr, Spz3_z.infScoreMean+Spz3_z.infScoreErr, color='green', alpha=a)

plt.plot(x, aSNAP_z.infScoreMean, marker='o', label='KD Driver 2', c='blue', markersize=ms)
plt.fill_between(x, aSNAP_z.infScoreMean-aSNAP_z.infScoreErr, aSNAP_z.infScoreMean+aSNAP_z.infScoreErr, color='blue', alpha=a)

plt.title('Infiltration along Z-axis', fontsize=16, c='k', fontweight='bold')

plt.ylabel('Infiltration \n(% cortex area)', fontsize=14, c='k', fontweight='bold')
plt.ylim(top=15)

plt.xlabel('% total brain depth \n (along z-axis)', fontsize=14, c='k', fontweight='bold')

plt.xticks(np.arange(10, 101, 10), labels=['10 \n ventral', '20', '30', '40', '50', '60', '70', '80', '90', '100 \n dorsal'], 
            fontweight='bold', fontsize=14)
plt.xlim(left=5, right=95)
plt.xticks(np.arange(10, 100, 10), fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)
plt.tick_params(axis='x', which='both',  labelbottom=True, color='k', labelsize=12, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=12, length=ticklength, width=tickwidth)

plt.legend(frameon=False, prop=dict(weight='bold', size=12))

plt.show()

#%%FIND RELATIONSHIP BETWEEN LOCAL SCORES

#spearman correlations
#window size
win_size=0.1
z_0=0
z_final=1-win_size

z_corr=[]

for i in np.arange(0, 100-win_size*100+1):
    z_end=round(z_0+win_size, 2)
    
    z_measurements = measurements.loc[(measurements.zNorm>z_0) & (measurements.zNorm<=z_end)]   
    z_measurements = z_measurements.groupby(['brainID', 'GT'], as_index=False).mean()
    
    x=z_measurements.AMI
    y=z_measurements.AIS
    
    rho, p = spearmanr(x, y)    
    z_corr.append([z_0, z_end, rho, p])
    
    z_0=round(z_0+0.01, 2)
    
z_corr=pd.DataFrame(z_corr, columns=['z_0', 'z_end', 'rho', 'p'])

#%%FIND WHERE RELATIONSHIP IS SIGNIFICANT TO INDICATE IT ON GRAPH

#find where p<0.05
sig_p=0.05
sig_corr=z_corr[z_corr.p>sig_p]
z_sig_min=sig_corr.z_0.min()*100+win_size/2
z_sig_max=sig_corr.z_0.max()*100+win_size/2
print(sig_corr)
print(z_sig_min)
print(z_sig_max)


#%%Spearman correlations w/ p

ticklength=4
tickwidth=2

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5.5,5))

#rho
color='k'
x=(z_corr.z_0+win_size/2)*100
y=z_corr.rho
ax1.plot(x, y, color,  marker='o', markersize=ms)
ax1.set_ylabel("Spearman's Rank \n"  + r'Correlation Coefficient ($\rho$)', color=color, fontweight='bold', fontsize=14)
ax1.tick_params(axis='y', labelcolor=color, which='both', labelsize=12, length=ticklength, width=tickwidth)
plt.yticks(fontweight='bold')
plt.xticks(np.arange(10, 101, 10), labels=['10 \n ventral', '20', '30', '40', '50', '60', '70', '80', '90', '100 \n dorsal'], 
            fontweight='bold', fontsize=14)
plt.tick_params(axis='x', which='both',  labelbottom=True, color='k', labelsize=12, length=ticklength, width=tickwidth)
plt.xlabel('% total brain depth \n (along z-axis)', fontsize=14, c='k', fontweight='bold')
plt.xlim(left=0, right=100)

#fill in yellow the part that's significant
x1, x2=ax1.get_xlim()
y1, y2=ax1.get_ylim()
c='yellow'
a=0.15
ax1.fill_between([x1, z_sig_min], [y1, y1], [y2, y2], color=c, alpha=a)
ax1.fill_between([z_sig_max, x2], [y1, y1], [y2, y2], color=c, alpha=a)
# ax1.fill_between([70, 95], [-0.01, -0.01], [0.5, 0.5], color='gray', alpha=0.2)
ax1.set_ylim(bottom=y1, top=y2)
ax1.set_xlim(left=5, right=95)
plt.xticks(np.arange(10, 100, 10), fontweight='bold', fontsize=14)

plt.title('Spearman Correlation \n along Z-axis', fontsize=16, c='k', fontweight='bold')
plt.yticks(fontweight='bold')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

