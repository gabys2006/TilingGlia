#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:31:12 2021

@author: gabrielasalazar

ANALYZE GLOBAL AIS/AIM AND FIND THEIR RELATIONSHIP

"""

#%%
import pandas as pd
import numpy as np
from scipy.stats import kruskal, sem, spearmanr
import scikit_posthocs as sp
import matplotlib.pyplot as plt

#%%LOAD DATA 
#these data are AMI/AIS scores for full zstacks quantified using Z_Stack_Quantification.py 

path='2021_05_31.csv'
measurements=pd.read_csv(path, index_col=False)
col=['GT', 'sliceID', 'brainID', 'AMI', 'AIS']
measurements.columns=col
print(measurements.columns)   


#%%CALCULATE GLOBAL SCORES
#collapse by brain ID to calculate average scores per animal
byBrain=measurements.groupby(['brainID', 'GT'], as_index=False).mean()

#%%ISOLATE AMI AND AIS SCORES BY EXPERIMENTAL GROUP
#isolate control scores
WT=byBrain.loc[byBrain.GT=='6L']
#isolate KD1
Spz3=byBrain.loc[byBrain.GT=='7L']
#isolate KD2
aSNAP=byBrain.loc[byBrain.GT=='8L']

#%%RUN STATISTICS ON TO COMPARE DISTRIBUTIONS

#calculate means
print('AMI Means')
print(np.average(WT.AMI))
print(np.average(Spz3.AMI))
print(np.average(aSNAP.AMI))
print()

#Kruskal-Wallis test, followed by pairwise Dunn comparisons
H, pKW =kruskal(WT.AMI, Spz3.AMI, aSNAP.AMI)
print('pKW', pKW)
iScore=[WT.AMI, Spz3.AMI, aSNAP.AMI]
dunn=sp.posthoc_dunn(iScore)
print(dunn)
print()

#calculate means
print('AIS Means')
print(np.average(WT.AIS))
print(np.average(Spz3.AIS))
print(np.average(aSNAP.AIS))
print()

#Kruskal-Wallis test, followed by pairwise Dunn comparisons
H, pKW =kruskal(WT.AIS, Spz3.AIS, aSNAP.AIS)
print('pKW', pKW)
iScore=[WT.AIS, Spz3.AIS, aSNAP.AIS]
dunn=sp.posthoc_dunn(iScore)
print(dunn)
print()

#%%PLOT DISTRIBUTIONS FOR *******AMI**************

tickfont=16
ticklength=6
tickwidth=3
lw=3

plt.figure(figsize=(6, 8), facecolor='w')

data=[WT.AMI, Spz3.AMI, aSNAP.AMI]
dx=0.7
xs=[1, 1+dx, 1+2*dx]

parts=plt.violinplot(data, xs, showmeans=False, showextrema=False)

colors=['gray', 'green', 'b']
#set violin colors
for pc, c in zip(parts['bodies'], colors):
    pc.set_facecolor(c)
    pc.set_edgecolor(c)
    pc.set_alpha(1)
    
#mean and error
widths=[0.23, 0.24, 0.24]

for x, scores, dx, in zip(xs, data, widths):
    plt.hlines(np.mean(scores), x-dx, x+dx, color='k', linewidth=lw)
    plt.errorbar(x, np.mean(scores), yerr=sem(scores), elinewidth=3, capsize=8, ecolor='k')
    
#sig
sigloc=np.max(WT.AMI)
dx=0.1
dy=1
plt.hlines(sigloc+dy, xs[0], xs[1], colors='k')     
plt.text((xs[0]+xs[1])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'})  

sigloc=np.max(Spz3.AMI)
dx=0.1
dy=1
plt.hlines(sigloc+dy, xs[1], xs[2], colors='k')     
plt.text((xs[1]+xs[2])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'}) 

sigloc=np.max(WT.AMI)
dx=0.2
dy=3
plt.hlines(sigloc+dy, xs[0], xs[2], colors='k')     
plt.text((xs[0]+xs[2])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'}) 
    
plt.title('Globularity \n Avg by Brain', fontsize=18, c='k', fontweight='bold')

plt.ylabel('perimeter \n (% cortex area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(top=20)

plt.xticks([1, 2, 3], labels=['Control', 'KD \n Driver 1', 'KD \n Driver 2'], color='k', fontweight='bold')
plt.yticks(np.arange(0, 21, 2), color='k', fontweight='bold')
plt.xticks(fontweight='bold')
plt.tick_params(axis='x', which='both', color='k', labelsize=tickfont, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  


plt.show()

#%%PLOT DISTRIBUTIONS FOR *******AIS**************

tickfont=16
ticklength=6
tickwidth=3
lw=3

plt.figure(figsize=(6, 8), facecolor='w')

data=[WT.AIS, Spz3.AIS, aSNAP.AIS]
dx=0.7
xs=[1, 1+dx, 1+2*dx]

parts=plt.violinplot(data, xs, showmeans=False, showextrema=False)

colors=['gray', 'green', 'b']
#set violin colors
for pc, c in zip(parts['bodies'], colors):
    pc.set_facecolor(c)
    pc.set_edgecolor(c)
    pc.set_alpha(1)
    
#mean and error
widths=[0.23, 0.24, 0.24]

for x, scores, dx, in zip(xs, data, widths):
    plt.hlines(np.mean(scores), x-dx, x+dx, color='k', linewidth=lw)
    plt.errorbar(x, np.mean(scores), yerr=sem(scores), elinewidth=3, capsize=8, ecolor='k')
    
#sig
sigloc=np.max(Spz3.AIS)
dx=0.1
dy=1
plt.hlines(sigloc+dy, xs[0], xs[1], colors='k')     
plt.text((xs[0]+xs[1])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'})  

sigloc=np.max(aSNAP.AIS)
dx=0.1
dy=1
plt.hlines(sigloc+dy, xs[1], xs[2], colors='k')     
plt.text((xs[1]+xs[2])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'}) 

sigloc=np.max(aSNAP.AIS)
dx=0.2
dy=3
plt.hlines(sigloc+dy, xs[0], xs[2], colors='k')     
plt.text((xs[0]+xs[2])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'}) 
    
plt.title('Aberrant Infiltration \n Avg by Brain', fontsize=18, c='k', fontweight='bold')

plt.ylabel('infiltration \n (% cortex area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(top=20)

plt.xticks([1, 2, 3], labels=['Control', 'KD \n Driver 1', 'KD \n Driver 2'], color='k', fontweight='bold')
plt.yticks(np.arange(0, 21, 2), color='k', fontweight='bold')
plt.xticks(fontweight='bold')
plt.tick_params(axis='x', which='both', color='k', labelsize=tickfont, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  


plt.show()

#%%FIND RELATIONSHIP BETWEEN AMI AND AIS 

#spearman correlatin
x=byBrain.AMI
y=byBrain.AIS

rho, p =spearmanr(x, y)
print(rho)
print(p)

#%%GRAPH CORRELATION

# bwidth=0.27
# cap=5
# tickfont=16
# ticklength=4
# tickwidth=2
# barlw=2
# ms=60

plt.figure(figsize=(8, 8), facecolor='w')

WTGS=WT.AMI
Spz3GS=Spz3.AMI
aSNAPGS=aSNAP.AMI

WTIS=WT.AIS
Spz3IS=Spz3.AIS
aSNAPIS=aSNAP.AIS

dotsize=60
plt.scatter(WTGS, WTIS, c='gray', s=dotsize, label='Control')
plt.scatter(Spz3GS, Spz3IS, c='green', s=dotsize, label='KD Driver 1')
plt.scatter(aSNAPGS, aSNAPIS, c='blue', s=dotsize, label='KD Driver 1')

plt.title('Globularity-Infiltration \n Correlation', fontsize=18, c='k', fontweight='bold')
plt.ylabel('Infiltration \n (% cortex area)', fontsize=18, c='k', fontweight='bold')
plt.xlabel('Globularity \n (100*perimeter/cortex area)', fontsize=18, c='k', fontweight='bold')

tx=14
ty=5.5
dy=-0.3

plt.text(tx, ty+dy, r'$\rho$ = ' + str(round(rho, 3)), fontsize=12)
plt.text(tx, ty+2*dy, 'p < 0.0001', fontsize=12)


plt.xticks(color='k', fontweight='bold')
plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='both', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  
plt.xlim(left=15, right=0)

plt.legend(frameon=False, loc='upper left')

plt.show()


