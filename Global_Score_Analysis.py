#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:31:12 2021

@author: gabrielasalazar
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import csv
import skimage
import scipy
from scipy import stats
import sklearn
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import f_oneway, sem, kruskal
import statsmodels.stats.multicomp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib
import re
import pandas as pd
import scikit_posthocs as sp
from matplotlib.ticker import FormatStrFormatter
from pylr2 import regress2


#%%load full data
path='InfRuns/2021_05_31/2021_05_31.csv'
measurements=pd.read_csv(path, index_col=False)

col=['GT', 'sliceID', 'brainID', 'neuronArea', 'cgArea', 'cgPer',
       'astroArea', 'ctxArea', 'infArea', 'globScore', 'infPerc']

measurements.columns=col

print(measurements.columns)   

#%%get rid of anything over 10%
topScore=100

a=len(measurements)

measurements=measurements.loc[measurements.infPerc<=topScore]

b=len(measurements)

outliers=a-b

print ('There are ', outliers, 'outliers, ', outliers/a*100, '%')


#%%collapse by brain ID
byBrain=measurements.reindex(columns=['brainID', 'GT', 'globScore', 'infPerc'])
byBrain=byBrain.groupby(['brainID', 'GT'], as_index=False).mean()

#%%get rid of really high brain ID
#%outlier
outlier=byBrain.brainID[byBrain.infPerc>8]
print(outlier)

byBrain= byBrain.loc[byBrain.infPerc<8]

#%%stats

WT=byBrain.loc[byBrain.GT=='6L']
WTGS=WT.globScore
Spz3=byBrain.loc[byBrain.GT=='7L']
Spz3GS=Spz3.globScore
aSNAP=byBrain.loc[byBrain.GT=='8L']
aSNAPGS=aSNAP.globScore

#sig - astro
print('Averages')
print(np.average(WTGS))
print(np.average(Spz3GS))
print(np.average(aSNAPGS))
print()

#stats
#do all samples come from the same distribution?
#auto
iScore=[WTGS, Spz3GS, aSNAPGS]
H, pKW =kruskal(WTGS, Spz3GS, aSNAPGS)
print('pKW', pKW)
print('auto')
dunn=sp.posthoc_dunn(iScore)
print(dunn)
print()

#comparing means
f, p = f_oneway(WTGS, Spz3GS, aSNAPGS)
print(p)

#Tukey
iScore=byBrain.globScore
group=byBrain.GT

comp=statsmodels.stats.multicomp.MultiComparison(iScore, group)

ph=comp.tukeyhsd(alpha=0.05)

print(ph)

#%%Glob Score counts by brain 
#%%Glob Score counts for everything GRAPH
bwidth=0.27
cap=5
tickfont=16
ticklength=6
tickwidth=3
lw=3
ms=60

plt.figure(figsize=(6, 8), facecolor='w')

WT=WTGS
Spz3=Spz3GS
aSNAP=aSNAPGS

data=[WT, Spz3, aSNAP]
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
sigloc=np.max(WT)
dx=0.1
dy=1
plt.hlines(sigloc+dy, xs[0], xs[1], colors='k')     
plt.text((xs[0]+xs[1])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'})  

sigloc=np.max(Spz3)
dx=0.1
dy=1
plt.hlines(sigloc+dy, xs[1], xs[2], colors='k')     
plt.text((xs[1]+xs[2])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'}) 

sigloc=np.max(WT)
dx=0.2
dy=3
plt.hlines(sigloc+dy, xs[0], xs[2], colors='k')     
plt.text((xs[0]+xs[2])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'}) 
    
# plt.title('Globularity \n Avg by Brain', fontsize=18, c='k', fontweight='bold')

# plt.ylabel('perimeter \n (% cortex area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(top=20)

# plt.xticks([1, 2, 3], labels=['Control', 'KD \n Driver 1', 'KD \n Driver 2'], color='k', fontweight='bold')
plt.yticks(np.arange(0, 21, 2), color='k', fontweight='bold')
plt.xticks(fontweight='bold')
plt.tick_params(axis='x', which='both', color='k', labelsize=tickfont, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  


plt.show()

#%%By Brain - Globularity - Individual pts
cap=5
tickfont=16
ticklength=4
tickwidth=2
barlw=2
ms=60

plt.figure(figsize=(3, 5), facecolor='w')

WT=WTGS
Spz3=Spz3GS
aSNAP=aSNAPGS

data=[WT, Spz3, aSNAP]

dotsize=20
dx=0.1
w=0.8
bwidth=2
ew=3
ct=2
dotcolor='gray'

dotcolor='k'
xloc=1
y=WT
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor)
plt.bar(xloc, np.mean(y), yerr=sem(y), width=w, edgecolor='k', fill=False, linewidth=bwidth, capsize=cap, 
        error_kw={'elinewidth': ew, 'capthick': ct})

dotcolor='green'
xloc=2
y=Spz3
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor)
plt.bar(xloc, np.mean(y), yerr=sem(y), width=w, edgecolor='k', fill=False, linewidth=bwidth, capsize=cap, 
        error_kw={'elinewidth': ew, 'capthick': ct})

dotcolor='blue'
xloc=3
y=aSNAP
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor)
plt.bar(xloc, np.mean(y), yerr=sem(y), width=w, edgecolor='k', fill=False, linewidth=bwidth, capsize=cap, 
        error_kw={'elinewidth': ew, 'capthick': ct})    

#sig
sigloc=np.max(WT)
dx=0.1
dy=1
plt.hlines(sigloc+dy, 1, 2, colors='k')     
plt.text((1+2)/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 18, 'fontweight': 'bold'})  

sigloc=np.max(Spz3)
dx=0.1
dy=1
plt.hlines(sigloc+dy, 2, 3, colors='k')     
plt.text((2+3)/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 18, 'fontweight': 'bold'})  

sigloc=np.max(WT)
dx=0.2
dy=3
plt.hlines(sigloc+dy, 1, 3, colors='k')     
plt.text((1+3)/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 18, 'fontweight': 'bold'})  
    
# plt.title('Globularity \n Avg by Brain', fontsize=18, c='k', fontweight='bold')

# plt.ylabel('Globularity \n (100 X perimeter/cortex area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(bottom=0, top=20)

# plt.xticks([1, 2, 3], labels=['Control', 'KD \n Driver 1', 'KD \n Driver 2'], color='k', fontweight='bold')
plt.yticks(np.arange(11)*2,  color='k', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=False,  top=False, labelbottom=False, color='k', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  

plt.show()


#%%
WT=byBrain.loc[byBrain.GT=='6L']
WTGS=WT.infPerc
Spz3=byBrain.loc[byBrain.GT=='7L']
Spz3GS=Spz3.infPerc
aSNAP=byBrain.loc[byBrain.GT=='8L']
aSNAPGS=aSNAP.infPerc

#sig - astro
print('Averages')
print(np.average(WTGS))
print(np.average(Spz3GS))
print(np.average(aSNAPGS))
print()

#stats
#do all samples come from the same distribution?
#auto
iScore=[WTGS, Spz3GS, aSNAPGS]
H, pKW =kruskal(WTGS, Spz3GS, aSNAPGS)
print('pKW', pKW)
print('auto')
dunn=sp.posthoc_dunn(iScore)
print(dunn)
print()

#comparing means
f, p = f_oneway(WTGS, Spz3GS, aSNAPGS)
print(p)

#Tukey
iScore=byBrain.infPerc
group=byBrain.GT

comp=statsmodels.stats.multicomp.MultiComparison(iScore, group)

ph=comp.tukeyhsd(alpha=0.05)

print(ph)
#%%Infiltration counts for everything GRAPH
# matplotlib.rc('axes', edgecolor='w', facecolor='k', linewidth=2)
# plt.rc_context({'xtick.color':'w', 'ytick.color':'w'})
# matplotlib.rcParams['text.color'] = 'w'

#%%Glob Score counts for everything GRAPH
bwidth=0.27
cap=5
tickfont=16
ticklength=6
tickwidth=3
lw=3
ms=60

plt.figure(figsize=(6, 8), facecolor='w')

WT=WTGS
Spz3=Spz3GS
aSNAP=aSNAPGS

data=[WT, Spz3, aSNAP]
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
widths=[0.25, 0.24, 0.24]

for x, scores, dx, in zip(xs, data, widths):
    plt.hlines(np.mean(scores), x-dx, x+dx, color='k', linewidth=lw)
    plt.errorbar(x, np.mean(scores), yerr=sem(scores), elinewidth=3, capsize=8, ecolor='k')
    
#sig
sigloc=np.max(Spz3)
dx=0.1
dy=0.5
plt.hlines(sigloc+dy, xs[0], xs[1], colors='k')     
plt.text((xs[0]+xs[1])/2-dx, sigloc+dy, r'***', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'})  

sigloc=np.max(aSNAP)
dx=0.1
dy=0.5
plt.hlines(sigloc+dy, xs[1], xs[2], colors='k')     
plt.text((xs[1]+xs[2])/2-dx, sigloc+dy, r'**', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'}) 

sigloc=np.max(aSNAP)
dx=0.2
dy=1.4
plt.hlines(sigloc+dy, xs[0], xs[2], colors='k')     
plt.text((xs[0]+xs[2])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'}) 
    
# plt.title('Globularity \n Avg by Brain', fontsize=18, c='k', fontweight='bold')

# plt.ylabel('perimeter \n (% cortex area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(top=10)

# plt.xticks([1, 2, 3], labels=['Control', 'KD \n Driver 1', 'KD \n Driver 2'], color='k', fontweight='bold')
plt.yticks(np.arange(0, 11, 2), color='k', fontweight='bold')
plt.xticks(fontweight='bold')
plt.tick_params(axis='x', which='both', color='k', labelsize=tickfont, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  


plt.show()

#%%By Brain - Globularity - Individual pts
cap=5
tickfont=16
ticklength=4
tickwidth=2
barlw=2
ms=60

plt.figure(figsize=(3, 5), facecolor='w')

WT=WTGS
Spz3=Spz3GS
aSNAP=aSNAPGS

data=[WT, Spz3, aSNAP]

dotsize=20
dx=0.1
w=0.8
bwidth=2
ew=3
ct=2

dotcolor='k'
xloc=1
y=WT
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor)
plt.bar(xloc, np.mean(y), yerr=sem(y), width=w, edgecolor='k', fill=False, linewidth=bwidth, capsize=cap, 
        error_kw={'elinewidth': ew, 'capthick': ct})

dotcolor='green'
xloc=2
y=Spz3
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor)
plt.bar(xloc, np.mean(y), yerr=sem(y), width=w, edgecolor='k', fill=False, linewidth=bwidth, capsize=cap, 
        error_kw={'elinewidth': ew, 'capthick': ct})

dotcolor='blue'
xloc=3
y=aSNAP
x=np.random.uniform(xloc-dx, xloc+dx, len(y))
plt.scatter(x, y, s=dotsize, c=dotcolor)
plt.bar(xloc, np.mean(y), yerr=sem(y), width=w, edgecolor='k', fill=False, linewidth=bwidth, capsize=cap, 
        error_kw={'elinewidth': ew, 'capthick': ct})    

#sig
sigloc=np.max(Spz3)
dx=0.1
dy=0.5
plt.hlines(sigloc+dy, 1, 2, colors='k')     
plt.text((1+2)/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 18, 'fontweight': 'bold'})  

sigloc=np.max(aSNAP)
dx=0.1
plt.hlines(sigloc+dy, 2, 3, colors='k')     
plt.text((2+3)/2-dx, sigloc+dy, r'***', {'color': 'k', 'fontsize': 18, 'fontweight': 'bold'})  

sigloc=np.max(aSNAP)
dx=0.2
dy=dy*3
plt.hlines(sigloc+dy, 1, 3, colors='k')     
plt.text((1+3)/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 18, 'fontweight': 'bold'})  
    
# plt.title('Infiltration \n Avg by Brain', fontsize=18, c='k', fontweight='bold')

# plt.ylabel('Infiltration \n (% cortex area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(top=10)

plt.xticks([1, 2, 3], labels=['Control', 'KD \n Driver 1', 'KD \n Driver 2'], color='k', fontweight='bold')
plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=False,  top=False, labelbottom=False, color='k', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  

plt.show()


#%%Correlation 
inf=byBrain.infPerc
glob=byBrain.globScore

x=glob
y=inf

# astro_m, astro_b, astro_r, astro_p, astro_se = scipy.stats.linregress(x, y)
# astro_r2=astro_r**2

# astro_y=astro_m*astro_x+astro_b

# print('astro_r2', astro_r2)
# print('astro_p', astro_p)
# print('astro_m', astro_m)
# print()

results = regress2(x, y, _method_type_2="reduced major axis")

astro_m=results['slope']
astro_b=results['intercept']
astro_SD=results['std_intercept']
astro_r=results['r']

astro_x=np.arange(np.min(x), np.max(x)+1, np.ptp(x)/10)
astro_y=astro_m*astro_x+astro_b

astro_r2=astro_r**2
astro_SE=astro_SD/np.sqrt(len(astro_x))

 # t = m/se(m)
t=astro_m/astro_SE

astro_p = stats.t.sf(np.abs(t), len(x)-1)*2

# https://www.socscistatistics.com/pvalues/tdistribution.aspx
# p-value is < .00001

print('RMA_repo_r2', astro_r2)
print('RMA_repo_m', astro_m)
print('astro_p', astro_p)

#95% CI
z=1.96

mL=astro_m-z*astro_SE
mH=astro_m+z*astro_SE

print()
print('mL', mL)
print('mH', mH)



#%%spearman correlation r
inf=byBrain.infPerc
glob=byBrain.globScore

x=glob
y=inf

rho, p =spearmanr(x, y)
print(rho)
print(p)

#%%Correlation Graph
bwidth=0.27
cap=5
tickfont=16
ticklength=4
tickwidth=2
barlw=2
ms=60

plt.figure(figsize=(8, 8), facecolor='w')

WTGS=byBrain.globScore[byBrain.GT=='6L']
Spz3GS=byBrain.globScore[byBrain.GT=='7L']
aSNAPGS=byBrain.globScore[byBrain.GT=='8L']

WTIS=byBrain.infPerc[byBrain.GT=='6L']
Spz3IS=byBrain.infPerc[byBrain.GT=='7L']
aSNAPIS=byBrain.infPerc[byBrain.GT=='8L']

dotsize=60
plt.scatter(WTGS, WTIS, c='gray', s=dotsize, label='Control')
plt.scatter(Spz3GS, Spz3IS, c='green', s=dotsize, label='KD Driver 1')
plt.scatter(aSNAPGS, aSNAPIS, c='blue', s=dotsize, label='KD Driver 1')

# plt.plot(astro_x, astro_y, c='gray')

# plt.title('Globularity-Infiltration \n Correlation', fontsize=18, c='k', fontweight='bold')
# plt.ylabel('Infiltration \n (% cortex area)', fontsize=18, c='k', fontweight='bold')
# plt.xlabel('Globularity \n (100*perimeter/cortex area)', fontsize=18, c='k', fontweight='bold')

tx=14
ty=5.5
dy=-0.3
# plt.text(tx, ty, 'slope=' + str(round(astro_m, 3)), fontsize=12, c='k')
# plt.text(tx, ty+dy, r'$\rho$ = ' + str(round(rho, 3)), fontsize=12)
# plt.text(tx, ty+2*dy, 'p < 0.0001', fontsize=12)


plt.xticks(color='k', fontweight='bold')
plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='both', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  
plt.xlim(left=15, right=0)

# plt.legend(frameon=False, loc='upper left')

plt.show()


