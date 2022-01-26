#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:38:05 2021

@author: gabrielasalazar

VALIDATE AIM BY COMPARING MANUAL AND AUTOMATED SCORES. 

"""
import os
import pandas as pd
from pylr2 import regress2
import numpy as np
from scipy import stats
from scipy.stats import kruskal, sem
import scikit_posthocs as sp
import matplotlib.pyplot as plt 


#%%load all manual scores
data='GradedbyGabby20210521'+'.csv'
manual=pd.read_csv(data, index_col=False)
manual.columns=['GT', 'sliceID', 'manAMI']
print (manual.columns)

#%%LOAD AUTOMATED DATA
#THESE ARE AMI/AIS SCORES FOR THE FULL DATASET QUANTIED USING Z_Stack_Quantification.py
newRun = '2021_05_31'
path=os.path.join(newRun+'.csv')
auto=pd.read_csv(path, index_col=False)
auto.columns=['GT', 'brainID', 'sliceID', 'autoAMI', 'autoAIS']


#%%MERGE WITH MANUAL SCORES
#WE MANUALLY SCORED ~1000 IMAGES WITH 0-4 ACCORDING TO SEVERITY OF GLOBULARITY
key=['GT', 'picID']
manAuto=pd.merge(manual, auto, on=key, how='outer')

#%%REDUCED MAJOR AXIS LINEAR REGRESSION 
x=manAuto.manAMI
y=manAuto.autoAMI
results = regress2(x, y, _method_type_2="reduced major axis")
#grab results from regression 
m=results['slope']
b=results['intercept']
SD=results['std_intercept']
r=results['r']

#calculate regression line for plotting
regline_x=np.arange(9)
regline_y=m*regline_x+b

#calculate r^2 to assess correlation
r2=r**2

#calculate standard error
SE=SD/np.sqrt(len(x))

#calc p_value 
t=m/SE
pval = stats.t.sf(np.abs(t), len(x)-1)*2

print('RMA_r2', r2)
print('RMA_m', m)
print('pval', pval)

#calculate 95% Confidence interval
z=1.96

mL=m-z*SE
mH=m+z*SE

print()
print('mL', mL)
print('mH', mH)

#%%GROUP AUTOMATED SCORES BY CORRESPONDING MANUAL SCORES FOR GRAPHING

GS0=manAuto.auto[manAuto.manAMI==0]
GS1=manAuto.auto[manAuto.manAMI==1]
GS2=manAuto.auto[manAuto.manAMI==2]
GS3=manAuto.auto[manAuto.manAMI==3]
GS4=manAuto.auto[manAuto.manAMI==4]

print(len(GS0))
print(len(GS1))
print(len(GS2))
print(len(GS3))
print(len(GS4))


#%%GRAPH RELATIONSHIP BETWEEN MANUAL AND AUTO SCORES

bwidth=0.27
cap=5
tickfont=16
ticklength=6
tickwidth=4
barlw=3
ms=60

plt.figure(figsize=(8, 8), facecolor='w')

data=[GS0, GS1, GS2, GS3, GS4]

x=np.arange(5)
parts=plt.violinplot(data, x, showmeans=False, showextrema=False)

#set violin colors
for pc in parts['bodies']:
    pc.set_facecolor('k')
    pc.set_edgecolor('k')
    pc.set_alpha(0.5)

#linear reg    
plt.plot(regline_x, regline_y, c='k', linewidth=4, linestyle='dotted')

#means and error
widths=[0.22, 0.24, 0.22, 0.21, 0.13]

for gs, scores, dx, in zip(x, data, widths):
    plt.hlines(np.mean(scores), gs-dx, gs+dx, color='k', linewidth=barlw)
    plt.errorbar(gs, np.mean(scores), yerr=sem(scores), elinewidth=3, capsize=8, ecolor='k')

#anotations
tx=1.7
ty=20
dy=-1.5
plt.text(tx, ty, 'slope=' + str(round(m, 3)) + '\n95%CI[' + str(round(mL, 3)) +', ' + str(round(mH, 3)) + ']', fontsize=14, c='k')
plt.text(tx, ty+dy, '$r^2$=' + str(round(r2, 2)), fontsize=14)
plt.text(tx, ty+2*dy, 'p<0.0001', fontsize=14)

#axes titles
plt.title('Globularity Validation', fontsize=18, c='k', fontweight='bold')
plt.xlabel('manual score', fontsize=18, c='k', fontweight='bold')
plt.ylabel('automated score \n (perimeter/cortex area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(top=23)

plt.xticks(x, fontweight='bold')
plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='k', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  

plt.show()

#%%SEPARATE BY EXP GROUP (WT, Spz3 KD, and aSNAP KD) 
#AND AUTO OR MANUAL SCORE

WT=manAuto.loc[manAuto.GT=='6L']
aWTscore=WT.auto_inf
mWTscore=WT.man_inf

Spz3=manAuto.loc[manAuto.GT=='7L']
aSpz3score=Spz3.auto_inf
mSpz3score=Spz3.man_inf

aSNAP=manAuto.loc[manAuto.GT=='8L']
aaSNAPscore=aSNAP.auto_inf
maSNAPscore=aSNAP.man_inf

#%%STATS
#KRUSKAL-WALLIS TEST FOLLOWED BY DUNN'S PAIRWISE COMPARISONS

#find averages 
print('Averages')
print('aWTscore', np.average(aWTscore))
print('aSpz3', np.average(aSpz3score))
print('aaSNAPscore', np.average(aaSNAPscore))
print()
print('mWTscore', np.average(mWTscore))
print('mSpz3score', np.average(mSpz3score))
print('mSNAPscore', np.average(maSNAPscore))
print()

#Kruskal-Wallis comparing all auto scores to each other and all manual scores to each other 
#auto
iScore=[aWTscore, aSpz3score, aaSNAPscore]
H, pKW =kruskal(aWTscore, aSpz3score, aaSNAPscore)
print('pKW', pKW)
print('auto')
dunn=sp.posthoc_dunn(iScore)
print(dunn)
print()

#manual
iScore=[mWTscore, mSpz3score, maSNAPscore]
H, pKW =kruskal(mWTscore, mSpz3score, maSNAPscore)
print('pKW', pKW)
print('manual')
dunn=sp.posthoc_dunn(iScore)
print(dunn)
print()


#%%GRAPH **MANUAL** SCORES BY EXPERIMENTAL GROUP 
WT=mWTscore
Spz3=mSpz3score
aSNAP=maSNAPscore


tickfont=16
ticklength=6
tickwidth=4
lw=1

plt.figure(figsize=(8, 8), facecolor='w')

data=[WT, Spz3, aSNAP]
xs=[1,2,3]

parts=plt.violinplot(data, showmeans=False, showextrema=False)

colors=['gray', 'green', 'b']
#set violin colors
for pc, c in zip(parts['bodies'], colors):
    pc.set_facecolor(c)
    pc.set_edgecolor(c)
    pc.set_alpha(1)
    
#mean and error
widths=[0.25, 0.12, 0.26]

for x, scores, dx, in zip(xs, data, widths):
    plt.hlines(np.mean(scores), x-dx, x+dx, color='k', linewidth=barlw)
    plt.errorbar(x, np.mean(scores), yerr=sem(scores), elinewidth=3, capsize=8, ecolor='k')
    
#sig
sigloc=np.max(Spz3)
dx=0.2
dy=0.1
plt.hlines(sigloc+dy, xs[0], xs[1], colors='k',  linewidth=lw)     
plt.text((xs[1]+xs[3])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 28, 'fontweight': 'bold'}) 

sigloc=np.max(Spz3)
dx=0.2
dy=0.1
plt.hlines(sigloc+dy, xs[0], xs[1], colors='k',  linewidth=lw)     
plt.text((xs[1]+xs[3])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 28, 'fontweight': 'bold'})   

sigloc=np.max(Spz3)
dx=0.2
dy=0.2
plt.hlines(sigloc+dy, xs[0], xs[1], colors='k',  linewidth=lw)     
plt.text((xs[1]+xs[3])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 28, 'fontweight': 'bold'}) 
    
plt.title('Globularity', fontsize=18, c='k', fontweight='bold')

plt.ylabel('manual score \n (perimeter/cortex area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(top=5)

plt.xticks([1, 2, 3], labels=['Control', 'KD \n Driver 1', 'KD \n Driver 2'], color='k', fontweight='bold')
plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='w', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  

plt.show()


#%%GRAPH **AUTO** SCORES BY EXPERIMENTAL GROUP 

WT=aWTscore
Spz3=aSpz3score
aSNAP=aaSNAPscore


tickfont=16
ticklength=6
tickwidth=4
lw=1

plt.figure(figsize=(8, 8), facecolor='w')

data=[WT, Spz3, aSNAP]

parts=plt.violinplot(data, showmeans=True, showextrema=False)

colors=['gray', 'green', 'b']
#set violin colors
for pc, c in zip(parts['bodies'], colors):
    pc.set_facecolor(c)
    pc.set_edgecolor(c)
    pc.set_alpha(1)
    
#mean and error
widths=[0.23, 0.19, 0.15]
xs=[1,2,3]

for x, scores, dx, in zip(xs, data, widths):
    plt.hlines(np.mean(scores), x-dx, x+dx, color='k', linewidth=barlw)
    plt.errorbar(x, np.mean(scores), yerr=sem(scores), elinewidth=3, capsize=8, ecolor='k')
    
#sig
sigloc=np.min(Spz3)
dx=0.2
dy=-1
plt.hlines(sigloc+dy, 1, 2, colors='k')     
plt.text((1+2)/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'})  

sigloc=np.min(Spz3)
dx=0.2
dy=-1
plt.hlines(sigloc+2*dy, 2, 3, colors='k')     
plt.text((2+3)/2-dx, sigloc+2*dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'})  

sigloc=np.min(Spz3)
dx=0.2
dy=-1.2
plt.hlines(sigloc++3*dy, 1, 3, colors='k')     
plt.text((1+3)/2-dx, sigloc+3*dy, r'****', {'color': 'k', 'fontsize': 22, 'fontweight': 'bold'})  
    
plt.title('Globularity', fontsize=18, c='k', fontweight='bold')

plt.ylabel('automated score \n (perimeter/cortex area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(bottom=21, top=-5)

plt.xticks([1, 2, 3], labels=['Control', 'KD \n Driver 1', 'KD \n Driver 2'], color='k', fontweight='bold')
plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='w', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  

plt.show()












