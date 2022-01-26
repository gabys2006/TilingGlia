#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 20 13:31:12 2021

@author: gabrielasalazar

VALIDATE AIS BY COMPARING MANUAL AND AUTOMATED SCORES. 

"""

#%%

import os
import csv
import skimage
import pandas as pd
from pylr2 import regress2
import numpy as np
from scipy import stats
from scipy.stats import kruskal, sem
import scikit_posthocs as sp
import matplotlib.pyplot as plt 

from TilingPipelineFunctions import quantify_slice

#%%QUANTIFY ALL VALIDATION IMAGES
#THESE ARE IMAGES WITH MANUAL AND AUTOMATED INFILTRATION SCORES

dataDir='ValidationImages'

#create results directory
newRun = 'TestRun'
if not os.path.exists(newRun):
    os.makedirs(newRun)

#create csv to write data as it is gathered
newRunPathcsv=os.path.join(newRun, newRun+'.csv')
with open(newRunPathcsv, "w", newline='') as rec:
    writer = csv.writer(rec, delimiter=',')
    writer.writerow(["GT", "sliceID", "AMI", "AIS"])
    rec.close()

#iterate through test images
#create list to hold files that caused errors
errors = []
for root, directory, file in os.walk(dataDir):
    for f in file:
        if '.tif' in f:
            try:
                ipath=os.path.join(root, f)
                #define sliceID
                sliceID = f
                #load image
                fullimg=skimage.io.imread(ipath, plugin="tifffile")
                #quantify image
                measurements = quantify_slice(fullimg, sliceID)
                #write data on csv
                with open(newRunPathcsv, 'a', newline='') as rec:
                    writer = csv.writer(rec, delimiter=',')
                    writer.writerow(measurements)
                    rec.close()
            except:
                #add errors to list
                errors.append(f)

#check if there are errors
print (errors)

#%%Load AUTO SCORES

newRun='TestRun'
path=os.path.join(newRun, newRun + '.csv')
auto=pd.read_csv(path, index_col=False)
print(auto)
print(auto.columns)

#load manual data 
path='Manual_Inf.20210528.csv'
manual=pd.read_csv(path)
print(manual)
col=['GT', 'sliceID', 'infPerc']
manual.columns=col
print(manual.columns)

#%%MATCH UP CORRESPONDING AUTO AND MANUAL SCORES

key=['GT', 'sliceID']
matchedScores=pd.merge(manual, auto, how='outer', on=key)
#rename columns
matchedScores.columns=['GT', 'sliceID', 'man_inf', 'AMI', 'auto_inf']
#delete NANS if there are any
print(matchedScores.isnull().sum().sum())
matchedScores=matchedScores.dropna()
print(matchedScores.isnull().sum().sum())


#%%REDUCED MAJOR AXIS LINEAR REGRESSION 
x=matchedScores.man_inf
y=matchedScores.auto_inf
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


# cap=5
# tickfont=16
ticklength=4
tickwidth=2
# barlw=2
# ms=60

plt.figure(figsize=(5, 5))

# tx=3
# ty=25
# dy=-1.6

x=matchedScores.man_inf
y=matchedScores.auto_inf

plt.scatter(x, y, c='k')

plt.title('Infiltration Scores Validation', fontsize=18, c='k', fontweight='bold')

plt.xlabel('Infiltration - Manual \n (% ctx area)', fontsize=18, c='k', fontweight='bold')
plt.ylabel('Infiltration - Auto \n (% ctx area)', fontsize=18, c='k', fontweight='bold')

plt.xticks(color='k', fontweight='bold')
plt.yticks(color='k', fontweight='bold')

plt.legend()

plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='k', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', bottom=True,  top=False, labelbottom=True, color='k', 
                labelsize=16, length=ticklength, width=tickwidth)
# plt.ylim(top=20)
# plt.xlim(right=20)
plt.show()

#%%GRAPH RELATIONSHIP BETWEEN MANUAL AND AUTO SCORES

bwidth=0.27
cap=5
tickfont=16
ticklength=12
tickwidth=4
barlw=2
ms=150
lw=4

plt.figure(figsize=(8, 8), facecolor='w')

x=matchedScores.loc[matchedScores.GT=='6L'].man_inf
y=matchedScores.loc[matchedScores.GT=='6L'].auto_inf
plt.scatter(x, y, c='dimgray', label='Control', s=ms)

x=matchedScores.loc[matchedScores.GT=='7L'].man_inf
y=matchedScores.loc[matchedScores.GT=='7L'].auto_inf
plt.scatter(x, y, c='green', label='Driver 1 KD', s=ms)

x=matchedScores.loc[matchedScores.GT=='8L'].man_inf
y=matchedScores.loc[matchedScores.GT=='8L'].auto_inf
plt.scatter(x, y, c='b', label='Driver 2 KD', s=ms)

#reg line
regline_x=np.arange(9)
regline_y=m*regline_x+b
plt.plot(regline_x, regline_y, c='k', label='regression line', linewidth=4)

#reg line error
x=np.arange(11)
lowy=mL*x+b
highy=mH*x+b

plt.fill_between(x, lowy, highy, color='darkgray', alpha=0.3)
#y=x
x=np.arange(11)
plt.plot(x, x, linestyle='dotted', c='k', label='y=x', linewidth=5)

ty=9.5
dy=-0.6
tx=2.8
plt.text(tx, ty, 'slope=' + str(round(m, 3)) + '95%CI[' + str(round(mL, 3)) +', ' + str(round(mH, 3)) + ']', fontsize=12, c='k')
tx=8.2
plt.text(tx, ty+dy, '$r^2$=' + str(round(r2, 2)), fontsize=12)
tx=7.9
plt.text(tx, ty+2*dy, 'p<0.0001', fontsize=12)

plt.title('Infiltration Scores Validation', fontsize=18, c='k', fontweight='bold')

plt.xlabel('Infiltration - Manual \n (% ctx area)', fontsize=18, c='k', fontweight='bold')
plt.ylabel('Infiltration - Auto \n (% ctx area)', fontsize=18, c='k', fontweight='bold')

plt.xticks(color='k', fontweight='bold')
plt.yticks(color='k', fontweight='bold')

# plt.xlim(left=0, right=10)
# plt.ylim(bottom=0, top=10)

plt.legend(frameon=False, loc='lower right')

plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='k', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', bottom=True,  top=False, labelbottom=True, color='k', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.show()

#%%SEPARATE BY EXP GROUP (WT, Spz3 KD, and aSNAP KD) 
#AND AUTO OR MANUAL SCORE

WT=matchedScores.loc[matchedScores.GT=='6L']
aWTscore=WT.auto_inf
mWTscore=WT.man_inf

Spz3=matchedScores.loc[matchedScores.GT=='7L']
aSpz3score=Spz3.auto_inf
mSpz3score=Spz3.man_inf

aSNAP=matchedScores.loc[matchedScores.GT=='8L']
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

#kruskal wallis test comparing auto and manual scores
KS, p=stats.ks_2samp(aWTscore, mWTscore)
print()
print('auto v manual')
print('WT')
print(p)

KS, p=stats.ks_2samp(aSpz3score, mSpz3score)
print('Spz3')
print(p)

KS, p=stats.ks_2samp(aaSNAPscore, maSNAPscore)
print('aSNAP')
print(p)

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


#%%GRAPH DISTRIBUTIONS OF SCORES GROUPED BY EXP GROUP AND MANUAL/AUTO 

tickfont=16
ticklength=4
tickwidth=2

plt.figure(figsize=(5, 5), facecolor='w')

data=[mWTscore, aWTscore, mSpz3score, aSpz3score, maSNAPscore, aaSNAPscore]
dx=1
dx2=0.3
start=0
Xs=[start, start+dx2, 
    start+dx, start+dx+dx2,
    start+2*dx, start+2*dx+dx2]

color=['gray', 'skyblue']*3
labels=['manual', 'auto', '', '', '', '']

for d, x, col, l in zip(data, Xs, color, labels):
    X=np.full(len(d), x)
    plt.scatter(X, d, c=col, label=l)

means=[]
err=[]
for d in data:
    m=np.mean(d)
    means.append(m)
    e=sem(d)
    err.append(e)

plt.bar(Xs, means, yerr=err, width=0.25, capsize=3, color='None', edgecolor='k')

#sig
sigloc=np.max(mWTscore)
dx=0.1
dy=0.5
x1=0
x2=1
plt.hlines(sigloc+dy, Xs[0], Xs[1], colors='k')     
plt.text((Xs[x1]+Xs[x2])/2-dx, sigloc+dy, r'NS', {'color': 'k', 'fontsize': 14, 'fontweight': 'bold'}) 

sigloc=np.max(aSpz3score)
dx=0.15
dy=0.5
x1=2
x2=3
plt.hlines(sigloc+dy, Xs[2], Xs[3], colors='k')     
plt.text((Xs[x1]+Xs[x2])/2-dx, sigloc+dy, r'****', {'color': 'k', 'fontsize': 14, 'fontweight': 'bold'}) 

sigloc=np.max(aaSNAPscore)
dx=0.1
dy=0.5
x1=4
x2=5
plt.hlines(sigloc+dy, Xs[4], Xs[5], colors='k')     
plt.text((Xs[x1]+Xs[x2])/2-dx, sigloc+dy, r'NS', {'color': 'k', 'fontsize': 14, 'fontweight': 'bold'})  
    
plt.title('Infiltration Validation', fontsize=18, c='k', fontweight='bold')

plt.ylabel('Infiltration \n (% ctx)', fontsize=18, c='k', fontweight='bold')
plt.ylim(top=11)

plt.xticks([(Xs[0]+Xs[1])/2, (Xs[2]+Xs[3])/2, (Xs[4]+Xs[5])/2], labels=['Control', 'KD \n Driver 1', 'KD \n Driver 2'], color='k', fontweight='bold')
plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='w', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  

plt.legend(frameon=False)

plt.show()

