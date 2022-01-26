#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:31:12 2021

@author: gabrielasalazar
"""


import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress
import csv
import skimage
import scipy
from scipy import stats
import sklearn
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import f_oneway, sem, kruskal, linregress
import statsmodels.stats.multicomp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib
import re
import pandas as pd
import scikit_posthocs as sp
from matplotlib.ticker import FormatStrFormatter
from pylr2 import regress2
import pingouin as pg
# import sklearn

#%%load auto data
recsDir='InfRuns'
run='20210531'
path=os.path.join(recsDir, run, run + '.csv')
auto=pd.read_csv(path, index_col=False)
print(auto)
print(auto.columns)

#load manual data 
path='ManualQuant/Manual_Inf.20210528.csv'
manual=pd.read_csv(path)
print(manual)
col=['GT', 'sliceID', 'infPerc']
manual.columns=col
print(manual.columns)

#%%correlation 
#match up scores
matchedGS=pd.merge(manual, auto, how='left', on='sliceID')
matchedGS=matchedGS.loc[:, ['GT_x', 'sliceID', 'infPerc_x', 'infPerc_y']]
matchedGS.columns=['GT', 'sliceID', 'man_inf', 'auto_inf']
#no nans
print(matchedGS.isnull().sum().sum())
matchedGS=matchedGS.dropna()
print(matchedGS.isnull().sum().sum())


#%%
discrepancy=10

#pick out outliers
outliers=matchedGS.loc[np.abs(matchedGS.auto_inf-matchedGS.man_inf)>discrepancy]

if len(outliers)>1:
    for o in outliers.sliceID:
        print(o)
else: 
    print('no outliers')
    
#scores w/o outliers
no_outliers=matchedGS.loc[np.abs(matchedGS.auto_inf-matchedGS.man_inf)<discrepancy]

#%%stats - RMA linear regression 
# x=matchedGS.man_inf
# y=matchedGS.auto_inf

x=no_outliers.man_inf
y=no_outliers.auto_inf

results = regress2(x, y, _method_type_2="reduced major axis")

astro_m=results['slope']
astro_b=results['intercept']
astro_SD=results['std_intercept']
astro_r=results['r']

astro_x=np.arange(9)
astro_y=astro_m*astro_x+astro_b

astro_r2=astro_r**2
astro_SE=astro_SD/np.sqrt(len(astro_x))

 # t = m/se(m)
t=astro_m/astro_SE

pval = stats.t.sf(np.abs(t), len(x)-1)*2

print('RMA_repo_r2', astro_r2)
print('RMA_repo_m', astro_m)
print('pval', pval)

#95% CI
z=1.96

mL=astro_m-z*astro_SE
mH=astro_m+z*astro_SE

print()
print('mL', mL)
print('mH', mH)

#%% manual-auto correlation graph - ALL scores
bwidth=0.27
cap=5
tickfont=16
ticklength=4
tickwidth=2
barlw=2
ms=60

plt.figure(figsize=(5, 5), facecolor='w')

tx=3
ty=25
dy=-1.6

x=matchedGS.man_inf
y=matchedGS.auto_inf

xout=outliers.man_inf
yout=outliers.auto_inf

plt.scatter(x, y, c='k')
plt.scatter(xout, yout, c='r', label='outliers')

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


#%% manual-auto correlation graph - no outliers
bwidth=0.27
cap=5
tickfont=16
ticklength=12
tickwidth=4
barlw=2
ms=150
lw=4

plt.figure(figsize=(8, 8), facecolor='w')

xout=outliers.man_inf
yout=outliers.auto_inf
plt.scatter(xout, yout, c='r')

x=no_outliers.loc[no_outliers.GT=='6L'].man_inf
y=no_outliers.loc[no_outliers.GT=='6L'].auto_inf
plt.scatter(x, y, c='dimgray', label='Control', s=ms)

x=no_outliers.loc[no_outliers.GT=='7L'].man_inf
y=no_outliers.loc[no_outliers.GT=='7L'].auto_inf
plt.scatter(x, y, c='green', label='Driver 1 KD', s=ms)

x=no_outliers.loc[no_outliers.GT=='8L'].man_inf
y=no_outliers.loc[no_outliers.GT=='8L'].auto_inf
plt.scatter(x, y, c='b', label='Driver 2 KD', s=ms)

#outlier
x=no_outliers.loc[no_outliers.auto_inf==no_outliers.auto_inf.max()].man_inf
y=no_outliers.loc[no_outliers.auto_inf==no_outliers.auto_inf.max()].auto_inf
plt.scatter(x, y, c='r', label='outlier', s=ms)

#reg line
astro_x=np.arange(9)
astro_y=astro_m*astro_x+astro_b
plt.plot(astro_x, astro_y, c='k', label='regression line', linewidth=4)

#reg line error
x=np.arange(11)
lowy=mL*x+astro_b
highy=mH*x+astro_b

plt.fill_between(x, lowy, highy, color='darkgray', alpha=0.3)
#y=x
x=np.arange(11)
plt.plot(x, x, linestyle='dotted', c='k', label='y=x', linewidth=5)

# ty=9.5
# dy=-0.6
# tx=2.8
# plt.text(tx, ty, 'slope=' + str(round(astro_m, 3)) + '95%CI[' + str(round(mL, 3)) +', ' + str(round(mH, 3)) + ']', fontsize=12, c='k')
# tx=8.2
# plt.text(tx, ty+dy, '$r^2$=' + str(round(astro_r2, 2)), fontsize=12)
# tx=7.9
# plt.text(tx, ty+2*dy, 'p<0.0001', fontsize=12)
# 
# plt.title('Infiltration Scores Validation', fontsize=18, c='k', fontweight='bold')

# plt.xlabel('Infiltration - Manual \n (% ctx area)', fontsize=18, c='k', fontweight='bold')
# plt.ylabel('Infiltration - Auto \n (% ctx area)', fontsize=18, c='k', fontweight='bold')

plt.xticks(color='k', fontweight='bold')
plt.yticks(color='k', fontweight='bold')

plt.xlim(left=0, right=10)
plt.ylim(bottom=0, top=10)

# plt.legend(frameon=False, loc='lower right')

plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='k', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', bottom=True,  top=False, labelbottom=True, color='k', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.show()

#%%filter by GT and auto v manual
#auto
WT=matchedGS.loc[matchedGS.GT=='6L']
aWTscore=WT.auto_inf
mWTscore=WT.man_inf

Spz3=matchedGS.loc[matchedGS.GT=='7L']
aSpz3score=Spz3.auto_inf
mSpz3score=Spz3.man_inf

aSNAP=matchedGS.loc[matchedGS.GT=='8L']
aaSNAPscore=aSNAP.auto_inf
maSNAPscore=aSNAP.man_inf

#%%Stats
print('Averages')
print('aWTscore', np.average(aWTscore))
print('aSpz3', np.average(aSpz3score))
print('aaSNAPscore', np.average(aaSNAPscore))
print()
print('mWTscore', np.average(mWTscore))
print('mSpz3score', np.average(mSpz3score))
print('mSNAPscore', np.average(maSNAPscore))
print()

#stats
#auto v manual
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

#Kruskal-Wallis
#do all samples come from the same distribution?
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


#%%auto
#comparing means 
f, p = f_oneway(aWTscore, aSpz3score, aaSNAPscore)
print('auto')
print(p)

#Tukey
iScore=matchedGS.auto_inf
group=matchedGS.GT
comp=statsmodels.stats.multicomp.MultiComparison(iScore, group)
ph=comp.tukeyhsd(alpha=0.05)
print(ph)

f, p = f_oneway(aWTscore, aSpz3score, aaSNAPscore)
print('manual')
print(p)

#Tukey
iScore=matchedGS.man_inf
group=matchedGS.GT
comp=statsmodels.stats.multicomp.MultiComparison(iScore, group)
ph=comp.tukeyhsd(alpha=0.05)
print(ph)


#%%Comparing all scores toghether manual v auto
auto=matchedGS.auto_inf
man=matchedGS.man_inf

H, pKW =kruskal(auto, man)
print('pKW', pKW)

plt.figure(figsize=(2, 2))

y=man
x=np.full((len(y)), 1)
plt.scatter(x, y)
plt.bar(1, np.mean(y), yerr=sem(y), fill=False)

y=auto
x=np.full((len(y)), 2)
plt.scatter(x, y)
plt.bar(2, np.mean(y), yerr=sem(y), fill=False)


plt.xticks([1, 2], ['man', 'auto'])

plt.show()


#%%Inf Scores - Manual vs Auto - All
# matplotlib.rc('axes', edgecolor='w', facecolor='k', linewidth=2)
# plt.rc_context({'xtick.color':'w', 'ytick.color':'w'})
# matplotlib.rcParams['text.color'] = 'w'

bwidth=0.27
cap=5
tickfont=16
ticklength=4
tickwidth=2
barlw=2
ms=60

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

# sigloc=np.max(aaSNAPscore)
# dx=0.1
# dy=2
# x1=1
# x2=5
# plt.hlines(sigloc+dy, Xs[1], Xs[5], colors='k')     
# plt.text((Xs[x1]+Xs[x2])/2-dx, sigloc+dy, r'*', {'color': 'k', 'fontsize': 18, 'fontweight': 'bold'}) 
    
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

#%%filter by GT and auto v manual
#auto
WT=no_outliers.loc[matchedGS.GT=='6L']
aWTscore=WT.auto_inf
mWTscore=WT.man_inf

Spz3=no_outliers.loc[matchedGS.GT=='7L']
aSpz3score=Spz3.auto_inf
mSpz3score=Spz3.man_inf

aSNAP=no_outliers.loc[matchedGS.GT=='8L']
aaSNAPscore=aSNAP.auto_inf
maSNAPscore=aSNAP.man_inf

#%%Stats
print('Averages')
print('aWTscore', np.average(aWTscore))
print('aSpz3', np.average(aSpz3score))
print('aaSNAPscore', np.average(aaSNAPscore))
print()
print('mWTscore', np.average(mWTscore))
print('mSpz3score', np.average(mSpz3score))
print('mSNAPscore', np.average(maSNAPscore))
print()

#stats
#auto v manual
H, p=stats.kruskal(aWTscore, mWTscore)
print()
print('auto v manual')
print('WT')
print(p)

H, p=stats.kruskal(aSpz3score, mSpz3score)
print('Spz3')
print(p)

H, p=stats.kruskal(aaSNAPscore, maSNAPscore)
print('aSNAP')
print(p)
print()

#Kruskal-Wallis
#do all samples come from the same distribution?
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


#%%auto
#comparing means 
f, p = f_oneway(aWTscore, aSpz3score, aaSNAPscore)
print('auto')
print(p)

#Tukey
iScore=matchedGS.auto_inf
group=matchedGS.GT
comp=statsmodels.stats.multicomp.MultiComparison(iScore, group)
ph=comp.tukeyhsd(alpha=0.05)
print(ph)

f, p = f_oneway(aWTscore, aSpz3score, aaSNAPscore)
print('manual')
print(p)

#Tukey
iScore=matchedGS.man_inf
group=matchedGS.GT
comp=statsmodels.stats.multicomp.MultiComparison(iScore, group)
ph=comp.tukeyhsd(alpha=0.05)
print(ph)


#%%Inf Scores - Manual vs Auto - No outliers
# matplotlib.rc('axes', edgecolor='w', facecolor='k', linewidth=2)
# plt.rc_context({'xtick.color':'w', 'ytick.color':'w'})
# matplotlib.rcParams['text.color'] = 'w'

bwidth=0.27
cap=5
tickfont=16
ticklength=4
tickwidth=2
barlw=2
ms=60

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

    
#sig - auto v manual
sigloc=np.max(mWTscore)
dx=0.1
dy=0.3
plt.hlines(sigloc+dy, Xs[0], Xs[1], colors='k')     
plt.text((Xs[0]+Xs[1])/2-dx, sigloc+dy, r'NS', {'color': 'k', 'fontsize': 14, 'fontweight': 'bold'}) 

sigloc=np.max(mSpz3score)
dx=0.15
dy=0.3
plt.hlines(sigloc+dy, Xs[2], Xs[3], colors='k')     
plt.text((Xs[2]+Xs[3])/2-dx, sigloc+dy, r'NS', {'color': 'k', 'fontsize': 14, 'fontweight': 'bold'}) 

sigloc=np.max(aaSNAPscore)
dx=0.1
dy=0.3
plt.hlines(sigloc+dy, Xs[4], Xs[5], colors='k')     
plt.text((Xs[4]+Xs[5])/2-dx, sigloc+dy, r'NS', {'color': 'k', 'fontsize': 14, 'fontweight': 'bold'})  

#sig - manual scores
#WT v Spz3
sigloc=np.max(mSpz3score)
dx=0.1
dy=1.2
plt.hlines(sigloc+dy, Xs[0], Xs[2], colors='gray')     
plt.text((Xs[0]+Xs[2])/2-dx, sigloc+dy, r'***', {'color': 'gray', 'fontsize': 18, 'fontweight': 'bold'}) 

# #Spz3 v aSNAP
# sigloc=np.max(mWTscore)
# dx=0.1
# dy=2
# plt.hlines(sigloc+dy, Xs[2], Xs[4], colors='gray')     
# plt.text((Xs[2]+Xs[4])/2-dx, sigloc+dy, r'NS', {'color': 'k', 'fontsize': 18, 'fontweight': 'bold'}) 

#WT v aSNAP
sigloc=np.max(mSpz3score)
dx=0.1
dy=3
plt.hlines(sigloc+dy, Xs[0], Xs[4], colors='gray')     
plt.text((Xs[0]+Xs[4])/2-dx, sigloc+dy, r'**', {'color': 'gray', 'fontsize': 18, 'fontweight': 'bold'}) 

#sig - AUTO scores#############################
#WT v Spz3
sigloc=np.max(mSpz3score)
dx=0.1
dy=2
plt.hlines(sigloc+dy, Xs[1], Xs[3], colors='skyblue')     
plt.text((Xs[1]+Xs[3])/2-dx, sigloc+dy, r'*', {'color': 'skyblue', 'fontsize': 18, 'fontweight': 'bold'}) 

# #Spz3 v aSNAP
# sigloc=np.max(mWTscore)
# dx=0.1
# dy=2
# plt.hlines(sigloc+dy, Xs[2], Xs[4], colors='skyblue')     
# plt.text((Xs[2]+Xs[4])/2-dx, sigloc+dy, r'NS', {'color': 'k', 'fontsize': 18, 'fontweight': 'bold'}) 

#WT v aSNAP
sigloc=np.max(mSpz3score)
dx=0.1
dy=5.3
plt.hlines(sigloc+dy, Xs[1], Xs[5], colors='skyblue')     
plt.text((Xs[1]+Xs[5])/2-dx, sigloc+dy, r'***', {'color': 'skyblue', 'fontsize': 18, 'fontweight': 'bold'}) 

plt.title('Infiltration Validation', fontsize=18, c='k', fontweight='bold')

plt.ylabel('Infiltration \n (% ctx area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(top=11)

plt.xticks([(Xs[0]+Xs[1])/2, (Xs[2]+Xs[3])/2, (Xs[4]+Xs[5])/2], labels=['Control', 'KD \n Driver 1', 'KD \n Driver 2'], color='k', fontweight='bold')
plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='w', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  

# plt.legend(frameon=False, loc='upper left')

plt.show()

#%%Graph with violins
bwidth=0.75
cap=5
tickfont=16
ticklength=6
tickwidth=4
barlw=2
ms=60
lw=2

plt.figure(figsize=(8, 8), facecolor='w')

data=[mWTscore, aWTscore, mSpz3score, aSpz3score, maSNAPscore, aaSNAPscore]
xs=[1,2, 3.25,4.25, 5.5,6.5]

parts=plt.violinplot(data, xs, widths=bwidth, showmeans=False, showextrema=False)

colors=['gray', 'skyblue']
colors=colors*3

#set violin colors
for pc, c in zip(parts['bodies'], colors):
    pc.set_facecolor(c)
    pc.set_edgecolor(c)
    pc.set_alpha(1)
    
#mean and error
widths=[0.28, 0.3, 0.32, 0.3, 0.32, 0.3]

for x, scores, dx, in zip(xs, data, widths):
    plt.hlines(np.mean(scores), x-dx, x+dx, color='k', linewidth=barlw)
    plt.errorbar(x, np.mean(scores), yerr=sem(scores), elinewidth=3, capsize=8, ecolor='k')
    
###############sig - auto v manual
sigloc=np.max(mWTscore)
dx=0.22
dy=0.3
cdy=1.3
plt.hlines(sigloc+dy, xs[0], xs[1], colors='k', linewidth=lw)     
plt.text((xs[0]+xs[1])/2-dx, sigloc+cdy*dy, r'NS', {'color': 'k', 'fontsize': 24, 'fontweight': 'bold'}) 

sigloc=np.max(mSpz3score)
# dx=0.22
# dy=0.3
plt.hlines(sigloc+dy, xs[2], xs[3], colors='k', linewidth=lw)     
plt.text((xs[2]+xs[3])/2-dx, sigloc+cdy*dy, r'NS', {'color': 'k', 'fontsize': 24, 'fontweight': 'bold'}) 

sigloc=np.max(aaSNAPscore)
# dx=0.21
# dy=0.3
plt.hlines(sigloc+dy, xs[4], xs[5], colors='k', linewidth=lw)     
plt.text((xs[4]+xs[5])/2-dx, sigloc+cdy*dy, r'NS', {'color': 'k', 'fontsize': 24, 'fontweight': 'bold'})  

##############sig - manual scores
#WT v Spz3
sigloc=np.max(mSpz3score)
dx=0.32
dy=0.9
plt.hlines(sigloc+dy, xs[0], xs[2], colors='dimgray', linewidth=lw)     
plt.text((xs[0]+xs[2])/2-dx, sigloc+dy, r'****', {'color': 'dimgray', 'fontsize': 28, 'fontweight': 'bold'}) 

# #Spz3 v aSNAP
# sigloc=np.max(mWTscore)
# dx=0.1
# dy=2
# plt.hlines(sigloc+dy, Xs[2], Xs[4], colors='dimgray', linewidth=lw)     
# plt.text((Xs[2]+Xs[4])/2-dx, sigloc+dy, r'NS', {'color': 'k', 'fontsize': 18, 'fontweight': 'bold'}) 

#WT v aSNAP
sigloc=np.max(mSpz3score)
dx=0.2
dy=2.8
plt.hlines(sigloc+dy, xs[0], xs[4], colors='dimgray', linewidth=lw)     
plt.text((xs[0]+xs[4])/2-dx, sigloc+dy, r'**', {'color': 'dimgray', 'fontsize': 28, 'fontweight': 'bold'}) 

#sig - AUTO scores#############################
#WT v Spz3
sigloc=np.max(mSpz3score)
dx=0.1
dy=1.7
plt.hlines(sigloc+dy, xs[1], xs[3], colors='skyblue',  linewidth=lw)     
plt.text((xs[1]+xs[3])/2-dx, sigloc+dy, r'*', {'color': 'skyblue', 'fontsize': 28, 'fontweight': 'bold'}) 

# #Spz3 v aSNAP
# sigloc=np.max(mWTscore)
# dx=0.1
# dy=2
# plt.hlines(sigloc+dy, Xs[2], Xs[4], colors='skyblue')     
# plt.text((Xs[2]+Xs[4])/2-dx, sigloc+dy, r'NS', {'color': 'k', 'fontsize': 18, 'fontweight': 'bold'}) 

#WT v aSNAP
sigloc=np.max(mSpz3score)
dx=0.5
dy=5.5
plt.hlines(sigloc+dy, xs[1], xs[5], colors='skyblue',  linewidth=lw)     
plt.text((xs[1]+xs[5])/2-dx, sigloc+dy, r'****', {'color': 'skyblue', 'fontsize': 28, 'fontweight': 'bold'}) 

    
# plt.title('Globularity', fontsize=18, c='k', fontweight='bold')

# plt.ylabel('automated score \n (perimeter/cortex area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(bottom=0, top=12)

# plt.xticks([1.5, 2.5, 3.5], labels=['Control', 'KD \n Driver 1', 'KD \n Driver 2'], color='k', fontweight='bold')
# plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='w', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  

plt.show()


#%%%MAPE
true=matchedGS.man_inf
pred=matchedGS.auto_inf

mape=np.mean(np.abs(true-pred))

print(mape)

#%%scrap
colors=['gray', 'skyblue']
print(colors)
colors=colors*3
print(colors)



