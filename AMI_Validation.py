#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:38:05 2021

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

#%%make list of manual scores - unblinded
dataDir='GlobScoreVal'
scorer='GradedbyGabby20210521'
dirName=os.path.join(dataDir, scorer)

GTs=['1BL', '2BL', '6L', '7L', '8L']

grades=[]

for root, directory, file in os.walk(dirName):
    for d in directory:
        dPath=os.path.join(root, d)
        for Root, Directory, File in os.walk(dPath):
            for f in File:
                for gt in GTs:
                    if gt in f:
                        GT = gt
                if 'Red' in f:
                    nName=f.replace('Red', 'Blue')
                    picID=f.replace('.Red', '_')
                if 'Green' in f:
                    nName=f.replace('Green', 'Blue')
                    picID=f.replace('.Green', '_')
                grade=d
                grades.append([GT, picID, grade])

col=['GT', 'picID', 'globScore']                
grades=pd.DataFrame(grades, columns=col)  
path=os.path.join(dataDir, scorer+'.csv')
grades.to_csv(path, index=False)


#%%make list of manual scores - blind
#using files of images
#load blinding list 
path='GlobScoreVal/ImagesToGrade_Ariana_Grace/BlindingIDS.csv'
unblinding=pd.read_csv(path, dtype='str')
# col=['picID', 'blindID']

dataDir='GlobScoreVal'
# scorer='GradedbyGrace'
# scorer='GradedbyAriana'
dirName=os.path.join(dataDir, scorer)

GTs=['1BL', '2BL', '6L', '7L', '8L']

grades=[]

for root, directory, file in os.walk(dirName):
    for d in directory:
        dPath=os.path.join(root, d)
        for Root, Directory, File in os.walk(dPath):
            for f in File:
                grade=d
                #unblind
                bID=f.replace('.tif', '')
                pID=unblinding.picID[unblinding.blindID==bID].values[0]
                #get GT
                for gt in GTs:
                    if gt in pID:
                        GT = gt
                grades.append([GT, pID, grade])

col=['GT', 'picID', 'globScore']                
grades=pd.DataFrame(grades, columns=col)  
path=os.path.join(dataDir, scorer+'.csv')
grades.to_csv(path, index=False)




#%%
#make list of manual scorer - blind
#using csv\
#load blinding list
path='GlobScoreVal/ImagesToGrade_Ariana_Grace/BlindingIDS.csv'
unblinding=pd.read_csv(path, dtype='str')
# col=['picID', 'blindID']
#add GT to unblinding
GTs=['1BL', '2BL', '6L', '7L', '8L']

genotypes=[]

#load manual grading list
dataDir='GlobScoreVal'
# scorer='Grace'
scorer='Ariana'
path=os.path.join(dataDir, scorer + 'Grades.csv')
col=['blindID', 'globScore']
mgrades=pd.read_csv(path, dtype='str')
mgrades.columns=col

#merge to unblind
key=['blindID']
unblind=pd.merge(mgrades, unblinding, how='left', on=key)
unblind=unblind.drop('blindID', axis=1)

path=os.path.join(dataDir, scorer + 'Final.csv')
unblind.to_csv(path, index=False)



#%%load all manual scores
dataDir='GlobScoreVal'
scorer='GradedbyGabby20210521'+'.csv'
path=os.path.join(dataDir, scorer)
gabby=pd.read_csv(path, index_col=False)
print(gabby.isna().sum().sum())


dataDir='GlobScoreVal'
scorer='GraceFinal'+'.csv'
path=os.path.join(dataDir, scorer)
grace=pd.read_csv(path, index_col=False)
print(grace.isna().sum().sum())
print(len(grace))


dataDir='GlobScoreVal'
scorer='ArianaFinal'+'.csv'
path=os.path.join(dataDir, scorer)
ariana=pd.read_csv(path, index_col=False)
print(ariana.isna().sum().sum())
print(len(ariana))

#%%
#merge scores
key=['picID']
gg=pd.merge(gabby, grace, on=key, how='inner')
gg.columns=['GT', 'picID', 'Gabby', 'Grace']
print(gg.isna().sum().sum())
print(len(gg))

#%%
key=['picID']
ga=pd.merge(gabby, ariana, on=key, how='inner')
ga.columns=['GT', 'picID', 'Gabby', 'Ariana']
print(ga.isna().sum().sum())
print(len(ga))

#%%
key=['picID', 'Gabby']
allGlobScores=pd.merge(gg, ga, on=key, how='outer')
print(allGlobScores.isna().sum().sum())
print(len(allGlobScores))

#%%get matching ones too
three=allGlobScores.dropna()

#%%gg w/o matching
gg2=pd.concat([gg, three]).drop_duplicates(subset=['picID'], keep=False)
print(len(gg2))

#%%gg w/o matching
ga2=pd.concat([ga, three]).drop_duplicates(subset=['picID'], keep=False)
print(len(ga2))

#%%graph- three scorers
bwidth=0.27
cap=5
tickfont=16
ticklength=4
tickwidth=2
barlw=2
ms=60

plt.figure(figsize=(15, 5), facecolor='w')

dx=15
#three scorers
x=np.arange(len(three))
a=1
plt.scatter(x, three.Gabby, label='Scorer 1', s=70, alpha=a, facecolor='none', edgecolor='k', marker='o')
a=1
plt.scatter(x, three.Ariana, label='Scorer 2', s=40, alpha=a, marker='x', c='m')
a=1
plt.scatter(x, three.Grace, label='Scorer 3', s=30, alpha=a, marker='o', facecolor='none', edgecolor='c')
plt.text(-2, 3.7, 'ICC(3,k)=0.987', fontsize=12)
plt.text(-2, 3.5, 'p<0.0001', fontsize=12)
plt.vlines(len(three)+dx/2, 0, 4, linestyles='dotted', color='k')

#ga
x=np.arange(len(ga2))+len(three)+dx
a=1
plt.scatter(x, ga2.Gabby, s=50, alpha=a, facecolor='none', edgecolor='k', marker='o')
a=1
plt.scatter(x, ga2.Ariana, s=40, alpha=a, marker='x', c='m')
plt.text(260, 3.7, 'ICC(3,k)=0.983', fontsize=12)
plt.text(260, 3.5, 'p<0.0001', fontsize=12)
plt.vlines(len(three)+len(ga2)+dx+dx/2, 0, 4, linestyles='dotted', color='k')

#gg
x=np.arange(len(gg2))+len(three)+len(ga2)+2*dx
a=1
plt.scatter(x, gg2.Gabby, s=50, alpha=a, facecolor='none', edgecolor='k', marker='o')
a=1
plt.scatter(x, gg2.Grace, s=30, alpha=a, marker='o', facecolor='none', edgecolor='c')
plt.text(525, 3.7, 'ICC(3,k)=0.994', fontsize=12)
plt.text(525, 3.5, 'p<0.0001', fontsize=12)

plt.legend(bbox_to_anchor=(1,1), fontsize=12)
# plt.legend(fontsize=12)
x1=len(three)/2+dx
x2=len(ga2)/2+len(three)+dx
x3=len(gg2)/2+len(three)+len(ga2)+2*dx
xticks=[x1, x2, x3]
xlabels=['all scorers', 'scorer 1 & \n scorer 2', 'scorer 1 & \n scorer 3']
plt.xticks(xticks, labels=xlabels, fontweight='bold')

# plt.xlabel('Image ID', fontweight='bold', fontsize=16)
plt.xticks(fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=False,  top=False, labelbottom=True, color='k', 
                labelsize=16, length=ticklength, width=tickwidth)


y=np.arange(5)
plt.yticks(y, color='k', fontweight='bold')
plt.ylabel('Manual Globularity Score', fontweight='bold', fontsize=16)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  

plt.xlim(left=-10)
plt.show()

#%%reformat data
name=np.full((len(ariana)), 'ariana')
arianaTidy=ariana.loc[:, ['globScore', 'picID']]
arianaTidy['scorer']=name

name=np.full((len(grace)), 'grace')
graceTidy=grace.loc[:, ['globScore', 'picID']]
graceTidy['scorer']=name

name=np.full((len(gabby)), 'gabby')
gabbyTidy=gabby.loc[:, ['globScore', 'picID']]
gabbyTidy['scorer']=name

allTidy=pd.concat([arianaTidy, graceTidy, gabbyTidy])

#%%three tidy
threeTidyPics=three.picID
threeTidy=pd.merge(threeTidyPics, allTidy, how='inner', on='picID')

print(len(threeTidy))
print(threeTidy.isna().sum().sum())

print(threeTidy.value_counts(subset='picID').min())

#%%ICC - for all three
icc = pg.intraclass_corr(data=threeTidy, targets='picID', raters='scorer',
                         ratings='globScore', nan_policy='omit')
icc.set_index("Type")

#%%2 tidy - GA
twotidyPics=ga2.picID
twoTidyGA=pd.merge(twotidyPics, allTidy, how='inner', on='picID')

print(len(twoTidyGA))
print(twoTidyGA.isna().sum().sum())

print(twoTidyGA.value_counts(subset='picID').max())

#%%ICC - for GA
icc = pg.intraclass_corr(data=twoTidyGA, targets='picID', raters='scorer',
                         ratings='globScore', nan_policy='omit')
icc.set_index("Type")

#%%2 tidy - GG
twotidyPics=gg2.picID
twoTidyGG=pd.merge(twotidyPics, allTidy, how='inner', on='picID')

print(len(twoTidyGG))
print(twoTidyGG.isna().sum().sum())

print(twoTidyGG.value_counts(subset='picID').max())

#%%ICC - for GA
icc = pg.intraclass_corr(data=twoTidyGG, targets='picID', raters='scorer',
                         ratings='globScore', nan_policy='omit')
icc.set_index("Type")

#%%heatmap for scores

# Defining index for the dataframe
idx = ['1', '2', '3', '4']
  
# Defining columns for the dataframe
cols = list('ABCD')
  
# Entering values in the index and columns  
# and converting them into a panda dataframe
df = pd.DataFrame([[10, 20, 30, 40], [50, 30, 8, 15],
                   [25, 14, 41, 8], [7, 14, 21, 28]],
                   columns = cols, index = idx)
  
# Displaying dataframe as an heatmap
# with diverging colourmap as RdYlBu
plt.imshow(df, cmap ="RdYlBu")
  
# Displaying a color bar to understand
# which color represents which range of data
plt.colorbar()
  
# Assigning labels of x-axis 
# according to dataframe
plt.xticks(range(len(df)), df.columns)
  
# Assigning labels of y-axis 
# according to dataframe
plt.yticks(range(len(df)), df.index)
  
# Displaying the figure
plt.show()

#%%heatmap for scores
#make DF with just scores

threeScores=three.loc[:, ['Gabby', 'Grace', 'Ariana']]
threeScores=threeScores.transpose()

#%%split into part bc graph is too long

matplotlib.rcParams['pdf.fonttype'] = 42

#part 1
beg=0
end=len(threeScores.columns)

# #part 2
# beg=50
# end=100

# threeScoresPiece=threeScores.iloc[:, beg:end]
#graph
ls=12
ticklength=2
tickwidth=1

fig = plt.figure(figsize=(37.5, 3))


plt.imshow(threeScores, cmap ="RdYlBu")

x=np.arange(beg+4, end, 5)
plt.xticks(ticks=x, labels=x+1, fontweight='bold')
# plt.xlabel('Image ID', fontweight='bold', fontsize=ls)
plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='k', labelsize=ls, length=ticklength, 
                width=tickwidth)

# d=0.3
# y=np.arange(3)
# yticks=[y[0]-d, y[1], y[2]+d]
# ylab=y+1
plt.yticks(ticks=[], labels=[], fontweight='bold')
# plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False, labelsize=11, length=ticklength, 
#                 width=tickwidth, color='k')  

#make grid around squares
lw=1
c='k'
plt.hlines([0.4, 1.4] , beg-0.5, end-0.5, color=c, linewidth=lw)

x=np.arange(beg, end)-0.5
yMin=-0.5
yMax=2.5
plt.vlines(x, yMin, yMax, color=c, linewidth=lw)

cbar=plt.colorbar(orientation="horizontal")
cbar.set_ticks(np.arange(5))
# cbar.set_ticklabels(np.arange(5), fontdict={'fontweight': 'bold'})

#save it so illustrator likes it
plt.savefig("SupFig1/Heatmap.pdf", transparent=True)

plt.show()


#%%
for i in np.arange(3)+1:
    print(i)



#%%

# Defining index for the dataframe
idx = ['1', '2', '3', '4']
  
# Defining columns for the dataframe
cols = list('ABCD')
  
# Entering values in the index and columns  
# and converting them into a panda dataframe
df = pd.DataFrame([[10, 20, 30, 40], [50, 30, 8, 15],
                   [25, 14, 41, 8], [7, 14, 21, 28]],
                   columns = cols, index = idx)
  
# Displaying dataframe as an heatmap
# with diverging colourmap as RdYlBu
plt.imshow(df, cmap ="RdYlBu")
  
# Displaying a color bar to understand
# which color represents which range of data
plt.colorbar()
  
# Assigning labels of x-axis 
# according to dataframe
plt.xticks(range(len(df)), df.columns)
  
# Assigning labels of y-axis 
# according to dataframe
plt.yticks(range(len(df)), df.index)
  
# Displaying the figure
plt.show()

#%%graph my scores vs manual 
#load auto
recordDir='InfRuns'
newRun = '2021_05_28'
path=os.path.join(recordDir, newRun, newRun+'.csv')
autoFull=pd.read_csv(path, index_col=False)
auto=autoFull.reindex(columns=['GT', 'sliceID', 'globScore'])
auto.columns=['GT', 'picID', 'auto']

#%%merge w/ manual 
key=['GT', 'picID']
manAuto=pd.merge(gabby, auto, on=key, how='inner')
manAuto.columns=['GT', 'picID', 'manual', 'auto']

#%%stats - RMA linear regression 
# x=matchedGS.man_inf
# y=matchedGS.auto_inf

x=manAuto.manual
y=manAuto.auto

results = regress2(x, y, _method_type_2="reduced major axis")

astro_m=results['slope']
astro_b=results['intercept']
astro_SD=results['std_intercept']
astro_r=results['r']

astro_x=np.linspace(-0.2, 4.3, 10)
astro_y=astro_m*astro_x+astro_b

astro_r2=astro_r**2
astro_SE=astro_SD/np.sqrt(len(astro_x))

 # t = m/se(m)
t=astro_m/astro_SE

pval = stats.t.sf(np.abs(t), len(x)-1)*2

# https://www.socscistatistics.com/pvalues/tdistribution.aspx
# p-value is < .00001

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

#%%auto v manual 
GS0=manAuto.auto[manAuto.manual==0]
GS1=manAuto.auto[manAuto.manual==1]
GS2=manAuto.auto[manAuto.manual==2]
GS3=manAuto.auto[manAuto.manual==3]
GS4=manAuto.auto[manAuto.manual==4]

print(len(GS0))
print(len(GS1))
print(len(GS2))
print(len(GS3))
print(len(GS4))


#%%graph correlation auto-manual 
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
plt.plot(astro_x, astro_y, c='k', linewidth=4, linestyle='dotted')

#means and error
widths=[0.22, 0.24, 0.22, 0.21, 0.13]

for gs, scores, dx, in zip(x, data, widths):
    plt.hlines(np.mean(scores), gs-dx, gs+dx, color='k', linewidth=barlw)
    plt.errorbar(gs, np.mean(scores), yerr=sem(scores), elinewidth=3, capsize=8, ecolor='k')


# #make whiskers fatter
# dx=0.2
# x=1
# plt.hlines(np.mean(GS0), x-dx, x+dx, color='red', linewidth=barlw)

# dx=0.12
# x=2
# plt.hlines(np.mean(GS1), x-dx, x+dx, color='red', linewidth=barlw)

# dx=0.2
# x=3
# plt.hlines(np.mean(GS2), x-dx, x+dx, color='red', linewidth=barlw)

# dx=0.2
# x=3
# plt.hlines(np.mean(GS3), x-dx, x+dx, color='red', linewidth=barlw)

# dx=0.2
# x=3
# plt.hlines(np.mean(GS2), x-dx, x+dx, color='red', linewidth=barlw)

##anotations
# tx=1.7
# ty=20
# dy=-1.5
# plt.text(tx, ty, 'slope=' + str(round(astro_m, 3)) + '\n95%CI[' + str(round(mL, 3)) +', ' + str(round(mH, 3)) + ']', fontsize=14, c='k')
# plt.text(tx, ty+dy, '$r^2$=' + str(round(astro_r2, 2)), fontsize=14)
# plt.text(tx, ty+2*dy, 'p<0.0001', fontsize=14)

##axes titles
# plt.title('Globularity Validation', fontsize=18, c='k', fontweight='bold')
# plt.xlabel('manual score', fontsize=18, c='k', fontweight='bold')
# plt.ylabel('automated score \n (perimeter/cortex area)', fontsize=18, c='k', fontweight='bold')
plt.ylim(top=23)

plt.xticks(x, fontweight='bold')
plt.yticks(color='k', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=True,  top=False, labelbottom=True, color='k', 
                labelsize=16, length=ticklength, width=tickwidth)
plt.tick_params(axis='y', which='both', labelsize=tickfont, length=ticklength, 
  width=tickwidth, color='k')  

plt.show()

#%%manual scores#####################
WT=manAuto.manual[manAuto.GT=='6L']
Spz3=manAuto.manual[manAuto.GT=='7L']
aSNAP=manAuto.manual[manAuto.GT=='8L']

#sig - astro
print('Averages')
print(np.average(WT))
print(np.average(Spz3))
print(np.average(aSNAP))
print()

#stats - kruskal-wallis
#auto
iScore=[WT, Spz3, aSNAP]
H, pKW =kruskal(WT, Spz3, aSNAP)
print('pKW', pKW)
print('auto')
dunn=sp.posthoc_dunn(iScore)
print(dunn)
print()

#comparing means
f, p = f_oneway(WT, Spz3, aSNAP)
print(p)

#Tukey
iScore=manAuto.manual
group=manAuto.GT

comp=statsmodels.stats.multicomp.MultiComparison(iScore, group)

ph=comp.tukeyhsd(alpha=0.05)

print(ph)


#%%Glob Score counts for everything GRAPH
bwidth=0.27
cap=5
tickfont=16
ticklength=6
tickwidth=4
barlw=3
ms=60

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


#%%auto scores#####################
WT=manAuto.auto[manAuto.GT=='6L']
Spz3=manAuto.auto[manAuto.GT=='7L']
aSNAP=manAuto.auto[manAuto.GT=='8L']

#sig - astro
print('Averages')
print(np.average(WT))
print(np.average(Spz3))
print(np.average(aSNAP))
print()

#stats
#stats - kruskal-wallis
#auto
iScore=[WT, Spz3, aSNAP]
H, pKW =kruskal(WT, Spz3, aSNAP)
print('pKW', pKW)
print('auto')
dunn=sp.posthoc_dunn(iScore)
print(dunn)
print()

#comparing means
f, p = f_oneway(WT, Spz3, aSNAP)
print(p)

#Tukey
iScore=manAuto.manual
group=manAuto.GT

comp=statsmodels.stats.multicomp.MultiComparison(iScore, group)

ph=comp.tukeyhsd(alpha=0.05)

print(ph)


#%%Glob Score counts for everything GRAPH
bwidth=0.27
cap=5
tickfont=16
ticklength=6
tickwidth=4
barlw=2
ms=60

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












