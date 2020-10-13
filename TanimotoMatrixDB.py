#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:08:46 2018

@author: sameermac
"""

#Computing the Tanimoto Matrix and Analyzing the results 

#from __future__ import print_function
#import csv
#import math
#import random


#from tqdm import tqdm.tqdm
#for i in tqdm(l):
#...stuff
#joblib


from scipy.cluster.hierarchy import linkage, dendrogram
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdMolAlign, rdShapeHelpers
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering



#from rdkit.Chem import Descriptors - May need later


#If we wish to append the arrays later (piecewise generation of the matrix)

#mol_1000 = mol_FULL[0:1000]
#mol_2000 = mol_FULL[1000:2000]
#mol_3000 = mol_FULL[2000:3000]
#mol_4000 = mol_FULL[3000:4000]
#mol_5000 = mol_FULL[4000:5000]

#Use in the code later if necessary



#Reading molecules from SDF file

MOL_open = open( '/Users/sameermac/Desktop/structures_DRUGBANK_approved.sdf','rb')
MOL_LIST = Chem.ForwardSDMolSupplier(MOL_open)
mol = [x for x in MOL_LIST if x is not None]
#Some elements empty because there are invalid molecules =/> SMILES Format



mol_FULL = mol

#Converting to SMILE and non-SMILE formats

Smile_mol_FULL = [Chem.MolToSmiles(m) for m in mol_FULL]

Mol_From_Smile_FULL = [Chem.MolFromSmiles(m) for m in Smile_mol_FULL]

finTanArray = [FingerprintMols.FingerprintMol(x) for x in Mol_From_Smile_FULL]


#Generating the Tanimoto Similarity Matrix

#Molecule Sequence Length
MSL = len(mol_FULL)
MSL_String = str(MSL)

#TanimotoSimilarity Matrix Generation and Conversion to TanimotoDistance Matrix Generation
TDA = []
for i in range(MSL):
    for j in range(MSL):
            TDA.append(1 - DataStructs.FingerprintSimilarity(finTanArray[i], finTanArray[j]))

#This produces as a single MSL x 1 list : We transform --> into a MSL x MSL matrix
            
TDM_list = np.array(TDA)
TDM_matdim = (MSL, MSL)
TDM = TDM_list.reshape(TDM_matdim)

#Computing Tanimoto Averages (Row by Row)

#(Vectorized) Averaging - TDM * [Ones(Matdim) * 1/MSL]

#Adjusted Average omits one value in average (essentially ignoring molecule x = molecule x)

#Method: molecule x = molecule x --> Tanimoto Distance = 0, so 0 in an average 
#with one point removed removes this molecule in the average count

AvgVec = (1/MSL) *  np.ones(MSL)

AdjVec = (1/(MSL-1)) * np.ones(MSL)

TDM_Row_Avg = np.matmul(TDM,AvgVec)

TDM_Row_Avg_Distinct = np.matmul(TDM,AdjVec)

#Analyzing Tanimoto Results 

TDM_SuperAverage = np.mean(TDM_Row_Avg)

TDM_SuperAverage_Distinct = np.mean(TDM_Row_Avg_Distinct)


#Line Plots

#Standard 
plt.figure() 
M_Index = np.arange(MSL)
Avg_Vals = TDM_Row_Avg

lines = []
for p in range(len(M_Index)):
    pair=[(M_Index[p],0), (M_Index[p], Avg_Vals[p])]
    lines.append(pair)

linecoll = matcoll.LineCollection(lines)
fig, ax = plt.subplots()
ax.add_collection(linecoll)

plt.scatter(M_Index,Avg_Vals)

plt.title(MSL_String + ' Average TD Values (molecule against other molecules)')

plt.xticks(M_Index)
plt.ylim(0,1)

plt.show()



#Adjusted
plt.figure() 
M_Index = np.arange(MSL)
Avg_Vals_Adj = TDM_Row_Avg_Distinct

lines = []
for q in range(len(M_Index)):
    pair=[(M_Index[q],0), (M_Index[q], Avg_Vals_Adj[q])]
    lines.append(pair)

linecoll = matcoll.LineCollection(lines)
fig, ax = plt.subplots()
ax.add_collection(linecoll)

plt.scatter(M_Index,Avg_Vals_Adj)

plt.title(MSL_String + ' Average (adjusted) TD Values (molecule against other molecules)')

plt.xticks(M_Index)
plt.ylim(0,1)

plt.show()


#Sparsity Pattern of Tanimoto Distance Matrix
plt.figure()
plt.spy(TDM)
plt.title('Sparsity Pattern of Tanimoto Distance Matrix')
plt.show()


#General Matrix Pattern of Tanimoto Distance Matrix
plt.figure()
plt.matshow(TDM)
plt.title('General Matrix Pattern of Tanimoto Distance Matrix')
plt.show()

#Tanimoto K-means and Spectral Clustering

#Selecting 2 clusters
TDM_Kmeans2 = KMeans(n_clusters=2, random_state=0).fit(TDM)
#TDM_Pred = TDM_Kmeans.predict(TDM_Kmeans)
TDM_Kmeans_Labels2 = TDM_Kmeans2.labels_
TDM_Kmeans_Centroids2 = TDM_Kmeans2.cluster_centers_

#-----------------------------------------------

TDM_SC2 = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=0).fit(TDM)
TDM_SC_Labels2 = TDM_SC2.labels_



#Selecting 5 clusters
TDM_Kmeans5 = KMeans(n_clusters=5, random_state=0).fit(TDM)
#TDM_Pred = TDM_Kmeans.predict(TDM_Kmeans)
TDM_Kmeans_Labels5 = TDM_Kmeans5.labels_
TDM_Kmeans_Centroids5 = TDM_Kmeans5.cluster_centers_


#-----------------------------------------------

TDM_SC5 = SpectralClustering(n_clusters=5,assign_labels="discretize",random_state=0).fit(TDM)
TDM_SC_Labels5 = TDM_SC5.labels_

#Selecting 10 clusters
TDM_Kmeans10 = KMeans(n_clusters=10, random_state=0).fit(TDM)
#TDM_Pred = TDM_Kmeans.predict(TDM_Kmeans)
TDM_Kmeans_Labels10 = TDM_Kmeans10.labels_
TDM_Kmeans_Centroids10 = TDM_Kmeans10.cluster_centers_

#-----------------------------------------------

TDM_SC10 = SpectralClustering(n_clusters=10,assign_labels="discretize",random_state=0).fit(TDM)
TDM_SC_Labels10 = TDM_SC10.labels_

#Selecting 100 clusters
TDM_Kmeans100 = KMeans(n_clusters=100, random_state=0).fit(TDM)
#TDM_Pred = TDM_Kmeans.predict(TDM_Kmeans)
TDM_Kmeans_Labels100 = TDM_Kmeans100.labels_
TDM_Kmeans_Centroids100 = TDM_Kmeans100.cluster_centers_

#-----------------------------------------------

TDM_SC100 = SpectralClustering(n_clusters=100, affinity='precomputed', assign_labels="discretize",random_state=0).fit(TDM)
TDM_SC_Labels100 = TDM_SC100.labels_




#Visualizing the Clusters??
#plt.figure()
#plt.scatter(TDM_Kmeans_Centroids[:,0],TDM_Kmeans_Centroids[:,1], c= TDM_Pred)
#plt.title("K-means Clustering of the Tanimoto Distance Matrix")
#plt.plot(TDM_Kmeans[:, 0], TDM_Kmeans[:, 1], 'k.', markersize=2)






















#-----------------------Discarded Code-----------------------


#for avgs in TDM_Row_Avg:
    #plt.axvline(x=avgs, color='k', linestyle='--')
    
    
#TSA = DataStructs.FingerprintSimilarity(finTanArray[i], finTanArray[j])
#TSA.append(DataStructs.FingerprintSimilarity(finTanArray[0], finTanArray[i]))

#Generating the Tanimoto Distance Matrix

#([dim: MSL x MSL | Matrix(ones)] - TanimotoSimilarity Matrix)

