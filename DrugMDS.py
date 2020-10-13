#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 23:39:07 2018

@author: sameermac
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:04:21 2018

@author: sameermac
"""



###                 Loaded Modules                        ###


#from __future__ import print_function
#import csv
#import math
#import random

#from tqdm import tqdm.tqdm
#for i in tqdm(l):
#...stuff
#joblib


#from ggplot import *
#from mpl_toolkits.mplot3d import Axes3d
from operator import itemgetter
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
import random
from rdkit.Chem import Descriptors
import requests
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
from rdkit.Chem import DataStructs
from scipy.spatial.distance import *
from sklearn import manifold
from rdkit import rdBase
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
import matplotlib.colors as colors
import seaborn as sns
from scipy import stats

#from pubchempy import get_compounds, Compound
#WHY WON'T IT INSTALL VIA CONDA AS INSTRUCTED


###                 Loaded Modules                        ###


MOL_open = open( '/Users/sameermac/Desktop/structures_DRUGBANK_approved.sdf','rb')
#MOL_open = open( '/Users/sameermac/Desktop/Thesis/gdb9.sdf','rb')
MOL_LIST = Chem.ForwardSDMolSupplier(MOL_open)

mol = [x for x in MOL_LIST if x is not None]
#Some elements empty because there are invalid molecules =/> SMILES Format

mol_FULL = mol

Smile_mol_FULL = [Chem.MolToSmiles(m) for m in mol_FULL]

InorganicPeriodicTable = ['[Li','[Na','[K','[Rb','[Cs','[Fr',
                          '[Be','[Mg','[Ca','[Sr', '[Ba', '[Ra',
                          '[Sc','[Y',
                          '[Ti','[Zr','[Hf','[Rf','[La','[Ac',
                          '[V','[Nb','[Ta','[Db','[Ce','[Th',
                          '[Cr','[Mo','[W','[Sg','[Pr','[Pa',
                          '[Mn','[Tc','[Re','[Bh','[Nd','[U',
                          '[Fe','[Ru','[Os','[Hs','[Pm','[Np',
                          '[Co','[Rh','[Ir','[Mt','[Sm','[Pu',
                          '[Ni','[Pd','[Pt','[Ds','[Eu','[Am',
                          '[Cu','[Ag','[Au','[Rg','[Gd','[Cm',
                          '[Zn','[Cd','[Hg','[Cn','[Tb','[Bk',
                          'B','[Al','[Ga','[In','[Tl','[Nh','[Dy','[Cf',
                          '[Si','[Ge','[Sn','[Pb','[Fl','[Ho','[Es',
                          '[As','[Sb','[Bi','[Mc','[Er','[Fm',
                          '[Se','[Te','[Po','[Lv','[Tm','[Md',
                          'Br','I','[At','[Ts','[Yb','[No',
                          '[He','[Ne','[Ar','[Kr','[Xe','[Rn',
                          '[Og','[Lu','[Lr','b']

#IssueMolecules = ['Cs','Sc','Np','Co','Ho','Sn']

#Eliminating any molecules that contain atoms other than:
#C,H,O,N,S,F,Cl,P

#Capturing "Bad" Molecules as a subset
Smile_mol_Captured = []

#If Further Filtering Needed

for j in range(len(Smile_mol_FULL)):
    for i in range(len(InorganicPeriodicTable)):
       if InorganicPeriodicTable[i] in Smile_mol_FULL[j]:
            Smile_mol_Captured.append(Smile_mol_FULL[j])
            
#Re-filtering to target leftover valid configurations
       
#Removing "Bad" Molecules from the original superset 

Smile_mol_Filtered = [m for m in Smile_mol_FULL if m not in Smile_mol_Captured]

#Checking if properly filtered - Undesirable Atoms Seperated

#Check = [i for i in Smile_mol_Filtered if i in Smile_mol_Captured]

#Convert to Mol Data Structures

#Original
#Mol_From_Smile_FULL = [Chem.MolFromSmiles(m) for m in Smile_mol_FULL]

Mol_From_Smile_FULL = [Chem.MolFromSmiles(m) for m in Smile_mol_Filtered]


#Mol_From_Smile_FULL --> Now Smile_mol_Filtered
Lipinski_Over_500 = []
Lipinski_Over_500_SMILE = []
for i in range(len(Smile_mol_Filtered)):
    if Chem.Descriptors.ExactMolWt(Mol_From_Smile_FULL[i]) > 500:
        Lipinski_Over_500.append(Mol_From_Smile_FULL[i])
        Lipinski_Over_500_SMILE.append(Smile_mol_FULL[i])

#Remove 500+ Dalton Molecules from set

#Preserve Smile Structures for Visualization if desired
        
Smile_mol_Filtered_Lipinski = [m for m in Smile_mol_FULL if m not in Lipinski_Over_500_SMILE]

#Data File (prior to conversion to Fingerprint)

mol_Filtered_Lipinski = [m for m in Mol_From_Smile_FULL if m not in Lipinski_Over_500]

#Generating Array Containing Strings of Known Molecules with respect to SMILES Representation

#We use PubChem Framework to do this

#Convert to Molecular Fingerprint Format


#Mol_From_Smile_FULL --> mol_Filtered_Lipinski
finTanArray = [FingerprintMols.FingerprintMol(x) for x in mol_Filtered_Lipinski]

#Molecule Sequence (List) Length
MSL = len(finTanArray)
MSL_String = str(MSL)

#Generating Tanimoto Distance Matrix

TDA = []
for i in range(MSL):
    for j in range(MSL):
            TDA.append(1 - DataStructs.FingerprintSimilarity(finTanArray[i], finTanArray[j]))

#This produces as a single MSL x 1 list : We transform --> into a MSL x MSL matrix
            
TDM_list = np.array(TDA)
TDM_matdim = (MSL, MSL)
TDM = TDM_list.reshape(TDM_matdim)

###########              Manifold Learning Methods               ##############



#3D Plotting Example
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter([x for x,y,z in coords3D], [y for x,y,z in coords3D],[z for x,y,z in coords3D], 
#c = ClustersLabels, cmap=plt.cm.Spectral)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(DX_TSNE, DY_TSNE,DZ_TSNE, c=ClustersLabels, cmap=plt.cm.Spectral)

#Multidimensional Scaling using TDM (2 Dimensions)

#MDS = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=3, n_jobs = 4, verbose=1,max_iter=1000)
#results = MDS.fit(TDM)
#coords = results.embedding_
#print('Final stress:',MDS.stress_)


#Multidimensional Scaling using TDM (3 Dimensions)

MDS3D = manifold.MDS(n_components=3, dissimilarity="precomputed", random_state=3, n_jobs = 4, verbose=1,max_iter=1000)
results3D = MDS3D.fit(TDM)
coords3D = results3D.embedding_
print('Final stress:',MDS3D.stress_)


#t-Stochastic Nearest Embedding using TDM (2D) ---------------------------------------

#TDM_TSNE = manifold.TSNE(n_components=2,metric = "precomputed").fit_transform(TDM)

#To view all these methods in 2D use this analogous method

#X_coord_TSNE = [x for x,y in TDM_TSNE]

#Y_coord_TSNE = [y for x,y in TDM_TSNE]

TDM_TSNE = manifold.TSNE(n_components=3,metric = "precomputed").fit_transform(TDM)

DX_TSNE = [x for x,y,z in TDM_TSNE]

DY_TSNE = [y for x,y,z in TDM_TSNE]

DZ_TSNE = [z for x,y,z in TDM_TSNE]

#plt.scatter(X_coord_TSNE, Y_coord_TSNE, c = np.arange(MSL), cmap=plt.cm.hot)

#---------------------------------------------




#Isomap using TDM (2D)

#TDM_Iso_Embed = manifold.Isomap(n_components=2)
#TDM_Iso_transformed = TDM_Iso_Embed.fit_transform(TDM)
#plt.scatter([x for x,y in TDM_Iso_transformed], [y for x,y in TDM_Iso_transformed],edgecolors='none')


#Locally Linear Embedding using TDM (2D)

#TDM_LLE_Embed = manifold.LocallyLinearEmbedding(n_components=2)
#TDM_LLE_transformed = TDM_LLE_Embed.fit_transform(TDM)
#plt.scatter([x for x,y in TDM_LLE_transformed], [y for x,y in TDM_LLE_transformed],edgecolors='none')



#Principal Component Analysis using TDM - Possibly to be used later




#Examining - For later use possibly

#Redesigned distance matrix based on 1/S - 1 with tolerance, where S is similarity
#TSM = 1 - TDM
#eps = 1e-4
#customDM = 1/(TSM + eps) - 1/(1+eps)
#
#
#MDS2 = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=3, n_jobs = 4, verbose=1,max_iter=1000)
#results2 = MDS2.fit(customDM)
#coords2 = results2.embedding_
#print('Final stress:',MDS2.stress_)


#Using Distance Matrix given non-metric status (To accomodate)



#Frequency Analysis of TDM
#sns.set(color_codes=True)
#sns.distplot(TDM_list) 

#Note: We use the array and not the matrix to avoid multi-dimensional array issues






















#Check all - Plotting Methods - To get meaningful color maps

#plt.scatter([x for x,y in TDM_TSNE], [y for x,y in TDM_TSNE],edgecolors='none')

#plt.scatter([x for x,y in TDM_TSNE], [y for x,y in TDM_TSNE],c='hot', cmap=plt.cm.Spectral)

#c = colors


#Examining Indices that correspond to the SMILE Representation against the FingerPrint Representation

#K_medoid_Smile = #[Chem.MolToSmiles(m) for m in K_medoid]

#for j in range(MSL):
#    for i in range(len(K_medoid)):

#Min = min(TanDistSums)

#KMinIndex = min(enumerate(TanDistSums), key=itemgetter(1))[0] 
    
#K_medoid_Update = K_medoid[KMinIndex] 


#if K_medoid[i] != finTanArray[j]:




#TanMedMat = np.reshape(TanDist, len(K_medoid),len(finTanArray)- len(K_medoid))



                
#for i in range(len(K_medoid)):
    




#Updating the medoid

#for i in range(len(K_medoid)):
#    if Tandist[i] = min(TanDist):
#        K_medoid
        
#Convergence Criteria for K-medoid
                


#Computing SC Clustering with Medoids

#-----------------------------------------------------
    
#i = 0
#j = 0
#while(i < len(Smile_mol_Filtered)):
#    while(j < len(InorganicPeriodicTable)):
#         if InorganicPeriodicTable[i] in Smile_mol_Filtered[j]:
#             Smile_mol_Filtered.remove(Smile_mol_Filtered[j])
#             j = j + 1
#    i = i + 1            
            #Takes too long to complete



