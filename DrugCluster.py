#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:04:21 2018

@author: sameermac
"""

#from __future__ import print_function
#import csv
#import math
#import random


#from tqdm import tqdm.tqdm
#for i in tqdm(l):
#...stuff
#joblib

from operator import itemgetter
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
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
from sklearn.manifold import MDS

MOL_open = open( '/Users/sameermac/Desktop/structures_DRUGBANK_approved.sdf','rb')
#MOL_open = open( '/Users/sameermac/Desktop/Thesis/gdb9.sdf','rb')
MOL_LIST = Chem.ForwardSDMolSupplier(MOL_open)

mol = [x for x in MOL_LIST if x is not None]
#Some elements empty because there are invalid molecules =/> SMILES Format

mol_FULL = mol

#If we wish to append the arrays later (piecewise generation of the matrix)

#mol_1000 = mol_FULL[0:1000]
#mol_2000 = mol_FULL[1000:2000]
#mol_3000 = mol_FULL[2000:3000]
#mol_4000 = mol_FULL[3000:4000]
#mol_5000 = mol_FULL[4000:5000]

#Use in the code later if necessary

#Converting to SMILE and non-SMILE formats

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

Mol_From_Smile_FULL = [Chem.MolFromSmiles(m) for m in Smile_mol_FULL]



Lipinski_Over_500 = []
Lipinski_Over_500_SMILE = []
for i in range(len(Mol_From_Smile_FULL)):
    if Chem.Descriptors.ExactMolWt(Mol_From_Smile_FULL[i]) > 500:
        Lipinski_Over_500.append(Mol_From_Smile_FULL[i])
        Lipinski_Over_500_SMILE.append(Smile_mol_FULL[i])

#Remove 500+ Dalton Molecules from set


#Preserve Smile Structures for Visualization if desired
        
Smile_mol_Filtered_Lipinski = [m for m in Smile_mol_FULL if m not in Lipinski_Over_500_SMILE]

#Data File (prior to conversion to Fingerprint)

mol_Filtered_Lipinski = [m for m in Mol_From_Smile_FULL if m not in Lipinski_Over_500]

#Convert to Molecular Fingerprint Format

finTanArray = [FingerprintMols.FingerprintMol(x) for x in Mol_From_Smile_FULL]

#Molecule Sequence (List) Length
MSL = len(mol_FULL)
MSL_String = str(MSL)

#Computing K-medoids based on Tanimoto Distance

#Randomly select K as the medoids for n data points

#Taking n = 50

#K clusters
n = 10


#Initial K_Medoid Computation
K_medoid= random.sample(finTanArray, n)

TanDist = []
for i in range(len(K_medoid)):
        for j in range(MSL):
                TanDist.append(1 - DataStructs.FingerprintSimilarity(K_medoid[i], finTanArray[j]))


#Computing K-Medoid Distance Sums
                
TanDistDim = (n,MSL)

TanDistMat = np.reshape(TanDist,TanDistDim)

TanDistSums = np.matmul(TanDistMat,np.ones(MSL))

K_medoid_new = random.sample(finTanArray, n)


#Swapping with non-medoid data points
TanDistNew = []
for i in range(len(K_medoid_new)):
        for j in range(MSL):
                TanDistNew.append(1 - DataStructs.FingerprintSimilarity(K_medoid_new[i], finTanArray[j]))

TanDistDimNew = (n,MSL)

TanDistMatNew = np.reshape(TanDistNew,TanDistDimNew)

TanDistSumsNew = np.matmul(TanDistMatNew,np.ones(MSL))


#Updating the Medoids 
TanDistCheckArray = TanDistSums > TanDistSumsNew

IndicesImprove = np.where(TanDistCheckArray)[0]

for i in range(len(TanDistCheckArray)):
    if i in IndicesImprove:
        K_medoid[i] = K_medoid_new[i]


#Labelling the clusters

ClusterCheck = []
ClustersLabels = []
for j in range(MSL):
    for i in range(len(K_medoid)):
        ClusterCheck.append(1 - DataStructs.FingerprintSimilarity(K_medoid[i], finTanArray[j]))
    
        
ClusterDim = (MSL,n)   
ClusterMat = np.reshape(ClusterCheck, ClusterDim)

#--------------------------------------------------

#Assigning Data Points to Medoids
for j in range(MSL):
    ClustersLabels.append(np.argmin(ClusterMat[j,:]))
    






































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



