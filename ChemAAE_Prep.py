#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:28:22 2019

@author: sameermac
"""

#High Dimensional Unsupervised Learning on Organic Chemicals

#from __future__ import print_function
#import csv
#import math
#import random


#from tqdm import tqdm.tqdm
#for i in tqdm(l):
#...stuff
#joblib

###                 Loaded Modules                        ###

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
from sklearn.manifold import MDS


#Note: %matplotlib auto  - Put this in the IPython console to get full figures

###     Unsupervised Learning Framework - This file     ###

#1) First the molecules are loaded (here done locally) - SDF Format
#2) Unconvertible molecules are removed
#3) Conversion to SMILES and Fingerprint Formats for computations and Filtering

#Filters Applied to the Molecular Set

#4) Inorganic/Toxic Molecules are filtered : Captured then Removed
#5) Filtering basedd off of Lipinski Rule of Five: Molecules < 500 Dalton

#Unsupervised Learning - Clustering

#6) Compute K-medoid Clusters (Labels): Vary Hyperparameters Clusters, Iterations
#7) Apply K-medoid Labels to High-Dimensional Manifold Learning Methods (MDS, TSNE, etc.)

#Applying the B-VAE Autoencoder Framework

#8) Apply the B-VAE network




###               Loading Molecules                ###

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
                          '[Og','[Lu','[Lr','b',
                          
                          'Li]','Na]','K]','Rb]','Cs]','Fr]',
                          'Be]','Mg]','Ca]','Sr]', 'Ba]', 'Ra]',
                          'Sc]','Y]',
                          'Ti]','Zr]','Hf]','Rf]','La]','Ac]',
                          'V]','Nb]','Ta]','Db]','Ce]','Th]',
                          'Cr]','Mo]','W]','Sg]','Pr]','Pa]',
                          'Mn]','Tc]','Re]','Bh]','Nd]','U]',
                          'Fe]','Ru]','Os]','Hs]','Pm]','Np]',
                          'Co]','Rh]','Ir]','Mt]','Sm]','Pu]',
                          'Ni]','Pd]','Pt]','Ds]','Eu]','Am]',
                          'Cu]','Ag]','Au]','Rg]','Gd]','Cm]',
                          'Zn]','Cd]','Hg]','Cn]','Tb]','Bk]',
                          'Al]','Ga]','In]','Tl]','Nh]','Dy]','Cf]',
                          'Si]','Ge]','Sn]','Pb]','Fl]','Ho]','Es]',
                          'As]','Sb]','Bi]','Mc]','Er]','Fm]',
                          'Se]','Te]','Po]','Lv]','Tm]','Md]',
                          'Br','I','At]','Ts]','Yb]','No]',
                          'He]','Ne]','Ar]','Kr]','Xe]','Rn]',
                          'Og]','Lu]','Lr]']
                          
                          #'.[Li','.[Na','.[K','.[Rb','.[Cs','.[Fr',
                          #'.[Be','.[Mg','.[Ca','.[Sr', '.[Ba', '.[Ra',
                          #'.[Sc','.[Y',
                          #'.[Ti','.[Zr','.[Hf','.[Rf','.[La','.[Ac',
                          #'.[V','.[Nb','.[Ta','.[Db','.[Ce','.[Th',
                          #'.[Cr','.[Mo','.[W','.[Sg','.[Pr','.[Pa',
                          #'.[Mn','.[Tc','.[Re','.[Bh','.[Nd','.[U',
                          #'.[Fe','.[Ru','.[Os','.[Hs','.[Pm','.[Np',
                          #'.[Co','.[Rh','.[Ir','.[Mt','.[Sm','.[Pu',
                          #'.[Ni','.[Pd','.[Pt','.[Ds','.[Eu','.[Am',
                          #'.[Cu','.[Ag','.[Au','.[Rg','.[Gd','.[Cm',
                          #'.[Zn','.[Cd','.[Hg','.[Cn','.[Tb','.[Bk',
                          #'.[B','.[Al','.[Ga','.[In','.[Tl','.[Nh','.[Dy','.[Cf',
                          #'.[Si','.[Ge','.[Sn','.[Pb','.[Fl','.[Ho','.[Es',
                          #'.[As','.[Sb','.[Bi','.[Mc','.[Er','.[Fm',
                          #'.[Se','.[Te','.[Po','.[Lv','.[Tm','.[Md',
                          #'.[Br','.[I','.[At','.[Ts','.[Yb','.[No',
                          #'.[He','.[Ne','.[Ar','.[Kr','.[Xe','.[Rn',
                          #'.[Og','.[Lu','.[Lr','.[b']


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



###########              K-medoid Clustering                     ##############

#Computing K-medoids based on Tanimoto Distance

#Randomly select K as the medoids for n data points


##### Select Hyperparameters #####

#K clusters
n = 10

#Iterations
Iters = 10

##### ^Select Hyperparameters^ #####




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

#Updating the Medoids - Multiple Iterations

for p in range(Iters):
    
    #Updating the Medoids - 1 Iteration 
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
    
#Making Cluster Label Matrix - Cluster Labels appended with Molecules
    
ClusterJoin = np.column_stack((ClustersLabels,mol_Filtered_Lipinski))

#Calling Desired Cluster (Starting with 0th) - Change as necessary in IDE

Choke = []
for i in range(MSL):
    if ClusterJoin[i][0] == 0:
        Choke.append(ClusterJoin[i])




##Filter by further criteria - see below for individual functions - 10 Samples Only based on Position
#Analyte1 = Choke[0][1]
#print('Molecular Weight of Analyte 1:', Chem.Descriptors.ExactMolWt(Analyte1))
#print('Log P Value of Analyte 1:', Chem.Descriptors.MolLogP(Analyte1))
#print('Polar Surface Area of Analyte 1:', Chem.Descriptors.TPSA(Analyte1))
#print('H-bond Donors of Analyte 1:', Chem.rdMolDescriptors.CalcNumHBD(Analyte1))
#print('(Lipinski) H-bond Donors of Analyte 1:', Chem.rdMolDescriptors.CalcNumLipinskiHBD(Analyte1))
#print('H-bond Acceptors of Analyte 1:', Chem.rdMolDescriptors.CalcNumHBA(Analyte1))
#print('(Lipinski) H-bond Acceptors of Analyte 1:', Chem.rdMolDescriptors.CalcNumLipinskiHBA(Analyte1))
#print('Number of Rotatable Bonds of Analyte 1:', Chem.rdMolDescriptors.CalcNumRotatableBonds(Analyte1))
#Analyte1
#
#Analyte2 = Choke[2][1]
#print('Molecular Weight of Analyte 2:', Chem.Descriptors.ExactMolWt(Analyte2))
#print('Log P Value of Analyte 2:', Chem.Descriptors.MolLogP(Analyte2))
#print('Polar Surface Area of Analyte 2:', Chem.Descriptors.TPSA(Analyte2))
#print('H-bond Donors of Analyte 2:', Chem.rdMolDescriptors.CalcNumHBD(Analyte2))
#print('(Lipinski) H-bond Donors of Analyte 2:', Chem.rdMolDescriptors.CalcNumLipinskiHBD(Analyte2))
#print('H-bond Acceptors of Analyte 2:', Chem.rdMolDescriptors.CalcNumHBA(Analyte2))
#print('(Lipinski) H-bond Acceptors of Analyte 2:', Chem.rdMolDescriptors.CalcNumLipinskiHBA(Analyte2))
#print('Number of Rotatable Bonds of Analyte 2:', Chem.rdMolDescriptors.CalcNumRotatableBonds(Analyte2))
#Analyte2
#
#Analyte3 = Choke[4][1]
#print('Molecular Weight of Analyte 3:', Chem.Descriptors.ExactMolWt(Analyte3))
#print('Log P Value of Analyte 3:', Chem.Descriptors.MolLogP(Analyte3))
#print('Polar Surface Area of Analyte 3:', Chem.Descriptors.TPSA(Analyte3))
#print('H-bond Donors of Analyte 3:', Chem.rdMolDescriptors.CalcNumHBD(Analyte3))
#print('(Lipinski) H-bond Donors of Analyte 3:', Chem.rdMolDescriptors.CalcNumLipinskiHBD(Analyte3))
#print('H-bond Acceptors of Analyte 3:', Chem.rdMolDescriptors.CalcNumHBA(Analyte3))
#print('(Lipinski) H-bond Acceptors of Analyte 3:', Chem.rdMolDescriptors.CalcNumLipinskiHBA(Analyte3))
#print('Number of Rotatable Bonds of Analyte 3:', Chem.rdMolDescriptors.CalcNumRotatableBonds(Analyte3))
#Analyte3
#
#Analyte4 = Choke[6][1]
#print('Molecular Weight of Analyte 4:', Chem.Descriptors.ExactMolWt(Analyte4))
#print('Log P Value of Analyte 4:', Chem.Descriptors.MolLogP(Analyte4))
#print('Polar Surface Area of Analyte 4:', Chem.Descriptors.TPSA(Analyte4))
#print('H-bond Donors of Analyte 4:', Chem.rdMolDescriptors.CalcNumHBD(Analyte4))
#print('(Lipinski) H-bond Donors of Analyte 4:', Chem.rdMolDescriptors.CalcNumLipinskiHBD(Analyte4))
#print('H-bond Acceptors of Analyte 4:', Chem.rdMolDescriptors.CalcNumHBA(Analyte4))
#print('(Lipinski) H-bond Acceptors of Analyte 4:', Chem.rdMolDescriptors.CalcNumLipinskiHBA(Analyte4))
#print('Number of Rotatable Bonds of Analyte 4:', Chem.rdMolDescriptors.CalcNumRotatableBonds(Analyte4))
#Analyte4
#
#Analyte5 = Choke[int(np.floor(len(Choke)/2) - 2)][1]
#print('Molecular Weight of Analyte 5:', Chem.Descriptors.ExactMolWt(Analyte5))
#print('Log P Value of Analyte 5:', Chem.Descriptors.MolLogP(Analyte5))
#print('Polar Surface Area of Analyte 5:', Chem.Descriptors.TPSA(Analyte5))
#print('H-bond Donors of Analyte 5:', Chem.rdMolDescriptors.CalcNumHBD(Analyte5))
#print('(Lipinski) H-bond Donors of Analyte 5:', Chem.rdMolDescriptors.CalcNumLipinskiHBD(Analyte5))
#print('H-bond Acceptors of Analyte 5:', Chem.rdMolDescriptors.CalcNumHBA(Analyte5))
#print('(Lipinski) H-bond Acceptors of Analyte 5:', Chem.rdMolDescriptors.CalcNumLipinskiHBA(Analyte5))
#print('Number of Rotatable Bonds of Analyte 5:', Chem.rdMolDescriptors.CalcNumRotatableBonds(Analyte5))
#Analyte5
#
#Analyte6 = Choke[int(len(Choke)/2)][1]
#print('Molecular Weight of Analyte 6:', Chem.Descriptors.ExactMolWt(Analyte6))
#print('Log P Value of Analyte 6:', Chem.Descriptors.MolLogP(Analyte6))
#print('Polar Surface Area of Analyte 6:', Chem.Descriptors.TPSA(Analyte6))
#print('H-bond Donors of Analyte 6:', Chem.rdMolDescriptors.CalcNumHBD(Analyte6))
#print('(Lipinski) H-bond Donors of Analyte 6:', Chem.rdMolDescriptors.CalcNumLipinskiHBD(Analyte6))
#print('H-bond Acceptors of Analyte 6:', Chem.rdMolDescriptors.CalcNumHBA(Analyte6))
#print('(Lipinski) H-bond Acceptors of Analyte 6:', Chem.rdMolDescriptors.CalcNumLipinskiHBA(Analyte6))
#print('Number of Rotatable Bonds of Analyte 6:', Chem.rdMolDescriptors.CalcNumRotatableBonds(Analyte6))
#Analyte6
#
#Analyte7 = Choke[int(len(Choke)/2 + 2)][1]
#print('Molecular Weight of Analyte 7:', Chem.Descriptors.ExactMolWt(Analyte7))
#print('Log P Value of Analyte 7:', Chem.Descriptors.MolLogP(Analyte7))
#print('Polar Surface Area of Analyte 7:', Chem.Descriptors.TPSA(Analyte7))
#print('H-bond Donors of Analyte 7:', Chem.rdMolDescriptors.CalcNumHBD(Analyte7))
#print('(Lipinski) H-bond Donors of Analyte 7:', Chem.rdMolDescriptors.CalcNumLipinskiHBD(Analyte7))
#print('H-bond Acceptors of Analyte 7:', Chem.rdMolDescriptors.CalcNumHBA(Analyte7))
#print('(Lipinski) H-bond Acceptors of Analyte 7:', Chem.rdMolDescriptors.CalcNumLipinskiHBA(Analyte7))
#print('Number of Rotatable Bonds of Analyte 7:', Chem.rdMolDescriptors.CalcNumRotatableBonds(Analyte7))
#Analyte7
#
#Analyte8 = Choke[-4][1]
#print('Molecular Weight of Analyte 8:', Chem.Descriptors.ExactMolWt(Analyte8))
#print('Log P Value of Analyte 8:', Chem.Descriptors.MolLogP(Analyte8))
#print('Polar Surface Area of Analyte 8:', Chem.Descriptors.TPSA(Analyte8))
#print('H-bond Donors of Analyte 8:', Chem.rdMolDescriptors.CalcNumHBD(Analyte8))
#print('(Lipinski) H-bond Donors of Analyte 8:', Chem.rdMolDescriptors.CalcNumLipinskiHBD(Analyte8))
#print('H-bond Acceptors of Analyte 8:', Chem.rdMolDescriptors.CalcNumHBA(Analyte8))
#print('(Lipinski) H-bond Acceptors of Analyte 8:', Chem.rdMolDescriptors.CalcNumLipinskiHBA(Analyte8))
#print('Number of Rotatable Bonds of Analyte 8:', Chem.rdMolDescriptors.CalcNumRotatableBonds(Analyte8))
#Analyte8
#
#Analyte9 = Choke[-2][1]
#print('Molecular Weight of Analyte 9:', Chem.Descriptors.ExactMolWt(Analyte9))
#print('Log P Value of Analyte 9:', Chem.Descriptors.MolLogP(Analyte9))
#print('Polar Surface Area of Analyte 9:', Chem.Descriptors.TPSA(Analyte9))
#print('H-bond Donors of Analyte 9:', Chem.rdMolDescriptors.CalcNumHBD(Analyte9))
#print('(Lipinski) H-bond Donors of Analyte 9:', Chem.rdMolDescriptors.CalcNumLipinskiHBD(Analyte9))
#print('H-bond Acceptors of Analyte 9:', Chem.rdMolDescriptors.CalcNumHBA(Analyte9))
#print('(Lipinski) H-bond Acceptors of Analyte 9:', Chem.rdMolDescriptors.CalcNumLipinskiHBA(Analyte9))
#print('Number of Rotatable Bonds of Analyte 9:', Chem.rdMolDescriptors.CalcNumRotatableBonds(Analyte9))
#Analyte9
#
#Analyte10 = Choke[-1][1]
#print('Molecular Weight of Analyte 10:', Chem.Descriptors.ExactMolWt(Analyte10))
#print('Log P Value of Analyte 10:', Chem.Descriptors.MolLogP(Analyte10))
#print('Polar Surface Area of Analyte 10:', Chem.Descriptors.TPSA(Analyte10))
#print('H-bond Donors of Analyte 10:', Chem.rdMolDescriptors.CalcNumHBD(Analyte10))
#print('(Lipinski) H-bond Donors of Analyte 10:', Chem.rdMolDescriptors.CalcNumLipinskiHBD(Analyte10))
#print('H-bond Acceptors of Analyte 10:', Chem.rdMolDescriptors.CalcNumHBA(Analyte10))
#print('(Lipinski) H-bond Acceptors of Analyte 10:', Chem.rdMolDescriptors.CalcNumLipinskiHBA(Analyte10))
#print('Number of Rotatable Bonds of Analyte 10:', Chem.rdMolDescriptors.CalcNumRotatableBonds(Analyte10))
#Analyte10

#ShaperFull = np.shape(Choke)
#ShaperTarget = ShaperFull[0]
#print('The given cluster has', ShaperTarget, 'molecules')

#Example Label in IDE: #K-medoids clustering (50 clusters) [0 - 49] : Cluster 2

#########  Molecular Description Tools for Calculation #####################

#MOLECULAR_WEIGHT 
#Chem.Descriptors.ExactMolWt(mol)


#JCHEM_DONOR_COUNT 
#rdkit.Chem.rdMolDescriptors.CalcNumHBD((Mol)mol) → int :
#rdkit.Chem.rdMolDescriptors.CalcNumLipinskiHBD((Mol)mol) → int :
#Chem.Descriptors.NumHDonors

#ALOGPS_LOGP
#rdkit.Chem.rdMolDescriptors.CalcCrippenDescriptors((Mol)mol[, (bool)includeHs=True[, (bool)force=False]]) → tuple :
#Chem.Descriptors.MolLogP <-- Use this one

#JCHEM_POLAR_SURFACE_AREA
#Chem.Descriptors.TPSA
#Chem.rdMolDescriptors.CalcTPSA

#JCHEM_ACCEPTOR_COUNT

#rdkit.Chem.rdMolDescriptors.CalcNumHBA((Mol)mol) → int :

#rdkit.Chem.rdMolDescriptors.CalcNumLipinskiHBA((Mol)mol) → int :
    
#Chem.Descriptors.NumHAcceptors

#>>> from rdkit.Chem import Descriptors
#>>> m = Chem.MolFromSmiles('c1ccccc1C(=O)O')
#>>> Descriptors.TPSA(m)
#37.3
#>>> Descriptors.MolLogP(m)
#1.3848
    

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

##MDS3D = manifold.MDS(n_components=3, dissimilarity="precomputed", random_state=3, n_jobs = 4, verbose=1,max_iter=1000)
##results3D = MDS3D.fit(TDM)
##coords3D = results3D.embedding_
##print('Final stress:',MDS3D.stress_)


#t-Stochastic Nearest Embedding using TDM (2D) ---------------------------------------

#TDM_TSNE = manifold.TSNE(n_components=2,metric = "precomputed").fit_transform(TDM)

#To view all these methods in 2D use this analogous method

#X_coord_TSNE = [x for x,y in TDM_TSNE]

#Y_coord_TSNE = [y for x,y in TDM_TSNE]

##TDM_TSNE = manifold.TSNE(n_components=3,metric = "precomputed").fit_transform(TDM)

##DX_TSNE = [x for x,y,z in TDM_TSNE]

##DY_TSNE = [y for x,y,z in TDM_TSNE]

##DZ_TSNE = [z for x,y,z in TDM_TSNE]

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




        
