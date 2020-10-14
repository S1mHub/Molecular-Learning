#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:52:23 2019

@author: sameermac
"""


#We create a new form of clustering based on hierarchical chemical clustering

###                 Loaded Modules                        ###

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
import os

MOL_open = open( '/Users/sameermac/Desktop/structures_DRUGBANK_approved.sdf','rb')
#MOL_open = open( '/Users/sameermac/Desktop/Thesis/gdb9.sdf','rb')
MOL_LIST = Chem.ForwardSDMolSupplier(MOL_open)
#Zinc15 - Try this

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
                          'Og]','Lu]','Lr]','.']
                          
#                          '.[Li','.[Na','.[K','.[Rb','.[Cs','.[Fr',
#                          '.[Be','.[Mg','.[Ca','.[Sr', '.[Ba', '.[Ra',
#                          '.[Sc','.[Y',
#                          '.[Ti','.[Zr','.[Hf','.[Rf','.[La','.[Ac',
#                          '.[V','.[Nb','.[Ta','.[Db','.[Ce','.[Th',
#                          '.[Cr','.[Mo','.[W','.[Sg','.[Pr','.[Pa',
#                          '.[Mn','.[Tc','.[Re','.[Bh','.[Nd','.[U',
#                          '.[Fe','.[Ru','.[Os','.[Hs','.[Pm','.[Np',
#                          '.[Co','.[Rh','.[Ir','.[Mt','.[Sm','.[Pu',
#                          '.[Ni','.[Pd','.[Pt','.[Ds','.[Eu','.[Am',
#                          '.[Cu','.[Ag','.[Au','.[Rg','.[Gd','.[Cm',
#                          '.[Zn','.[Cd','.[Hg','.[Cn','.[Tb','.[Bk',
#                          '.[B','.[Al','.[Ga','.[In','.[Tl','.[Nh','.[Dy','.[Cf',
#                          '.[Si','.[Ge','.[Sn','.[Pb','.[Fl','.[Ho','.[Es',
#                          '.[As','.[Sb','.[Bi','.[Mc','.[Er','.[Fm',
#                          '.[Se','.[Te','.[Po','.[Lv','.[Tm','.[Md',
#                          '.[Br','.[I','.[At','.[Ts','.[Yb','.[No',
#                          '.[He','.[Ne','.[Ar','.[Kr','.[Xe','.[Rn',
#                          '.[Og','.[Lu','.[Lr','.[b']


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
    if Chem.Descriptors.ExactMolWt(Mol_From_Smile_FULL[i]) > 500 or Chem.Descriptors.ExactMolWt(Mol_From_Smile_FULL[i]) < 68:
        Lipinski_Over_500.append(Mol_From_Smile_FULL[i])
        Lipinski_Over_500_SMILE.append(Smile_mol_FULL[i])

#Remove 500+ Dalton Molecules from set

#Preserve Smile Structures for Visualization if desired
        
Smile_mol_Filtered_Lipinski = [m for m in Smile_mol_FULL if m not in Lipinski_Over_500_SMILE]

#Data File (prior to conversion to Fingerprint)

mol_Filtered_Lipinski = [m for m in Mol_From_Smile_FULL if m not in Lipinski_Over_500]

#Generating Array Containing Strings of Known Molecules with respect to SMILES Representation

#Mol_From_Smile_FULL --> mol_Filtered_Lipinski
#finTanArray = [FingerprintMols.FingerprintMol(x) for x in mol_Filtered_Lipinski]

#Molecule Sequence (List) Length
#MSL = len(finTanArray)
#MSL_String = str(MSL)

#Feature Clustering

#MOLECULAR_WEIGHT Filtering
#Chem.Descriptors.ExactMolWt(mol)

WeightClass400_500 = []
WeightClass300_400 = []
WeightClass200_300 = []
WeightClass100_200 = []
WeightClass0_100 = []


for i in range(len(mol_Filtered_Lipinski)):
    if Chem.Descriptors.ExactMolWt(mol_Filtered_Lipinski[i]) < 500 and Chem.Descriptors.ExactMolWt(mol_Filtered_Lipinski[i]) > 400:
        WeightClass400_500.append(mol_Filtered_Lipinski[i])
    elif Chem.Descriptors.ExactMolWt(mol_Filtered_Lipinski[i]) < 400 and Chem.Descriptors.ExactMolWt(mol_Filtered_Lipinski[i]) > 300:
        WeightClass300_400.append(mol_Filtered_Lipinski[i])
    elif Chem.Descriptors.ExactMolWt(mol_Filtered_Lipinski[i]) < 300 and Chem.Descriptors.ExactMolWt(mol_Filtered_Lipinski[i]) > 200:
        WeightClass200_300.append(mol_Filtered_Lipinski[i])  
    elif Chem.Descriptors.ExactMolWt(mol_Filtered_Lipinski[i]) < 200 and Chem.Descriptors.ExactMolWt(mol_Filtered_Lipinski[i]) > 100:
        WeightClass200_300.append(mol_Filtered_Lipinski[i]) 
    else:
        WeightClass100_200.append(mol_Filtered_Lipinski[i]) 
        
        

#FingerprintMols.FingerprintMol(x)
        
        
#Chem.Draw.MolsToGridImage(tuple(WeightClass200_300))
