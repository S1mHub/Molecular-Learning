#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:11:03 2018

@author: sameermac
"""

from operator import itemgetter
from scipy.cluster.hierarchy import linkage, dendrogram
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
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
from scipy.spatial import distance
from sklearn import manifold
from rdkit import rdBase


#This file if for testing tools of RDKIT or conducting basic experiments with software kits

##Determining Whether RDKIT Uses Tanimoto Distance (Jaccard Index) or Tanimoto Distance (Torsion-Based)

#We test a Jaccard Index computation vs. Tanimoto Distance computation


#Loading a Data-Set
MOL_open_TEST = open( '/Users/sameermac/Desktop/structures_DRUGBANK_approved.sdf','rb')
#MOL_open = open( '/Users/sameermac/Desktop/Thesis/gdb9.sdf','rb')
MOL_LIST_TEST = Chem.ForwardSDMolSupplier(MOL_open_TEST)

mol_TEST = [x for x in MOL_LIST_TEST if x is not None]
#Some elements empty because there are invalid molecules =/> SMILES Format

mol_FULL_TEST = mol_TEST

Smile_mol_FULL_TEST = [Chem.MolToSmiles(m) for m in mol_FULL_TEST]

Mol_From_Smile_FULL_TEST = [Chem.MolFromSmiles(m) for m in Smile_mol_FULL_TEST]

finTanArrayTEST = [FingerprintMols.FingerprintMol(x) for x in Mol_From_Smile_FULL_TEST]

#Computing random Tanimoto Distances

TDTest1 = 1 - DataStructs.FingerprintSimilarity(finTanArrayTEST[0], finTanArrayTEST[1])

TDTest2 = 1 - DataStructs.FingerprintSimilarity(finTanArrayTEST[3], finTanArrayTEST[6])

TDTest3 = 1 - DataStructs.FingerprintSimilarity(finTanArrayTEST[5], finTanArrayTEST[7])

TDTest4 = 1 - DataStructs.FingerprintSimilarity(finTanArrayTEST[9], finTanArrayTEST[9])

#Computing random Jaccard Indexes

JITest1 = distance.jaccard(finTanArrayTEST[0], finTanArrayTEST[1])

JITest2 = distance.jaccard(finTanArrayTEST[3], finTanArrayTEST[6])

JITest3 = distance.jaccard(finTanArrayTEST[5], finTanArrayTEST[7])

JITest4 = distance.jaccard(finTanArrayTEST[9], finTanArrayTEST[9])


#Comparing 
Truth1 = TDTest1 == JITest1

Truth2 = TDTest2 == JITest2

Truth3 = TDTest3 == JITest3

Truth4 = TDTest4 == JITest4

print('Truth1:',Truth1)
print('Truth2:',Truth2)
print('Truth3:',Truth3)
print('Truth4:',Truth4)

#Testing SDF Files
MOL_Amm = open( '/Users/sameermac/Desktop/SDFMolFingerPrintTest/ammonia.sdf','rb')
MOL_Eth = open('/Users/sameermac/Desktop/SDFMolFingerPrintTest/ethane.sdf','rb')
MOL_EthAmm = open('/Users/sameermac/Desktop/SDFMolFingerPrintTest/ethilammonia.sdf', 'rb')
MOL_Meth = open('/Users/sameermac/Desktop/SDFMolFingerPrintTest/methane.sdf', 'rb')
MOL_MethAmm = open('/Users/sameermac/Desktop/SDFMolFingerPrintTest/methilammonia.sdf', 'rb')
MOL_Prop = open('/Users/sameermac/Desktop/SDFMolFingerPrintTest/propane.sdf','rb')
                
MOL1 = list(Chem.ForwardSDMolSupplier(MOL_Amm))[0]
#Mol2 = list(Chem.ForwardSDMolSupplier(MOL_Eth))[0]
#Mol3 = Chem.ForwardSDMolSupplier(MOL_EthAmm)
Mol4 = list(Chem.ForwardSDMolSupplier(MOL_Meth))[0]
#Mol5 = Chem.ForwardSDMolSupplier(MOL_MethAmm)
#Mol6 = Chem.ForwardSDMolSupplier(MOL_Prop)




#TestMOL = [MOL1,Mol2,Mol3,Mol4,Mol5,Mol6]

TestMOL = [MOL1,Mol4]

Smile_mol_FULL_TEST2 = [Chem.MolToSmarts(m) for m in TestMOL]


Mol_From_Smile_FULL_TEST2 = [Chem.MolFromSmarts(m) for m in Smile_mol_FULL_TEST2]
#Mol_From_Smile_FULL_TEST2 = [Chem.MolFromSmiles(m) for m in Smile_mol_FULL_TEST2]

finTanArrayTEST2 = [FingerprintMols.FingerprintMol(x) for x in TestMOL]


#To view (FingerPrint) molecular bit string files

#a = []
#ConvertToNumpyArray(MoleculeFingerPrintVariable, a)
#for _ in a:
   #print(_)
   
   


