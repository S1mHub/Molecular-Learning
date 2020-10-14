#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 22:03:23 2018

@author: sameermac
"""

#Computing Tanimoto Distance and uniquenesses of 50 molecules from QM9 Database

#from __future__ import print_function
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
import random
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdMolAlign, rdShapeHelpers
from rdkit.Chem import Draw
#from rdkit.Chem import Descriptors - May need later

#Reading molecules from SDF file

MOL_open = open( '/Users/sameermac/Desktop/Thesis/gdb9.sdf','rb')
MOL_LIST = Chem.ForwardSDMolSupplier(MOL_open)
mol = [x for x in MOL_LIST if x is not None]
#Some elements empty because there are invalid molecules =/> SMILES Format



#Picking 50 random molecules
mol_50 = random.sample(mol, 50)

#Converting to SMILES format

Smile_mol = [Chem.MolToSmiles(m) for m in mol_50]

Mol_From_Smile = [Chem.MolFromSmiles(m) for m in Smile_mol]


#Computing number of Unique Chemicals

UniquenessIndex = len(set(Smile_mol)) / len(Smile_mol)

#Computing Tanimoto Distance (using RDKIT Fingerprint)
finTan = [FingerprintMols.FingerprintMol(x) for x in Mol_From_Smile]
TanimotoSimilarity = DataStructs.FingerprintSimilarity(finTan[1], finTan[2])
TanimotoDistance = 1 - TanimotoSimilarity

#Note Default measure is Tanimoto in FingerprintSimilarity

 #Draw.MolToImage(mol_50[0]) - see first molecule in viewer




















































#Error Bad Conformer ID



#Erased Code


#TanimotoDistance = rdShapeHelpers.ShapeTanimotoDist(Chem.MolFromSmiles(Smile_mol[1]), Chem.MolFromSmiles(Smile_mol[2]))
#TanimotoDistance = rdShapeHelpers.ShapeTanimotoDist(Smile_mol[1], Smile_mol[2])

#SmileMOLs = Chem.MolToSmiles(mol)

#def Smile_Conversion(MOL_LIST):

#for i in mol:  
    #smileMOLs = Chem.MolToSmiles(mol)
#return MOL_LIST


#DataStructs.DiceSimilarity(pairFps[0],pairFps[1])
#fig, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(refmol, mol, lambda m,idx: SimilarityMaps.GetMorganFingerprint(m, atomId=idx, radius=1, fpType='count'), metric=DataStructs.TanimotoSimilarity)


#metric=DataStructs.TanimotoSimilarity

