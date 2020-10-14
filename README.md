# Molecular-Learning
A repository for unsupervised learning of molecular classes

This repository contains numerous experimental results, building blocks and applications of unsupervised learning algorithms (such as K-medoids clustering modified to develop labelled clusters based off of Tanimoto Distance), and advanced visualization and dimension reduction algorithms (such as MDS, t-SNE, LLE, etc.) using sci-kit learn with molecular processing and filtering using RDKit. All code is written in Python. Such code and results are based only on 2D representations of molecules with a SMILES or SMARTS representation. Ligand-binding behavior is not explicitly explored in this code or demonstrated in any results. These methods and analysis are designed specifically to be used to prepare molecular classes for more advanced generative models for small drug-like molecular generation.

In addition to these components, there is some select research papers retained on this repository relating to machine learning and generative models for molecular generation, general machine learning methods for generative models and semi-supervised learning, as well as research articles and resources relating to molecular processing in a computer-aided drug design (CADD) context. 

To see only the code, the folder TD_Cluster_Code contains all the relevant code. If you use any of the code from TD_Cluster_Code, please consider citing this GitHub repository:

@Unpublished{RawatGithub, <br />
<space><space> author = {Sameer Rawat}, <br />
<space><space> title = {Molecular-Learning}, <br />
<space><space> Month = {October}, <br />
<space><space> year = {2020}, <br />
<space><space> note =   {Github Repository}, <br />
<space><space> url = {https://github.com/S1mHub/Molecular-Learning/} <br />
}
