---
title: "FuzzyRoughShuffledFrog"
author: "Javad Rahimipour Anaraki"
date: '08/08/18'
---

## Use case
To determine the most important features using binary Fuzzy-Rough Shuffled Frog Leaping Algorithm as described in [A Fuzzy-Rough Feature Selection based on Binary Shuffled Frog Leaping Algorithm](https://waset.org/conference/2018/09/vancouver/ICFIE) By Javad Rahimipour Anaraki, Saeed Samet, Mahdi Eftekhari and Chang Wook Ahn

A long version of the paper can be accessed in [A Fuzzy-Rough based Binary Shuffled Frog Leaping Algorithm for Feature Selection](https://arxiv.org/abs/1808.00068)

## Compile
To compile this program follow these steps:

1. Be sure that you have the latest GCC/G++ compiler installed
2. Use `g++ -o SFLA main.cpp clsWeka.cpp clsData.cpp -std=c++11 -Iinclude -pthread` to compile the program
3. To improve its performance one can use `-O1` or `-O2` or `-O3`

## Run
To run the program use `./SFLA /Data/{a dataset name}OK.csv {maximum number of stalled} {number of threads} {maximum iteration}`
For instance, run the code for Breast Tissue dataset with maximum three stall situations, four threads for 10 times, it should be run using: `./SFLA /Data/BreastTissueOK.csv 3 4 10` . The classification results will be stored in *classification* folder and the convergence process is stored in *results* folder.

All datasets are stored in *data* folder and originally adopted from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

## Note
Datasets end with *OK.csv* in their names should have no column and/or row names, and the class values should be numeric and sorted in ascending order. The ones ending with *W.csv* should have columns name and the class values should be alphabetical.
