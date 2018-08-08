---
title: "FuzzyRoughShuffledFrog"
author: "Javad Rahimipour Anaraki"
date: '08/08/18'
---

## Use case
To determine the most important features using binary Fuzzy-Rough Shuffled Frog Leaping Algorithm as described in [A Fuzzy-Rough Feature Selection based on Binary Shuffled Frog Leaping Algorithm](https://waset.org/conference/2018/09/vancouver/ICFIE) By Javad Rahimipour Anaraki, Saeed Samet, Mahdi Eftekhari and Chang Wook Ahn

Long version of the paper can be accessed in [A Fuzzy-Rough based Binary Shuffled Frog Leaping Algorithm for Feature Selection](https://arxiv.org/abs/1808.00068)

## Compile
This code can be run using MATLAB R2006a and above

## Run
To run the code, open `main.m` and choose a dataset to apply the method to. At first, the overall dependency degree is calculated using `dependency.m`. Then, indisernible objects are determined and returned to `main.m`. Finally, feature are selected if they have the best dependecy degree among all. This process stops if `(overall dependency degree - current dependency degree) x number of samples < 1` or maximum dependency degree does not improve with adding extra features.

All datasets are stored in *Data* folder and originally adopted from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

## Note
Datasets should have no column and/or row names, and the class values should be all numeric
