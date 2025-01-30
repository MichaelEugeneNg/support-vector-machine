# Support Vector Machine
Implementations of an SVM model for MNIST and Spam email classification

This repository contains 3 files:
1. **linear_svm.py**:
    - Visualizes hard-margin SVMs on the toy-data.npz dataset
2. **featurize.py**:
    - Performs feature engineering on a corpus of emails to return a feature vector for each email
4. **cs189_hw1.ipynb**:
    - Performs data partitioning and evaluation metrics
    - Hyperparameter tuning 
    - K-fold cross validation to find the optimal C-value to use in the SVM
    - Contains code used to generate test predictions for MNIST and SPAM kaggle competition
