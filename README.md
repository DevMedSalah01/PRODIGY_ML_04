# PRODIGY_ML_04
Implement a Support Vector Machine (SVM) to Classify Images of Cats and Dogs from the Kaggle Dataset



# Implement a Support Vector Machine (SVM) to Classify Images of Cats and Dogs from the Kaggle Dataset

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Saving the Model](#saving-the-model)
- [Conclusion](#conclusion)
- [File Structure](#file-structure)
- [References](#references)

## Introduction
This project aims to classify images of cats and dogs using a Support Vector Machine (SVM) model. The dataset used for this task is sourced from Kaggle. The steps include data loading, preprocessing, model training, evaluation, and saving the trained model.

## Installation
To run this project, you need the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `opencv-python`
- `matplotlib`
- `keras`
- `tensorflow`

You can install these dependencies using pip:

bash

#pip install numpy pandas scikit-learn opencv-python matplotlib keras tensorflow

**Dataset
The dataset used is the "Dogs vs. Cats" dataset from Kaggle. Ensure you have the dataset downloaded and extracted in the following path: C:/Users/ASUS/Documents/Prodigy_Tasks/Dataset/train.

**Preprocessing
The preprocessing steps include reading the images, resizing them to a uniform size, normalizing the pixel values, and labeling them as either cat (0) or dog (1).

**Model Architecture
We use a Pipeline that includes PCA for dimensionality reduction and an SVM for classification:

**Training
Split the dataset into training and testing sets and fit the pipeline:

**Evaluation
Evaluate the model using accuracy score, confusion matrix, and classification report:

**Saving the Model
Save the trained model using joblib:
------------------------------------------------------------------------------------------
Conclusion
The SVM model achieves an accuracy of approximately 67.46%. The model can be improved further with hyperparameter tuning and more advanced techniques.

***********************************************************
References
Kaggle Dataset: Dogs vs. Cats
Scikit-learn Documentation: https://scikit-learn.org/stable/
Keras Documentation: https://keras.io/

