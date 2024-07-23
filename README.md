Name: Ahmad Hirbawy
Email: ahmadhir33@gmail.com

Overview
This project is aimed at building and training an Artificial Neural Network (ANN) to predict binary outcomes based on a dataset.
The dataset includes various features related to tree seedling functional traits and plant-soil feedback. This readme provides a detailed
description of the functionality of the code, how it works, the features used, and those excluded.

Functionality
The code performs the following steps:

Data Preprocessing:

Importing the dataset.
Handling missing values.
Excluding unwanted columns.
Handling non-numeric values in the 'Adult' column.
Setting negative values in the 'Phenolics' column to zero.
Data Encoding:

Encoding categorical features using OneHotEncoder.
Data Splitting:

Splitting the dataset into training and test sets.
Feature Scaling:

Standardizing the features.
Building the ANN:

Creating a Sequential model.
Adding input, hidden, and output layers.
Training the ANN:

Compiling and training the model.
Saving the Trained Model:

Optionally saving the trained model.
Predicting Test Set Results:

Making predictions on the test set.
Evaluation:

Generating a confusion matrix and calculating the accuracy of the model.
How It Works
Data Preprocessing:

The dataset is imported using pandas.
Rows with missing values in the 'Event' column are removed.
Unwanted columns (No, Subplot, EMF, Harvest, Alive, Event, PlantDate) are excluded from the features.
Non-numeric values in the 'Adult' column are identified and replaced with the mean of the numeric values.
Negative values in the 'Phenolics' column are set to zero.
Features (x) and target (y) variables are extracted.
Data Encoding:

The categorical features are encoded using OneHotEncoder via ColumnTransformer.
Data Splitting:

The dataset is split into training (80%) and test (20%) sets.
Feature Scaling:

Features are standardized using StandardScaler.
Building the ANN:

A Sequential ANN model is created.
Two hidden layers with 6 neurons each and ReLU activation functions are added.
An output layer with 1 neuron and sigmoid activation function is added.
Training the ANN:

The model is compiled using the Adam optimizer and binary cross-entropy loss function.
The model is trained for 100 epochs with a batch size of 32.
Saving the Trained Model:

The trained model can be saved by specifying a path and uncommenting the relevant lines.
Predicting Test Set Results:

Predictions are made on the test set.
Predictions are thresholded at 0.5 to obtain binary outcomes.
Evaluation:

A confusion matrix is generated.
The accuracy of the model is calculated.
Features Used and Excluded
Used Features:
Adult
Phenolics
Other numeric and categorical features not listed in the unwanted columns
Excluded Features:
No
Subplot
EMF
Harvest
Alive
Event
PlantDate
These exclusions were based on relevance and the need to avoid potential data leakage.

Prerequisites
Python 3.x
pandas
numpy
scikit-learn
tensorflow
Usage
To run the code, ensure you have the necessary libraries installed and the dataset (Data.csv) available in the working directory.
 Execute the script to preprocess the data, build and train the ANN, and evaluate its performance.
