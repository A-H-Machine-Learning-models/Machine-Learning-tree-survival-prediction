import numpy as np
import pandas as pd
import tensorflow as tf

# Data Preprocessing
# Import Dataset
dataset = pd.read_csv('Data.csv')

# Remove rows with missing 'Event' values
dataset = dataset[~dataset['Event'].isna()]

# Exclude unwanted columns
unwanted_columns = ['No','Subplot', 'EMF', 'Harvest', 'Alive', 'Event', 'PlantDate']
features = [col for col in dataset.columns if col not in unwanted_columns]

# Identify non-numeric values in 'Adult' column
non_numeric_mask = pd.to_numeric(dataset['Adult'], errors='coerce').isna()

# Calculate mean of numeric values in 'Adult' column
numeric_adult = pd.to_numeric(dataset.loc[~non_numeric_mask, 'Adult'], errors='coerce')
mean_adult = np.mean(numeric_adult.dropna())

# Replace non-numeric values with mean
dataset.loc[non_numeric_mask, 'Adult'] = int(mean_adult)

# Set the negative values of phenolics to 0 because it can't be negative
dataset['Phenolics'] = dataset['Phenolics'].apply(lambda x: max(x, 0))

# Extract features (x) and target (y)
x = dataset[features]
y = dataset['Event']

# Data Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#use oneHotEncoder to encode the non-numeric values
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [1, 3, 5, 7, 8, 9, 10])
    ],
    remainder='passthrough'
)

x = np.array(ct.fit_transform(x))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Building the ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(6, activation='relu'))
ann.add(tf.keras.layers.Dense(6, activation='relu'))
# Adding output layer
ann.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Training the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(x_train, y_train, batch_size=32, epochs=100)

# Saving the trained model, write the path you want then uncomment the 3 lines
# model_path = ''
# ann.save(model_path)
# print(f"Model saved to {model_path}")

# Predicting the Test set results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy: ", accuracy_score(y_test, y_pred))
