# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:23:11 2023

@author: MDL Workstation
"""

# pip install scikit-learn

import joblib
import pandas as pd
import glob
import numpy as np
import os
import scipy.stats as stats
import statistics
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import norm
from scipy import interpolate
from sklearn import preprocessing
from scipy.stats import norm
from scipy.stats import gaussian_kde

from sklearn.model_selection import train_test_split
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from scipy.stats import multivariate_normal

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


framerate = 10

path = 'C:/Aman/Research papers/Quality assessment paper/Dataset/UI-PRMD/Data/Segmented Movements/Vicon/Positions'

folder_path = 'C:/Aman/Research papers/Quality assessment paper/Trained GMM models (UI-PRMD)/Vicon_UI_PRMD'  # Path where the trained model will be saved
model_filename = folder_path + '/m10.pkl' 

allFiles = glob.glob(path + "/*.txt") 

print("The length of the files are ", len(allFiles))


current_exercise = 'm10'



# data = pd.DataFrame()
data = []
count = 0

for f in allFiles:
  try:
    currFileName = os.path.basename(f).split('.')[0]
    # print("The file name is ", currFileName)
    currAct = currFileName.split('_')[0:1]
    # currentActivity = "_".join(currAct)
    # print(currAct)
    
    if currAct[0] == current_exercise:
        myFile = pd.read_csv(f)
        myFile = pd.DataFrame(myFile)
        data.append(myFile)
        # data = pd.concat([data, myFile], axis = 0, ignore_index=True)
        # data = data.append(myFile)
        print(myFile)
        
    else:
        pass

  except:
    print("File reding error")





## Now analyse for the lengths to make the data homogeneous
maxLength = 0
for val in data:
    currLen = len(val)
    print("Length is ", currLen)
    
    if currLen >= maxLength:
        maxLength = currLen
        
    else:
        pass
    
print("The maximum length found is ", maxLength)


## Zero padding
paddedData = []

for val in data:
    currLen, currWidth = np.array(val).shape
    diff = maxLength - currLen
    
    print("The file's length is ", currLen, "and the current difference is ", diff)
    
    for i in range(0, diff):
        val.loc[len(val)] = 0
    
    paddedData.append(val)


data = paddedData
data = np.array(data)
data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])


data = pd.DataFrame(data)

data = data.dropna()
data = data.reset_index(drop = True)


data= data.round(10)
print(data)

 
sensor_data = np.array(data)

# Try different numbers of components and calculate AIC for each
n_components_range = range(1, 11)
aic_values = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(sensor_data)
    aic = gmm.aic(sensor_data)
    aic_values.append(aic)

# Plot the AIC values for different numbers of components
plt.figure(figsize=(8, 6))
plt.plot(n_components_range, aic_values, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Components')
plt.ylabel('AIC Value')
plt.title('AIC vs. Number of Components')
plt.grid(True)

# Find the optimal number of components with the lowest AIC
optimal_n_components = n_components_range[np.argmin(aic_values)]
print(f"Optimal number of components (AIC): {optimal_n_components}")

# Fit the GMM with the optimal number of components
best_gmm = GaussianMixture(n_components=optimal_n_components, random_state=0)
best_gmm.fit(sensor_data)

# Visualize the clustering result
plt.figure(figsize=(8, 6))
labels = best_gmm.predict(sensor_data)
plt.scatter(sensor_data[:, 0], sensor_data[:, 1], c=labels, cmap='viridis', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('GMM Clustering Result')
plt.grid(True)
plt.show()


num_components = optimal_n_components  # You can adjust this based on your data
gmm = GaussianMixture(n_components=num_components, random_state=0)
# gmm.fit(sensor_data)



log_likelihoods = []
max_iterations = 5

for i in range(max_iterations):
    gmm.fit(sensor_data)
    likelihood_value = gmm.score(sensor_data)
    log_likelihoods.append(likelihood_value)

plt.plot(range(max_iterations), log_likelihoods)
plt.xlabel('Iterations')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood vs. Iterations')
plt.show()


# Save the trained GMM model to a file
joblib.dump(gmm, model_filename)

print(f"Trained GMM model saved to {model_filename}")


# Calculate the log-likelihood of the data
log_likelihood = -1 * gmm.score_samples(sensor_data)

# Print the log-likelihood values for each data point
# print("The log likelihood values are ", log_likelihood)


performanceScore = []

for i in log_likelihood:
    value = (1 - (i - np.min(log_likelihood))/(np.max(log_likelihood) - np.min(log_likelihood))) * 100
    performanceScore.append(value)
    
print("The scaled performance score values are ", performanceScore)
data.insert(0, 'Score', performanceScore)


# Get cluster assignments for each data point
cluster_assignments = gmm.predict(sensor_data)

# Alternatively, you can estimate the probability of each point belonging to each cluster
cluster_probabilities = gmm.predict_proba(sensor_data)


plt.plot(cluster_probabilities)
print(cluster_probabilities)
plt.scatter(sensor_data[:, 0], sensor_data[:, 1], c=cluster_assignments, cmap='viridis')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Gaussian Mixture Model Clusters')
plt.show()
