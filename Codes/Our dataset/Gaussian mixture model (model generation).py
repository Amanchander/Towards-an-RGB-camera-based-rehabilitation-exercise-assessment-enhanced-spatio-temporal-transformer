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


curr_exercise = 'E8'
path = "_Enter_path_for_exercise_data_" +'/' + curr_exercise

folder_path = '_Enter_path_to_save_trained_GMM_model_'  # Path where the trained model will be saved
model_filename = folder_path + '/GMM_model_' + curr_exercise

allFiles = glob.glob(path + "/*.csv") 

print("Total number of files are ", len(allFiles))

data = pd.DataFrame()


for f in allFiles:
  try:
    myFile = pd.read_csv(f);

    currFileName = os.path.basename(f).split('.')[0]
    crr_incrr = currFileName.split('_')[1]   
    # currentActivity = "_".join(currAct)
    # print("Current activity is  ", currentActivity)
    
    if crr_incrr == 'crr':
        print("The current file added is ", currFileName)
        
        data  = pd.concat([data, myFile])
    
    else:
        pass

  except:
    print("File reding error")


data.drop(['Time in ms from Jan1 1970'], axis=1, inplace=True)
data = data.dropna()
data = data.reset_index(drop = True)
print("Total data is ", data)


myDesiredDf = data
 
sensor_data = np.array(myDesiredDf)

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
