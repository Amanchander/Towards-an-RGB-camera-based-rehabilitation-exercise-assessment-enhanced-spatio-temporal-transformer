# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:45:07 2024

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


curr_exercise = 'E5'
path = '_Path_to_get_exercise_data_with_scores_' + curr_exercise


store_Path = '_Path_to_store_final_data_with_scaled_final_scores_'

allFiles = glob.glob(path + "/*.csv") 

print("Total number of files are ", len(allFiles))

data = []


for f in allFiles:
  try:
    myFile = pd.read_csv(f);

    currFileName = os.path.basename(f).split('.')[0]
    crr_incrr = currFileName.split('_')[1]   

    print("The current file added is ", currFileName)
        
    data.append(myFile)
    
 
  except:
    print("File reding error")



likelihood_mean = []


for i in data:
    mean_val = i.loc[:,'Likelihood'].mean()
    likelihood_mean.append(mean_val)


performanceScore = []

for f in allFiles:
    myFile = pd.read_csv(f)
    currFileName = os.path.basename(f).split('.')[0]
    
    mean_val = myFile.loc[:,'Likelihood'].mean()
    print("The mean value is ", mean_val)
    
    score_array = []
    score = (mean_val - np.min(likelihood_mean))/(np.max(likelihood_mean) - np.min(likelihood_mean))
    performanceScore.append(score)
    
    for i in range(0, len(myFile)):
        score_array.append(score)
    
    
    myFile.insert(0, 'Score', score_array)
    path_to_store = store_Path + '/' + curr_exercise + '/' + currFileName
    myFile.to_csv(path_to_store + ".csv", index = False)



crr_scores   = performanceScore[0 : int(len(performanceScore)/2)]
incrr_scores = performanceScore[int(len(performanceScore)/2) : len(performanceScore)]

crr_loglikelihood = likelihood_mean[0:int(len(likelihood_mean)/2)]
incrr_loglikelihood = likelihood_mean[int(len(likelihood_mean)/2):len(likelihood_mean)]

val = []
[val.append(i) for i in range(0, len(crr_scores))]
x_plt = val

print("The scores are ", performanceScore)

legend_labels = ["Correct", "Incorrect"]
plt.scatter(x_plt, crr_loglikelihood, label = legend_labels[0], marker = 'x')
plt.scatter(x_plt, incrr_loglikelihood, label = legend_labels[1], marker = 's')
plt.xlabel("Sequence Number")
plt.ylabel("Log-likelihood")
plt.title("Exercise E5")
plt.legend()
plt.savefig('_Path_to_store_image_', dpi = 1200, bbox_inches="tight")
plt.show()

legend_labels = ["Correct", "Incorrect"]
plt.scatter(x_plt, crr_scores, label = legend_labels[0], marker = 'x')
plt.scatter(x_plt, incrr_scores, label = legend_labels[1], marker = 's')
plt.xlabel("Sequence Number")
plt.ylabel("Quality Score")
plt.title("Exercise E5")
plt.legend()
plt.savefig('_Path_to_store_image_', dpi = 1200, bbox_inches="tight")
plt.show()











