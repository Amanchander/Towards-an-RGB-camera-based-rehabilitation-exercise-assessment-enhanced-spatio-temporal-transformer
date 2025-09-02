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
from sklearn.preprocessing import StandardScaler



curr_exercise = 'E5'

path = '_Path_for_exercise_data_csv_files_'
targetDirectory = path + curr_exercise

folder_path = '_Enter_path_where_trianed_models_have_been_saved_'  # Path where the trained model has been saved
model_path = folder_path + curr_exercise

folder_path_to_store_scoredFiles = '_Path_to_store_data_with_scores_'+'/'
store_Path_for_final_scores_data = '_Path_to_store_final_data_with_scores_'+'/' # Path to sto store the final files with scores
path_to_store_graph_images = '_Path_to_store_graph_images_'+ '/' 

headers = ['Score', 'Log-likelihood', '0_x', '0_y', '0_z', '0_vis', '1_x', '1_y', '1_z', '1_vis', '2_x', '2_y', '2_z', '2_vis', '3_x', '3_y', '3_z', '3_vis', '4_x', '4_y', '4_z', '4_vis', '5_x', '5_y', '5_z', '5_vis', '6_x', '6_y', '6_z', '6_vis', '7_x', '7_y', '7_z', '7_vis', '8_x', '8_y', '8_z', '8_vis', '9_x', '9_y', '9_z', '9_vis', '10_x', '10_y', '10_z', '10_vis', '11_x', '11_y', '11_z', '11_vis', '12_x', '12_y', '12_z','12_vis','13_x', '13_y', '13_z', '13_vis', '14_x', '14_y', '14_z', '14_vis', '15_x', '15_y', '15_z', '15_vis', '16_x', '16_y', '16_z', '16_vis', '17_x', '17_y', '17_z', '17_vis', '18_x', '18_y', '18_z', '18_vis', '19_x', '19_y', '19_z', '19_vis', '20_x', '20_y', '20_z', '20_vis', '21_x', '21_y', '21_z', '21_vis', '22_x', '22_y', '22_z', '22_vis', '23_x', '23_y', '23_z', '23_vis', '24_x', '24_y', '24_z', '24_vis', '25_x', '25_y', '25_z', '25_vis', '26_x', '26_y', '26_z', '26_vis', '27_x', '27_y', '27_z', '27_vis', '28_x', '28_y', '28_z', '28_vis', '29_x', '29_y', '29_z', '29_vis', '30_x', '30_y', '30_z', '30_vis', '31_x', '31_y', '31_z', '31_vis', '32_x', '32_y', '32_z', '32_vis']



allfiles = glob.glob(targetDirectory + "/*.csv")

print("The length of the files are ", len(allfiles))


check = False

max_likelihood = -10000
min_likelihood = 10000


maxSizeFound = 0

for f in allfiles:
    try:
        data = pd.read_csv(f)
        # data  = data.append(myFile)     
        # data.drop(['Action'], axis=1, inplace=True)
        # data.drop(['Time in ms from Jan1 1970'], axis=1, inplace=True)
        
        data = data.dropna()
        data = data.reset_index(drop = True)
        # print("Total data is ", data)
        
        lengthOfFile = len(data)
        
        if lengthOfFile > maxSizeFound:
            maxSizeFound = lengthOfFile
            
        else:
            pass
        
        print("The length of the file is ", lengthOfFile)

    except:
        print("File reding error")

print("The max file size found is ", maxSizeFound)




for file in allfiles:

    currFileName = os.path.basename(file).split('.')[0]

    print("The current filename is ", currFileName)
    
    print(file, " -- ", type(file))
    # This prints: videofiles\Burglary.mp4  --  <class 'str'>
    myFile = pd.read_csv(file);
    file = pd.DataFrame(myFile)
    
    model_path = folder_path + 'GMM_model_' + curr_exercise
    # Load the GMM model from the file
    gmm = joblib.load(model_path)

   
    data = file
    
    data.drop(['Time in ms from Jan1 1970'], axis=1, inplace=True)
    data = data.dropna()
    

    
    data = data.reset_index(drop = True)
    
        
    
    df1 = data.loc[:,'0_x':'32_vis']
    column_headers = list(df1.columns.values)

 
    data = data.dropna()
    data = data.reset_index(drop = True)
     
    myDesiredDf = data
     
    sensor_data = np.array(myDesiredDf)

    
    # Calculate the log-likelihood of the data
    log_likelihood = gmm.score_samples(sensor_data)
    
    # Print the log-likelihood values for each data point
    # print("The log likelihood values are ", log_likelihood)
    
    
    performanceScore = []
 
    
    for i in log_likelihood:
        value = (i - np.min(log_likelihood))/(np.max(log_likelihood) - np.min(log_likelihood))
        performanceScore.append(value)

    myDesiredDf.insert(0, 'Likelihood', log_likelihood)
    # print("The scaled performance score values are ", performanceScore)
    # myDesiredDf.insert(1, 'Score', performanceScore)
     
    try:
        # Directory
        directory = curr_exercise
          
        # Parent Directory path
        parent_dir = folder_path_to_store_scoredFiles
          
        # Path
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
    
    except:
        pass
    
    path_to_store = folder_path_to_store_scoredFiles + curr_exercise + '/' + currFileName
    myDesiredDf.to_csv(path_to_store + ".csv", index = False)
    
    

 
path = folder_path_to_store_scoredFiles + curr_exercise
store_Path = store_Path_for_final_scores_data
 
allFiles = glob.glob(path + "/*.csv") 

print("Total number of files are ", len(allFiles))

data = []


for f in allFiles:
    try:
        myFile = pd.read_csv(f);
    
        currFileName = os.path.basename(f).split('.')[0]
        crr_incrr = currFileName.split('_')[1]   
        # currentActivity = "_".join(currAct)
        # print("Current activity is  ", currentActivity)
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
    score = round(score, 5)
    performanceScore.append(score)
    
    for i in range(0, len(myFile)):
        score_array.append(score)
    
    
    myFile.insert(0, 'Score', score_array)
    
    rows_to_append = maxSizeFound - len(myFile)
    
    # Create zero rows
    zero_rows = np.zeros((rows_to_append, myFile.shape[1]))
    
    # Append zero rows to x
    myFile = np.vstack([myFile, zero_rows])
    
    myFile = pd.DataFrame(myFile, columns = headers)
    
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
plt.title("Exercise data")
plt.legend()
plt.savefig(path_to_store_graph_images + 'Log-likelihood plot.png', dpi = 1200, bbox_inches="tight")
plt.show()

legend_labels = ["Correct", "Incorrect"]
plt.scatter(x_plt, crr_scores, label = legend_labels[0], marker = 'x')
plt.scatter(x_plt, incrr_scores, label = legend_labels[1], marker = 's')
plt.xlabel("Sequence Number")
plt.ylabel("Quality Score")
plt.title("Exercise data")
plt.legend()
plt.savefig(path_to_store_graph_images + 'Quality scores.png', dpi = 1200, bbox_inches="tight")
plt.show()

   
   
   
   