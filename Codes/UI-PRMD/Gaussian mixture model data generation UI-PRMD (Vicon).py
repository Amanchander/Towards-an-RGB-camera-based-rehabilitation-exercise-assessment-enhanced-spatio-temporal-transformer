# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:27:21 2023

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



path = '_path_to_Vicon_positional_data_'
current_exercise = 'm10'

allFiles = glob.glob(path + "/*.txt") 

print("The length of the files are ", len(allFiles))


headers = ['Likelihood', 'Score', '0_x', '0_y', '0_z', '0_vis', '1_x', '1_y', '1_z', '1_vis', '2_x', '2_y', '2_z', '2_vis', '3_x', '3_y', '3_z', '3_vis', '4_x', '4_y', '4_z', '4_vis', '5_x', '5_y', '5_z', '5_vis', '6_x', '6_y', '6_z', '6_vis', '7_x', '7_y', '7_z', '7_vis', '8_x', '8_y', '8_z', '8_vis', '9_x', '9_y', '9_z', '9_vis', '10_x', '10_y', '10_z', '10_vis', '11_x', '11_y', '11_z', '11_vis', '12_x', '12_y', '12_z','12_vis','13_x', '13_y', '13_z', '13_vis', '14_x', '14_y', '14_z', '14_vis', '15_x', '15_y', '15_z', '15_vis', '16_x', '16_y', '16_z', '16_vis', '17_x', '17_y', '17_z', '17_vis', '18_x', '18_y', '18_z', '18_vis', '19_x', '19_y', '19_z', '19_vis', '20_x', '20_y', '20_z', '20_vis', '21_x', '21_y', '21_z', '21_vis', '22_x', '22_y', '22_z', '22_vis', '23_x', '23_y', '23_z', '23_vis', '24_x', '24_y', '24_z', '24_vis', '25_x', '25_y', '25_z', '25_vis', '26_x', '26_y', '26_z', '26_vis', '27_x', '27_y', '27_z', '27_vis', '28_x', '28_y', '28_z', '28_vis', '29_x', '29_y', '29_z', '29_vis', '30_x', '30_y', '30_z', '30_vis', '31_x', '31_y', '31_z', '31_vis', '32_x', '32_y', '32_z', '32_vis']


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
        # myFile.set_axis(headers[0:66], axis = 1, inplace = True)
        data.append(myFile)
        # data = pd.concat([data, myFile], axis = 0, ignore_index=True)
        # data = data.append(myFile)
        # print(myFile)
        print("My file name is ", currFileName)
        
    else:
        pass

  except:
    print("File reding error")



check = False

for file in allFiles:
    # try:
    currFileName = os.path.basename(file).split('.')[0]
    currAct = currFileName.split('_')[0:1]
    # print("Current activity is  ", currentActivity)
    # print("The current filename is ", currFileName)
    
    if currAct[0] == current_exercise:
        
        print(file, " -- ", type(file))
        # This prints: videofiles\Burglary.mp4  --  <class 'str'>
        myFile = pd.read_csv(file);
        file = pd.DataFrame(myFile)
        
        path = '_path_to_trained_GMM_models_/' + current_exercise + '.pkl'
        # path = model_filename
        # Load the GMM model from the file
        gmm = joblib.load(path)
    
        recordingFrameRate = 30 # Frame rate at which the video was recorded 
        
        data = file
        
        data = data.dropna()
        data = data.reset_index(drop = True)
        # print("Total data is ", data)
        
        
        # myDesiredDf = data.loc[:,'0_x':'32_vis']
        myDesiredDf = data.dropna() 
        myDesiredDf = myDesiredDf.reset_index(drop = True)
        
        
        myDesiredDf= myDesiredDf.round(10)
        # print(myDesiredDf)
        
         
        sensor_data = np.array(myDesiredDf)
    
        
        # Calculate the log-likelihood of the data
        log_likelihood = -1 * gmm.score_samples(sensor_data)
        
        # Print the log-likelihood values for each data point
        # print("The log likelihood values are ", log_likelihood)
        
        
        performanceScore = []
        
        for i in log_likelihood:
            value = (1 - (i - np.min(log_likelihood))/(np.max(log_likelihood) - np.min(log_likelihood))) * 100
            performanceScore.append(value)
    
        data.insert(0, 'Likelihood', log_likelihood)
        # print("The scaled performance score values are ", performanceScore)
        data.insert(1, 'Score', performanceScore)
        
        data.set_axis(headers[0:119], axis = 1, inplace = True)
    
        
        if check == False:
            check = True
            # Directory
            directory = current_exercise
              
            # Parent Directory path
            parent_dir = "C:/Aman/Research papers/Quality assessment paper/GMM training data UI-PRMD/Vicon"
            
            # Path
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)
        
        
        print("The stored file is ", current_exercise + '/' + currFileName)
        # os.makedirs(os.path.dirname('_path_to_store_score_values_/'+ currentActivity + '/'))
         
      
        data.to_csv("_path_to_save_UI-PRMD_VICON_data_for_training_"+ current_exercise + '/' + currFileName + ".csv", index = False)
    
    
        # except:
        #     print("Error in the file ", currFileName)
        
    else:

        pass
