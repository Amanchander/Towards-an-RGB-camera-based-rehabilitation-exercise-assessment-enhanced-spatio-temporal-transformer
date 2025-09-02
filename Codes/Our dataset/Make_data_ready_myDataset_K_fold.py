# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:20:17 2024

@author: MDL Workstation
"""

import logging
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# Transformer model reference was taken from https://www.tensorflow.org/text/tutorials/transformer 

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

# pip install scikit-learn

import joblib
import pandas as pd
import glob
import numpy as np
import os, random
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
print(tf.__version__)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold  

current_exercise = 'E1' # Exercise number


# Parent Directory path
input_data_path = '_Path_to_store_final_data_with_scores_' + current_exercise
parent_dir = '_Path_to_store_pkl_files'


allFiles = glob.glob(input_data_path + "/*.csv") 


print("The length of the files are ", len(allFiles))


maxSizeFound = 0

for f in allFiles:
    try:
        data = pd.read_csv(f)
        
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

maxFrameSize = maxSizeFound


crr_data = []
crr_score =[]

incrr_data = []
incrr_score = []


for f in allFiles:
    try:
        currFileName = os.path.basename(f).split('.')[0]

        crr_incrr = currFileName.split('_')[1]   

        

        myFile = pd.read_csv(f)

        
        myFile = myFile.dropna()
        myFile = myFile.reset_index(drop = True)
        curr_score = myFile.loc[0]['Score']
        myFile.drop(['Log-likelihood'], axis=1, inplace=True)
        myFile.drop(['Score'], axis=1, inplace=True)
        myFile = np.array(myFile)
        
        rows_to_append = maxSizeFound - len(myFile)
        
        # Create zero rows
        zero_rows = np.zeros((rows_to_append, myFile.shape[1]))
        
        # Append zero rows to x
        myFile = np.vstack([myFile, zero_rows])
        
        if crr_incrr == 'crr':
            print("The correct file added is ", currFileName)
            crr_data.append(myFile)
            crr_score.append(curr_score)

        elif crr_incrr == 'incrr':
            print("The incorrect file added is ", currFileName)
            incrr_data.append(myFile)
            incrr_score.append(curr_score)
  
        
    except:
        print("File reading error")

print("The max file size found is ", maxSizeFound)




try: # Try to make the directory if its not there already otherwise it will give an error and it will simply pass 
    
    directory = current_exercise
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)

except: 
    pass



k = 4
k_fold = KFold(n_splits = k, random_state = None)  


split_no = 1

for training_index, testing_index in k_fold.split(crr_data):  
    
    print("Training index is ", training_index, "  testing index is ", testing_index)
    
    crr_data    = np.array(crr_data)
    crr_score   = np.array(crr_score)
    
    incrr_data  = np.array(incrr_data)
    incrr_score = np.array(incrr_score)
    
    
    
    X_train_crr = crr_data[training_index,:]
    X_test_crr  = crr_data[testing_index,:]  
    
    Y_train_crr = crr_score[training_index]
    Y_test_crr  = crr_score[testing_index] 
    
    print("The size of X_train_crr is ",    X_train_crr.shape)
    print("The size of X_test_crr is ",    X_test_crr.shape)
    
    
    
    X_train_incrr = incrr_data[training_index,:]
    X_test_incrr  = incrr_data[testing_index,:]  
    
    Y_train_incrr = incrr_score[training_index]
    Y_test_incrr  = incrr_score[testing_index] 
    
    print("The size of X_train_incrr is ",    X_train_incrr.shape)
    print("The size of X_test_incrr is ",    X_test_incrr.shape)
    
    

    # Training set: data and labels
    train_x = np.concatenate((X_train_crr, X_train_incrr))
    print(train_x.shape, 'training data')
    train_y = np.concatenate((np.squeeze(Y_train_crr), (np.squeeze(Y_train_incrr))))
    print(train_y.shape, 'training labels')

    # Validation set: data and labels
    valid_x = np.concatenate((X_test_crr, X_test_incrr))
    print(valid_x.shape, 'validation data')
    valid_y = np.concatenate((np.squeeze(Y_test_crr),np.squeeze(Y_test_incrr)))
    print(valid_y.shape, 'validation labels')
    
    
    split1path = path + '/' + current_exercise  + '_split_' + str(split_no) + '.pkl'
    print(split1path)
    split1_data = {'train': {'labels': train_y, 'data': train_x}, 'test': {'labels': valid_y, 'data': valid_x}}
    
    joblib.dump(split1_data, split1path)
    
    split_no+=1
    
