# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 19:08:23 2024

@author: MDL Workstation
"""

# conda install cudatoolkit
# conda install numba


# Transformer model reference was taken from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
# import numba
# from numba import jit, cuda

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

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
from math import sqrt
print(tf.__version__)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Input, LSTM, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
import torch
import torch.nn.utils as torch_utils
import random 
import copy

random.seed(42) 
np.random.seed(42) # For reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




current_exercise = 'E1' # Exercise number 
split = 'split_1' # Split number for cross validation 


parentDir = "_Path_to_the_saved_PKL_files_" + "/pkl files/" # For UI-PRMD Vicon data




split_data_path = parentDir + current_exercise + '/' + current_exercise + '_' + split + '.pkl' 
split_data = joblib.load(split_data_path)


X_train = split_data['train']['data']
y_train = split_data['train']['labels']


X_test = split_data['test']['data']
y_test = split_data['test']['labels']





import numpy as np

# Replace NaN values with mean in training data
nan_mask_train_X = np.isnan(X_train)
mean_values_train_X = np.nanmean(X_train)
X_train[nan_mask_train_X] = mean_values_train_X

nan_mask_train_y = np.isnan(y_train)
mean_values_train_y = np.nanmean(y_train)
y_train[nan_mask_train_y] = mean_values_train_y

# Replace NaN values with mean in testing data
nan_mask_test_X = np.isnan(X_test)
mean_values_test_X = np.nanmean(X_test)
X_test[nan_mask_test_X] = mean_values_test_X

nan_mask_test_y = np.isnan(y_test)
mean_values_test_y = np.nanmean(y_test)
y_test[nan_mask_test_y] = mean_values_test_y



# Convert data to PyTorch tensors if not done already
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)

X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)



print("The X_train size is ", X_train.shape)
print("The y_train size is ", y_train.shape)
print("The X_test size is ", X_test.shape)
print("The y_test size is ", y_test.shape) 





##################################################################################################################################################################################
##################################################################################################################################################################################

# import joblib
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# import matplotlib.pyplot as plt

# # Train Random Forest model and compute feature importance
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# feature_importances = []

# for i in range(X_train.shape[2]):  # Loop through each feature separately
#     print("i is ", i)
#     X_train_feature = X_train[:, :, i]
#     y_train_feature = y_train  # Ensure the dimensions match

#     rf.fit(X_train_feature, y_train_feature)
#     feature_importances.append(rf.feature_importances_[0])


# # Plot feature importance for each feature
# plt.figure(figsize=(10, 6))
# plt.bar(range(X_train.shape[2]), feature_importances)
# plt.xlabel('Feature Index', fontsize=14)  # Increase font size of x-label
# plt.ylabel('Feature Importance', fontsize=14)  # Increase font size of y-label
# plt.title('Feature Importance for each feature', fontsize=16)  # Increase font size of title

# plt.xticks(fontsize=14)  # Increase font size of x-ticks
# plt.yticks(fontsize=14)  # Increase font size of x-ticks


# plt.savefig("C:/Aman/Research papers/Quality assessment paper/Paper/All paper files/Image files/Feature_importance_myDataset.png", dpi = 600)
# plt.show()


##################################################################################################################################################################################
##################################################################################################################################################################################
import torch.nn.functional as F
from torch import nn
# from transformer_encoder.utils import PositionalEncoding
from einops import rearrange
from einops.layers.torch import Rearrange

# classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


        
    
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head *  heads
        self.alpha = nn.Parameter(data = torch.tensor(dim_head, dtype=torch.float32), requires_grad=True)  # Initialize alpha with dim_head
        # self.alpha = dim_head
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=True)
        
        
    def forward(self, x):
        # print("Input shape to the attention layer is ", x.shape)
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        scale = self.alpha ** -0.5
        dots = torch.matmul(q, k.transpose(-1, -2)) * scale 
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([]) 
        self.relu  = nn.ReLU()
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            
        return self.norm(x)




def makeFrames(x, frame_size, stride):  
    x = copy.deepcopy(x) # In case you want to trim the features, to select all features jsut select 'nose'
    vertical_divisibility = (len(x) - frame_size) % stride
    no_of_frames = ((len(x) - frame_size) // stride) + 1
    # print("The size of x before appending is ", x.shape)
    
    if vertical_divisibility != 0:
        # Calculate the number of rows to append
        rows_to_append = stride - vertical_divisibility 
        
        # Create zero rows
        zero_rows = np.zeros((rows_to_append, x.shape[1]))
        
        # Append zero rows to x
        x = np.vstack([x, zero_rows])
    
    assert (len(x) - frame_size) % stride == 0, 'Cannot cover all the values of the input matrix vertically, please change the patch height value or vertical stride value!'
    
    no_of_frames = ((len(x) - frame_size) // stride) + 1
    frames = []
    frames_transposed = []
    
    for i in range(0, no_of_frames):
        val = x[(i * stride) : (i * stride) + frame_size, :]
        frames.append(copy.deepcopy(val))
        val = np.transpose(val)
        frames_transposed.append(copy.deepcopy(val))
   
    return frames, frames_transposed





import math
from torch import nn, Tensor
import time
import joblib

class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
      super().__init__()
      self.dropout = nn.Dropout(p=dropout)

      position = torch.arange(max_len).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
      pe = torch.zeros(max_len, 1, d_model)
      pe[:, 0, 0::2] = torch.sin(position * div_term)
      pe[:, 0, 1::2] = torch.cos(position * div_term)
      self.register_buffer('pe', pe)

  def forward(self, x: Tensor) -> Tensor:
      """
      Arguments:
          x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
      """
      x = x + self.pe[:x.size(0)]    
      return self.dropout(x)  
 



class mySpatioTemporalTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, dim_head, frame_size, no_of_frames, num_heads, num_layers, dim_feedforward, stride, dropout, dense_layer_neurons):
        super().__init__()

        self.spatial_transformer_transposed = Transformer(model_dim, num_layers, num_heads, dim_head, dim_feedforward)
        self.spatial_transformer            = Transformer(model_dim, num_layers, num_heads, dim_head, dim_feedforward)
        self.temporal_transformer           = Transformer(model_dim, num_layers, num_heads, dim_head, dim_feedforward)
        
        self.spatial_embedding_transposed   = nn.Linear(frame_size, model_dim)
        self.spatial_embedding              = nn.Linear(input_dim[-1], model_dim)
        self.temporal_embedding             = nn.Linear(model_dim , model_dim)
        
        self.positional_encoding_transposed = PositionalEncoding(model_dim)
        self.positional_encoding            = PositionalEncoding(model_dim)
        self.temporal_positional_encoding   = PositionalEncoding(model_dim)
        self.weights1                       = nn.Linear(model_dim, model_dim)
        self.weights2                       = nn.Linear(model_dim, model_dim)
        
        
        self.flatten                        = nn.Flatten()
        self.frame_size                     = frame_size
        self.stride                         = stride
        
        self.dense_1                        = nn.Linear((((frame_size + input_dim[-1]) * no_of_frames) * model_dim) , dense_layer_neurons)  # Adjust units as needed
        self.relu                           = nn.ReLU()
        self.dropout                        = nn.Dropout(dropout)
        self.dense_2                        = nn.Linear(dense_layer_neurons, 1)
        self.input_dim                      = input_dim
        
         
    def forward(self, x):

        total_frames, total_frames_transposed = makeFrames(x.cpu().numpy(), self.frame_size, self.stride)  # Move to CPU and convert to NumPy
        transformer_output = torch.empty(0, 1, model_dim, device=x.device)

        for frame, frame_transposed in zip(total_frames, total_frames_transposed):

            frame_transposed = torch.tensor(frame_transposed, dtype=torch.float, device=x.device)  # Move to the device of x
            frame_transposed = frame_transposed.reshape(frame_transposed.shape[0], 1, frame_transposed.shape[1])
            val_transposed = self.spatial_embedding_transposed(frame_transposed)
            frame_transposed_init = val_transposed
            val_transposed = self.positional_encoding_transposed(val_transposed)
            val_transposed = self.spatial_transformer_transposed(val_transposed)
            val_transposed = val_transposed + self.weights1(frame_transposed_init.clone().detach())
 

            frame = torch.tensor(frame, dtype=torch.float, device=x.device)  # Move to the device of x
            frame = frame.reshape(frame.shape[0], 1, frame.shape[1])          
            val = self.spatial_embedding(frame)
            frame_init = val
            val = self.positional_encoding(val)
            val = self.spatial_transformer(val)       
            val = val + self.weights2(frame_init.clone().detach())            
          
            val_cat = torch.cat((val, val_transposed), 0)
            
            transformer_output = torch.cat((transformer_output, val_cat), 0)
           


        x = transformer_output
        x = self.temporal_positional_encoding(x)
        x = self.temporal_transformer(x)
        x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2])
        x = self.dense_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = torch.sigmoid(x)
        
        return x




def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)



def trainMyModel(model, train_data, val_data, epochs):
    cuda_available = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    X_train, y_train = train_data.to(device), val_data.to(device)

    criterion = nn.BCELoss() # Binary cross entropy

    

    min_train_val_loss = 1000
    best_model = model
    desired_epoch_number = 0
    
    start_time = time.time()
    train_val_loss = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_epoch_loss = 0.0
        train_val_epoch_loss = 0.0
        
        X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle = True)

        train_count = 0 
        for val, target in zip(X_train_val, y_train_val):
            optimizer.zero_grad()
            output = model(val) 
            target = target.unsqueeze(dim = 0)
            individual_loss  = criterion(output, target)

            individual_loss.backward()
            
            # Gradient clipping
            torch_utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Adjust max_norm as needed
            
            optimizer.step()
            train_epoch_loss += individual_loss.item()
            train_count += 1

        train_loss = train_epoch_loss / len(X_train_val)
        train_losses.append(train_loss)
          
        model.eval()
            
        train_val_count = 0
        train_val_values = []
        
        
        with torch.no_grad():
            for val, target in zip(X_test_val, y_test_val):
                y_pred_val = model(val)
                target = target.unsqueeze(dim = 0)
                individual_loss  = criterion(y_pred_val, target)
                train_val_values.append(individual_loss.item())
                train_val_epoch_loss += individual_loss.item() 
                train_val_count += 1
            
            train_val_loss = train_val_epoch_loss / len(X_test_val)
            val_losses.append(train_val_loss)
               
  
        if epoch > 15:
            if train_val_loss <= min_train_val_loss: 
                min_train_val_loss = train_val_loss
                desired_epoch_number = epoch + 1
                best_model = copy.deepcopy(model)
            
        model_at_last_epoch = copy.deepcopy(model) # Save the model at last epoch
        

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.6f}    Val loss: {train_val_loss:.6f}    Best model at epoch: {desired_epoch_number} ")
        

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    time_taken = convert_time_format(time.time() - start_time)
    
    print("The best model was found at epoch nunmber ", desired_epoch_number)
    print("--- %s seconds ---" % time_taken) 
    print("The training loss is ", train_losses[-1])
    
    return best_model, model_at_last_epoch







def evaluateMyModel(model, test_data, test_labels):    
    X_test = test_data
    y_test = test_labels
    

    criterion = nn.BCELoss() # Binary cross entropy


    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_epoch_loss = 0
        predicted_test_values = []
        
        for val, target in zip(X_test, y_test):
            target = target.unsqueeze(dim=0)

            y_pred_val = model(val)
            loss = criterion(y_pred_val, target)
            test_epoch_loss += loss.item()
            predicted_test_values.append(y_pred_val)  # Convert PyTorch tensor to NumPy array
        
        test_loss = test_epoch_loss / len(X_test)
        print("The test loss is ", test_loss)
        
        predicted_test_values = [tensor.cpu().numpy() for tensor in predicted_test_values]
    
        predicted_test_values = np.array(predicted_test_values)

        
        test_dev = abs(np.squeeze(predicted_test_values)-y_test.numpy())
        # Cumulative deviation
        mean_abs_dev = np.mean(test_dev)
        # RMS deviation
        rms_dev = sqrt(mean_squared_error(predicted_test_values, y_test.numpy()))
        print("For exercise :    ", current_exercise, "     and split : ", split)
        print(f'Mean absolute deviation        : {mean_abs_dev : .6f}') 
        print(f'RMS deviation:: {rms_dev : .6f}')
    

        
    return mean_abs_dev

def convert_time_format(sec):
    print("Total time in seconds : ", sec)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec) 



def count_parameters(model): # This method is used to count the number of learnable parameters 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#######################################################################################################################################################################



import torch
torch.cuda.empty_cache()
cuda_available = torch.cuda.is_available()
print("The cuda availability is ", cuda_available)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("The cuda device is ", device)


import time
import joblib

input_dim           = X_train.shape  # Input dimension 
model_dim           = 48 # Associated with the capturing of different features
num_heads           = 4
dim_feedforward     = 56
num_layers          = 6
dropout             = 0.1
stride              = 60
frame_size          = 60 
dense_layer_neurons = 118
epochs              = 100
learning_rate       = 0.00001





all_frames, _ = makeFrames(X_train[0], frame_size, stride)
no_of_frames = len(all_frames)



dim_head = model_dim
frame_shape = all_frames[0].shape
input_dim = all_frames[0].shape  # Input dimension 
print("All frames shape is ", input_dim)
print("The frame size is ", frame_shape)

print("The number of frames are ", no_of_frames, " , the frame size is ", frame_size, " and stride is ", stride)

model = mySpatioTemporalTransformer(input_dim, model_dim, dim_head, frame_size, no_of_frames, num_heads, num_layers, dim_feedforward, stride, dropout, dense_layer_neurons)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_train, y_train = X_train.to(device), y_train.to(device) 


print("Current device:", device)
print("Model output is ", model(X_train[0]))
print("The no. of frames are ", no_of_frames)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 


model.apply(weights_init)

from torchsummary import summary


# Training loop
start_time = time.time()

train_losses = []
val_losses = []

folder_path = 'C:/Aman/Research papers/Quality assessment paper/Codes/Models saved during training/My spatio temporal/'

files = glob.glob(folder_path + "/*.pkl")

print("The files stored are ", files)
print("For exercise : ", current_exercise, "  and split : ", split)
# Remove all the previously stored models

for f in files: 
    os.remove(f) 

print(f"The hyperparameters are : model_dim is = {model_dim} , num_heads = {num_heads} , dim_feedforward = {dim_feedforward} , num_layers = {num_layers} , dropout = {dropout} , stride = {stride} , frame_size = {frame_size} , dense_layer_neurons = {dense_layer_neurons} , epochs = {epochs}, learning_rate = {learning_rate}")

best_model, model_at_last_epoch = trainMyModel(model, X_train, y_train, epochs)
# best_model, model_at_last_epoch = trainMyModelHubberLoss(model, X_train, y_train, epochs)

print("Performance with the best model is :")
evaluateMyModel(best_model, X_test, y_test)
print("Performance with the last epoch model is :")
evaluateMyModel(model_at_last_epoch, X_test, y_test)


print(f"The hyperparameters are : model_dim is = {model_dim} , num_heads = {num_heads} , dim_feedforward = {dim_feedforward} , num_layers = {num_layers} , dropout = {dropout} , stride = {stride} , frame_size = {frame_size} , dense_layer_neurons = {dense_layer_neurons} , epochs = {epochs}, learning_rate = {learning_rate}")
