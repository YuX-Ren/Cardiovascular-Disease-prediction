#!/usr/bin/env python
# coding: utf-8



## Standard libraries
import os
import math
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()
## Progress bar
from tqdm import tqdm
import torch
print("Using torch", torch.__version__)
torch.manual_seed(42) # Setting the seed
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


url = './cardiovascular.txt'
data = pd.read_csv(url,sep=';',decimal=',')

# let's separate index from other columns
data.index = data.iloc[:,0]
df = data.iloc[:,1:]
df = df.drop(['chd'],axis=1)
df.famhist=[i=='Present' for i in df.famhist ]
df = df.astype('float')
df=StandardScaler().fit_transform(df)
# sbp	tobacco	ldl	adiposity	famhist	typea	obesity	alcohol	age
df=pd.DataFrame(df,columns=['sbp', 'tobacco', 'ldl', 'adiposity','famhist','typea', 'obesity','alcohol','age'])
df.index=data.index


#PCA
pca=PCA(n_components=1) # split in 5 components

principalComponents = pca.fit_transform(df)

factors_Df = pd.DataFrame(data = principalComponents)#columns =['PC1','PC2','PC3','PC4','PC5'])

factors_Df.index=df.index
# df=factors_Df


df=np.array(df)
y=data['chd']
y=np.array(y)
class SimpleClassifier(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, 32)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 32)
        self.linear6 = nn.Linear(32, num_outputs)
        # self.linear7 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.act_fn(x)        
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        x = self.act_fn(x)
        x = self.linear4(x)
        x = self.act_fn(x)
        x = self.linear5(x)
        x = self.act_fn(x)
        x = self.linear6(x)
        # x = self.act_fn(x)
        # x = self.linear1(x)
        # x = self.act_fn(x)
        # x = x+self.linear2(x)
        # x = self.act_fn(x)
        # x = x+self.linear3(x)
        # x = self.act_fn(x)
        # x = x+self.linear4(x)
        # x = self.act_fn(x)
        # x = x+self.linear5(x)
        # x = self.act_fn(x)
        # x = x+self.linear6(x)
        # x = self.act_fn(x)
        # x = self.linear7(x)
        return x


# model = SimpleClassifier(num_inputs=9, num_hidden=128, num_outputs=1)
# Printing a module shows all its submodules
# print(model)

import torch.utils.data as data
class XORDataset(data.Dataset):

    def __init__(self):
        super().__init__()
        self.generate()

    def generate(self):
        self.data,self.label = torch.Tensor(df),torch.Tensor(y)

    def __len__(self):
        return y.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label



loss_module = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
dataset = XORDataset()
size= y.size
# train_dataset, test_dataset = data.random_split(dataset,[int(size*0.9),size-int(size*0.9)])
# train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)


num_folds = 20
from sklearn.model_selection import KFold
from torch.utils.data import Subset
# Create an instance of the KFold class
kf = KFold(n_splits=num_folds)

# Initialize an empty list to store the accuracy scores
acc_scores = []
acc=np.array([0.0]*20)
i=0
# Iterate over the folds
for train_index, test_index in kf.split(dataset):
    # Split the data into training and test sets

    model = SimpleClassifier(num_inputs=9, num_hidden=128, num_outputs=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_dataset, test_dataset = Subset(dataset,train_index),Subset(dataset,test_index)
    train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
        # Set model to train mode
        model.train()

        # Training loop
        for epoch in tqdm(range(num_epochs)):
            for data_inputs, data_labels in data_loader:

                ## Step 1: Move input data to device (only strictly necessary if we use GPU)
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)
                preds = model(data_inputs)
                preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
                loss = loss_module(preds, data_labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



    train_model(model, optimizer, train_data_loader, loss_module)




    def eval_model(model, data_loader):
        model.eval() # Set model to eval mode
        true_preds, num_preds = 0., 0.

        with torch.no_grad(): # Deactivate gradients for the following code
            for data_inputs, data_labels in data_loader:

                # Determine prediction of model on dev set
                data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
                preds = model(data_inputs)
                preds = preds.squeeze(dim=1)
                preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
                pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1

                # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
                true_preds += (pred_labels == data_labels).sum()
                num_preds += data_labels.shape[0]

        acc[i] = true_preds / num_preds
        # print(f"Accuracy of the model: {100.0*acc[i]:4.2f}%")


    # drop_last -> Don't drop the last batch although it is smaller than 128
    test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)
    eval_model(model,test_data_loader)
    i+=1
print("%0.2f accuracy with a standard deviation of %0.2f by NN" % (acc.mean(), acc.std()))








