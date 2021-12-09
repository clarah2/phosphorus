## NOTE: This is Python 3 code.
import pandas as pd
import numpy as np
import os
import torch
import torch.optim as optim
from sklearn import preprocessing
import matplotlib.pyplot as plt
import copy


# Data loading
fn = "DATASET_for_Python.xlsx"
Raw = pd.read_excel(fn, sheet_name = 'Raw')
PE = pd.read_excel(fn, sheet_name = 'PE')
SFE = pd.read_excel(fn, sheet_name = 'SFE')
Etc = pd.read_excel(fn, sheet_name = 'Etc')

#Extract Mass Flow
#Raw
Raw.usecols = ['Date', 'T_Raw_MGD', 'Temp_C',
       'TPRaw_Mass_kg.d',
       'SPRaw_Mass_kg.d', 'VSSRaw_Mass_kg.d',
       'FerricRaw_Mass_kg.d']
Raw = Raw.loc[:,Raw.usecols]

#PE
PE.usecols = ['PE Flow_MGD', 'SPPE_Mass_kg.d','TSSPE_Mass_kg.d','VSSPE_Mass_kg.d','TPPE_Mass_kg.d']
PE = PE.loc[:,PE.usecols]

# Merge data for estimating TP
df = pd.concat([Raw,PE], axis=1)
# Set date column as index
df = df.set_index('Date')
print(df.info())     


#### Missing TP - eliminating correspond row
df_processed = copy.copy(df)
idx = df_processed[df_processed['TPPE_Mass_kg.d'].isnull()].index
df_processed.drop(idx, inplace=True)
## filling na 
df_processed = df_processed.fillna(method="ffill")

############# Normalizing from 0 to 1
min_max_scaler = preprocessing.MinMaxScaler()
scaled_df_processed = min_max_scaler.fit_transform(df_processed)

##whole dataset
scaled_df_whole = min_max_scaler.fit_transform(df)

#####Y2 for inverse
y2= df_processed.iloc[:,10:]
y2 = pd.DataFrame(y2).to_numpy()
y2 = y2.reshape(1,-1)
scaler_y = preprocessing.StandardScaler().fit(y2)

#Spliting x_train and y_train
x_data = scaled_df_processed[:,:10]
y_data = scaled_df_processed[:,10:] 

# convert from Dataframe to Arrary
x_data = pd.DataFrame(x_data).to_numpy()
y_data = pd.DataFrame(y_data).to_numpy()


########################################################
#Rogistic regression
########################################################
torch.manual_seed(1)
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
print(x_train.shape) 
print(y_train.shape)

# w, b
W = torch.zeros((10, 1), requires_grad=True) # n 10 x 1
b = torch.zeros(1, requires_grad=True)

hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
print(hypothesis) 


# Cost function
#1. comparison
print(hypothesis)
print(y_train)

# Loss
losses = -(y_train * torch.log(hypothesis) + 
           (1 - y_train) * torch.log(1 - hypothesis))
print(losses)
cost = losses.mean()
print(cost)

# Trainning
# Initializing Model
W = torch.zeros((10, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost calculation
    hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
    cost = -(y_train * torch.log(hypothesis) + 
             (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # H(x) improvement by cost
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # print epoch 
    if epoch % 200 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
        
# Result comparison
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)
#classification
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
print(W)
print(b)

# Using W and b, estimating missing TP and plug into the original dataset 

###### Prediction and Comparison
df_processed = pd.DataFrame(df_processed).to_numpy()
x_pred = df_processed[:,:10]
y_pred = df_processed[:,10:] 

x_pred = torch.FloatTensor(x_pred)
y_pred = torch.FloatTensor(y_pred)


x = y_pred
y = hypothesis

x.size()
y.size()
y.squeeze().size()

x = x.detach().numpy()
y = y.detach().numpy()
w = W.detach().numpy()
plt.plot(x)

y_predicted = scaler_y.inverse_transform(y.T)
plt.plot(y_predicted.T)


print(np.array_equal(x, y))

