import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os


## Data Loading - Mass, bioreactor
#path = '/Users/cyjun/OneDrive - Umich/WORKING/Group pjt - Machine learning application'
fn = 'DATASET_for_Python.xlsx'
Raw_raw = pd.read_excel(fn, sheet_name = 'Raw')
PE_raw = pd.read_excel(fn, sheet_name = 'PE')
SFE_raw = pd.read_excel(fn, sheet_name = 'SFE')
Etc_raw = pd.read_excel(fn, sheet_name = 'Etc')



##Extract Variables interested in.
# Raw
Raw_raw.usecols = ['Date','Month','Temp_C','FerricRaw_Mass_kg.d']
Raw = Raw_raw.loc[:,Raw_raw.usecols]
Raw.head(5)
# PE
PE_raw.usecols = ['PE Flow_MGD', 'BODPE_Mass_kg.d', 'VSSPE_Mass_kg.d', 'SPPE_Mass_kg.d','f.TPPE_Mass_kg.d']
PE = PE_raw.loc[:,PE_raw.usecols]
PE.head(5)
# SFE
SFE_raw.usecols = ['SPSFE_Mass_kg.d', 'TPSFE_Mass_kg.d', 'TSSSFE_Mass_kg.d']
SFE = SFE_raw.loc[:,SFE_raw.usecols]
SFE.head(5)
# Etc.
Etc_raw.usecols = ['RAS_FlowMGD', 'MVLSSmg.l','SRT_PredDays', 'Sludge_Blanket_Depth_ft']
Etc = Etc_raw.loc[:,Etc_raw.usecols]
Etc.head(5)

## Merge data 
df = pd.concat([Raw,PE,SFE,Etc], axis=1)


## Data Processing
# Set Date to Index
df['Date'] = pd.to_datetime(df['Date']).dt.date #remove time from datetime
df = df.set_index('Date') # set index with Date
df.info()

# Simplification of Columns name
df.columns = ["Month", "Temp", "Ferric", "PE_flow", "PE_BOD", "PE_VSS", "PE_SP", "PE_TP", "SFE_SP", "SFE_TP", "SFE_TSS", "RAS_flow",
              "MLVSS", "SRT", "SLBk"]
print(df)
#df.head(5)
#df.shape
#df.columns

# Cleaning NaN from dataset
df=df.fillna(df.mean()) # Mean to NaN
#df=df.fillna(method="ffill") #fill with previous value
#df.where(pd.notnull(df), df.mean(), axis='columns')
print(df.info())    

# Normalizing from  0 to 1
min_max_scaler = preprocessing.MinMaxScaler()
scaled_df = min_max_scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, index=df.index, columns=df.columns)


# Set inputdata and outputdata
X = scaled_df.loc[:,["Month", "Temp", "Ferric", "PE_flow", "PE_BOD", "PE_VSS", "PE_SP", "PE_TP", "SFE_SP", "SFE_TP", "SFE_TSS", "RAS_flow",
              "MLVSS", "SRT", "SLBk"]]

unscaled_y = df.loc[:,'SFE_TP']

X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, unscaled_y, test_size=0.25, random_state=113021)
X_train, X_test, y_train, y_test = np.array(X_train_df),  np.array(X_test_df),  np.array(y_train_df),  np.array(y_test_df)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

