import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
# from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model


## Data Loading - Concentration
#path = '/Users/cyjun/OneDrive - Umich/WORKING/Group pjt - Machine learning application'
fn = 'DATASET_for_Python.xlsx'
Raw_raw = pd.read_excel(fn, sheet_name = 'Raw')
PE_raw = pd.read_excel(fn, sheet_name = 'PE')
SFE_raw = pd.read_excel(fn, sheet_name = 'SFE')
Etc_raw = pd.read_excel(fn, sheet_name = 'Etc')


## Select concentration based data and extending data source from bioreacor to entire system.

#Raw Sheet
Raw_raw.usecols = ['Date', 'Month', 'T_Raw_MGD', 'Recycle_MGD', 'Temp_C','BODRaw_Conc._mg.l','NH3Raw_Conc._mg.l',
                   'TPRaw_Conc._mg.l','SPRaw_Conc._mg.l','TSSRaw_Conc._mg.l','VSSRaw_Conc._mg.l','Aver. FerricRaw_Conc._mg.l']
Raw = Raw_raw.loc[:,Raw_raw.usecols]

#PE Sheet
PE_raw.usecols = ['PE Flow_MGD','BODPE_Conc._mg.l','CODPE_Conc._mg.l','TSSPE_Conc._mg.l','VSSPE_Conc._mg.l','SPPE_Conc._mg.l',
                  'f.TPPE_Conc']
PE = PE_raw.loc[:,PE_raw.usecols]

#SFE Sheet
SFE_raw.usecols = ['BODSFE_Conc._mg.l','NH3SFE_Conc._mg.l','SPSFE_Conc._mg.l','TPSFE_Conc._mg.l',
                   'TSSSFE_Conc._mg.l','TSSFE_Conc._mg.l']
SFE = SFE_raw.loc[:,SFE_raw.usecols]

# Etc. Sheet
Etc_raw.usecols = ['RAS_FlowMGD','RAW_TSSmg.l','Pred_RASmg.l','MLSSmg.l','MVLSSmg.l','MVLSS.MLSS%',
                   'SVIml.mg','SRT_PredDays','SRT_MeasuredDays','Sludge_Blanket_Depth_ft',]
Etc = Etc_raw.loc[:,Etc_raw.usecols]

#Merging df
df = pd.concat([Raw,PE,SFE,Etc], axis=1)
df['Date'] = pd.to_datetime(df['Date']).dt.date #remove time from datetime
df = df.set_index('Date') # set index with Date
df.info()

df.columns = ["Month","Raw_Flow", "Recycle_Flow", "Temp", "Raw_BOD","Raw_NH3", "Raw_TP","Raw_SP","Raw_TSS","Raw_VSS","Ferric",
              "PE_Flow","PE_BOD","PE_COD","PE_TSS","PE_VSS","PE_SP","PE_fTP",
              "SFE_BOD","SFE_NH3","SFE_SP","SFE_TP","SFE_TSS","SFE_TS",
              "RAS_Flow","RAS_TSS","Pred_RAS_TSS","MLSS","MLVSS","MLVSS_MLSS","SVI","SRT","SRT_cal","SLBk"]

# Adding additional parameter written by Anar
#week day
df['week_day'] = np.arange(2,3167)%7
df['Mn'] = (df['week_day'].values == 1)+np.zeros(3165)
df['Ts'] = (df['week_day'].values == 2)+np.zeros(3165)
df['Wed'] = (df['week_day'].values == 3)+np.zeros(3165)
df['Th'] = (df['week_day'].values == 4)+np.zeros(3165)
df['Fr'] = (df['week_day'].values == 5)+np.zeros(3165)
df['Sat'] = (df['week_day'].values == 6)+np.zeros(3165)
df['Sun'] = (df['week_day'].values == 0)+np.zeros(3165)

#season
df['Winter'] = ((df['Month'] > 11)  | (df["Month"]<3)) + np.zeros(3165)
df['Summer'] = ((df['Month'] > 5)  & (df["Month"]<9)) + np.zeros(3165)
df['Fall'] = ((df['Month'] > 8)  & (df["Month"]<12)) + np.zeros(3165)
df['Spring'] = ((df['Month'] > 2)  & (df["Month"]<6)) + np.zeros(3165)

# Moving Average > choose ma1
df['t1_TP'] = df['SFE_TP'].rolling(window=1).mean()
df['t1_TSS'] = df['SFE_TSS'].rolling(window=1).mean()
df['t1_SP'] = df['SFE_SP'].rolling(window=1).mean()
# lag a row by group '1d'
df['t1_TP'] = df['t1_TP'].shift(1)
df['t1_TSS'] = df['t1_TSS'].shift(1)
df['t1_SP'] = df['t1_SP'].shift(1)

# Delete first row to remove NA in t1 values
df = df.dropna(subset=['t1_TP'])
df = df.fillna(df.mean(axis=0)) # Mean to NaN
#df=df.fillna(method="ffill", axis=0) #fill with previous value
#df.where(pd.notnull(df), df.mean(), axis='columns')
print(df.info())    

#Normalizing dataset
min_max_scaler = preprocessing.MinMaxScaler()
scaled_df_conc = min_max_scaler.fit_transform(df)
scaled_df_conc = pd.DataFrame(scaled_df_conc, index=df.index, columns=df.columns)

X = scaled_df_conc.loc[:,["Month", "Temp", "Raw_TP","Raw_SP","Ferric",
              "PE_Flow","PE_SP","PE_fTP","MLVSS","SRT","SLBk",
              't1_TP','t1_TSS', 't1_SP']]
unscaled_y = df.loc[:,'SFE_TP'] # removed Outliers

np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(X, unscaled_y, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train.info())  

Lasso_model = linear_model.Lasso(alpha=0.001)
Lasso_model.fit(X_train, y_train)
print(round(Lasso_model.score(X_test, y_test),3), "Lasso Regression")

usefull_columns = []

for i in range(len(X_train.columns)):
  if abs(Lasso_model.coef_[i]) > 0.00001:
    usefull_columns.append(X_train.columns[i])
print(mean_squared_error(y_test, Lasso_model.predict(X_test)))

len(usefull_columns)

X = scaled_df_conc.loc[:,usefull_columns]
unscaled_y = df.loc[:,'SFE_TP'] # removed Outliers

np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(X, unscaled_y, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X.info()) 