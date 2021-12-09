import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
from sklearn import linear_model
from tensorflow import keras
from keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam



## Data Loading
#path = r'/Users/cyjun/Documents/GLWA Datset '
fn = 'DATASET_for_Python.xlsx'
Raw_raw = pd.read_excel(fn, sheet_name = 'Raw')
PE_raw = pd.read_excel(fn, sheet_name = 'PE')
SFE_raw = pd.read_excel(fn, sheet_name = 'SFE')
Etc_raw = pd.read_excel(fn, sheet_name = 'Etc')

# Concentration based df
Raw_raw.usecols = ['Date', 'Month', 'T_Raw_MGD', 'Recycle_MGD', 'Temp_C', 'BODRaw_Conc._mg.l','NH3Raw_Conc._mg.l','TPRaw_Conc._mg.l','SPRaw_Conc._mg.l','TSSRaw_Conc._mg.l','VSSRaw_Conc._mg.l','Aver. FerricRaw_Conc._mg.l']
Raw_conc = Raw_raw.loc[:,Raw_raw.usecols]

PE_raw.usecols = ['PE Flow_MGD','BODPE_Conc._mg.l','CODPE_Conc._mg.l','TSSPE_Conc._mg.l','VSSPE_Conc._mg.l','SPPE_Conc._mg.l','TPPE_Conc._mg.l','f.TPPE_Mass_kg.d']
PE_conc = PE_raw.loc[:,PE_raw.usecols]
# SFE
SFE_raw.usecols = ['BODSFE_Conc._mg.l',
'NH3SFE_Conc._mg.l',
'SPSFE_Conc._mg.l',
'TPSFE_Conc._mg.l',
'TSSSFE_Conc._mg.l',
'TSSFE_Conc._mg.l']
SFE_conc = SFE_raw.loc[:,SFE_raw.usecols]
SFE_conc.head(5)
# Etc.
Etc_raw.usecols = ['RAS_FlowMGD',
'RAW_TSSmg.l',
'Pred_RASmg.l',
'MLSSmg.l',
'MVLSSmg.l',
'MVLSS.MLSS%',
'SVIml.mg',
'SRT_PredDays',
'SRT_MeasuredDays',
'Sludge_Blanket_Depth_ft',]
Etc_conc = Etc_raw.loc[:,Etc_raw.usecols]
Etc_conc.head(5)

df_conc = pd.concat([Raw_conc,PE_conc,SFE_conc,Etc_conc], axis=1)
df_conc['Date'] = pd.to_datetime(df_conc['Date']).dt.date #remove time from datetime
df_conc = df_conc.set_index('Date') # set index with Date
df_conc.info()

df_conc.columns = ["Month","Raw_Flow", "Recycle_Flow", "Temp", "Raw_BOD","Raw_NH3", "Raw_TP","Raw_SP","Raw_TSS","Raw_VSS","Ferric",
              "PE_Flow","PE_BOD","PE_COD","PE_TSS","PE_VSS","PE_SP","PE_TP","PE_fTP",
              "SFE_BOD","SFE_NH3","SFE_SP","SFE_TP","SFE_TSS","SFE_TS",
              "RAS_Flow","RAS_TSS","Pred_RAS_TSS","MLSS","MLVSS","MLVSS_MLSS","SVI","SRT","SRT_cal","SLBk"]
#["Month", "Temp", "Raw_TP", "Raw_SP", "Ferric", "PE_flow", "PE_BOD", "PE_VSS", "PE_SP", "PE_TP", "SFE_SP", "SFE_TP", "SFE_TSS", "RAS_flow", "MLVSS", "SRT", "SLBk"]
df_conc['week_day'] = np.arange(2,3167)%7

df_conc['Mn'] = (df_conc['week_day'].values == 1)+np.zeros(3165)
df_conc['Ts'] = (df_conc['week_day'].values == 2)+np.zeros(3165)
df_conc['Wed'] = (df_conc['week_day'].values == 3)+np.zeros(3165)
df_conc['Th'] = (df_conc['week_day'].values == 4)+np.zeros(3165)
df_conc['Fr'] = (df_conc['week_day'].values == 5)+np.zeros(3165)
df_conc['Sat'] = (df_conc['week_day'].values == 6)+np.zeros(3165)
df_conc['Sun'] = (df_conc['week_day'].values == 0)+np.zeros(3165)

df_conc['Winter'] = ((df_conc['Month'] > 11)  | (df_conc["Month"]<3)) + np.zeros(3165)
df_conc['Summer'] = ((df_conc['Month'] > 5)  & (df_conc["Month"]<9)) + np.zeros(3165)
df_conc['Fall'] = ((df_conc['Month'] > 8)  & (df_conc["Month"]<12)) + np.zeros(3165)
df_conc['Spring'] = ((df_conc['Month'] > 2)  & (df_conc["Month"]<6)) + np.zeros(3165)

# Moving Average
#df['ma3'] = df['SFE_TP'].rolling(window=3).mean()
df_conc['ma1'] = df_conc['SFE_TP'].rolling(window=1).mean()
# lag a row by group 'id'
#df['ma3'] =  df['ma3'].shift(1)
df_conc['ma1'] =  df_conc['ma1'].shift(1)


# Delete row with NaN
df_conc = df_conc.dropna(subset=['ma1'])
df_conc = df_conc.fillna(df_conc.mean(axis=0)) # Mean to NaN
#df=df.fillna(method="ffill", axis=0) #fill with previous value
#df.where(pd.notnull(df), df.mean(), axis='columns')
print(df_conc.info())    


def outlier(df_conc,col,z): 
    return df_conc[abs(df_conc[col]-np.mean(df_conc[col]))/np.std(df_conc[col])>z].index

outlier = df_conc.loc[outlier(df_conc,"SFE_TP",2)]
print(len(outlier))

test = []
for i in df_conc.index:
    if i not in outlier.index:
        test.append(i)
        
df_conc_clean = df_conc.loc[test]
print(df_conc_clean.info())   

min_max_scaler = preprocessing.MinMaxScaler()
scaled_df_conc = min_max_scaler.fit_transform(df_conc_clean)
scaled_df_conc = pd.DataFrame(scaled_df_conc, index=df_conc_clean.index, columns=df_conc_clean.columns)

X = scaled_df_conc.loc[:,["Month","Raw_Flow", "Recycle_Flow", "Temp", "Raw_BOD","Raw_NH3", "Raw_TP","Raw_SP","Raw_TSS","Raw_VSS","Ferric",
              "PE_Flow","PE_BOD","PE_COD","PE_TSS","PE_VSS","PE_SP","PE_TP","PE_fTP",
              "SFE_BOD","SFE_NH3","SFE_SP","SFE_TSS","SFE_TS",
              "RAS_Flow","RAS_TSS","Pred_RAS_TSS","MLSS","MLVSS","MLVSS_MLSS","SVI","SRT","SRT_cal","SLBk",'Winter', 'Fall', 'Summer', 'Spring','Mn', 'Ts','Wed','Th','Fr','Sat','Sun']]
y = scaled_df_conc.loc[:,'SFE_TP']
unscaled_y = df_conc_clean.loc[:,'SFE_TP']

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
y = scaled_df_conc.loc[:,'SFE_TP']
unscaled_y = df_conc_clean.loc[:,'SFE_TP']
X_train, X_test, y_train, y_test = train_test_split(X, unscaled_y, test_size=0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


backward = 7
X_train_GRU = []
for i in range(backward, X_train.shape[0]):
  X_train_GRU.append(X[i-backward+1:i+1])

X_train_GRU=np.array(X_train_GRU)

X_test_GRU = []
length = X_train.shape[0]
for i in range(backward, X_test.shape[0]):
  X_test_GRU.append(X[i+length-backward+1:i+length+1])

X_test_GRU = np.array(X_test_GRU)

y_test_GRU = unscaled_y[length+backward:]
y_train_GRU = unscaled_y[backward:length]

modelGRU = keras.Sequential()
modelGRU.add(GRU(50, return_sequences=True))
modelGRU.add(GRU(20, return_sequences=False))
modelGRU.add(Dense(10, activation='relu'))
modelGRU.add(Dense(1, activation='linear'))

modelGRU.compile(optimizer=Adam(learning_rate=0.005, beta_1=0.9, beta_2 = 0.999, epsilon=1e-8, amsgrad=False), loss='mean_squared_error')
modelGRU.fit(X_train_GRU, y_train_GRU, batch_size=100, epochs=50)

y_pred_GRU = modelGRU.predict(X_test_GRU)
modelGRU.evaluate(X_test_GRU, y_test_GRU)

print('r2 score', r2_score(y_test_GRU, y_pred_GRU))

plt.scatter(y_test_GRU, y_pred_GRU, c='crimson')
x = [0, 1]
y = [0, 1]
plt.plot(x, y, c='dodgerblue')
plt.title('GRU')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.savefig('GRU.png')
plt.show()
