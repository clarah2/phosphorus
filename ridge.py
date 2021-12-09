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


## Data Loading
#path = r'/Users/cyjun/Documents/GLWA Datset '
fn = 'DATASET_for_Python.xlsx'
Raw_raw = pd.read_excel(fn, sheet_name = 'Raw')
PE_raw = pd.read_excel(fn, sheet_name = 'PE')
SFE_raw = pd.read_excel(fn, sheet_name = 'SFE')
Etc_raw = pd.read_excel(fn, sheet_name = 'Etc')
print(Raw_raw.columns)
print(PE_raw.columns)
print(SFE_raw.columns)
print(Etc_raw.columns)

##Extract Variables interested in.
# Raw
Raw_raw.usecols = ['Date','Month','Temp_C','TPRaw_Mass_kg.d', 'SPRaw_Mass_kg.d','FerricRaw_Mass_kg.d']
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
df.columns = ["Month", "Temp", "Raw_TP", "Raw_SP", "Ferric", "PE_flow", "PE_BOD", "PE_VSS", "PE_SP", "PE_TP", "SFE_SP", "SFE_TP", "SFE_TSS", "RAS_flow", "MLVSS", "SRT", "SLBk"]
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

column_headings = ['Month', 'PE_TP','Temp', 'Raw_TP', 'Ferric', 'PE_flow', 'PE_BOD', 'SFE_TSS', 'MLVSS', 'SRT', 'SLBk']
important_features = ['PE_TP', 'SFE_TSS']

# Set inputdata and outputdata
X = scaled_df.loc[:,column_headings]
y = df.loc[:,'SFE_TP']

# Split into Train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=113021)
X_train, X_test, y_train, y_test = np.array(X_train),  np.array(X_test),  np.array(y_train),  np.array(y_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
y_train = y_train.astype('int')

print(len(y_train))

plt.plot(range(0, len(y_train)), y_train)
plt.show()

def print_coefs(coefs, names, sort = False):
  itemList = zip(coefs, names)
  if sort:
    itemList = sorted(itemList,  key = lambda x:-np.abs(x[0]))
  return " + ".join("%s * %s" % (round(coef, 3), name)for coef, name in itemList)

# Ridge regression
print('--- Ridge Regression ---')
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:      ', rmse)
print('R^2 Score: ', ridge.score(X_train, y_train))
mycoefs = ridge.coef_
print('--- Ridge Coefficients ---')
print(print_coefs(mycoefs, column_headings))

# Plotting
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')

maxVal = max(max(y_pred), max(y_test))
minVal = min(min(y_pred), min(y_test))
plt.plot([maxVal, minVal], [maxVal, minVal], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

# Plotting with Log Scale
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

maxVal = max(max(y_pred), max(y_test))
minVal = min(min(y_pred), min(y_test))
plt.plot([maxVal, minVal], [maxVal, minVal], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

alphas = np.linspace(0.1, 100, 100)
ridge = Ridge(normalize=True)
coefs = []
for a in alphas:
  ridge.set_params(alpha = a)
  ridge.fit(X_train, y_train)
  coefs.append(ridge.coef_)

alphaPlot = plt.gca()
alphaPlot.plot(alphas, coefs)
#alphaPlot.set_xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Weights')
plt.ylim([-1000, 5000])
plt.show()

kf = KFold(n_splits=5, random_state=113021, shuffle=True)
kf.get_n_splits(X_train)

avg_mses = []
for alph in alphas:
    mses = []
    for train_i, test_i in kf.split(X_train):
        x_in, x_v = X_train[train_i], X_train[test_i]
        y_in, y_v = y_train[train_i], y_train[test_i]
            
        clf = Ridge(alpha=alph)
        clf.fit(x_in, y_in)
            
        preds = clf.predict(x_v)
        mse = mean_squared_error(y_v, preds)
        mses.append(mse)
        
    mse_avg = sum(mses) / len(mses)
    avg_mses.append(mse_avg)

min_val = min(avg_mses)
min_i = avg_mses.index(min_val)
print("Minimum MSE:", round(min_val, 2), "when alpha =", alphas[min_i])

cross_validation_model = RidgeClassifierCV(alphas).fit(X_train, y_train)
print(cross_validation_model.best_score_)


X = scaled_df.loc[:,important_features]
y = df.loc[:,'SFE_TP']

# Split into Train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=113021)
X_train, X_test, y_train, y_test = np.array(X_train),  np.array(X_test),  np.array(y_train),  np.array(y_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
y_train = y_train.astype('int')

# Ridge regression with two features: PE_TP and SFE_TSS
print('--- Ridge Regression ---')
print('PE_TP and SFE_TSS')
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:      ', rmse)
print('R^2 Score: ', ridge.score(X_train, y_train))
mycoefs = ridge.coef_
print('--- Ridge Coefficients ---')
print(print_coefs(mycoefs, important_features))

# Plotting
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')

maxVal = max(max(y_pred), max(y_test))
minVal = min(min(y_pred), min(y_test))
plt.plot([maxVal, minVal], [maxVal, minVal], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

# Plotting with Log Scale
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

maxVal = max(max(y_pred), max(y_test))
minVal = min(min(y_pred), min(y_test))
plt.plot([maxVal, minVal], [maxVal, minVal], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()