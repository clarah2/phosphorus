# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# This notebook contains Lasso, SVR, and Decision Tree Models
# -

# ### Import modules

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ### Data preprocessing from Anar's notebook

fn = "DATASET_for_Python.xlsx"
Raw_raw = pd.read_excel(fn, sheet_name = 'Raw')
PE_raw = pd.read_excel(fn, sheet_name = 'PE')
SFE_raw = pd.read_excel(fn, sheet_name = 'SFE')
Etc_raw = pd.read_excel(fn, sheet_name = 'Etc')

Raw_raw.usecols = ['Date','Month','Temp_C','TPRaw_Mass_kg.d', 'SPRaw_Mass_kg.d','FerricRaw_Mass_kg.d']
Raw = Raw_raw.loc[:,Raw_raw.usecols]
PE_raw.usecols = ['PE Flow_MGD', 'BODPE_Mass_kg.d', 'VSSPE_Mass_kg.d', 'SPPE_Mass_kg.d','f.TPPE_Mass_kg.d']
PE = PE_raw.loc[:,PE_raw.usecols]
# SFE
SFE_raw.usecols = ['SPSFE_Mass_kg.d', 'TPSFE_Mass_kg.d', 'TSSSFE_Mass_kg.d']
SFE = SFE_raw.loc[:,SFE_raw.usecols]
SFE.head(5)
# Etc.
Etc_raw.usecols = ['RAS_FlowMGD', 'MVLSSmg.l','SRT_PredDays', 'Sludge_Blanket_Depth_ft']
Etc = Etc_raw.loc[:,Etc_raw.usecols]
Etc.head(5)

df = pd.concat([Raw,PE,SFE,Etc], axis=1)
df['Date'] = pd.to_datetime(df['Date']).dt.date #remove time from datetime
df = df.set_index('Date') # set index with Date
df.info()

df.columns = ["Month", "Temp", "Raw_TP", "Raw_SP", "Ferric", "PE_flow", "PE_BOD", "PE_VSS", "PE_SP", "PE_TP", "SFE_SP", "SFE_TP", "SFE_TSS", "RAS_flow", "MLVSS", "SRT", "SLBk"]
df['week_day'] = np.arange(2,3167)%7

df['Mn'] = (df['week_day'].values == 1)+np.zeros(3165)
df['Ts'] = (df['week_day'].values == 2)+np.zeros(3165)
df['Wed'] = (df['week_day'].values == 3)+np.zeros(3165)
df['Th'] = (df['week_day'].values == 4)+np.zeros(3165)
df['Fr'] = (df['week_day'].values == 5)+np.zeros(3165)
df['Sat'] = (df['week_day'].values == 6)+np.zeros(3165)
df['Sun'] = (df['week_day'].values == 0)+np.zeros(3165)

df['Winter'] = ((df['Month'] > 11)  | (df["Month"]<3)) + np.zeros(3165)
df['Summer'] = ((df['Month'] > 5)  & (df["Month"]<9)) + np.zeros(3165)
df['Fall'] = ((df['Month'] > 8)  & (df["Month"]<12)) + np.zeros(3165)
df['Spring'] = ((df['Month'] > 2)  & (df["Month"]<6)) + np.zeros(3165)

df=df.fillna(df.mean(axis=0)) # Mean to NaN
#df=df.fillna(method="ffill", axis=0) #fill with previous value
#df.where(pd.notnull(df), df.mean(), axis='columns')
print(df.info())    

min_max_scaler = preprocessing.MinMaxScaler()
scaled_df = min_max_scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, index=df.index, columns=df.columns)

# X = scaled_df.loc[:,['PE_TP','Temp', 'Raw_TP', 'Ferric', 'PE_flow', 'PE_BOD', 'SFE_TSS', 'MLVSS', 'SRT', 'SLBk', 'Winter', 'Fall', 'Summer', 'Spring', 'Mn', 'Ts', 'Wed', 'Th',
#                      'Fr', 'Sat', 'Sun']]
X = scaled_df.loc[:,['PE_TP','Temp', 'Raw_TP', 'Ferric', 'PE_flow', 'PE_BOD', 'SFE_TSS', 'MLVSS', 'SRT', 'SLBk']]
y = scaled_df.loc[:,'SFE_TP']

unscaled_y = df.loc[:,'SFE_TP']

X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, unscaled_y, test_size=0.25, random_state=113021)
X_train, X_test, y_train, y_test = np.array(X_train_df),  np.array(X_test_df),  np.array(y_train_df),  np.array(y_test_df)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# ### Lasso

# +
# K Fold cross validation for alpha hyperparameter
alphas = [0.01, 0.05, 0.1, 0.25, 0.5, 1]

kf = KFold(n_splits=5, random_state=113021, shuffle=True)
kf.get_n_splits(X_train)

avg_mses = []
for alph in alphas:
    mses = []
    for train_i, test_i in kf.split(X_train):
        x_in, x_v = X_train[train_i], X_train[test_i]
        y_in, y_v = y_train[train_i], y_train[test_i]
            
        clf = linear_model.Lasso(alpha=alph)
        clf.fit(x_in, y_in)
            
        preds = clf.predict(x_v)
        mse = mean_squared_error(y_v, preds)
        mses.append(mse)
        
    mse_avg = sum(mses) / len(mses)
    avg_mses.append(mse_avg)

print(avg_mses)
min_val = min(avg_mses)
min_i = avg_mses.index(min_val)
print("Minimum MSE:", round(min_val, 2), "when alpha =", alphas[min_i])
# -

# Determined best value of alpha is 0.05
lasso_mod = linear_model.Lasso(alpha=0.05)
lasso_mod.fit(X_train, y_train)
print(lasso_mod.coef_)
print(X_train_df.columns[lasso_mod.coef_ == 0])

preds = lasso_mod.predict(X_test)
print("Lasso MSE:", mean_squared_error(y_test, preds))

plt.scatter(y_test, preds)
x = [0, 5000]
y = [0, 5000]
plt.plot(x, y)
plt.title('Lasso Regression')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# ### SVR

reg_range = np.logspace(-1,1,3)     # Regularization paramters
kpara_range = np.logspace(-2, 0, 3) # Kernel parameters 

# K fold cross validation for regularization parameter and kernel parameter
for C in reg_range:
    for gamma in kpara_range:
        mses = []
        for train_i, test_i in kf.split(X_train):
            x_in, x_v = X_train[train_i], X_train[test_i]
            y_in, y_v = y_train[train_i], y_train[test_i]

            svr = SVR(kernel="rbf", C=C, gamma=gamma).fit(x_in, y_in)
            preds = svr.predict(x_v)
            mse = mean_squared_error(y_v, preds)
            mses.append(mse)
                  
        svr_avg = sum(mses) / len(mses)
        print("C: ", C, "Gamma: ", gamma, "SVR Avg. MSE: ", svr_avg)

# Lowest MSE identified with C=10 and gamma=1
svr = SVR(kernel="rbf", C=10, gamma=1).fit(X_train, y_train)
preds = svr.predict(X_test)
print("SVR MSE:", mean_squared_error(y_test, preds))

plt.scatter(y_test, preds)
x_vals = [0, 5000]
y_vals = [0, 5000]
plt.plot(x_vals, y_vals)
plt.title('SVR Regression')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# ### Regression Tree

# +
# K fold cross validation for max depth
max_depths = [1, 5, 10, 25, 50, 100]

for d in max_depths:
    mses = []
    for train_i, test_i in kf.split(X_train):
        x_in, x_v = X_train[train_i], X_train[test_i]
        y_in, y_v = y_train[train_i], y_train[test_i]

        regressor = DecisionTreeRegressor(max_depth = d, random_state=113021)
        regressor.fit(x_in, y_in)
        preds = regressor.predict(x_v)
        mse = mean_squared_error(y_v, preds)
        mses.append(mse)
            
        
    rt_avg = sum(mses) / len(mses)
    print("Depth: ", d, "Regression Tree Avg. MSE: ", rt_avg)
# -

# Lowest MSE identified with max_depth = 5
regressor = DecisionTreeRegressor(max_depth = 5, random_state=113021)
regressor.fit(X_train, y_train)
preds = regressor.predict(X_test)
print("Decision tree MSE:", mean_squared_error(y_test, preds))

plt.scatter(y_test, preds)
x_vals = [0, 5000]
y_vals = [0, 5000]
plt.plot(x_vals, y_vals)
plt.title('Decision Tree Regression')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()


