# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:01:34 2025

@author: thoma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import sem, t, ttest_rel, ttest_1samp, ttest_rel
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

df_acled = pd.read_excel('Acled.xlsx', 'Non_HRP')
df_acled_1 = pd.read_excel('Acled.xlsx', 'HRP_1')
df_acled_2 = pd.read_excel('Acled.xlsx', 'HRP_2')

df_acled['Date'] = pd.to_datetime(df_acled['Year'].astype(str) + ' ' + df_acled['Month'] + ' 1', 
                           format='%Y %B %d')
df_acled = df_acled.pivot_table(index='Date', columns='Country',values='Fatalities', 
    aggfunc='sum',fill_value=0)
df_acled= df_acled.loc[:, (df_acled != 0).any(axis=0)]
df_acled_1['Date'] = pd.to_datetime(df_acled_1['Year'].astype(str) + ' ' + df_acled_1['Month'] + ' 1', 
                           format='%Y %B %d')
df_acled_1 = df_acled_1.pivot_table(index='Date', columns='Country',values='Fatalities', 
    aggfunc='sum',fill_value=0)
df_acled_1= df_acled_1.loc[:, (df_acled_1 != 0).any(axis=0)]
df_acled_2['Date'] = pd.to_datetime(df_acled_2['Year'].astype(str) + ' ' + df_acled_2['Month'] + ' 1', 
                           format='%Y %B %d')
df_acled_2 = df_acled_2.pivot_table(index='Date', columns='Country',values='Fatalities', 
    aggfunc='sum',fill_value=0)
df_acled_2= df_acled_2.loc[:, (df_acled_2 != 0).any(axis=0)]

df_acled = pd.concat([df_acled,df_acled_1,df_acled_2],axis=1)
df_acled=df_acled.iloc[:-1,:]
df_tot_m = df_acled.copy()

df_roll = pd.read_csv('Data/Conf_rolling.csv',index_col=0,parse_dates=True)
df_etnic = pd.read_csv('Data/Etnic.csv',index_col=0,parse_dates=True)
df_demos= pd.read_csv('Data/Demos.csv',index_col=0,parse_dates=True)
df_popu = pd.read_csv('Data/Population.csv',index_col=0,parse_dates=True)
df_gdp= pd.read_csv('Data/GDP.csv',index_col=0,parse_dates=True)
df_tot_m=df_tot_m.loc[:,df_tot_m.columns.intersection(df_roll.columns)]



all_features = {
    'df_roll': df_roll,
    'df_etnic': df_etnic,
    'df_demos': df_demos,
    'df_popu': df_popu,
    'df_gdp': df_gdp
}

# Scale all DataFrames using training data up to 2015
train_end = '2015-12-31'
scalers = {}

df_log_m = np.log(df_tot_m+1)
df_log_m = df_log_m.loc[:,(df_log_m>0).sum()>50]
df_log_m.index = pd.date_range('1997-01-31','2025-08-31',freq='M')

for key in all_features:
    all_features[key] = all_features[key].loc[:, df_log_m.columns]

start_date = pd.to_datetime('2018-01-31')
end_date = pd.to_datetime('2024-12-31')

preds_model1 = []
preds_model2 = []
preds_model3 = []
actuals = []

for current_date in pd.date_range(start=start_date, end=end_date, freq='M'):
    idx = df_log_m.index.get_loc(current_date)
    if idx < 6:
        continue

    past_dates = df_log_m.index[idx-6:idx]
    if any(date not in df_log_m.index for date in past_dates):
        continue
    
    if current_date.month==1:
        X_train = []
        Y_train = []
        for ti in range(6, idx):
            dates_window = df_log_m.index[ti-6:ti]
            x_seq = df_log_m.loc[dates_window].values.T
            roll_vals = all_features['df_roll'].loc[df_log_m.index[ti-1]].values.reshape(-1, 1)
            x_aug = np.hstack([x_seq, roll_vals])
            y_seq = df_log_m.loc[df_log_m.index[ti]].values
            X_train.append(x_aug)
            Y_train.append(y_seq)
        X_train = np.vstack(X_train)
        Y_train = np.hstack(Y_train)
        model1 = RandomForestRegressor()
        model1.fit(X_train, Y_train)

    X_pred = df_log_m.loc[past_dates].values.T
    roll_pred = all_features['df_roll'].loc[df_log_m.index[idx-1]].values.reshape(-1, 1)
    X_pred1 = np.hstack([X_pred, roll_pred])
    Y_true = df_log_m.loc[current_date].values
    pred1 = model1.predict(X_pred1)

    exog_at_t_minus_1 = [df.loc[df_log_m.index[idx-1]].values for df in all_features.values()]
    exog_at_t_minus_1 = np.vstack(exog_at_t_minus_1).T
    X_pred2 = np.hstack([X_pred, exog_at_t_minus_1])

    if current_date.month==1:
        X_train2 = []
        for ti in range(6, idx):
            dates_window = df_log_m.index[ti-6:ti]
            x_seq = df_log_m.loc[dates_window].values.T
            exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
            X_train2.append(np.hstack([x_seq, exog_seq]))
    
        X_train2 = np.vstack(X_train2)

        model2 = RandomForestRegressor()
        model2.fit(X_train2, Y_train)
        
    pred2 = model2.predict(X_pred2)


    X_pred3 = exog_at_t_minus_1
    if current_date.month==1:
        X_train3 = []
        for ti in range(6, idx):
            exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
            X_train3.append(exog_seq)
        X_train3 = np.vstack(X_train3)
        model3 = RandomForestRegressor()
        model3.fit(X_train3, Y_train)
        
    pred3 = model3.predict(X_pred3)
    preds_model1.append(pred1)
    preds_model2.append(pred2)
    preds_model3.append(pred3)
    actuals.append(Y_true)
    
preds_model1 = np.vstack(preds_model1).flatten()
preds_model2 = np.vstack(preds_model2).flatten()
preds_model3 = np.vstack(preds_model3).flatten()
actuals = np.vstack(actuals).flatten()

preds_model1 = pd.Series(preds_model1)
preds_model2 = pd.Series(preds_model2)
preds_model3 = pd.Series(preds_model3)
actuals = pd.Series(actuals)


L = len(df_log_m.columns)
N = len(preds_model1)
num_seqs = N // L

seq_tr=[]
for df in [preds_model1,preds_model2,preds_model3,actuals]:
    trimmed = df.values[:num_seqs * L]
    sequences = trimmed.reshape(num_seqs, L).T  # Transpose to get L rows
    sequences_df = pd.DataFrame(sequences, columns=[f'seq_{i}' for i in range(num_seqs)])
    seq_tr.append(sequences_df.T)

preds_model1,preds_model2,preds_model3,actuals=seq_tr

err_1_l=[]
err_2_l=[]
err_3_l=[]
log_diff_1=[]
log_diff_2=[]
log_diff_3=[]
for col in range(L):
    for i in range(int(len(actuals)/12)):
        err_1 = mean_absolute_error(actuals.iloc[i*12:(i+1)*12,col], preds_model1.iloc[i*12:(i+1)*12,col])
        err_2 = mean_absolute_error(actuals.iloc[i*12:(i+1)*12,col], preds_model2.iloc[i*12:(i+1)*12,col])
        err_3 = mean_absolute_error(actuals.iloc[i*12:(i+1)*12,col], preds_model3.iloc[i*12:(i+1)*12,col])
        
        log_diff_2.append(err_2 - err_1)
        log_diff_3.append(err_3 - err_1)
        log_diff_1.append(err_3 - err_2)
        
        err_1_l.append(err_1)
        err_2_l.append(err_2)
        err_3_l.append(err_3)
        
    
def mean_ci(data):
    mean_val = np.mean(data)
    ci95 = t.ppf(0.975, len(data)-1) * sem(data)
    return mean_val, ci95

mae1=pd.Series(log_diff_2).mean()
mae1_ci = mean_ci(log_diff_2)[1]
mae2=pd.Series(log_diff_3).mean()
mae2_ci = mean_ci(log_diff_3)[1]
mae3=pd.Series(log_diff_1).mean()
mae3_ci = mean_ci(log_diff_1)[1]


rmse_means = [mae1, mae2,mae3]
rmse_cis = [mae1_ci, mae2_ci,mae3_ci]
model_labels = ['AR/Cov - AR','Cov - AR','Cov - AR/Cov']
x = np.arange(len(model_labels))

fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(x, rmse_means, yerr=rmse_cis, fmt='o', capsize=5, markersize=8, color='black')
ax.set_xticks(x,model_labels,fontsize=16)
ax.set_ylabel('MAE',fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axhline(0,linestyle='--',color='grey')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.xlim(-0.5,2.5)
plt.show()


# =============================================================================
# Hyperparameters
# =============================================================================

tuning_end_date = pd.to_datetime('2017-12-31')
start_date = pd.to_datetime('2018-01-31')
end_date = pd.to_datetime('2024-12-31')

# Build training dataset for grid search (all data before 2018)
X_tune1, X_tune2, X_tune3 = [], [], []
Y_tune = []

for current_date in pd.date_range(start=df_log_m.index[6], end=tuning_end_date, freq='M'):
    idx = df_log_m.index.get_loc(current_date)
    if idx < 6:
        continue
    
    dates_window = df_log_m.index[idx-6:idx]
    x_seq = df_log_m.loc[dates_window].values.T
    
    # Model 1 features
    roll_vals = all_features['df_roll'].loc[df_log_m.index[idx-1]].values.reshape(-1, 1)
    x_aug1 = np.hstack([x_seq, roll_vals])
    
    # Model 2 features
    exog_seq = np.vstack([df.loc[df_log_m.index[idx-1]].values for df in all_features.values()]).T
    x_aug2 = np.hstack([x_seq, exog_seq])
    
    # Model 3 features
    x_aug3 = exog_seq
    
    y_seq = df_log_m.loc[df_log_m.index[idx]].values
    
    X_tune1.append(x_aug1)
    X_tune2.append(x_aug2)
    X_tune3.append(x_aug3)
    Y_tune.append(y_seq)

X_tune1 = np.vstack(X_tune1)
X_tune2 = np.vstack(X_tune2)
X_tune3 = np.vstack(X_tune3)
Y_tune = np.hstack(Y_tune)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 0.5]
}

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=3)

def grid_search_rf(X, Y, param_grid, cv):
    """Simple grid search for RandomForest"""
    best_score = float('inf')
    best_params = None
    
    # Generate all parameter combinations
    from itertools import product
    keys = param_grid.keys()
    values = param_grid.values()
    
    total_combinations = np.prod([len(v) for v in values])

    for i, combination in enumerate(product(*values)):
        params = dict(zip(keys, combination))
        scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_train, Y_train)
            pred = model.predict(X_val)
            rmse = mean_squared_error(Y_val, pred, squared=False)
            scores.append(rmse)
        
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
        
    return best_params, best_score

best_params1, best_score1 = grid_search_rf(X_tune1, Y_tune, param_grid, tscv)
best_params2, best_score2 = grid_search_rf(X_tune2, Y_tune, param_grid, tscv)
best_params3, best_score3 = grid_search_rf(X_tune3, Y_tune, param_grid, tscv)

preds_model1 = []
preds_model2 = []
preds_model3 = []
actuals = []
for current_date in pd.date_range(start=start_date, end=end_date, freq='M'):
    idx = df_log_m.index.get_loc(current_date)
    if idx < 6:
        continue
    past_dates = df_log_m.index[idx-6:idx]
    if any(date not in df_log_m.index for date in past_dates):
        continue
    
    if current_date.month == 1:
        X_train = []
        Y_train = []
        for ti in range(6, idx):
            dates_window = df_log_m.index[ti-6:ti]
            x_seq = df_log_m.loc[dates_window].values.T
            roll_vals = all_features['df_roll'].loc[df_log_m.index[ti-1]].values.reshape(-1, 1)
            x_aug = np.hstack([x_seq, roll_vals])
            y_seq = df_log_m.loc[df_log_m.index[ti]].values
            X_train.append(x_aug)
            Y_train.append(y_seq)
        X_train = np.vstack(X_train)
        Y_train = np.hstack(Y_train)
        model1 = RandomForestRegressor(**best_params1, random_state=42, n_jobs=-1)
        model1.fit(X_train, Y_train)
    
    X_pred = df_log_m.loc[past_dates].values.T
    roll_pred = all_features['df_roll'].loc[df_log_m.index[idx-1]].values.reshape(-1, 1)
    X_pred1 = np.hstack([X_pred, roll_pred])
    Y_true = df_log_m.loc[current_date].values
    pred1 = model1.predict(X_pred1)
    
    exog_at_t_minus_1 = [df.loc[df_log_m.index[idx-1]].values for df in all_features.values()]
    exog_at_t_minus_1 = np.vstack(exog_at_t_minus_1).T
    X_pred2 = np.hstack([X_pred, exog_at_t_minus_1])
    
    if current_date.month == 1:
        X_train2 = []
        for ti in range(6, idx):
            dates_window = df_log_m.index[ti-6:ti]
            x_seq = df_log_m.loc[dates_window].values.T
            exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
            X_train2.append(np.hstack([x_seq, exog_seq]))
        
        X_train2 = np.vstack(X_train2)
        model2 = RandomForestRegressor(**best_params2, random_state=42, n_jobs=-1)
        model2.fit(X_train2, Y_train)
    
    pred2 = model2.predict(X_pred2)
    X_pred3 = exog_at_t_minus_1
    
    if current_date.month == 1:
        X_train3 = []
        for ti in range(6, idx):
            exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
            X_train3.append(exog_seq)
        X_train3 = np.vstack(X_train3)
        model3 = RandomForestRegressor(**best_params3, random_state=42, n_jobs=-1)
        model3.fit(X_train3, Y_train)
    
    pred3 = model3.predict(X_pred3)
    preds_model1.append(pred1)
    preds_model2.append(pred2)
    preds_model3.append(pred3)
    actuals.append(Y_true)

preds_model1 = np.vstack(preds_model1).flatten()
preds_model2 = np.vstack(preds_model2).flatten()
preds_model3 = np.vstack(preds_model3).flatten()
actuals = np.vstack(actuals).flatten()

preds_model1 = pd.Series(preds_model1)
preds_model2 = pd.Series(preds_model2)
preds_model3 = pd.Series(preds_model3)
actuals = pd.Series(actuals)

L = len(df_log_m.columns)
N = len(preds_model1)
num_seqs = N // L

seq_tr=[]
for df in [preds_model1,preds_model2,preds_model3,actuals]:
    trimmed = df.values[:num_seqs * L]
    sequences = trimmed.reshape(num_seqs, L).T  # Transpose to get L rows
    sequences_df = pd.DataFrame(sequences, columns=[f'seq_{i}' for i in range(num_seqs)])
    seq_tr.append(sequences_df.T)

preds_model1,preds_model2,preds_model3,actuals=seq_tr

err_1_l=[]
err_2_l=[]
err_3_l=[]
log_diff_1=[]
log_diff_2=[]
log_diff_3=[]
for col in range(L):
    for i in range(int(len(actuals)/12)):
        err_1 = mean_absolute_error(actuals.iloc[i*12:(i+1)*12,col], preds_model1.iloc[i*12:(i+1)*12,col])
        err_2 = mean_absolute_error(actuals.iloc[i*12:(i+1)*12,col], preds_model2.iloc[i*12:(i+1)*12,col])
        err_3 = mean_absolute_error(actuals.iloc[i*12:(i+1)*12,col], preds_model3.iloc[i*12:(i+1)*12,col])
        
        log_diff_2.append(err_2 - err_1)
        log_diff_3.append(err_3 - err_1)
        log_diff_1.append(err_3 - err_2)
        
        err_1_l.append(err_1)
        err_2_l.append(err_2)
        err_3_l.append(err_3)
        
    
def mean_ci(data):
    mean_val = np.mean(data)
    ci95 = t.ppf(0.975, len(data)-1) * sem(data)
    return mean_val, ci95

mae1=pd.Series(log_diff_2).mean()
mae1_ci = mean_ci(log_diff_2)[1]
mae2=pd.Series(log_diff_3).mean()
mae2_ci = mean_ci(log_diff_3)[1]
mae3=pd.Series(log_diff_1).mean()
mae3_ci = mean_ci(log_diff_1)[1]


rmse_means = [mae1, mae2,mae3]
rmse_cis = [mae1_ci, mae2_ci,mae3_ci]
model_labels = ['AR/Cov - AR','Cov - AR','Cov - AR/Cov']
x = np.arange(len(model_labels))

fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(x, rmse_means, yerr=rmse_cis, fmt='o', capsize=5, markersize=8, color='black')
ax.set_xticks(x,model_labels,fontsize=16)
ax.set_ylabel('MAE',fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axhline(0,linestyle='--',color='grey')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.xlim(-0.5,2.5)
plt.show()