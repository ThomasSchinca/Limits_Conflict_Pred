# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 15:15:28 2025

@author: thoma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from scipy.stats import sem, t, ttest_rel, ttest_1samp, ttest_rel
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

df_tot_m= pd.read_csv('Data/Conf.csv',index_col=0,parse_dates=True)
df_roll = pd.read_csv('Data/Conf_rolling.csv',index_col=0,parse_dates=True)
df_etnic = pd.read_csv('Data/Etnic.csv',index_col=0,parse_dates=True)
df_demos= pd.read_csv('Data/Demos.csv',index_col=0,parse_dates=True)
df_popu = pd.read_csv('Data/Population.csv',index_col=0,parse_dates=True)
df_gdp= pd.read_csv('Data/GDP.csv',index_col=0,parse_dates=True)

all_features = {
    'df_roll': df_roll,
    'df_etnic': df_etnic,
    'df_demos': df_demos,
    'df_popu': df_popu,
    'df_gdp': df_gdp
}


df_log_m = np.log(df_tot_m+1)
df_log_m = df_log_m.loc[:,(df_log_m>0).sum()>50]

for key in all_features:
    all_features[key] = all_features[key].loc[:, df_log_m.columns]

tuning_end_date = pd.to_datetime('2017-12-31')
start_date = pd.to_datetime('2018-01-31')
end_date = pd.to_datetime('2024-12-31')

# # Build training dataset for grid search (all data before 2018)
# X_tune1, X_tune2, X_tune3 = [], [], []
# Y_tune = []

# for current_date in pd.date_range(start=df_log_m.index[6], end=tuning_end_date, freq='M'):
#     idx = df_log_m.index.get_loc(current_date)
#     if idx < 6:
#         continue
    
#     dates_window = df_log_m.index[idx-6:idx]
#     x_seq = df_log_m.loc[dates_window].values.T
    
#     # Model 1 features
#     roll_vals = all_features['df_roll'].loc[df_log_m.index[idx-1]].values.reshape(-1, 1)
#     x_aug1 = np.hstack([x_seq, roll_vals])
    
#     # Model 2 features
#     exog_seq = np.vstack([df.loc[df_log_m.index[idx-1]].values for df in all_features.values()]).T
#     x_aug2 = np.hstack([x_seq, exog_seq])
    
#     # Model 3 features
#     x_aug3 = exog_seq
    
#     y_seq = df_log_m.loc[df_log_m.index[idx]].values
    
#     X_tune1.append(x_aug1)
#     X_tune2.append(x_aug2)
#     X_tune3.append(x_aug3)
#     Y_tune.append(y_seq)

# X_tune1 = np.vstack(X_tune1)
# X_tune2 = np.vstack(X_tune2)
# X_tune3 = np.vstack(X_tune3)
# Y_tune = np.hstack(Y_tune)
# param_grid = {
#     'n_estimators': [25,50,100,200],
#     'max_depth': [3,7,12],
#     'min_samples_split': [2, 5, 8],
#     'max_features': ['sqrt', 0.5]
# }

# # Time series cross-validation
# tscv = TimeSeriesSplit(n_splits=3)

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
            
            model = RandomForestRegressor(**params, random_state=0, n_jobs=-1)
            model.fit(X_train, Y_train)
            pred = model.predict(X_val)
            rmse = mean_squared_error(Y_val, pred, squared=False)
            scores.append(rmse)
        
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
        
    return best_params, best_score

# best_params1, best_score1 = grid_search_rf(X_tune1, Y_tune, param_grid, tscv)
# best_params2, best_score2 = grid_search_rf(X_tune2, Y_tune, param_grid, tscv)
# best_params3, best_score3 = grid_search_rf(X_tune3, Y_tune, param_grid, tscv)


# preds_model1 = []
# preds_model2 = []
# preds_model3 = []
# actuals = []
# for current_date in pd.date_range(start=start_date, end=end_date, freq='M'):
#     idx = df_log_m.index.get_loc(current_date)
#     if idx < 6:
#         continue
#     past_dates = df_log_m.index[idx-6:idx]
#     if any(date not in df_log_m.index for date in past_dates):
#         continue
    
#     if current_date.month == 1:
#         X_train = []
#         Y_train = []
#         for ti in range(6, idx):
#             dates_window = df_log_m.index[ti-6:ti]
#             x_seq = df_log_m.loc[dates_window].values.T
#             roll_vals = all_features['df_roll'].loc[df_log_m.index[ti-1]].values.reshape(-1, 1)
#             x_aug = np.hstack([x_seq, roll_vals])
#             y_seq = df_log_m.loc[df_log_m.index[ti]].values
#             X_train.append(x_aug)
#             Y_train.append(y_seq)
#         X_train = np.vstack(X_train)
#         Y_train = np.hstack(Y_train)
#         model1 = RandomForestRegressor(**best_params1, random_state=42, n_jobs=-1)
#         model1.fit(X_train, Y_train)
    
#     X_pred = df_log_m.loc[past_dates].values.T
#     roll_pred = all_features['df_roll'].loc[df_log_m.index[idx-1]].values.reshape(-1, 1)
#     X_pred1 = np.hstack([X_pred, roll_pred])
#     Y_true = df_log_m.loc[current_date].values
#     pred1 = model1.predict(X_pred1)
    
#     exog_at_t_minus_1 = [df.loc[df_log_m.index[idx-1]].values for df in all_features.values()]
#     exog_at_t_minus_1 = np.vstack(exog_at_t_minus_1).T
#     X_pred2 = np.hstack([X_pred, exog_at_t_minus_1])
    
#     if current_date.month == 1:
#         X_train2 = []
#         for ti in range(6, idx):
#             dates_window = df_log_m.index[ti-6:ti]
#             x_seq = df_log_m.loc[dates_window].values.T
#             exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
#             X_train2.append(np.hstack([x_seq, exog_seq]))
        
#         X_train2 = np.vstack(X_train2)
#         model2 = RandomForestRegressor(**best_params2, random_state=42, n_jobs=-1)
#         model2.fit(X_train2, Y_train)
    
#     pred2 = model2.predict(X_pred2)
#     X_pred3 = exog_at_t_minus_1
    
#     if current_date.month == 1:
#         X_train3 = []
#         for ti in range(6, idx):
#             exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
#             X_train3.append(exog_seq)
#         X_train3 = np.vstack(X_train3)
#         model3 = RandomForestRegressor(**best_params3, random_state=42, n_jobs=-1)
#         model3.fit(X_train3, Y_train)
    
#     pred3 = model3.predict(X_pred3)
#     preds_model1.append(pred1)
#     preds_model2.append(pred2)
#     preds_model3.append(pred3)
#     actuals.append(Y_true)

# preds_model1 = np.vstack(preds_model1).flatten()
# preds_model2 = np.vstack(preds_model2).flatten()
# preds_model3 = np.vstack(preds_model3).flatten()
# actuals = np.vstack(actuals).flatten()

# preds_model1 = pd.Series(preds_model1)
# preds_model2 = pd.Series(preds_model2)
# preds_model3 = pd.Series(preds_model3)
# actuals = pd.Series(actuals)

# preds_model1.to_csv("preds_model1.csv")
# preds_model2.to_csv("preds_model2.csv")
# preds_model3.to_csv("preds_model3.csv")
# actuals.to_csv("actuals.csv")


# =============================================================================
# Results
# =============================================================================

preds_model1=pd.read_csv("preds_model1.csv",index_col=(0),parse_dates=True)
preds_model2=pd.read_csv("preds_model2.csv",index_col=(0),parse_dates=True)
preds_model3=pd.read_csv("preds_model3.csv",index_col=(0),parse_dates=True)
actuals=pd.read_csv("actuals.csv",index_col=(0),parse_dates=True)



def mean_ci(data):
    mean_val = np.mean(data)
    ci95 = t.ppf(0.975, len(data)-1) * sem(data)
    return mean_val, ci95


mae1_1m = mean_absolute_error(actuals, preds_model1)
mae2_1m = mean_absolute_error(actuals, preds_model2)
mae3_1m = mean_absolute_error(actuals, preds_model3)
mae1_1m_ci = mean_ci(np.abs(preds_model1 - actuals).values.flatten())[1]
mae2_1m_ci = mean_ci(np.abs(preds_model2 - actuals).values.flatten())[1]
mae3_1m_ci = mean_ci(np.abs(preds_model3 - actuals).values.flatten())[1]

mask_1m = actuals.values.flatten() != 0
actuals_1m_filtered = actuals.values.flatten()[mask_1m]
preds1_1m_filtered = preds_model1.values.flatten()[mask_1m]
preds2_1m_filtered = preds_model2.values.flatten()[mask_1m]
preds3_1m_filtered = preds_model3.values.flatten()[mask_1m]
mape1_1m = mean_absolute_percentage_error(actuals_1m_filtered, preds1_1m_filtered) * 100
mape2_1m = mean_absolute_percentage_error(actuals_1m_filtered, preds2_1m_filtered) * 100
mape3_1m = mean_absolute_percentage_error(actuals_1m_filtered, preds3_1m_filtered) * 100
mape1_1m_ci = mean_ci(np.abs((actuals_1m_filtered - preds1_1m_filtered) / actuals_1m_filtered) * 100)[1]
mape2_1m_ci = mean_ci(np.abs((actuals_1m_filtered - preds2_1m_filtered) / actuals_1m_filtered) * 100)[1]
mape3_1m_ci = mean_ci(np.abs((actuals_1m_filtered - preds3_1m_filtered) / actuals_1m_filtered) * 100)[1]

metrics = ['MAE', 'MAPE']
model1_scores = [mae1_1m, mape1_1m]
model2_scores = [mae2_1m, mape2_1m]
model3_scores = [mae3_1m, mape3_1m]
model1_ci = [mae1_1m_ci, mape1_1m_ci]
model2_ci = [mae2_1m_ci, mape2_1m_ci]
model3_ci = [mae3_1m_ci, mape3_1m_ci]
x = np.arange(len(metrics))
width = 0.15

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
colors = ['#404040', 'grey', 'lightgrey']
labels = ['AR', 'AR+Cov', 'Cov']
for i, ax in enumerate(axes):
    x = np.arange(3)
    scores = [model1_scores[i], model2_scores[i], model3_scores[i]]
    cis = [model1_ci[i], model2_ci[i], model3_ci[i]]
    ax.bar(x, scores, 1, yerr=cis, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(metrics[i])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='both', which='both', length=3)

plt.tight_layout()
plt.show()

def pct_diff(a, b):
    return 100 * (b - a) / a

latex = r"""\begin{table}[ht!]
\centering
\begin{tabular}{lccc}
\hline
Metric & AR & AR+Cov & Cov \\
\hline
"""

for i, metric in enumerate(metrics):
    # Values with CIs
    row = f"{metric} & " \
          f"{model1_scores[i]:.3f} ({model1_ci[i]:.3f}) & " \
          f"{model2_scores[i]:.3f} ({model2_ci[i]:.3f}) & " \
          f"{model3_scores[i]:.3f} ({model3_ci[i]:.3f}) \\\\ \n"

    # Percentage differences
    diff_row = f" &  & " \
                f"{pct_diff(model1_scores[i], model2_scores[i]):+.1f}\\% vs AR & " \
                f"{pct_diff(model1_scores[i], model3_scores[i]):+.1f}\\% vs AR \\\\ \n"

    latex += row + diff_row + r"\hline" + "\n"

latex += r"""\end{tabular}
\caption{Model comparison: mean scores with 95\% CI (in parentheses) and relative differences vs AR.}
\label{tab:model_comparison}
\end{table}
"""




preds_model1=pd.read_csv("preds_model1.csv",index_col=(0),parse_dates=True)
preds_model2=pd.read_csv("preds_model2.csv",index_col=(0),parse_dates=True)
preds_model3=pd.read_csv("preds_model3.csv",index_col=(0),parse_dates=True)
actuals=pd.read_csv("actuals.csv",index_col=(0),parse_dates=True)

L = len(df_log_m.columns)
N = len(preds_model1.iloc[:, 0])
num_seqs = N // L

seq_tr=[]
for df in [preds_model1,preds_model2,preds_model3,actuals]:
    trimmed = df.iloc[:, 0].values[:num_seqs * L]
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
        
    
ttest_1samp(log_diff_2,0)    
ttest_1samp(log_diff_3,0)    

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
# Results (unlogged)
# =============================================================================

preds_model1=pd.read_csv("preds_model1.csv",index_col=(0),parse_dates=True)
preds_model2=pd.read_csv("preds_model2.csv",index_col=(0),parse_dates=True)
preds_model3=pd.read_csv("preds_model3.csv",index_col=(0),parse_dates=True)
actuals=pd.read_csv("actuals.csv",index_col=(0),parse_dates=True)

preds_model1 = np.exp(preds_model1)-1
preds_model2 = np.exp(preds_model2)-1
preds_model3 = np.exp(preds_model3)-1
actuals = np.exp(actuals)-1

def mean_ci(data):
    mean_val = np.mean(data)
    ci95 = t.ppf(0.975, len(data)-1) * sem(data)
    return mean_val, ci95


mae1_1m = mean_absolute_error(actuals, preds_model1)
mae2_1m = mean_absolute_error(actuals, preds_model2)
mae3_1m = mean_absolute_error(actuals, preds_model3)
mae1_1m_ci = mean_ci(np.abs(preds_model1 - actuals).values.flatten())[1]
mae2_1m_ci = mean_ci(np.abs(preds_model2 - actuals).values.flatten())[1]
mae3_1m_ci = mean_ci(np.abs(preds_model3 - actuals).values.flatten())[1]

mask_1m = actuals.values.flatten() != 0
actuals_1m_filtered = actuals.values.flatten()[mask_1m]
preds1_1m_filtered = preds_model1.values.flatten()[mask_1m]
preds2_1m_filtered = preds_model2.values.flatten()[mask_1m]
preds3_1m_filtered = preds_model3.values.flatten()[mask_1m]
mape1_1m = mean_absolute_percentage_error(actuals_1m_filtered, preds1_1m_filtered) * 100
mape2_1m = mean_absolute_percentage_error(actuals_1m_filtered, preds2_1m_filtered) * 100
mape3_1m = mean_absolute_percentage_error(actuals_1m_filtered, preds3_1m_filtered) * 100
mape1_1m_ci = mean_ci(np.abs((actuals_1m_filtered - preds1_1m_filtered) / actuals_1m_filtered) * 100)[1]
mape2_1m_ci = mean_ci(np.abs((actuals_1m_filtered - preds2_1m_filtered) / actuals_1m_filtered) * 100)[1]
mape3_1m_ci = mean_ci(np.abs((actuals_1m_filtered - preds3_1m_filtered) / actuals_1m_filtered) * 100)[1]

metrics = ['MAE', 'MAPE']
model1_scores = [mae1_1m, mape1_1m]
model2_scores = [mae2_1m, mape2_1m]
model3_scores = [mae3_1m, mape3_1m]
model1_ci = [mae1_1m_ci, mape1_1m_ci]
model2_ci = [mae2_1m_ci, mape2_1m_ci]
model3_ci = [mae3_1m_ci, mape3_1m_ci]
x = np.arange(len(metrics))
width = 0.15

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
colors = ['#404040', 'grey', 'lightgrey']
labels = ['AR', 'AR+Cov', 'Cov']
for i, ax in enumerate(axes):
    x = np.arange(3)
    scores = [model1_scores[i], model2_scores[i], model3_scores[i]]
    cis = [model1_ci[i], model2_ci[i], model3_ci[i]]
    ax.bar(x, scores, 1, yerr=cis, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(metrics[i])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='both', which='both', length=3)

plt.tight_layout()
plt.show()

L = len(df_log_m.columns)
N = len(preds_model1.iloc[:, 0])
num_seqs = N // L

seq_tr=[]
for df in [preds_model1,preds_model2,preds_model3,actuals]:
    trimmed = df.iloc[:, 0].values[:num_seqs * L]
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
        
    
ttest_1samp(log_diff_2,0)  

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
# Importance score
# =============================================================================

current_date = pd.to_datetime('2024-12-31')
idx = df_log_m.index.get_loc(current_date)
past_dates = df_log_m.index[idx-6:idx]
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
X_pred = df_log_m.loc[past_dates].values.T
roll_pred = all_features['df_roll'].loc[df_log_m.index[idx-1]].values.reshape(-1, 1)
X_pred1 = np.hstack([X_pred, roll_pred])
Y_true = df_log_m.loc[current_date].values
model1 = RandomForestRegressor()
model1.fit(X_train, Y_train)


exog_at_t_minus_1 = [df.loc[df_log_m.index[idx-1]].values for df in all_features.values()]
exog_at_t_minus_1 = np.vstack(exog_at_t_minus_1).T
X_pred2 = np.hstack([X_pred, exog_at_t_minus_1])
X_train2 = []
for ti in range(6, idx):
    dates_window = df_log_m.index[ti-6:ti]
    x_seq = df_log_m.loc[dates_window].values.T
    exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
    X_train2.append(np.hstack([x_seq, exog_seq]))
X_train2 = np.vstack(X_train2)
model2 = RandomForestRegressor()
model2.fit(X_train2, Y_train)

X_pred3 = exog_at_t_minus_1
X_train3 = []
for ti in range(6, idx):
    exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
    X_train3.append(exog_seq)

X_train3 = np.vstack(X_train3)
model3 = RandomForestRegressor()
model3.fit(X_train3, Y_train)

labels = [f't-{6-i}' for i in range(6)] + ['Conflict Inc.','Ethic frac.', 'Polity IV', 'Population', 'GDP']
imp1 = model1.feature_importances_
imp2 = model2.feature_importances_
imp3 = model3.feature_importances_

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(labels))
width = 0.25
ax.bar(x - width, np.concatenate([imp1,[0]*4]), width, label='AR',color='#404040')
ax.bar(x, imp2, width, label='AR/Cov',color='grey')
ax.bar(x + width, np.concatenate([[0]*6, imp3]), width, label='Cov',color='lightgrey')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=60,fontsize=15)
ax.set_ylabel('Variable Importance',fontsize=15)
ax.legend(fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=13)
plt.tight_layout()
plt.show()

# =============================================================================
# Distribution fatalities
# =============================================================================

ad = pd.DataFrame(np.zeros((432,74)))
ad.index = df_tot_m.index
df_tot_m_full = pd.concat([df_tot_m,ad],axis=1)
data = np.log10(df_tot_m_full + 1).values.flatten()
tick_vals = np.arange(int(np.floor(data.min())), int(np.ceil(data.max())) + 1)
tick_labels = [0]+[f'$10^{int(i)}$' for i in tick_vals if i>0]
plt.figure(figsize=(10, 6))
plt.hist(data, bins=25, color='grey')
plt.xlabel('Number of fatalities per month', fontsize=20)
plt.xticks(tick_vals, tick_labels, fontsize=16)
plt.yticks([],[])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()

plt.figure(figsize=(10, 6))
sns.ecdfplot(data,color='#404040',linewidth=5)
plt.xlabel('Number of fatalities per month', fontsize=20)
plt.ylabel('Proportion', fontsize=20)
plt.xticks(tick_vals, tick_labels, fontsize=16)
plt.yticks([0,0.2,0.4,0.6,0.8,1], ['0','0.2','0.4','0.6','0.8','1'], fontsize=16)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


data = df_tot_m_full.values.flatten()
data=data+1
sorted_data = np.sort(data)
grid_size = len(sorted_data)
ccdf_y = 1.0 - np.arange(1, grid_size + 1) / grid_size
ccdf_x = sorted_data
plt.figure(figsize=(10, 6))
plt.plot(ccdf_x, ccdf_y, marker='.', linestyle='none', alpha=0.7, color='#404040')
plt.xlabel('Number of Fatalities+1 (x)', fontsize=20)
plt.ylabel('P(X ≥ x)', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()


# =============================================================================
# Percentage of zero 
# =============================================================================

test = (df_tot_m==0).sum()/len(df_tot_m)*100
test.sort_values(inplace=True)
selected_indices = np.linspace(0, len(test) - 1, 25, dtype=int)
test_subset = test.iloc[selected_indices]

plt.figure(figsize=(12, 8))
plt.bar(test_subset.index, test_subset.values, color='grey')
plt.ylabel('Percentage of zero fatality months', fontsize=20)
plt.xticks(fontsize=16, rotation=90)
plt.yticks(fontsize=16)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(axis='y')
plt.show()

# =============================================================================
# ACF
# =============================================================================

acf_list=[]
for col in range(len(df_log_m.columns)):
    acf_list.append(acf(df_log_m.iloc[:,col],nlags=15))

acf_list=pd.DataFrame(acf_list)
acf_list=acf_list.iloc[:,1:]
mean_vals = acf_list.mean()
sem_vals = acf_list.sem()
ci95 = 1.96 * sem_vals
lags = acf_list.columns.astype(int)

plt.figure(figsize=(10, 6))
plt.errorbar(lags, mean_vals, yerr=ci95, fmt='-o', color='black',
             ecolor='black', capsize=0)
plt.xlabel('Lag', fontsize=20)
plt.ylabel('Mean ACF ± 95% CI', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

# =============================================================================
# Probabilities
# =============================================================================

df_tot_m_bin = df_tot_m.copy()
df_tot_m_bin[df_tot_m_bin>0]=1

# Shift by -1 to align t and t+1
df_t = df_tot_m_bin.iloc[:-1]     # rows from t=0 to T-2
df_tp1 = df_tot_m_bin.iloc[1:]    # rows from t=1 to T-1

t = df_t.values.flatten()
tp1 = df_tp1.values.flatten()

p_tp1_0_given_t_0 = np.sum((tp1 == 0) & (t == 0)) / np.sum(t == 0)
p_tp1_1_given_t_1 = np.sum((tp1 == 1) & (t == 1)) / np.sum(t == 1)
p_tp1_1_given_t_0 = np.sum((tp1 == 1) & (t == 0)) / np.sum(t == 0)
p_tp1_0_given_t_1 = np.sum((tp1 == 0) & (t == 1)) / np.sum(t == 1)

print(f"P(t+1=0 | t=0): {p_tp1_0_given_t_0:.4f}")
print(f"P(t+1=1 | t=1): {p_tp1_1_given_t_1:.4f}")
print(f"P(t+1=1 | t=0): {p_tp1_1_given_t_0:.4f}")
print(f"P(t+1=0 | t=1): {p_tp1_0_given_t_1:.4f}")

# =============================================================================
# Rolling Window - Exemples
# =============================================================================

df_rolling_mean = np.log(df_tot_m+1).rolling(window=6,min_periods=6).sum()
countries = ['Nigeria', 'Pakistan', 'Colombia', 'Philippines']
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
#axs[0][0].text(x=2000,y=0,s='Fatalities Sum Rolling Window (log)',rotation=90)
for ax, country in zip(axs.flat, countries):
    ax.plot(df_rolling_mean[country],color='grey')
    ax.set_title(country, fontsize=18)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    # for spine in ax.spines.values():
    #     spine.set_visible(False)
    #ax.set_xticks([])
    #ax.set_yticks([])
plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.show()

# =============================================================================
# Increase/Decrease
# =============================================================================
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, sem

# Load 1-month horizon data
preds_model1 = pd.read_csv("preds_model1.csv", index_col=(0), parse_dates=True)
preds_model2 = pd.read_csv("preds_model2.csv", index_col=(0), parse_dates=True)
preds_model3 = pd.read_csv("preds_model3.csv", index_col=(0), parse_dates=True)
actuals = pd.read_csv("actuals.csv", index_col=(0), parse_dates=True)

# Flatten to series
preds_model1_flat = preds_model1.values.flatten()
preds_model2_flat = preds_model2.values.flatten()
preds_model3_flat = preds_model3.values.flatten()
actuals_flat = actuals.values.flatten()

# We need to reconstruct the temporal structure to calculate differences
# Assuming actuals has the same structure as the predictions
# We need actuals at t-1 for each prediction at t

# Reconstruct original df_log_m indices for the prediction period
start_date = pd.to_datetime('2018-01-31')
end_date = pd.to_datetime('2024-12-31')
pred_dates = pd.date_range(start=start_date, end=end_date, freq='M')

# Get actual values at t and t-1
actuals_t = []
actuals_t_minus_1 = []

for current_date in pred_dates:
    if current_date not in df_log_m.index:
        continue
    idx = df_log_m.index.get_loc(current_date)
    if idx < 1:
        continue
    
    # Get all countries' values at t and t-1
    vals_t = df_log_m.loc[current_date].values
    vals_t_minus_1 = df_log_m.loc[df_log_m.index[idx-1]].values
    
    actuals_t.extend(vals_t)
    actuals_t_minus_1.extend(vals_t_minus_1)

actuals_t = np.array(actuals_t)
actuals_t_minus_1 = np.array(actuals_t_minus_1)

# Calculate percentage change
# When actuals[t-1] = 0, divide by 1
denominators = np.where(actuals_t_minus_1 == 0, 1, actuals_t_minus_1)
pct_change = ((actuals_t - actuals_t_minus_1) / denominators) * 100

# Categorize into decrease, stable, increase
categories = np.where(pct_change < -0.1, 'Decrease',
                     np.where(pct_change > 0.1, 'Increase', 'Stable'))

# Make sure we have the same length
min_len = min(len(preds_model1_flat), len(actuals_flat), len(categories))
preds_model1_flat = preds_model1_flat[:min_len]
preds_model2_flat = preds_model2_flat[:min_len]
preds_model3_flat = preds_model3_flat[:min_len]
actuals_flat = actuals_flat[:min_len]
categories = categories[:min_len]

# Calculate MAE for each category and model
def mean_ci(data):
    mean_val = np.mean(data)
    ci95 = t.ppf(0.975, len(data)-1) * sem(data)
    return mean_val, ci95

category_types = ['Decrease', 'Stable', 'Increase']
mae_model1 = []
mae_model2 = []
mae_model3 = []
ci_model1 = []
ci_model2 = []
ci_model3 = []

for cat in category_types:
    mask = categories == cat
    
    # Model 1
    errors1 = np.abs(preds_model1_flat[mask] - actuals_flat[mask])
    mae1 = np.mean(errors1)
    _, ci1 = mean_ci(errors1)
    mae_model1.append(mae1)
    ci_model1.append(ci1)
    
    # Model 2
    errors2 = np.abs(preds_model2_flat[mask] - actuals_flat[mask])
    mae2 = np.mean(errors2)
    _, ci2 = mean_ci(errors2)
    mae_model2.append(mae2)
    ci_model2.append(ci2)
    
    # Model 3
    errors3 = np.abs(preds_model3_flat[mask] - actuals_flat[mask])
    mae3 = np.mean(errors3)
    _, ci3 = mean_ci(errors3)
    mae_model3.append(mae3)
    ci_model3.append(ci3)
    
    print(f"{cat}: {mask.sum()} observations")

# Plot
x = np.arange(len(category_types))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, mae_model1, width, yerr=ci_model1, label='AR', capsize=5, color='#404040')
ax.bar(x, mae_model2, width, yerr=ci_model2, label='AR+Cov', capsize=5, color='grey')
ax.bar(x + width, mae_model3, width, yerr=ci_model3, label='Cov', capsize=5, color='lightgrey')
ax.set_ylabel('MAE')
ax.set_xticks(x)
ax.set_xticklabels(category_types)
ax.legend()
plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 6))
# plt.scatter(pct_change,np.abs(preds_model1_flat - actuals_flat)-np.abs(preds_model2_flat - actuals_flat))
# plt.axhline(0,linestyle='--',color='grey')
# plt.xlabel('Diff in observed fatalities (in %)')
# plt.ylabel('Absolute difference in MAE : AR - AR+Cov')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.scatter(pct_change,np.abs(preds_model1_flat - actuals_flat)-np.abs(preds_model3_flat - actuals_flat),marker='x',color='dimgrey')
# plt.axhline(0,linestyle='--',color='grey')
# plt.xlabel('Diff in observed fatalities (in %)')
# plt.ylabel('Absolute difference in MAE : AR - Cov')
# plt.show()


# =============================================================================
# Table Creation
# =============================================================================

df = pd.read_csv("Data/Conf.csv", index_col=0, parse_dates=True)

n_countries = len(df.columns)
n_country_months = df.shape[0]*df.shape[1]
zero_fatality_pct = (df == 0).mean().mean() * 100
log_fatalities = pd.Series(np.log1p(df).to_numpy().flatten())
fatalities = pd.Series(df.to_numpy().flatten())
summary = {
    'Raw': {
        'Mean': fatalities.mean(),
        'Median': fatalities.median(),
        'Range': (fatalities.min(), fatalities.max())
    },
    'Log': {
        'Mean': log_fatalities.mean(),
        'Median': log_fatalities.median(),
        'Range': (log_fatalities.min(), log_fatalities.max())
    }
}
# Example: overall summary
summary_df = pd.DataFrame({
    'Countries': [n_countries],
    'Country-Months': [n_country_months],
    'Zero-Fatality (%)': [zero_fatality_pct],
    'Mean (Raw)': [summary['Raw']['Mean']],
    'Median (Raw)': [summary['Raw']['Median']],
    'Range (Raw)': [f"{summary['Raw']['Range'][0]}–{summary['Raw']['Range'][1]}"],
    'Mean (Log)': [summary['Log']['Mean']],
    'Median (Log)': [summary['Log']['Median']],
    'Range (Log)': [f"{summary['Log']['Range'][0]}–{summary['Log']['Range'][1]}"]
})
summary_df.to_latex('summary_table.tex', index=False, float_format="%.2f")


