# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 14:04:46 2025

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

# ==================== 3-MONTH HORIZON ====================
print("Training and predicting for 3-month horizon...")
start_date = pd.to_datetime('2018-01-31')
end_date = pd.to_datetime('2024-10-31')  

preds_model1_3m = []
preds_model2_3m = []
preds_model3_3m = []
actuals_3m = []

for current_date in pd.date_range(start=start_date, end=end_date, freq='M'):
    idx = df_log_m.index.get_loc(current_date)
    if idx < 6 or idx + 2 >= len(df_log_m.index):
        continue
    
    past_dates = df_log_m.index[idx-6:idx]
    if any(date not in df_log_m.index for date in past_dates):
        continue
    
    # Retrain models every January
    if current_date.month == 1:
        X_train = []
        Y_train_3m = []
        
        for ti in range(6, idx):
            if ti + 2 >= len(df_log_m.index):
                continue
            
            dates_window = df_log_m.index[ti-6:ti]
            x_seq = df_log_m.loc[dates_window].values.T
            roll_vals = all_features['df_roll'].loc[df_log_m.index[ti-1]].values.reshape(-1, 1)
            x_aug = np.hstack([x_seq, roll_vals])
            
            # Sum of next 3 months
            y_3m = df_log_m.loc[df_log_m.index[ti:ti+3]].sum(axis=0).values
            
            X_train.append(x_aug)
            Y_train_3m.append(y_3m)
        
        X_train = np.vstack(X_train)
        Y_train_3m = np.hstack(Y_train_3m)
        
        # Model 1
        model1_3m = RandomForestRegressor()
        model1_3m.fit(X_train, Y_train_3m)
        
        # Model 2
        X_train2 = []
        for ti in range(6, idx):
            if ti + 2 >= len(df_log_m.index):
                continue
            dates_window = df_log_m.index[ti-6:ti]
            x_seq = df_log_m.loc[dates_window].values.T
            exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
            X_train2.append(np.hstack([x_seq, exog_seq]))
        
        X_train2 = np.vstack(X_train2)
        model2_3m = RandomForestRegressor()
        model2_3m.fit(X_train2, Y_train_3m)
        
        # Model 3
        X_train3 = []
        for ti in range(6, idx):
            if ti + 2 >= len(df_log_m.index):
                continue
            exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
            X_train3.append(exog_seq)
        
        X_train3 = np.vstack(X_train3)
        model3_3m = RandomForestRegressor()
        model3_3m.fit(X_train3, Y_train_3m)
    
    # Predict
    X_pred = df_log_m.loc[past_dates].values.T
    roll_pred = all_features['df_roll'].loc[df_log_m.index[idx-1]].values.reshape(-1, 1)
    X_pred1 = np.hstack([X_pred, roll_pred])
    
    Y_true_3m = df_log_m.loc[df_log_m.index[idx:idx+3]].sum(axis=0).values
    
    pred1_3m = model1_3m.predict(X_pred1)
    
    exog_at_t_minus_1 = [df.loc[df_log_m.index[idx-1]].values for df in all_features.values()]
    exog_at_t_minus_1 = np.vstack(exog_at_t_minus_1).T
    X_pred2 = np.hstack([X_pred, exog_at_t_minus_1])
    pred2_3m = model2_3m.predict(X_pred2)
    
    X_pred3 = exog_at_t_minus_1
    pred3_3m = model3_3m.predict(X_pred3)
    
    preds_model1_3m.append(pred1_3m)
    preds_model2_3m.append(pred2_3m)
    preds_model3_3m.append(pred3_3m)
    actuals_3m.append(Y_true_3m)

preds_model1_3m = np.vstack(preds_model1_3m).flatten()
preds_model2_3m = np.vstack(preds_model2_3m).flatten()
preds_model3_3m = np.vstack(preds_model3_3m).flatten()
actuals_3m = np.vstack(actuals_3m).flatten()

preds_model1_3m = pd.Series(preds_model1_3m)
preds_model2_3m = pd.Series(preds_model2_3m)
preds_model3_3m = pd.Series(preds_model3_3m)
actuals_3m = pd.Series(actuals_3m)

mae1 = mean_absolute_error(actuals_3m, preds_model1_3m)
mae2 = mean_absolute_error(actuals_3m, preds_model2_3m)
mae3 = mean_absolute_error(actuals_3m, preds_model3_3m)
rmse1 = mean_squared_error(actuals_3m, preds_model1_3m, squared=False)
rmse2 = mean_squared_error(actuals_3m, preds_model2_3m, squared=False)
rmse3 = mean_squared_error(actuals_3m, preds_model3_3m, squared=False)

def mean_ci(data):
    mean_val = np.mean(data)
    ci95 = t.ppf(0.975, len(data)-1) * sem(data)
    return mean_val, ci95

mae1_ci = mean_ci(np.abs(preds_model1_3m - actuals_3m))[1]
mae2_ci = mean_ci(np.abs(preds_model2_3m - actuals_3m))[1]
mae3_ci = mean_ci(np.abs(preds_model3_3m - actuals_3m))[1]
rmse1_ci = mean_ci((preds_model1_3m - actuals_3m)**2)[1] / (2 * rmse1)
rmse2_ci = mean_ci((preds_model2_3m - actuals_3m)**2)[1] / (2 * rmse2)
rmse3_ci = mean_ci((preds_model3_3m - actuals_3m)**2)[1] / (2 * rmse3)

metrics = ['MAE', 'RMSE']
model1_scores = [mae1, rmse1]
model2_scores = [mae2, rmse2]
model3_scores = [mae3, rmse3]
model1_ci = [mae1_ci, rmse1_ci]
model2_ci = [mae2_ci, rmse2_ci]
model3_ci = [mae3_ci, rmse3_ci]
x = np.arange(len(metrics))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width, model1_scores, width, yerr=model1_ci, label='AR', capsize=5,color='#404040')
ax.bar(x, model2_scores, width, yerr=model2_ci, label='AR+Cov', capsize=5,color='grey')
ax.bar(x + width, model3_scores, width, yerr=model3_ci, label='Cov', capsize=5,color='lightgrey')
ax.set_ylabel('Error')
ax.set_title('MAE and RMSE with 95% CI - 3 months')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.tight_layout()
plt.show()





# ==================== 6-MONTH HORIZON ====================

start_date = pd.to_datetime('2018-01-31')
end_date = pd.to_datetime('2024-06-30')  # Adjusted to ensure 12 months ahead available

preds_model1_12m = []
preds_model2_12m = []
preds_model3_12m = []
actuals_12m = []

for current_date in pd.date_range(start=start_date, end=end_date, freq='M'):
    idx = df_log_m.index.get_loc(current_date)
    if idx < 6 or idx + 11 >= len(df_log_m.index):
        continue
    
    past_dates = df_log_m.index[idx-6:idx]
    if any(date not in df_log_m.index for date in past_dates):
        continue
    
    # Retrain models every January
    if current_date.month == 1:
        X_train = []
        Y_train_12m = []
        
        for ti in range(6, idx):
            if ti + 11 >= len(df_log_m.index):
                continue
            
            dates_window = df_log_m.index[ti-6:ti]
            x_seq = df_log_m.loc[dates_window].values.T
            roll_vals = all_features['df_roll'].loc[df_log_m.index[ti-1]].values.reshape(-1, 1)
            x_aug = np.hstack([x_seq, roll_vals])
            
            # Sum of next 12 months
            y_12m = df_log_m.loc[df_log_m.index[ti:ti+6]].sum(axis=0).values
            
            X_train.append(x_aug)
            Y_train_12m.append(y_12m)
        
        X_train = np.vstack(X_train)
        Y_train_12m = np.hstack(Y_train_12m)
        
        # Model 1
        model1_12m = RandomForestRegressor()
        model1_12m.fit(X_train, Y_train_12m)
        
        # Model 2
        X_train2 = []
        for ti in range(6, idx):
            if ti + 11 >= len(df_log_m.index):
                continue
            dates_window = df_log_m.index[ti-6:ti]
            x_seq = df_log_m.loc[dates_window].values.T
            exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
            X_train2.append(np.hstack([x_seq, exog_seq]))
        
        X_train2 = np.vstack(X_train2)
        model2_12m = RandomForestRegressor()
        model2_12m.fit(X_train2, Y_train_12m)
        
        # Model 3
        X_train3 = []
        for ti in range(6, idx):
            if ti + 11 >= len(df_log_m.index):
                continue
            exog_seq = np.vstack([df.loc[df_log_m.index[ti-1]].values for df in all_features.values()]).T
            X_train3.append(exog_seq)
        
        X_train3 = np.vstack(X_train3)
        model3_12m = RandomForestRegressor()
        model3_12m.fit(X_train3, Y_train_12m)
    
    # Predict
    X_pred = df_log_m.loc[past_dates].values.T
    roll_pred = all_features['df_roll'].loc[df_log_m.index[idx-1]].values.reshape(-1, 1)
    X_pred1 = np.hstack([X_pred, roll_pred])
    
    Y_true_12m = df_log_m.loc[df_log_m.index[idx:idx+6]].sum(axis=0).values
    
    pred1_12m = model1_12m.predict(X_pred1)
    
    exog_at_t_minus_1 = [df.loc[df_log_m.index[idx-1]].values for df in all_features.values()]
    exog_at_t_minus_1 = np.vstack(exog_at_t_minus_1).T
    X_pred2 = np.hstack([X_pred, exog_at_t_minus_1])
    pred2_12m = model2_12m.predict(X_pred2)
    
    X_pred3 = exog_at_t_minus_1
    pred3_12m = model3_12m.predict(X_pred3)
    
    preds_model1_12m.append(pred1_12m)
    preds_model2_12m.append(pred2_12m)
    preds_model3_12m.append(pred3_12m)
    actuals_12m.append(Y_true_12m)

preds_model1_12m = np.vstack(preds_model1_12m).flatten()
preds_model2_12m = np.vstack(preds_model2_12m).flatten()
preds_model3_12m = np.vstack(preds_model3_12m).flatten()
actuals_12m = np.vstack(actuals_12m).flatten()

preds_model1_12m = pd.Series(preds_model1_12m)
preds_model2_12m = pd.Series(preds_model2_12m)
preds_model3_12m = pd.Series(preds_model3_12m)
actuals_12m = pd.Series(actuals_12m)

mae1 = mean_absolute_error(actuals_12m, preds_model1_12m)
mae2 = mean_absolute_error(actuals_12m, preds_model2_12m)
mae3 = mean_absolute_error(actuals_12m, preds_model3_12m)

mape1 = mean_absolute_percentage_error(actuals_12m, preds_model1_12m) 
mape2 = mean_absolute_percentage_error(actuals_12m, preds_model2_12m) 
mape3 = mean_absolute_percentage_error(actuals_12m, preds_model3_12m)

def mean_ci(data):
    mean_val = np.mean(data)
    ci95 = t.ppf(0.975, len(data)-1) * sem(data)
    return mean_val, ci95

mae1_ci = mean_ci(np.abs(preds_model1_12m - actuals_12m))[1]
mae2_ci = mean_ci(np.abs(preds_model2_12m - actuals_12m))[1]
mae3_ci = mean_ci(np.abs(preds_model3_12m - actuals_12m))[1]

# Calculate MAPE CI
mape1_ci = mean_ci(np.abs((actuals_12m - preds_model1_12m) / actuals_12m) )[1]
mape2_ci = mean_ci(np.abs((actuals_12m - preds_model2_12m) / actuals_12m) )[1]
mape3_ci = mean_ci(np.abs((actuals_12m - preds_model3_12m) / actuals_12m) )[1]

metrics = ['MAE', 'MAPE (%)']
model1_scores = [mae1, mape1]
model2_scores = [mae2, mape2]
model3_scores = [mae3, mape3]
model1_ci = [mae1_ci, mape1_ci]
model2_ci = [mae2_ci, mape2_ci]
model3_ci = [mae3_ci, mape3_ci]

x = np.arange(len(metrics))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width, model1_scores, width, yerr=model1_ci, label='AR', capsize=5, color='#404040')
ax.bar(x, model2_scores, width, yerr=model2_ci, label='AR+Cov', capsize=5, color='grey')
ax.bar(x + width, model3_scores, width, yerr=model3_ci, label='Cov', capsize=5, color='lightgrey')

ax.set_ylabel('Error')
ax.set_title('MAE and MAPE with 95% CI - 6 months')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.tight_layout()
plt.show()

preds_model1=pd.read_csv("preds_model1.csv",index_col=(0),parse_dates=True)
preds_model2=pd.read_csv("preds_model2.csv",index_col=(0),parse_dates=True)
preds_model3=pd.read_csv("preds_model3.csv",index_col=(0),parse_dates=True)
actuals=pd.read_csv("actuals.csv",index_col=(0),parse_dates=True)


# ==================== MAE PLOT ====================
# 1-month horizon
mae1_1m = mean_absolute_error(actuals, preds_model1)
mae2_1m = mean_absolute_error(actuals, preds_model2)
mae3_1m = mean_absolute_error(actuals, preds_model3)
mae1_1m_ci = mean_ci(np.abs(preds_model1 - actuals).values.flatten())[1]
mae2_1m_ci = mean_ci(np.abs(preds_model2 - actuals).values.flatten())[1]
mae3_1m_ci = mean_ci(np.abs(preds_model3 - actuals).values.flatten())[1]

# 3-month horizon
mae1_3m = mean_absolute_error(actuals_3m, preds_model1_3m)
mae2_3m = mean_absolute_error(actuals_3m, preds_model2_3m)
mae3_3m = mean_absolute_error(actuals_3m, preds_model3_3m)
mae1_3m_ci = mean_ci(np.abs(preds_model1_3m - actuals_3m))[1]
mae2_3m_ci = mean_ci(np.abs(preds_model2_3m - actuals_3m))[1]
mae3_3m_ci = mean_ci(np.abs(preds_model3_3m - actuals_3m))[1]

# 6-month horizon (using the 12m variables you mentioned you changed)
mae1_6m = mean_absolute_error(actuals_12m, preds_model1_12m)
mae2_6m = mean_absolute_error(actuals_12m, preds_model2_12m)
mae3_6m = mean_absolute_error(actuals_12m, preds_model3_12m)
mae1_6m_ci = mean_ci(np.abs(preds_model1_12m - actuals_12m))[1]
mae2_6m_ci = mean_ci(np.abs(preds_model2_12m - actuals_12m))[1]
mae3_6m_ci = mean_ci(np.abs(preds_model3_12m - actuals_12m))[1]

horizons = ['1-month', '3-month', '6-month']
model1_mae = [mae1_1m, mae1_3m, mae1_6m]
model2_mae = [mae2_1m, mae2_3m, mae2_6m]
model3_mae = [mae3_1m, mae3_3m, mae3_6m]
model1_mae_ci = [mae1_1m_ci, mae1_3m_ci, mae1_6m_ci]
model2_mae_ci = [mae2_1m_ci, mae2_3m_ci, mae2_6m_ci]
model3_mae_ci = [mae3_1m_ci, mae3_3m_ci, mae3_6m_ci]

x = np.arange(len(horizons))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, model1_mae, width, yerr=model1_mae_ci, label='AR', capsize=5, color='#404040')
ax.bar(x, model2_mae, width, yerr=model2_mae_ci, label='AR+Cov', capsize=5, color='grey')
ax.bar(x + width, model3_mae, width, yerr=model3_mae_ci, label='Cov', capsize=5, color='lightgrey')

ax.set_ylabel('MAE')
ax.set_title('MAE Comparison Across Horizons with 95% CI')
ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.legend()
plt.tight_layout()
plt.show()

# ==================== MAPE PLOT ====================
# Filter out zeros for MAPE calculations
# 1-month horizon
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

# 3-month horizon
mask_3m = actuals_3m != 0
actuals_3m_filtered = actuals_3m[mask_3m]
preds1_3m_filtered = preds_model1_3m[mask_3m]
preds2_3m_filtered = preds_model2_3m[mask_3m]
preds3_3m_filtered = preds_model3_3m[mask_3m]

mape1_3m = mean_absolute_percentage_error(actuals_3m_filtered, preds1_3m_filtered) * 100
mape2_3m = mean_absolute_percentage_error(actuals_3m_filtered, preds2_3m_filtered) * 100
mape3_3m = mean_absolute_percentage_error(actuals_3m_filtered, preds3_3m_filtered) * 100
mape1_3m_ci = mean_ci(np.abs((actuals_3m_filtered - preds1_3m_filtered) / actuals_3m_filtered) * 100)[1]
mape2_3m_ci = mean_ci(np.abs((actuals_3m_filtered - preds2_3m_filtered) / actuals_3m_filtered) * 100)[1]
mape3_3m_ci = mean_ci(np.abs((actuals_3m_filtered - preds3_3m_filtered) / actuals_3m_filtered) * 100)[1]

# 6-month horizon
mask_6m = actuals_12m != 0
actuals_6m_filtered = actuals_12m[mask_6m]
preds1_6m_filtered = preds_model1_12m[mask_6m]
preds2_6m_filtered = preds_model2_12m[mask_6m]
preds3_6m_filtered = preds_model3_12m[mask_6m]

mape1_6m = mean_absolute_percentage_error(actuals_6m_filtered, preds1_6m_filtered) * 100
mape2_6m = mean_absolute_percentage_error(actuals_6m_filtered, preds2_6m_filtered) * 100
mape3_6m = mean_absolute_percentage_error(actuals_6m_filtered, preds3_6m_filtered) * 100
mape1_6m_ci = mean_ci(np.abs((actuals_6m_filtered - preds1_6m_filtered) / actuals_6m_filtered) * 100)[1]
mape2_6m_ci = mean_ci(np.abs((actuals_6m_filtered - preds2_6m_filtered) / actuals_6m_filtered) * 100)[1]
mape3_6m_ci = mean_ci(np.abs((actuals_6m_filtered - preds3_6m_filtered) / actuals_6m_filtered) * 100)[1]

model1_mape = [mape1_1m, mape1_3m, mape1_6m]
model2_mape = [mape2_1m, mape2_3m, mape2_6m]
model3_mape = [mape3_1m, mape3_3m, mape3_6m]
model1_mape_ci = [mape1_1m_ci, mape1_3m_ci, mape1_6m_ci]
model2_mape_ci = [mape2_1m_ci, mape2_3m_ci, mape2_6m_ci]
model3_mape_ci = [mape3_1m_ci, mape3_3m_ci, mape3_6m_ci]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, model1_mape, width, yerr=model1_mape_ci, label='AR', capsize=5, color='#404040')
ax.bar(x, model2_mape, width, yerr=model2_mape_ci, label='AR+Cov', capsize=5, color='grey')
ax.bar(x + width, model3_mape, width, yerr=model3_mape_ci, label='Cov', capsize=5, color='lightgrey')

ax.set_ylabel('MAPE (%)')
ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.legend()
plt.tight_layout()
plt.show()