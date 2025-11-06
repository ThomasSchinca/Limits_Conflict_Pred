# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 23:02:53 2025

@author: thoma
"""

import numpy as np
import pandas as pd

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
