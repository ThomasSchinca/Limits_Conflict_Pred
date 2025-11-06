# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 15:15:28 2025

@author: thoma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas_datareader.wb as wb

df = pd.read_csv("https://ucdp.uu.se/downloads/ged/ged251-csv.zip",
                  parse_dates=['date_start','date_end'],low_memory=False)

df_tot = pd.DataFrame(columns=df.country.unique(),index=pd.date_range(df.date_start.min(),
                                          df.date_end.max()))
df_tot=df_tot.fillna(0)
for i in df.country.unique():
    df_sub=df[df.country==i]
    for j in range(len(df_sub)):
        if df_sub.date_start.iloc[j].month == df_sub.date_end.iloc[j].month:
            df_tot.loc[df_sub.date_start.iloc[j],i]=df_tot.loc[df_sub.date_start.iloc[j],i]+df_sub.best.iloc[j]
        else:
            pass                                                    
                                                     
df_tot_m=df_tot.resample('M').sum()
df_tot_m.drop(['Cameroon', 'France', 'India', 'Malta', 'Mozambique', 'Papua New Guinea','Venezuela'],axis=1,inplace=True)
df_tot_m.to_csv('Data/Conf.csv')

df_rolling_mean = df_tot_m.shift(1).rolling(window=12, min_periods=12).sum()
df_rolling_mean = df_rolling_mean.fillna(method='bfill')
df_rolling_mean= np.log(df_rolling_mean+1)
df_rolling_mean.to_csv('Data/Conf_rolling.csv')

df_etnic = pd.read_csv('HIEF_data.csv')
df_etnic = df_etnic.drop_duplicates()
df_etnic = df_etnic.pivot(index='Year',columns='Country',values='EFindex')
df_etnic = df_etnic.loc[1988:]
df_etnic.loc[1988:1990,'Croatia'] = df_etnic.loc[1988:1990,'Yugoslavia']
df_etnic.loc[1988:1990,'Serbia'] = df_etnic.loc[1988:1990,'Yugoslavia']
df_etnic.loc[1988:1990,'Slovenia'] = df_etnic.loc[1988:1990,'Yugoslavia']
df_etnic.loc[1988:1990,'Macedonia'] = df_etnic.loc[1988:1990,'Yugoslavia']
df_etnic.loc[1988:1992,'Bosnia-Herzegovina'] = df_etnic.loc[1988:1992,'Yugoslavia']
df_etnic.loc[1988:1992,'Czech Republic'] = df_etnic.loc[1988:1992,'Czechoslovakia']
df_etnic.loc[1988:1992,'Slovakia'] = df_etnic.loc[1988:1992,'Czechoslovakia']
df_etnic.loc[1988:1990,'Russia'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Ukraine'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Belarus'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Georgia'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Latvia'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Uzbekistan'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Armenia'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Azerbaijan'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Kazakhstan'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Lithuania'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Moldova'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Kyrgyz Republic'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Turkmenistan'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Tajikistan'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[1988:1990,'Estonia'] = df_etnic.loc[1988:1990,'USSR']
df_etnic.loc[:,'South Sudan'] = df_etnic.loc[:,'Sudan']
df_repeated = df_etnic.loc[df_etnic.index.repeat(12)].copy()
full_index = pd.date_range(start='1989-01', end='2025-01', freq='M')
df_full = pd.DataFrame(columns=df_repeated.columns,index=full_index)
df_full.iloc[:len(df_repeated), :] = df_repeated.values
df_etnic = df_full.fillna(method='ffill').fillna(method='bfill')
df_etnic = df_etnic.rename(columns={'Cambodia':'Cambodia (Kampuchea)','Democratic Republic of Congo':'DR Congo (Zaire)', 
                 'German Democratic Republic':'Germany', 'Cote d\'Ivoire':'Ivory Coast','Swaziland':'Kingdom of eSwatini (Swaziland)', 
                 'Kyrgyz Republic':'Kyrgyzstan', 'Madagascar':'Madagascar (Malagasy)', 'Myanmar':'Myanmar (Burma)', 
                 'Macedonia':'North Macedonia','Russia':'Russia (Soviet Union)', 'Serbia':'Serbia (Yugoslavia)',
                 'Yemen Arab Republic':'Yemen (North Yemen)', 'Zimbabwe':'Zimbabwe (Rhodesia)'})
df_etnic = df_etnic.loc[:,df_tot_m.columns.tolist()]
df_etnic.to_csv('Data/Etnic.csv')


df_demos = pd.read_csv('democracy-index-polity.csv')
df_demos = df_demos.dropna()
df_demos = df_demos.pivot(index='Year',columns='Entity',values='Democracy')
df_demos = df_demos.loc[1988:]
df_repeated = df_demos.loc[df_demos.index.repeat(12)].copy()
full_index = pd.date_range(start='1989-01', end='2025-01', freq='M')
df_full = pd.DataFrame(columns=df_repeated.columns,index=full_index)
df_full.iloc[:len(df_repeated), :] = df_repeated.values
df_demos = df_full.fillna(method='ffill').fillna(method='bfill')
df_demos = df_demos.rename(columns={'Cambodia':'Cambodia (Kampuchea)','Democratic Republic of Congo':'DR Congo (Zaire)', 
                 'Bosnia and Herzegovina':'Bosnia-Herzegovina', 'Cote d\'Ivoire':'Ivory Coast','Eswatini':'Kingdom of eSwatini (Swaziland)', 
                 'Madagascar':'Madagascar (Malagasy)', 'Myanmar':'Myanmar (Burma)', 
                 'Russia':'Russia (Soviet Union)', 'Serbia':'Serbia (Yugoslavia)',
                 'United States':'United States of America',
                 'Yemen Arab Republic':'Yemen (North Yemen)', 'Zimbabwe':'Zimbabwe (Rhodesia)'})
df_demos = df_demos.loc[:,df_tot_m.columns.tolist()]
df_demos.to_csv('Data/Demos.csv')

indicators = ['NY.GDP.PCAP.KD', 'SP.POP.TOTL']
df = wb.download(indicator=indicators, country='all', start=1988, end=2023)
df = df.reset_index()
df = df.rename(columns={
    'NY.GDP.PCAP.KD': 'gdp_per_capita',
    'SP.POP.TOTL': 'population'
})
df = df.dropna()
df['log_gdp_per_capita'] = np.log(df['gdp_per_capita'])
df['log_population'] = np.log(df['population'])
df = df.loc[1764:]
df_popu = df.pivot(index='year',columns='country',values='log_population')
df_repeated = df_popu.loc[df_popu.index.repeat(12)].copy()
full_index = pd.date_range(start='1989-01', end='2025-01', freq='M')
df_full = pd.DataFrame(columns=df_repeated.columns,index=full_index)
df_full.iloc[:len(df_repeated), :] = df_repeated.values
df_popu = df_full.fillna(method='ffill').fillna(method='bfill')
df_popu = df_popu.rename(columns={'Cambodia':'Cambodia (Kampuchea)','Congo, Dem. Rep.':'DR Congo (Zaire)', 
                 'Bosnia and Herzegovina':'Bosnia-Herzegovina', 'Congo, Rep.':'Congo','Cote d\'Ivoire':'Ivory Coast','Eswatini':'Kingdom of eSwatini (Swaziland)', 
                 'Madagascar':'Madagascar (Malagasy)', 'Myanmar':'Myanmar (Burma)','Egypt, Arab Rep.':'Egypt','Gambia, The':'Gambia','Iran, Islamic Rep.':'Iran',
                 'Russian Federation':'Russia (Soviet Union)', 'Serbia':'Serbia (Yugoslavia)',
                 'Kyrgyz Republic':'Kyrgyzstan','Lao PDR':'Laos','Syrian Arab Republic':'Syria','Turkiye':'Turkey',
                 'United States':'United States of America',
                 'Yemen, Rep.':'Yemen (North Yemen)', 'Zimbabwe':'Zimbabwe (Rhodesia)'})
df_popu = df_popu.loc[:,df_tot_m.columns.tolist()]
df_popu.to_csv('Data/Population.csv')

df_gdp = df.pivot(index='year',columns='country',values='log_gdp_per_capita')
df_repeated = df_gdp.loc[df_gdp.index.repeat(12)].copy() 
full_index = pd.date_range(start='1989-01', end='2025-01', freq='M')
df_full = pd.DataFrame(columns=df_repeated.columns,index=full_index)
df_full.iloc[:len(df_repeated), :] = df_repeated.values
df_gdp = df_full.fillna(method='ffill').fillna(method='bfill')
df_gdp = df_gdp.rename(columns={'Cambodia':'Cambodia (Kampuchea)','Congo, Dem. Rep.':'DR Congo (Zaire)', 
                 'Bosnia and Herzegovina':'Bosnia-Herzegovina', 'Congo, Rep.':'Congo','Cote d\'Ivoire':'Ivory Coast','Eswatini':'Kingdom of eSwatini (Swaziland)', 
                 'Madagascar':'Madagascar (Malagasy)', 'Myanmar':'Myanmar (Burma)','Egypt, Arab Rep.':'Egypt','Gambia, The':'Gambia','Iran, Islamic Rep.':'Iran',
                 'Russian Federation':'Russia (Soviet Union)', 'Serbia':'Serbia (Yugoslavia)',
                 'Kyrgyz Republic':'Kyrgyzstan','Lao PDR':'Laos','Syrian Arab Republic':'Syria','Turkiye':'Turkey',
                 'United States':'United States of America',
                 'Yemen, Rep.':'Yemen (North Yemen)', 'Zimbabwe':'Zimbabwe (Rhodesia)'})
df_gdp = df_gdp.loc[:,df_tot_m.columns.tolist()]
df_gdp.to_csv('Data/GDP.csv')


