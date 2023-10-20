import pandas as pd
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_studerende = pd.read_excel('Frafald_grunddata_20230921.xlsx')
df_eksamen = pd.read_excel('Frafald_grunddata_karakter_20230921.xlsx')

#%% FJERNER KOLONNER
#%%
#dropout relaterede columnes
remove_dropout_related_columns = ['frafald_beslutter', 'frafald_begrundelse', 'udmeldbegrundelse', 'studie_status', 'studieslut', 'studie_event']
df_studerende.drop(columns=remove_dropout_related_columns, inplace=True)

#%% SLETTER MISSING VALUES
#sletter de resterende missing values (sletter 425 rækker)
df_studerende.dropna(inplace=True)

#%% STEP 1: KARAKTERER

#opretter en pivot tabel for at tælle antallet af hver karakter for hver studerende
karakter_pivot = pd.pivot_table(df_eksamen, values='eksamenstype', index='studerende_id', columns='karakter', aggfunc='count', fill_value=0).reset_index()

#merge df_studerende og karakter_pivot
df = pd.merge(df_studerende, karakter_pivot, on='studerende_id', how='left')

#lister alle mulige karakterer
karakterer = ['IB', 'U', '-3', '00', '02', '4', '7', '10', '12', 'B']

#erstatter manglende værdier med 0
df[karakterer] = df[karakterer].fillna(0).astype(int)

#%% STEP 2: ORDINÆR- OG RE-EKSAMEN

#opretter en pivot tabel for at tælle antallet af ordinære og re-eksamener for hver studerende
eksamenstype_pivot = pd.pivot_table(df_eksamen, values='karakter', index='studerende_id', columns='eksamenstype', aggfunc='count', fill_value=0).reset_index()

#merge df_studerende og eksamenstype_pivot
df = pd.merge(df, eksamenstype_pivot, on='studerende_id', how='left')

#lister de forskellige typer
eksamentyper = ['o', 'r']

#erstatter NaN værdier med 0
df[eksamentyper] = df[eksamentyper].fillna(0)

df['o'] = df['o'].astype(int) #omkoder til int i stedet for float (nemmere for modellerne at håndtere)
df['r'] = df['r'].astype(int) #omkoder til int i stedet for float (nemmere for modellerne at håndtere)

#%% STEP 3: MERGER SKALA OG ECTS KOLONNER

#nedenstående merger de manglende kolonner og sætter missing values til at være 0

for col in ['skala_kode', 'eksamen_ects']:
    pivot = pd.pivot_table(df_eksamen, values= 'karakter', index = 'studerende_id', columns=col, aggfunc='count', fill_value=0).reset_index()
    pivot.columns = [f"{col}_{x}" if x != 'studerende_id' else x for x in pivot.columns]
    df = pd.merge(df, pivot, on='studerende_id', how='left')

df.fillna(0, inplace=True)


#%% LABEL ENCODING
df['koen'] = df['koen'].replace({'Mand': 0, 'Kvinde': 1})

df['prioritet'] = df['prioritet'].replace({'Lavere prioritet': 0, '1. prioritet': 1})

df['optagsrunde'] = df['optagsrunde'].replace({'1. runde': 0, '2. runde': 1})

df['uddannelse_type'] = df['uddannelse_type'].replace({'Professionsbachelor': 0, 'Bachelor': 1})

#%%
def convert_float_to_int(df, columns_to_int):
    for col in columns_to_int:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

columns_to_int = ['alder_ved_studiestart', 'prio_nr', 'adgangsgivende_eksamen_aar', 'gnms_a_bonus']

df = convert_float_to_int(df, columns_to_int)

#%%
df = pd.get_dummies(df, columns= df.select_dtypes(include=['object']).columns, drop_first=True)

#%%

def bool_to_int(df):
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)
    return df

df = bool_to_int(df)

#%%

def float_to_int(df):
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].astype(int)
    return df

df = float_to_int(df)

#%%

def datetime_to_int(df):
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = (df[col].astype('int64') / 10**9).astype('int64')
    return df

df = datetime_to_int(df)

#%%
from sklearn.ensemble import RandomForestClassifier

X = df.drop('frafald_event', axis=1)
y = df['frafald_event']

rf = RandomForestClassifier()
rf.fit(X, y)

importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
importance = importance.sort_values('importance', ascending=False)
print(importance)
importance.to_csv('feature_importance.csv', index=False)

#%%

eksamen_ects_importance = importance[importance['feature'].str.startswith('eksamen_ects_')]
print(eksamen_ects_importance)

#%%
institutionsland_importance = importance[importance['feature'].str.startswith('institutionsland_')]
print(institutionsland_importance)

#%%
institutionslandkode_importance = importance[importance['feature'].str.startswith('institutionslandekode_')]
print(institutionslandkode_importance)

#%%
nationalitet_importance = importance[importance['feature'].str.startswith('nationalitet_')]
print(nationalitet_importance)

#%%

institutionsnavn_importance = importance[importance['feature'].str.startswith('institutionsnavn_')]
print(institutionsnavn_importance)

#%%

studieretning_importance = importance[importance['feature'].str.startswith('studieretning_')]
print(studieretning_importance)

#%%

uddannelse_importance = importance[importance['feature'].str.startswith('uddannelse_')]
print(uddannelse_importance)

#%%
eksamensnavn_importance = importance[importance['feature'].str.startswith('eksamen_navn_')]
print(eksamensnavn_importance)

#%%
bedomdato_importance = importance[importance['feature'].str.startswith('bedoemmelsesdato_')]
print(bedomdato_importance)

#%%

eksamenkode_importance = importance[importance['feature'].str.startswith('eksamen_kode_')]
print(eksamenkode_importance)

#%%

resultatid_importance = importance[importance['feature'].str.startswith('resultat_id_')]
print(resultatid_importance)


#%%

top_10_importance = importance.head(10)
print(top_10_importance)
