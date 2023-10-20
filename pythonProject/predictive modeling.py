#%%
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import numpy as np
from imblearn.over_sampling import SMOTE


df = pd.read_csv('df_pre_model.csv', index_col=False)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f'Proportion of target in the original data \n{df["frafald_event"].value_counts()/len(df)}\n\n'+
      f'Proportion of target in training set \n{train_df["frafald_event"].value_counts()/len(train_df)}\n\n'+
      f'Proportion of target in test set \n{test_df["frafald_event"].value_counts()/len(test_df)}')

#%% kfold CV med stratified sampling og SMOTE

kfoldstrat = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
splits = kfoldstrat.split(df, df['frafald_event'])
smote = SMOTE(random_state=42)

for n, (train_index, test_index) in enumerate(splits):
    X_train, y_train = df.drop("frafald_event", axis=1).loc[train_index], df.loc[train_index, "frafald_event"]
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f'Split no {n + 1}\ntraining set size: {len(train_index)}' +
          f'\ttest set size: {len(test_index)}\nNumber of target instances in training set BEFORE SMOTE\n' +
          f'{y_train.value_counts()}\nNumber of target instances in training set AFTER SMOTE\n' +
          f'{y_train_smote.value_counts()}\nNumber of target instances in the test set\n' +
          f'{df.loc[test_index, "frafald_event"].value_counts()}\n\n')
