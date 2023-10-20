import pandas as pd
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import time


df = pd.read_csv('df_pre_model_final_sletningafvif.csv', index_col='studerende_id')

#%%
correlation_matrix = df.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=.5)
plt.title("Korrelationsmatrix")
plt.show()

#%%

df1 = pd.read_csv('df_pre_model_final_sletningaftrevariable.csv', index_col='studerende_id')

correlation_matrix1 = df1.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix1, annot=False, cmap="coolwarm", linewidths=.5)
plt.title("Korrelationsmatrix")
plt.show()
#%%

df3 = pd.read_csv('df_pre_model_final.csv', index_col='Unnamed: 0')

correlation_matrix2 = df3.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix2, annot=False, cmap="coolwarm", linewidths=.5)
plt.title("Korrelationsmatrix")
plt.show()


#%%

y = df['dropout_event']
X = df.drop('dropout_event', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#opret en instans af StratifiedKfold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#%% ADABOOSTING

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report


ada_model = AdaBoostClassifier()

pipeline_grid_ada = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', ada_model)
])

param_grid_ada = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__algorithm': ['SAMME', 'SAMME.R']
}

grid_ada = GridSearchCV(pipeline_grid_ada, param_grid_ada, cv=skf)


#træner modellen
start_time_ada = time.time()
grid_ada.fit(X_train, y_train)
end_time_ada = time.time()

training_time_ada = end_time_ada - start_time_ada

best_ada = grid_ada.best_estimator_
print(cross_val_score(best_ada, X, y, cv=skf))

y_pred_ada = best_ada.predict(X_test)

print(classification_report(y_test, y_pred_ada))
print(f"Best parameters: {grid_ada.best_params_}")
print(f"Best score: {grid_ada.best_score_}")
print(f"Training time for Adaboosting: {training_time_ada: .2f} seconds")

#%% ADABOOST
from sklearn.metrics import average_precision_score
y_prop_ada = best_ada.predict_proba(X_test)[:, 1]


#beregn og udskriv de forskellige metrikker
print("Performance on test set: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ada)}")
print(f"Precision: {precision_score(y_test, y_pred_ada)}")
print(f"Recall: {recall_score(y_test, y_pred_ada)}")
print(f"F1 score: {f1_score(y_test, y_pred_ada)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_ada)}")
print(f"AUC PR: {average_precision_score(y_test, y_prop_ada)}")


y_train_pred_ada = best_ada.predict(X_train)
y_train_prop_ada = best_ada.predict_proba(X_train)[:, 1]


#beregn og udskriv de forskellige metrikker
print("Performance on training set: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_ada)}")
print(f"Precision: {precision_score(y_train, y_train_pred_ada)}")
print(f"Recall: {recall_score(y_train, y_train_pred_ada)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_ada)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_ada)}")
print(f"AUC PR: {average_precision_score(y_train, y_train_prop_ada)}")