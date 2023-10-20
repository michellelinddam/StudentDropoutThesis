
#%%


y = df['frafald_event']
X = df.drop('frafald_event', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#%% LOGISTISK REGRESSION MED STRATIFIED K-FOLD CV

#opret en instans af logistisk regression
logistic_model = LogisticRegression(max_iter=10000, solver='saga')
#LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.9, C=1.0, max_iter=10000)

#opret en instans af StratifiedKfold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#opret en pipeline, der først standardiserer data og derefter træner
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', logistic_model)
])

#%% ELASTICNET
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, balanced_accuracy_score

#definerer scores
scores_lr = {
    'f1_score': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score),
    'balanced_accuracy': make_scorer(balanced_accuracy_score)
}

#definer parametrene
param_grid = {
    'model__penalty': ['elasticnet'],
    'model__solver': ['saga'],
    'model__l1_ratio': [0.1, 0.5, 0.9],
    'model__C': [0.001, 0.01, 0.1, 1.0, 10],
    'model__max_iter': [10000]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=skf,
    scoring=scores_lr,
    refit='f1_score',
    n_jobs=-1,
    verbose=2)

#træn modellen
grid_search.fit(X_train, y_train)

#resultater
print(f"Best parameters for Elastic Net: {grid_search.best_params_}")
print(f"Best score for Elastic Net: {grid_search.best_score_}")

#%%

params_l1_l2 = {
    'model__penalty': ['l1', 'l2'],
    'model__C': [0.001, 0.01, 0.1, 1, 10],
    'model__solver': ['liblinear', 'saga'],
    'model__max_iter': [10000]
}

grid_search_l1_l2 = GridSearchCV(
    estimator=pipeline,
    param_grid=params_l1_l2,
    cv=skf,
    scoring=scores_lr,
    refit='f1_score',
    n_jobs=-1,
    verbose=2)

grid_search_l1_l2.fit(X_train, y_train)

print("Best parameters for L1/L2:", grid_search_l1_l2.best_params_)
print("Best score for L1/L2:", grid_search_l1_l2.best_score_)

#%%
#bruger den bedste estimator fundet af GridSearchCV
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

#beregn og udskriv de forskellige metrikker
print("Performance on test set: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1 score: {f1_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")

#%%

y_train_pred = best_model.predict(X_train)

#beregn og udskriv de forskellige metrikker
print("Performance on training set: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred)}")
print(f"Precision: {precision_score(y_train, y_train_pred)}")
print(f"Recall: {recall_score(y_train, y_train_pred)}")
print(f"F1 score: {f1_score(y_train, y_train_pred)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred)}")

#%%

from sklearn.utils import resample
from sklearn.metrics import accuracy_score

#antal bootstrap iterationer
n_iterations = 5000
bootstrap_scores = [] #tom liste til at gemme estimater

for i in range(n_iterations):
    #trækker en tilfældig prøve fra data med resample
    X_resample, y_resample = resample(X_test, y_test)
    #træner modellen på resampled date
    y_pred_bs = best_model.predict(X_resample)
    score = accuracy_score(y_resample, y_pred_bs)
    bootstrap_scores.append(score)

#95% konfidensinterval
lower = np.percentile(bootstrap_scores, 2.5)
upper = np.percentile(bootstrap_scores, 97.5)

print(f"95% konfidensinterval for nøjagtighed: ({lower: .2f}, {upper: .2f}")

#%% RANDOM FOREST MED STRATIFIED K-FOLD CV - CREATING THE PIPELINE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


rf_model = RandomForestClassifier(random_state=42)

pipeline_rf = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', rf_model)
])

#%%  RANDOM PARAMETER SEARCH

param_distributions = {
    'model__n_estimators': [350, 400],
    'model__max_depth': [5, 10, 20],
    'model__min_samples_split': [5, 10, 20],
    'model__min_samples_leaf': [5, 10, 20],
    'model__max_features': ['sqrt'] ,
    'model__bootstrap': [True, False],
    'model__class_weight': ['balanced'],
    'model__ccp_alpha': [0.1, 0.5]
}

random_search = RandomizedSearchCV(
    estimator= pipeline_rf,
    param_distributions=param_distributions,
    n_iter=500,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

#træner modellen
random_search.fit(X_train, y_train)

#resultater
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_}")

#Best parameters fra random search: {'model__n_estimators': 300, 'model__min_samples_split': 5, 'model__min_samples_leaf': 1, 'model__max_features': 'sqrt', 'model__max_depth': 20, 'model__class_weight': 'balanced', 'model__ccp_alpha': 0.0, 'model__bootstrap': True}
#Best score: 0.8504748392983771 - her er det F1

#Best parameters: {'model__n_estimators': 350, 'model__min_samples_split': 5, 'model__min_samples_leaf': 20, 'model__max_features': 'sqrt', 'model__max_depth': 5, 'model__class_weight': 'balanced', 'model__ccp_alpha': 0.1, 'model__bootstrap': False}
#Best score: 0.9106797320541625 - her er det ROC AUC


from sklearn.model_selection import GridSearchCV

param_grid_rf = {
    'model__n_estimators': [350, 400, 450],
    'model__max_depth': [5, 10, 15],
    'model__min_samples_split': [5, 10, 15],
    'model__min_samples_leaf': [20, 25, 30],
    'model__max_features': ['sqrt'],
    'model__bootstrap': [True, False],
    'model__class_weight': ['balanced'],
    'model__ccp_alpha': [0.1, 0.2, 0.3]
}

grid_search_rf = GridSearchCV(
    estimator= pipeline_rf,
    param_grid=param_grid_rf,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

#træner modellen
grid_search_rf.fit(X_train, y_train)

#resultater
print(f"Best parameters: {grid_search_rf.best_params_}")
print(f"Best score: {grid_search_rf.best_score_}")

#Best parameters: {'model__bootstrap': True, 'model__ccp_alpha': 0.0, 'model__class_weight': 'balanced', 'model__max_depth': 20, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 1, 'model__min_samples_split': 5, 'model__n_estimators': 300}
#Best score: 0.8504748392983771 #f1

#Best parameters: {'model__bootstrap': False, 'model__ccp_alpha': 0.1, 'model__class_weight': 'balanced', 'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 20, 'model__min_samples_split': 5, 'model__n_estimators': 350}
#Best score: 0.9106797320541625 #roc auc


#%%

from sklearn.metrics import confusion_matrix, classification_report

#bruger den bedste estimator fundet af GridSearchCV
best_model_rf = grid_search_rf.best_estimator_
y_pred_rf = best_model_rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))

matrix = confusion_matrix(y_test, y_pred_rf)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16, 7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)

class_names = ['Not dropout', 'Dropout']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation = 25)
plt.yticks(tick_marks2, class_names, rotation = 0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

#%%
#beregn og udskriv de forskellige metrikker
print("Performance on test set: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf)}")
print(f"Recall: {recall_score(y_test, y_pred_rf)}")
print(f"F1 score: {f1_score(y_test, y_pred_rf)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_rf)}")

# Performance on test set:
# Accuracy: 0.8718381112984823
# Precision: 0.7428851815505397
# Recall: 0.7960042060988434
# F1 score: 0.7685279187817259
# ROC AUC: 0.8477527743190802
#%%
from joblib import dump
dump(best_model_rf, 'best_rf_model.joblib')

import pickle
with open('best_rf_model.pkl', 'wb') as file:
    pickle.dump(best_model_rf, file)

#%%

y_train_pred_rf = best_model_rf.predict(X_train)

#beregn og udskriv de forskellige metrikker
print("Performance on training set: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_rf)}")
print(f"Precision: {precision_score(y_train, y_train_pred_rf)}")
print(f"Recall: {recall_score(y_train, y_train_pred_rf)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_rf)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_rf)}")


#%%
metrics_train = {"accuracy": 0.88, "precision": 0.76, "recall": 0.79, "f1": 0.77, "roc_auc": 0.85}
metrics_test = {"accuracy": 0.87, "precision": 0.74, "recall": 0.80, "f1": 0.77, "roc_auc": 0.85}

labels = metrics_train.keys()
train_vals = metrics_train.values()
test_vals = metrics_test.values()

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train_vals, width, label='Train')
rects2 = ax.bar(x + width/2, test_vals, width, label='Test')

ax.set_ylabel('Scores')
ax.set_title('Performance metrics for train and test datasets')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

# Performance on test set:
# Accuracy: 0.8718381112984823
# Precision: 0.7428851815505397
# Recall: 0.7960042060988434
# F1 score: 0.7685279187817259
# ROC AUC: 0.8477527743190802

# Performance on training set:
# Accuracy: 0.8760894011807703
# Precision: 0.7571176618795666
# Recall: 0.7899579390115667
# F1 score: 0.7731892448218192
# ROC AUC: 0.8487395220767734

#%% ADABOOSTING

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report


ada_model = AdaBoostClassifier()

pipeline_grid_ada = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', ada_model)
])

param_grid_ada = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__algorithm': ['SAMME', 'SAMME.R']
}

grid_ada = GridSearchCV(pipeline_grid_ada, param_grid_ada, cv=skf)

grid_ada.fit(X_train, y_train)

best_ada = grid_ada.best_estimator_
print(cross_val_score(best_ada, X, y, cv=skf))

y_pred_ada = best_ada.predict(X_test)

print(classification_report(y_test, y_pred_ada))

#%%
print(f"Best parameters: {grid_ada.best_params_}")
print(f"Best score: {grid_ada.best_score_}")

#%% ADABOOST

#beregn og udskriv de forskellige metrikker
print("Performance on test set: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ada)}")
print(f"Precision: {precision_score(y_test, y_pred_ada)}")
print(f"Recall: {recall_score(y_test, y_pred_ada)}")
print(f"F1 score: {f1_score(y_test, y_pred_ada)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_ada)}")

y_train_pred_ada = best_ada.predict(X_train)

#beregn og udskriv de forskellige metrikker
print("Performance on training set: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_ada)}")
print(f"Precision: {precision_score(y_train, y_train_pred_ada)}")
print(f"Recall: {recall_score(y_train, y_train_pred_ada)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_ada)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_ada)}")


#%% ADABOOST

matrix_ada = confusion_matrix(y_test, y_pred_ada)
matrix_ada = matrix_ada.astype('float') / matrix_ada.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16, 7))
sns.set(font_scale=1.4)
sns.heatmap(matrix_ada, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)

class_names = ['Not dropout', 'Dropout']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation = 25)
plt.yticks(tick_marks2, class_names, rotation = 0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for ADAboost Model')
plt.show()
#%% GRADIENT BOOSTING

from sklearn.ensemble import GradientBoostingClassifier

gradient = GradientBoostingClassifier()

pipeline_grid_grad = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', gradient)
])

param_grid_grad = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.8, 0.9, 1.0],
    'model__max_depth': [3, 4, 5],
    'model__min_samples_split': [2, 3, 4],
    'model__min_samples_leaf': [1, 2, 3]
}

grid_grad = GridSearchCV(pipeline_grid_grad, param_grid_grad, cv=skf)

grid_grad.fit(X_train, y_train)

best_grad = grid_grad.best_estimator_
print(cross_val_score(best_grad, X, y, cv=skf))


y_pred_grad = best_grad.predict(X_test)

print(classification_report(y_test, y_pred_grad))

#%%
print(f"Best parameters: {grid_grad.best_params_}")
print(f"Best score: {grid_grad.best_score_}")

#%% GRADIENT BOOSTING

#beregn og udskriv de forskellige metrikker
print("Performance on test set: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_grad)}")
print(f"Precision: {precision_score(y_test, y_pred_grad)}")
print(f"Recall: {recall_score(y_test, y_pred_grad)}")
print(f"F1 score: {f1_score(y_test, y_pred_grad)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_grad)}")

y_train_pred_grad = best_grad.predict(X_train)

#beregn og udskriv de forskellige metrikker
print("Performance on training set: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_grad)}")
print(f"Precision: {precision_score(y_train, y_train_pred_grad)}")
print(f"Recall: {recall_score(y_train, y_train_pred_grad)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_grad)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_grad)}")


#%% GRADIENT BOOSTING

matrix_grad = confusion_matrix(y_test, y_pred_grad)
matrix_grad = matrix_grad.astype('float') / matrix_grad.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16, 7))
sns.set(font_scale=1.4)
sns.heatmap(matrix_grad, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)

class_names = ['Not dropout', 'Dropout']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation = 25)
plt.yticks(tick_marks2, class_names, rotation = 0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Gradient Boosting Model')
plt.show()

#%% XGBoost

import xgboost as xgb

xg_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

pipeline_grid_xg = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', xg_model)
])

param_grid_xg = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.8, 0.9, 1.0],
    'model__max_depth': [3, 4, 5],
    'model__gamma': [0, 0.1, 0.2] ,
    'model__colsample_bytree': [0.8, 0.9, 1]
}

grid_xg = GridSearchCV(pipeline_grid_xg, param_grid_xg, cv=skf)

grid_xg.fit(X_train, y_train)

best_xg = grid_xg.best_estimator_
print(cross_val_score(best_xg, X, y, cv=skf))

# [0.92551996 0.92662356 0.92268766 0.92324993 0.92409334]

#%%
print(f"Best parameters: {grid_xg.best_params_}")
print(f"Best score: {grid_xg.best_score_}")

#%%
y_pred_xg = best_xg.predict(X_test)

print(classification_report(y_test, y_pred_xg))

#%% EXTREME BOOSTING

#beregn og udskriv de forskellige metrikker
print("Performance on test set: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xg)}")
print(f"Precision: {precision_score(y_test, y_pred_xg)}")
print(f"Recall: {recall_score(y_test, y_pred_xg)}")
print(f"F1 score: {f1_score(y_test, y_pred_xg)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_xg)}")

y_train_pred_xg = best_xg.predict(X_train)

#beregn og udskriv de forskellige metrikker
print("Performance on training set: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_xg)}")
print(f"Precision: {precision_score(y_train, y_train_pred_xg)}")
print(f"Recall: {recall_score(y_train, y_train_pred_xg)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_xg)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_xg)}")


#%% XGBOOST

matrix_xg = confusion_matrix(y_test, y_pred_xg)
matrix_xg = matrix_xg.astype('float') / matrix_xg.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16, 7))
sns.set(font_scale=1.4)
sns.heatmap(matrix_xg, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, linewidths=0.2)

class_names = ['Not dropout', 'Dropout']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation = 25)
plt.yticks(tick_marks2, class_names, rotation = 0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for XGBoost model')
plt.show()

#%% RECALL KURVE FOR ALLE TRE

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

y_scores_ada = best_ada.predict_proba(X_test)[:, 1]
y_scores_grad = best_grad.predict_proba(X_test)[:, 1]
y_scores_xg = best_xg.predict_proba(X_test)[:, 1]

precision_ada, recall_ada, _ = precision_recall_curve(y_test, y_scores_ada)
precision_grad, recall_grad, _ = precision_recall_curve(y_test, y_scores_grad)
precision_xg, recall_xg, _ = precision_recall_curve(y_test, y_scores_xg)

plt.figure(figsize=(10, 7))
plt.plot(recall_ada, precision_ada, label=f'AdaBoost (AP = {average_precision_score(y_test, y_scores_ada):.2f}')
plt.plot(recall_grad, precision_grad, label=f'Gradient Boosting (AP = {average_precision_score(y_test, y_scores_grad):.2f}')
plt.plot(recall_xg, precision_xg, label=f'XGBoost (AP = {average_precision_score(y_test, y_scores_xg):.2f}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

#%% BOOTSTRAP TIL RANDOM FOREST

#antal bootstrap iterationer
n_iterations = 5000
bootstrap_scores = [] #tom liste til at gemme estimater

for i in range(n_iterations):
    #trækker en tilfældig prøve fra data med resample
    X_resample, y_resample = resample(X_test, y_test)
    #træner modellen på resampled date
    y_pred_bs = best_model_rf.predict(X_resample)
    score = accuracy_score(y_resample, y_pred_bs)
    bootstrap_scores.append(score)

#95% konfidensinterval
lower = np.percentile(bootstrap_scores, 2.5)
upper = np.percentile(bootstrap_scores, 97.5)

print(f"95% konfidensinterval for nøjagtighed: ({lower: .2f}, {upper: .2f}")


#%%
nan_count = df.isnull().sum()
print("Antal NaN værdier i df: ")
print(nan_count)
print("\n")


#%% EDA
#%%
print(df.describe())

#%% mål for skævhed of kurtosis
skewness = df.skew()
kurtosis = df.kurt()

print(skewness)
print(kurtosis)

#%% korrelationsmatrix
corr_matrix = df.corr()
corr_matrix_str = corr_matrix.round(2).to_string()
print(corr_matrix_str)

with open('korrelationsmatrix.txt', 'w') as file:
    file.write(corr_matrix_str)

#%%
f, ax = plt.subplots(figsize=(30, 30))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)

plt.show()

#%%


#%%
#print(df['frafald_event'].value_counts()) #13031 ikke droppet ud, 4755 droppet ud

#%%
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# sns.countplot(x='frafald_event', data= df)
# plt.show()

#%%
correlations = df.corr()['frafald_event'].sort_values()
print('most positive correlations: \n', correlations.tail(20))
print('most negative correlations: \n', correlations.head(20))

#%%
# correlation = df['karakter_diff'].corr(df['frafald_event'])
# print(f"korrelation mellem difference i karakter og frafald: {correlation}")

#%%

from sklearn.ensemble import RandomForestClassifier

X = df.drop('frafald_event', axis=1)
y = df['frafald_event']

rf = RandomForestClassifier()
rf.fit(X, y)

importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
importance = importance.sort_values('importance', ascending=False)
print(importance.head(592))

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.utils import resample

y = df['frafald_event']
X = df.drop('frafald_event', axis = 1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42)

#%%
train_data = pd.concat([X_train, y_train], axis=1)

df_majority = train_data[train_data.frafald_event==0]
df_minority = train_data[train_data.frafald_event==1]

df_minority_upsampled = resample(df_minority,
                                 replace= True,
                                 n_samples=len(df_majority),
                                 random_state= 42)

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

X_train_resampled = df_upsampled.drop('frafald_event', axis=1)
y_train_resampled = df_upsampled['frafald_event']

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

#Opret en instans a modellen med L1 regularisering
logistic_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)


#standardiser data
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
X_train_resampled_scaled = scalar.fit_transform(X_train_resampled)
X_test_scaled = scalar.transform(X_test)
#X_test_scaled = scalar.fit_transform(X_test)

#Træn modellen
logistic_model.fit(X_train_resampled_scaled, y_train_resampled)

#forudsigelser på testsættet
y_pred_logistic = logistic_model.predict(X_test_scaled)
y_prob_logistic = logistic_model.predict_proba(X_test_scaled)[:, 1] #sandsynlighed for positive klasse

print("Classification Report: ")
print(classification_report(y_test, y_pred_logistic))
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Precision:", precision_score(y_test, y_pred_logistic))
print("Recall:", recall_score(y_test, y_pred_logistic))
print("F1 score:", f1_score(y_test, y_pred_logistic))
print("ROC AUC:", roc_auc_score(y_test, y_pred_logistic))

#%%
y_pred_train_logistic = logistic_model.predict(X_train_resampled_scaled)

print("classification report on training data: ")
print(classification_report(y_train_resampled, y_pred_train_logistic))

#%%

from sklearn.model_selection import StratifiedKFold

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_logistic = cross_val_score(logistic_model, X_scaled, y, cv=stratified_kfold, scoring='accuracy')

print("Stratified cross-validation scores: ", cv_scores_logistic)
print("Mean CV scores: ", np.mean(cv_scores_logistic))
print("Standard Deviation of CV scores: ", np.std(cv_scores_logistic))

#%% SMOTE til resampling

from imblearn.over_sampling import SMOTE

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled_rf = scalar.transform(X_test)

smote = SMOTE(random_state = 42)
X_resampled_rf, y_resampled_rf = smote.fit_resample(X_train_scaled, y_train)

#%% Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

#definer parametergitteret
param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]
              }

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator=rf, param_grid = param_grid, cv = 3, n_jobs=-1, verbose=2)

grid_search.fit(X_resampled_rf, y_resampled_rf)

print("Bedste parametre: ", grid_search.best_params_)

#%%

#opret en instans af modellen
rf_model = RandomForestClassifier(bootstrap= False, n_estimators=200, max_depth=3, min_samples_split=2, min_samples_leaf=1)

#træn modellen
rf_model.fit(X_resampled_rf, y_resampled_rf)

#forudsigelser på testsættet
y_pred_rf = rf_model.predict(X_test_scaled_rf)

#evaluering

print("Classification Report: ")
print(classification_report(y_test, y_pred_rf))

#%%
print("Accuracy: ", accuracy_score(y_test, y_pred_rf))
print("Precision: ", precision_score(y_test, y_pred_rf))
print("Recall: ", recall_score(y_test, y_pred_rf))
print("F1 score: ", f1_score(y_test, y_pred_rf))
print("ROC AUC: ", roc_auc_score(y_test, y_pred_rf))


#%%
y_pred_train_rf = rf_model.predict(X_resampled_rf)
print("Classification report on training data: ")
print(classification_report(y_resampled_rf, y_pred_train_rf))

#%%

cv_scores_rf = cross_val_score(rf_model, X_resampled_rf, y_resampled_rf, cv=stratified_kfold)
print("Stratified cross-validation scores:", cv_scores_rf)
print("Mean: ", np.mean(cv_scores_rf))
print("Standard Deviation: ", np.std(cv_scores_rf))

#%% GRADIENT BOOSTING

from sklearn.ensemble import GradientBoostingClassifier

param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}

#opret en instans af gradientboosting
gb = GradientBoostingClassifier(random_state=42, n_iter_no_change=3, tol=0.001)

#opret en instans af GridSearchCV
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=3, n_jobs=-1, verbose=2)

#træn modellen
grid_search_gb.fit(X_resampled_rf, y_resampled_rf)

#brug den bedste model til at lave forudsigelser
best_gb_model = grid_search_gb.best_estimator_
y_pred_gb = best_gb_model.predict(X_test_scaled_rf)

#evaluering
print("Classification Report for Gradient Boosting: ")
print(classification_report(y_test, y_pred_gb))
print("Accuracy: ", accuracy_score(y_test, y_pred_gb))
print("Precision: ", precision_score(y_test, y_pred_gb))
print("Recall: ", recall_score(y_test, y_pred_gb))
print("F1 score: ", f1_score(y_test, y_pred_gb))
print("ROC AUC: ", roc_auc_score(y_test, y_pred_gb))

#evaluering på træningsdata
y_pred_train_gb = best_gb_model.predict(X_resampled_rf)
print("Classification report on training data for Gradient Boosting: ")
print(classification_report(y_resampled_rf, y_pred_train_gb))

#%%
from sklearn.ensemble import VotingClassifier

#opret en votingclassifier
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb)], voting='soft')

#træn ensemblemodellen
ensemble_model.fit(X_resampled_rf, y_resampled_rf)

y_pred_ensemble = ensemble_model.predict(X_test_scaled_rf)

#evaluering
print("Classification Report for Ensemble Model: ")
print(classification_report(y_test, y_pred_ensemble))
print("Accuracy: ", accuracy_score(y_test, y_pred_ensemble))
print("Precision: ", precision_score(y_test, y_pred_ensemble))
print("Recall: ", recall_score(y_test, y_pred_ensemble))
print("F1 score: ", f1_score(y_test, y_pred_ensemble))
print("ROC AUC: ", roc_auc_score(y_test, y_pred_ensemble))

#forudsigelser og evaluering
print("Classification Report for Ensemble model: ")
print(classification_report(y_test, y_pred_ensemble))

#evaluering på træningsdata
y_pred_train_ensemble = ensemble_model.predict(X_resampled_rf)
print("Classification report on training data for Ensemble model: ")
print(classification_report(y_resampled_rf, y_pred_train_ensemble))

#%%
# def cramers_v(x, y):
#     confusion_matrix = pd.crosstab(x, y)
#     chi2 = ss.chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     phi2 = chi2 / n
#     r, k = confusion_matrix.shape
#     phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
#     rcorr = r - ((r-1)**2)/(n-1)
#     kcorr = k - ((k-1)**2) /(n-1)
#     return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
#
# y = df['frafald_event']
# potential_xs = df.drop(['frafald_event' ], axis=1)
#
# correlations = {}
#
# for col in potential_xs.columns:
#     if potential_xs[col].dtype == 'O' or potential_xs[col].dtype == 'datetime64[ns]':  # Kategorisk eller datetime
#         if y.dtype == 'O':  # Kategorisk
#             corr = cramers_v(y, potential_xs[col])
#         else:  # Numerisk
#             continue  # Spring denne kombination over, da den ikke er gyldig
#     else:  # Numerisk
#         # Fjern eller erstat manglende værdier
#         y_clean = y.dropna()
#         xs_clean = potential_xs[col].loc[y_clean.index].dropna()
#         y_clean = y_clean.loc[xs_clean.index]
#
#         if y_clean.dtype == 'O':  # Kategorisk
#             corr, _ = ss.pointbiserialr(xs_clean, y_clean)
#         else:  # Numerisk
#             corr, _ = ss.pearsonr(y_clean, xs_clean)
#     correlations[col] = corr
#
# correlations_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
# print(correlations_df)

#%% normalisering


