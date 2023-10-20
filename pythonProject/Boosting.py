import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import accuracy_score
import time


df = pd.read_csv('df_pre_model_final_sletningafvif.csv', index_col='studerende_id')

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
    'model__n_estimators': [200, 300, 400],
    'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
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

#%% Threshold
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prop_ada)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.legend(loc="best")
plt.title("Precision-Recall vs Threshold")
plt.show()


#%% ADABOOST

def plot_confusion_matrix(y_true, y_pred, title):
    matrix = confusion_matrix(y_true, y_pred)
    percent_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    labels = [f"{value}\n({percent:.2%})" for value, percent in zip(matrix.flatten(), percent_matrix.flatten())]
    labels = np.array(labels).reshape(2,2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=labels, fmt="", cmap="Greens", cbar=False)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(ticks=[0.5, 1.5], labels=["Not Dropout", "Dropout"])
    plt.yticks(ticks=[0.5, 1.5], labels=["Not Dropout", "Dropout"], va="center")
    plt.show()

#lasso confusion matrix
plot_confusion_matrix(y_test, y_pred_ada, "Confusion Matrix for ADA boosting")


#%% GRADIENT BOOSTING

from sklearn.ensemble import GradientBoostingClassifier

gradient = GradientBoostingClassifier()

pipeline_grid_grad = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', gradient)
])

param_grid_grad = {
    'model__n_estimators': [200, 300, 400],
    'model__learning_rate': [0.001, 0.01, 0.1],
    'model__subsample': [0.8, 0.9, 1.0],
    'model__max_depth': [3, 4, 5],
    'model__min_samples_split': [2, 3, 4],
    'model__min_samples_leaf': [1, 2, 3]
}

grid_grad = GridSearchCV(pipeline_grid_grad, param_grid_grad, cv=skf)


start_time_grad = time.time()
grid_grad.fit(X_train, y_train)
end_time_grad = time.time()

training_time_grad = end_time_grad - start_time_grad

best_grad = grid_grad.best_estimator_
print(cross_val_score(best_grad, X, y, cv=skf))

y_pred_grad = best_grad.predict(X_test)
y_prop_grad = best_grad.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_grad))
print(f"Best parameters: {grid_grad.best_params_}")
print(f"Best score: {grid_grad.best_score_}")
print(f"Training time for Gradient Boosting: {training_time_grad: .2f} seconds")

#%% GRADIENT BOOSTING

#beregn og udskriv de forskellige metrikker
print("Performance on test set: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_grad)}")
print(f"Precision: {precision_score(y_test, y_pred_grad)}")
print(f"Recall: {recall_score(y_test, y_pred_grad)}")
print(f"F1 score: {f1_score(y_test, y_pred_grad)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_grad)}")
print(f"AUC PR: {average_precision_score(y_test, y_prop_grad)}")


y_train_pred_grad = best_grad.predict(X_train)
y_train_prop_grad = best_grad.predict_proba(X_train)[:, 1]


#beregn og udskriv de forskellige metrikker
print("Performance on training set: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_grad)}")
print(f"Precision: {precision_score(y_train, y_train_pred_grad)}")
print(f"Recall: {recall_score(y_train, y_train_pred_grad)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_grad)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_grad)}")
print(f"AUC PR: {average_precision_score(y_train, y_train_prop_grad)}")

#%%
def plot_confusion_matrix(y_true, y_pred, title):
    matrix = confusion_matrix(y_true, y_pred)
    percent_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    labels = [f"{value}\n({percent:.2%})" for value, percent in zip(matrix.flatten(), percent_matrix.flatten())]
    labels = np.array(labels).reshape(2,2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=labels, fmt="", cmap="Greens", cbar=False)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(ticks=[0.5, 1.5], labels=["Not Dropout", "Dropout"])
    plt.yticks(ticks=[0.5, 1.5], labels=["Not Dropout", "Dropout"], va="center")
    plt.show()

#lasso confusion matrix
plot_confusion_matrix(y_test, y_pred_grad, "Confusion Matrix for Gradient boosting")

#%% XGBoost

import xgboost as xgb

xg_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

pipeline_grid_xg = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', xg_model)
])

param_grid_xg = {
    'model__n_estimators': [200, 300],
    'model__learning_rate': [0.001, 0.01, 0.1],
    'model__subsample': [0.8, 0.9, 1.0],
    'model__max_depth': [3, 4, 5],
    'model__gamma': [0, 0.1, 0.2] ,
    'model__colsample_bytree': [0.8, 0.9, 1]
}

grid_xg = GridSearchCV(pipeline_grid_xg, param_grid_xg, cv=skf)

start_time_xg = time.time()
grid_xg.fit(X_train, y_train)
end_time_xg = time.time()

training_time_xg = end_time_xg - start_time_xg


best_xg = grid_xg.best_estimator_
print(cross_val_score(best_xg, X, y, cv=skf))

print(f"Best parameters: {grid_xg.best_params_}")
print(f"Best score: {grid_xg.best_score_}")
print(f"Training time for XGboost: {training_time_xg: .2f} seconds")


y_pred_xg = best_xg.predict(X_test)
print(classification_report(y_test, y_pred_xg))



#%% EXTREME BOOSTING

y_prop_xg = best_xg.predict_proba(X_test)[:, 1]
threshold = 0.66588
y_pred_xg_thresholded = (y_prop_xg >= threshold).astype(int)


#beregn og udskriv de forskellige metrikker
print("Performance on test set: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xg)}")
print(f"Precision: {precision_score(y_test, y_pred_xg)}")
print(f"Recall: {recall_score(y_test, y_pred_xg)}")
print(f"F1 score: {f1_score(y_test, y_pred_xg)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_xg)}")
print(f"AUC PR: {average_precision_score(y_test, y_prop_xg)}")


y_train_pred_xg = best_xg.predict(X_train)
y_train_prop_xg = best_xg.predict_proba(X_train)[:, 1]

#beregn og udskriv de forskellige metrikker
print("Performance on training set: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_xg)}")
print(f"Precision: {precision_score(y_train, y_train_pred_xg)}")
print(f"Recall: {recall_score(y_train, y_train_pred_xg)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_xg)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_xg)}")
print(f"AUC PR: {average_precision_score(y_train, y_train_prop_xg)}")



#%% XGBOOST

def plot_confusion_matrix(y_true, y_pred, title):
    matrix = confusion_matrix(y_true, y_pred)
    percent_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    labels = [f"{value}\n({percent:.2%})" for value, percent in zip(matrix.flatten(), percent_matrix.flatten())]
    labels = np.array(labels).reshape(2,2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=labels, fmt="", cmap="Greens", cbar=False)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(ticks=[0.5, 1.5], labels=["Not Dropout", "Dropout"])
    plt.yticks(ticks=[0.5, 1.5], labels=["Not Dropout", "Dropout"], va="center")
    plt.show()

#lasso confusion matrix
plot_confusion_matrix(y_test, y_pred_xg, "Confusion Matrix for XGboosting")


#%% RECALL KURVE FOR ALLE TRE

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For AdaBoost
y_proba_ada = best_ada.predict_proba(X_test)[:, 1]
precision_ada, recall_ada, _ = precision_recall_curve(y_test, y_proba_ada)
average_precision_ada = average_precision_score(y_test, y_proba_ada)

# For Gradient Boosting
y_proba_grad = best_grad.predict_proba(X_test)[:, 1]
precision_grad, recall_grad, _ = precision_recall_curve(y_test, y_proba_grad)
average_precision_grad = average_precision_score(y_test, y_proba_grad)

# For XGBoost
y_proba_xg = best_xg.predict_proba(X_test)[:, 1]
precision_xg, recall_xg, _ = precision_recall_curve(y_test, y_proba_xg)
average_precision_xg = average_precision_score(y_test, y_proba_xg)

plt.figure(figsize=(8, 6))
plt.step(recall_ada, precision_ada, where='post', label=f'AdaBoost (AP = {average_precision_ada:.2f})')
plt.step(recall_grad, precision_grad, where='post', label=f'Gradient Boosting (AP = {average_precision_grad:.2f})')
plt.step(recall_xg, precision_xg, where='post', label=f'XGBoost (AP = {average_precision_xg:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curves for boosting models')
plt.legend()
plt.show()


#%%

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prop_xg)

desired_precision = 0.90
closest_precision_index = (np.abs(precisions - desired_precision)).argmin()
optimal_threshold_for_desired_precision = thresholds[closest_precision_index]

print(f"Optimal threshold for desired precision {desired_precision}: {optimal_threshold_for_desired_precision}")


#%% Learning curve for gradient boosting

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_grad, X, y, cv=skf, n_jobs= -1, train_sizes=np.linspace(.1, 1.0, 5))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Learning Curve for Gradient Boosting")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.grid()
plt.tight_layout()
plt.show()


#%% Learning curve for adaboost

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_ada, X, y, cv=skf, n_jobs= -1, train_sizes=np.linspace(.1, 1.0, 5))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Learning Curve for ADA Boosting")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.grid()
plt.tight_layout()
plt.show()


#%% Learning curve

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_xg, X, y, cv=skf, n_jobs= -1, train_sizes=np.linspace(.1, 1.0, 5))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Learning Curve for XGBoost")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.grid()
plt.tight_layout()
plt.show()

#%% SHAP

import shap
from IPython.display import display

shap.initjs = lambda : None

#initialiser SHAPs explainer med gradient model
explainer = shap.TreeExplainer(best_grad.named_steps['model'])

#beregn shap værdier for test data
shap_values_gb = explainer.shap_values(X_train)


shap.summary_plot(shap_values_gb, X_train)

#%% bar plot

shap.summary_plot(shap_values_gb, X_test, plot_type="bar")

#%%
from joblib import dump
from joblib import load

dump(best_ada, 'adaptive_boosting.joblib')
dump(best_grad, 'gradient_boosting.joblib')
dump(best_xg, 'XGboosting.joblib')


#%%

plt.figure(figsize=(8, 6))
plt.step(recall_lasso, precision_lasso, where='post', label=f'Logistic regression (AP = {average_precision_lasso:.2f})')
plt.step(recall_specified, precision_specified, where='post', label=f'Random forest (AP = {average_precision_specified:.2f})')
plt.step(recall_grad, precision_grad, where='post', label=f'Boosting (AP = {average_precision_grad:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curves for Logistic regression, Random Forest and Boosting')
plt.legend()
plt.show()

#%%
from sklearn.metrics import roc_curve, auc

#funktion ti lat plotte ROC kurve
def plot_roc_curve(y_true, y_pred_prob, label):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label}(area={roc_auc:.2f})')

plt.figure(figsize=(10, 8))

#logistisk regression
plot_roc_curve(y_test, y_pred_lasso, "Logistic Regression")

#random
plot_roc_curve(y_test, y_pred_rf_random, "Random Forest")

#boosting
plot_roc_curve(y_test, y_pred_grad, "Boosting")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curves')
plt.legend(loc="lower right")
plt.show()

#%% Få model til at printe studerende

predicted_classes = best_grad.named_steps['model'].predict(X_test)
students_predicted_to_dropout = X_test[predicted_classes == 1]
first_10_ids = students_predicted_to_dropout.head(10).index

for i, student_id in enumerate(first_10_ids, start=1):
    print(f"{i}: {student_id}")