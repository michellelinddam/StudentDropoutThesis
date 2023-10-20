

#%% LOGISTISK REGRESSION MED STRATIFIED K-FOLD CV

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#opret en instans af logistisk regression
logistic_model = LogisticRegression()

#opret en pipeline, der først standardiserer data og derefter træner
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', logistic_model)
])

#%% ELASTICNET
#definer parametrene
elasticnet_params = {
    'model__penalty': ['elasticnet'],
    'model__solver': ['saga'],
    'model__l1_ratio': np.linspace(0, 1), #0 er ren ridge, 1 er ren lasso
    'model__C': np.logspace(-4, 4, 10),  #logaritmisk skala
    'model__max_iter': [10000]
}

grid_search_elasticnet = GridSearchCV(
    estimator=pipeline,
    param_grid=elasticnet_params,
    cv=skf,
    scoring='f1',
    n_jobs=-1,
    verbose=2)

start_time_elasticnet = time.time()
#træn modellen
grid_search_elasticnet.fit(X_train, y_train)
end_time_elasticnet = time.time()

#udregn tiden
training_time_elasticnet = end_time_elasticnet - start_time_elasticnet

#resultater
print(f"Best parameters for Elastic Net: {grid_search_elasticnet.best_params_}")
print(f"Best score for Elastic Net: {grid_search_elasticnet.best_score_}")
print(f"Training time for Elastic Net: {training_time_elasticnet: .2f} seconds")


lasso_params = {
    'model__penalty': ['l1'],
    'model__C': np.logspace(-4, 4, 10),
    'model__solver': ['liblinear', 'saga'],
    'model__max_iter': [10000]
}

grid_search_lasso = GridSearchCV(
    estimator=pipeline,
    param_grid=lasso_params,
    cv=skf,
    scoring='f1',
    n_jobs=-1,
    verbose=2)

start_time_lasso = time.time()
grid_search_lasso.fit(X_train, y_train)
end_time_lasso = time.time()

#udregn tiden
training_time_lasso = end_time_lasso - start_time_lasso

print("Best parameters for Lasso:", grid_search_lasso.best_params_)
print("Best score for Lasso:", grid_search_lasso.best_score_)
print(f"Training time for Lasso: {training_time_lasso: .2f} seconds")


ridge_params = {
    'model__penalty': ['l2'],
    'model__C': np.logspace(-4, 4, 10),
    'model__solver': ['liblinear', 'saga'],
    'model__max_iter': [10000]
}

grid_search_ridge = GridSearchCV(
    estimator=pipeline,
    param_grid=ridge_params,
    cv=skf,
    scoring='f1',
    n_jobs=-1,
    verbose=2)

start_time_ridge = time.time()
grid_search_ridge.fit(X_train, y_train)
end_time_ridge = time.time()

#udregn tiden
training_time_ridge = end_time_ridge - start_time_ridge

print("Best parameters for Ridge:", grid_search_ridge.best_params_)
print("Best score for Ridge:", grid_search_ridge.best_score_)
print(f"Training time for Ridge: {training_time_ridge: .2f} seconds")

#%%
from sklearn.metrics import average_precision_score

#bruger den bedste estimator for ElasticNet
best_model_elasticnet = grid_search_elasticnet.best_estimator_
y_pred_elasticnet = best_model_elasticnet.predict(X_test)
y_prop_elasticnet = best_model_elasticnet.predict_proba(X_test)[:, 1]

#beregn og udskriv de forskellige metrikker
print("Performance on test set for Elastic Net: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_elasticnet)}")
print(f"Precision: {precision_score(y_test, y_pred_elasticnet)}")
print(f"Recall: {recall_score(y_test, y_pred_elasticnet)}")
print(f"F1 score: {f1_score(y_test, y_pred_elasticnet)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_elasticnet)}")
print(f"AUC PR: {average_precision_score(y_test, y_prop_elasticnet)}")


y_train_pred_elasticnet = best_model_elasticnet.predict(X_train)
y_train_prop_elasticnet = best_model_elasticnet.predict_proba(X_train)[:, 1]

#beregn og udskriv de forskellige metrikker
print("Performance on training set for Elastic Net: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_elasticnet)}")
print(f"Precision: {precision_score(y_train, y_train_pred_elasticnet)}")
print(f"Recall: {recall_score(y_train, y_train_pred_elasticnet)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_elasticnet)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_elasticnet)}")
print(f"AUC PR: {average_precision_score(y_train, y_train_prop_elasticnet)}")


#%%
#bruger den bedste estimator for Lasso
best_model_lasso = grid_search_lasso.best_estimator_
y_pred_lasso = best_model_lasso.predict(X_test)
y_prop_lasso = best_model_lasso.predict_proba(X_test)[:, 1]

#beregn og udskriv de forskellige metrikker
print("Performance on test set for Lasso: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lasso)}")
print(f"Precision: {precision_score(y_test, y_pred_lasso)}")
print(f"Recall: {recall_score(y_test, y_pred_lasso)}")
print(f"F1 score: {f1_score(y_test, y_pred_lasso)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_lasso)}")
print(f"AUC PR: {average_precision_score(y_test, y_prop_lasso)}")


y_train_pred_lasso = best_model_lasso.predict(X_train)
y_train_prop_lasso = best_model_lasso.predict_proba(X_train)[:, 1]

#beregn og udskriv de forskellige metrikker
print("Performance on training set for Lasso: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_lasso)}")
print(f"Precision: {precision_score(y_train, y_train_pred_lasso)}")
print(f"Recall: {recall_score(y_train, y_train_pred_lasso)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_lasso)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_lasso)}")
print(f"AUC PR: {average_precision_score(y_train, y_train_prop_lasso)}")


#%%
#bruger den bedste estimator for Ridge
best_model_ridge = grid_search_ridge.best_estimator_
y_pred_ridge = best_model_ridge.predict(X_test)
y_prop_ridge = best_model_ridge.predict_proba(X_test)[:, 1]


#beregn og udskriv de forskellige metrikker
print("Performance on test set for Ridge Regression: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ridge)}")
print(f"Precision: {precision_score(y_test, y_pred_ridge)}")
print(f"Recall: {recall_score(y_test, y_pred_ridge)}")
print(f"F1 score: {f1_score(y_test, y_pred_ridge)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_ridge)}")
print(f"AUC PR: {average_precision_score(y_test, y_prop_ridge)}")


y_train_pred_ridge = best_model_ridge.predict(X_train)
y_train_prop_ridge = best_model_ridge.predict_proba(X_train)[:, 1]


#beregn og udskriv de forskellige metrikker
print("Performance on training set: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_ridge)}")
print(f"Precision: {precision_score(y_train, y_train_pred_ridge)}")
print(f"Recall: {recall_score(y_train, y_train_pred_ridge)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_ridge)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_ridge)}")
print(f"AUC PR: {average_precision_score(y_train, y_train_prop_ridge)}")

#%% Confusion matrix

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

#confusion matrix
plot_confusion_matrix(y_test, y_pred_ridge, "Ridge Confusion Matrix")

#%% Learning curve

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # Brug learning_curve funktionen fra sklearn
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="roc_auc")

    # Beregn middelværdi og standardafvigelse for scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt


# Plot learning curve for bedste Random Forest model fra GridSearchCV
plot_learning_curve(best_model_ridge, "Learning Curve for Ridge", X_train, y_train, cv=skf)
plt.show()

#%%

#elasticnet
precision_elastic, recall_elastic, _ = precision_recall_curve(y_test, y_prop_elasticnet)
average_precision_elastic = average_precision_score(y_test, y_prop_elasticnet)

#lasso
precision_lasso, recall_lasso, _ = precision_recall_curve(y_test, y_prop_lasso)
average_precision_lasso = average_precision_score(y_test, y_prop_lasso)

#ridge
precision_ridge, recall_ridge, _ = precision_recall_curve(y_test, y_prop_ridge)
average_precision_ridge = average_precision_score(y_test, y_prop_ridge)


plt.figure(figsize=(8, 6))
plt.step(recall_elastic, precision_elastic, where='post', label=f'Logistic regression, ElasticNet (AP = {average_precision_elastic:.2f})')
plt.step(recall_lasso, precision_lasso, where='post', label=f'Logistic regression, Lasso (AP = {average_precision_lasso:.2f})')
plt.step(recall_ridge, precision_ridge, where='post', label=f'Logistic regression, Ridge (AP = {average_precision_ridge:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curves for Logistic regression')
plt.legend()
plt.show()

#%% Koefficienternes størrelse

coef_lasso = best_model_lasso.named_steps['model'].coef_[0]
coef_ridge = best_model_ridge.named_steps['model'].coef_[0]
coef_elastic = best_model_elasticnet.named_steps['model'].coef_[0]

feature_names = X.columns

plt.figure(figsize=(20, 5))

plt.subplot(131)
plt.stem(np.arange(len(coef_lasso)), coef_lasso, linefmt='b-', markerfmt='bo', basefmt='b-')
plt.title("Lasso")
plt.xticks(rotation=45, fontsize=8)

plt.subplot(132)
plt.stem(np.arange(len(coef_ridge)), coef_ridge, linefmt='r-', markerfmt='ro', basefmt='r-')
plt.title("Ridge")
plt.xticks(rotation=45, fontsize=8)


plt.subplot(133)
plt.stem(np.arange(len(coef_elastic)), coef_elastic, linefmt='g-', markerfmt='go', basefmt='g-')
plt.title("Elastic Net")
plt.xticks(rotation=45, fontsize=8)

plt.tight_layout()
plt.show()

#%% Feature importance Random Forest
#SHAP


shap.initjs = lambda : None

logistisk_model = best_model_lasso.named_steps['model']
feature_names_lasso = X_train.columns.tolist()

#initialiser SHAPs explainer med gradient model
explainer_lasso = shap.LinearExplainer(logistisk_model, X_test)

#beregn shap værdier for test data
shap_values_lasso = explainer_lasso.shap_values(X_test)

#visualiser den første prve

shap.summary_plot(shap_values_lasso, X_test, feature_names=feature_names_lasso)
