
rf_model = RandomForestClassifier(random_state=42)

#pipeline
pipeline_rf = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', rf_model)
])


param_distributions = {
    'model__n_estimators': [500, 600],
    'model__max_depth': [5, 7, 9],
    'model__min_samples_split': [2, 4],
    'model__min_samples_leaf': [5, 7],
    'model__max_features': ['sqrt', 'log2', None],
    'model__bootstrap': [True, False],
    'model__class_weight': ['balanced'],
    'model__ccp_alpha': [0.0, 0.1, 0.2]
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
start_time_randomrf = time.time()
random_search.fit(X_train, y_train)
end_time_randomrf = time.time()
training_time_randomrf = end_time_randomrf - start_time_randomrf


#resultater
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_}")
print(f"Training time for Random Search Random Forest: {training_time_randomrf: .2f} seconds")


#%% MODEL DER SKAL BRUGES:

from sklearn.metrics import average_precision_score

#feedback loop
best_rf_random = random_search.best_estimator_
y_pred_rf_random = best_rf_random.predict(X_test)
y_prop_rf_random = best_ada.predict_proba(X_test)[:, 1]

print("Performance after RandomSearchCV")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_random)}")
print(f"Precision: {precision_score(y_test, y_pred_rf_random)}")
print(f"Recall: {recall_score(y_test, y_pred_rf_random)}")
print(f"F1 score: {f1_score(y_test, y_pred_rf_random)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_rf_random)}")
print(f"AUC PR: {average_precision_score(y_test, y_prop_rf_random)}")

feedback_threshold = 0.88  # Dette er en vilkårlig tærskelværdi. Juster efter behov.
if accuracy_score(y_test, y_pred_rf_random) > feedback_threshold:
    print("RandomSearch performance is satisfactory. No need for GridSearchCV!")
    best_model_rf = best_rf_random
else:
    print("Proceeding with GridSearchCV...")

y_train_pred_rf_random = best_rf_random.predict(X_train)
y_train_prop_rf_random = best_ada.predict_proba(X_train)[:, 1]

#beregn og udskriv de forskellige metrikker
print("Performance on training set RandomSearch: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_rf_random)}")
print(f"Precision: {precision_score(y_train, y_train_pred_rf_random)}")
print(f"Recall: {recall_score(y_train, y_train_pred_rf_random)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_rf_random)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_rf_random)}")
print(f"AUC PR: {average_precision_score(y_train, y_train_prop_rf_random)}")


#%% MODEL DER OVERFITTER

param_grid_rf = {
    'model__n_estimators': [550],
    'model__max_depth': [15],
    'model__min_samples_split': [2],
    'model__min_samples_leaf': [5],
    'model__max_features': ['sqrt'],
    'model__bootstrap': [False],
    'model__class_weight': ['balanced'],
    'model__ccp_alpha': [0.0]
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
start_time_rf = time.time()
grid_search_rf.fit(X_train, y_train)
end_time_rf = time.time()

training_time_rf = end_time_rf - start_time_rf


#resultater
print(f"Best parameters: {grid_search_rf.best_params_}")
print(f"Best score: {grid_search_rf.best_score_}")
print(f"Training time for overfitting Random Forest: {training_time_rf: .2f} seconds")

#%% Confusion matrix + classification report (MODEL DER OVERFITTER)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

best_model_rf = grid_search_rf.best_estimator_
y_pred_rf = best_model_rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))

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
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")


#%% Resultatmetrikker

y_prop_grid_rf = best_model_rf.predict_proba(X_test)[:, 1]


# Resultatmetrikker på test sæt
print("Performance on test set: ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf)}")
print(f"Recall: {recall_score(y_test, y_pred_rf)}")
print(f"F1 score: {f1_score(y_test, y_pred_rf)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_rf)}")
print(f"AUC PR: {average_precision_score(y_test, y_prop_grid_rf)}")


#Restultatmetrikker på træningssæt
y_train_pred_rf = best_model_rf.predict(X_train)
y_train_prop_grid_rf = best_model_rf.predict_proba(X_train)[:, 1]


#beregn og udskriv de forskellige metrikker
print("Performance on training set: ")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred_rf)}")
print(f"Precision: {precision_score(y_train, y_train_pred_rf)}")
print(f"Recall: {recall_score(y_train, y_train_pred_rf)}")
print(f"F1 score: {f1_score(y_train, y_train_pred_rf)}")
print(f"ROC AUC: {roc_auc_score(y_train, y_train_pred_rf)}")
print(f"AUC PR: {average_precision_score(y_train, y_train_prop_grid_rf)}")

#%%

from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt


# Definer en funktion til at tegne learning curves
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


# Plot learning curve for the best Random Forest model from GridSearchCV
plot_learning_curve(best_rf_random, "Learning Curve for Random Forest", X_train, y_train, cv=skf)
plt.show()

#%%

# For overfitting
precision_overfit, recall_overfit, _ = precision_recall_curve(y_test, y_prop_grid_rf)
average_precision_overfit = average_precision_score(y_test, y_prop_grid_rf)

# For specified
precision_specified, recall_specified, _ = precision_recall_curve(y_test, y_prop_rf_random)
average_precision_specified = average_precision_score(y_test, y_prop_rf_random)


plt.figure(figsize=(8, 6))
plt.step(recall_overfit, precision_overfit, where='post', label=f'Random forest, overfit (AP = {average_precision_overfit:.2f})')
plt.step(recall_specified, precision_specified, where='post', label=f'Random forest, specified (AP = {average_precision_specified:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curves for Random forest models')
plt.legend()
plt.show()



#%% Feature importance Random Forest
#SHAP


shap.initjs = lambda : None

rf_model = best_rf_random.named_steps['model']

#initialiser SHAPs explainer med gradient model
explainer = shap.TreeExplainer(best_rf_random.named_steps['model'])

#beregn shap værdier for test data
shap_values = explainer.shap_values(X_train)

#visualiser den første prve

shap.summary_plot(shap_values[1], X_train)

#%%

from sklearn.tree import plot_tree

trees = best_rf_random.named_steps['model']

feature_names_list = X_train.columns.tolist()

#vælg et træ
chosen_tree = trees.estimators_[0]

#visualer
plt.figure(figsize=(40, 20))
plot_tree(chosen_tree, filled=True, feature_names=feature_names_list, class_names=['Not Dropout', 'Dropout'], rounded=True, max_depth=3)
plt.show()


#%%
dump(best_model_rf, 'gridsearchmodelrf.joblib')
dump(best_rf_random, 'bestrandomrf.joblib')
