import  pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sciopt import BayesSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

column_names = ['col1', 'col2', 'col3', 'col4', 'col5', 'labels']
data_import = pd.read_csv("a2.csv.txt", names=column_names)

y = data_import["labels"] > 0
X = data_import.iloc[:, :-1]

# Evaluating model/CONF MATRIX
def model_eval(y_test, predictions):
  conf_matrix = confusion_matrix(y_test, predictions)
  accuracy = accuracy_score(y_test, predictions)
  precision = precision_score(y_test, predictions)
  recall = recall_score(y_test, predictions)
  f1 = f1_score(y_test, predictions)

  return conf_matrix, accuracy, precision, recall, f1

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaled_X_trained = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

#SupportVectormachineClassifier
svc_param_grid = [
    {
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    }
]

svc = SVC()
bayes_op = BayesSearchCV(svc, svc_param_grid, cv=3, scoring="accuracy", return_train_score=True)
bayes_op.fit(scaled_X_trained, y_train)
svc_clf = bayes_op.best_estimator_
svc_pred = (svc_clf.predict(scaled_X_test)
model_eval(y_test, svc_pred)

#RandomForestClasifier
param_grid = [
    {"n_estimators": [3, 10, 30, 50, 100, 150],
     "max_features": [2, 3, 4, 5],
     "bootstrap": [True],
     }
]
rfc = RandomForestClassifier(max_features=3, n_estimators=150)
grid_search = GridSearchCV(rfc, param_grid, cv=5, scoring="accuracy", return_train_score=True)
grid_search.fit(scaled_X_trained, y_train)
rfc_clf = grid_search.best_estimator_
rfc_pred = rfc_clf.predict(scaled_X_test)
model_eval(y_test, rfc_pred)

# GradientBoostingClassifier
gbc_param_grid = [
    {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
     }
]
gbc = GradientBoostingClassifier(max_depth=7, n_estimators=150, learning_rate=0.01)
gbc_grid_search = GridSearchCV(gbc, gbc_param_grid, cv=5, scoring="accuracy", return_train_score=True)
gbc_grid_search.fit(scaled_X_trained, y_train)
gbc_clf = gbc_grid_search.best_estimator_
gbc_pred = gbc_clf.predict(scaled_X_test)
model_eval(y_test, gbc_pred)
