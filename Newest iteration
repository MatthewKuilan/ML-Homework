import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

column_names = ['col1', 'col2', 'col3', 'col4', 'col5', 'labels']
test_column_names = ['col1', 'col2', 'col3', 'col4', 'col5']
data_import = pd.read_csv('a2.csv.txt', names=column_names)
new_data_import = pd.read_csv('a1_test.csv.txt', names=test_column_names)

y = data_import["labels"] > 0
X = data_import.iloc[:, :-1]
X_nd = new_data_import.iloc[:, :]

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.0909090909, random_state=42)
scaled_X_trained = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
scaled_X_nd = scaler.transform(X_nd)

# Evaluating model/CONF MATRIX
def model_eval(predictions):
    conf_matrix = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return conf_matrix, accuracy, precision, recall, f1


# Support Vector machine Classifier
svc_clf = SVC()
svc_clf = svc_clf.fit(scaled_X_trained, y_train)
svc_pred = svc_clf.predict(scaled_X_test)
svc_metrics = model_eval(svc_pred)
print(model_eval(svc_pred))

# RandomForestClassifier
param_grid = [
    {"n_estimators": [150, 350],
     "max_features": [2, 3, 4, 5, 7 , 9],
     "bootstrap": [True],
     }
]
rfc = RandomForestClassifier()
grid_search = GridSearchCV(rfc, param_grid, cv=5, scoring="accuracy", return_train_score=True)
grid_search.fit(scaled_X_trained, y_train)
rfc_clf = grid_search.best_estimator_
rfc_pred = rfc_clf.predict(scaled_X_test)
rfc_metrics = model_eval(rfc_pred)
print(model_eval(rfc_pred))

# GradientBoostingClassifier
gbc_param_grid = [
    {'learning_rate': [0.01, 0.05, 0.1],
     'n_estimators': [50, 100, 150],
     'max_depth': [3, 5, 7],
     }
]
gbc = GradientBoostingClassifier()
gbc_grid_search = GridSearchCV(gbc, gbc_param_grid, cv=5, scoring="accuracy", return_train_score=True)
gbc_grid_search.fit(scaled_X_trained, y_train)
gbc_clf = gbc_grid_search.best_estimator_
gbc_pred = gbc_clf.predict(scaled_X_test)
gbc_metrics = model_eval(gbc_pred)
print(model_eval(gbc_pred))

nd_pred = rfc_clf.predict(scaled_X_nd)
