from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def fit_decision_tree_classifier(x_train, y_train, cv=5,scoring='f1_weighted', random_state=None):

    param = {'criterion': ['gini'],
             'max_depth': [None, 5, 10, 15, 20],
             'min_samples_leaf': [1, 2, 3, 5, 10],
             'min_impurity_decrease': [0, 0.1, 0.2, 0.5]}
    grid = GridSearchCV(DecisionTreeClassifier(random_state=random_state),
                        param_grid=param, cv=cv,
                        scoring=scoring)
    grid.fit(x_train, y_train)
    print("best_estimator_", grid.best_estimator_)
    print("best_score_", grid.best_score_)
    return grid.best_estimator_

def Linear_SVR(C=1.0, gamma=0.1, epsilon=1):
    return Pipeline([
        ("std_scaler", MinMaxScaler()),
        ("model", SVR(kernel="linear", C=C, gamma=gamma, epsilon=epsilon))
    ])

def Linear_SVC(C=1.0, gamma=0.1, epsilon=1):
    return Pipeline([
        ("std_scaler", MinMaxScaler()),
        ("model", SVC(kernel="linear", C=C, gamma=gamma))
    ])
def RBF_SVC(C=1.0, gamma=1, epsilon=1):
    return Pipeline([
        ("std_scaler", MinMaxScaler()),
        ("model", SVC(kernel="rbf", C=C, gamma=gamma))
    ])

def RBF_SVR(C=1.0, gamma=1, epsilon=1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon))
    ])
