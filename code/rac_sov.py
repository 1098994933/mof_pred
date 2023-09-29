"""
solvent_RAC ml
"""

from catboost import CatBoostClassifier
from config import config
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
# from eval_method import cal_metric
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np
import pickle
import rdkit
from rdkit.Chem import MolFromSmiles, RDKFingerprint
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import os
import scipy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from util.alg import Linear_SVC, RBF_SVC
import random

CV_FOLD = 5


def main():
    random.seed(0)
    np.random.seed(0)
    dataset = pd.read_csv("../data/RAC_train.csv")
    targets = ['param1', 'param2', 'param3', 'param4', 'param5']
    features = list(dataset.columns[1:-12])
    X = dataset[features]
    alg_dict = {
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "LinearRegression": LinearRegression(),
        # 'LinearSVR': SVR(kernel='linear'),
        "GradientBoosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "ExtraTrees": ExtraTreesRegressor(),
        "RandomForest": RandomForestRegressor(),
        "KNeighbors": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(),
        # 'RbfSVR': SVR(kernel='rbf')
    }

    y_trans_dict = {
        # "param1": lambda x: np.log10(x + 0.15),
        "param3": lambda x: np.log10(x + 0.15),
        "param4": np.exp,
    }
    y_inv_trans_dict = {
        # "param1": lambda x: 10 ** x - 0.15,
        "param3": lambda x: 10 ** x - 0.15,
        "param4": np.log,
    }
    # region modeling
    for Y_col in targets:
        print(Y_col)
        Y = dataset[Y_col]
        if Y_col in y_trans_dict.keys():
            Y = y_trans_dict[Y_col](Y)
        best_model = None
        best_score = 10 ** 10
        for alg_name in alg_dict.keys():
            model = alg_dict[alg_name]
            # model.fit(X_train, Y_train)
            # y_predict = model.predict(X_test)
            # score = -r2_score(Y_test, y_predict)
            score = - np.mean(cross_val_score(model, X, Y, cv=CV_FOLD))
            print(f"{alg_name} {score}")
            if score < best_score:
                best_model = model
                best_score = score
        # save the best model
        print(f"best score {best_score} best model {best_model}")
        model_final = best_model
        model_final.fit(X, Y)
        with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
            pickle.dump(model_final, file)
        model_final = RandomForestRegressor()
        model_final.fit(X, Y)
        # evaluation
        score = np.median(cross_val_score(model_final, X, Y, cv=CV_FOLD))
        print(f"Y_col: {Y_col} , score:{score}")
        # save model
        with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
            pickle.dump(model_final, file)
        # endregion
    # predict region
    dataset = pd.read_csv("../data/RAC_test.csv")
    predict_df = pd.DataFrame()
    predict_df['mof'] = list(range(1, len(dataset) + 1))
    features = list(dataset.columns[1:])
    X = dataset[features]
    for Y_col in targets:
        # load model
        with open(f'../data/rac_model_{Y_col}.pkl', 'rb') as file:
            model_final = pickle.load(file)
            y_predict = model_final.predict(X)
            if Y_col in y_trans_dict.keys():
                predict_df[Y_col] = y_inv_trans_dict[Y_col](y_predict)
            else:
                predict_df[Y_col] = y_predict
    print(predict_df)
    predict_df.to_csv("../data/rac_sov.csv", index=False)


if __name__ == '__main__':
    main()
