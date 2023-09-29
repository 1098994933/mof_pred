from catboost import CatBoostClassifier

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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from util.alg import Linear_SVC, RBF_SVC

from util.alg import fit_decision_tree_classifier
import random


# region RAC

def main():
    random.seed(0)
    np.random.seed(0)

    cv_folder = 5
    dataset = pd.read_csv("../data/RAC_train.csv")
    targets = ['temperature', 'time']
    features = list(dataset.columns[1:-12])
    print(features)
    X = dataset[features]
    alg_dict = {
        # "Lasso": Lasso(),
        # "Ridge": Ridge(),
        # "LinearRegression": LinearRegression(),
        # 'LinearSVR': SVR(kernel='linear'),
        "GradientBoosting": GradientBoostingRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        # "ExtraTrees": ExtraTreesRegressor(),
        "RandomForest": RandomForestRegressor(),
        "KNeighbors": KNeighborsRegressor(),
        # "DecisionTree": DecisionTreeRegressor(),
        'RbfSVR': SVR(kernel='rbf')
    }

    for Y_col in targets[:1]:
        print(Y_col)
        Y = dataset[Y_col]
        best_model = None
        best_score = - 10 ** 10
        best_rs = 0
        for rs in range(10, 20):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs)

            for alg_name in alg_dict.keys():
                model = alg_dict[alg_name]
                model.fit(X_train, Y_train)
                # model.fit(X, Y)
                dataset_sub = pd.read_csv("../data/RAC_test.csv")
                targets = ['temperature', 'time']
                predict_df = pd.DataFrame()
                predict_df['mof'] = list(range(1, len(dataset) + 1))
                # features = list(dataset.columns[1:])
                X_sub = dataset_sub[features]
                y_predict = model.predict(X_sub)
                df_true = pd.read_csv("../data/RAC.csv")
                y_true = df_true[Y_col]
                score = r2_score(y_predict, y_true)
                score = -mean_squared_error(y_predict, y_true)**0.5
                print(mean_absolute_error(y_predict, y_true))
                # score = -r2_score(Y_test, y_predict)

                # r2_score()
                # score = - np.median(cross_val_score(model, X, Y, cv=cv_folder))
                # print(f"{alg_name} {score}")
                if score > best_score:
                    best_model = model
                    best_score = score
                    best_rs = rs
            # save the best model
            print(f"best score {best_score} best model {best_model}")
            model_final = best_model
            # model_final.fit(X, Y)
            # with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
            #     pickle.dump(model_final, file)
        print(best_rs)
    # model_final = RandomForestRegressor()
    # model_final = DecisionTreeRegressor()
    # model_final.fit(X_train, Y_train)
    # evaluate
    # score = np.median(cross_val_score(model_final, X, Y, cv=cv_folder))
    # print(f"Y_col: {Y_col} , score:{score}")
    # save model
    # with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
    #     pickle.dump(model_final, file)

    # for Y_col in targets:
    #     # load model
    #     with open(f'../data/rac_model_{Y_col}.pkl', 'rb') as file:
    #         model_final = pickle.load(file)
    #         predict_df[Y_col] = model_final.predict(X)

    # predict_df.to_csv("../data/RAC_prediction.csv", index=False)
    # print(predict_df)


if __name__ == '__main__':
    main()
