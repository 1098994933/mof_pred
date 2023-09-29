"""
data prediction
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
    X = dataset[features]
    alg_dict = {
        # "Lasso": Lasso(),
        # "Ridge": Ridge(),
        # "LinearRegression": LinearRegression(),
        # 'LinearSVR': SVR(kernel='linear'),
        f"Random Forest": RandomForestRegressor(random_state=0,
                              min_samples_split=4,
                              # min_samples_leaf=i,
                              n_estimators=100)
        # f"Random Forest{i}": RandomForestRegressor(random_state=0,
        #                                            min_samples_split=4,
        #                                            # min_samples_leaf=i,
        #                                            n_estimators=i)
        # # for i in range(2, 10)
        # for i in [50,100,200,500]
        # "GradientBoosting": GradientBoostingRegressor(random_state=0),
        # "GradientBoosting": GradientBoostingRegressor(min_samples_split=8),
        # "GradientBoosting2": GradientBoostingRegressor(min_samples_split=4),
        # "GradientBoosting3": GradientBoostingRegressor(min_samples_split=2),
        # "GradientBoosting4": GradientBoostingRegressor(),
        # "AdaBoost": AdaBoostRegressor(),
        # "ExtraTrees": ExtraTreesRegressor(),
        # "RandomForest": RandomForestRegressor(),
        # "RandomForest2": RandomForestRegressor(n_estimators=50, min_samples_split=8),
        # "RandomForest2": RandomForestRegressor(n_estimators=200, min_samples_split=8),
        # "RandomForest3": RandomForestRegressor(n_estimators=10, min_samples_split=4),
        # "RandomForest4": RandomForestRegressor(n_estimators=10),
        # "KNeighbors": KNeighborsRegressor(),
        # "DecisionTree": DecisionTreeRegressor(),
        # 'RbfSVR': SVR(kernel='rbf')
    }

    for Y_col in targets:
        print(Y_col)
        Y = dataset[Y_col]
        best_model = None
        best_score = - 10 ** 10
        for alg_name in alg_dict.keys():
            model = alg_dict[alg_name]
            # model.fit(X_train, Y_train)
            # y_predict = model.predict(X_test)
            # score = -r2_score(Y_test, y_predict)
            score = np.mean(cross_val_score(model, X, Y, cv=cv_folder, scoring='neg_root_mean_squared_error'))
            print(f"{alg_name} {score}")
            if score > best_score:
                best_model = model
                best_score = score
        # save the best model
        print(f"best score {best_score} best model {best_model}")
        model_final = best_model
        model_final.fit(X, Y)
        with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
            pickle.dump(model_final, file)
    # model_final = RandomForestRegressor()
    # model_final = DecisionTreeRegressor()
    model_final.fit(X, Y)
    # evaluate
    # score = np.median(cross_val_score(model_final, X, Y, cv=cv_folder))
    # print(f"Y_col: {Y_col} , score:{score}")
    # save model
    # with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
    #     pickle.dump(model_final, file)
    dataset = pd.read_csv("../data/RAC_test.csv")
    targets = ['temperature', 'time']
    predict_df = pd.DataFrame()
    predict_df['mof'] = list(range(1, len(dataset) + 1))
    features = list(dataset.columns[1:])
    X = dataset[features]
    for Y_col in targets:
        # load model
        with open(f'../data/rac_model_{Y_col}.pkl', 'rb') as file:
            model_final = pickle.load(file)
            predict_df[Y_col] = model_final.predict(X)
    predict_df.to_csv("../data/RAC_prediction.csv", index=False)
    print(predict_df)


if __name__ == '__main__':
    main()
