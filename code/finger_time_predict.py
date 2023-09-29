"""
finger temperature prediction by molecular descriptor
f1: finger print Similarity
f2: mordred feature
f3: mol2vec
f4: metal elements features
"""
import pickle
import random

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import os
import scipy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from util.alg import Linear_SVR, RBF_SVR
from config import config

val_config = {
    "feature_num": 2446,
    "cv_fold": 5,
    'test_ratio': 0.20  # extra test data ratio
}


def main():
    random.seed(0)
    np.random.seed(0)
    dataset = pd.read_csv("../data/finger_train.csv")
    Y_col = 'time'
    ele_features = dataset.columns[1:-5]
    df_f4 = dataset[ele_features]
    # df_f0 = pd.read_csv("../data/finger_train_f0.csv")
    df_f1 = pd.read_csv("../data/finger_train_f1.csv")
    df_f2 = pd.read_csv("../data/finger_train_f22.csv")
    df_f3 = pd.read_csv("../data/finger_train_f3.csv")
    df_f5 = pd.read_csv("../data/External/train_formula_magpie_features.csv")
    df_f5 = df_f5[list(df_f5.columns[2:])]
    df_f6 = pd.read_csv("../data/finger_train_m_magpie.csv").fillna(-1)
    df_f7 = pd.read_csv("../data/train_f6.csv")
    # df_f8 = pd.read_csv("../data/finger_train_f8.csv")
    print("test shape", df_f1.shape)
    print("test shape", df_f2.shape)
    print("test shape", df_f3.shape)
    print("test shape", df_f4.shape)
    print("test shape", df_f5.shape)
    print("test shape", df_f6.shape)
    print("test shape", df_f7.shape)
    #df_f8
    ml_dataset = pd.concat([df_f7, df_f6, df_f5, df_f2, df_f3, df_f4, dataset[[Y_col]]], axis=1)
    ml_dataset.to_csv("../data/ml_dataset_time.csv", index=False)
    features = list(ml_dataset.columns)[:-1]
    print("features", len(features), features)
    X = ml_dataset[features]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Y = ml_dataset[Y_col]
    Y[Y < 12] = 12
    Y = np.log10(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=val_config['test_ratio'],
                                                        random_state=1)
    # from sklearn.feature_selection import VarianceThreshold
    # var = VarianceThreshold(threshold=0)
    # X = var.fit_transform(X)
    print(X_train.shape, Y_train.shape)
    X_train = pd.DataFrame(X_train, columns=features)
    columns_with_nulls = X_train.columns[X_train.isnull().any()]
    print(columns_with_nulls)
    feature_selection = SelectKBest(f_regression, k=val_config['feature_num']).fit(X_train, Y_train)

    feature_scores = feature_selection.scores_
    print('feature_scores:', feature_scores)
    indices = np.argsort(feature_scores)[::-1]
    best_features = list(X_train.columns.values[indices[0:val_config['feature_num']]])

    X_train = feature_selection.transform(X_train)
    X_test = feature_selection.transform(X_test)
    sc = MinMaxScaler()

    alg_dict = {
        # "Lasso": Lasso(),
        # "Ridge": Ridge(),
        # "LinearRegression": LinearRegression(),
        # 'LinearSVR': Linear_SVR(C=1),
        # 'LinearSVR2': Linear_SVR(C=100),
        # 'LinearSVR3': Linear_SVR(C=10),
        # "GradientBoosting": GradientBoostingRegressor(),
        # "AdaBoost": AdaBoostRegressor(),
        # "ExtraTrees": ExtraTreesRegressor(),
        "RandomForest": RandomForestRegressor(random_state=1),
        # "RandomForest2": RandomForestRegressor(random_state=2),
        # "RandomForest": RandomForestRegressor(min_samples_split=9, random_state=0),
        # f"Random Forest{i}": RandomForestRegressor(random_state=0,
        #                                            min_samples_split=i,
        #                                            # min_samples_leaf=1,
        #                                            n_estimators=10)
        # for i in range(2, 10)
        # "RandomForest3": RandomForestRegressor(random_state=3),
        # "RandomForest4": RandomForestRegressor(random_state=5),
        # "RandomForest5": RandomForestRegressor(random_state=6),
        # "KNeighbors": KNeighborsRegressor(),
        # "DecisionTree": DecisionTreeRegressor(),
        # 'RbfSVR': RBF_SVR(C=1),
        # 'RbfSVR1': RBF_SVR(C=10, gamma=0.20),
        # 'RbfSVR2': RBF_SVR(C=100, gamma=0.10),
        # 'RbfSVR3': RBF_SVR(C=1000, gamma=0.05),
        # 'RbfSVR4': RBF_SVR(C=0.1, gamma=0.01),
    }
    best_model = None
    best_score = 10 ** 10
    for alg_name in alg_dict.keys():
        model = alg_dict[alg_name]
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        #score = -r2_score(Y_test, y_predict)
        score = - np.mean(cross_val_score(model, X, Y, cv=5))
        bias = np.mean(Y_test) - np.mean(y_predict)
        # print(f"bias: {bias}")
        # score = - np.median(cross_val_score(model, X, Y, cv=5))
        print(f"{alg_name} {score}")
        if score < best_score:
            best_model = model
            best_score = score
    # save the best model
    print(f"best score {best_score} best model {best_model}")
    model_final = best_model
    X_df = pd.DataFrame(feature_selection.transform(X), columns=best_features)
    model_final.fit(X_df, Y)

    with open(f'../data/model_{Y_col}.pkl', 'wb') as file:
        pickle.dump(model_final, file)

    # test data prediction
    dataset = pd.read_csv(config['test_data_finger'])
    ele_features = dataset.columns[1:-3]
    df_f4 = dataset[ele_features]
    # df_f0 = pd.read_csv("../data/finger_test_f0.csv")
    df_f1 = pd.read_csv("../data/finger_test_f1.csv")
    df_f2 = pd.read_csv("../data/finger_test_f22.csv")
    df_f3 = pd.read_csv("../data/finger_test_f3.csv")
    df_f5 = pd.read_csv("../data/External/test_formula_magpie_features.csv").fillna(-1)
    df_f5 = df_f5[list(df_f5.columns[2:])]
    df_f6 = pd.read_csv("../data/finger_test_m_magpie.csv").fillna(-1)
    df_f7 = pd.read_csv("../data/test_f6.csv")
    # df_f8 = pd.read_csv("../data/finger_test_f8.csv")
    # X_t = pd.concat([df_f8, df_f7, df_f6, df_f5, df_f2, df_f3, df_f4], axis=1)
    X_t = pd.concat([df_f7, df_f6, df_f5, df_f2, df_f3, df_f4], axis=1)

    X_t.to_csv("../data/test_ml_dataset.csv", index=False)
    print("test shape", df_f1.shape)
    print("test shape", df_f2.shape)
    print("test shape", df_f3.shape)
    print("test shape", df_f4.shape)
    X_t = scaler.transform(X_t)
    X_t = feature_selection.transform(X_t)
    y_sub = model_final.predict(X_t)

    # save prediction
    predict_df = pd.DataFrame()
    predict_df['mof'] = list(range(1, len(y_sub) + 1))
    predict_df[Y_col] = 10 ** y_sub
    predict_df.to_csv(f"../data/finger_{Y_col}.csv", index=False)


if __name__ == '__main__':
    main()
