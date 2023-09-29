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
from sklearn.metrics import mean_squared_error
import random

random.seed(0)
np.random.seed(0)


def best_eval_rac_temperature():
    """
    :return: best score
    rmse mean: 27.913885899848374
    r2 mean 0.3049718349809741
    """
    # log?
    dataset = pd.read_csv("../data/RAC_train.csv")
    Y_col = "temperature"
    features = list(dataset.columns[1:-12])
    X = dataset[features]
    Y = dataset[Y_col]

    score_list = []
    score2_list = []
    for rs in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs)
        # model = GradientBoostingRegressor()
        model = RandomForestRegressor(random_state=0,
                                      min_samples_split=4,
                                      n_estimators=100)
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        score = mean_squared_error(Y_test, y_predict) ** 0.5
        score2 = r2_score(Y_test, y_predict)
        print("rmse", score)
        score_list.append(score)
        print("r2", score2)
        score2_list.append(score2)
    print("===============================")
    print("rmse mean:", np.array(score_list).mean())
    print("r2 mean", np.array(score2_list).mean())
    # save model
    model_final = model
    model_final.fit(X, Y)
    with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
        pickle.dump(model_final, file)
    return 0

def best_eval_rac_param3():
    """
    rmse mean: 0.21854560732796796
    r2 mean -0.09836647993741547
    :return:
    """
    dataset = pd.read_csv("../data/RAC_train.csv")
    Y_col = "param3"
    features = list(dataset.columns[1:-12])
    X = dataset[features]
    Y = dataset[Y_col]

    score_list = []
    score2_list = []
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

    for rs in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs)

        if Y_col in y_trans_dict.keys():
            Y_train = y_trans_dict[Y_col](Y_train)
        # model = GradientBoostingRegressor()
        model = RandomForestRegressor()
        #print(Y_train)
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        if Y_col in y_trans_dict.keys():
            y_predict = y_inv_trans_dict[Y_col](y_predict)
        else:
            y_predict = y_predict
        score = mean_squared_error(Y_test, y_predict) ** 0.5
        score2 = r2_score(Y_test, y_predict)
        print("rmse", score)
        score_list.append(score)
        print("r2", score2)
        score2_list.append(score2)
    print("===============================")
    print("rmse mean:", np.array(score_list).mean())
    print("r2 mean", np.array(score2_list).mean())
    # save model
    model_final = model
    model_final.fit(X, Y)
    with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
        pickle.dump(model_final, file)
    return 0

def best_eval_rac_param1():
    """
    rmse mean: 0.13091757459909356
    r2 mean 0.08839754298725284
    :return:
    """
    dataset = pd.read_csv("../data/RAC_train.csv")
    Y_col = "param1"
    features = list(dataset.columns[1:-12])
    X = dataset[features]
    Y = dataset[Y_col]

    score_list = []
    score2_list = []
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

    for rs in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs)

        if Y_col in y_trans_dict.keys():
            Y_train = y_trans_dict[Y_col](Y_train)
        # model = GradientBoostingRegressor()
        model = RandomForestRegressor()
        #print(Y_train)
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        if Y_col in y_trans_dict.keys():
            y_predict = y_inv_trans_dict[Y_col](y_predict)
        else:
            y_predict = y_predict
        score = mean_squared_error(Y_test, y_predict) ** 0.5
        score2 = r2_score(Y_test, y_predict)
        print("rmse", score)
        score_list.append(score)
        print("r2", score2)
        score2_list.append(score2)
    print("===============================")
    print("rmse mean:", np.array(score_list).mean())
    print("r2 mean", np.array(score2_list).mean())
    # save model
    model_final = model
    model_final.fit(X, Y)
    with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
        pickle.dump(model_final, file)
    return 0

def best_eval_rac_param5():
    """
    rmse mean: 0.19542425901168362
    r2 mean 0.06406502334615626
    :return:
    """
    dataset = pd.read_csv("../data/RAC_train.csv")
    Y_col = "param5"
    features = list(dataset.columns[1:-12])
    X = dataset[features]
    Y = dataset[Y_col]

    score_list = []
    score2_list = []
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

    for rs in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs)

        if Y_col in y_trans_dict.keys():
            Y_train = y_trans_dict[Y_col](Y_train)
        # model = GradientBoostingRegressor()
        model = RandomForestRegressor()
        #print(Y_train)
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        if Y_col in y_trans_dict.keys():
            y_predict = y_inv_trans_dict[Y_col](y_predict)
        else:
            y_predict = y_predict
        score = mean_squared_error(Y_test, y_predict) ** 0.5
        score2 = r2_score(Y_test, y_predict)
        print("rmse", score)
        score_list.append(score)
        print("r2", score2)
        score2_list.append(score2)
    print("===============================")
    print("rmse mean:", np.array(score_list).mean())
    print("r2 mean", np.array(score2_list).mean())
    # save model
    model_final = model
    model_final.fit(X, Y)
    with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
        pickle.dump(model_final, file)
    return 0

def best_eval_rac_param2():
    """
    rmse mean: 0.2662453880334337
    r2 mean 0.16477114353684935
    :return:
    """
    dataset = pd.read_csv("../data/RAC_train.csv")
    Y_col = "param2"
    features = list(dataset.columns[1:-12])
    X = dataset[features]
    Y = dataset[Y_col]

    score_list = []
    score2_list = []
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

    for rs in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs)

        if Y_col in y_trans_dict.keys():
            Y_train = y_trans_dict[Y_col](Y_train)
        # model = GradientBoostingRegressor()
        model = RandomForestRegressor()
        #print(Y_train)
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        if Y_col in y_trans_dict.keys():
            y_predict = y_inv_trans_dict[Y_col](y_predict)
        else:
            y_predict = y_predict
        score = mean_squared_error(Y_test, y_predict) ** 0.5
        score2 = r2_score(Y_test, y_predict)
        print("rmse", score)
        score_list.append(score)
        print("r2", score2)
        score2_list.append(score2)
    print("===============================")
    print("rmse mean:", np.array(score_list).mean())
    print("r2 mean", np.array(score2_list).mean())
    # save model
    model_final = model
    model_final.fit(X, Y)
    with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
        pickle.dump(model_final, file)
    return 0

def best_finger_time():
    """
    rmse mean: 51.85976851997334
    r2 mean -0.19207650925471745
    :return:
    """
    # finger
    dataset = pd.read_csv("../data/ml_dataset_time.csv")
    Y_col = 'time'
    features = list(dataset.columns[:-1])
    X = dataset[features]
    Y = dataset[dataset.columns[-1]]

    score_list = []
    score2_list = []
    for rs in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs)
        model = RandomForestRegressor()
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        score = mean_squared_error(Y_test, y_predict) ** 0.5
        score2 = r2_score(Y_test, y_predict)
        print("rmse", score)
        score_list.append(score)
        print("r2", score2)
        score2_list.append(score2)
    print("===============================")
    print("rmse mean:", np.array(score_list).mean())
    print("r2 mean", np.array(score2_list).mean())


if __name__ == '__main__':
    # best_eval_rac_temperature()
    # best_eval_rac_param3()
    # best_eval_rac_param1()
    # best_eval_rac_param2()
    # best_eval_rac_param5()

    # rac
    # dataset = pd.read_csv("../data/RAC_train.csv")
    # targets = ['temperature', 'time', 'param1', 'param2', 'param3', 'param4', 'param5']
    # Y_col = targets[2]
    # features = list(dataset.columns[1:-12])
    # X = dataset[features]
    # Y = dataset[Y_col]
    #
    # score_list = []
    # score2_list = []
    # for rs in range(10):
    #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs)
    #     model = GradientBoostingRegressor()
    #     model.fit(X_train, Y_train)
    #     y_predict = model.predict(X_test)
    #     score = mean_squared_error(Y_test, y_predict) ** 0.5
    #     score2 = r2_score(Y_test, y_predict)
    #     print("rmse", score)
    #     score_list.append(score)
    #     print("r2", score2)
    #     score2_list.append(score2)
    # print("===============================")
    # print("rmse mean:", np.array(score_list).mean())
    # print("r2 mean", np.array(score2_list).mean())


    # finger
    Y_col = 'temperature'
    dataset = pd.read_csv(f"../data/ml_dataset_{Y_col}.csv")

    features = list(dataset.columns[:-1])
    X = dataset[features]
    Y = dataset[dataset.columns[-1]]

    score_list = []
    score2_list = []
    for rs in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs)
        model = RandomForestRegressor()
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        score = mean_squared_error(Y_test, y_predict) ** 0.5
        score2 = r2_score(Y_test, y_predict)
        print("rmse", score)
        score_list.append(score)
        print("r2", score2)
        score2_list.append(score2)
    print("===============================")
    print("rmse mean:", np.array(score_list).mean())
    print("r2 mean", np.array(score2_list).mean())

