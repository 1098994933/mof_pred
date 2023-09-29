import gplearn
from sklearn.metrics import r2_score, mean_squared_error

from config import config
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def evaluate_model_plot(y_true, y_predict, residual=False, show=True):
    """
    评价模型误差
    y_true: 真实值
    y_predict: 预测值
    """
    if show == True:
        # 图形基础设置
        print("开始画图")
        plt.figure(figsize=(7, 5), dpi=400)
        plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        plt.grid(linestyle="--")  # 设置背景网格线为虚线
        ax = plt.gca()  # 获取坐标轴对象
        plt.scatter(y_true, y_predict, color='red')
        plt.plot(y_predict, y_predict, color='blue')
        # 画200%误差线
        # plt.plot(y_predict, y_predict+0.301, color ='blue',linestyle = "--")
        # plt.plot(y_predict, y_predict-0.301, color ='blue',linestyle = "--")
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.xlabel("Measured", fontsize=12, fontweight='bold')
        plt.ylabel("Predicted", fontsize=12, fontweight='bold')
        # plt.xlim(4, 9)  # 设置x轴的范围
        # plt.ylim(4, 9)
        plt.savefig('./genetic.svg', format='svg')
        plt.show()

        # 预测值 真实值图
        # plt.subplot(1, 2, 1)
        # plt.scatter(y_true, y_predict, color='red')
        # plt.plot(y_predict, y_predict, color='blue')
        ##画200%误差线
        ##plt.plot(y_predict, y_predict+0.301, color ='blue',linestyle = "--")
        ##plt.plot(y_predict, y_predict-0.301, color ='blue',linestyle = "--")
        # plt.xticks(fontsize=12, fontweight='bold')
        # plt.yticks(fontsize=12, fontweight='bold')
        # plt.xlabel("Measured", fontsize=12, fontweight='bold')
        # plt.ylabel("Predicted", fontsize=12, fontweight='bold')
        # plt.xlim(4, 9)  # 设置x轴的范围
        # plt.ylim(4, 9)
        ##plt.title("fit effect",fontsize = 30)
        ## 残差分布图
        # plt.subplot(1, 2, 2)
        plt.figure(figsize=(7, 5), dpi=400)
        plt.hist(np.array(y_true) - np.array(y_predict), 40)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Residual", fontsize=20)
        plt.ylabel("Freq", fontsize=20)
        # plt.title("Residual=y_true-y_pred",fontsize = 30)
        plt.show()
    from sklearn.metrics import mean_absolute_error

    n = len(y_true)
    MSE = mean_squared_error(y_true, y_predict)
    RMSE = pow(MSE, 0.5)
    MAE = mean_absolute_error(y_true, y_predict)
    R2 = r2_score(y_true, y_predict)
    MRE = 0
    print("样本个数 ", round(n))
    print("均方根误差RMSE ", round(RMSE, 3))
    print("均方差MSE ", round(MSE, 3))
    print("平均绝对误差MAE ", round(MAE, 3))
    print("R2：", round(R2, 3))
    return dict({"n": n, "MSE": MSE, "RMSE": RMSE, "MSE": MSE, "MAE": MAE, "R2": R2, "MRE": MRE})
if __name__ == '__main__':
    # 使用 SymbolicTransformer 在训练集上生成新的特征
    CV_FOLD = 5
    data_path = config['data_path']

    dataset = pd.read_csv(data_path + "RAC_train.csv")
    targets = ['temperature', 'time']
    Y_col = targets[1]
    features = list(dataset.columns[1:-12])
    X = dataset[features]
    Y = dataset[Y_col]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    feature_names = features
    # 定义操作符
    function_set = ['add', 'sub', 'mul', 'div',
                    'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos']
    from gplearn.genetic import SymbolicTransformer

    gp = SymbolicTransformer(generations=6,  # 最大迭代次数
                             population_size=2000,  # 每代个体数
                             hall_of_fame=100,  # hall_of_fame=100 选择生成100个与因变量相关系数最大的特征
                             n_components=10,  # n_components=10 表示从100个特征选择10个特征直接相关性最小的 以较少冗余
                             function_set=function_set,
                             parsimony_coefficient=0.005,  # 复杂度惩罚系数C
                             max_samples=0.8,  # 使用90%数据训练 剩余作为包外估计（OOB）
                             tournament_size=100,
                             verbose=1,
                             random_state=0, n_jobs=6, feature_names=feature_names
                             )
    gp.fit(X_train, Y_train)

    # 加入新生成的10个特征
    X_train_GP = np.hstack((X_train, gp.transform(X_train)))
    X_test_GP = np.hstack((X_test, gp.transform(X_test)))

    model = GradientBoostingRegressor()
    model.fit(X_train_GP, Y_train)
    y_pred = model.predict(X_test_GP)
    evaluation = evaluate_model_plot(Y_test, y_pred)
    print(evaluation)

