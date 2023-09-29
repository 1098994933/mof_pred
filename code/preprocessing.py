"""
data preprocessing and features calculation
"""
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# from eval_method import cal_metric
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import scipy.spatial as spt
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import pickle
import rdkit
from rdkit.Chem import MolFromSmiles, RDKFingerprint
from rdkit.DataStructs import FingerprintSimilarity
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.impute import SimpleImputer
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MolFromSmiles
import pandas as pd
from rdkit.Chem import MolFromSmiles
import numpy as np
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
# from matminer.featurizers.conversions import StrToComposition
# from matminer.featurizers.base import MultipleFeaturizer
# from matminer.featurizers import composition as cf
from config import config


def clean(s):
    try:
        s = float(s)
    except:
        return -1

    if isinstance(s, str):
        return np.nan
    else:
        return s


def preprocess_step1():
    """
    data preprocessing and features calculation
    """
    # finger
    # generate finger similarity
    dataset = pd.read_csv("../data/finger_train.csv")
    # f0
    train_fps = [RDKFingerprint(MolFromSmiles(mol)) for mol in dataset.linker1smi]
    fps_binary = [mol.ToList() for mol in train_fps]
    train_f0 = np.array(fps_binary)
    features_names = [f"f0_{i}" for i in range(train_f0.shape[1])]
    df_f = pd.DataFrame(train_f0, columns=features_names)
    print("train f0 shape", df_f.shape)
    df_f.to_csv('../data/finger_train_f0.csv', index=False)

    # # f1 sim
    train_f1 = np.array([[FingerprintSimilarity(train_fps[k], train_fps[i])
                          for i in range(len(train_fps))]
                         for k in range(len(train_fps))])
    features_names = [f"f1_{i}" for i in range(train_f1.shape[1])]
    df_f = pd.DataFrame(train_f1, columns=features_names)
    print("train f1 shape", df_f.shape)
    df_f.to_csv('../data/finger_train_f1.csv', index=False)

    # f2: mo
    calc = Calculator(descriptors, ignore_3D=False)
    train_f2 = np.array([calc(MolFromSmiles(mol)) for mol in dataset.linker1smi])

    features_names = [f"f2_{i}" for i in range(train_f2.shape[1])]
    df_f = pd.DataFrame(train_f2, columns=features_names)
    print("train f2 shape", df_f.shape)
    df_f.to_csv('../data/finger_train_f2.csv', index=False)

    # generate features for test set
    test_dataset = pd.read_csv(config['test_data_finger'])
    test_fps = [RDKFingerprint(MolFromSmiles(mol)) for mol in test_dataset.linker1smi]
    test_fps_binary = [mol.ToList() for mol in test_fps]

    test_f0 = np.array(test_fps_binary)
    features_names = [f"f0_{i}" for i in range(train_f0.shape[1])]
    df_f = pd.DataFrame(test_f0, columns=features_names)
    print("train f0 shape", df_f.shape)
    print(df_f)
    df_f.to_csv('../data/finger_test_f0.csv', index=False)

    # get the sim of test data with train data
    test_f2 = np.array([[FingerprintSimilarity(train_fps[k], train_fps[i])
                         for i in range(len(train_fps))]
                        for k in range(len(test_fps))])
    features_names = [f"f1_{i}" for i in range(test_f2.shape[1])]
    df_f = pd.DataFrame(test_f2, columns=features_names)
    print("test f1 shape", df_f.shape)
    df_f.to_csv('../data/finger_test_f1.csv', index=False)

    calc = Calculator(descriptors, ignore_3D=False)
    test_f2 = np.array([calc(MolFromSmiles(mol)) for mol in test_dataset.linker1smi])
    features_names = [f"f2_{i}" for i in range(test_f2.shape[1])]
    df_f = pd.DataFrame(test_f2, columns=features_names)
    print("test f2 shape", df_f.shape)
    df_f.to_csv('../data/finger_test_f2.csv', index=False)


def preprocess_step2():
    """
    run after data preprocessing
    fill null values
    """
    for data in ['train', 'test']:
        train_f2 = pd.read_csv(f"../data/finger_{data}_f2.csv")
        train_f2 = train_f2.applymap(clean)
        print(data, train_f2.shape)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(train_f2)
        # train_rdkit_feature_df = pd.DataFrame(train_f2, columns=des_list)
        train_f2.to_csv(f"../data/finger_{data}_f22.csv")


def preprocess_step3():
    """
    cal magpie features for molecular
    """

    # train magpie features
    for data in ['train']:
        dataset = pd.read_csv(f"../data/finger_{data}.csv")
        # calculate formula
        formulas = [CalcMolFormula(MolFromSmiles(smile)) for smile in dataset.linker1smi]
        formulas_process = []
        for formula in formulas:
            if '-' in formula:
                print(formula)
                formulas_process.append(formula.split("-")[0])
            else:
                formulas_process.append(formula)

        df = pd.DataFrame({"formula": formulas_process})
        df.to_csv(f"../data/{data}_formula.csv", index=False)
        #
        # data_path = "../data/"
        # file_name = "train_formula.csv"
        # df_chemistry_formula = pd.read_csv(data_path + file_name)
        # df_magpie = StrToComposition(target_col_id='composition_obj').featurize_dataframe(df_chemistry_formula,
        #                                                                                   'formula')
        # feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
        #                                           cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
        # feature_labels = feature_calculators.feature_labels()
        # df_magpie = feature_calculators.featurize_dataframe(df_magpie, col_id='composition_obj');
        # df_magpie.to_csv(data_path + 'train_formula_magpie_features.csv', index=False)
        #
        # file_name = "test_formula.csv"
        # df_chemistry_formula = pd.read_csv(data_path + file_name)
        # df_magpie = StrToComposition(target_col_id='composition_obj').featurize_dataframe(df_chemistry_formula,
        #                                                                                   'formula')
        # feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
        #                                           cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
        # feature_labels = feature_calculators.feature_labels()
        # df_magpie = feature_calculators.featurize_dataframe(df_magpie, col_id='composition_obj');
        # df_magpie.to_csv(data_path + 'test_formula_magpie_features.csv', index=False)

    dataset = pd.read_csv(config['test_data_finger'])
    data = 'test'
    # calculate formula
    formulas = [CalcMolFormula(MolFromSmiles(smile)) for smile in dataset.linker1smi]
    formulas_process = []
    for formula in formulas:
        if '-' in formula:
            print(formula)
            formulas_process.append(formula.split("-")[0])
        else:
            formulas_process.append(formula)

    df = pd.DataFrame({"formula": formulas_process})
    df.to_csv(f"../data/{data}_formula.csv", index=False)

def preprocess_step4():
    """
    get metal magpie features
    """
    ele_data = pd.read_excel("../data/External/elements.xlsx")
    df = pd.read_csv("../data/finger_train.csv")
    metal_magpie = pd.merge(df, ele_data, how='left', left_on=['metal'], right_on=['formula'])
    print(metal_magpie.shape)
    print(metal_magpie.columns[25:])
    col_names = [f"m_{i}" for i in list(metal_magpie.columns[25:])]
    df_train_metal = pd.DataFrame(metal_magpie[list(metal_magpie.columns[25:])], columns=col_names)

    df_train_metal.to_csv("../data/finger_train_m_magpie.csv", index=False)

    df = pd.read_csv(config['test_data_finger'])
    metal_magpie = pd.merge(df, ele_data, how='left', left_on=['metal'], right_on=['formula'])
    print(metal_magpie.columns[23:])
    df_train_metal = pd.DataFrame(metal_magpie[list(metal_magpie.columns[23:])], columns=col_names)
    df_train_metal.to_csv("../data/finger_test_m_magpie.csv", index=False)


def preprocess_step6():
    train_df = pd.read_csv(f"../data/finger_train_f3.csv")
    Y = pd.read_csv(f"../data/finger_train.csv").temperature
    A = np.array(train_df)
    # A是一个向量矩阵：euclidean代表欧式距离
    distA = pdist(A, metric='euclidean')
    # 将distA数组变成一个矩阵
    distB = squareform(distA)
    # 查找最近点
    point = np.array(train_df)  # 二维样本点数组
    kt = spt.KDTree(data=point, leafsize=10)  # 用于快速查找的KDTree类
    ckt = spt.cKDTree(point)  # 用C写的查找类，执行速度更快
    find_point = point[300]
    d, x = kt.query(find_point, 5)  # 返回最近邻点的距离d和在数组中的顺序x
    y_pre = []
    sim_train_features = []
    for i in range(len(point)):
        find_point = point[i]
        k = 8
        d, indexes = kt.query(find_point, k)  # 返回最近邻点的距离d和在数组中的顺序x
        # dis.append(d[1])
        # y_dis.append(abs(Y[x[0]] - Y[x[1]]))
        feature = [Y[index] for index in indexes if index != i]  # 排除自己
        feature.append(np.array(feature).mean())
        feature.append(np.array(feature).max())
        feature.append(np.array(feature).min())
        feature.append(np.array(feature).std())
        y_pre.append(np.array(feature).mean())
        print(feature, Y[i])
        sim_train_features.append(feature[:k + 3])
    train_f6 = pd.DataFrame(np.array(sim_train_features))

    train_f6.to_csv(f"../data/train_f6.csv", index=False)

    train_df = pd.read_csv(f"../data/finger_train_f3.csv")
    Y = pd.read_csv(f"../data/finger_train.csv").temperature
    A = np.array(train_df)
    # A是一个向量矩阵：euclidean代表欧式距离
    distA = pdist(A, metric='euclidean')
    # 将distA数组变成一个矩阵
    distB = squareform(distA)

    # 查找最近点
    test_df = pd.read_csv(f"../data/finger_test_f3.csv")
    point = np.array(test_df)  # 二维样本点数组
    kt = spt.KDTree(data=point, leafsize=10)  # 用于快速查找的KDTree类
    ckt = spt.cKDTree(point)  # 用C写的查找类，执行速度更快

    y_pre = []
    sim_train_features = []
    for i in range(len(point)):
        find_point = point[i]
        k = 8
        d, indexes = kt.query(find_point, k)  # 返回最近邻点的距离d和在数组中的顺序x
        # dis.append(d[1])
        # y_dis.append(abs(Y[x[0]] - Y[x[1]]))
        feature = [Y[index] for index in indexes if index != i]  # 排除自己
        feature.append(np.array(feature).mean())
        feature.append(np.array(feature).max())
        feature.append(np.array(feature).min())
        feature.append(np.array(feature).std())
        y_pre.append(np.array(feature).mean())
        print(feature, Y[i])
        sim_train_features.append(feature[:k + 3])

    print(sim_train_features)
    train_f6 = pd.DataFrame(np.array(sim_train_features))
    print(train_f6.shape)
    # print(pearsonr(Y, y_pre))
    train_f6.to_csv(f"../data/test_f6.csv", index=False)


def preprocess_step5():
    """
    embedding a molecular with 300 dim embedding by mod2vec
    """
    model = word2vec.Word2Vec.load('../data/External/model_300dim.pkl')
    # generate train f3
    dataset = pd.read_csv("../data/finger_train.csv")
    dataset['mol'] = dataset['linker1smi'].apply(lambda x: MolFromSmiles(x))
    mdf = dataset

    mdf['sentence'] = mdf.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)

    # Extracting embeddings to a numpy.array
    # Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
    mdf['mol2vec'] = [DfVec(x) for x in sentences2vec(mdf['sentence'], model, unseen='UNK')]

    train_f3 = np.array([x.vec for x in mdf['mol2vec']])
    features_names = [f"f3_{i}" for i in range(train_f3.shape[1])]
    df_f = pd.DataFrame(train_f3, columns=features_names)
    df_f.to_csv('../data/finger_train_f3.csv', index=False)

    # generate test
    dataset = pd.read_csv(config['test_data_finger'])
    dataset['mol'] = dataset['linker1smi'].apply(lambda x: MolFromSmiles(x))
    mdf = dataset
    mdf['sentence'] = mdf.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)

    # Extracting embeddings to a numpy.array
    # Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
    mdf['mol2vec'] = [DfVec(x) for x in sentences2vec(mdf['sentence'], model, unseen='UNK')]
    test_f3 = np.array([x.vec for x in mdf['mol2vec']])
    features_names = [f"f3_{i}" for i in range(test_f3.shape[1])]
    df_f = pd.DataFrame(test_f3, columns=features_names)
    df_f.to_csv('../data/finger_test_f3.csv', index=False)
    print(train_f3.shape)

def main():
    preprocess_step1()
    preprocess_step2()
    preprocess_step3()
    preprocess_step4()
    preprocess_step5()
    preprocess_step6()

if __name__ == '__main__':
    main()
