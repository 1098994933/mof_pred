elements.xlsx : 元素的特征数据
model_300dim.pkl: mol2vec 模型预训练来自kaggle：https://www.kaggle.com/code/vladislavkisin/tutorial-ml-in-chemistry-research-rdkit-mol2vec/input?select=model_300dim.pkl  
下面两个文件由于环境与当前不兼容，具体生成方法为将preprocess 中preprocess_step3()函数生成的化学式文件train_formula.csv 和 test_formula.csv 
上传到colab，运行magpie.ipynb 即可得到下面文件  
    test_formula_magpie_features.csv  
    train_formula_magpie_features.csv
requirment:
pandas==1.4.4
mordred==1.2.0
mol2vec @ git+https://github.com/samoturk/mol2vec@850d944d5f48a58e26ed0264332b5741f72555aa
numpy==1.23.3
shap==0.39.0
matplotlib==3.6.0
scikit-learn==1.1.2
scipy==1.9.1
seaborn @ file:///tmp/build/80754af9/seaborn_1608578541026/work
