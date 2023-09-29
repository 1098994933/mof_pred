from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from config import config
def main():
    dataset = pd.read_csv("../data/RAC_train.csv")
    category_targets = ['additive_category']
    features = list(dataset.columns[1:-12])
    X = dataset[features]
    for Y_col in category_targets:
        Y = dataset[Y_col]
        model_final = DecisionTreeClassifier(random_state=6)
        # evaluate
        cv_scores = cross_val_score(model_final, X, Y, cv=30, scoring='f1_weighted')
        y_predict_cv = cross_val_predict(model_final, X, Y, cv=30)
        score = np.mean(cv_scores)
        score_std = np.std(cv_scores)
        score_max = np.max(cv_scores)
        score_min = np.min(cv_scores)
        print(f"Y_col: {Y_col} , score:{score:3f} "
              f"{score - 2 * score_std:3f}~{score + 2 * score_std:3f}  "
              f"min:{score_min:3f} max:{score_max:3f}",
              )
        model_final.fit(X, Y)
        # save model
        with open(f'../data/rac_model_{Y_col}.pkl', 'wb') as file:
            pickle.dump(model_final, file)
        dataset = pd.read_csv("../data/RAC_test.csv")
        targets = ['temperature', 'time', 'param1', 'param2', 'param3', 'param4', 'param5']
        predict_df = pd.DataFrame()
        predict_df['mof'] = list(range(1, len(dataset) + 1))
        features = list(dataset.columns[1:])
        X = dataset[features]
        # predict
        category_targets = ['additive_category']
        for Y_col in category_targets:
            # load model
            with open(f'../data/rac_model_{Y_col}.pkl', 'rb') as file:
                model_final = pickle.load(file)
                predict_df[Y_col] = model_final.predict(X)
        predict_df.rename(columns={'additive_category': 'additive'}, inplace=True)
        predict_df.to_csv(f"../data/rac_{Y_col}.csv", index=False)
        #print(predict_df)
if __name__ == '__main__':
    main()
