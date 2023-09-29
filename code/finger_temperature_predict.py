import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from config import config


def main():
    df = pd.read_csv("../data/finger_train.csv")
    Y_col = 'temperature'
    X = df.iloc[:, 1:-5]
    Y = df.iloc[:, -2]
    ml_dataset = pd.concat([X, Y], axis=1)
    ml_dataset.to_csv(f"../data/ml_dataset_{Y_col}.csv", index=False)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("root Mean Squared Error:", mse**0.5)
    print("R-squared:", r2)

    df_test = pd.read_csv(config['test_data_finger'])
    X_test = df_test[X.columns]
    y_submit = model.predict(X_test)
    predict_df = pd.DataFrame()
    predict_df['mof'] = list(range(1, len(df_test) + 1))

    predict_df[Y_col] = y_submit
    predict_df.to_csv(f"../data/finger_{Y_col}.csv", index=False)


if __name__ == '__main__':
    main()
