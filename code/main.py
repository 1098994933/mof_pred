import rac_additive
import rac_sov
import rac_temperature
import pandas as pd
import preprocessing
import finger_temperature_predict
import finger_time_predict

if __name__ == '__main__':
    # finger prediction
    preprocessing.main()
    finger_time_predict.main()
    finger_temperature_predict.main()

    df1 = pd.read_csv('../data/finger_temperature.csv')
    df2 = pd.read_csv('../data/finger_time.csv')
    df2 = df2[df2.columns[1:]]
    df_submit = pd.concat([df1, df2], axis=1)
    df_submit.to_csv("../submit/finger_prediction.csv", index=False)

    # rac prediction
    rac_additive.main()
    rac_sov.main()
    rac_temperature.main()

    # merge prediction to submit
    df1 = pd.read_csv("../data/rac_prediction.csv")
    df2 = pd.read_csv("../data/rac_sov.csv")
    df2 = df2[df2.columns[1:]]
    df3 = pd.read_csv("../data/rac_additive_category.csv")
    df3 = df3[df3.columns[1:]]
    rac_submit = pd.concat([df1, df2, df3], axis=1)
    rac_submit.to_csv("../submit/RAC_prediction.csv", index=False)
    #print(rac_submit)

