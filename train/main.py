import pandas as pd

from m_xgboost import M_XGBOOST
from m_lstm import M_LSTM

def process_lstm():
    lstminst = M_LSTM()

    # Data process
    x_train_scaled, y_train_scaled, x_test_scaled, y_test, mu_test_list, std_test_list = \
        lstminst.data_process(df, 0.2)

    # Train
    # lstminst.train(x_train_scaled, y_train_scaled)

    # Predict
    rmse, mape, est = lstminst.predict(x_test_scaled, \
                                        y_test, \
                                        mu_test_list, \
                                        std_test_list)

    # Print RMSE
    print("RMSE on test set = %0.3f" % rmse)

    # Print MAPE
    print("MAPE on test set = %0.3f%%" % mape)


def process_xbg():
    xgbinst = M_XGBOOST()

    ## feature engineering
    X_train_scaled, y_train_scaled, test, X_test_scaled, y_test = \
        xgbinst.data_process(df, 0.2)

    ## train
    # xgbinst.train(X_train_scaled, y_train_scaled)

    # predict
    rmse, mape, est = xgbinst.predict(X_test_scaled, y_test, \
                                      test['adj_close_mean'], test['adj_close_std'])

    # Print RMSE
    print("RMSE on test set = %0.3f" % rmse)

    # Print MAPE
    print("MAPE on test set = %0.3f%%" % mape)

if __name__ == "__main__":
    stk_path = "../data/VTI.csv"

    df = pd.read_csv(stk_path, sep=",")
    # process_lstm()
    process_xbg()