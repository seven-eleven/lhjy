import pandas as pd
import sys

from m_xgboost import M_XGBOOST
from m_lstm import M_LSTM

def process_lstm():
    lstminst = M_LSTM()

    ## Data process
    x_train_scaled, y_train_scaled, x_test_scaled, y_test, mu_test_list, std_test_list = \
        lstminst.data_process(df, 0.2)

    ## Train
    # lstminst.train(x_train_scaled, y_train_scaled)

    ## Predict
    rmse, mape, est = lstminst.predict(x_test_scaled, \
                                        y_test, \
                                        mu_test_list, \
                                        std_test_list)

    ## Evaluate
    print("RMSE on test set = %0.3f" % rmse) # RMSE
    print("MAPE on test set = %0.3f%%" % mape) # MAPE

    ## Profits
    # print(type(y_test))
    # print(type(est))
    lstminst.profit(y_test, est, mape)

def process_xbg():
    xgbinst = M_XGBOOST()

    ## Data Process
    X_train_scaled, y_train_scaled, test, X_test_scaled, y_test = \
        xgbinst.data_process(df, 0.2)

    ## Train
    # xgbinst.train(X_train_scaled, y_train_scaled)

    ## Predict
    rmse, mape, est = xgbinst.predict(X_test_scaled, y_test, \
                                      test['adj_close_mean'], test['adj_close_std'])

    ## Evaluate
    print("RMSE on test set = %0.3f" % rmse) # RMSE
    print("MAPE on test set = %0.3f%%" % mape) # MAPE

    ## Frofit
    # print(y_test.head())
    # print(est.head())
    xgbinst.profit(y_test, est, mape)

def help():
    print("run command: python main.py {model_name}\n model_name support xgboost and lstm\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        help()
        exit(-1)

    stk_path = "../data/VTI.csv"
    df = pd.read_csv(stk_path, sep=",")

    if sys.argv[1] == "lstm":
        process_lstm()
    elif sys.argv[1] == "xgboost":
        process_xbg()
    else:
        help()
