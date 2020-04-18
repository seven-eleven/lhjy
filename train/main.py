import pandas as pd
import sys

from m_xgboost import M_XGBOOST
from m_lstm import M_LSTM

def process_lstm(train):
    lstminst = M_LSTM()

    ## Data process
    x_train_scaled, y_train_scaled, x_test_scaled, y_test, mu_test_list, std_test_list = \
        lstminst.data_process(df, 0.2)

    ## Train
    if train:
        print("start to train")
        lstminst.train(x_train_scaled, y_train_scaled)

    ## Predict
    rmse, mape, est = lstminst.predict(x_test_scaled, \
                                        y_test, \
                                        mu_test_list, \
                                        std_test_list)

    ## Evaluate
    print("RMSE on test set = %0.3f" % rmse) # RMSE
    print("MAPE on test set = %0.3f%%" % mape) # MAPE
    lstminst.draw(y_test, est)

    ## Profits
    # print(type(y_test))
    # print(type(est))
    lstminst.profit(y_test, est, mape)

def process_xbg(train):
    xgbinst = M_XGBOOST()

    ## Data Process
    X_train_scaled, y_train_scaled, test, X_test_scaled, y_test = \
        xgbinst.data_process(df, 0.2)

    ## Train
    if train:
        print("start to train")
        xgbinst.train(X_train_scaled, y_train_scaled)

    ## Predict
    rmse, mape, est = xgbinst.predict(X_test_scaled, y_test, \
                                      test['close_mean'], test['close_std'])

    ## Evaluate
    print("RMSE on test set = %0.3f" % rmse) # RMSE
    print("MAPE on test set = %0.3f%%" % mape) # MAPE
    xgbinst.draw(y_test, est)

    ## Profits
    # print(y_test.head())
    # print(est.head())
    xgbinst.profit(y_test, est, mape)

def help():
    print("run command: python main.py {model} {train} {data}\n" + \
          "{model} input: 'xgboost' or 'lstm'\n" + \
          "{train} input: 'train' or other else\n" + \
          "{data} input the data file path")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        help()
        exit(-1)

    model = sys.argv[1]
    train = sys.argv[2]
    data = sys.argv[3] # data = "../data/VTI.csv"
    df = pd.read_csv(data, sep=",")

    with_train = True if train == "train" else False
    if model == "lstm":
        process_lstm(with_train)
    elif model == "xgboost":
        process_xbg(with_train)
    else:
        help()
