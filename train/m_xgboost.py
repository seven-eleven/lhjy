import math
import numpy as np
import pandas as pd
import pickle
import utils

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
from xgboost import XGBRegressor

class M_XGBOOST(object):
    def __init__(self):
        self.N = 3  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        self.n_estimators = 20  # Number of boosted trees to fit. default = 100
        self.max_depth = 5  # Maximum tree depth for base learners. default = 3
        self.learning_rate = 0.1  # Boosting learning rate (xgb’s “eta”). default = 0.1
        self.min_child_weight = 13  # Minimum sum of instance weight(hessian) needed in a child. default = 1
        self.subsample = 1  # Subsample ratio of the training instance. default = 1
        self.colsample_bytree = 1  # Subsample ratio of columns when constructing each tree. default = 1
        self.colsample_bylevel = 1  # Subsample ratio of columns for each split, in each level. default = 1
        self.gamma = 0  # Minimum loss reduction required to make a further partition on a leaf node of the tree. default=0
        self.model_seed = 100

    def data_process(self, df, test_size):
        '''
        :param df: original data formated in pandas data frame
        :param test_size: proportion of dataset to be used as test set
        :return:
        '''
        # Convert Date column to datetime
        df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

        # Change all column headings to be lower case, and remove spacing
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

        # Get month of each sample
        df['month'] = df['date'].dt.month

        # Sort by datetime
        df.sort_values(by='date', inplace=True, ascending=True)

        print(df.head())

        # Get difference between high and low of each day
        df['range_hl'] = df['high'] - df['low']
        df.drop(['high', 'low'], axis=1, inplace=True)

        # Get difference between open and close of each day
        df['range_oc'] = df['open'] - df['close']
        df.drop(['open', 'close'], axis=1, inplace=True)

        print(df.head())

        # Add a column 'order_day' to indicate the order of the rows by date
        df['order_day'] = [x for x in list(range(len(df)))]

        # merging_keys
        merging_keys = ['order_day']

        # List of columns that we will use to create lags
        lag_cols = ['adj_close', 'range_hl', 'range_oc', 'volume']

        shift_range = [x + 1 for x in range(self.N)]

        for shift in tqdm_notebook(shift_range):
            train_shift = df[merging_keys + lag_cols].copy()

            # E.g. order_day of 0 becomes 1, for shift = 1.
            # So when this is merged with order_day of 1 in df, this will represent lag of 1.
            train_shift['order_day'] = train_shift['order_day'] + shift

            foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
            train_shift = train_shift.rename(columns=foo)

            df = pd.merge(df, train_shift, on=merging_keys, how='left')  # .fillna(0)

        del train_shift

        # Remove the first N rows which contain NaNs
        df = df[self.N:]

        print(df.head())

        cols_list = [
            "adj_close",
            "range_hl",
            "range_oc",
            "volume"
        ]

        for col in cols_list:
            df = self._get_mov_avg_std(df, col, self.N)
        df.head()

        ## split data
        # Get sizes of each of the datasets
        num_test = int(test_size * len(df))
        num_train = len(df) - num_test
        print("num_train = " + str(num_train))
        print("num_test = " + str(num_test))

        # Split into train, and test
        train = df[:num_train]
        test = df[num_train:]
        print("train.shape = " + str(train.shape))
        print("test.shape = " + str(test.shape))

        ## scale data
        cols_to_scale = [
            "adj_close"
        ]

        for i in range(1, self.N + 1):
            cols_to_scale.append("adj_close_lag_" + str(i))
            cols_to_scale.append("range_hl_lag_" + str(i))
            cols_to_scale.append("range_oc_lag_" + str(i))
            cols_to_scale.append("volume_lag_" + str(i))

        # Do scaling for train set
        # Here we only scale the train dataset, and not the entire dataset to prevent information leak
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train[cols_to_scale])
        print("scaler.mean_ = " + str(scaler.mean_))
        print("scaler.var_ = " + str(scaler.var_))
        print("train_scaled.shape = " + str(train_scaled.shape))

        # Convert the numpy array back into pandas dataframe
        train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)
        train_scaled[['date', 'month']] = train.reset_index()[['date', 'month']]
        print("train_scaled.shape = " + str(train_scaled.shape))
        print(train_scaled.head())

        # Do scaling for test set
        test_scaled = test[['date']]
        for col in tqdm_notebook(cols_list):
            feat_list = [col + '_lag_' + str(shift) for shift in range(1, self.N + 1)]
            temp = test.apply(lambda row: self._scale_row(row[feat_list], row[col + '_mean'], row[col + '_std']), axis=1)
            test_scaled = pd.concat([test_scaled, temp], axis=1)

        # Now the entire test set is scaled
        print(test_scaled.head())

        # split into x and y
        features = []
        for i in range(1, self.N + 1):
            features.append("adj_close_lag_" + str(i))
            features.append("range_hl_lag_" + str(i))
            features.append("range_oc_lag_" + str(i))
            features.append("volume_lag_" + str(i))

        target = "adj_close"

        # Split into X and y
        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]
        print("X_train.shape = " + str(X_train.shape))
        print("y_train.shape = " + str(y_train.shape))
        print("X_test.shape = " + str(X_test.shape))
        print("y_test.shape = " + str(y_test.shape))

        # Split into X and y
        X_train_scaled = train_scaled[features]
        y_train_scaled = train_scaled[target]
        X_test_scaled = test_scaled[features]
        print("X_train_scaled.shape = " + str(X_train_scaled.shape))
        print("y_train_scaled.shape = " + str(y_train_scaled.shape))
        print("X_test_scaled.shape = " + str(X_test_scaled.shape))

        return X_train_scaled, y_train_scaled, test, X_test_scaled, y_test

    def train(self, X_train_scaled, y_train_scaled):
        '''
        Train model
        Inputs
            X_train_scaled     : features for training. Scaled to have mean 0 and variance 1
            y_train_scaled     : target for training. Scaled to have mean 0 and variance 1
        '''
        model = XGBRegressor(seed=self.model_seed,
                             n_estimators=self.n_estimators,
                             max_depth=self.max_depth,
                             learning_rate=self.learning_rate,
                             min_child_weight=self.min_child_weight,
                             subsample=self.subsample,
                             colsample_bytree=self.colsample_bytree,
                             colsample_bylevel=self.colsample_bylevel,
                             gamma=self.gamma)

        # Train the regressor
        model.fit(X_train_scaled, y_train_scaled)

        self._save_model(model)

    def predict(self, X_test_scaled, y_test, col_mean, col_std):
        '''
        predict
        Inputs
            X_test_scaled      : features for test. Each sample is scaled to mean 0 and variance 1
            y_test             : target for test. Actual values, not scaled.
            col_mean           : means used to scale each sample of X_test_scaled. Same length as X_test_scaled and y_test
            col_std            : standard deviations used to scale each sample of X_test_scaled. Same length as X_test_scaled and y_test
        Outputs
            rmse               : root mean square error of y_test and est
            mape               : mean absolute percentage error of y_test and est
            est                : predicted values. Same length as y_test
        '''
        model = self._load_model()

        # Get predicted labels and scale back to original range
        est_scaled = model.predict(X_test_scaled)
        est = est_scaled * col_std + col_mean

        # Calculate RMSE
        rmse = utils.get_rmse(y_test, est)

        # Calculate MAPE
        mape = utils.get_mape(y_test, est)

        return rmse, mape, est

    def _scale_row(self, row, feat_mean, feat_std):
        """
        Given a pandas series in row, scale it to have 0 mean and var 1 using feat_mean and feat_std
        Inputs
            row      : pandas series. Need to scale this.
            feat_mean: mean
            feat_std : standard deviation
        Outputs
            row_scaled : pandas series with same length as row, but scaled
        """
        # If feat_std = 0 (this happens if adj_close doesn't change over N days),
        # set it to a small number to avoid division by zero
        feat_std = 0.001 if feat_std == 0 else feat_std

        row_scaled = (row - feat_mean) / feat_std

        return row_scaled

    def _get_mov_avg_std(self, df, col, N):
        """
        Given a dataframe, get mean and std dev at timestep t using values from t-1, t-2, ..., t-N.
        Inputs
            df         : dataframe. Can be of any length.
            col        : name of the column you want to calculate mean and std dev
            N          : get mean and std dev at timestep t using values from t-1, t-2, ..., t-N
        Outputs
            df_out     : same as df but with additional column containing mean and std dev
        """
        mean_list = df[col].rolling(window=N, min_periods=1).mean()  # len(mean_list) = len(df)
        std_list = df[col].rolling(window=N, min_periods=1).std()  # first value will be NaN, because normalized by N-1

        # Add one timestep to the predictions
        mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
        std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))

        # Append mean_list to df
        df_out = df.copy()
        df_out[col + '_mean'] = mean_list
        df_out[col + '_std'] = std_list

        return df_out

    def _save_model(self, model):
        '''
        Save model using pickle
        :param model: xgboost model to be saved
        :return: None
        '''
        pickle.dump(model, open("../model/xgboost.dat", "wb"))

    def _load_model(self):
        '''
        Load model using pickle
        :return: xgboost model
        '''
        model = pickle.load(open("../model/xgboost.dat", "rb"))
        return model












