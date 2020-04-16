import numpy as np
import pandas as pd
import utils

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM

class M_LSTM(object):
    def __init__(self):
        self.N = 3              # for feature at day t, we use lags from t-1, t-2, ..., t-N as features.
        self.lstm_units=128     # lstm param.
        self.dropout_prob=1     # lstm param.
        self.optimizer='nadam'  # lstm param.
        self.epochs=50          # lstm param.
        self.batch_size=8       # lstm param.
        self.model_seed = 100

    def train(self, x_train_scaled, y_train_scaled):
        '''
        Train model
        Inputs
            x_train_scaled  : e.g. x_train_scaled.shape=(451, 9, 1). Here we are using the past 9 values to predict the next value
            y_train_scaled  : e.g. y_train_scaled.shape=(451, 1)
        '''
        # Create the LSTM network
        model = Sequential()
        model.add(LSTM(units=self.lstm_units, return_sequences=True, input_shape=(x_train_scaled.shape[1], 1)))
        model.add(Dropout(self.dropout_prob))  # Add dropout with a probability of 0.5
        model.add(LSTM(units=self.lstm_units))
        model.add(Dropout(self.dropout_prob))  # Add dropout with a probability of 0.5
        model.add(Dense(1))

        # Compile and fit the LSTM network
        model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        model.fit(x_train_scaled, y_train_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        # Print model summary
        model.summary()

        # Save model
        self._save_model(model)

    def predict(self, \
              x_cv_scaled, \
              y_cv, \
              mu_cv_list, \
              std_cv_list):
        '''
        Train model, do prediction, scale back to original range and do evaluation
        Use LSTM here.
        Returns rmse, mape and predicted values
        Inputs
            x_cv_scaled     : use this to do predictions
            y_cv            : actual value of the predictions
            mu_cv_list      : list of the means. Same length as x_scaled and y
            std_cv_list     : list of the std devs. Same length as x_scaled and y
        Outputs
            rmse            : root mean square error
            mape            : mean absolute percentage error
            est             : predictions
        '''
        model = self._load_model()

        # Do prediction
        est_scaled = model.predict(x_cv_scaled)
        est = (est_scaled * np.array(std_cv_list).reshape(-1, 1)) + np.array(mu_cv_list).reshape(-1, 1)

        # Calculate RMSE and MAPE
        rmse = utils.get_rmse(y_cv, est)
        mape = utils.get_mape(y_cv, est)

        return rmse, mape, est

    def _save_model(self, model):
        model.save('../model/lstm.h5')

    def _load_model(self):
        model = load_model('../model/lstm.h5')
        return model

    def data_process(self, df, test_size):
        '''

        :param df:
        :param test_size:
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

        df.head()

        # Split into train and test sets
        # Get sizes of each of the datasets
        num_test = int(test_size * len(df))
        num_train = len(df) - num_test
        print("num_train = " + str(num_train))
        print("num_test = " + str(num_test))

        # Split into train, and test
        train = df[:num_train][['date', 'adj_close']]
        test = df[num_train:][['date', 'adj_close']]

        print("train.shape = " + str(train.shape))
        print("test.shape = " + str(test.shape))

        # Converting dataset into x_train and y_train
        # Here we only scale the train dataset, and not the entire dataset to prevent information leak
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(np.array(train['adj_close']).reshape(-1, 1))
        print("scaler.mean_ = " + str(scaler.mean_))
        print("scaler.var_ = " + str(scaler.var_))

        # Split into x and y
        x_train_scaled, y_train_scaled = self._get_x_y(train_scaled, self.N, self.N)
        print("x_train_scaled.shape = " + str(x_train_scaled.shape))  # (446, 7, 1)
        print("y_train_scaled.shape = " + str(y_train_scaled.shape))  # (446, 1)

        # Scale the test dataset
        # Split into x and y
        x_test_scaled, y_test, mu_test_list, std_test_list = self._get_x_scaled_y(np.array(df['adj_close']).reshape(-1, 1), self.N,
                                                                    num_train)
        print("x_test_scaled.shape = " + str(x_test_scaled.shape))
        print("y_test.shape = " + str(y_test.shape))
        print("len(mu_test_list) = " + str(len(mu_test_list)))
        print("len(std_test_list) = " + str(len(std_test_list)))

        return x_train_scaled, y_train_scaled, x_test_scaled, y_test, mu_test_list, std_test_list

    def _get_x_y(self, data, N, offset):
        """
        Split data into x (features) and y (target)
        """
        x, y = [], []
        for i in range(offset, len(data)):
            x.append(data[i - N:i])
            y.append(data[i])
        x = np.array(x)
        y = np.array(y)

        return x, y

    def _get_x_scaled_y(self, data, N, offset):
        """
        Split data into x (features) and y (target)
        We scale x to have mean 0 and std dev 1, and return this.
        We do not scale y here.
        Inputs
            data     : pandas series to extract x and y
            N
            offset
        Outputs
            x_scaled : features used to predict y. Scaled such that each element has mean 0 and std dev 1
            y        : target values. Not scaled
            mu_list  : list of the means. Same length as x_scaled and y
            std_list : list of the std devs. Same length as x_scaled and y
        """
        x_scaled, y, mu_list, std_list = [], [], [], []
        for i in range(offset, len(data)):
            mu_list.append(np.mean(data[i - N:i]))
            std_list.append(np.std(data[i - N:i]))
            x_scaled.append((data[i - N:i] - mu_list[i - offset]) / std_list[i - offset])
            y.append(data[i])
        x_scaled = np.array(x_scaled)
        y = np.array(y)

        return x_scaled, y, mu_list, std_list


