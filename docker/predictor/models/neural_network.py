import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

from utils import helpers
from utils import constants as const

"""
btc_data = h.parse('bitcoin', 'BTC')
eth_data = h.parse("ethereum", 'ETH')
market_data = h.join(btc_data, eth_data)
m = nn.NeuralNetwork(market_data, ['BTC', 'ETH'])
m.create_model()
m.split_data()
m.drop('Date', 1)
m.create_inputs()
y_train_btc, y_test_btc = m.create_outputs('BTC')
#
import gc
gc.collect()
np.random.seed(202)
m.build_model()
btc_history = m.fit(y_train_btc, y_test_btc)
"""

class NeuralNetwork:

    def __init__(self, data: pd.DataFrame, coins: list, date_column: str='Date') -> None:
        self.data = data
        self.coins = coins
        self.date_column = date_column
        self.train_set = None
        self.test_set = None
        self.x_train = None
        self.x_test = None
        self.model = None
        self.fitted_model = None

    def add_volatility(self) -> 'NeuralNetwork':
        for coin in self.coins:
            kwargs = {
              '{}_change'.format(coin): lambda x:  \
                  (x['{}_Close**'.format(coin)] - x['{}_Open*'.format(coin)]) / x['{}_Open*'.format(coin)],
              '{}_close_off_high'.format(coin): lambda x: \
                  2 * (x['{}_High'.format(coin)] - x['{}_Close**'.format(coin)]) / (x['{}_High'.format(coin)] - x['{}_Low'.format(coin)]) - 1,
              '{}_volatility'.format(coin): lambda x: \
                  (x['{}_High'.format(coin)] - x['{}_Low'.format(coin)]) / (x['{}_Open*'.format(coin)])
            }
            self.data = self.data.assign(**kwargs)
        return self

    # TODO: delete (prev: create_model_data)
    def create_model(self,
                     metrics: list=['Close**','Volume']) -> 'NeuralNetwork':
        self.data = self.data[[self.date_column] + ['{}_{}'.format(coin, metric) for coin in self.coins for metric in metrics]]
        self.data = self.data.sort_values(by='Date')
        return self


    def add_coin(self, *args) -> 'NeuralNetwork':
        self.coins += [*args]
        return self


    def split_data(self, training_size: float=const.TRAINING_SIZE) -> 'NeuralNetwork':
        self.train_set = self.data[:int(training_size*len(self.data))]
        self.test_set = self.data[int(training_size*len(self.data)):]
        return self

    def drop(self, column: str, axis: int=1, splitted_set: bool=True) -> 'NeuralNetwork':
        if splitted_set:
            self.train_set = self.train_set.drop(column, axis)
            self.test_set = self.test_set.drop(column, axis)
        else:
            self.data = self.data.drop(column, axis)
        return self


    # TODO: delete (
    #    X_train = create_inputs(train_set)
    #    X_test = create_inputs(test_set)
    #    X_train, X_test = to_array(X_train), to_array(X_test)
    # )
    def create_inputs(self,
                      metrics: list=['Close**', 'Volume'],
                      window_len: int=const.WINDOW_LEN) -> 'NeuralNetwork':
        norm_cols = ['{}_{}'.format(coin, metric) for coin in self.coins for metric in metrics]
        results = []
        for data in [self.train_set, self.test_set]:
            inputs = []
            for i in range(len(data) - window_len):
                temp_set = data[i:(i + window_len)].copy()
                inputs.append(temp_set)
                for col in norm_cols:
                    inputs[i].loc[:, col] = inputs[i].loc[:, col] / inputs[i].loc[:, col].iloc[0] - 1
            results.append(helpers.to_array(inputs))
        self.x_train, self.x_test = results
        return self

    # TODO: delete (
    #    Y_train_btc = create_outputs(train_set, coin='BTC')
    #    Y_test_btc = create_outputs(test_set, coin='BTC')
    # )
    def create_outputs(self, coin: str, window_len: int=const.WINDOW_LEN) -> list:
        results = []
        for data in [self.train_set, self.test_set]:
            results.append(
                (data['{}_Close**'.format(coin)][window_len:].values / \
                data['{}_Close**'.format(coin)][:-window_len].values) - 1
            )
        return results


    def build_model(self, output_size: int=1, neurons: int=const.NEURONS,
                    activ_func: str=const.ACTIVATION_FUNCTION,
                    dropout: float=const.DROPOUT, loss: str=const.LOSS,
                    optimizer: str=const.OPTIMIZER) -> 'NeuralNetwork':
        self.model = Sequential()
        self.model.add(LSTM(neurons, return_sequences=True, input_shape=(self.x_train.shape[1], self.x_train.shape[2]), activation=activ_func))
        self.model.add(Dropout(dropout))
        self.model.add(LSTM(neurons, return_sequences=True, activation=activ_func))
        self.model.add(Dropout(dropout))
        self.model.add(LSTM(neurons, activation=activ_func))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(units=output_size))
        self.model.add(Activation(activ_func))
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
        self.model.summary()
        return self


    def fit(self, y_train: pd.DataFrame, y_test: pd.DataFrame,
            epochs:int=const.EPOCHS, batch_size: int=const.BATCH_SIZE,
            verbose: int=1, shuffle: bool=False) -> pd.DataFrame:

        self.fitted_model = self.model.fit(self.x_train, y_train, epochs=epochs,
                                           batch_size=batch_size, verbose=verbose,
                                           validation_data=(self.x_test, y_test), shuffle=shuffle)
        return self

    # def plot_results(self, model, Y_target, coin, X_train, test_set, X_test, market_data):
    #     plt.figure(figsize=(25, 20))
    #     plt.subplot(311)
    #     plt.plot(self.fitted_model.epoch, self.fitted_model.history['loss'], )
    #     plt.plot(self.fitted_model.epoch, self.fitted_model.history['val_loss'])
    #     plt.xlabel('Number of Epochs')
    #     plt.ylabel('Loss')
    #     plt.title(coin + ' Model Loss')
    #     plt.legend(['Training', 'Test'])
    #     plt.subplot(312)
    #     plt.plot(Y_target)
    #     plt.plot(model.predict(X_train))
    #     plt.xlabel('Dates')
    #     plt.ylabel('Price')
    #     plt.title(coin + ' Single Point Price Prediction on Training Set')
    #     plt.legend(['Actual','Predicted'])
    #     ax1 = plt.subplot(313)
    #     plt.plot(test_set[coin + '_Close**'][window_len:].values.tolist())
    #     plt.plot(((np.transpose(model.predict(X_test)) + 1) * test_set[coin + '_Close**'].values[:-window_len])[0])
    #     plt.xlabel('Dates')
    #     plt.ylabel('Price')
    #     plt.title(coin + ' Single Point Price Prediction on Test Set')
    #     plt.legend(['Actual','Predicted'])
    #     date_list = h.date_labels(market_data, X_test)
    #     ax1.set_xticks([x for x in range(len(date_list))])
    #     for label in ax1.set_xticklabels([date for date in date_list], rotation='vertical')[::2]:
    #     label.set_visible(False)
    #     plt.show()
    #
    #     plot_results(btc_history, m.model, y_train_btc, coin='BTC')
