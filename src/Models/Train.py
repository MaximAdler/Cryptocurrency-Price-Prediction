import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import sys
import io
import urllib
import keras

sys.path.insert(0, sys.path[0]+'/../Utils')
from Plot import Plot
from NeuralNetwork import NeuralNetwork


class Train(object):
    def __init__(self):
        ''' Constructor '''

    @staticmethod
    def trainCoin(coin):
        model_path = ('assets/%s_model.h5' % coin)

        if coin == 'btc':
            start_date = '20131227'
            end_date = '20180425'
            split_date = '2018-02-25'
            window_len = 20
            epochs = 100
            batch_size = 32
            neurons_lv1 = 50
            neurons_lv2 = 25
            activation_func = 'linear'
            dropout = 0.5
            loss = 'mean_squared_error'
            optimizer = 'adam'

            btc_data = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=" + start_date + "&end=" + end_date)[0]
            btc_data = btc_data.assign(Date=pd.to_datetime(btc_data['Date']))

            print(btc_data.head())
            print('Shape: {}'.format(btc_data.shape))
            Plot.drawTrend(btc_data, "Bitcoin")
            print('\n')

            btc_data.columns = [btc_data.columns[0]] + [coin + '_' + i for i in btc_data.columns[1:]]
            kwargs = {coin + '_day_diff': lambda x: (x[coin + '_Close'] - x[coin + '_Open']) / x[coin + '_Open']}
            btc_data = btc_data.assign(**kwargs)

            print(btc_data.head())
            print('Shape: {}'.format(btc_data.shape))
            print('\n')
            Plot.drawTest(btc_data, split_date, "btc_Close", "Bitcoin")

            kwargs = {coin + '_close_off_high': lambda x: 2 * (x[coin + '_High'] - x[coin + '_Close']) / (x[coin + '_High'] - x[coin + '_Low']) -1,
                      coin + '_volatility': lambda x: (x[coin + '_High'] - x[coin + '_Low']) / (x[coin + '_Open'])}
            btc_data = btc_data.assign(**kwargs)
            model_data = btc_data[['Date'] + [coin + metric for metric in ['_Close', '_Volume', '_close_off_high', '_volatility', '_day_diff', '_Market Cap']]]
            model_data = model_data.sort_values(by='Date')

            print(model_data.head())
            print('Shape: {}'.format(model_data.shape))
            print("\n")

            training_set, test_set = model_data[model_data['Date'] < split_date], model_data[model_data['Date'] >= split_date]
            training_set = training_set.drop('Date', 1)
            test_set =  test_set.drop('Date', 1)
            normalise_cols = [coin + metric for metric in ['_Close', '_Volume', '_Market Cap']]

            LSTM_training_inputs = NeuralNetwork.normaliseInput(training_set, normalise_cols, window_len)
            LSTM_training_outputs = NeuralNetwork.normaliseOutput(training_set, coin+'_Close', window_len)
            LSTM_test_inputs = NeuralNetwork.normaliseInput(test_set, normalise_cols, window_len)
            LSTM_test_outputs = NeuralNetwork.normaliseOutput(test_set, coin+'_Close', window_len)

            print("\nNumber Of Input Training's sequences: {}".format(len(LSTM_training_inputs)))
            print("\nNumber Of Output Training's sequences: {}".format(len(LSTM_training_outputs)))
            print("\nNumber Of Input Test's sequences: {}".format(len(LSTM_test_inputs)))
            print("\nNumber Of Output Test's sequences: {}".format(len(LSTM_test_outputs)))

            LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
            LSTM_training_inputs = np.array(LSTM_training_inputs)
            LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
            LSTM_test_inputs = np.array(LSTM_test_inputs)

            model = NeuralNetwork.buildModel(LSTM_training_inputs, output_size=1, neurons_lv1=neurons_lv1,
                                             neurons_lv2=neurons_lv2, activation_func=activation_func,
                                             dropout=dropout, loss=loss, optimizer=optimizer)
            train_model =  model.fit(LSTM_training_inputs, LSTM_training_outputs, epochs=epochs,
                                     batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.2,
                                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min'),
                                                keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
            scores = model.evaluate(LSTM_test_inputs, LSTM_test_outputs, verbose=1, batch_size=batch_size)
            print('\nMSE: {}'.format(scores[1]))
            print('\nMAE: {}'.format(scores[2]))
            print('\nR^2: {}'.format(scores[3]))

            return True
