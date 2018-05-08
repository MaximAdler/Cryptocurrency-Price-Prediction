from scipy.stats.stats import pearsonr
from keras.models import load_model
import pandas as pd
import numpy as np
import os
import urllib
import time

from NeuralNetwork import NeuralNetwork


class Predict(object):
    def __init__(self):
        ''' Constructor '''

    @staticmethod
    def predictCoin(coin):
        coin = coin.lower()
        model_path = ('assets/%s_model.h5' % coin)
        predictions = []
        window_len = 20
        start_dates = ['20180408', '20180409', '20180410', '20180411', '20180412', '20180413', '20180414', '20180415']
        end_dates = ['20180428', '20180429', '20180430', '20180501', '20180502', '20180503', '20180504', '20180505']

        if coin == 'btc':
            real_currency = [9419.08, 9240.55, 9119.01, 9235.92, 9743.86, 9700.76, 9858.15, 9654.80]
            coin_url = 'https://coinmarketcap.com/currencies/bitcoin/historical-data/?start='
        elif coin == 'eth':
            real_currency = [688.88, 669.92, 673.61, 687.15, 779.54, 785.62, 816.12, 792.31]
            coin_url = 'https://coinmarketcap.com/currencies/ethereum/historical-data/?start='
        else:
            return False
        
        if os.path.isfile(model_path):
            for start_date, end_date in zip(start_dates, end_dates):
                time.strftime("%Y%m%d")
                crypto_data = pd.read_html(coin_url + start_date + "&end=" + end_date)[0]
                crypto_data = crypto_data.assign(Date=pd.to_datetime(crypto_data['Date']))

                crypto_data.columns = [crypto_data.columns[0]] + [coin + '_' + i for i in crypto_data.columns[1:]]
                kwargs = {coin + '_day_diff': lambda x: (x[coin + '_Close'] - x[coin + '_Open']) / x[coin + '_Open']}
                crypto_data = crypto_data.assign(**kwargs)

                kwargs = {coin + '_close_off_high': lambda x: 2 * (x[coin + '_High'] - x[coin + '_Close']) / (x[coin + '_High'] - x[coin + '_Low']) - 1,
                          coin + '_volatility': lambda x: (x[coin + '_High'] - x[coin + '_Low']) / (x[coin + '_Open'])}
                crypto_data = crypto_data.assign(**kwargs)
                model_data = crypto_data[['Date'] + [coin + metric for metric in ['_Close', '_Volume', '_close_off_high', '_volatility', '_day_diff', '_Market Cap']]]
                model_data = model_data.sort_values(by='Date')

                model_data = model_data.drop('Date', 1)
                normalised_cols = [coin + metric for metric in ['_Close', '_Volume', '_Market Cap']]

                LSTM_test_inputs = NeuralNetwork.normaliseInput(model_data, normalised_cols, window_len)
                LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
                LSTM_test_inputs = np.array(LSTM_test_inputs)

                estimator = load_model(model_path, custom_objects={'r2': NeuralNetwork.r2})
                print((((np.transpose(estimator.predict(LSTM_test_inputs)) + 1) * model_data[coin + '_Close'].values[:-window_len])[0])[0])
                predictions.append((((np.transpose(estimator.predict(LSTM_test_inputs)) + 1) * model_data[coin + '_Close'].values[:-window_len])[0])[0])

            print(predictions)
            print(pearsonr(predictions, real_currency))
        else:
            print 'Firstly, train the Network!\n'

        return True
