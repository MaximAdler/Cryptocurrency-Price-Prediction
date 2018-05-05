import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import sys
import io
import urllib
import keras
from PIL import Image

sys.path.insert(0, sys.path[0]+'/../Utils')
from Plot import Plot
from NeuralNetwork import NeuralNetwork


class Train(object):
    def __init__(self):
        ''' Constructor '''

    @staticmethod
    def trainCoin(coin):
        model_path = ('../../assets/%s_model.h5' % coin)

        if coin == 'btc':
            start_date = '20131227'
            end_date = '20180425'
            split_date = '2018-02-25'
            window_len = 20
            btc_epochs = 100
            btc_batch_size = 32
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
            model_data = btc_data[['Date'] + [coin + metric for metric in ['_close', '_volume', '_close_off_high', '_volatility', '_day_diff', '_market_cap']]]
            model_data = model_data.sort_values(by='Date')

            print(model_data.head())
            print('Shape: {}'.format(model_data.shape))
            print("\n")

            traning_set, test_set = model_data[model_data['Date'] < split_date], model_data[model_data['Date'] >= split_date]
            training_set =  training_set.drop('Date', 1)
            test_set =  test_set.drop('Date', 1)
            


            return True
