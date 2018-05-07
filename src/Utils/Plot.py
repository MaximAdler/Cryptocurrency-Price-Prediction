import datetime
import matplotlib.pyplot as plt
import numpy as np


class Plot(object):
    def __init__(self):
        ''' Constructor '''

    @staticmethod
    def drawTrend(data, coin):
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[5, 1]}, figsize=(10, 10))

        ax1.set_ylabel('Closing Price ($)', fontsize=12)
        ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
        ax1.set_xticklabels('')


        ax2.set_ylabel('Volume ($ ' + coin + ')', fontsize=12)
        ax2.set_yticks([int('%d000000000' %i) for i in range(10)])
        ax2.set_yticklabels(range(10))
        ax2.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
        ax2.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y')  for i in range(2013, 2019) for j in [1, 7]])

        ax1.plot(data['Date'].astype(datetime.datetime), data['Open'])
        ax2.bar(data['Date'].astype(datetime.datetime).values, data['Volume'].values)
        fig.tight_layout()
        plt.show()
        fig.savefig("assets/" + coin + "Trend.png")

    @staticmethod
    def drawTest(data, split_date, target, coin):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
        ax.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y')  for i in range(2013, 2019) for j in [1, 7]])
        ax.plot(data[data['Date'] < split_date]['Date'].astype(datetime.datetime),
                data[data['Date'] < split_date][target],
                color='#B08FC7')
        ax.plot(data[data['Date'] >= split_date]['Date'].astype(datetime.datetime),
                data[data['Date'] >= split_date][target],
                color='#8FBAC8')
        ax.set_ylabel(coin + ' Price ($)', fontsize=12)
        plt.tight_layout()
        plt.show()
        fig.savefig("assets/" + coin + "Test.png")

    @staticmethod
    def drawError(model, train_model, coin):
        fig_err, ax = plt.subplots(1, 1)
        ax.plot(train_model.epoch, train_model.history['loss'])
        ax.set_title('Training Error')
        if model.loss == 'mae':
            ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
        else:
            ax.set_ylabel('Model Loss', fontsize=12)
        ax.set_xlabel('Epochs', fontsize=12)
        plt.show()
        fig_err.savefig("assets/" + coin + "Error")

    @staticmethod
    def drawDiff(model, model_data, test_set, LSTM_test_inputs, split_date, window_len, coin):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xticks([datetime.date(2018, i + 1, 1) for i in range(6)])
        ax.set_xticklabels([datetime.date(2018, i+1, 1).strftime('%b %d %Y') for i in range(6)])
        ax.plot(model_data[model_data['Date'] >= split_date]['Date'][window_len:].astype(datetime.datetime),
                test_set[coin + '_Close'][window_len:], label='Actual')
        ax.plot(model_data[model_data['Date'] >= split_date]['Date'][window_len:].astype(datetime.datetime),
                ((np.transpose(model.predict(LSTM_test_inputs)) + 1) * test_set[coin + '_Close'].values[:-window_len])[0],
                label='Predicted')
        ax.annotate('MAE: %.4f' % np.mean(np.abs((np.transpose(model.predict(LSTM_test_inputs)) + 1) - \
                                                 (test_set[coin + '_Close'].values[window_len:]) / (test_set[coin + '_Close'].values[:-window_len]))),
                                                 xy=(0.75, 0.9), xycoords='axes fraction',
                                                 xytext=(0.75, 0.9), textcoords='axes fraction')
        ax.set_title('Test Set: Single Timepoint Prediction', fontsize=13)
        ax.set_ylabel(coin.upper() + ' Price ($)', fontsize=12)
        ax.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
        plt.show()
        fig.savefig("assets/" + coin.upper() + "Diff.png")
