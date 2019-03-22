from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout
from keras import metrics
import keras.backend as Backend
import statsmodels.api as sm
from pandas import DataFrame
from sklearn.metrics import r2_score
import ml_metrics as metrics


class NeuralNetwork(object):
    def __init__(self):
        ''' Constructor '''

    @staticmethod
    def r2(y_true, y_pred):
        square_sum = Backend.sum(Backend.square(y_true - y_pred))
        total_square_sum = Backend.sum(Backend.square(y_true - Backend.mean(y_true)))
        return (1 - square_sum / (total_square_sum + Backend.epsilon()))

    @staticmethod
    def arima(data):
        model = sm.tsa.ARIMA(data, order=(1, 1, 1)).fit(disp=0)
        print model.summary()
        q_test = sm.tsa.stattools.acf(model.resid, qstat=True)
        print DataFrame({'Q-stat':q_test[1], 'p-value':q_test[2]})

        pred = model.predict(1567, 1587, typ='levels')
        trn = data[1567:]
        r2 = r2_score(trn, pred[1:21])
        print 'R^2: %1.2f' % r2
        print metrics.rmse(trn,pred[1:21])
        print metrics.mae(trn,pred[1:21])

        return '\nARIMA Success'

    @staticmethod
    def normaliseInput(dataset, cols, window_len):
        LSTM_inputs = []
        for i in range(len(dataset) - window_len):
            temp_set = dataset[i:(i + window_len)].copy()
            for col in cols:
                temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
            LSTM_inputs.append(temp_set)
        return LSTM_inputs

    @staticmethod
    def normaliseOutput(dataset, target, window_len):
        return (dataset[target][window_len:].values / dataset[target][:-window_len].values) - 1

    @staticmethod
    def buildModel(inputs, output_size, neurons_lv1, neurons_lv2, activation_func="linear",
                   dropout=0.3, loss="mean_squared_error", optimizer="adam"):
        model = Sequential()
        model.add(LSTM(input_shape=(inputs.shape[1],inputs.shape[2]),
                       units=neurons_lv1,
                       return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=neurons_lv2,
                       return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(units=output_size))
        model.add(Activation(activation_func))
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics.mse, metrics.mae, NeuralNetwork.r2])

        print(model.summary())
        return model
