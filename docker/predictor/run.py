import gc
import numpy as np

from models import neural_network as nn
from utils import helprs as h

btc_data = h.parse('bitcoin', 'BTC')
eth_data = h.parse("ethereum", 'ETH')
market_data = h.join(btc_data, eth_data)

with nn.NeuralNetwork(market_data, ['BTC', 'ETH']) as prediction:
    prediction \
        .create_model() \
        .split_data() \
        .drop('Date', 1) \
        .create_inputs()

    for coin in ['BTC', 'ETC']:
        y_train, y_test = prediction.create_outputs(coin)

        gc.collect()
        np.random.seed(202)

        prediction \
            .build_model() \
            .fit(y_train, y_test) \
            .plot_results(y_train=y_train,
                          coin=coin,
                          market_data=market_data)
