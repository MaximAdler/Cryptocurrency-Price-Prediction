# import gc
# import numpy as np
#
# from models import neural_network as nn
# from utils import helpers as h
#
# btc_data = h.parse('bitcoin', 'BTC')
# eth_data = h.parse("ethereum", 'ETH')
# market_data = h.join(btc_data, eth_data)
# m = nn.NeuralNetwork(market_data, ['BTC', 'ETH'])
# m.create_model()
# m.split_data()
# m.drop('Date', 1)
# m.create_inputs()
# y_train_btc, y_test_btc = m.create_outputs('BTC')
#
#
# gc.collect()
# np.random.seed(202)
# m.build_model()
# btc_history = m.fit(y_train_btc, y_test_btc)
import wget

print('Beginning file download with wget module')

url = 'https://tpc.googlesyndication.com/simgad/3519940189745092650'
wget.download(url, '/opt/predictor/iamg.jpg')
