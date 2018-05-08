#!/usr/bin/env python

import sys, getopt

sys.path.insert(0, sys.path[0]+'/src/Models')

from Train import Train
from Predict import Predict

def main(argv):
    print('\n')
    print('#'*34)
    print('# CRYPTOCURRENCY PRICE PREDICTOR #')
    print('#'*34)

    def helpMsg(err=False):
        print('\nTo train network: ')
        print('python main.py --train <cryptocurrency>')
        print('\nTo predict cryptocurrency: ')
        print('python main.py --predict <cryptocurrency>\n')
        if err:
            sys.exit(2)
        else:
            sys.exit()

    if not argv:
        helpMsg()
    train = ''
    predict = ''
    try:
        opts, args = getopt.getopt(argv,"hlt:p:",["list","train=","predict="])
    except getopt.GetoptError:
        helpMsg(True)
    for opt, arg in opts:
        if opt == '-h' or opt == '':
            helpMsg()
        elif opt in ("-l", "--list"):
            print("\nAvailable cryptocurrencies:")
            for i, c in enumerate(['BTC', 'ETH']):
                print(str(i+1) + '. ' + c)
            print("\n")
        elif opt in ("-t", "--train"):
            train = arg
            print('\nTrained cryptocurrency: %s\n' % train.upper())
            Train.trainCoin(train)
        elif opt in ("-p", "--predict"):
            predict = arg
            print('\nPredicted cryptocurrency: %s\n' % predict.upper())
            Predict.predictCoin(predict)

if __name__ == '__main__':
    main(sys.argv[1:])
