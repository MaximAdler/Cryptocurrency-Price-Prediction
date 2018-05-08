#!/usr/bin/env python

import sys, getopt

sys.path.insert(0, sys.path[0]+'/src/Models')

from Train import Train
from Predict import Predict

def helpMsg(err=False):
    print('\n$ python main.py\n')
    print('USAGE:')
    print('     python main.py command [arguments...]\n')
    print('COMMANDS:')
    print('     --help,    -h                  Show help')
    print('     --list,    -l                  Show list of available cryptocurrency')
    print('     --train,   -t <cryptocurrency> Train network')
    print('     --predict, -p <cryptocurrency> Predict cryptocurrency\n')
    if err:
        sys.exit(2)
    else:
        sys.exit()

def availableCrypto():
    print("\nAvailable cryptocurrencies:")
    for i, c in enumerate(['BTC', 'ETH']):
        print(str(i+1) + '. ' + c)
    print("\n")

def main(argv):
    print('\n')
    print('#'*34)
    print('# CRYPTOCURRENCY PRICE PREDICTOR #')
    print('#'*34)

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
            availableCrypto()
        elif opt in ("-t", "--train"):
            train = arg
            print('\nTrained cryptocurrency: %s\n' % train.upper())
            train_coin = Train.trainCoin(train)
            if not train_coin:
                print('\nNo valid cryptocurrency!')
                availableCrypto()
        elif opt in ("-p", "--predict"):
            predict = arg
            print('\nPredicted cryptocurrency: %s\n' % predict.upper())
            predict_coin = Predict.predictCoin(predict)
            if not predict_coin:
                print('\nNo valid cryptocurrency!')
                availableCrypto()

if __name__ == '__main__':
    main(sys.argv[1:])
