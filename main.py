#!/usr/bin/env python

import sys

sys.path.insert(0, sys.path[0]+'/src/Models')
from Train import Train

if __name__ == '__main__':
    print Train.trainCoin('eth')
    # print Train.trainCoin('btc')
