# **Cryptocurrency Price Predictor**

### What is this?
The Neural Network for Cryptocurrency.

## Setup guide
```shell
git clone https://github.com/MaximAdler/Cryptocurrency-Price-Prediction.git

cd Cryptocurrency-Price-Prediction

virtualenv venv

source venv/bin/activate

pip install -r requirements.txt

python main.py <command>
```

## Command line options
```shell
$ python main.py

USAGE:
     python main.py command [arguments...]

COMMANDS:
     --help,    -h                  Show help
     --list,    -l                  Show list of available cryptocurrency
     --train,   -t <cryptocurrency> Train network
     --predict, -p <cryptocurrency> Predict cryptocurrency
```
## Available Cryptocurrencies:
- **BTC**
- **ETH**
