import time
import datetime

import pandas as pd
import numpy as np

from utils import constants as const


# TODO: delete (prev: get_market_data)
def parse(market: str,
          tag: str,
          date_column: str='Date',
          volume_column: str='Volume') -> pd.DataFrame:

    df = pd.read_html(
        'https://coinmarketcap.com/currencies/{}/historical-data/?start=20130428&end={}'.format(
            market, time.strftime('%Y%m%d')
        ), flavor='html5lib')[0]
    df = df.assign(Date=pd.to_datetime(df[date_column]))
    df[volume_column] = pd.to_numeric(df[volume_column], errors='coerce').fillna(0)
    df.columns = [df.columns[0]] + ['{}_{}'.format(tag, _col) for _col in df.columns[1:]]
    return df


# TODO: delete (prev: merge_data)
def join(df1: pd.DataFrame,
         df2: pd.DataFrame,
         date_column: str='Date',
         start_date: str=const.MERGE_DATE) -> pd.DataFrame:
  df = pd.merge(df1, df2, on=[date_column])
  df = df[df[date_column] >= start_date]
  return df


def to_array(data: list) -> np.array:
  return np.array([np.array(data[i]) for i in range(len(data))])


def date_labels(market_data: pd.DataFrame, x_test: pd.DataFrame) -> list:
    last_date = market_data.iloc[0, 0]
    date_list = [last_date - datetime.timedelta(days=x) for x in range(len(x_test))]
    return[date.strftime('%m/%d/%Y') for date in date_list][::-1]
