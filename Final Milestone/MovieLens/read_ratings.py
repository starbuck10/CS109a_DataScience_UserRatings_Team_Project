from datetime import datetime

import pandas as pd

from common import elapsed_time


def date_parse(time_in_secs):
    return datetime.utcfromtimestamp(float(time_in_secs))


def read_ratings_df_with_timestamp(file_name):
    with elapsed_time('loaded csv'):
        ratings_df = pd.read_csv(file_name, parse_dates=['timestamp'], date_parser=date_parse)
    return ratings_df


def read_ratings_df(file_name):
    with elapsed_time('loaded csv'):
        ratings_df = pd.read_csv(file_name)
    return ratings_df
