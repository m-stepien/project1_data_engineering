import pandas as pd
import numpy as np


def calc_statistic_for_all_columns(data):
    for column in data.colums:
        if is_numerical(column):
            mean = column.mean()
            std = column.std()
            median = column.median()
            q1 = column.quantile(0.25)
            q3 = column.quantile(0.75)
        else:
            return "kolumna zawiera dane nie numeryczne"
    pass


def calc_corr_between_two_columns(col1, col2):
    return col1.corr(col2)

def calc_corr_for_all(data):
    return data.corr()


def is_numerical(data):
    pass
