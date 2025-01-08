import pandas as pd
import numpy as np


def calc_statistic_for_all_columns(data):
    for column in data.colums:
        mean = column.mean()
        std = column.std()
        median = column.median()
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)
    pass


def calc_corr_between_two_columns(col1, col2):
    return col1.corr(col2)


def calc_corr_for_all(data):
    return data.corr()


def split_df_to_categorical_and_numerical(data):
    numerical_df = data.select_dtypes(include=['number'])
    categorical_df = data.select_dtypes(exclude=['number'])
    return numerical_df, categorical_df

