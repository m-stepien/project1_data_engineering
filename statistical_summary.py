import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, \
    mean_absolute_error, r2_score


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


def classification_model_evaluation(y, y_pred, y_pred_prob):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_prob)


def regresion_model_evaluation(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
