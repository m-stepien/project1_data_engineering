import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, \
    mean_absolute_error, r2_score


def calc_statistic_for_all_columns(data):
    stats = {}
    stats_categorical = {}
    for column in data.select_dtypes(include=['number']).columns:
        stats[column] = {
            'mean': data[column].mean(),
            'std': data[column].std(),
            'median': data[column].median(),
            'q1': data[column].quantile(0.25),
            'q3': data[column].quantile(0.75)
        }
    for column in data.select_dtypes(include=['object', 'category']).columns:
        stats_categorical[column] = {
            'unique_values': data[column].nunique(),
            'most_common': data[column].mode()[0] if not data[column].mode().empty else None,
        }
    return pd.DataFrame(stats).T, pd.DataFrame(stats_categorical).T


def calc_corr_between_two_columns(col1, col2):
    return col1.corr(col2)


def calc_corr_for_all(data):
    return split_df_to_categorical_and_numerical(data)[0].corr()


def split_df_to_categorical_and_numerical(data):
    numerical_df = data.select_dtypes(include=['number'])
    categorical_df = data.select_dtypes(exclude=['number'])
    return numerical_df, categorical_df


def classification_model_evaluation(y, y_pred, y_pred_prob):
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, average="weighted"),
        "Recall": recall_score(y, y_pred, average="weighted"),
        "F1 Score": f1_score(y, y_pred, average="weighted"),
        "AUC": roc_auc_score(y, y_pred_prob, multi_class="ovr")
    }
    return pd.DataFrame(metrics, index=["Score"])


def regresion_model_evaluation(y, y_pred):
    metrics = {
        "MSE": mean_squared_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "MAE": mean_absolute_error(y, y_pred),
        "R2": r2_score(y, y_pred)
    }
    return pd.DataFrame(metrics, index=["Score"]).T
