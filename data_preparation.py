import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def removing_missing_data(df):
    df.dropna(inplace=True)


def imputation_missing_data(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in ['i', 'u', 'f', 'c']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def data_standardization(df):
    df = df.copy()
    numerical_cols = df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


def data_normalization(df):
    df = df.copy()
    numerical_cols = df.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return pd.DataFrame(df, columns=df.columns)


def one_hot_encoding(df):
    categorical_column_names = df.select_dtypes(exclude=[np.number]).columns.tolist()
    encoded_df = pd.get_dummies(df, columns=categorical_column_names)
    return encoded_df


def label_encoding(df):
    categorical_column_names = df.select_dtypes(exclude=[np.number]).columns.tolist()
    encoded_df = df.copy()
    label_encoder = LabelEncoder()
    for col in categorical_column_names:
        encoded_df[col] = label_encoder.fit_transform(encoded_df[col])
    return encoded_df


def split_to_test_training(x, y, test_size=0.3, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def split_to_features_and_labels(df, features_name_list, labels_name):
    X = df[features_name_list]
    y = df[labels_name]
    return X, y
