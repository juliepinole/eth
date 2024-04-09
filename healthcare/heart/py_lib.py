import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import namedtuple
import torch
import torch.nn.functional as F


def standardize_features(
    df: pd.DataFrame,
    cols_to_standardize: list = None,
):
    scaler = MinMaxScaler()
    if cols_to_standardize is None:
        untouched_columns = list(df.columns)
    else:
        untouched_columns = [col for col in df.columns if col not in cols_to_standardize]
    df_standardized = scaler.fit_transform(df[cols_to_standardize].copy())
    return pd.concat(
        [df[untouched_columns], pd.DataFrame(df_standardized, columns=cols_to_standardize)],
        axis=1
        )

def adding_one_hot_encoded(
        df: pd.DataFrame,
        cols_obj_pure: list,
        features: list = None,
        # drop_one_category: bool = False,
        ):
    df_hot = pd.get_dummies(df, columns=cols_obj_pure)
    all_features = list(df_hot.columns)
    categorical_features = list(set(all_features) - set(features))
    return df_hot, categorical_features, all_features


def pre_process_features(
        df: pd.DataFrame(),
        num_features: list,
        categorical_features: list,
        label_col: str = 'HeartDisease',
        add_one_hot_encoded: bool = False,
        stand_features: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
):
    # Step 1: Adding One-Hot_Encoded columns
    if add_one_hot_encoded:
        df_x, categorical_features, all_features = adding_one_hot_encoded(df[num_features + categorical_features], categorical_features, num_features)
    else:
        df_x = df[num_features].copy()
        all_features = num_features

    # Step 2: Standardizing features
    if stand_features:
        df_x = standardize_features(df_x, cols_to_standardize=all_features)
    
    # Step 3: Split test/ train
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(
    df_x, df[label_col], test_size=test_size, random_state=random_state)
    data_dataframes = {
        'X_train': X_train_0,
        'X_test': X_test_0,
        'y_train': y_train_0,
        'y_test': y_test_0,
    }

    # Step 4: Create Tensors
    X_train, X_test = torch.Tensor(X_train_0.to_numpy()),torch.Tensor(X_test_0.to_numpy())
    y_train, y_test = torch.Tensor(y_train_0.to_numpy()),torch.Tensor(y_test_0.to_numpy())
    data_tensors = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }

    # Step 5: Store the results in a named tuple
    Train_test_results = namedtuple("train_test_results", "dataframes tensors")
    train_test_results = Train_test_results(
        dataframes=data_dataframes,
        tensors=data_tensors,
        )
    return train_test_results, categorical_features