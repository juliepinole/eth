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
        num_features: list = None,
        category_to_drop: dict = None,
        drop_first: bool = False,
        # drop_one_category: bool = False,
        ):
    # We set drop_first=False when we want to select manually which dummy column to drop, to have a benchmark that makes sense.
    df_hot = pd.get_dummies(df, columns=cols_obj_pure, drop_first=drop_first)
    if category_to_drop is not None:
        for var, category in category_to_drop.items():
            df_hot.drop('_'.join([var, category]), axis=1, inplace=True)
    all_features = list(df_hot.columns)
    categorical_features = list(set(all_features) - set(num_features))
    return df_hot, categorical_features, all_features


def pre_process_features(
        df: pd.DataFrame,
        num_features: list,
        categorical_features: list,
        label_col: str = 'HeartDisease',
        add_one_hot_encoded: bool = False,
        stand_features: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
        category_to_drop: dict = None,
        split_data: bool = True,
):
    """
    Pre-process the features of the dataset.
    :param df: pd.DataFrame: the dataset
    :param num_features: list: numerical features
    :param categorical_features: list: categorical features
    :param label_col: str: the label column
    :param add_one_hot_encoded: bool: whether to add one-hot encoded columns
    :param stand_features: bool: whether to standardize features
    :param test_size: float: test size
    :param random_state: int: random state
    :param category_to_drop: dict: category to drop
    :param split_data: bool: whether to split data into train and test
    :return: namedtuple: train_test_results, categorical_features
    """
    # Step 1: Adding One-Hot_Encoded columns
    if add_one_hot_encoded:
        df_x, categorical_features, all_features = adding_one_hot_encoded(
            df[num_features + categorical_features],
            categorical_features,
            num_features = num_features,
            category_to_drop=category_to_drop,
            drop_first=False,
            )
    else:
        df_x = df[num_features].copy()
        all_features = num_features

    # Step 2: Standardizing features
    if stand_features:
        df_x = standardize_features(df_x, cols_to_standardize=all_features)
    
    # Step 3: Split test/ train
    if split_data:
        X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(
        df_x, df[label_col], test_size=test_size, random_state=random_state)
    else:
        X_train_0 = df_x
        X_test_0 = np.zeros(1)
        y_train_0 = df[label_col]
        y_test_0 = np.zeros(1)
    # Organize data in a dictionary
    data_dataframes = {
        'X_train': X_train_0,
        'X_test': X_test_0,
        'y_train': y_train_0,
        'y_test': y_test_0,
    }

    # Step 4: Create Tensors
    if split_data:
        X_train, X_test = torch.Tensor(X_train_0.to_numpy()),torch.Tensor(X_test_0.to_numpy())
        y_train, y_test = torch.Tensor(y_train_0.to_numpy()),torch.Tensor(y_test_0.to_numpy())
        data_tensors = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }
    else:
        X_train = torch.Tensor(X_train_0.to_numpy())
        y_train = torch.Tensor(y_train_0.to_numpy())
        data_tensors = {
            'X_train': X_train,
            'y_train': y_train,
        }

    # Step 5: Store the results in a named tuple
    Train_test_results = namedtuple("train_test_results", "dataframes tensors")
    train_test_results = Train_test_results(
        dataframes=data_dataframes,
        tensors=data_tensors,
        )
    return train_test_results, categorical_features