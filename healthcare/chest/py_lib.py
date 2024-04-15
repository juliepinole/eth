import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import namedtuple
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


##################################################################
# Integrated Gradients
##################################################################

def interpolate_images(
    baseline,
    image,
    alphas,
):
  alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x = tf.expand_dims(image, axis=0)
  input_x = tf.cast(
    input_x, tf.float32
  )
  delta = input_x - baseline_x
  images = baseline_x +  alphas_x * delta
  return images


def compute_gradients(
    model,
    images,
    alpha_length=32,
    target_is_class_zero=False,
    ):
  with tf.GradientTape() as tape:
    tape.watch(images)
    # Normalize the data, as was done in training
    images_for_prediction = tf.divide(images, 255)
    images_for_prediction = tf.reshape(images_for_prediction, [alpha_length, 150, 150, 1])
  
    # Note: using model(.) instead of model.predict(.), as the latter transforms the 
    # tensor with the attribute .numpy() which is not possible in the mode decorator
    # function @tf, see https://stackoverflow.com/questions/52357542/attributeerror-tensor-object-has-no-attribute-numpy
    probs = model(images_for_prediction) # Proba of the class 1 = normal
    if target_is_class_zero:
      probs = 1 - probs
  return tape.gradient(probs, images)


def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients


@tf.function
def one_batch(model, baseline, image, alpha_batch, target_is_class_zero=False):
  # Generate interpolated inputs between baseline and input.
  interpolated_path_input_batch = interpolate_images(
    baseline=baseline,
    image=image,
    alphas=alpha_batch,
    )

  # Compute gradients between model outputs and interpolated inputs.
  gradient_batch = compute_gradients(
     model=model,
    images=interpolated_path_input_batch,
    alpha_length=len(alpha_batch),
    target_is_class_zero=target_is_class_zero,
    )
  return gradient_batch


def integrated_gradients(
  model,
  baseline,
  image,
  m_steps=50,
  batch_size=32,
  target_is_class_zero=False,
  ):
  # Generate alphas.
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

  # Collect gradients.    
  gradient_batches = []

  # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
  for alpha in tf.range(0, len(alphas), batch_size):
    from_ = alpha
    to = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]

    gradient_batch = one_batch(model, baseline, image, alpha_batch, target_is_class_zero=target_is_class_zero)
    gradient_batches.append(gradient_batch)

  # Concatenate path gradients together row-wise into single tensor.
  total_gradients = tf.concat(gradient_batches, axis=0)

  # Integral approximation through averaging gradients.
  avg_gradients = integral_approximation(gradients=total_gradients)

  # Scale integrated gradients with respect to input.
  integrated_gradients = (image - baseline) * avg_gradients

  return integrated_gradients


def plot_img_attributions(
  model,
  baseline,
  image,
  target_is_class_zero=False,
  m_steps=50,
  img_size: int = 150,
  cmap=None,
  overlay_alpha=0.4,
  figsize=(8, 8),
  suptitle=None,
  ):
  attributions = integrated_gradients(
     model=model,
    baseline=baseline,
    image=image,
    target_is_class_zero=target_is_class_zero,
    m_steps=m_steps,
    )

  # Sum of the attributions across color channels for visualization.
  # The attribution mask shape is a grayscale image with height and width
  # equal to the original image.
  # attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
  attribution_mask = tf.reshape(attributions, [img_size, img_size])

  fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=figsize)

  axs[0, 0].set_title('Baseline image')
  axs[0, 0].imshow(baseline)
  axs[0, 0].axis('off')

  axs[0, 1].set_title('Original image')
  axs[0, 1].imshow(image)
  axs[0, 1].axis('off')

  axs[1, 0].set_title('Attribution mask')
  axs[1, 0].imshow(attribution_mask, cmap=cmap)
  axs[1, 0].axis('off')

  axs[1, 1].set_title('Overlay')
  axs[1, 1].imshow(attribution_mask, cmap=cmap)
  axs[1, 1].imshow(image, alpha=overlay_alpha)
  axs[1, 1].axis('off')
  
  if suptitle is not None:
    plt.suptitle(suptitle)

  plt.tight_layout()
  return fig, attribution_mask



##################################################################
# Images Processing
##################################################################

def get_training_data(
  data_dir,
  img_size: int = 150,
  labels: list = ['PNEUMONIA', 'NORMAL'],
  ): 
  data = [] 
  for label in labels:
    path = os.path.join(data_dir, label)
    class_num = labels.index(label)
    for img in os.listdir(path):
      try:
        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
        data.append([resized_arr, class_num])
      except Exception as e:
        print(e)
  return data


##################################################################
# Heart-related functions, check whether still useful
##################################################################


def encode_categorical_features(
        df: pd.DataFrame,
        categorical_features: list,
) ->tuple[pd.DataFrame,dict]:
    '''
    Encode the categorical features of the dataset.
    :param df: pd.DataFrame: the dataset
    :param categorical_features: list: categorical features
    :return: pd.DataFrame: encoded dataset
    '''
    label_encoders = {}
    for cat_col in categorical_features:
        # TODO(jpinole): is it ok to use an encoder designed for labels and not features?
        label_encoders[cat_col] = LabelEncoder()
        df[cat_col] = label_encoders[cat_col].fit_transform(df[cat_col])
    return df, label_encoders

        

def standardize_features(
    df: pd.DataFrame,
    cols_to_standardize: list = None,
) -> pd.DataFrame:
    '''
    Standardize the features of the dataset.
    :param df: pd.DataFrame: the dataset
    :param cols_to_standardize: list: columns to standardize
    :return: pd.DataFrame: standardized dataset
    '''
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
    print(df['Sex'].unique())
    df_hot = pd.get_dummies(df, columns=cols_obj_pure, drop_first=drop_first)
    print(list(df_hot.columns))
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
        add_embeddings: bool = False,
        stand_features: bool = False,
        stand_embeddings: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
        category_to_drop: dict = None,
        split_data: bool = True,
        max_emb_dim: int = 50,
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
    if add_embeddings and add_one_hot_encoded:
        raise ValueError("You cannot add embeddings and one-hot encoded columns at the same time.")
    print(df['Sex'].unique())
    # Step 0: Encode categorical features
    label_encoders = {}
    emb_dims = []
    if add_embeddings:
        print('adding embeddings')
        df_aux, label_encoders = encode_categorical_features(df, categorical_features)
        df_x = df_aux[num_features + categorical_features].copy()
        all_features = num_features + categorical_features
        cat_dims = [int(df_x[col].nunique()) for col in categorical_features]
        # TODO(jpinole): check if this is the best way to calculate the embedding dimensions
        emb_dims = [(x, min(max_emb_dim, (x + 1) // 2)) for x in cat_dims]
        df_x_cat = df_x[categorical_features]
        df_x_num = df_x[num_features].astype(np.float32)
    
    # Step 1: Adding One-Hot_Encoded columns
    elif add_one_hot_encoded:
        print('adding One Hot Encoded')
        df_x, categorical_features, all_features = adding_one_hot_encoded(
            df[num_features + categorical_features],
            categorical_features,
            num_features = num_features,
            category_to_drop=category_to_drop,
            drop_first=False,
            )
        df_x_cat = None
        df_x_num = df_x.astype(np.float32) # After OHE, categorical features have become numerical

    else:
        df_x = df[num_features].copy()
        df_x_cat = None
        df_x_num = df_x.astype(np.float32).copy()
        all_features = num_features

    # Step 2: Standardizing features
    if stand_features:
        df_x_num = standardize_features(df_x_num, cols_to_standardize=df_x_num.columns)
    if stand_embeddings:
        df_x_cat = standardize_features(df_x_cat, cols_to_standardize=df_x_cat.columns)
    if df_x_cat is not None:
        df_x = pd.concat([df_x_num, df_x_cat], axis=1)
    else:
        df_x = df_x_num
    
    # Step 3: Split test/ train
    if split_data:
        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(
        df_x, df[label_col], test_size=test_size, random_state=random_state)
        print(list(x_train_0.columns))
    else:
        x_train_0 = df_x
        y_train_0 = df[label_col]
        print(list(x_train_0.columns))
    
    # Organize data in a dictionary
    if split_data:
        data_dataframes = {
            'X_train': {
                'num': x_train_0[num_features],
                'cat': x_train_0[categorical_features],
                'all': x_train_0,  
            },
            'X_test': {
                'num': x_test_0[num_features],
                'cat': x_test_0[categorical_features],
                'all': x_test_0,  
            },
            'y_train': y_train_0,
            'y_test': y_test_0,
        }
    else:
        data_dataframes = {
            'X_train': {
                'num': x_train_0[num_features],
                'cat': x_train_0[categorical_features],
                'all': x_train_0,  
            },
            'y_train': y_train_0,
        }


    # Step 4: Create Tensors
    x_train_dict = {}
    x_test_dict = {}
    y_train = torch.Tensor(y_train_0.to_numpy())
    for features_type, df in data_dataframes['X_train'].items():
        # x_train_dict[features_type] = torch.Tensor(df.to_numpy())
        x_train_dict[features_type] = torch.from_numpy(df.to_numpy())

    if split_data:
        y_test = torch.Tensor(y_test_0.to_numpy())
        for features_type, df in data_dataframes['X_test'].items():
            x_test_dict[features_type] = torch.from_numpy(df.to_numpy())
        data_tensors = {
            'X_train': x_train_dict,
            'X_test': x_test_dict,
            'y_train': y_train,
            'y_test': y_test,
        }
    else:
        data_tensors = {
            'X_train': x_train_dict,
            'y_train': y_train,
        }

    # Step 5: Store the results in a named tuple
    Train_test_results = namedtuple("train_test_results", "dataframes tensors")
    train_test_results = Train_test_results(
        dataframes=data_dataframes,
        tensors=data_tensors,
        )
    return (
        train_test_results,
        categorical_features,
        all_features,
        {'label_encoders': label_encoders, 'emb_dims': emb_dims}
        )