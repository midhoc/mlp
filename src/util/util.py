import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.util.scaler import CustomStandardScaler

def custom_train_test_split(data, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_size)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return data.iloc[train_indices], data.iloc[test_indices]

def data_to_x_y(df, scaler, train=False):
    x = df[df.columns[1:]]
    y = df[1]
    if train:
        x = scaler.fit_transform(x)
    else:
        x = scaler.transform(x)
    return x.to_numpy(), y.to_numpy()

def custom_accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of true labels and predicted labels do not match.")

    correct_predictions = np.sum(np.array(y_true) == np.array(y_pred))
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples

    return accuracy


def data_prep(path, train=False):
    df = pd.read_csv(path, header=None, index_col=0)
    df[1] = (df[1] == 'M').astype(int)
    
    drop_columns = [4,5,9,14,15,22,24,25]
    drop_columns_index = [item-1 for item in drop_columns]
    df.drop(df.columns[drop_columns_index], axis = 1, inplace=True)
    scaler = CustomStandardScaler()
    
    if train:
        x, y = data_to_x_y(df, scaler, train=train)
        scaler.save('data/scaler.npz')
    else:
        scaler.load('data/scaler.npz')
        x, y = data_to_x_y(df, scaler)

    return x, y


def draw_histograms(df, r = 7):
    target_column = df.columns[0]
    features = df.columns[1:]

    class_0 = df[df[target_column] == 0]
    class_1 = df[df[target_column] == 1]

    num_features = len(features)
    num_rows = (num_features + r-1) // r 
    plt.figure(figsize=(15, r*num_rows))

    for i, column in enumerate(features):
        plt.subplot(num_rows, r, i+1)
        sns.histplot(class_0[column], color='blue', kde=True, label='B', alpha=0.5)
        sns.histplot(class_1[column], color='red', kde=True, label='M', alpha=0.5)
        plt.title(column)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot(y1, y2, x_label, y_lable, title, y1_label, y2_label):
    x = range(1, len(y1) + 1)
    plt.plot(x,y1, label=y1_label)
    plt.plot(x,y2, label=y2_label)
    plt.xlabel(x_label)
    plt.ylabel(y_lable)
    plt.title(title)
    plt.legend()
    plt.show()