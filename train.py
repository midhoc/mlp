import pandas as pd
import numpy as np
import click
from src.util.util import data_prep
from src.util.util import plot
from src.network import Network
from src.layers.FC_layer import FCLayer
from src.layers.activation_layer import ActivationLayer
from src.layers.activation import ReLu, sigmoid, Tanh
from src.layers.loss import BinaryCrossEntropy


@click.command()
@click.option('--path-train', default='data/train.csv', help='Path to the training CSV file.')
@click.option('--path-val', default='data/val.csv', help='Path to the validation CSV file.')
@click.option('--epoch', default=200, help='Number of training epochs.')
@click.option('--learning-rate', default=0.01, help='Learning rate for the optimizer.')
@click.option('--batch-size', default=16, help='Batch size for training.')
@click.option('--patience', default=10, help='Patience for early stopping.')
@click.option('--random-state', default=42, help='Seed for the random number generator.')
def main(path_train, path_val, epoch, learning_rate, batch_size, patience, random_state):
    np.random.seed(random_state)
    x_train, y_train = data_prep(path_train, train=True)
    x_val, y_val = data_prep(path_val)

    net = Network()
    net.add(FCLayer(len(x_train[0]), 32))
    net.add(ActivationLayer(Tanh()))
    net.add(FCLayer(32, 32))
    net.add(ActivationLayer(Tanh()))
    net.add(FCLayer(32, 1))
    net.add(ActivationLayer(sigmoid()))

    # train
    net.use(BinaryCrossEntropy())
    out_dict = net.fit(x_train, y_train, epochs=epoch, learning_rate=learning_rate,
                       batch_size=batch_size, x_val=x_val, y_val=y_val, patience=patience)
    df = pd.DataFrame(out_dict)
    df.to_csv('data/metrics.csv', index=False)
    plot(out_dict["train_losses"], out_dict["val_losses"], "epoch", "loss", "Loss evolution", "train_losses", "val_losses")
    plot(out_dict["train_accuracies"], out_dict["val_accuracies"], "epoch", "accuracy", "Accuracy evolution", "train_accuracies", "val_accuracies")
    plot(out_dict["train_precisions"], out_dict["val_precisions"], "epoch", "precisions", "precisions evolution", "train_precisions", "val_precisions")
    plot(out_dict["train_recalls"], out_dict["val_recalls"], "epoch", "recalls", "recalls evolution", "train_recalls", "val_recalls")
    plot(out_dict["train_f1_scores"], out_dict["val_f1_scores"], "epoch", "f1_scores", "f1_scores evolution", "train_f1_scores", "val_f1_scores")

if __name__ == '__main__':
    main()
