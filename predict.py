import pandas as pd
import numpy as np
import click

from src.util.metrics import custom_accuracy_score
from src.util.util import data_prep

from src.network import Network
from src.layers.loss import BinaryCrossEntropy


@click.command()
@click.option('--path-val', default='data/val.csv', help='Path to the validation CSV file.')
@click.option('--path-model', default='data/best_model.pickle', help='Path to the model.')
@click.option('--path-save', default='data/', help='Path to the model.')
@click.option('--random-state', default=42, help='Seed for the random number generator.')
def main(path_val,path_model,path_save,random_state):
    np.random.seed(random_state)
    x_val, y_val = data_prep(path_val)
    loss = BinaryCrossEntropy()
    net = Network()
    net = net.load_model(path_model)
    out = net.predict(x_val)
    out = np.array(out).reshape(-1)
    loss_error = loss(y_val, out)
    out = (out >= 0.5).astype(int)
    accuracy = custom_accuracy_score(y_val, out)
    df = pd.read_csv(path_val, header=None, index_col=0)
    df[-1] = ['M' if item == 1 else 'B' for item in out]
    df.to_csv(f'{path_save}/prediction.csv', header=False)
    print(f'Loss : {loss_error}   Accuracy : {accuracy}')
    
if __name__ == '__main__':
    main()