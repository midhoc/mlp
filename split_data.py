import pandas as pd
import numpy as np
import click
import seaborn as sns
import matplotlib.pyplot as plt
from src.util.util import custom_train_test_split, draw_histograms



@click.command()
@click.option('--path', default='data/data.csv', help='Path to the input CSV file.')
@click.option('--save-dir', default='data/', help='Directory to save the train and val CSV files.')
@click.option('--test-size', default=0.3, help='Proportion of the dataset to include in the validation split.')
@click.option('--random-state', default=42, help='Seed for the random number generator.')
def main(path, save_dir, test_size, random_state):
    df = pd.read_csv(path, header=None, index_col=0)
    df_train, df_val = custom_train_test_split(df, test_size=test_size, random_state=random_state)
    df_train.to_csv(f'{save_dir}/train.csv', header=False)
    df_val.to_csv(f'{save_dir}/val.csv', header=False)
    df[1] = (df[1] == 'M').astype(int)
    draw_histograms(df)
    sns.heatmap(df.corr())
    plt.show()
    
if __name__ == '__main__':
    main()