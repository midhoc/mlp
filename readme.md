# Multilayer Perceptron (MLP) From Scratch

This repository contains a **Python implementation of a Multilayer Perceptron (MLP)** built entirely from scratch. It demonstrates how neural networks work at their core, providing insight into the mechanics of forward propagation, backward propagation, and training.

## Features

- Fully connected layers with customizable input and output sizes.
- Multiple activation functions, including ReLU, Tanh, and Sigmoid.
- Binary Cross-Entropy loss function for binary classification tasks.
- Backpropagation and gradient descent for training.
- Early stopping to prevent overfitting.
- Custom metrics, including accuracy, precision, recall, and F1-score.

## Project Structure

The project is organized into the following files and directories:

```plaintext
├── src
│   ├── layers
│   │   ├── activation.py         # Activation functions (ReLU, Tanh, Sigmoid, etc.)
│   │   ├── activation_layer.py   # Activation layer for non-linear transformations
│   │   ├── base_layer.py         # Base layer class
│   │   ├── FC_layer.py           # Fully connected layer implementation
│   │   ├── loss.py               # Loss functions (Binary Cross-Entropy, MSE, etc.)
│   ├── util
│   │   ├── metrics.py            # Custom metrics for evaluation
│   │   ├── scaler.py             # Standard scaler for data normalization
│   │   ├── util.py               # Utility functions for data preparation and visualization
│   ├── network.py                # Main network implementation
│   ├── save_load.py              # Functions for saving and loading models
├── train.py                      # Script to train the MLP
├── predict.py                    # Script to make predictions
├── split_data.py                 # Script to split data into training and validation sets
```

## Requirements

To run the project, install the following dependencies:
* **Python 3.8 or higher**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Click** (for CLI support)

You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
Split the dataset into training and validation sets using `split_data.py`:

```bash
python split_data.py --path data/data.csv --save-dir data/ --test-size 0.3 --random-state 42
```

### 2. Training the MLP
Train the MLP using `train.py`. Customize hyperparameters like learning rate, batch size, and number of epochs:

```bash
python train.py --path-train data/train.csv --path-val data/val.csv --epoch 200 --learning-rate 0.01 --batch-size 16 --patience 10
```

### 3. Making Predictions
Use the trained model to make predictions on the validation set:

```bash
python predict.py --path-val data/val.csv --path-model data/best_model.pickle --path-save data/
```

## Customization

The network structure can be customized in `train.py`. For example:

```python
net.add(FCLayer(input_size=30, output_size=32))
net.add(ActivationLayer(Tanh()))
net.add(FCLayer(32, 32))
net.add(ActivationLayer(Tanh()))
net.add(FCLayer(32, 1))
net.add(ActivationLayer(sigmoid()))
```

## References

This project is complemented by a detailed article explaining the theory and implementation of MLPs. [Read the article here](https://medium.com/@hocine.midoun/demystifying-multilayer-perceptrons-a-deep-dive-80ced0438ccd).