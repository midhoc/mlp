import pickle
import numpy as np
from src.util.metrics import *

class Network:
    def __init__(self, best_model_filepath = 'data/best_model.pickle'):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.best_model = None
        self.best_model_filepath = best_model_filepath

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss):
        self.loss = loss
        self.loss_prime = loss.prime

    # predict output for given input
    def predict(self, input_data):
        samples = len(input_data)
        result = []

        # Run network over all samples
        for i in range(samples):
            output = input_data[i]

            # Forward propagation through layers
            for layer in self.layers:
                output = layer.forward_propagation(output)
            
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate, batch_size, x_val=None, y_val=None, patience=None):
        samples = len(x_train)
        best_val_loss = float('inf')
        best_val_acc = 0
        early_stop_counter = 0

        for i in range(epochs):
            epoch_error = self.train_one_epoch(x_train, y_train, samples, batch_size, learning_rate)
            
            
            epoch_error, train_acc, train_precision, train_recall, train_f1 = self.calculate_metrics(x_train, y_train)
            self.train_losses.append(epoch_error)
            self.train_accuracies.append(train_acc) 
            self.train_precisions.append(train_precision)
            self.train_recalls.append(train_recall)
            self.train_f1_scores.append(train_f1)
            print(f'Epoch {i+1}/{epochs}   Error={epoch_error}  accuracy={train_acc}')
            
            

            if x_val is not None and y_val is not None:
                val_loss, val_acc, val_precision, val_recall, val_f1 = self.calculate_metrics(x_val, y_val)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                self.val_precisions.append(val_precision)
                self.val_recalls.append(val_recall)
                self.val_f1_scores.append(val_f1)
                print(f'    Validation Loss: {val_loss} Validation Accuracy: {val_acc}')

                if patience:
                    early_stop_counter = self.Increment_early_stopping(val_loss, best_val_loss, early_stop_counter)
                    if early_stop_counter >= patience:
                        print(f'Early stopping at epoch {i+1}...')
                        print(f'Best Validation Loss: {best_val_loss}')
                        print(f'Validation Accuracy: {best_val_acc}')
                        break

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    self.best_model = self
                    self.save_model()
                    
                    
        if x_val is None or y_val is None:
            self.best_model = self
            self.save_model()
        out = {
            "train_losses" : self.train_losses, 
            "train_accuracies" : self.train_accuracies, 
            "val_losses" : self.val_losses, 
            "val_accuracies" : self.val_accuracies,
            "train_precisions" : self.train_precisions,
            "val_precisions" : self.val_precisions,
            "train_recalls" : self.train_recalls,
            "val_recalls" : self.val_recalls,
            "train_f1_scores" : self.train_f1_scores,
            "val_f1_scores" : self.val_f1_scores,
        }
        return out

    
    def calculate_metrics(self, x_data, y_data):
        predictions = self.predict(x_data)

        # Convert predictions to binary values (0 or 1)
        predictions = np.array(predictions).reshape(-1)
        binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions]
        loss = self.loss(y_data, predictions)
        accuracy = custom_accuracy_score(y_data, binary_predictions)
        precision = custom_precision_score(y_data, binary_predictions)
        recall = custom_recall_score(y_data, binary_predictions)
        f1 = custom_f1_score(y_data, binary_predictions)

        return loss, accuracy, precision, recall, f1
    
    def train_one_epoch(self, x_train, y_train, samples, batch_size, learning_rate):
        err = 0
        for j in range(0, samples, batch_size):
            end_idx = min(j + batch_size, samples)
            x_batch = x_train[j:end_idx]
            y_batch = y_train[j:end_idx]

            batch_err = self.train_batch(x_batch, y_batch, learning_rate)
            err += batch_err

        err /= (samples / batch_size)
        return err

    def train_batch(self, x_batch, y_batch, learning_rate):
        batch_err = 0
        for k in range(len(x_batch)):
            output = x_batch[k]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            
            batch_err += self.loss(y_batch[k], output)

            error = self.loss_prime(y_batch[k], output)
            for layer in reversed(self.layers):
                error = layer.backward_propagation(error, learning_rate)

        batch_err /= len(x_batch)
        return batch_err

    def Increment_early_stopping(self, val_loss, best_val_loss, early_stop_counter):
        if val_loss < best_val_loss:
            early_stop_counter = 0
        else:
            early_stop_counter += 1  # Increment the counter
        return early_stop_counter

    def save_model(self):
        with open(self.best_model_filepath, 'wb') as file:
            pickle.dump(self.best_model, file)
        print(f'    Model saved to {self.best_model_filepath}')

    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
