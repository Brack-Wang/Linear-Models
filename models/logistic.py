"""Logistic regression model."""

import numpy as np
import copy
from sklearn.metrics import accuracy_score
import math

class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.b = 0

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1 / (1 + np.exp(-z))
    
    def gradients_decent(self, X_train, y_train, y_pred):
        db = np.mean(y_pred - y_train)
        dw = np.matmul(X_train.transpose(), y_pred - y_train)
        dw = np.array([np.mean(grad) for grad in dw])
        return dw, db

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.w = np.zeros(X_train.shape[1])

        for i in range(self.epochs):
            x_w = np.matmul(self.w, X_train.transpose()) + self.b
            pred = self.sigmoid(x_w)
            dw, db = self.gradients_decent(X_train, y_train, pred)
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        prob = self.sigmoid(np.matmul(X_test, self.w.transpose()) + self.b)
        return [1 if p > 0.5 else 0 for p in prob]
