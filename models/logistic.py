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
        z_list = []
        for i in range(len(z)):
            if z[i] < 0:
                z_list.append(np.exp(z[i]) / (1 + np.exp(z[i])))
            else:
                z_list.append(1 / (1 + np.exp(-z[i])))
        return np.array(z_list)
    
    def gradients_decent(self, X_train, y_train, y_pred):
        dw = np.matmul(X_train.transpose(), y_pred - y_train)
        dw = np.array([np.mean(grad) for grad in dw])
        return dw

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        self.w = np.zeros(X_train.shape[1])
        m, n = X_train.shape

        for epoch in range(self.epochs):
            for index in range(m):
                label = y_train[index]
                data = X_train[index, :]
                w_yi = self.w[label, :]
                wyi_xi =  - label * np.dot(w_yi, data.T)
                self.w += self.lr * dw
            # x_w = np.matmul(self.w, X_train.transpose()) + self.b
            # pred = self.sigmoid(x_w)
            # dw, db = self.gradients_decent(X_train, y_train, pred)
            # dw = np.matmul(X_train.transpose(), pred - y_train)
            # dw = np.array([np.mean(grad) for grad in dw])
            # self.w = self.w - self.lr * dw
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
