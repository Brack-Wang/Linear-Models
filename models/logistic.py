"""Logistic regression model."""

import numpy as np


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
        self.avoid_zero = 1e-10

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        result = 1 / (1 + np.exp(-z))
        return result

    def compute_loss(self, pred, y_train):
        loss_list = y_train * np.log(pred + self.avoid_zero) + (1-y_train) * np.log(1 - pred + self.avoid_zero)
        return -np.mean(loss_list)


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        m, n = X_train.shape
        self.w = np.zeros(n)
        epoch = 0
        loss = 100000

        while loss > self.threshold and epoch < self.epochs :
            pred = self.sigmoid(np.dot(self.w, X_train.T) )
            gradient = np.dot(X_train.T, pred - y_train) /m
            self.w = self.w - self.lr * gradient
            loss = self.compute_loss(y_train, pred)
            epoch += 1
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
        X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
        result = []
        for index in range(len(X_test)):
            data = X_test[index,:]
            prob = np.dot(self.w, data.T)
            pred = np.round(self.sigmoid(prob))
            result.append(pred)
        return np.array(result)

