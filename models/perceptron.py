"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
    

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        self.w = np.random.random((self.n_class, X_train.shape[1]))

        for epoch in range(self.epochs):
            if epoch % 2 == 0:
                self.lr /= 10
            for index in range(len(X_train)):
                label = y_train[index]
                data = X_train[index, :]
                w_yi = self.w[label, :]
                wyi_xi =  np.dot(w_yi, data.T)
                for class_index in range(self.n_class):
                    if (class_index != label):
                        w_c = self.w[class_index, :]
                        wc_xi = np.dot(w_c, data.T)
                        if wc_xi > wyi_xi:
                            self.w[label,:] = self.w[label,:] + self.lr * data
                            self.w[class_index, :] = self.w[class_index, :] - self.lr *data
        
        # for epoch in range(self.epochs):
        #     pred = np.dot(self.w, X_train.T)
        #     for index in range(len(X_train)):
        #         pred_label= np.argmax(pred[:, index])
        #         train_label = y_train[index]
        #         if pred_label!=train_label:
        #             self.w[:, :] -=  self.lr * X_train[index, :]
        #             self.w[train_label, :] +=  2* self.lr * X_train[index, :]
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
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        m, n = X_test.shape
        result = []

        for index in range(m):
            data = X_test[index,:]
            prob = np.dot(self.w, data.T)
            pred = np.argmax(prob)
            result.append(pred)
        return np.array(result)

