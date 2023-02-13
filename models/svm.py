"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, batch: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.batch_size = batch

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        m, n = X_train.shape
        dw = np.zeros_like(self.w)
        for index in range(m):
            label = y_train[index]
            data = X_train[index, :]
            w_yi = self.w[label, :]
            wyi_xi =  np.dot(w_yi, data.T)
            for class_index in range(self.n_class):
                if (class_index != label):
                    w_c = self.w[class_index, :]
                    wc_xi = np.dot(w_c, data.T)
                    if wc_xi > wyi_xi + 1:
                        dw[label] += data
                        dw[class_index] -= data
        dw /= m
        wc = self.w + self.lr * dw
        return wc

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        m, n = X_train.shape
        self.w = np.random.random((self.n_class, X_train.shape[1]))
        for epoch in range(self.epochs):
            if epoch % 1 == 0:
                self.lr /= 10
            for i in range((m-1)//self.batch_size + 1):
                start = i * self.batch_size
                end = start + self.batch_size
                if end > m:
                    end = m
                X = X_train[start : end]
                y = y_train[start : end]
                wc = self.calc_gradient(X, y)
                self.w = (1 - self.lr * self.reg_const / m) * wc
        return

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
            pred = np.argmax(prob)
            result.append(pred)
        return np.array(result)
