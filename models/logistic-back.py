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
        self.batch_size = 10
        self.b = 0
        self.losses = []
        self.train_accuracies = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        result = 1.0 /((1+np.exp(-z))*1.0)
        return result

    def loss(self, y, y_hat):
        print("y", y)
        print("y_hat", y_hat)
        loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
        return loss

    def gradients(self, X, y, y_hat):
        # X --> Input.
        # y --> true/target value.
        # y_hat --> hypothesis/predictions.
        # w --> weights (parameter).
        # b --> bias (parameter).
        
        # m-> number of training examples.
        m = X.shape[0]
        
        # Gradient of loss w.r.t weights.

        dw = (1/m)*np.dot(X.T, (y_hat - y))
        
        return dw

    def normalize(self, X):
        # X --> Input.
        # m-> number of training examples
        # n-> number of features 
        m, n = X.shape
        # Normalizing all the n features of X.
        for i in range(n):
            X = (X - X.mean(axis=0))/X.std(axis=0)
        return X

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        m, n = X_train.shape
        self.w = np.zeros((n, 1))
        X = self.normalize(X_train)
        y = np.where(y_train == 0, -1 ,1)
        losses = []
        for epoch in range(self.epochs):
            for i in range((m-1)//self.batch_size + 1):
                start_i = i * self.batch_size
                end_i = start_i + self.batch_size
                # xb: [10, 11]
                xb = X[start_i : end_i, :]
                #  yb: [10, 1]
                yb = y[start_i : end_i]

                # y_hat [10, 1]
                y_hat = self.sigmoid(np.dot(xb, self.w))
     
                dw = self.gradients(xb, yb, y_hat)

                self.w = self.w - self.lr * dw

            l = self.loss(y, self.sigmoid(np.dot(X, self.w)))
            losses.append(l)
        return self.w, self.b, losses

               
                
        # pass

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
        x = self.normalize(X_test)

        preds = self.sigmoid(np.dot(x, self.w) + self.b)

        pred_class = []

        pred_class = [1 if i >0.5 else 0 for i in preds]
        return np.array(pred_class)
