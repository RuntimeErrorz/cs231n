from builtins import range
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]  # 500
    for i in range(num_train):
        scores = X[i].dot(W)  # X[i] = (1, 3073) W = (3073, 10)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        loss += -np.log(probs[y[i]])
        probs_reshape = probs.reshape(1, -1)  # (1, 10)
        probs_reshape[:, y[i]] -= 1
        dW += X[i].reshape(-1, 1).dot(probs_reshape)  # (3073, 1) * (1, 10)
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg * 2 * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    scores = X.dot(W) # (500, 10)
    probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) #不加keepdims=True的话，会出现(500, )的情况 
    loss = np.sum(-np.log(probs[np.arange(X.shape[0]), y]))
    loss /= X.shape[0]
    loss += reg * np.sum(W * W)

    probs[np.arange(X.shape[0]), y] -= 1
    dW = X.T.dot(probs) # (3073, 500) (500, 10)
    dW /= X.shape[0]
    dW += reg * 2 * W
    return loss, dW
