from builtins import range
import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. (3073, 10)
    - X: A numpy array of shape (N, D) containing a minibatch of data.  (500, 3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)  # X[i] = (1, 3073) W = (3073, 10)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):  # 1...500
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # s[j] - s[y[i]] > -1
            # s[y[i]] - s[j] <  1
            # s = X[i]W
            # s[j] = X[i]W[j]
            # s[y[i]] = X[i]W[y[i]]
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape)
    num_train = X.shape[0]
    scores = X.dot(W)  # (500, 3073) (3073, 10) = (500, 10)
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)   # (500, 1)
    margins = np.maximum(0, scores - correct_class_scores + 1) 
    margins[np.arange(num_train), y] = 0 # y=[9,2,7,..]  y[i] = c means that X[i] has label c, where 0 <= c < C. 
    loss = np.sum(margins) / num_train + reg * np.sum(W * W)

    masks = np.zeros(margins.shape)  #(500, 10)
    masks[margins > 0] = 1
    # masks[i][j] 代表第i个样本的第j个类别的梯度需要的代数值（无法描述），
    # 如果masks[i][j] = 1，说明第i个样本的第j个类别的margin大于0，需要计算梯度、
    # 如果masks[i][j] = 0，说明第i个样本的第j个类别的margin小于等于0，不需要计算梯度。

    masks[np.arange(num_train), y] = -np.sum(masks, axis=1)
    """
    这里对500行的正确值进行了处理，masks每一行为1（即需要计入的X[j]）的和（参照前面代码+1个就要-1个）的相反数。
    arr = np.array([[1, 1, 1], [0, 0, 0]])
    arr[np.arange(2), [0, 2]] = [4, 5]
    arr = [[4 1 1] 
           [0 0 5]] 这样赋值是不是有点奇怪？
    """
    dW = X.T.dot(masks) / num_train + 2 * reg * W # (3073, 500) (500, 10) = (3073, 10) = W.shape，
    return loss, dW
