from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        # input_dim   15 // 特征数，例如32*32=1024
        # hidden_dims [20, 30]
        # num_classes 10
        for i in range(self.num_layers):  # 0 1 2
            if i == 0:
                self.params[f'W{i+1}'] = np.random.normal(
                    loc=0, scale=weight_scale, size=(input_dim, hidden_dims[i]))
                self.params[f'b{i+1}'] = np.zeros(hidden_dims[i])
            # W(15, 20)

            elif i < len(hidden_dims):  # 1
                self.params[f'W{i+1}'] = np.random.normal(
                    loc=0, scale=weight_scale, size=(hidden_dims[i-1], hidden_dims[i]))
                self.params[f'b{i+1}'] = np.zeros(hidden_dims[i])
            # W(20, 30)
            else:  # 2
                self.params[f'W{i+1}'] = np.random.normal(
                    loc=0, scale=weight_scale, size=(hidden_dims[i-1], num_classes))
                self.params[f'b{i+1}'] = np.zeros(num_classes)
            # 　W(30, 10)
            if (self.normalization == 'batchnorm' or self.normalization == 'layernorm') and i < len(hidden_dims):
                self.params[f'gamma{i+1}'] = np.ones(hidden_dims[i])
                self.params[f'beta{i+1}'] = np.ones(hidden_dims[i])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
            self.bn_params = [{"mode": "train", "eps": 1e-5, "momentum": 0.9}
                              for _ in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode

        if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
            for bn_param in self.bn_params:
                bn_param['layernorm'] = self.normalization == 'layernorm'

        scores = None
        caches = []
        for i in range(self.num_layers - 1):
            if self.normalization == "batchnorm" or self.normalization == "layernorm":
                gamma = self.params[f'gamma{i+1}']
                beta = self.params[f'beta{i+1}']
                bn_param = self.bn_params[i]
                X, cache = affine_norm_relu_forward(
                    X, self.params[f'W{i+1}'], self.params[f'b{i+1}'], gamma, beta, bn_param)  # (100, 5) (100, 100) (100,)
            else:
                X, cache = affine_relu_forward(
                    X, self.params[f'W{i+1}'], self.params[f'b{i+1}'])
            if self.use_dropout:
                X, cache_dropout = dropout_forward(X, self.dropout_param)
                cache = (cache, cache_dropout)
            else:
                cache = (cache, None)
            caches.append(cache)
        scores, cache = affine_forward(
            X, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
        caches.append(cache)

        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        loss, dout = softmax_loss(scores, y)
        for i in range(self.num_layers):
            loss += 0.5*self.reg*np.sum(self.params[f'W{i+1}']**2)
            if i == 0:  # W3
                dout, grads[f'W{self.num_layers - i}'], db = affine_backward(
                    dout, caches[self.num_layers - i - 1])
            else:  # W2, W1
                if self.use_dropout:
                    dout = dropout_backward(
                        dout, caches[self.num_layers - i - 1][1])
                if self.normalization:
                    dout, dw, db, dgamma, dbeta = affine_norm_relu_backward(
                        dout, caches[self.num_layers - i - 1][0])
                    grads[f'W{self.num_layers - i}'] = dw
                    grads[f'gamma{self.num_layers - i}'] = dgamma
                    grads[f'beta{self.num_layers - i}'] = dbeta
                else:
                    dout, grads[f'W{ self.num_layers - i}'], db = affine_relu_backward(
                        dout, caches[self.num_layers - i - 1][0])
            grads[f'W{self.num_layers - i}'] += self.reg * \
                self.params['W{}'.format(self.num_layers - i)]
            grads[f'b{self.num_layers - i}'] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
