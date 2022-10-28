from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
      scores = X[i].dot(W)
      scores -= np.max(scores)
      scores_exp = np.sum(np.exp(scores))
      correct_exp = np.exp(scores[y[i]])
    
      
      loss -= np.log(correct_exp / scores_exp)

      for j in range(num_classes):
        if j == y[i]:
          continue
        dW[:, j] += X[i]/scores_exp * np.exp(scores[j])         
      dW[:, y[i]] += (X[i]/scores_exp*np.exp(scores[y[i]]) - X[i])

    loss /= num_train
    dW /= num_train 

    loss += reg * np.sum(W*W)
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores)
    scores_exp =  np.sum(np.exp(scores), axis=1) # 500*1
    correct_scores_exp = np.exp(scores[np.arange(num_train), y])

    loss = correct_scores_exp / scores_exp
    loss = - np.sum(np.log(loss)/num_train + reg*np.sum(W*W))
    s = np.divide(np.exp(scores), scores_exp.reshape(num_train, 1))
    s[range(num_train), y] = 0
    s[range(num_train), y] = -(scores_exp - correct_scores_exp) / scores_exp
    dW = X.T.dot(s)
    dW /= num_train
    dW += 2*reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
