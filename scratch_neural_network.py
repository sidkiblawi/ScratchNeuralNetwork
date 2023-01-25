# MNIST data setup
# ----------------
#
# We will use the classic `MNIST <http://deeplearning.net/data/mnist/>`_ dataset,
# which consists of black-and-white images of hand-drawn digits (between 0 and 9).
#
# We will use `pathlib <https://docs.python.org/3/library/pathlib.html>`_
# for dealing with paths (part of the Python 3 standard library), and will
# download the dataset using
# `requests <http://docs.python-requests.org/en/master/>`_. We will only
# import modules when we use them, so you can see exactly what's being
# used at each point.

from pathlib import Path
import requests
import torch
import pickle
import gzip
from matplotlib import pyplot
import math
import numpy as np
from IPython.core.debugger import set_trace

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

###############################################################################
# This dataset is in numpy array format, and has been stored using pickle,
# a python-specific format for serializing data.


with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

###############################################################################
# Each image is 28 x 28, and is being stored as a flattened row of length
# 784 (=28x28). Let's take a look at one; we need to reshape it to 2d
# first.


pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

###############################################################################
# PyTorch uses ``torch.tensor``, rather than numpy arrays, so we need to
# convert our data.


x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
print(x_train, y_train)
print(x_train.shape)
print(y_train.shape)

###############################################################################
# Neural net from scratch (no torch.nn)
# ---------------------------------------------
#

# PyTorch provides methods to create random or zero-filled tensors, which we will
# use to create our weights and bias for a simple linear model. These are just regular
# tensors, with one very special addition: we tell PyTorch that they require a
# gradient. This causes PyTorch to record all of the operations done on the tensor,
# so that it can calculate the gradient during back-propagation *automatically*!
#
# For the weights, we set ``requires_grad`` **after** the initialization, since we
# don't want that step included in the gradient. (Note that a trailing ``_`` in
# PyTorch signifies that the operation is performed in-place.)
#
# .. note:: We are initializing the weights here with
#    `Xavier initialisation <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
#    (by multiplying with 1/sqrt(n)).


weights = torch.randn(784, 10) / math.sqrt(784)  # output layer 10 units
weights.requires_grad = True
bias = torch.zeros(10, requires_grad=True)


###############################################################################
# Thanks to PyTorch's ability to calculate gradients automatically, we can
# use any standard Python function (or callable object) as a model! So
# let's just write a plain matrix multiplication and broadcasted addition
# to create a simple linear model. We also need an activation function, so
# we'll write `log_softmax` and use it. Remember: although PyTorch
# provides lots of pre-written loss functions, activation functions, and
# so forth, you can easily write your own using plain python. PyTorch will
# even create fast GPU or vectorized CPU code for your function
# automatically.

# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()
  app = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):
    return log_softmax(xb @ weights + bias)  # @ is matrix multiplication
###############################################################################
# Backprop  by hand

# initiate variables
bs = 64  # batch size
xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]
print("batch shapes: ", xb.shape, yb.shape)

logits = xb @ weights + bias # hidden layer pre-activation
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
probs = counts * counts_sum_inv
preds = probs.log()
loss = -preds[range(bs), yb].mean()

for p in [weights, bias]:
    p.grad = None

for t in [logits,logit_maxes,norm_logits,counts,counts_sum,counts_sum_inv,probs,preds]:
    t.retain_grad()

loss.backward()

dpreds = torch.zeros_like(preds)
dpreds[range(bs), yb] = -1.0/bs
dprobs = (1.0 / probs) * dpreds
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
dcounts = counts_sum_inv * dprobs
dcounts_sum = (-counts_sum ** -2) * dcounts_sum_inv
dcounts += torch.ones_like(counts) * dcounts_sum
dnorm_logits = counts * dcounts
dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits += torch.nn.functional.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
dweights = xb.T @ dlogits
dbias = dlogits.sum(0)

cmp('preds', dpreds, preds)
cmp('probs', dprobs, probs)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
cmp('counts_sum', dcounts_sum, counts_sum)
cmp('counts', dcounts, counts)
cmp('norm_logits', dnorm_logits, norm_logits)
cmp('logit_maxes', dlogit_maxes, logit_maxes)
cmp('logits', dlogits, logits)
cmp('weights',dweights,weights)
cmp('bias',dbias,bias)

###############################################################################
# As you see, the ``preds`` tensor contains not only the tensor values, but also a
# gradient function. We'll use this later to do backprop.
#
# Let's implement negative log-likelihood to use as the loss function
# (again, we can just use standard Python):


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


loss_func = nll

###############################################################################
# Let's check our loss with our random model, so we can see if we improve
# after a backprop pass later.

yb = y_train[0:bs]
print(loss_func(preds, yb))


###############################################################################
# Let's also implement a function to calculate the accuracy of our model.
# For each prediction, if the index with the largest value matches the
# target value, then the prediction was correct.

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


###############################################################################
# Let's check the accuracy of our random model, so we can see if our
# accuracy improves as our loss improves.

print(accuracy(preds, yb))

###############################################################################
# We can now run a training loop.  For each iteration, we will:
#
# - select a mini-batch of data (of size ``bs``)
# - use the model to make predictions
# - calculate the loss
# - ``loss.backward()`` updates the gradients of the model, in this case, ``weights``
#   and ``bias``.
#
# We now use these gradients to update the weights and bias.  We do this
# within the ``torch.no_grad()`` context manager, because we do not want these
# actions to be recorded for our next calculation of the gradient.  You can read
# more about how PyTorch's Autograd records operations
# `here <https://pytorch.org/docs/stable/notes/autograd.html>`_.
#
# We then set the
# gradients to zero, so that we are ready for the next loop.
# Otherwise, our gradients would record a running tally of all the operations
# that had happened (i.e. ``loss.backward()`` *adds* the gradients to whatever is
# already stored, rather than replacing them).
#
# .. tip:: You can use the standard python debugger to step through PyTorch
#    code, allowing you to check the various variable values at each step.
#    Uncomment ``set_trace()`` below to try it out.
#


lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():  # we don't want weight updates added to graph
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

###############################################################################
# That's it: we've created and trained a minimal neural network (in this case, a
# logistic regression, since we have no hidden layers) entirely from scratch!
#
# Let's check the loss and accuracy and compare those to what we got
# earlier. We expect that the loss will have decreased and accuracy to
# have increased, and they have.

print(loss_func(model(xb), yb), accuracy(model(xb), yb))
