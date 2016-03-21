
import os

from preProceso import loadFile
from preProceso import build_matrices
from recurrentNetworks import RNNNumpy


PATH = os.getcwd()
FILENAME = 'reddit-comments-2015-08.csv'
FILENAME = os.path.join(PATH, FILENAME)

## Load the dataset
sentences = loadFile(FILENAME)
X_train, y_train = build_matrices(sentences)

# RNN
o = RNNNumpy(word_dim=800)
o.forward_propagation(X_train[10])

