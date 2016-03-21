

import numpy as np

def softmax(w, t=1.0):
    e = np.exp(np.array(w)/t)
    dist = e / np.sum(e)
    return dist

class RNNNumpy:

    def __init__(self, word_dim, hidden_dim=100, btt_truncate=4):
        """
        word_dim: the size of the vocabulary.
        hidden_dim: size of the hidden layer.
        btt_truncate:
        """
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.btt_truncate = btt_truncate
        # Random initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
                                   (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),
                                    (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),
                                    (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        """
        x:
        """
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in
        # s because need them later.
        # We add one additional element for the initial hiden, which
        # we set to 0
        s = np.zeros((T+1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.hidden_dim))
        # For each time step ...
        for t in np.arange(T):
            # Note that we are indixing U by x[t]. This is the same as
            # multipluing U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        """
        x:
        """
        # Perform forward propagation and return index of the highest score.
        o, s = self.forward(o, axis=1)

    def calculate_loss(self, x, y):
        """
        x: the prediction words.
        y: the correct words.
        """
        L = 0
        # For each sentence ...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our predict of yhe "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples.
        N = np.sum(len(y_i) for y_i in y)
        return self.calculate_total_loss(x,y)/N

    def bptt(self, x, y):
        """
        x: the prediction words.
        y: the correct words.
        """
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We acumulate the gradients in these variables.
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        # For each output backwards ...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[y], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print ----
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdW[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold = 0.01):
        """
        x: the predict words.
        y: the correct words.
        h:
        error_threshold:
        """
        # Calculate the gradients using backpropagation. We want to checker
        # if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            pass

