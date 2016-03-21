
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

class NeuronalNetwork:
    def __init__(self, layers, activation="tanh"):
        if activation == "sigmoid":
            self.activation= sigmoid
            self.activation_prime= sigmoid_prime
        if activation == "tanh":
            self.activation= tanh
            self.activation_prime= tanh_prime
        # Set weights
        self.weights= []
        # layers = [2,2,1]
        # range of weights (-1,1)
        # input and hidden layers -
        for i in range(1, len(layers)-1):
            r= 2*np.random.random((layers[i-1]+1,layers[i]+1))-1
            self.weights.append(r)
        # oyput layer -
        # I dont like this implementation
        r= 2*np.random.random((layers[i]+1, layers[i+1]))-1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Add column of ones X
        # This is to add the bias unit to the input layer
        ones= np.atleast_2d(np.ones(X.shape[0]))
        X= np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):
            i= np.random.randint(X.shape[0])
            a= [X[i]]

            for l in range(len(self.weights)):
                dot_values= np.dot(a[l], self.weights[l])
                activation= self.activation(dot_values)
                a.append(activation)
            # output layers
            error= y[i] - a[-1]
            deltas= [error * self.activation_prime(a[-1])]
            # we need to begin at the second to last layers
            # (a layer before the output layer)
            for l in range(len(a)-2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
            # reverse
            # [level3->level2] => [level2->level3]
            deltas.reverse()

            # 1. Multiply its output delta and input activation
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.

            for i in range(len(self.weights)):
                layer= np.atleast_2d(a[i])
                delta= np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: print "epochs: ", k

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)
        for l in range(0, len(self.weights)):
            a= self.activation(np.dot(a, self.weights[l]))
            print a
        return a



def net():
    nn= NeuronalNetwork([2,2,1])
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 0])
    nn.fit(X, y)
    for e in X:
        print(e,nn.predict(e))

def main():
    x= np.linspace(-10,10,100)
    plt.figure(1)
    plt.plot(x, sigmoid(x), lw=2, label="Sigmoid")
    plt.plot(x, sigmoid_prime(x), lw=2, label="Sigmoid prime")
    plt.title("Activation Function")
    plt.legend(loc=0); plt.show()

    f, (ax1, ax2)= plt.subplots(2, 1, sharex=True)
    ax1.plot(x, tanh(x), lw=2, label="Tanh")
    ax1.set_title("Activation Function")
    ax1.legend(loc=0)
    ax2.plot(x, tanh_prime(x), lw=2, c="g", label="Tanh prime")
    ax2.legend(loc=0); plt.show()

if __name__=="__main__":
    main()
    net()

