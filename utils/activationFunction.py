
import numpy as np

class LogSig:
    """
    Logarithmic sigmoid transfer function.
    """
    def __call__(self, x):
        return 1/(1+np.exp(-x))

    def deriv(self, x, y):
        """
        Derivative of transfer function LogSig.
        """
        return y * (1 - y)


class TanSig:
    """
    Hyperbolic tangent sigmoid transfer function.
    """
    def __call__(self, x):
        return  np.tanh(x)

    def deriv(self, x, y):
        """
        Derivative of transfer function TanSig.
        """
        return 1.0 - np.square(y)


class HardLim:
    """
    Hard limit transfer function
    """
    def __call__(self, x):
        return (x > 0) * 1.0

    def deriv(self):
        pass


class HardLims:
    """
    Symmetric hard limit transfer function.
    """
    def __call__(self, x):
        return (x > 0) * 2.0 - 1.0
        #return -1 if x<0 else +1



# Testing the classes
def main():
    x = np.array([-5, -0.1, 0, 0.1, 100])
    o = HardLims()
    print o(x)

if __name__ == "__main__":
    main()

