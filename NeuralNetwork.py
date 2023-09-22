import random
# import random.randint
from time import sleep

import jax
# JAX's syntax is (for the most part) same as NumPy's!
# There is also a SciPy API support (jax.scipy)
import jax.numpy as jnp
# Special transform functions (we'll understand what these are very soon!)
from jax import grad
from jax import random


# JAX's low level API
# (lax is just an anagram for XLA, not completely sure how they came up with name JAX)


def sigmoid(x):
    return 2 * (1 / (1 + jnp.exp(-x))) - 1


def sigmoidPrime(x):
    try:
        exped = jnp.exp(-x)
    except OverflowError:
        return 0.0
    return 2 * ((1 + exped) ** (-2)) * exped


# def softmax(inputs, axis=0):
#     expedInputs = np.exp(inputs)
#     denominator = np.sum(expedInputs, axis=axis)
#     for _ in range(expedInputs.shape[axis]):
#         denominator = np.concatenate((denominator, denominator), axis) # prob need backwards axis
#     expedInputs /= denominator
#     return expedInputs
def derivativeOfSoftmax(inputs, indexOfNumerator, indexOfDerivingVar):
    result = inputs[:, indexOfNumerator]
    if indexOfNumerator == indexOfDerivingVar:
        result *= (1 - inputs[:, indexOfDerivingVar])
    else:
        result *= -inputs[:, indexOfDerivingVar]
    return result


def relu(x):
    return jnp.max(0, x)


# def leakyRelu(x):


class NeuralNetwork():
    def __init__(self, layerSizes, learningRate=0.1, penaltyForNotLearning=1, activaction=sigmoid,
                 derivativeOfActivation=sigmoidPrime):
        """ layerSizes :list of ints, the values are the number of nodes
                        in the layer. Includes the input and output layer
        """
        self.weights = []  ## list of matrices that is the connections between the nodes
        for i in range(len(layerSizes)):
            try:
                self.weights.append(random.normal(jax.random.PRNGKey(100), shape=(layerSizes[i], layerSizes[i + 1])))
            except IndexError:
                pass

        ## init class vars
        self.learningRate = learningRate
        self.penaltyForNotLearning = penaltyForNotLearning
        self.activaction = activaction
        self.derivativeOfActivation = derivativeOfActivation
        self.layerSizes = layerSizes
        self.avgPastWeights = []  # rolling avg of past weights (give slight gradient away from their)
        self.ouputFunction = activaction
        self.x = []
        self.y = []

    def forwardPass(self, x, weights=[], lazyness=0):
        """ x : is input, has shape of
            :returns: output of NeuralNetwork and deltas ="""
        if weights == []:
            weights = self.weights
        deltas = []
        if x.shape[1] != weights[0].shape[0]:
            print("ERROR: input is wrong size")
            raise Exception("ERROR: input is wrong size\ninput size: " + str(x.shape[1]) + "\nweight size: " +
                            str(weights[0].shape[0]))
        output = x
        for i in range(len(weights)):  ## for each layer of neural network
            if lazyness > 0:
                sleep(lazyness)
            weightLayer = weights[i]
            outputBeforeActivation = jnp.matmul(output, weightLayer)
            output = self.activaction(outputBeforeActivation)  # pass through the layer
        # print(output)
        # print(deltas)
        return output

    def getError(self, weights, lazyness=0):
        yhat = self.forwardPass(self.x, weights)
        error = jnp.sum((self.ouputFunction(yhat) - self.y) ** 2)
        return error

    def train(self, x, y, numIter=2, lazyness=0):
        self.x = x  # probably not the best strat
        self.y = y
        for i in range(numIter):
            yhat = self.forwardPass(x, lazyness=lazyness)
            error = jnp.sum((self.ouputFunction(yhat) - y) ** 2)
            print("iteration:", i, "error", error)
            if jnp.isnan(error):
                print("error is nan, stopping")
                break
            ## update
            weightGrads = grad(self.getError)(self.weights)
            for index in range(len(self.weights)):
                self.weights[index] -= weightGrads[index] * self.learningRate

            # ranNum = random.randint(0,yhat.shape[1]-1)

            print(jnp.argmax(self.ouputFunction(yhat), axis=1))
            print(jnp.argmax(y, axis=1))
            print(jnp.sum(jnp.argmax(self.ouputFunction(yhat), axis=1) == jnp.argmax(y, axis=1)) / y.shape[0])
            # print(self.ouputFunction(yhat)[ranNum],jnp.argmax(yhat[ranNum]))
            # print(y[ranNum], jnp.argmax(y[ranNum]))
            print("__________")
