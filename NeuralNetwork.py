import random
# import random.randint
from time import sleep

import jax
# JAX's syntax is (for the most part) same as NumPy's!
# There is also a SciPy API support (jax.scipy)
import jax.numpy as jnp
# Special transform functions (we'll understand what these are very soon!)
from jax import grad
import random
import numpy as np


# JAX's low level API
# (lax is just an anagram for XLA, not completely sure how they came up with name JAX)


def sigmoid(x):
    exped = jnp.exp(-x)
    # exped = exped.at[exped > 100000000].set(100000000)
    return 2 * (1 / (1 +exped)) - 1


def sigmoidPrime(x):
    exped = jnp.exp(-x)
    # exped = exped.at[exped > 100000000].set(1000)
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
    def __init__(self, layerSizes, learningRate=0.1, penaltyForNotLearning=1, activaction=jnp.tanh,
                 derivativeOfActivation=sigmoidPrime):
        """ layerSizes :list of ints, the values are the number of nodes
                        in the layer. Includes the input and output layer
        """
        self.weights = []  ## list of matrices that is the connections between the nodes
        for i in range(len(layerSizes)):
            try:
                self.weights.append(jax.random.normal(jax.random.PRNGKey(100), shape=(layerSizes[i], layerSizes[i + 1])))
            except IndexError:
                pass

        ## init class vars
        self.learningRate = learningRate
        self.penaltyForNotLearning = penaltyForNotLearning
        self.activaction = activaction
        self.derivativeOfActivation = derivativeOfActivation
        self.layerSizes = layerSizes
        self.latestGradSteps = [self.weights, ] * 10  # last 10 weight grads (just init so don't get shape error)
        self.ouputFunction = activaction

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

    def getError(self,x,y, weights, lazyness=0):
        yhat = self.forwardPass(x, weights)
        pastWeightGrads1 = self.latestGradSteps[-2]
        pastWeightGrads2 = self.latestGradSteps[-1]
        diffSum = 0
        for layerIndex in range(len(pastWeightGrads1)):
            diffSum += jnp.linalg.norm(
                pastWeightGrads1[layerIndex] / jnp.linalg.norm(pastWeightGrads1[layerIndex]) - pastWeightGrads2[
                    layerIndex] / jnp.linalg.norm(pastWeightGrads2[layerIndex]))
        diffSum /= layerIndex + 1

        error = jnp.mean((self.ouputFunction(yhat) - y) ** 2)
        if not jnp.isnan(diffSum) and diffSum != 0:
            error /= diffSum
        return error

    def train(self, x, y, batchSize = 50000000, numIter=2, lazyness=0):
        gradFunc = grad(self.getError, argnums=2)
        for i in range(numIter):
            printIter = i % 100 == 0
            # if batchSize != -1
            numBatches = (x.shape[0]/batchSize)
            randBatchNumber = random.randint(0,round(numBatches))
            batchX = x[randBatchNumber*batchSize:(randBatchNumber + 1)*batchSize]  # probably not the best strat
            batchY = y[randBatchNumber*batchSize:(randBatchNumber + 1)*batchSize]

            ## update
            weightGrads = gradFunc(batchX,batchY,self.weights)
            # print(jax.make_jaxpr(grad(self.getError)))

            weightGradsTimesLearningRate = []


            for index in range(len(self.weights)):
                # momentumGrad = jnp.zeros_like(self.weights[index])
                # for pastWeightGrad in self.avgPastWeights:
                #     momentumGrad += pastWeightGrad[index]
                # momentumGrad /= len(self.avgPastWeights)
                # weightGrads[index] = weightGrads[index].at[weightGrads[index] > 100000000].set(0)
                # weightGrads[index] = weightGrads[index].at[jnp.isnan(weightGrads[index])].set(0)
                self.weights[index] -= (weightGrads[index] * self.learningRate) #+ momentumGrad
                if printIter:
                    weightGradsTimesLearningRate.append(weightGrads[index] * self.learningRate)


            # ranNum = random.randint(0,yhat.shape[1]-1)
            if printIter:
                self.latestGradSteps.pop(0)
                self.latestGradSteps.append(weightGradsTimesLearningRate)
                for pastWeightGradIndex in range(len(self.latestGradSteps) - 1):
                    pastWeightGrads1 = self.latestGradSteps[pastWeightGradIndex]
                    pastWeightGrads2 = weightGradsTimesLearningRate
                    diffSum = 0
                    for layerIndex in range(len(pastWeightGrads1)):
                        diffSum += jnp.linalg.norm(pastWeightGrads1[layerIndex]/jnp.linalg.norm(pastWeightGrads1[layerIndex]) - pastWeightGrads2[layerIndex]/jnp.linalg.norm(pastWeightGrads2[layerIndex]))
                    diffSum /= layerIndex + 1
                    print("gradient distances", len(self.latestGradSteps) - pastWeightGradIndex - 1, "iterations away:", diffSum)
                print("----------")
                yhat = self.forwardPass(x, lazyness=lazyness)
                error = jnp.sum((self.ouputFunction(yhat) - y) ** 2)
                print("iteration:", i, "error", error)
                if jnp.isnan(error):
                    print("error is nan, stopping")
                    break
                print(jnp.argmax(self.ouputFunction(yhat), axis=1))
                print(jnp.argmax(y, axis=1))
                print(jnp.sum(jnp.argmax(self.ouputFunction(yhat), axis=1) == jnp.argmax(y, axis=1)) / y.shape[0])
                # print(self.ouputFunction(yhat)[ranNum],jnp.argmax(yhat[ranNum]))
                # print(y[ranNum], jnp.argmax(y[ranNum]))
                print("__________")
