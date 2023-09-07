import numpy as np
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x):
    try:
        exped = np.exp(-x)
    except OverflowError:
        return 0.0
    return ((1+exped)**(-2))*exped

def relu(x):
    return np.max(0, x)


# def leakyRelu(x):


class NeuralNetwork():
    def __init__(self, layerSizes, learningRate=0.01, penaltyForNotLearning=1, activaction=sigmoid, derivativeOfActivation=sigmoidPrime):
        """ layerSizes :list of ints, the values are the number of nodes
                        in the layer. Includes the input and output layer
        """
        self.weights = [] ## list of matrices that is the connections between the nodes
        for i in range(len(layerSizes)):
            try:
                self.weights.append(np.random.normal(size=(layerSizes[i], layerSizes[i+1])))
            except IndexError:
                pass



        ## init class vars
        self.learningRate = learningRate
        self.penaltyForNotLearning = penaltyForNotLearning
        self.activaction = activaction
        self.derivativeOfActivation = derivativeOfActivation
        self.layerSizes = layerSizes
    def forwardPass(self, x):
        """ x : is input, has shape of
            :returns: output of NeuralNetwork and deltas ="""
        deltas = []
        if x.shape[1] != self.weights[0].shape[0]:
            print("ERROR: input is wrong size")
            sys.exit(1)
        output = x
        for i in range(len(self.weights)): ## for each layer of neural network
            weightLayer = self.weights[i]
            outputBeforeActivation = np.dot(output,weightLayer)


            # calculate deltas (so we can train)
            # calculate deltas for weights more than 1 layer back
            for deltaLayer in deltas: ## update earlier layers
                for a in range(weightLayer.shape[0]):
                    for b in range(weightLayer.shape[1]):
                        for k in range(deltaLayer.shape[-1]):
                            try:
                                # print("start")
                                derivative = self.derivativeOfActivation(outputBeforeActivation)[:,k]
                                Wcol = weightLayer[:, k]
                                deltCol = deltaLayer[a,b, :Wcol.shape[0]]

                                deltaLayer[a][b][k] = derivative * np.dot(Wcol, deltCol)
                            except IndexError:
                                pass
            # calculate deltas for latest weights
            deltas.append(np.zeros(shape=weightLayer.shape + (max(self.layerSizes),x.shape[0]))) ## init deltas with respect to first layer weights

            for a in range(deltas[-1].shape[0]):
                for b in range(deltas[-1].shape[1]):
                    deltas[-1][a][b][b] = np.dot(self.derivativeOfActivation(outputBeforeActivation)[:,b],output[:,a])


            output = self.activaction(outputBeforeActivation)  # pass through the layer
        # print(output)
        # print(deltas)
        return output, deltas
    def train(self, x, y, numIter=100):

        for i in range(numIter):
            yhat, deltas = self.forwardPass(x)
            error = np.sum((yhat - y) ** 2)
            print("error", error)
            if np.isnan(error):
                print("error is nan, stopping")
                break
            ## update


            weightLayerIndex = -1
            for weightLayer in self.weights:
                weightLayerIndex += 1
                # for deltaLayer in deltas:  ## update earlier layers
                weightChangesWanted = np.zeros_like(weightLayer)
                for a in range(weightLayer.shape[0]):
                    for b in range(weightLayer.shape[1]):
                        for k in range(weightLayer.shape[-1]):
                            # print(deltas[weightLayerIndex][a][b,:yhat.shape[1]].shape)

                            weightChangesWanted[a,b] = np.mean(np.dot(yhat - y, deltas[weightLayerIndex][a][b][:yhat.shape[1]]))
                weightLayer -=  weightChangesWanted * self.learningRate
                # need to condense deltas into self.weights-like array buy dot product with yhat - y

            for j in range(1):
                print(yhat[j])
                print("vs")
                print(y[j])
                print("__________")