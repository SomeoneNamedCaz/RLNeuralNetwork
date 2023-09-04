import numpy as np
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidPrime(x):
    exped = np.exp(-x)
    return ((1+exped)**2)*exped

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
    def forwardPass(self, x):
        """ x : is input, has shape of
            :returns: output of NeuralNetwork and deltas ="""
        deltas = []
        if x.shape[1] != self.weights[0].shape[0]:
            print("ERROR: input is wrong size")
            sys.exit(1)
        output = x
        for i in range(len(self.weights)): ## for each layer of neural network
            print(i)
            weightLayer = self.weights[i]
            outputBeforeActivation = np.dot(output,weightLayer)


            # calculate deltas (so we can train)
            deltas.append(np.zeros(shape=weightLayer.shape + (self.weights[-1].shape[1],))) ## init deltas with respect to first layer weights

            for a in range(weightLayer.shape[0]):
                for b in range(weightLayer.shape[1]):
                    for k in range(weightLayer.shape[1]): # FIX: k needs to have different max values depending on layer
                        if b == k:
                            print("self.derivativeOfActivation(outputBeforeActivation)",self.derivativeOfActivation(outputBeforeActivation))
                            print(output[:,a])
                            deltas[-1][a][b][k] = self.derivativeOfActivation(outputBeforeActivation)[:,k] * output[:,a]
            for deltaLayer in deltas[:-1]: ## update earlier layers
                for a in range(weightLayer.shape[0]):
                    for b in range(weightLayer.shape[1]):
                        for k in range(weightLayer.shape[1]):
                            print("start")
                            print("self.derivativeOfActivation(outputBeforeActivation)[:,k]",self.derivativeOfActivation(outputBeforeActivation)[:,
                                                      k])
                            print("weightLayer[:,k]",weightLayer.shape,weightLayer[:,k].shape)
                            print("deltaLayer[a][b]",deltaLayer.shape,deltaLayer[a,b].shape)
                            print("np.dot(weightLayer[:,k], deltaLayer[a][b])",np.dot(weightLayer[:,k], deltaLayer[a,b]))
                            deltaLayer[a][b][k] = self.derivativeOfActivation(outputBeforeActivation)[:,
                                                      k] * np.dot(weightLayer[:,k], deltaLayer[a,b])

            output = self.activaction(outputBeforeActivation)  # pass through the layer
        print(output)
        print(deltas)
        return output, deltas