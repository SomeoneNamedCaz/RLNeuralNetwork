import numpy as np
import sys
import random
import tensorflow

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
        self.avgPastWeights = [] # rolling avg of past weights (give slight gradient away from their)
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

            for deltaLayer in deltas: ## for delta corresponding to first layer
                for a in range(deltaLayer.shape[0]):
                    for b in range(deltaLayer.shape[1]): # 103 seconds vs 285
                                # #print("start")
                                # derivative = self.derivativeOfActivation(outputBeforeActivation[:,k])
                                # Wcol = weightLayer[:, k]
                                # deltCol = deltaLayer[a,b, :Wcol.shape[0]]
                                # print(self.derivativeOfActivation(outputBeforeActivation).shape) # (numSamples, next layer size)
                                # print(weightLayer.shape) # (previous layer size, next layer size
                                # print(deltaLayer.shape) # (size of inputs of weights, size of outputs of weights
                                    # (self.weights[delts.indexof[deltlayer], layer after that size, numSamples

                                # (240, 10) (40, 10) (64, 40, 64, 240)
                        allTogether = np.matmul(np.transpose(weightLayer), deltaLayer[a, b, :weightLayer.shape[0]])
                        allTogether = np.append(allTogether, np.zeros(shape=(deltaLayer[a][b].shape[0] - allTogether.shape[0], allTogether.shape[1])),axis=0)
                        deltaLayer[a][b] = allTogether
            # calculate deltas for latest weights
            #           i/x               a   b           k (only going to use part)     num samples
            deltas.append(np.zeros(shape=weightLayer.shape + (max(self.layerSizes),x.shape[0]))) ## init deltas with respect to first layer weights

            # has shape of weightLayer
            deltasOfSameLayer = np.matmul(np.transpose(output), self.derivativeOfActivation(outputBeforeActivation))
            # if self.debugMode:
                #print("doudwx shape", deltasOfSameLayer.shape)
                #print("delts", deltas[-1].shape)
                # deltas = np.transpose(deltas,axes=(3,2,0,1))
                #print("delts", deltas[-1].shape)
            # for sampleNum in deltas[-1].shape[0]:


            # for _ in range(deltas[-1].shape[0]):

            # for a in range(deltas[-1].shape[0]):
            #     for b in range(deltas[-1].shape[1]):
            #         deltasOfSameLayer[a,b]
            # for a in range(deltas[-1].shape[0]):
            #     for b in range(deltas[-1].shape[1]):

                    # if not (np.dot(self.derivativeOfActivation(outputBeforeActivation)[:, b],output[:, a]) == deltasOfSameLayer[a,b]).all():
                    #     print("uh oh")
                    #                                                                                 out of last layer
                    # deltas[-1][:,b,b] = deltasOfSameLayer[:,b]
            deltas[-1] = np.transpose(deltas[-1],axes=(2,1,0,3))
            for b in range(deltas[-1].shape[0]): # now axis 0
                deltas[-1][b] = deltasOfSameLayer
            output = self.activaction(outputBeforeActivation)  # pass through the layer
        # print(output)
        # print(deltas)
        return output, deltas
    def train(self, x, y, numIter=2):

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


            ranNum = random.randint(0,yhat.shape[1]-1)

            print(tensorflow.nn.softmax(yhat[ranNum]))
            print("vs")
            print(y[ranNum])
            print(np.argmax(yhat[ranNum]), np.argmax(y[ranNum]))
            print("__________")