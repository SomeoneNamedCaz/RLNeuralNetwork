from NeuralNetwork import NeuralNetwork
import numpy as np
import sklearn.datasets as datasets
import cProfile
import re
# import matplotlib.pyplot as plt
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ## test neural network
    layerSizes = [64,64,40,10]
    nn = NeuralNetwork(layerSizes)
    ## test weight sizes
    # print("size", len(nn.weights))

    assert len(nn.weights) == len(layerSizes) - 1
    for i in range(len(nn.weights)):
        assert nn.weights[i].shape[0] == layerSizes[i] ## is input size right
        assert nn.weights[i].shape[1] == layerSizes[i+1] ## is output size right

    digits = datasets.load_digits()
    digits.keys()
    # data = np.random.uniform(size=[1,10])
    # nn.forwardPass(digits[0])
    oneHotTarget = np.zeros(shape=(digits["target"].shape[0],10)) # magic number
    i = -1
    for answer in digits["target"]:
        i += 1
        oneHotTarget[i, answer] = 1
    numDigitsToRead = 1000
    cProfile.run(nn.train(digits["data"][:numDigitsToRead], oneHotTarget[:numDigitsToRead]))
