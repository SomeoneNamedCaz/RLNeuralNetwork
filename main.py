from NeuralNetwork import NeuralNetwork
import numpy as np
import sklearn.datasets as datasets
import cProfile
import re
# import matplotlib.pyplot as plt
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

prof = cProfile.Profile()
prof.enable()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ## test neural network
    layerSizes = [64,40,10]
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
    numDigitsToRead = 240
    nn.train(digits["data"][:numDigitsToRead], oneHotTarget[:numDigitsToRead], numIter=2)

    # x = np.zeros(shape=(2,4,5,6,7,))
    # # print(np.matmul(x, x.transpose()).shape)
    # # for i in range(5):
    # print("transpos",i, "times")
    # print(x.shape)
    # x = np.transpose(x,axes=(4,3,2,1,0))
    # print(x.shape)

prof.disable()
prof.create_stats()
prof.print_stats()

# import pstats, io
# # from pstats import SortKey
# s = io.StringIO()
# # sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(prof, stream=s)#.sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())