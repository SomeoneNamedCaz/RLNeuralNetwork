from NeuralNetwork import NeuralNetwork
import numpy as np
import sklearn.datasets as datasets
import cProfile
import re
# import matplotlib.pyplot as plt
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# from NeuralNetwork import NeuralNetwork
import numpy as np
import sklearn.datasets as datasets
from time import time
# JAX_DEBUG_NANS=True
# from jax import config
# config.update("jax_debug_nans", True)
import cProfile
import re
# import matplotlib.pyplot as plt
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# prof = cProfile.Profile()
# prof.enable()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t1 = time()
    ## test neural network
    hiddenLayerSizes = [30,20,10]

    ## test weight sizes
    # print("size", len(nn.weights))

    # dataset = datasets.load_iris()
    dataset = datasets.load_digits()
    dataset.keys()
    # data = np.random.uniform(size=[1,10])
    # nn.forwardPass(digits[0])

    numOutputGroups = max(dataset["target"]) + 1
    oneHotTarget = np.zeros(shape=(dataset["target"].shape[0], numOutputGroups)) # magic number
    hiddenLayerSizes.append(numOutputGroups) # add output layers
    hiddenLayerSizes = [dataset["data"].shape[1]] + hiddenLayerSizes
    i = -1
    for answer in dataset["target"]:
        i += 1
        oneHotTarget[i, answer] = 1

    NUM_SAMPLES_TO_READ = 2400 # everything
    SLEEP_TIME = 0.0

    nn = NeuralNetwork(hiddenLayerSizes)
    assert len(nn.weights) == len(hiddenLayerSizes) - 1
    for i in range(len(nn.weights)):
        assert nn.weights[i].shape[0] == hiddenLayerSizes[i] ## is input size right
        assert nn.weights[i].shape[1] == hiddenLayerSizes[i + 1] ## is output size right


    # nn.learningRate = 0.5
    # nn.train(dataset["data"][:NUM_SAMPLES_TO_READ], oneHotTarget[:NUM_SAMPLES_TO_READ], numIter=5, lazyness=SLEEP_TIME)
    nn.learningRate = 0.1
    nn.train(dataset["data"][:NUM_SAMPLES_TO_READ], oneHotTarget[:NUM_SAMPLES_TO_READ], numIter=1000, lazyness=SLEEP_TIME)
    nn.learningRate = 0.01
    nn.train(dataset["data"][:NUM_SAMPLES_TO_READ], oneHotTarget[:NUM_SAMPLES_TO_READ], numIter=1000, lazyness=SLEEP_TIME)
    # nn.learningRate = 0.001
    # nn.train(dataset["data"][:NUM_SAMPLES_TO_READ], oneHotTarget[:NUM_SAMPLES_TO_READ], numIter=20, lazyness=SLEEP_TIME)
    # x = np.zeros(shape=(2,4,5,6,7,))
    # # print(np.matmul(x, x.transpose()).shape)
    # # for i in range(5):
    # print("transpos",i, "times")
    # print(x.shape)
    # x = np.transpose(x,axes=(4,3,2,1,0))
    # print(x.shape)
    print(time()-t1)
# prof.disable()
# prof.create_stats()
# prof.print_stats()


