NUM_SAMPLES_TO_READ = 240000  # everything
SLEEP_TIME = 0.001
MAKE_ONE_HOT_ENCODING = True
from NeuralNetwork import NeuralNetwork
import numpy as np
import sklearn.datasets as datasets
from time import time
import tensorflow as tf
import jax.numpy as jnp
tf.config.set_visible_devices([], device_type='GPU')
# JAX_DEBUG_NANS=True
# from jax import config
# config.update("jax_debug_nans", True)
import cProfile
import re
# import matplotlib.pyplot as plt
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

prof = cProfile.Profile()
prof.enable()
# Press the green button in the gutter to run the script.
def importRNAData(fileName):
    firstLine = True
    with open(fileName) as dataFile:
        reactivityFirstIndex = -1
        reactivityLastIndex = -1
        indexesOfReactivity = []
        for line in dataFile:
            cols = line.split(',')
            if firstLine:
                for index in range(len(cols)): # there's also "experiment_type,dataset_name,reads,signal_to_noise,SN_filter" that might be useful
                    if cols[index] == 'sequence_id':
                        indexOfSeqId = index
                    elif cols[index] == 'sequence':
                        indexOfSeqNucs = index
                    elif "reactivity_" in cols[index] and not ("error" in cols[index].split("_")):
                        if reactivityFirstIndex == -1:
                            reactivityFirstIndex = index
                        reactivityLastIndex = index
                firstLine = False
                # continue
            seqId = cols[indexOfSeqId]
            seqNucs = cols[indexOfSeqNucs]
            reactivity = cols[reactivityFirstIndex:reactivityLastIndex]
            print(reactivity)

if __name__ == '__main__':

    # importRNAData("train_data.csv") # TODO: finish later

    t1 = time()
    ## test neural network
    hiddenLayerSizes = [512, 256, 256]

    ## test weight sizes
    # print("size", len(nn.weights))

    # dataset = datasets.load_iris()
    # dataset = datasets.load_digits()
    # xTrain = dataset["data"]
    # yTrain = dataset["target"]
    train, test = tf.keras.datasets.mnist.load_data()
    xTrain = jnp.array(train[0])
    xTrain = jnp.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1]*xTrain.shape[2]))
    yTrain = jnp.array(train[1])
    # dataset = tf.data.
    # print(dataset.keys())
    # data = np.random.uniform(size=[1,10])
    # nn.forwardPass(digits[0])

    numOutputGroups = max(yTrain) + 1
    oneHotTarget = np.zeros(shape=(yTrain.shape[0], numOutputGroups)) # magic number
    hiddenLayerSizes.append(numOutputGroups) # add output layers
    hiddenLayerSizes = [xTrain.shape[1]] + hiddenLayerSizes

    if MAKE_ONE_HOT_ENCODING:
        i = -1
        for answer in yTrain:
            i += 1
            oneHotTarget[i, answer] = 1
        yTrain = oneHotTarget


    nn = NeuralNetwork(hiddenLayerSizes)
    assert len(nn.weights) == len(hiddenLayerSizes) - 1
    for i in range(len(nn.weights)):
        assert nn.weights[i].shape[0] == hiddenLayerSizes[i] ## is input size right
        assert nn.weights[i].shape[1] == hiddenLayerSizes[i + 1] ## is output size right

    # nn.learningRate = 0.5
    # nn.train(dataset["data"][:NUM_SAMPLES_TO_READ], oneHotTarget[:NUM_SAMPLES_TO_READ], numIter=5, lazyness=SLEEP_TIME)
    nn.learningRate = 0.01
    nn.train(xTrain[:NUM_SAMPLES_TO_READ], yTrain[:NUM_SAMPLES_TO_READ], numIter=100, lazyness=SLEEP_TIME)
    nn.learningRate = 0.000001
    nn.train(xTrain[:NUM_SAMPLES_TO_READ], yTrain[:NUM_SAMPLES_TO_READ], numIter=100000, lazyness=SLEEP_TIME)
    nn.learningRate = 0.001
    print('increase')
    nn.train(xTrain[:NUM_SAMPLES_TO_READ], yTrain[:NUM_SAMPLES_TO_READ], numIter=1000, lazyness=SLEEP_TIME)
    # x = np.zeros(shape=(2,4,5,6,7,))
    # # print(np.matmul(x, x.transpose()).shape)
    # # for i in range(5):
    # print("transpos",i, "times")
    # print(x.shape)
    # x = np.transpose(x,axes=(4,3,2,1,0))
    # print(x.shape)
    print("time",time()-t1)
prof.disable()
prof.create_stats()
prof.print_stats()


