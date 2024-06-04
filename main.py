import argparse
import math
import keras
import tensorflow as tf
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from model.builder import Builder
from model.layers import setModelLayersHook, setInputShapeHook
from tuner.bayesopt import BayesOpt, Continuous, Discrete, OptimizableFunction


DATA_FILE = "./data/processed.csv"
TARGET_PREDICTION_IDX = -1 # update accordingly with column idx in data used as target

TEST_SPLIT = .2


def loadDataRaw() -> np.ndarray:
    data = pd.read_csv(DATA_FILE)
    
    return data.to_numpy()


def loadDataSet():
    data = loadDataRaw()
    n = data.shape[0]
    trainSplit = math.trunc(n * (1 - TEST_SPLIT))
    
    trainingData = data[:trainSplit,:]
    #! we subtract 200 (max window size) as a hack to ensure we get enough data to test with
    testingData = data[trainSplit - 200:, :]
    
    return trainingData, testingData


def createDataset(data: np.ndarray, windowSize: int) -> tf.data.Dataset:
    X = np.delete(data[: -windowSize, :], TARGET_PREDICTION_IDX, axis=1) # set X offset and delete target column
    Y = data[windowSize:, TARGET_PREDICTION_IDX]
    return keras.utils.timeseries_dataset_from_array(
        X,
        Y,
        windowSize,
    )


def compileModel(learningRate: float,
              windowSize: int,
              featureSize: int,
              lstmLayers: int,
              lstmUnits: int,
              denseUnits: int):
    builder = Builder()
    builder.addPreCompileHook(
        setInputShapeHook(windowSize, featureSize)
    )
    builder.addPreCompileHook(
        setModelLayersHook(lstmLayers, lstmUnits, denseUnits)
    )
    model = builder.build(learningRate)
    
    return model


def train(model: keras.Model,
          dataset: tf.data.Dataset,
          epochs: int) -> keras.Model:
    model.fit(x=dataset, epochs=epochs, shuffle=False)

    return model


def evaluate(model: keras.Model, dataset: tf.data.Dataset) -> float:
    result = model.evaluate(dataset, return_dict=True)
    
    return result["loss"]


def createOptimizableFunction(trainingData: np.ndarray, testingData: np.ndarray) -> OptimizableFunction:
    featureSize = trainingData.shape[1] - 1

    def f(windowSize: int,
          lstmLayers,
          lstmUnits: int,
          denseUnits: int,
          epochs: int,
          learningRate: float) -> float:
        trainDataset = createDataset(trainingData, windowSize)
        params = {
                "windowSize": windowSize,
                "lstmLayers": lstmLayers,
                "lstmUnits": lstmUnits,
                "denseUnits": denseUnits,
                "epochs": epochs,
                "learningRate": learningRate,
            }
        print(f"INFO: training model with hyperparameters: {params}")

        model = compileModel(learningRate, windowSize, featureSize, lstmLayers, lstmUnits, denseUnits)
        model = train(model, trainDataset, epochs)
        print("INFO: the model was successfully trained")

        testDataset = createDataset(testingData, windowSize)
        loss = evaluate(model, testDataset)
        print(f"INFO: model loss over the test dataset: {loss}")

        return loss

    return f


def main(spaceFillingSize, sampleBudgetSize):
    trainingData, testingData = loadDataSet()

    featureSize = trainingData.shape[1] - 1 # remove one for the target column
    tuner = BayesOpt(spaceFillingSize=spaceFillingSize, sampleBudgetSize=sampleBudgetSize)
    tuner.RegisterPredictor("windowSize", Discrete(100, 200))
    tuner.RegisterPredictor("lstmLayers", Discrete(1, 5))
    tuner.RegisterPredictor("lstmUnits", Discrete(128, featureSize))
    tuner.RegisterPredictor("denseUnits", Discrete(128, featureSize))
    tuner.RegisterPredictor("epochs", Discrete(1, 25))
    tuner.RegisterPredictor("learningRate", Continuous(0, 1))
    tuner.RegisterTargetFunction(createOptimizableFunction(trainingData, testingData))

    print(f"INFO: Parameter set that minimizes MSE over test dataset: {tuner.Fit()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the real time LSTM architecture and search for a set of hyperparameters via Bayes Opt.")
    parser.add_argument("-spaceFilling", default=30, type=int,
                    help="Number of times a random set of hyperparameters is tested to \"fill\" regions of the search space")
    parser.add_argument("-sampleBudget", default=50, type=int,
                    help="Computation budget for the search procedure. It controls the total amount of times the LSTM model is trained. It must be greater than spaceFilling.")
    args = parser.parse_args()
    
    main(args.spaceFilling, args.sampleBudget)
