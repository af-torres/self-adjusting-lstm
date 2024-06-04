import keras
from model.builder import Builder
import numpy as np
import pandas as pd

from model.layers import setModelLayersHook, setInputShapeHook

def load_data() -> np.ndarray:
    DATA_FILE = "./data/processed.csv"
    data = pd.read_csv(DATA_FILE)
    
    return data.to_numpy()


def compile_model(learning_rate: float,
              windowSize: int,
              featureSize: int,
              lstm_layers: int,
              lstm_units: int,
              dense_units: int):
    builder = Builder()
    builder.addPreCompileHook(
        setInputShapeHook(windowSize, featureSize)
    )
    builder.addPreCompileHook(
        setModelLayersHook(lstm_layers, lstm_units, dense_units)
    )
    model = builder.build(learning_rate)
    
    return model


def train(model: keras.Model,
          X: np.ndarray,
          Y: np.ndarray,
          windowSize: int,
          epochs: int) -> keras.Model:
    dataset = keras.utils.timeseries_dataset_from_array(
        X,
        Y,
        windowSize
    )

    history = model.fit(x=dataset, epochs=epochs, shuffle=False)

    return model


def main():
    TARGET_PREDICTION_IDX = -1 # update accordingly with what we want to predict

    data = load_data()
    featureSize = data.shape[1]
    
    windowSize = 200 # make this optimizable
    lstm_layers = 1 # make this optimizable
    lstm_units = featureSize # make this optimizable
    dense_units = featureSize # make this optimizable
    epochs = 10 # make this optimizable
    learning_rate = 0.001 # make this optimizable
    
    X = data[: -windowSize, :]
    Y = data[windowSize:, TARGET_PREDICTION_IDX]

    model = compile_model(learning_rate, windowSize, featureSize, lstm_layers, lstm_units, dense_units)
    model = train(model, X, Y, windowSize, epochs)


if __name__ == "__main__":
    main()
