import keras

from model.builder import PreCompileHook


def setInputShapeHook(timeSteps: int, featureSize: int) -> PreCompileHook:
    def hook(model: keras.Sequential):
        model.add(
            keras.layers.Input(shape=(timeSteps, featureSize))
        )

        return model
    
    return hook


def setModelLayersHook(layers: int,
                       lstmUnits: int,
                       denseUnits: int,
                       outputSize: int = 1) -> PreCompileHook:
    def hook(model: keras.Sequential):
        for i in range(0, layers):
            retSeq = False
            if i < layers - 1 : retSeq = True

            l = keras.layers.LSTM(
                lstmUnits,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=retSeq,
            )
            model.add(
                l
            )

        model.add(
            keras.layers.Dense(denseUnits, activation="relu")
        )
        model.add(
            keras.layers.Dense(outputSize, activation="linear")
        )
        
        return model
    
    return hook
