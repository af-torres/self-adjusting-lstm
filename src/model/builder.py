
from typing import Callable, List
import keras

PreCompileHook = Callable[[keras.Sequential], keras.Sequential]

class Builder:
    def __init__(self):
        self.__precompile: List[PreCompileHook] = []

    def addPreCompileHook(self, hook: PreCompileHook) -> None:
        self.__precompile.append(hook)

    def build(self, learningRate: float) -> keras.Model:
        model = keras.Sequential()
        for h in self.__precompile:
            model = h(model)

        optimizer = keras.optimizers.Adam(learningRate)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mean_squared_error"])

        return model
