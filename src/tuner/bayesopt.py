from typing import Callable, Dict, List

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from tuner.acquisition import ei


OptimizableFunction = Callable[..., float]


class Predictor:
    def RandomSample(self) -> int | float:
        raise NotImplementedError


class Discrete(Predictor):
    def __init__(self, min: int, max: int):
        self.choices = np.arange(start=min, stop=max + 1).astype(int)

    def RandomSample(self, size: int = None) -> int | List[int]:
        assert isinstance(size, int) or size is None
        
        sample = np.random.choice(self.choices, size=size)
        if size is None:
            return int(sample)
        else:
            return [int(x) for x in sample]


class Continuous(Predictor):
    def __init__(self, min: float, max: float) -> None:
        self.min = min
        self.max = max

    def RandomSample(self, size: int = None) -> float | List[float]:
        assert isinstance(size, int) or size is None

        return np.random.uniform(self.min, self.max, size)


class BayesOpt:
    __predictors = {}
    __params = []
    __outputs = []
    __keys = []
    

    def __init__(self,
                 acq = ei,
                 spaceFillingSize:int = 10,
                 sampleBudgetSize:int = 30,
                 nIter: int = 500):
        assert spaceFillingSize <= sampleBudgetSize

        self.__sampleBudgetSize = sampleBudgetSize
        self.__spaceFillingSize = spaceFillingSize

        self.__gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
        )
        self.__nIter = nIter
        self.__acq = acq


    def RegisterPredictor(self, name: str, predictor: Predictor)-> None:
        self.__predictors[name] = predictor
        self.__keys.append(name)


    def RegisterTargetFunction(self, f: OptimizableFunction) -> None:
        self.__target = f


    def __evalTarget(self, params: List) -> float:
        if len(params) != len(self.__keys): raise ValueError(
            f"Size of array {len(params)}) is different than the " +
                f"expected number of parameters ({len(self.keys)}).")

        inputs = dict(zip(self.__keys, params))
        output = self.__target(**inputs)

        self.__params.append(params)
        self.__outputs.append(output)

        return output
    

    def __randomParamSample(self, size = None) -> List:
        params = []

        for k in self.__keys:
            predictor: Predictor = self.__predictors[k]
            sample = predictor.RandomSample(size)
            params.append(sample)

        if size == None:
            return params # no need to transpose

        return [[params[j][i] for j in range(len(params))] for i in range(len(params[0]))] # manually transpose to avoid losing dtypes


    def __min(self) -> float:
        if len(self.__outputs) == 0: raise ValueError("No items in output space found. Make sure to fill the space.")
        return np.min(self.__outputs)


    def __suggest(self) -> np.ndarray:
        sample = self.__randomParamSample(self.__nIter)
        y = self.__acq(sample, self.__gp, self.__min())
        return sample[y.argmin()]
        
    
    def __fillSpace(self) -> None:
        for _ in range(self.__spaceFillingSize):
            inputs = self.__randomParamSample()
            self.__evalTarget(inputs)


    def Fit(self) -> Dict:
        self.__fillSpace()
        self.__gp = self.__gp.fit(np.array(self.__params), np.array(self.__outputs))

        for _ in range(self.__spaceFillingSize, self.__sampleBudgetSize):
            inputs = self.__suggest()
            self.__evalTarget(inputs)
            self.__gp = self.__gp.fit(np.array(self.__params), np.array(self.__outputs))

        return dict(zip(self.__keys, self.__params[np.argmin(self.__outputs)]))
