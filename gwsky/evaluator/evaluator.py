from abc import ABCMeta, abstractmethod

from ..typing import ParameterDict, ParaNameList, FisherMatrix


class BaseEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def snr(self, parameters: ParameterDict) -> float:
        pass

    @abstractmethod
    def fisher(self, parameters: ParameterDict) -> FisherMatrix:
        pass

    @property
    @abstractmethod
    def fisher_parameters(self):
        pass
