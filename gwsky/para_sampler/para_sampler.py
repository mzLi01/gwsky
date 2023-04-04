from abc import ABCMeta, abstractmethod

from ..typing import ParaNameList, ParameterVector


class ParameterSampler(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def keys(self) -> ParaNameList:
        pass

    @abstractmethod
    def sample(self) -> ParameterVector:
        pass
