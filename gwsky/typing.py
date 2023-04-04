import numpy as np
from typing import List, Dict, Tuple, Union

ParameterDict = Dict[str, float]
ParameterVector = List[float]
ParaNameList = List[str]

Value = Union[float, complex, np.ndarray]
DerivateDict = Dict[str, Value]
FisherMatrix = np.ndarray

SampleResult = List[Tuple[ParameterVector, float, FisherMatrix]]

SHModeLM = Tuple[int, int]
SHModes = Dict[SHModeLM, complex]