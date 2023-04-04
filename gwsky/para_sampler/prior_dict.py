from bilby.core.prior import PriorDict

from .para_sampler import ParameterSampler

from ..typing import ParaNameList, ParameterVector


class PriorDictSampler(ParameterSampler):
    def __init__(self, priors: PriorDict) -> None:
        self.priors = priors
        self._keys = list(priors.keys())

    @property
    def keys(self) -> ParaNameList:
        return self._keys

    def sample(self) -> ParameterVector:
        para_dict = self.priors.sample()
        return [para_dict[key] for key in self.keys]
