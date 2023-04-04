import numpy as np

from pathos import multiprocessing
from datetime import datetime

from .evaluator import BaseEvaluator
from .result import Result
from .para_sampler import ParameterSampler

from typing import List, Optional
from .typing import ParameterVector, ParameterDict, SampleResult


class Sampler:
    def __init__(self, para_samplers: List[ParameterSampler], default_parameters: ParameterDict):
        self.para_samplers = para_samplers
        self.keys = sum([sampler.keys for sampler in para_samplers], [])

        self.default_parameters = default_parameters

    def sample_points(self) -> ParameterVector:
        samples = sum([sampler.sample() for sampler in self.para_samplers], [])
        return samples

    def get_parameter_from_samples(self, samples: ParameterVector) -> ParameterDict:
        parameters = self.default_parameters.copy()
        parameters.update({name: value for name, value in zip(self.keys, samples)})
        return parameters

    def sample_parameter(self) -> ParameterDict:
        return self.get_parameter_from_samples(self.sample_points())

    def _real_sample(self, evaluator: BaseEvaluator, nsamples: int, njobs: int) -> List[SampleResult]:
        def _sample_wrapper(*args) -> SampleResult:
            np.random.seed()
            samples = self.sample_points()
            parameters = self.get_parameter_from_samples(samples)
            snr = evaluator.snr(parameters)
            fisher = evaluator.fisher(parameters)
            return samples, snr, fisher

        pool = multiprocessing.ProcessPool(nodes=njobs)
        return pool.map(_sample_wrapper, range(nsamples))

    def sample_fisher(self, nsamples: int, evaluator: BaseEvaluator, result_path: str,
                      nprocess: int = 1, max_memory_sample: Optional[int] = None) -> Result:
        """
        run the MC sample, calculate SNR and fisher matrix for sample points

        Args:
            nsamples (int): number of sample points
            evaluator (BaseEvaluator): evaluator instance to calculate SNR and fisher matrix
            result_path (str): path to result file
            nprocess (int, optional): number of processes to use. Defaults to 1.
            max_memory_sample (Optional[int]):
                Max numbers of samples that can be saved in memory.
                All samples are saved to disk and result will be cleared when
                number of samples in result exceeds this value.
                Defaults to None, when no memory limit is presented.

        Returns:
            Result: result instance, containing the last `nsamples`%`max_memory_sample` samples.
        """               
        if max_memory_sample is None:
            max_memory_sample = nsamples + 1  # act as infinity

        start_time = datetime.now()

        result = Result(keys=self.keys, fisher_parameters=evaluator.fisher_parameters)
        for _ in range(nsamples//max_memory_sample):
            samples = self._real_sample(evaluator, max_memory_sample, nprocess)
            result.append_samples(samples)
            result.save_csv(result_path)
            result.clear()
        samples = self._real_sample(evaluator, nsamples%max_memory_sample, nprocess)
        result.append_samples(samples)
        result.save_csv(result_path)

        end_time = datetime.now()
        print(f'Sampled {nsamples} points, used time {end_time-start_time}')

        return result
