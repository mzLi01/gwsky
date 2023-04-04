import numpy as np
import healpy as hp

from bilby.gw import WaveformGenerator
from bilby.gw.detector import Interferometer, InterferometerList
from bilby.gw.utils import noise_weighted_inner_product

from .evaluator import BaseEvaluator
from ..converter import RotationConverter
from .utils import derivate_central
from ..utils import rotation_matrix_from_vec

from ..typing import ParameterDict, FisherMatrix, ParaNameList, DerivateDict


class BilbyEvaluator(BaseEvaluator):
    def __init__(self, network: InterferometerList, waveform_generator: WaveformGenerator,
                 fisher_parameters: ParaNameList, step: float = 1e-7) -> None:
        self.network = network
        self.waveform_generator = waveform_generator
        self.network.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.waveform_generator.sampling_frequency,
            duration=self.waveform_generator.duration,
            start_time=self.waveform_generator.start_time)

        self._fisher_parameters = fisher_parameters.copy()
        self._step = step

    def _interferometer_response(self, parameters: ParameterDict, interferometer: Interferometer):
        waveform = self.waveform_generator.frequency_domain_strain(parameters)
        return interferometer.get_detector_response(
            waveform_polarizations=waveform, parameters=parameters)

    def snr(self, parameters: ParameterDict) -> float:
        total_snr_square = sum([
            ifo.optimal_snr_squared(
                self._interferometer_response(parameters, ifo)).real
            for ifo in self.network])
        return np.sqrt(total_snr_square)

    @property
    def fisher_parameters(self):
        return self._fisher_parameters
    
    @fisher_parameters.setter
    def fisher_parameters(self, value: ParaNameList):
        self._fisher_parameters = value

    def _interferometer_signal_derivate(self, interferometer: Interferometer, parameters: ParameterDict,
                                        fisher_parameters: ParaNameList) -> DerivateDict:
        """
        calculate derivate of masked waveform

        Args:
            interferometer (Interferometer)
            parameters (ParameterDict): waveform parameters
            fisher_parameters (ParaNameList): names of parameters to derivate on

        Returns:
            DerivateDict: derivate of frequency-masked waveform
        """
        def waveform(parameters):
            return self._interferometer_response(parameters, interferometer)                 
        return derivate_central(waveform, parameters, deriv_para=fisher_parameters, step=self._step)

    def _signal_derivate_to_fisher_matrix(self, interferometer: Interferometer, signal_derivate:DerivateDict,
                                          fisher_parameters: ParaNameList) -> FisherMatrix:
        signal_array_len = signal_derivate[fisher_parameters[0]].shape[0]
        if signal_array_len == interferometer.frequency_array.shape[0]:
            masked_derivate = {
                para: deriv[interferometer.frequency_mask] for para, deriv in signal_derivate.items()}
        elif signal_array_len == np.sum(interferometer.frequency_mask):
            masked_derivate = signal_derivate
        else:
            raise ValueError('signal derivate shape does not match frequency array of interferometer.')

        psd = interferometer.power_spectral_density_array[interferometer.frequency_mask]

        fisher = np.zeros([len(fisher_parameters)]*2)
        for i, para_i in enumerate(fisher_parameters):
            for j in range(i+1):
                para_j = fisher_parameters[j]
                fisher[i, j] = noise_weighted_inner_product(
                    masked_derivate[para_i], masked_derivate[para_j],
                    power_spectral_density=psd, duration=interferometer.duration).real
                fisher[j, i] = fisher[i, j]
        return fisher

    def _interferometer_fisher(self, interferometer: Interferometer, parameters: ParameterDict,
                               fisher_parameters: ParaNameList):
        signal_derivate = self._interferometer_signal_derivate(
            interferometer, parameters, fisher_parameters)
        return self._signal_derivate_to_fisher_matrix(interferometer, signal_derivate, fisher_parameters)

    def fisher(self, parameters: ParameterDict) -> FisherMatrix:
        return sum([
            self._interferometer_fisher(interferometer, parameters, self.fisher_parameters)
            for interferometer in self.network])


class BilbyEvaluatorRotated(BilbyEvaluator):
    def __init__(self, network: InterferometerList, waveform_generator: WaveformGenerator, fisher_parameters: ParaNameList, step: float = 1e-7) -> None:
        super().__init__(network, waveform_generator, fisher_parameters, step)

    def converter_from_parameter(self, parameters: ParameterDict) -> RotationConverter:
        theta, phi = np.pi/2-parameters['dec'], parameters['ra']
        rot_mat = rotation_matrix_from_vec(
            orig_vec=hp.dir2vec(theta, phi), dest_vec=[1, 0, 0])
        return RotationConverter(rot_mat)

    def _interferometer_signal_derivate(self, interferometer: Interferometer, parameters: ParameterDict, fisher_parameters: ParaNameList) -> DerivateDict:
        converter = self.converter_from_parameter(parameters)
        parameters = converter(parameters)

        def waveform(parameters):
            return self._interferometer_response(converter.reverse_convert(parameters), interferometer)

        return derivate_central(waveform, parameters, deriv_para=fisher_parameters, step=self._step)
