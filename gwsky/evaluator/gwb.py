import numpy as np

from gwbench.network import Network

from .evaluator import BaseEvaluator

from typing import List, Optional
from ..typing import ParameterDict, ParaNameList, FisherMatrix


class GWBEvaluator(BaseEvaluator):
    def __init__(self, network_specs: List[str], frequency_array: np.ndarray, waveform_approximant: str,
                 deriv_parameters: ParaNameList, convert_cos: Optional[ParaNameList] = None, convert_log: Optional[ParaNameList] = None,
                 rotate: bool = False):
        self.network = Network(network_specs)
        self.network.set_wf_vars(
            wf_model_name='lal_bbh',
            wf_other_var_dic=dict(approximant=waveform_approximant))
        self.network.set_net_vars(
            f=frequency_array, deriv_symbs_string=' '.join(deriv_parameters),
            conv_cos=convert_cos, conv_log=convert_log, use_rot=rotate)
        self.network.setup_psds()

        self._fisher_parameters = deriv_parameters.copy()
        for convert_func, convert_para in {'cos':convert_cos, 'log':convert_log}.items():
            if convert_para is not None:
                for para in convert_para:
                    self._fisher_parameters[self._fisher_parameters.index(para)]=f'{convert_func}_{para}'

    def snr(self, parameters: ParameterDict) -> float:
        self.network.set_net_vars(inj_params=parameters)
        self.network.setup_ant_pat_lpf()
        self.network.calc_det_responses()
        self.network.calc_snrs_det_responses(only_net=True)
        return self.network.snr

    def fisher(self, parameters: ParameterDict) -> FisherMatrix:
        self.network.set_net_vars(inj_params=parameters)
        self.network.setup_ant_pat_lpf()
        self.network.calc_det_responses_derivs_num()
        self.network.calc_errors(only_net=True)
        return self.network.fisher
    
    @property
    def fisher_parameters(self):
        return self._fisher_parameters
