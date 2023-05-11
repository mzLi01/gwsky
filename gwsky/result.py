import json
import numpy as np
import pandas as pd

from .catalog import GWCatalog
from .converter import ParameterConverter, NoConvert

from typing import Optional, Union, Callable, List, Dict
from .typing import ParameterDict, ParaNameList, FisherMatrix, ParameterVector, SampleResult


ConverterFunc = Callable[[ParameterDict], ParameterConverter]
ConverterGen = Union[ParameterConverter, ConverterFunc]


class Result:
    def __init__(self, keys: ParaNameList, fisher_parameters: ParaNameList):
        self.keys = keys
        self.fisher_parameters = fisher_parameters

        self.last_save_path = ''
        self.samples = None

    @property
    def fisher_parameters(self):
        return self._fisher_parameters

    @fisher_parameters.setter
    def fisher_parameters(self, fisher_parameters: ParaNameList):
        self._fisher_parameters = fisher_parameters
        self._fisher_tril_indices = np.tril_indices(len(fisher_parameters))
        self.fisher_columns = [f'fisher_{i}{j}' for i, j in zip(*self._fisher_tril_indices)]

    def set_samples(self, samples: pd.DataFrame, fisher_parameters: Optional[ParaNameList] = None):
        self.samples = samples
        self.keys = list(samples.columns[:samples.columns.get_loc('snr')])
        if fisher_parameters is not None:
            self.fisher_parameters = fisher_parameters

    def _subpara_fisher_columns(self, sub_parameters: ParaNameList) -> List:
        para_to_index = {para: i for i, para in enumerate(self.fisher_parameters)}
        def sub_element_to_main(sub_i, sub_j):
            main_i = para_to_index[sub_parameters[sub_i]]
            main_j = para_to_index[sub_parameters[sub_j]]
            if main_i >= main_j:
                return main_i, main_j
            else:
                return main_j, main_i

        columns = []
        for i, j in zip(*np.tril_indices(len(sub_parameters))):
            main_i, main_j = sub_element_to_main(i, j)
            columns.append(f'fisher_{main_i}{main_j}')
        return columns

    def _fisher_matrix_to_columns(self, fisher: FisherMatrix) -> List:
        return list(fisher[self._fisher_tril_indices])

    def _fisher_columns_to_matrix(self, columns: List, ndim: Optional[int] = None) -> FisherMatrix:
        if ndim is None:
            ndim = len(self.fisher_parameters)
        fisher = np.zeros((ndim, ndim))
        fisher[np.tril_indices(ndim)] = columns
        return fisher + np.triu(fisher.T, k=1)  # symmetrize

    def _sample_result_to_df_item(self, sample_points: ParameterVector, snr: float, fisher: FisherMatrix) -> List:
        return sample_points+[snr]+self._fisher_matrix_to_columns(fisher)

    def append_samples(self, sample_results: List[SampleResult]):
        df_items = [self._sample_result_to_df_item(
            *result) for result in sample_results]
        new_samples = pd.DataFrame(
            df_items, columns=self.keys+['snr']+self.fisher_columns)
        if self.samples is None:
            self.samples = new_samples
        else:
            self.samples = pd.concat([self.samples, new_samples], ignore_index=True)

    def get_sample_points(self, indexes: Optional[pd.Index] = None, keys: Optional[ParaNameList] = None) -> np.ndarray:
        if indexes is None:
            indexes = self.samples.index
        if keys is None:
            keys = self.keys
        return self.samples.loc[indexes, keys].to_numpy()

    def get_sample_parameter_dict(self, indexes: Optional[pd.Index] = None, keys: Optional[ParaNameList] = None) -> List[Dict]:
        if indexes is None:
            indexes = self.samples.index
        if keys is None:
            keys = self.keys
        return [{key: self.samples.loc[i, key] for key in keys}
                for i in indexes]

    def get_snr(self, indexes: Optional[pd.Index] = None) -> np.ndarray:
        if indexes is None:
            indexes = self.samples.index
        return np.array(self.samples.loc[indexes, 'snr'])

    def get_fisher_matrix(self, indexes: Optional[pd.Index] = None,
                          fisher_parameters: Optional[ParaNameList] = None) -> np.ndarray:
        if indexes is None:
            indexes = self.samples.index
        if fisher_parameters is None:
            fisher_columns = self.fisher_columns
            ndim = len(self.fisher_parameters)
        else:
            fisher_columns = self._subpara_fisher_columns(fisher_parameters)
            ndim = len(fisher_parameters)

        fisher_value = np.array(self.samples.reindex(index=indexes, columns=fisher_columns))
        return np.array([self._fisher_columns_to_matrix(f, ndim=ndim) for f in fisher_value])

    def clear(self):
        self.samples.drop(index=self.samples.index, inplace=True)

    @property
    def metadata(self):
        return {'keys': self.keys, 'fisher_parameters': self.fisher_parameters}

    def save_csv(self, path: str, append_mode: Optional[bool] = None):
        if append_mode is None:
            append_mode = (path == self.last_save_path)

        if append_mode:
            self.samples.to_csv(path, index=False, mode='a', header=False)
        else:
            self.samples.to_csv(path, index=False)
        self.last_save_path = path

        metadata_path = path + '.metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    @classmethod
    def load_csv(cls, path: str):
        metadata_path = path + '.metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        keys = metadata['keys']
        fisher_parameters = metadata['fisher_parameters']

        result = cls(keys=keys, fisher_parameters=fisher_parameters)
        samples = pd.read_csv(path, index_col=False)
        result.samples = samples
        return result

    def regenerate_event(self, real_parameter: ParameterDict, fisher: FisherMatrix,
                         fisher_parameter: ParaNameList, num: Optional[int] = None) -> Union[ParameterDict, List[ParameterDict]]:
        calc_num = 1 if num is None else num

        cov_para_names = list(filter(lambda p: p in fisher_parameter, real_parameter.keys()))
        cov_para_list = [real_parameter[para] for para in cov_para_names]
        no_cov_para = {para: value for para, value in real_parameter.items() if para not in fisher_parameter}

        cov_full = np.linalg.inv(fisher)
        fisher_index_map = {para: i for i, para in enumerate(fisher_parameter)}
        catalog_index = [fisher_index_map[para] for para in cov_para_names]
        cov = cov_full[catalog_index, :][:, catalog_index]

        detected_para_lists = np.random.multivariate_normal(mean=cov_para_list, cov=cov, size=calc_num)
        detected_params = []
        for params in detected_para_lists:
            detected = {para: value for para, value in zip(cov_para_names, params)}
            detected.update(no_cov_para)
            detected_params.append(detected)
        if num is None:
            return detected_params[0]
        else:
            return detected_params

    def regenerate_converted_event(self, parameters: ParameterDict, fisher: FisherMatrix, snr: float,
                                   converter_func: ConverterFunc, paras: ParaNameList,
                                   num: Optional[int] = None) -> Union[ParameterVector, List[ParameterVector]]:
        calc_num = 1 if num is None else num

        converter = converter_func(parameters)
        converted = converter(parameters)
        try:
            event_para_converted = self.regenerate_event(
                real_parameter=converted, fisher=fisher,
                fisher_parameter=self.fisher_parameters, num=calc_num)
        except np.linalg.LinAlgError:
            event_para_converted = [parameters] * calc_num
            snr = -1

        event_params = [converter.reverse_convert(converted) for converted in event_para_converted]
        catalog_vectors = [[params[para] for para in paras]+[snr] for params in event_params]
        if num is None:
            return catalog_vectors[0]
        else:
            return catalog_vectors

    def _get_converter_func(self, converter_gen: Optional[ConverterGen] = None) -> ConverterFunc:
        if converter_gen is None:
            converter_gen = NoConvert()
        if isinstance(converter_gen, ParameterConverter):
            def converter_func(p: ParameterDict):
                return converter_gen
        else:
            converter_func = converter_gen
        return converter_func

    def generate_catalog(self, converter_from_parameter: Optional[ConverterGen] = None,
                         indexes: Optional[pd.Index] = None, paras: Optional[ParaNameList] = None) -> GWCatalog:
        """
        generate observed catalog using parameters and fisher matrix value

        Args:
            converter (Union[ParameterConverter, Callable[[ParameterDict], ParameterConverter]]): 
                converter that convert stored parameters to parameters that corresponding to fisher matrix
                or a function that generate converter from parameters (can vary for different parameters)
            indexes (Iterable[pandas.Index]):
                choose indexes in result samples to generate catalog. Default to choose all samples

        Returns:
            GWCatalog: GW catalog, containing observed parameter value and SNR
        """
        if paras is None:
            paras = self.keys
        converter_func = self._get_converter_func(converter_from_parameter)

        fishers = self.get_fisher_matrix(indexes=indexes)
        parameters = self.get_sample_parameter_dict(indexes=indexes, keys=paras)
        snrs = self.get_snr(indexes=indexes)

        catalog_list = [self.regenerate_converted_event(
            parameter_i, fisher_i, snr_i, converter_func=converter_func, paras=paras)
            for parameter_i, fisher_i, snr_i in zip(parameters, fishers, snrs)]
        catalog_df = pd.DataFrame(catalog_list, columns=paras+['snr'])
        catalog = GWCatalog(events=catalog_df)
        return catalog

    def bootstrap_catalog(self, n_resample: int, converter_from_parameter: Optional[ConverterGen] = None,
                          indexes: Optional[pd.Index] = None, paras: Optional[ParaNameList] = None,
                          memory_batch: Optional[int] = None, save_path: Optional[str] = None) -> GWCatalog:
        if indexes is None:
            indexes = self.samples.index
        if paras is None:
            paras = self.keys
        if memory_batch is None:
            memory_batch = n_resample + 1
        else:
            assert save_path is not None
        converter_func = self._get_converter_func(converter_from_parameter)

        resample_j = np.random.randint(0, len(indexes), size=n_resample)  # j is the index of `indexes` parameter
        j_value, j_count = np.unique(resample_j, return_counts=True)

        fishers = self.get_fisher_matrix(indexes=indexes)
        parameters = self.get_sample_parameter_dict(indexes=indexes, keys=paras)
        snrs = self.get_snr(indexes=indexes)

        catalog = GWCatalog(keys=paras+['snr'])
        events = []
        save_append = False
        for j, n in zip(j_value, j_count):
            events += self.regenerate_converted_event(
                parameters[j], fishers[j], snrs[j],
                converter_func=converter_func, paras=paras, num=n)
            if len(events) >= memory_batch:
                catalog.append_events(events)
                events = []
                catalog.save_csv(save_path, append=save_append)
                catalog.clear()
                save_append = True
        catalog.append_events(events)
        if save_path is not None:
            catalog.save_csv(save_path, append=save_append)
        return catalog
