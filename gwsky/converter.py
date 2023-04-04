from abc import ABCMeta, abstractmethod

import numpy as np
import healpy as hp

from typing import Callable
from .typing import ParameterDict, ParaNameList


class ParameterConverter(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, parameters: ParameterDict) -> ParameterDict:
        pass

    @abstractmethod
    def reverse_convert(self, converted: ParameterDict) -> ParameterDict:
        pass

    @abstractmethod
    def name_convert(self, name_list: ParaNameList) -> ParaNameList:
        pass

    @abstractmethod
    def reverse_name_convert(self, converted_name_list: ParaNameList) -> ParaNameList:
        pass

    def __mul__(self, other: 'ParameterConverter')->'ParameterConverter':
        class TmpConverter(ParameterConverter):
            def __call__(self_, parameters: ParameterDict) -> ParameterDict:
                return other(self(parameters))

            def reverse_convert(self_, converted: ParameterDict) -> ParameterDict:
                return self.reverse_convert(other.reverse_convert(converted))

            def name_convert(self_, name_list: ParaNameList) -> ParaNameList:
                return other.name_convert(self.name_convert(name_list))

            def reverse_name_convert(self_, converted_name_list: ParaNameList) -> ParaNameList:
                return self.reverse_name_convert(other.reverse_name_convert(converted_name_list))

        return TmpConverter()


class NoConvert(ParameterConverter):
    def __call__(self, parameters: ParameterDict) -> ParameterDict:
        return parameters.copy()
    
    def reverse_convert(self, converted: ParameterDict) -> ParameterDict:
        return converted.copy()
    
    def name_convert(self, name_list: ParaNameList) -> ParaNameList:
        return name_list.copy()
    
    def reverse_name_convert(self, converted_name_list: ParaNameList) -> ParaNameList:
        return converted_name_list.copy()


class FuncConverter(ParameterConverter):
    def __init__(self, convert_para: ParaNameList, convert_name: str,
                 convert_func: Callable[[float], float],
                 revert_convert_func: Callable[[float], float]) -> None:
        super().__init__()
        self.convert_para = convert_para
        self.convert_name = convert_name
        self.convert_func = convert_func
        self.revert_convert_func = revert_convert_func

    def _para_name_convert(self, para):
        return f'{self.convert_name}_{para}'

    def __call__(self, parameters: ParameterDict) -> ParameterDict:
        converted = parameters.copy()
        for para in self.convert_para:
            converted[self._para_name_convert(
                para)] = self.convert_func(converted.pop(para))
        return converted

    def reverse_convert(self, converted: ParameterDict) -> ParameterDict:
        parameters = converted.copy()
        for para in self.convert_para:
            parameters[para] = self.revert_convert_func(
                parameters.pop(self._para_name_convert(para)))
        return parameters

    def name_convert(self, name_list: ParaNameList) -> ParaNameList:
        converted_name_list = name_list.copy()
        for para in self.convert_para:
            converted_name_list.remove(para)
            converted_name_list.append(self._para_name_convert(para))
        return converted_name_list

    def reverse_name_convert(self, converted_name_list: ParaNameList) -> ParaNameList:
        name_list = converted_name_list.copy()
        for para in self.convert_para:
            name_list.remove(self._para_name_convert(para))
            name_list.append(para)
        return name_list


class RotationConverter(ParameterConverter):
    def __init__(self, rotation_matrix: np.ndarray, ra_dec: bool = True) -> None:
        self.rot_mat = rotation_matrix
        self.ra_dec = ra_dec

        self._inv_rot_mat = None

    @property
    def inv_rot_mat(self):
        if self._inv_rot_mat is None:
            self._inv_rot_mat = np.linalg.inv(self.rot_mat)
        return self._inv_rot_mat

    def rotate_from_parameters(self, rot_mat: np.ndarray, parameters: ParameterDict) -> ParameterDict:
        parameters = parameters.copy()
        if self.ra_dec:
            theta, phi = np.pi/2-parameters['dec'], parameters['ra']
        else:
            theta, phi = parameters['theta'], parameters['phi']

        rot_theta, rot_phi = hp.rotator.rotateDirection(rot_mat, theta, phi)

        if self.ra_dec:
            parameters['ra'], parameters['dec'] = rot_phi, np.pi/2-rot_theta
        else:
            parameters['theta'], parameters['phi'] = rot_theta, rot_phi
        return parameters

    def __call__(self, parameters: ParameterDict) -> ParameterDict:
        return self.rotate_from_parameters(self.rot_mat, parameters)

    def name_convert(self, name_list: ParaNameList) -> ParaNameList:
        return name_list.copy()

    def reverse_convert(self, converted: ParameterDict) -> ParameterDict:
        return self.rotate_from_parameters(self.inv_rot_mat, converted)

    def reverse_name_convert(self, converted_name_list: ParaNameList) -> ParaNameList:
        return converted_name_list.copy()
