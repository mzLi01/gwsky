import numpy as np
from numdifftools import Gradient

from typing import Callable, Tuple, Union
from ..typing import ParaNameList, ParameterDict, ParameterVector, Value, DerivateDict


def derivate_central(func, parameters: ParameterDict, deriv_para: ParaNameList, step: float = 1e-7) -> DerivateDict:
    deriv = {}
    for para in deriv_para:
        forward_parameters = parameters.copy()
        backward_parameters = parameters.copy()
        forward_parameters[para] += step
        backward_parameters[para] -= step
        deriv[para] = (func(forward_parameters) - func(backward_parameters)) / (2*step)
    return deriv



def complex_to_amplitude_phase(z: Value) -> Tuple[Value, Value]:
    return np.abs(z), np.unwrap(np.angle(np.atleast_1d(z)))


def amplitude_phase_to_complex(amplitude: Value, phase: Value) -> Value:
    return amplitude*np.exp(1j*phase)


def derivative_from_amplitude_and_phase(amplitude: Value, phase: Value, d_amplitude: Value, d_phase: Value) -> Value:
    return d_amplitude * np.exp(1j*phase) + amplitude * np.exp(1j*phase) * 1j * d_phase


def complex_gradient(func: Callable, parameter_vector: ParameterVector, return_value: bool,
                     step: float, **gradient_kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Value]]:
    """
    Calculate gradient for complex function by calculating derivative for amplitude and phase seperately.
    Using numerical derivative package numdifftools.

    Args:
        func: function for gradient calculation
              Return value of func should be a 1d or 2d array.
        parameter_vector (ParameterVector): values of parameters(real)
        return_value (bool): whether to return value of func at parameter_vector
            is func is expensive to calculate, this parameter should be set to True,
            since func(parameter_vector) is calculated once in this function,
            it will be returned to the caller who may need to use this value.
        step (float) : step of numerical derivative

    Returns:
        array, shape: len(parameter_vector) x shape of return value of func
            gradient of func
        value: value of func(parameter_vector). returns if `return_value` is True
    """
    value = func(parameter_vector)
    amplitude, phase = complex_to_amplitude_phase(value)

    def amp_pha_func(parameter_vector):
        # always returns a 2d array, no matter return value of func is 1d or 2d array
        return np.vstack(complex_to_amplitude_phase(func(parameter_vector)))
    d_amp_pha = Gradient(amp_pha_func, step=step, **gradient_kwargs)(parameter_vector)
    # 3d array, first index matches parameter_vector
    d_amp_pha = np.transpose(d_amp_pha, axes=(1, 0, 2))
    d_amplitude = d_amp_pha[:, :d_amp_pha.shape[1]//2, :]
    d_phase = d_amp_pha[:, d_amp_pha.shape[1]//2:, :]

    gradient_shape = [len(parameter_vector)] + list(amplitude.shape)
    gradient = derivative_from_amplitude_and_phase(
        amplitude, phase, d_amplitude, d_phase).reshape(gradient_shape)

    if return_value:
        return gradient, value
    else:
        return gradient


def combine_product_derivate(deriv_parameters: ParaNameList, deriv_dict1: DerivateDict, deriv_dict2: DerivateDict,
                             value1: Value, value2: Value) -> DerivateDict:
    product_deriv = {}
    for para in deriv_parameters:
        product_deriv[para] = deriv_dict1.get(para, 0) * value2 \
                            + deriv_dict2.get(para, 0) * value1
    return product_deriv


def convert_derivate_cos(value: Value, derivate: Value):
    return -1/np.sin(value) * derivate


def convert_derivate_log(value: Value, derivate: Value):
    return value * derivate
