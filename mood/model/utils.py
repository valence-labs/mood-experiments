from typing import Callable
import torch


def linear_interpolation(step, duration, max_value, start: int = 0, min_value: float = 0):
    if max_value < min_value:
        raise ValueError("max_value cannot be smaller than min value")
    if step < start:
        return min_value
    if step >= start + duration:
        return max_value
    step_size = (max_value - min_value) / duration
    step = step - start
    return step_size * step


def get_activation(activation_spec):

    if isinstance(activation_spec, Callable):
        return activation_spec
    if activation_spec is None:
        return None

    activation_fs = vars(torch.nn.modules.activation)
    for activation in activation_fs:
        if activation.lower() == activation_spec.lower():
            return activation_fs[activation]()

    raise ValueError(f"{activation_spec} is not a valid activation function")
