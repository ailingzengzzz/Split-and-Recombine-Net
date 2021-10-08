import torch
import numpy as np


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result