import inspect
import types

import torch
from TensorProxy import TensorProxy


class TorchWrapper:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        attr = getattr(self.target, name)
        return attr

    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)


def analyzeVar(var):
    if isinstance(var, types.FunctionType):
        signature = inspect.signature(var)
        parameters = signature.parameters
        return_type = signature.return_annotation
        return var

    if callable(var):
        return var

    return var


def cycleObj(obj):
    vars = list(obj.keys())

    for var in vars:
        val = torch.get(var)
        if isinstance(val, object):
            torch[var] = TorchWrapper(val)


cycleObj(torch)
