import inspect
import types

import torch
from torch import Tensor as TorchTensor

from .TensorProxy import TensorProxy
from .TensorsManager import TensorsManager

tensorsManager = TensorsManager()


class TorchWrapper:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        attr = getattr(self.target, name)
        return attr

    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)


###
### Analyze
###


def analyzeVar(var, name):
    if isinstance(var, (int, float, str)):
        return var
    if isinstance(
        var, (types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType)
    ):
        return method_wrapper(var)
    elif isinstance(var, type):  # Is class
        return analyzeClass(var, name)  # TorchWrapper(var)
    elif isinstance(var, object):
        return cycleObj(var, name)

    # if callable(var):
    #    return var

    return var


###
### Class
###


def method_wrapper(func):
    def wrapper(*args, **kwargs):
        # print(f"Before calling {func.__name__}")
        result = func(*args, **kwargs)
        # print(f"After calling {func.__name__}")

        if isinstance(result, TorchTensor):
            return TensorProxy(result, tensorsManager)

        return result

    return wrapper


def class_wrapper(cls):
    for name, method in cls.__dict__.items():
        if callable(method):
            setattr(cls, name, method_wrapper(method))
    return cls


def analyzeClass(var, name):
    methods = [
        member
        for member in dir(var)
        if callable(getattr(var, member)) and not member.startswith("__")
    ]

    print("Methods of MyClass:")
    for method in methods:
        print(method)

    # If you want to include methods from base classes as well, use inspect.getmembers() with a filter
    all_methods = inspect.getmembers(var, predicate=inspect.isfunction)

    print("\nAll methods of MyClass (including base classes):")
    for method_name, method in all_methods:
        print(method_name)

    return var


cycledObjs = []


def cycleObj(obj, name):
    vars = []  # dir(obj)

    if name == "torch":
        vars = ["rand"]

    for var in vars:
        if not var.startswith("_"):
            val = getattr(obj, var)
            if True or val not in cycledObjs:
                cycledObjs.append(val)
                val = analyzeVar(val, name + "." + var)
                setattr(obj, var, val)

    return obj


torch = cycleObj(torch, "torch")
torch.Tensor = TensorProxy
