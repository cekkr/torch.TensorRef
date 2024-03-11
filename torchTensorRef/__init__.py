#import inspect
import types
import copy

import torch
from torch import Tensor as TorchTensor

from .TensorRef import TensorRef
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

"""
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
"""

###
### Class
###


def method_wrapper(func):
    def wrapper(*args, **kwargs):
        # print(f"Before calling {func.__name__}")
        result = func(*args, **kwargs)
        # print(f"After calling {func.__name__}")

        if isinstance(result, TorchTensor):
            return TensorRef(result, tensorsManager)

        return result

    return wrapper


def class_wrapper(cls):
    for name, method in cls.__dict__.items():
        if callable(method):
            setattr(cls, name, method_wrapper(method))
    return cls

def classBoggart_creator(parent_class):
    try:
        # Define a new subclass with customized behavior or properties if needed
        class ClassBoggart(parent_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                args = list(args)
                for a in range(0, len(args)):
                    arg = args[a]
                    if isinstance(arg, TensorRef):
                        args[a] = arg.target
                args = tuple(args)

                res = super().__call__(*args, **kwargs)
                return res

        # Return the dynamically created subclass
        return ClassBoggart

    except:
        return parent_class

def analyzeClass(var):
    return classBoggart_creator(var)

class TorchLazyWrapper:
    def __init__(self, target):
        setattr(self, "__target", target)

    def __getattr__(self, name):
        if name == "__target" or name == "_TorchLazyWrapper__target":
            return super().__getattribute__("__target")

        attr = getattr(self.__target, name)

        if isinstance(attr, (int, float, str)):
            return attr
        elif isinstance(attr, type):  # Is class
            return analyzeClass(attr)
        elif isinstance(
            attr,
            (
                types.FunctionType,
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
                types.MethodType,
            ),
        ) or callable(attr):
            return method_wrapper(attr)
        elif isinstance(attr, object):
            return TorchLazyWrapper(attr)


torch = TorchLazyWrapper(torch)
torch.__target.Tensor = TensorRef
