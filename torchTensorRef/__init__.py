import inspect
import types

import torch

from .TensorProxy import TensorProxy


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


def analyzeVar(var, depth=0):
    if isinstance(var, (int, float, str)):
        return var
    if isinstance(var, types.FunctionType):
        signature = inspect.signature(var)
        parameters = signature.parameters
        return_type = signature.return_annotation
        return var
    elif isinstance(var, type):  # Is class
        return analyzeClass(var)  # TorchWrapper(var)
    elif isinstance(var, object):
        if depth < 3:
            return cycleObj(var, depth + 1)

    # if callable(var):
    #    return var

    return var


###
### Class
###


def method_wrapper(func):
    def wrapper(*args, **kwargs):
        print(f"Before calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"After calling {func.__name__}")
        return result

    return wrapper


def class_wrapper(cls):
    for name, method in cls.__dict__.items():
        if callable(method):
            setattr(cls, name, method_wrapper(method))
    return cls


def analyzeClass(var):
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


def cycleObj(obj, depth=0):
    vars = dir(obj)

    for var in vars:
        try:
            if not var.startswith("_"):
                val = getattr(obj, var)
                if val not in cycledObjs:
                    cycledObjs.append(val)
                    val = analyzeVar(val, depth)
                    setattr(obj, var, val)
        except:
            null = None

    return obj


torch = cycleObj(torch)
