###
###
###

#import inspect
import types
import copy

###
###
###

import numpy

def is_builtin_type(obj):
    builtin_types = (int, float, str, list, dict, tuple, set, bool, bytes)
    return isinstance(obj, builtin_types) or type(obj) in vars(types).values()

_dtype = str
class TorchLazyWrapper:
    def __init__(self, target, name=None, locals=None, globals=None, fromlist=None, level=None):
        super().__setattr__('__target', target)

        super().__setattr__('__name', name)
        super().__setattr__('__locals', locals)
        super().__setattr__('__globals', globals)
        super().__setattr__('__fromlist', fromlist)
        super().__setattr__('__level', level)

    def __check(self):
        if self.__target == False:
            target = noisy_importer(
                super().__getattribute__('__name'),
                super().__getattribute__('__locals'),
                super().__getattribute__('__globals'),
                super().__getattribute__('__fromlist'),
                super().__getattribute__('__level'),
                True
            )
            super().__setattr__('__target', target)

    def __getattr__(self, name):
        if name == "__target" or name == "_TorchLazyWrapper__target":
            return super().__getattribute__("__target")

        if name == '__check':
            return super().__getattribute__('__check')

        self.__check()

        attr = None
        try:
            attr = self.__target.__getattribute__(name)
        except:
            return None

        global _dtype
        if name == 'dtype':
            _dtype = attr

        if isinstance(attr, (int, float, str, TorchLazyWrapper, _dtype)) or name == '_C':
            return attr
        elif isinstance(attr, type):  # Is class
            return analyzeClass(attr)
        elif isinstance(attr, types.ModuleType):
            return TorchLazyWrapper(attr)
        elif isinstance(
            attr,
            (
                types.FunctionType,
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
                types.MethodType,
            ),
        ) and callable(attr):
            return method_wrapper(attr)

        return attr

    def __setattr__(self, key, value):
        self.__target.__setattr__(key, value)

    def __iter__(self, *args, **kwargs):
        return self.__target.__iter__(*args, **kwargs)

    def __next__(self, *args, **kwargs):
        return self.__target.__next__(*args, **kwargs)

    def __getitem__(self, key):
        return self.__target.__getitem__(key)

    def __setitem__(self, key, value):
        self.__target.__setitem__(key, value)

    def __delitem__(self, key):
        self.__target.__delitem__(key)

    def __call__(self, *args, **kwargs):
        try:
            return self.__target(*args, **kwargs)
        except:
            return self.__target.__call__(*args, **kwargs)


###
### Import hook (ugly solutions for lazy people)
###

TensorRef = None
TorchTensor = None

old_import = __import__

def noisy_importer(name, locals={}, globals={}, fromlist=[], level=0, forceLoad=False):
    print(f'name: {name!r}')
    print(f'fromlist: {fromlist}')
    print(f'level: {level}')

    if name == 'torch.Tensor':
        TorchTensor = name

    res = None

    if name.startswith('torch') and not forceLoad:
        bi = None
        try:
            bi = locals['builtins']
        except Exception as err:
            ignore = True

        if bi is None:
            bi = __import__('builtins', locals, globals, fromlist, 0)
            locals['builtins'] = bi

        bi.__import__ = noisy_importer

        if not name.startswith('torch._C'): # and isinstance(res, types.ModuleType)
            res = TorchLazyWrapper(False, name, locals, globals, fromlist, level)
        else:
            def whoCares(*args, **kwargs):
                print("seriously, who cares")
            res = old_import(name, locals, globals, fromlist, level)
            res.__dict__['_add_docstr'] = whoCares
    else:
        res = old_import(name, locals, globals, fromlist, level)

    if res is None:
        print("damn, import is none")

    return res

import builtins
builtins.__import__ = noisy_importer

###
### torch wrappers
###

def method_wrapper(func):
    def wrapper(*args, **kwargs):

        args = list(args)
        for a in range(0, len(args)):
            arg = args[a]
            if TensorRef is not None:
                if isinstance(arg, TensorRef):
                    args[a] = arg.target
        args = tuple(args)

        # print(f"Before calling {func.__name__}")
        result = func(*args, **kwargs)
        # print(f"After calling {func.__name__}")

        if TorchTensor is not None:
            if isinstance(result, TorchTensor):
                return TensorRef(result, tensorsManager)

        return result

    wrapper.__doc__ = "basic"

    return wrapper

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

                vars = dir(self)
                for v in vars:
                    try:
                        attr = getattr(self, v)
                        if callable(attr):
                            setattr(self, v, method_wrapper(attr))
                    except:
                        ignore = True

        # Return the dynamically created subclass
        return ClassBoggart

    except:
        return parent_class

def analyzeClass(var):
    try:
        if len(var.__parameters__) != 0:
            return var
    except:
        ignore = True

    return classBoggart_creator(var)

###
### torch
###

#import sys
#setattr(sys.modules['torch'], 'TensorBase', 'dummy')
#from torch import Tensor as TorchTensor
from .TensorRef import TensorRef
from .TensorsManager import TensorsManager

'''
import importlib
import sys
setattr(sys.modules['torch'], 'TensorBase', 'dummy')
importlib.reload(torch)
'''

tensorsManager = TensorsManager()

#torch = TorchLazyWrapper(torch)
#torch.__target.Tensor = TensorRef
