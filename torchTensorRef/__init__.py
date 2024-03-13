###
###
###
import inspect
# import inspect
# import copy
import types

def is_builtin_type(obj):
    builtin_types = (int, float, str, list, dict, tuple, set, bool, bytes)
    return isinstance(obj, builtin_types) or type(obj) in vars(types).values()


###
### Import hook (ugly solutions for lazy people)
###

def method_wrapper(func):
    if func.__name__ is 'wrapper':
        return func

    class wrapper:
        def __new__(cls, *args, **kwargs):
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

    # wrapper.__doc__ = "basic"

    try:
        wrapModule(func)
    except Exception as err:
        ignore = True

    wrapper.__name__ = func.__name__

    return wrapper

TensorRef = None
TorchTensor = None

def wrapModule(mod):
    if mod.__name__ == 'wrapper':
        return mod

    wrappedVars = 0
    try:
        wrappedVars = mod.__dict__['__wrapped']
    except:
        ignore = True

    vars = dir(mod)
    mod.__dict__['__wrapped'] = len(vars)

    if wrappedVars is len(vars):
        return mod

    for v in vars:
        try:
            attr = mod.__dict__[v]

            #TODO: try to invert the conditions?
            if isinstance(
                attr,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    types.BuiltinMethodType,
                    types.MethodType
                ),
            ) and callable(attr):
                mod.__dict__[v] = method_wrapper(attr)

            elif inspect.isclass(attr) or inspect.ismodule(attr):
                if attr.__module__.startswith('torch'):
                    mod.__dict__[v] = wrapModule(attr)
        except:
            ignore = True

    return mod


old_import = __import__

importCache = {}

def noisy_importer(
    name,
    locals={},
    globals={},
    fromlist=[],
    level=0,
    forceLoad=False,
    defaultImport=None,
):
    if defaultImport is None:
        defaultImport = old_import

    #print(f'name: {name!r}')

    if forceLoad:
        del sys.modules[name]

    res = None

    '''
    try:
        res = importCache[name]
        if res is not None and not forceLoad:
            res.__check()
            return res
    except:
        ignore = True
    '''

    inside = None
    try:
        inside = defaultImport.__dict__['inside']
    except:
        ignore = True

    if inside is None:
        try:
            inside = locals['__name__']
        except Exception as err:
            ignore = True

    ### Obtain builtins import to hook

    '''
    bi = None
    try:
        bi = locals["__builtins__"]
    except Exception as err:
        ignore = True

    if bi is None:
        bi = defaultImport("builtins", locals, globals, fromlist, 0)
        locals["__builtins__"] = bi
    '''

    #TODO: check if originalImport is really useful
    originalImport = None
    try:
        originalImport = globals['__import__']
    except:
        ignore = True

    try:
        originalImport = globals['__builtins__']['__import__']
    except:
        ignore = True

    try:
        originalImport = locals['__builtins__']['__import__']
    except:
        ignore = True

    try:
        originalImport = locals['__import__']
    except:
        ignore = True

    if isinstance(originalImport, types.BuiltinFunctionType):
        def wrapImport(
                name, locals={}, globals={}, fromlist=[], level=0, forceLoad=False
        ):
            return noisy_importer(
                name, locals, globals, fromlist, level, forceLoad, originalImport
            )

        wrapImport.__dict__['inside'] = name
        wrapImport.__dict__['original'] = originalImport
        locals['__import__'] = wrapImport
        defaultImport = originalImport
    else:
        try:
            defaultImport = originalImport.__dict__['original']
        except:
            ignore = True

    #if name.startswith('torch.nn'):
    #    print("check")

    if (name.startswith("torch") or (name.startswith('.') and inside != None and inside.startswith('torch'))) and inside != 'torch._tensor':
        res = defaultImport(name, locals, globals, fromlist, level)
        if name != 'builtins': # still necessary?
            res = wrapModule(res)
    else:
        try:
            res = defaultImport(name, locals, globals, fromlist, level)
        except Exception as err:
            raise err


    '''
    if res is None:
        print("damn, import is none")
    else:
        importCache[name] = res
    '''

    return res


import builtins

builtins.__import__ = noisy_importer


###
### torch
###

import torch
from torch import Tensor as TorchTensor

from .TensorRef import TensorRef
from .TensorsManager import TensorsManager

tensorsManager = TensorsManager()

print("torch.TensorRef injected.")