###
###
###

# import inspect
# import copy
import types

###
### Import hook (ugly solutions for lazy people)
###

TensorRef = None
TorchTensor = None

def wrapModule(mod):
    vars = dir(mod)
    for v in vars:
        try:
            attr = mod.__dict__[v]
            if isinstance(
                attr,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    types.BuiltinMethodType,
                    types.MethodType,
                ),
            ) and callable(attr):
                mod.__dict__[v] = method_wrapper(attr)
        except:
            ignore = True


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

    # print(f'name: {name!r}')

    if forceLoad:
        del sys.modules[name]

    res = None

    try:
        res = importCache[name]
        if res is not None and not forceLoad:
            res.__check()
            return res
    except:
        ignore = True

    if name.startswith("torch") and not forceLoad:

        bi = None
        try:
            bi = locals["builtins"]
        except Exception as err:
            ignore = True

        if bi is None:
            bi = __import__("builtins", locals, globals, fromlist, 0)
            locals["builtins"] = bi

        origImport = bi.__import__

        if isinstance(origImport, types.BuiltinFunctionType):

            def wrapImport(
                name, locals={}, globals={}, fromlist=[], level=0, forceLoad=False
            ):
                return noisy_importer(
                    name, locals, globals, fromlist, level, forceLoad, origImport
                )

            bi.__import__ = wrapImport

        res = defaultImport(name, locals, globals, fromlist, level)
        wrapModule(res)
    else:
        res = defaultImport(name, locals, globals, fromlist, level)

    if res is None:
        print("damn, import is none")
    else:
        importCache[name] = res

    return res


import builtins

builtins.__import__ = noisy_importer

###
### torch wrappers
###


def method_wrapper(func):
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

    return wrapper


###
### torch
###

from torch import Tensor as TorchTensor

from .TensorRef import TensorRef
from .TensorsManager import TensorsManager

tensorsManager = TensorsManager()

print("torch.TensorRef injected.")