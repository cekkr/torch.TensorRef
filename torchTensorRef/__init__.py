###
###
###
import inspect
# import inspect
# import copy
import types

from .hook import Hooks

def is_builtin_type(obj):
    builtin_types = (int, float, str, list, dict, tuple, set, bool, bytes)
    return isinstance(obj, builtin_types) or type(obj) in vars(types).values()


###
### Import hook (ugly solutions for lazy people)
###

injectTo = ['torch']
exclude = ['torch.fx', 'torch.jit', 'torch.autograd', 'torchgen', 'torchTensorRef', 'torch.storage', 'functools', 'torch.utils', 'torch.library']

def startsWith(str, arr):
    for a in arr:
        if str.startswith(a): # or str.endswith(a):
            return True 
    return False
    
itsMe = []
def method_wrapper(func):
    name = func.__module__+'.'+func.__name__
    if func in itsMe or startsWith(name, exclude) or not startsWith(name, injectTo):
        return func

    passAsRef = False
    if name.startswith('torch.nn.modules'):
        passAsRef = True

    #print(name)

    #func_signature = inspect.signature(func)

    ignore = False
    if name == 'torch._C._add_docstr':
        ignore = True

    class classWrapper:
        def funWrapper(*args, **kwargs):
            
            if ignore:
                return None

            args = list(args)

            refs = []
            for a in range(0, len(args)):
                arg = args[a]
                if TensorRef is not None:
                    if isinstance(arg, TorchTensor):
                        args[a] = TensorRef(arg, tensorsManager)
                    if not passAsRef:
                        if isinstance(args[a], TensorRef):
                            refs.append(args[a])
                            args[a] = args[a].toGPU()
                    '''
                    else:
                        # https://github.com/huggingface/accelerate/blob/main/src/accelerate/big_modeling.py
                        if isinstance(args[a], TensorRef):
                            args[a] = None
                    '''
            args = tuple(args)

            # print(f"Before calling {func.__name__}")
            result = func(*args, **kwargs)
            # print(f"After calling {func.__name__}")

            for r in refs:
                r.toCPU()

            if TorchTensor is not None:
                if isinstance(result, TorchTensor):
                    ref = TensorRef(result, tensorsManager)
                    ref.toCPU()
                    return ref

            return result

    wrapper = classWrapper.funWrapper

    try:
        wrapModule(func)
    except Exception as err:
        pass


    vars = dir(func)
    for v in vars:
        if v != '__new__':
            try:
                attr = getattr(func, v)
                setattr(wrapper, v, attr)
            except:
                pass

    itsMe.append(wrapper)
    return wrapper

TensorRef = None
TorchTensor = None

cachedModules = {}
def wrapModule(mod):
    if startsWith(mod.__name__, exclude):
        return mod

    wrappedVars = 0
    try:
        wrappedVars = mod.__dict__['__wrapped']
    except:
        pass

    vars = dir(mod)

    try:
        mod.__dict__['__wrapped'] = len(vars)
    except:
        pass

    name = ''
    try:
        name = mod.__module__ + '.'
    except:
        pass

    name += mod.__name__

    if wrappedVars == len(vars):
        try:
            return cachedModules[name]
        except:
            pass

    def trySet(name, attr):
        try:
            mod.__dict__[v] = attr
        except:
            setattr(mod, name, attr)

    def tryHook(name, attr, hook):
        if attr.__name__ != hook.__name__:
            trySet(name, hook)

    for v in vars:
        if v.startswith('__'):
            continue        

        try:
            attr = getattr(mod, v)

            if name.startswith('torch.nn.modules'):
                if v == 'register_parameter':
                    tryHook(v, attr, Hooks.module_register_parameter)
                if v == 'register_buffer':
                    tryHook(v, attr, Hooks.module_register_buffer)

            if inspect.isclass(attr) or inspect.ismodule(attr):
                if startsWith(attr.__module__+'.'+attr.__name__, injectTo) and not startsWith(attr.__module__+'.'+attr.__name__, exclude):
                    wr = wrapModule(attr)
                    trySet(v, wr)
            elif isinstance(
                attr,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    types.BuiltinMethodType,
                    types.MethodType
                ),
            ) and callable(attr):
                wr = method_wrapper(attr)
                trySet(v, wr)

        except Exception as err:
            ignore = True

    cachedModules[name] = mod
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

    if (startsWith(name, injectTo) or (name.startswith('.') and inside != None and startsWith(inside, injectTo))) and not startsWith(name, exclude):
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
