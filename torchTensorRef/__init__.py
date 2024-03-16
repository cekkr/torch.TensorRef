###
###
###
import inspect
# import inspect
# import copy
import types
import typing

from .hook import Hooks
from .common import VERBOSE_HOOK
from .basic import Stack

torch = None
TensorRef = None
tensorRefsTracker = None

def is_builtin_type(obj):
    builtin_types = (int, float, str, list, dict, tuple, set, bool, bytes)
    return isinstance(obj, builtin_types) or type(obj) in vars(types).values()

def tryHook(self, name, attr, hook):
    if attr.__name__ != hook.__name__:
        setattr(self, name, hook)

def checkSelf(self):
    if torch is not None:
        t = type(self)
        if issubclass(t, torch.nn.Module):
            if '__wrapped_nn_module' not in self.__dict__:
                methods = ['register_parameter', 'register_buffer']
                for m in methods:
                    try:
                        attr = getattr(self, m)      
                        if m == 'register_parameter':
                            tryHook(self, m, attr, Hooks.module_register_parameter)
                        if m == 'register_buffer':
                            tryHook(self, m, attr, Hooks.module_register_buffer)
                    except Exception as err:
                        pass
                self.__dict__['__wrapped_nn_module'] = True

###
### Import hook (ugly solutions for lazy people)
###

injectTo = ['torch']
exclude = [
            'torch.fx', 'torch.jit', 'torch.autograd', 'torchgen', 'torch.storage', 'functools', 'torch.utils', 'torch.library',
            'torchTensorRef',
            'torch._tensor', 'torch._C'
            #'torch._'
           ]

def startsWith(str, arr):
    for a in arr:
        if str.startswith(a): # or str.endswith(a):
            return True 
    return False


methodStack = Stack()

itsMe = []
def method_wrapper(func):
    name = func.__module__+'.'+func.__name__
    if func in itsMe or startsWith(name, exclude) or not startsWith(name, injectTo):
        return func

    passAsRef = False
    if name.startswith('torch.nn.modules'):
        passAsRef = True

    passTensorRefs = False
    '''
    passTensorRefs = passTensorRefs or name.startswith('torch._refs')
    passTensorRefs = passTensorRefs or name.startswith('torch._prims')
    '''
    returnNormalTensor = name.endswith('_maybe_convert_to_dtype')

    #print(name)

    #func_signature = inspect.signature(func)

    class classWrapper:
        def funWrapper(*args, **kwargs):

            global methodStack

            if VERBOSE_HOOK:
                print('Fun Hook: ', name)

            methodStack = methodStack.enter()
            methodStack.name = name

            _returnNormalTensor = returnNormalTensor

            if methodStack.get('inOp') is True:
                _returnNormalTensor = True

            if name.startswith('torch._refs') or name == 'torch.group_norm':
                methodStack.set('inOp', True)

            args = list(args)

            refs = []
            for a in range(0, len(args)):
                arg = args[a]
                checkSelf(arg)
                if TensorRef is not None:
                    if isinstance(arg, TorchTensor):
                        args[a] = TensorRef(arg, tensorsManager)
                    if isinstance(args[a], TensorRef):
                        refs.append(args[a])
                        res = args[a].toGPU()
                        if not passTensorRefs:
                            args[a] = res
                    '''
                    else:
                        # https://github.com/huggingface/accelerate/blob/main/src/accelerate/big_modeling.py
                        if isinstance(args[a], TensorRef):
                            args[a] = None
                    '''
            args = tuple(args)

            #if name == 'torch.group_norm':
            #    print("debug")

            # print(f"Before calling {func.__name__}")
            result = func(*args, **kwargs)
            # print(f"After calling {func.__name__}")

            for r in refs:
                r.toCPU()
                r.uncount()

            methodStack = methodStack.exit()

            if not _returnNormalTensor and TorchTensor is not None:
                if isinstance(result, TorchTensor):
                    ref = TensorRef(result, tensorsManager)
                    ref.toCPU()
                    return ref

            tensorRefsTracker.checkTensors()

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
            attrName = name + '.' + v

            if attrName.startswith('torch.Tensor'):
                continue

            if name.startswith('torch.nn.modules'):
                if v == 'register_parameter':
                    tryHook(v, attr, Hooks.module_register_parameter)
                    attr = Hooks.module_register_parameter
                if v == 'register_buffer':
                    tryHook(v, attr, Hooks.module_register_buffer)
                    attr = Hooks.module_register_buffer


            isNnModule = attrName == 'torch.nn.modules.container.Module'
            if isNnModule:
                defaultSetAttr = getattr(attr, '__setattr__')
                def setAttr(self, name, val):
                    if isinstance(val, TensorRef) and not isinstance(val, torch.Tensor):
                        val = val.target
                    defaultSetAttr(self, name, val)
                setattr(attr, '__setattr__', setAttr)


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
importToWrap = []
firstWrapping = True

setTensorLikeTo = None

def flushWrap():
    global importToWrap
    global firstWrapping
    for mod in importToWrap:
        wrapModule(mod)
    importToWrap = []
    firstWrapping = False

origTensorLike = None

def noisy_importer(
    name,
    locals={},
    globals={},
    fromlist=[],
    level=0,
    forceLoad=False,
    defaultImport=None,
):
    global setTensorLikeTo
    global origTensorLike

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
    '''
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
    '''

    #if name.startswith('torch.nn'):
    #    print("check")

    if (startsWith(name, injectTo) or (name.startswith('.') and inside != None and startsWith(inside, injectTo))) and not startsWith(name, exclude):
        res = defaultImport(name, locals, globals, fromlist, level)
        if firstWrapping:
            importToWrap.append(res)
        else:
            res = wrapModule(res)

        #if '__alreadyOnWrap' not in res.__dict__:
        #    res.__dict__['__alreadyOnWrap'] = True
        #    res = wrapModule(res)
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

    if name == 'torch':
        torch = res

    if name == 'torch._tensor':
        hook.props['tensor'] = res.Tensor

    if name.endswith('_custom_op.impl'):
        res.SUPPORTED_RETURN_TYPES[hook.TensorRefBase] = 'TensorRefBase'

    if name == 'torch._C':
        def funIgnore(fun, *args, **kwargs):
            return fun
        res.__dict__['_add_docstr'] = funIgnore

    if name == 'torch.overrides':
        # temp solution
        def funAlwaysTrue(*args, **kwargs):
            return True
        res.__dict__['is_tensor_like'] = funAlwaysTrue

    if name.endswith('inspect'):
        if '/torch/' in locals['__file__']:
            res = hook.Hooks.inspect(res)

    if name.endswith('_prims_common'):
        #setTensorLikeTo = res
        try:
            setattr(res, 'TensorLike', hook.TensorRefBase)

            if origTensorLike is None:
                origTensorLike = res.TensorLikeType

            if locals['__file__'].endswith('_refs/__init__.py'): # or locals['__file__'].endswith('_prims_common/wrappers.py'):
                setattr(res, 'TensorLikeType', hook.TensorRefBase)
            else:
                setattr(res, 'TensorLikeType', origTensorLike)

            '''
            print(globals['__file__'])
            if globals['__file__'].endswith('_prims_common/wrappers.py'):
                setattr(res, 'TensorLikeType', (hook.TensorRefBase, origTensorLike))
            else:
                setattr(res, 'TensorLikeType', origTensorLike)
            '''

            '''
            if globals['__file__'].endswith('_refs/__init__.py')\
                    or globals['__file__'].endswith('_prims_common/wrappers.py'):
                setattr(res, 'TensorLikeType', (hook.TensorRefBase, importCache['torch'].Tensor))
            else:
                setattr(res, 'TensorLikeType', hook.TensorRefBase)
            '''
            #setattr(res, 'TensorLike', (res.TensorLike, hook.TensorRefBase))
        except Exception as err:
            pass

    return res


import builtins
from .common import tensorsManager

def init():
    global TensorRef
    global TorchTensor
    global torch
    global tensorRefsTracker

    builtins.__import__ = noisy_importer

    ###
    ### torch
    ###

    import torch
    from torch import Tensor as TorchTensor

    flushWrap()

    from .TensorRef import TensorRef, tensorRefsTracker

    hook.props = { 'tensor': torch.Tensor, 'tensorRef': TensorRef }

#if setTensorLikeTo is not None:
#    setattr(setTensorLikeTo, 'TensorLike', (setTensorLikeTo.TensorLike, hook.TensorRefBase))

###
###
###

# Save the original isinstance function
original_isinstance = isinstance

# Define a new function that extends isinstance
def custom_isinstance(obj, classinfo):
    # Example: Let's say you want all instances to be considered a member of a special class, SpecialClass
    if classinfo is hook.TensorRefBase:
        return original_isinstance(obj, TensorRef) or original_isinstance(obj, torch.Tensor)
    # For all other cases, defer to the original isinstance
    return original_isinstance(obj, classinfo)

# Replace the built-in isinstance with the custom one
isinstance = custom_isinstance

###
###
###

print("torch.TensorRef injected.")
