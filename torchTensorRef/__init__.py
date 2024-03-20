###
###
###
import inspect
# import inspect
# import copy
import types
import typing
import time
import copy
import sys

from .hook import Hooks
from .common import VERBOSE_HOOK, properties
from .basic import Stack
from .inspection import GetNumArgs

torch = None
TensorRef = None
tensorRefsTracker = None
retrieveTensorRef = None

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
excludeFromInjection = ['timm']
exclude = [
            'torch.fx', 'torch.jit', 'torch.autograd', 'torchgen', 'torch.storage', 'functools', 'torch.utils', 'torch.library', 'torch.cuda',
            #"torch.collections", "torch.tensor",
            'torchTensorRef',
            #'torch._tensor', 'torch._C', 'torch._utils'
            'torch._',
            'torch.is_grad_enabled', 'torch.get_default_dtype', 'torch.no_grad',
            #'torch.load', 'torch.serialization',
]

functionsAsIs = [
    'torch.is_grad_enabled', 'torch.get_default_dtype', # 'torch.cat', 'torch.stack', 'torch.isfinite', 'torch.isnan', 'torch.ceil'
]

def startsWith(str, arr):
    for a in arr:
        if str.startswith(a):
            return True 
    return False


def endsWith(str, arr):
    for a in arr:
        if str.endswith(a):
            return True
    return False

methodStack = Stack()

itsMe = []
origFunctions = {}

def method_wrapper(func):
    name = func.__module__+'.'+func.__name__
    if func in itsMe or startsWith(name, exclude) or not startsWith(name, injectTo):
        return func

    if VERBOSE_HOOK:
        print("hooking function " + name)

    origFunctions[name] = func

    passAsRef = name not in ['torch.nn.modules.module._load_from_state_dict'] and not startsWith(name, ['torch.serialization'])
    ignoreNaNChecker = name in ['torch.nn.modules.module.load_state_dict', 'torch.tensor']
    #if name.startswith('torch.nn.modules'):
    #    passAsRef = True

    passTensorRefs = False
    '''
    passTensorRefs = passTensorRefs or name.startswith('torch._refs')
    passTensorRefs = passTensorRefs or name.startswith('torch._prims')
    '''
    returnNormalTensor = name.endswith('_maybe_convert_to_dtype')

    #print(name)
    #func_signature = inspect.signature(func)
    wrapArguments = False

    class classWrapper:
        def funWrapper(*args, **kwargs):

            global methodStack

            if methodStack.get('asYouAre') is True:
                return func(*args, **kwargs)

            if VERBOSE_HOOK:
                print('Fun Hook: ', name + ' \t', methodStack.level)

            maxStackLevel = 5
            tensorsBackToCPU = True # methodStack.level <= maxStackLevel

            methodStack = methodStack.enter(name)
            #stackFullName = methodStack.getFullName() #TODO: methodStack.getFullName() creates an infinity loop, check it

            inMaxLevel = False # methodStack.level > maxStackLevel

            if ((len(name.split('.'))) == 2 or startsWith(name, ['torch.nn.functional.'])): # basic function
                tensorsBackToCPU = True
                inMaxLevel = True
            else:
                pass # for debugging

            refAsGPU = True  # set it as false make the algorithm stop working

            argsAsRef = classWrapper.argsAsRef
            moveToAccelerator = True #TODO: change of logic: create the opportunity to eveluate the best device at the moment
            simpleFunction = False
            if (name in functionsAsIs
                    or startsWith(name, ['torch.nn.parameter.', 'torch._refs.']) or endsWith(name, ['load_from_state_dict', 'load_state_dict'])):
                argsAsRef = False
                #moveToAccelerator = False
                refAsGPU = False
                simpleFunction = True

            if name in ['torch.load'] and False:
                tensorsBackToCPU = False
                inMaxLevel = True

            if VERBOSE_HOOK:
                #print("Stack: " + stackFullName)
                pass

            _returnNormalTensor = returnNormalTensor

            if methodStack.get('inOp') is True:
                _returnNormalTensor = True

            if name.startswith('torch._refs') or name == 'torch.group_norm':
                methodStack.set('inOp', True)

            # If at lower level, force passing as Tensor
            if inMaxLevel:
                argsAsRef = False
                _returnNormalTensor = _returnNormalTensor or not tensorsBackToCPU

            isModelLoading = (name.startswith("torch.nn.modules")
                              and ('state' in name or 'parameter' in name or 'dict' in name or 'named' in name))
            if methodStack.avgPreparationTime > methodStack.avgExecTime or isModelLoading:
                moveToAccelerator = False

            def runFunc(*args, **kwargs):
                if classWrapper.isStaticFunction:
                    self, *args = args
                    return func(*args, *kwargs)
                else:
                    try:
                        func(*args, *kwargs)
                    except Exception as err:
                        se = str(err)
                        if 'positional argument' in se and 'but' in se and 'given' in se:
                            classWrapper.isStaticFunction = True 
                            return runFunc(*args, **kwargs)
                        else:
                            raise err

            refs = []
            newRefs = []
            embeddings = []
            def argToRef(arg):
                if TensorRef is not None:
                    if isinstance(arg, TorchTensor): #not tensorsBackToCPU and
                        arg = retrieveTensorRef(arg, tensorsManager)
                        newRefs.append(arg)
                    if isinstance(arg, TensorRef):
                        refs.append(arg)
                        if refAsGPU:
                            if moveToAccelerator:
                                arg.toGPU()
                            else:
                                arg.toCPU()
                    if isinstance(arg, torch.nn.Module):
                        props = dir(arg)
                        embeddings.append(arg)
                        methodStack.set('asYouAre', True)
                        for p in props:
                            try:
                                tensor = getattr(arg, p)
                                if isinstance(tensor, TorchTensor):
                                    ref = retrieveTensorRef(tensor, tensorsManager)
                                    if refAsGPU:
                                        if moveToAccelerator:
                                            ref.toGPU()
                                        else:
                                            ref.toCPU()
                                    setattr(arg, p, ref.target)
                            except:
                                pass
                        methodStack.set('asYouAre', False)
                if isinstance(arg, list) and False: # this cause an error during the loading of the checkpoints... (size mismatch)
                    for a in range(0, len(arg)):
                        arg[a] = argToRef(arg[0])
                if isinstance(arg, dict):
                    for k,v in arg.items():
                        arg[k] = argToRef(v)
                return arg

            beginStart = time.time_ns()

            args = list(args)
            for a in range(0, len(args)):
                arg = args[a]
                checkSelf(arg)
                args[a] = argToRef(args[a])
            args = tuple(args)

            for key, value in kwargs.items():
                kwargs[key] = argToRef(value)

            beginEnd = time.time_ns()
            beginDiff = beginEnd - beginStart

            execStart = time.time_ns()

            for ref in refs:
                ref.onUsage()
                ref.stackEnter()

            ###
            ###
            ###
            result = None
            if argsAsRef:
                try:
                    result = runFunc(*args, **kwargs)

                    if not classWrapper.nanChecked:
                        try:
                            if origFunctions['torch.isnan'](result).any():
                                argsAsRef = False # classWrapper.argsAsRef =
                        except:
                            pass
                        classWrapper.nanChecked = True

                except Exception as err:
                    argsAsRef = False # classWrapper.argsAsRef =

            if not argsAsRef:
                def argToTensor(arg):
                    if isinstance(arg, TensorRef):
                        if refAsGPU:
                            arg = arg.target
                        else:
                            if moveToAccelerator:
                                arg = arg.toGPU()
                            else:
                                arg = arg.toCPU()
                    if isinstance(arg, list):
                        for a in range(0, len(arg)):
                            arg[a] = argToTensor(arg[a])
                    if isinstance(arg, dict):
                        for k, v in arg.items():
                            arg[k] = argToTensor(v)
                    return arg

                args = list(args)
                for a in range(0, len(args)):
                    args[a] = argToTensor(args[a])
                args = tuple(args)

                for key, value in kwargs.items():
                    kwargs[key] = argToTensor(value)

                for emb in embeddings:
                    props = dir(emb)
                    methodStack.set('asYouAre', True)
                    for p in props:
                        tensor = getattr(emb, p)
                        if isinstance(tensor, TensorRef):
                            tens = None
                            if refAsGPU:
                                tens = tensor.target
                            else:
                                if moveToAccelerator:
                                    tens = tensor.toGPU()
                                else:
                                    tens = tensor.toCPU()
                            setattr(emb, p, tens)
                    methodStack.set('asYouAre', False)

                result = runFunc(*args, **kwargs)

            ###
            ### Result
            ###

            execEnd = time.time_ns()
            execDiff = execEnd - execStart

            resultStart = time.time_ns()
            for r in refs:
                if tensorsBackToCPU:
                    r.toCPU()
                # r.uncount()
                r.stackExit()

            if not tensorsBackToCPU:
                for r in newRefs:
                    r.release()

            def checkReturnArg(arg):
                if isinstance(arg, dict):
                    for k,v in arg.items():
                        arg[k] = checkReturnArg(v)

                isBaseTensor = False
                if isinstance(arg, TorchTensor):
                    isBaseTensor = True
                    if not _returnNormalTensor:
                        arg = retrieveTensorRef(arg, tensorsManager)
                if isinstance(arg, TensorRef):
                    tens = arg.toCPU()
                    if isBaseTensor:
                        arg = tens
                return arg

            args = list(args)
            for a in range(0, len(args)):
                args[a] = checkReturnArg(args[a])

            for k,v in kwargs.items():
                kwargs[k] = checkReturnArg(v)

            for e in embeddings:
                props = dir(e)
                methodStack.set('asYouAre', True)
                for p in props:
                    try:
                        tensor = getattr(e, p)
                        if isinstance(tensor, TorchTensor):
                            ref = retrieveTensorRef(tensor, tensorsManager)
                            tens = ref.target
                            if tensorsBackToCPU:
                                tens = ref.toCPU()
                            if _returnNormalTensor:
                                ref = tens
                            setattr(e, p, ref)
                    except:
                        pass
                methodStack.set('asYouAre', False)
            methodStack = methodStack.exit()

            resultEnd = time.time_ns()

            resultDiff = resultEnd - resultStart
            preparationTime = resultDiff + beginDiff

            methodStack.avgPreparationTime = (preparationTime+methodStack.avgPreparationTime)/2
            methodStack.avgExecTime = (execDiff+methodStack.avgExecTime)/2

            if moveToAccelerator and tensorsBackToCPU:
                tensorRefsTracker.checkTensors()

            if isinstance(result, TorchTensor):
                ref = retrieveTensorRef(result, tensorsManager)
                if tensorsBackToCPU:
                    ref.toCPU()
                if _returnNormalTensor:
                    return ref.target
                return ref

            return result

    classWrapper.argsAsRef = passAsRef
    classWrapper.nanChecked = name in ['torch.isnan'] or ignoreNaNChecker
    classWrapper.isStaticFunction = False 

    wrapper = classWrapper.funWrapper

    numArgs = -1 # GetNumArgs(func) # function disabled
    if numArgs >= 0: 
        # scala reale
        if numArgs == 0:
            def compiledWrapper():
                wrapper()
            return compiledWrapper
        elif numArgs == 1:
            def compiledWrapper(a0):
                wrapper(a0)
            return compiledWrapper   
        elif numArgs == 2:
            def compiledWrapper(a0, a1):
                wrapper(a0, a1)
            return compiledWrapper   
        elif numArgs == 3:
            def compiledWrapper(a0, a1, a2):
                wrapper(a0, a1, a2)
            return compiledWrapper   
        elif numArgs == 4:
            def compiledWrapper(a0, a1, a2, a3):
                wrapper(a0, a1, a2, a3)
            return compiledWrapper   
        elif numArgs == 5:
            def compiledWrapper(a0, a1, a2, a3, a4):
                wrapper(a0, a1, a2, a3, a4)
            return compiledWrapper   
        elif numArgs == 6:
            def compiledWrapper(a0, a1, a2, a3, a4, a5):
                wrapper(a0, a1, a2, a3, a4, a5)
            return compiledWrapper   
        elif numArgs == 7:
            def compiledWrapper(a0, a1, a2, a3, a4, a5, a6):
                wrapper(a0, a1, a2, a3, a4, a5, a6)
            return compiledWrapper   
        elif numArgs == 8:
            def compiledWrapper(a0, a1, a2, a3, a4, a5, a6, a7):
                wrapper(a0, a1, a2, a3, a4, a5, a6, a7,)
            return compiledWrapper   
        elif numArgs == 9:
            def compiledWrapper(a0, a1, a2, a3, a4, a5, a6, a7, a8):
                wrapper(a0, a1, a2, a3, a4, a5, a6, a7, a8)
            return compiledWrapper   
        elif numArgs == 10:
            def compiledWrapper(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
                wrapper(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9)
            return compiledWrapper   

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

def shallow_copy_module(original_module):
    if isinstance(original_module, (types.FunctionType, type)):
        return copy.copy(original_module)

    # Create a new module object
    new_module = types.ModuleType(original_module.__name__)
    
    # Copy attributes from the original module to the new module
    new_module.__dict__.update(original_module.__dict__)
    
    return new_module

cachedModules = {}
def wrapModule(mod):
    if startsWith(mod.__name__, exclude):
        return mod

    orig = mod
    vars = dir(mod)

    wrappedVars = 0
    idMod = id(mod)
    if idMod in cachedModules:
        mod = cachedModules[idMod]
        try:
            wrappedVars = mod.__dict__['__wrapped']
        except:
            pass

        if wrappedVars == len(vars):
            return mod
    else:
        mod = cachedModules[idMod] = shallow_copy_module(mod)

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
    #####

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

    return mod


old_import = __import__

importCache = {}
importToWrap = {}
modulesToWrap = []
firstWrapping = True

setTensorLikeTo = None

# it doesn't work anymore
def flushWrap():
    global importToWrap
    global modulesToWrap
    global firstWrapping

    module2Wrap = {}
    for mod in modulesToWrap:
        module2Wrap[id(mod)] = wrapModule(mod)

    for idLocals, local in importToWrap.items():
        for mid, mod in module2Wrap.items():
            for name, val in local.items():
                if(id(val) == mid):
                    local[name] = mod

    importToWrap = {}
    firstWrapping = False

origTensorLike = None
#origModules = {}
moduleExcludeStack = 0

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
    #global origModules
    global moduleExcludeStack
    global excludeFromInjection
    global modulesToWrap
    global importToWrap

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

    #todo: clone the hooked module to provide original one when needed
    if startsWith(name, excludeFromInjection):
            moduleExcludeStack += 1

    if moduleExcludeStack <= 0:
        if (startsWith(name, injectTo) or (name.startswith('.') and inside != None and startsWith(inside, injectTo))) and not startsWith(name, exclude):
            orig = res = defaultImport(name, locals, globals, fromlist, level)

            if firstWrapping:
                idLocals = id(locals)
                if idLocals not in importToWrap:
                    importToWrap[idLocals] = locals
                if res not in modulesToWrap:
                    modulesToWrap.append(res)
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
    else:
        res = defaultImport(name, locals, globals, fromlist, level)

    if name in excludeFromInjection:
        moduleExcludeStack -= 1


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
    global retrieveTensorRef

    builtins.__import__ = noisy_importer

    ###
    ### torch
    ###

    import torch
    from torch import Tensor as TorchTensor

    flushWrap()

    from .TensorRef import TensorRef, tensorRefsTracker, retrieveTensorRef
    tensorsManager.refsTracker = tensorRefsTracker
    tensorRefsTracker.manager = tensorsManager

    hook.props = { 'tensor': torch.Tensor, 'tensorRef': TensorRef }

    ### Calculate default parameters
    tref = torch.rand(1, 1)
    defRefs = tref.countReferences()
    print("Tensor default references: ", defRefs)
    properties['minRefsTensorRef'] = defRefs[0]
    properties['minRefsTensor'] = defRefs[1]

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
