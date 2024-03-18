import torch
from torch import Tensor
from abc import ABC
from functools import partial
import time

from .common import tensorsManager, VERBOSE_HOOK, VERBOSE_TENSOR_TRANSFER
from .hook import TensorRefBase
from .TensorRefsTracker import TensorRefsTracker, SetTensorRefType

tensorRefsTracker = TensorRefsTracker()

def retrieveTensorRef(tensor, tensorsManager=None):
    idTensor = id(tensor)
    if idTensor in tensorRefsTracker.refByTensor:
        return tensorRefsTracker.refByTensor[idTensor]

    if tensorsManager is not None:
        return TensorRef(tensor, tensorsManager)

    return None

class ProxyInfo:
    def __init__(self):
        self.device = "cpu"
        self.usageNs = 0
        self.locked = False
        self.stacks = 0

def levelArg(arg, ref):
    if isinstance(arg, Tensor):
        arg = retrieveTensorRef(arg, ref['tensorsManager'])
        arg.proxyInfo.locked = True
    if isinstance(arg, TensorRef):
        ref['proxies'].append(arg)
        if not ref['onCPU']:
            arg = arg.toGPU()
        else:
            arg = arg.toCPU()

    if isinstance(arg, tuple):
        arg = list(arg)
        for a in range(0, len(arg)):
            arg[a] = levelArg(arg[a], ref)
        arg = tuple(arg)
    if isinstance(arg, list):
        for a in range(0, len(arg)):
            arg[a] = levelArg(arg[a], ref)
    return arg

def levelTensorsArgs(args, kwargs, opts={}):
    if len(args) == 0:
        return None, args, kwargs

    self = args[0]
    manager = tensorsManager
    if isinstance(self, TensorRef):
        manager = self.proxyInfo.tensorsManager

    ref = { 'proxies': [], 'tensorsManager': manager, 'onCPU': False }
    if 'onCPU' in opts:
        ref['onCPU'] = opts['onCPU']

    args = list(args)

    # Lock immediately TensorRefs
    for arg in args:
        if isinstance(arg, TensorRef):
            arg.proxyInfo.locked = True

    for a in range(0, len(args)):
        args[a] = levelArg(args[a], ref)
    args = tuple(args)

    for key, value in kwargs.items():
        kwargs[key] = levelArg(value, ref)

    return ref, args, kwargs

class TensorRef(ABC, TensorRefBase):

    def __init__(self, target, tensorsManager):
        setattr(self, "target", target)
        setattr(self, "proxyInfo", ProxyInfo())
        self.proxyInfo.tensorsManager = tensorsManager

        tensorRefsTracker.countTensor(self)        
        self.onUsage()

    '''
    def __del__(self):
        tensorRefsTracker.uncountTensor(self, False)
        tensorRefsTracker.remTensorRef(self)
    '''
    
    def __setattr__(self, key, value):
        if key == "proxyInfo" or key == "target":
            super().__setattr__(key, value)
        else:
            setattr(self.target, key, value)

    def __getattr__(self, name):
        if name == "target" or name == "proxyInfo":
            try:
                return super().__getattribute__(name)
            except:
                if name == "target":
                    return self

        if VERBOSE_HOOK:
            print('TRef Attr: ', name)

        # Delegate attribute access to the target object
        attr = getattr(self.target, name)

        if name == "to":

            def ignore(*args, **kwargs):
                dev = None

                if len(args) > 0:
                    dev = args[0]

                if isinstance(dev, str):
                    self.proxyInfo.device = dev
                elif dev is not None or kwargs:
                    self.target = attr(*args, **kwargs)
                return self

            return ignore

        # Hole-fillers
        if attr is None:
            if name == 'detach':
                def stillMe():
                    print("deatch none")
                    return self
                return stillMe

        if callable(attr):
            if name in ['shape']:
                return attr

            # If the original attribute is callable, we return a new wrapper function
            def wrapper(*args, **kwargs):
                # Here you can analyze the arguments before calling the original function
                if VERBOSE_HOOK:            
                    print(f"Calling {name}")

                if name not in ['set_', 'numpy', 'detach']:
                    self.toGPU()
                    self.proxyInfo.locked = True

                attr = getattr(self.target, name) # you have to retrieve it again

                # look for tensors on CPU
                proxies = []
                args = list(args)
                for a in range(0, len(args)):
                    value = args[a]
                    if isinstance(value, Tensor):
                        value = retrieveTensorRef(value, self.proxyInfo.tensorsManager)
                    if isinstance(value, TensorRef):
                        proxies.append(value)
                        args[a] = value.toGPU()
                        value.proxyInfo.locked = True

                    if name == '__torch_function__':
                        if args[a] is tuple:
                            types = list(args[a])
                            for t in range(0, len(types)):
                                if types[t] is TensorRef:
                                    types[t] = Tensor
                            args[a] = tuple(types)
                args = tuple(args)

                for key, value in kwargs.items():
                    #print(f"{key}: {value}")
                    if isinstance(value, Tensor):
                        value = retrieveTensorRef(value, self.proxyInfo.tensorsManager)
                    if isinstance(value, TensorRef):
                        proxies.append(value)
                        kwargs[key] = value.toGPU()

                '''
                isCpu = self.target.is_cpu
                for arg in args:
                    if isinstance(arg, (Tensor, TensorRef)):
                        tensor = arg
                        if isinstance(arg, TensorRef):
                            tensor = tensor.target
                        if tensor.is_cpu != isCpu:
                            print("debug")
                '''

                # Perform the call to the original function
                result = attr(*args, **kwargs)
                # Optionally, process the result before returning

                # back to CPU
                for value in proxies:
                    value.proxyInfo.locked = False
                    value.toCPU()

                self.proxyInfo.locked = False
                self.toCPU()

                def toRef(result):
                    result = retrieveTensorRef(result, self.proxyInfo.tensorsManager)
                    result.toCPU()
                    return result

                if isinstance(result, Tensor):
                    if name == "cpu":
                        self.target = result
                    else:
                        result = toRef(result)
                elif isinstance(result, (list, tuple)): #todo: do the same in fun hook
                    l = list(result)
                    for i in range(0, len(l)):
                        ll = l[i]
                        if isinstance(ll, Tensor):
                            l[i] = toRef(ll)
                    if isinstance(result, tuple):
                        result = tuple(l)
                    else:
                        result = l

                if VERBOSE_HOOK:            
                    print(f"Returning {name}")

                tensorRefsTracker.printStatus()
                tensorRefsTracker.checkTensors()

                return result

            return wrapper
        else:
            return attr
        
    def onUsage(self):
        self.proxyInfo.usageNs = (time.time_ns() + self.proxyInfo.usageNs)/2

    def toGPU(self):
        if isinstance(self.target, Tensor):
            if self.target.is_cpu:
                if VERBOSE_TENSOR_TRANSFER:
                    print('TensorRef.toGPU')
                dev = self.proxyInfo.tensorsManager.device
                if dev is not None and dev != "cpu":
                    dt = self.target.dtype
                    if dt is torch.float64:
                        dt = torch.float32

                    tensorRefsTracker.uncountTensor(self)
                    res = self.target.to(device=dev, dtype=dt)                    

                    if isinstance(self.target, torch.nn.Parameter) and not isinstance(res, torch.nn.Parameter):
                        res = torch.nn.Parameter(res)
                    else:
                        #res = res.to(torch.get_default_dtype())
                        pass

                    self.target = res

                    tensorRefsTracker.countTensor(self)
                    return res
            else:
                pass

        return self.target

    def toCPU(self):
        if isinstance(self.target, Tensor):
            if not self.target.is_cpu:
                if VERBOSE_TENSOR_TRANSFER:
                    print('TensorRef.toCPU')
                tensorRefsTracker.uncountTensor(self)
                res = self.target.to(device="cpu")
                #res = res.to(torch.get_default_dtype()) # ensure Tensor default type

                if isinstance(self.target, torch.nn.Parameter) and not isinstance(res, torch.nn.Parameter):
                    res = torch.nn.Parameter(res)

                self.target = res
                tensorRefsTracker.countTensor(self)
            else:
                pass

        return self.target

    def uncount(self):
        tensorRefsTracker.uncountTensor(self)
        tensorRefsTracker.remTensorRef(self)

    def stackEnter(self):
        self.proxyInfo.stacks += 1

    def stackExit(self):
        self.proxyInfo.stacks -= 1

    def countReferences(self):
        return tensorRefsTracker.countTensorRefReferences(self)

    ### Iterate

    # The __iter__ method returns the iterator object itself
    def __iter__(self):
        return self.target.__iter__()

    def __repr__(self, *args, **kwargs):
        return self.target.__repr__(*args, **kwargs)

    def __str__(self, *args, **kwargs):
        return self.target.__str__(*args, **kwargs)
    
    def __getitem__(self, key):
        res = self.target.__getitem__(key)

        if isinstance(key, (slice, tuple, int)) or key is None:
            return TensorRef(res, self.proxyInfo.tensorsManager)

        return res

SetTensorRefType(TensorRef)

#TensorRef.register(Tensor)
#TensorRef.register(torch.nn.Parameter)

Tensor.__bases__ = (TensorRefBase,) + Tensor.__bases__
#TensorRefBase.__bases__ = (Tensor,) + TensorRefBase.__bases__

# Create math operation magic functions
ops = [
        "add", "sub", "truediv", "floordiv", "mul", "mod", "divmod", "pow", "and", "or", "lshift", "rshift", "xor", "matmul",
        "cmp", "eq", "nq", "ne", "lt", "gt", "le", "ge"
    ]

def applyMagicMethod_math(op, dev=''):
    op = "__" + op + "__"

    try:
        method = getattr(Tensor, op)
        if method is not None:

            def mathWrapper(self, other):
                if VERBOSE_HOOK:
                    print('TRef Math: ', m)

                self.proxyInfo.locked = True
                self.toGPU()
                self.stackEnter()
                self.onUsage()

                res = None
                withBaseTensor = False
                if isinstance(other, Tensor):
                    #res = method(self.target, other)
                    other = retrieveTensorRef(other, self.proxyInfo.tensorsManager)
                    withBaseTensor = True

                otherTarget = other
                if isinstance(other, TensorRef):
                    other.proxyInfo.locked = True
                    other.stackEnter()
                    other.onUsage()
                    otherTarget = other.toGPU()
                else:
                    print("Debug: this shouldn't happen 0x45647538")

                res = method(self.target, otherTarget)

                self.proxyInfo.locked = False
                self.stackExit()

                if res is NotImplemented:
                    if op == '__pow__':
                        res = torch.pow(self.target, otherTarget)
                        return res
                    else:
                        raise Exception("Not implemented")

                if isinstance(other, TensorRef):
                    other.proxyInfo.locked = False
                    other.toCPU()
                    other.stackExit()

                if isinstance(res, Tensor):
                    res = retrieveTensorRef(res, self.proxyInfo.tensorsManager)
                    res.toCPU()

                self.toCPU()

                #if withBaseTensor:
                #    return self.target

                return res

            setattr(TensorRef, dev+op, mathWrapper)
    except:
        ignore = True

for op in ops:
    applyMagicMethod_math(op)
    applyMagicMethod_math(op, 'r')
    applyMagicMethod_math(op, 'i')

# Generic magic proxy functions
magics = dir(Tensor)
magicsAttr = {}

def createMagicWrapper(m):
    magic = getattr(Tensor, m)
    magicsAttr[m] = magic

    isTorchFun = m == '__torch_function__'

    magicRef = None
    try:
        magicRef = getattr(TensorRef, m)
    except:
        pass

    if magicRef is None:
        def makeWrapper(m, magic):
            def magicWrapper(*args, **kwargs):

                if VERBOSE_HOOK:
                    print('TRef Magic: ', m)

                '''
                refs = []
                args = list(args)
                for a in range(0, len(args)):
                    if isinstance(args[a], TensorRef):
                        refs.append(args[a])
                        args[a] = args[a].toGPU()
                '''

                opts = {}

                # todo: __array__ issue fixed at the root (of calling numpy). Try to remove this and the next
                if m == '__array__':  # conversion to numpy must be done on CPU
                    opts['onCPU'] = True

                # What an ugly thing...
                if isTorchFun:
                    args = list(args)

                    fun = args[1]

                    if fun.__name__.endswith('__array__'): # sub with ==
                        opts['onCPU'] = True

                    types = args[2]
                    tup = args[3]
                    # tensor = args[0]

                    types = list(types)
                    for t in range(0, len(types)):
                        if types[t] == TensorRef:
                            types[t] = Tensor
                    types = tuple(types)

                    tref, tup, _ = levelTensorsArgs(tup, {}, opts) #todo: handle proxies(?)
                    #for p in tref['proxies']:
                    #    ref['proxies'].append(p)

                    args[0] = fun
                    args[1] = types
                    args[2] = tup
                    del args[3]

                    args = tuple(args)

                self = args[0]
                ref, args, kwargs = levelTensorsArgs(args, kwargs, opts)

                try:
                    for arg in args:
                        if isinstance(arg, TensorRef):
                            arg.stackEnter()
                            arg.onUsage()

                    res = magic(*args, **kwargs)

                    for arg in args:
                        if isinstance(arg, TensorRef):
                            arg.stackExit()

                    if res is NotImplemented:
                        return getattr(TensorRef, '__'+m[3:])(ref, *args, **kwargs)

                    for proxy in ref['proxies']:
                        proxy.proxyInfo.locked = False
                        proxy.toCPU()

                    if isinstance(res, Tensor):
                        manager = tensorsManager
                        if isinstance(self, TensorRef):
                            manager = self.proxyInfo.tensorsManager

                        res = retrieveTensorRef(res, manager)
                        res.toCPU()

                    return res
                except Exception as err:
                    raise err
            return magicWrapper

        setattr(TensorRef, m, makeWrapper(m, magic))

for m in magics:
    if m.startswith('__'):
        try:
            createMagicWrapper(m)
        except Exception as err:
            pass