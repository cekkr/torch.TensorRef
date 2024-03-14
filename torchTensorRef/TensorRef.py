import torch
from torch import Tensor
from abc import ABC
from functools import partial

from .hook import Hooks

class ProxyInfo:
    def __init__(self):
        self.device = "cpu"

def tryHook(self, name, attr, hook):
    if attr.__name__ != hook.__name__:
        setattr(self, name, attr)

def checkSelf(self):
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
                except:
                    pass
            self.__dict__['__wrapped_nn_module'] = True

class TensorRef(ABC):

    def __init__(self, target, tensorsManager):
        setattr(self, "target", target)
        setattr(self, "proxyInfo", ProxyInfo())
        self.proxyInfo.tensorsManager = tensorsManager

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

        checkSelf(self)

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
                    return self
                return stillMe

        if callable(attr):
            # If the original attribute is callable, we return a new wrapper function
            def wrapper(*args, **kwargs):
                # Here you can analyze the arguments before calling the original function
                #print(f"Calling {name}")

                # look for tensors on CPU
                proxies = []
                for a in range(0, len(args)):
                    value = args[a]
                    if isinstance(value, Tensor):
                        value = TensorRef(value, self.proxyManager.tensorsManager)
                    if isinstance(value, TensorRef):
                        proxies.append(value)
                        args[a] = value.toGPU()

                for key, value in kwargs.items():
                    #print(f"{key}: {value}")
                    if isinstance(value, Tensor):
                        value = TensorRef(value, self.proxyManager.tensorsManager)
                    if isinstance(value, TensorRef):
                        proxies.append(value)
                        kwargs[key] = value.toGPU()

                # Perform the call to the original function
                result = attr(*args, **kwargs)
                # Optionally, process the result before returning

                # back to CPU
                for value in proxies:
                    value.toCPU()

                if isinstance(result, Tensor):
                    if name == "cpu":
                        self.target = result
                    else:
                        result = TensorRef(result, self.proxyInfo.tensorsManager)
                        result.toCPU()

                return result

            return wrapper
        else:
            return attr

    def toGPU(self):
        if isinstance(self.target, Tensor):
            if self.target.is_cpu:
                dev = self.proxyInfo.tensorsManager.device
                if dev is not None and dev != "cpu":
                    res = self.target.to(dev)
                    if isinstance(self.target, torch.nn.Parameter) and not isinstance(res, torch.nn.Parameter):
                        res = torch.nn.Parameter(res)
                    self.target = res

        return self.target

    def toCPU(self):
        if isinstance(self.target, Tensor):
            if not self.target.is_cpu:
                self.target = self.target.to("cpu")

        return self.target

    ### Iterate

    # The __iter__ method returns the iterator object itself
    def __iter__(self):
        return self.target.__iter__()

    def __repr__(self, *args, **kwargs):
        return self.target.__repr__(*args, **kwargs)

    def __str__(self, *args, **kwargs):
        return self.target.__str__(*args, **kwargs)
    
    def __getitem__(self, key):
        return self.target.__getitem__(key)

TensorRef.register(Tensor)
TensorRef.register(torch.nn.Parameter)

# Create math operation magic functions
ops = ["add", "sub", "truediv", "floordiv", "mul", "mod", "divmod", "pow", "and", "or", "lshift", "rshift", "xor"]

def applyMagicMethod_math(op, dev=''):
    op = "__" + op + "__"

    try:
        method = getattr(Tensor, op)
        if method is not None:

            def mathWrapper(self, other):
                checkSelf(self)

                self.toGPU()
                res = None
                withBaseTensor = False
                if isinstance(other, Tensor):
                    #res = method(self.target, other)
                    other = TensorRef(other, self.proxyInfo.tensorsManager)
                    withBaseTensor = True

                otherTarget = other
                if isinstance(other, TensorRef):
                    otherTarget = other.toGPU()

                res = method(self.target, otherTarget)

                if res is NotImplemented:
                    if op == '__pow__':
                        res = torch.pow(self.target, otherTarget)
                        return res
                    else:
                        raise Exception("Not implemented")

                if isinstance(other, TensorRef):
                    other.toCPU()

                res = TensorRef(
                    res, self.proxyInfo.tensorsManager
                )

                res.toCPU()

                cpu = self.toCPU()

                if withBaseTensor:
                    return self.target

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

    magicRef = None
    try:
        magicRef = getattr(TensorRef, m)
    except:
        pass

    if magicRef is None:
        def makeWrapper(m, magic):
            def magicWrapper(*args, **kwargs):
                self, *args = args
                checkSelf(self)

                ref = None
                if isinstance(self, TensorRef):
                    ref = self
                    self = ref.toGPU()

                try:
                    res = magic(self, *args, **kwargs)

                    if res is NotImplemented:
                        return getattr(TensorRef, '__'+m[3:])(ref, *args, **kwargs)

                    if ref is not None:
                        ref.toCPU()

                    return TensorRef(res, ref.proxyInfo.tensorsManager)
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
