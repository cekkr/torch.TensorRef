import torch
from torch import Tensor
from abc import ABC
from functools import partial

class ProxyInfo:
    def __init__(self):
        self.device = "cpu"

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
                args = list(args)
                for a in range(0, len(args)):
                    value = args[a]
                    if isinstance(value, Tensor):
                        value = TensorRef(value, self.proxyManager.tensorsManager)
                    if isinstance(value, TensorRef):
                        proxies.append(value)
                        args[a] = value.toGPU()

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
                    res = self.target.to(device=dev,)
                    if isinstance(self.target, torch.nn.Parameter) and not isinstance(res, torch.nn.Parameter):
                        res = torch.nn.Parameter(res)
                    self.target = res

        return self.target

    def toCPU(self):
        if isinstance(self.target, Tensor):
            if not self.target.is_cpu:
                self.target = self.target.to(device="cpu")
                self.target = self.target.to(torch.get_default_dtype()) # ensure Tensor default type

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
        res = self.target.__getitem__(key)
        return res

#TensorRef.register(Tensor)
#TensorRef.register(torch.nn.Parameter)

# Create math operation magic functions
ops = [
        "add", "sub", "truediv", "floordiv", "mul", "mod", "divmod", "pow", "and", "or", "lshift", "rshift", "xor",
        "cmp", "eq", "nq", "ne", "lt", "gt", "le", "ge"
    ]

def applyMagicMethod_math(op, dev=''):
    op = "__" + op + "__"

    try:
        method = getattr(Tensor, op)
        if method is not None:

            def mathWrapper(self, other):
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

                if isinstance(res, Tensor):
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

    isTorchFun = m == '__torch_function__'

    magicRef = None
    try:
        magicRef = getattr(TensorRef, m)
    except:
        pass

    if magicRef is None:
        def makeWrapper(m, magic):
            def magicWrapper(*args, **kwargs):

                refs = []
                args = list(args)
                for a in range(0, len(args)):
                    if isinstance(args[a], TensorRef):
                        refs.append(args[a])
                        args[a] = args[a].toGPU()

                # What an ugly thing...
                if isTorchFun:
                    fun = args[1]
                    types = args[2]
                    tup = args[3]
                    tens = args[0]

                    types = list(types)
                    for t in range(0, len(types)):
                        if types[t] == TensorRef:
                            types[t] = Tensor
                    types = tuple(types)

                    tup = list(tup)
                    defTensMan = None
                    for t in range(0, len(tup)):
                        if isinstance(tup[t], Tensor):
                            tup[t] = TensorRef(tup[t],defTensMan)
                        if isinstance(tup[t], TensorRef):
                            defTensMan = tup[t].proxyInfo.tensorsManager
                            tup[t] = tup[t].toGPU()
                    tup = tuple(tup)

                    args[0] = fun
                    args[1] = types
                    args[2] = tup
                    #args[3] = tens
                    del args[3]

                args = tuple(args)

                try:
                    res = magic(*args, **kwargs)

                    if res is NotImplemented:
                        return getattr(TensorRef, '__'+m[3:])(ref, *args, **kwargs)

                    for ref in refs:
                        ref.toCPU()

                    if isinstance(res, Tensor):
                        res = TensorRef(res, ref.proxyInfo.tensorsManager)
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
