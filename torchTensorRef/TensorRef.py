from torch import Tensor


class ProxyInfo:
    def __init__(self):
        self.device = "cpu"


class TensorRef:
    """
    def __init__(self, *args, **kwargs):
        # Initialize the target object, passing all arguments
        target = Tensor(*args, **kwargs)
        setattr(self, "target", target)

        setattr(self, "proxyInfo", ProxyInfo())
    """

    def __init__(self, target, tensorsManager):
        setattr(self, "target", target)
        setattr(self, "proxyInfo", ProxyInfo())
        self.proxyInfo.tensorsManager = tensorsManager

    """
    def __add__(self, other):
        self.toGPU()
        res = None
        if isinstance(other, (Tensor)):
            res = TensorRef(self.target + other, self.proxyInfo.tensorsManager)
        elif isinstance(other, TensorRef):
            other.toGPU()
            res = TensorRef(self.target + other.target, self.proxyInfo.tensorsManager)
            other.toCPU()
        else:
            raise TypeError("Unsupported type for addition")

        self.toCPU()
        return res

    def __sub__(self, other):
        self.toGPU()
        res = None
        if isinstance(other, (Tensor)):
            res = TensorRef(self.target - other, self.proxyInfo.tensorsManager)
        elif isinstance(other, TensorRef):
            other.toGPU()
            res = TensorRef(self.target - other.target, self.proxyInfo.tensorsManager)
            other.toCPU()
        else:
            raise TypeError("Unsupported type for addition")

        self.toCPU()
        return res

    def __mul__(self, other):
        self.toGPU()
        res = None
        if isinstance(other, (Tensor)):
            res = TensorRef(self.target * other, self.proxyInfo.tensorsManager)
        elif isinstance(other, TensorRef):
            other.toGPU()
            res = TensorRef(self.target * other.target, self.proxyInfo.tensorsManager)
            other.toCPU()
        else:
            raise TypeError("Unsupported type for addition")

        self.toCPU()
        return res
    """

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
                dev = args[0]
                if isinstance(dev, str):
                    self.proxyInfo.device = dev
                    return self
                else:
                    return attr(*args, **kwargs)

            return ignore

        if callable(attr):
            # If the original attribute is callable, we return a new wrapper function
            def wrapper(*args, **kwargs):
                # Here you can analyze the arguments before calling the original function
                print(f"Calling {name}")

                # look for tensors on CPU
                proxies = []
                for a in range(0, len(args)):
                    value = args[a]
                    if isinstance(value, TensorRef):
                        proxies.append(value)
                        args[a] = value.toGPU()

                for key, value in kwargs.items():
                    print(f"{key}: {value}")
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
        if self.target.is_cpu:
            dev = self.proxyInfo.tensorsManager.device
            if dev is not None and dev is not "cpu":
                self.target = self.target.to(dev)

        return self.target

    def toCPU(self):
        if not self.target.is_cpu:
            self.target = self.target.to("cpu")

        return self.target


# Create math operation magic functions
ops = ["add", "sub", "truediv", "floordiv", "mul", "mod", "divmod", "pow"]


def applyMagicMethod(op):
    op = "__" + op + "__"

    try:
        method = getattr(Tensor, op)
        if method is not None:

            def mathWrapper(self, other):
                self.toGPU()
                res = None
                if isinstance(other, Tensor):
                    res = method(self.target, other)
                elif isinstance(other, TensorRef):
                    other.toGPU()
                    res = method(self.target, other.target)
                    other.toCPU()
                else:
                    raise TypeError("Unsupported type for addition")

                res = TensorRef(
                    res, self.proxyInfo.tensorsManager
                )
                
                res.toCPU()

                self.toCPU()
                return res

            setattr(TensorRef, op, mathWrapper)
    except:
        ignore = True

for op in ops:
    applyMagicMethod(op)
    applyMagicMethod("r" + op)
    applyMagicMethod("i" + op)
