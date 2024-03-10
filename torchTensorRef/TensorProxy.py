class Tensor:

    def __init__(self, *args, **kwargs):
        # Initialize the target object, passing all arguments
        target = TensorBase(*args, **kwargs)
        setattr(self, "target", target)

    def __add__(self, other):
        self.toGPU()
        res = None
        if isinstance(other, (TensorBase)):
            res = Tensor(self.target + other, self.gpuDevice)
        elif isinstance(other, Tensor):
            other.toGPU()
            res = Tensor(self.target + other.target, self.gpuDevice)
            other.toCPU()
        else:
            raise TypeError("Unsupported type for addition")

        self.toCPU()
        return res

    def __sub__(self, other):
        self.toGPU()
        res = None
        if isinstance(other, (TensorBase)):
            res = Tensor(self.target - other, self.gpuDevice)
        elif isinstance(other, Tensor):
            other.toGPU()
            res = Tensor(self.target - other.target, self.gpuDevice)
            other.toCPU()
        else:
            raise TypeError("Unsupported type for addition")

        self.toCPU()
        return res

    def __mul__(self, other):
        self.toGPU()
        res = None
        if isinstance(other, (TensorBase)):
            res = Tensor(self.target * other, self.gpuDevice)
        elif isinstance(other, Tensor):
            other.toGPU()
            res = Tensor(self.target * other.target, self.gpuDevice)
            other.toCPU()
        else:
            raise TypeError("Unsupported type for addition")

        self.toCPU()
        return res

    def __setattr__(self, key, value):
        if key == "gpuDevice" or key == "target":
            super().__setattr__(key, value)
        else:
            setattr(self.target, key, value)

    def __getattr__(self, name):
        if name == "target" or name == "gpuDevice":
            try:
                return super().__getattribute__(name)
            except:
                if name == "target":
                    return self

        # Delegate attribute access to the target object
        attr = getattr(super(), name)

        if name == "to":

            def ignore(*args, **kwargs):
                dev = args[0]
                if isinstance(dev, str):
                    setattr(self, "gpuDevice", dev)
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
                    if isinstance(value, Tensor):
                        proxies.append(value)
                        args[a] = value.toGPU()

                for key, value in kwargs.items():
                    print(f"{key}: {value}")
                    if isinstance(value, Tensor):
                        proxies.append(value)
                        kwargs[key] = value.toGPU()

                # Perform the call to the original function
                result = attr(*args, **kwargs)
                # Optionally, process the result before returning

                # back to CPU
                for value in proxies:
                    value.toCPU()

                if isinstance(result, TensorBase):
                    if name == "cpu":
                        self.target = result
                    else:
                        result = Tensor(result, self.gpuDevice)
                        result.toCPU()

                return result

            return wrapper
        else:
            return attr

    def toGPU(self):
        if self.target.is_cpu:
            dev = getattr(self, "gpuDevice")
            if dev is not None and dev is not "cpu":
                self.target = self.target.to(self.gpuDevice)

        return self.target

    def toCPU(self):
        if self.target.is_cuda:
            self.target = self.target.to("cpu")

        return self.target
