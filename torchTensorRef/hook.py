import inspect
import copy

origInspectSignature = inspect.signature

props = {}

class EmptyObj:
    pass

class Hooks:
    def module_register_parameter(self, name, param):
        self._parameters[name] = param

    def module_register_buffer(self, name, tensor, persistent):
        self._buffers[name] = tensor
        #todo: persistent

    '''
    def hook_class_Module(mod):
        class ModuleHook(mod):
            def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
                super().__setattr__(name, value)
        return ModuleHook
    '''

    def inspect(mod):
        def signature(fn):
            res = origInspectSignature(fn)

            if 'tensor' in props:
                '''
                if res.return_annotation is props['tensor']:
                    nres = EmptyObj()
                    setattr(nres, 'parameters', res.parameters)
                    setattr(nres, 'return_annotation', TensorRefBase)
                    res = nres
                '''

                if res.return_annotation is TensorRefBase:
                    nres = EmptyObj()
                    setattr(nres, 'parameters', res.parameters)
                    setattr(nres, 'return_annotation', props['tensor'])
                    res = nres

            return res

        mod.signature = signature 
        return mod

class TensorRefBase:
    pass

TensorRefBase.__module__ = "__torch__.torch.classes.TensorRef"