
def Hooks():
    def module_register_parameter(self, name, param):
        self._parameters[name] = param

    def module_register_buffer(self, name, tensor, persistent):
        self._buffers[name] = tensor
        #todo: persistent