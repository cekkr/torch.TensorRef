import torch


class TorchWrapper:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        attr = getattr(self.target, name)
        return attr

    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)


"""
global_vars = [] #list(globals().keys())

for var in global_vars:
    val = globals().get(var)
    if isinstance(val, object):
        globals()[var] = TorchWrapper(val)
"""

global_vars = list(torch.keys())

for var in global_vars:
    val = torch.get(var)
    if isinstance(val, object):
        torch[var] = TorchWrapper(val)
