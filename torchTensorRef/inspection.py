import inspect
import types

def IsCompiledFunction(func):
    # Check if the function is a compiled (built-in) function
    if not isinstance(func, types.BuiltinFunctionType):
        return -1
    
    # If it is a compiled function, analyze its signature
    try:
        sig = inspect.signature(func)
        # Count the number of parameters that are not *args or **kwargs
        params = [param for param in sig.parameters.values()
                  if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
        return len(params)
    except ValueError:
        # If inspect.signature() cannot process the function, return -1
        # This can happen with some built-in functions
        return -1

