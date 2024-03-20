import inspect
import types

def find_arg_count(func):
    arg_count = 0
    while True:
        args = [None] * arg_count
        try:
            func(*args)
        except TypeError as e:
            se = str(e)
            if 'missing' in se:
                spl = se.split('missing ')
                spl = spl[1].splut(' ')[0]
                return arg_count + int(spl)

            if "positional argument" in str(e) or "expected at most" in str(e):
                #print(f"Function '{func.__name__}' expects {arg_count-1} arguments.")
                break
            elif "takes no arguments" in str(e):
                #print(f"Function '{func.__name__}' expects 0 arguments.")
                return 0
        except Exception as e:
            # Catch all other exceptions to prevent the loop from breaking due to unrelated errors
            #print(f"An exception occurred: {e}")
            se = str(e)
            if "can't be None" not in se:
                return arg_count

        arg_count += 1

        if arg_count > 10:
            return -1
    return arg_count


def GetNumArgs(func):
    # Check if the function is a compiled (built-in) function
    #if not isinstance(func, types.BuiltinFunctionType):
    #    return -1
    
    # Attempt to retrieve the function's signature
    try:
        sig = inspect.signature(func)
    except ValueError as e:
        # Handle the case where a signature cannot be found
        nArgs = find_arg_count(func)
        return nArgs

    # Count the parameters, excluding *args and **kwargs
    params = [param for param in sig.parameters.values()
              if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
    return len(params)


