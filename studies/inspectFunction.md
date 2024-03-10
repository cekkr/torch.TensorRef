Yes, it is possible to inspect a Python function to check what arguments it requires and what its return type is, especially in Python 3.5 and above, which introduced type hints. The `inspect` module in the Python Standard Library provides tools for introspecting live objects, including functions, and can help you extract this information.

### Inspecting Function Arguments

You can use `inspect.signature()` to get the signature of a function, which includes its parameters:

```python
import inspect

def example_function(param1: int, param2: str = "default") -> str:
    return f"param1: {param1}, param2: {param2}"

signature = inspect.signature(example_function)
parameters = signature.parameters

for name, param in parameters.items():
    print(f"Name: {name}")
    print(f"  Type: {param.annotation}")
    print(f"  Default: {param.default}")
```

This code will give you detailed information about the parameters of `example_function`, including their names, expected types (annotations), and default values.

### Inspecting Return Type

The return type of a function can also be inspected using the same `signature` object:

```python
return_type = signature.return_annotation
print(f"Return type: {return_type}")
```

### Example

Putting it all together to inspect both arguments and return type:

```python
import inspect

def example_function(param1: int, param2: str = "default") -> str:
    return f"param1: {param1}, param2: {param2}"

signature = inspect.signature(example_function)
parameters = signature.parameters
return_type = signature.return_annotation

print("Parameters:")
for name, param in parameters.items():
    print(f"  Name: {name}")
    print(f"    Type: {param.annotation}")
    print(f"    Default: {param.default if param.default is not param.empty else 'No default'}")

print(f"Return type: {return_type}")
```

### Caveats

- **Dynamic Types**: Type annotations in Python are not enforced at runtime. They are primarily for documentation and tooling purposes. Therefore, the actual types of the arguments or the return value at runtime might not match the annotations.
- **Type Annotations Must Be Present**: This approach relies on type annotations being present in the function definition. If the function does not include annotations, the `param.annotation` and `return_annotation` will be `inspect.Parameter.empty` and `inspect.Signature.empty`, respectively.

- **Generics and Complex Types**: For more complex type annotations (e.g., generics like `List[int]`), the `inspect` module will report these annotations as they are written, but interpreting them correctly in all contexts might require additional logic or third-party libraries like `typing`.

Inspecting functions in this manner can be especially useful for building tools, documentation generators, or for dynamic invocation scenarios where you need to validate inputs or outputs based on their types.
