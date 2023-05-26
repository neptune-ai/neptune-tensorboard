from functools import wraps


def register_pre_hook(original, neptune_hook, run, base_namespace):
    @wraps(original)
    def wrapper(*args, **kwargs):
        neptune_hook(*args, **kwargs, run=run, base_namespace=base_namespace)
        return original(*args, **kwargs)

    return wrapper
