import os
from contextlib import contextmanager

@contextmanager
def temp_envvar(envvar_name: str, value: str) -> None:
    """
    Set an environment variable temporarily for the duration of a `with` block

    Args:
    ----
        key (str): environment variable name
        value (str): value to set the environment variable to

    """
    original_value = os.environ.get(envvar_name)
    os.environ[envvar_name] = value
    try:
        yield
    finally:
        if original_value is None:
            del os.environ[envvar_name]
        else:
            os.environ[envvar_name] = original_value
