"""Small validation helpers shared by the new physical models."""
from math import isfinite


def number(value, name, *, minimum=0.0, strictly_positive=False, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if (not isfinite(result) or result < minimum
            or (strictly_positive and result <= 0)
            or (maximum is not None and result > maximum)):
        raise ValueError(f"Invalid {name}: {value}")
    return result


def boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value
