import json


def is_serializable(d: dict) -> bool:
    """Checks whether a dictionary is JSON-serializable.

    Args:
        d (dict): Input dictionary to test.

    Returns:
        bool: True when serializable, otherwise False.
    """
    try:
        json.dumps(d)
        return True
    except (TypeError, OverflowError):
        return False