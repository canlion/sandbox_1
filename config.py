from collections import namedtuple


def dict2namedtuple(elem: dict)-> namedtuple:
    """Convert configuration dict to namedtuple"""
    if isinstance(elem, dict):
        for key, val in elem.items():
            elem[key] = dict2namedtuple(val)
        temp_template = namedtuple('temp_template', sorted(elem))
        return temp_template(**elem)
    else:
        return elem
