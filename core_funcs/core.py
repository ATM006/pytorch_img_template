""""
Inherit from fastai library - fastai.core
`fastai.core` contains essential util functions to format and split data
"""
from imports.core import *

def num_cpus() -> int:
    """
    Get number of cpus
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()