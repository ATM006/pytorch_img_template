"""
This file is based on fastai's torch_core module
"""
from imports.torch import *
from core_funcs.core import *
from torch.utils.data._utils.collate import default_collate


ItemList = Collection[Union[Tensor, ItemBase, 'ItemList', float, int]]


def to_data(b: ItemList):
    """
    Recursively map lists of items in `b` to their wrapped data
    """
    return recurse(lambda x: x.data if isinstance(x, ItemBase) else x, b)


def data_collate(batch: ItemList) -> Tensor:
    "Convert `batch` items to tensor data"
    return default_collate(to_data(batch))

