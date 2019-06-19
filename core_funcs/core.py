""""
Inherit from fastai library - fastai.core
`fastai.core` contains essential util functions to format and split data
"""
from imports.core import *


ListOrItem = Union[Collection[Any], int, float, str]
OptListOrItem = Optional[ListOrItem]


def num_cpus() -> int:
    """
    Get number of cpus
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


_default_cpus = min(16, num_cpus())
defaults = SimpleNamespace(cpus=_default_cpus,
                           cmap='viridis',
                           return_fig=False,
                           silent=False)


def is_listy(x: Any) -> bool:
    return isinstance(x, (tuple, list))


def is_tuple(x: Any) -> bool:
    return isinstance(x, tuple)


def is_dict(x: Any) -> bool:
    return isinstance(x, dict)


def is_pathlike(x: Any) -> bool:
    return isinstance(x, (str, Path))


def recurse(func: Callable, x: Any, *args, **kwargs) -> Any:
    if is_listy(x):
        return [recurse(func, o, *args, **kwargs) for o in x]
    if is_dict(x):
        return {k: recurse(func, v, *args, **kwargs) for k, v in x.items()}
    return func(x, *args, **kwargs)


def recurse_eq(arr1, arr2):
    if is_listy(arr1):
        return is_listy(arr2) and len(arr1) == len(arr2) and np.all([recurse_eq(x, y)
                                                                     for x, y in zip(arr1, arr2)])
    else:
        return np.all(np.atleast_1d(arr1 == arr2))


def listify(p: OptListOrItem=None, q: OptListOrItem=None):
    """
    Make `p` listy and the same length as `q`
    """
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    # Rank 0 Tensor in Pytorch are Iterable but do not have a length
    else:
        try:
            a = len(p)
        except:
            p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1:
        p = p * n
    assert len(p) == n, f"List len mismatch ({len(p)} vs {n})"
    return list(p)




class ItemBase:
    """
    Base item type in fastai library
    """
    def __init__(self, data: Any):
        self.data = self.obj = data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {str(self)}"

    def show(self, ax: plt.Axes, **kwargs):
        """
        Subclass this method for further customization
        """
        ax.set_title(str(self))

    def apply_tfms(self, tfms: Collection, **kwargs):
        """
        Subclass this method if you want to apply data augmentation with `tfms` to this `ItemBase`
        """
        if tfms:
            raise Exception(f"Not implemented: You can't apply transforms to this type of item ({self.__class__.__name__})")
        return self

    def __eq__(self, other):
        return recurse_eq(self.data, other.data)
