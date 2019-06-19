from core_funcs.torch_core import *
from torch.utils.data._utils.collate import default_collate

DatasetType = Enum('DatasetType', 'Train Valid Test Single Fix')
__all__ = ['DataBunch', 'DeviceDataLoader', 'DatasetType', 'load_data']

old_dl_init = torch.utils.data.DataLoader.__init__


def intercept_args(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                   num_workers=0, collate_fn=default_collate, pin_memory=True, drop_last=False,
                   timeout=0, worker_init_fn=None):
    """
    Re-initialize the DataLoader object.
    For more information about the parameters:
    https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html
    """
    self.init_kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'sampler': sampler,
                        'batch_sampler': batch_sampler, 'num_workers': num_workers,
                        'collate_fn': collate_fn, 'pin_memory': pin_memory, 'drop_last': drop_last,
                        'timeout': timeout, 'worker_init_fn': worker_init_fn}
    old_dl_init(self, dataset, **self.init_kwargs)


torch.utils.data.DataLoader.__init__ = intercept_args


def DataLoader___getattr__(dl, k: str) -> Any:
    return getattr(dl.dataset, k)


def DataLoader___setattr__(dl, data: Any):
    dl.__dict__.update(data)


@dataclass
class DeviceDataLoader():
    """
    Bind a `DataLoader` to a `torch.device`"
    """
    # Define types
    dl: DataLoader
    device: torch.device
    tfms: List[Callable]=None  # List of transformation
    collate_fn: Callable=data_collate
    def __post_init__(self):
        self.dl.collate_fn = self.collate_fn
        self.tfms = listify(self.tfms)








