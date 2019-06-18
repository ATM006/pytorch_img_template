"""
Inherit from fastai library - fastai.imports.core
"""
import csv, gc, gzip, os, pickle, shutil, sys, warnings, yaml, io, subprocess
import math, matplotlib.pyplot as plt, numpy as np, pandas as pd, random
import abc, collections, itertools, json, operator, pathlib
import mimetypes, inspect, typing, functools, importlib, weakref
import html, re, requests, tarfile, numbers, bz2

from abc import abstractmethod, abstractproperty
from collections import abc, Counter, defaultdict, Iterable, namedtuple, OrderedDict

import concurrent
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass, field, InitVar
from enum import Enum, IntEnum
from functools import partial, reduce
from pdb import set_trace
from matplotlib import patches, patheffects
from numpy import array, cos, exp, log, tan, tanh
from operator import attrgetter, itemgetter
from pathlib import Path
from warnings import warn
from contextlib import contextmanager
from fastprogress.fastprogress import MasterBar, ProgressBar
from fastprogress.fastprogress import master_bar, progress_bar
from matplotlib.patches import Patch
from pandas import Series, DataFrame
from io import BufferedWriter, BytesIO

import pkg_resources

# For type annotation
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union
from types import SimpleNamespace

