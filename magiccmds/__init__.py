#__init__.py
__all__ = ['timeit', 'time', 'prun', 'version']
from .implementations import (magic_timeit as timeit, magic_time as time, magic_prun as prun)
version = "0.0.13.1-2"
