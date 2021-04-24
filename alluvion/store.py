from ._alluvion import Store as Store_
from .variable import Variable
from typing import Iterable, Union


class Store(Store_):
    def create_float(self, n: int, shape: Union[int, Iterable[int]]):
        if isinstance(shape, int):
            shape = [shape]
        dim = len(shape)
        create_func = getattr(super(), f"create{dim}DF{n if n>1 else ''}")
        return Variable(create_func(shape))
