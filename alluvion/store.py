from ._alluvion import Store as Store_
from .variable import variable_class_dict
from typing import Iterable, Union
import numpy as np


class Store(Store_):
    def _create(self,
                is_graphical,
                shape: Union[int, Iterable[int]],
                n: int,
                dtype=np.float32):
        if isinstance(shape, int):
            shape = [shape]
        dim = len(shape)
        numeric_type_str = Store.dtype_to_string(dtype)
        variable_postfix = f"{dim}D{numeric_type_str}{n if n>1 else ''}"
        create_str = "create_graphical" if is_graphical else "create"
        variable_prefix = "Graphical" if is_graphical else ""
        create_func = getattr(super(), f"{create_str}{variable_postfix}")
        return create_func(shape)

    def create_graphical(self,
                         shape: Union[int, Iterable[int]],
                         n: int,
                         dtype=np.float32):
        return self._create(True, shape, n, dtype)

    def create(self,
               shape: Union[int, Iterable[int]],
               n: int,
               dtype=np.float32):
        return self._create(False, shape, n, dtype)

    def create_coated(self,
                      shape: Union[int, Iterable[int]],
                      n: int,
                      dtype=np.float32):
        return self.coat(self.create(shape, n, dtype))

    @classmethod
    def dtype_to_string(cls, dtype):
        if dtype == np.float32:
            return 'float'
        if dtype == np.float64:
            return 'double'
        if dtype == np.uint32:
            return 'uint'
        if dtype == np.int32:
            return 'int'
        raise NotImplementedError(f'{dtype} not yet implemented.')

    @classmethod
    def coat(cls, var):
        return variable_class_dict[type(var).__name__](var)
