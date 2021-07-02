import importlib
from ._alluvion import Store
from .variable import variable_class_dict
from typing import Iterable, Union
import numpy as np


class Depot(Store):
    def __init__(self, default_dtype=np.float32):
        self.default_dtype = default_dtype
        numeric_type_str = self.dtype_to_string(self.default_dtype)
        float_typed_classnames = [
            'Runner', 'Pile', 'SolverDf', 'SolverIi', 'DisplayProxy'
        ]
        _al = importlib.import_module("._alluvion", "alluvion")
        for classname in float_typed_classnames:
            setattr(self, classname,
                    getattr(_al, f"{classname}{numeric_type_str}"))

        float_typed_dg_classnames = [
            'BoxDistance', 'CylinderDistance', 'Distance', 'SphereDistance'
        ]
        for classname in float_typed_dg_classnames:
            setattr(self, classname,
                    getattr(_al.dg, f"{classname}{numeric_type_str}"))

        # cannot use setattr
        self.cn = getattr(self, f"cn{numeric_type_str}")
        self.copy_cn = getattr(self, f"copy_cn{numeric_type_str}")
        self.get_display_proxy = getattr(
            self, f"get_display_proxy{numeric_type_str}")

        float_typed_vector_names = ['3', '4']
        for name in float_typed_vector_names:
            setattr(self, f"f{name}", getattr(_al,
                                              f"{numeric_type_str}{name}"))

        super().__init__()

    def _create(self,
                is_graphical,
                shape: Union[int, Iterable[int]],
                n: int,
                dtype=np.float32):
        if isinstance(shape, int):
            shape = [shape]
        dim = len(shape)
        numeric_type_str = self.dtype_to_string(dtype)
        variable_postfix = f"{dim}D{numeric_type_str}{n if n>1 else ''}"
        create_str = "create_graphical" if is_graphical else "create"
        variable_prefix = "Graphical" if is_graphical else ""
        create_func = getattr(super(), f"{create_str}{variable_postfix}")
        return create_func(shape)

    def create_graphical(self,
                         shape: Union[int, Iterable[int]],
                         n: int,
                         dtype=None):
        return self._create(True, shape, n, dtype)

    def create(self, shape: Union[int, Iterable[int]], n: int, dtype=None):
        return self._create(False, shape, n, dtype)

    def create_coated(self,
                      shape: Union[int, Iterable[int]],
                      n: int,
                      dtype=None):
        return self.coat(self.create(shape, n, dtype))

    def dtype_to_string(self, dtype):
        if dtype is None:
            dtype = self.default_dtype
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
