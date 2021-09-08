import importlib
from ._alluvion import NumericType
import numpy as np


def type_enum_to_dtype_func(self):
    numeric_type = self.get_type()
    if (numeric_type == NumericType.f32):
        return np.float32
    if (numeric_type == NumericType.f64):
        return np.float64
    if (numeric_type == NumericType.i32):
        return np.int32
    if (numeric_type == NumericType.u32):
        return np.uint32
    return numeric_type


def get_func(self, shape=None):
    dtype = self.type_enum_to_dtype()
    if shape is None:
        shape = self.get_shape()
    if type(shape) is not list:
        shape = [shape]
    dst = np.empty(
        np.prod(shape) * self.get_num_primitives_per_element(), dtype)
    self.get_bytes(dst.view(np.ubyte))
    if self.get_num_primitives_per_element() == 1:
        return dst.reshape(shape)
    else:
        return dst.reshape(*shape, -1)


def set_func(self, src, offset=0):
    dtype = self.type_enum_to_dtype()
    if src.dtype != dtype:
        src = src.astype(dtype)
    self.set_bytes(src.view(np.ubyte), offset)


_al = importlib.import_module("._alluvion", "alluvion")

variable_class_dict = {}
for class_name in dir(_al):
    if not class_name.startswith("Variable") and not class_name.startswith(
            "GraphicalVariable"):
        continue
    is_graphical = class_name.startswith("GraphicalVariable")
    coated_class_name = "Coated" + class_name
    variable_class_dict[class_name] = type(
        coated_class_name, (getattr(_al, class_name), ), {
            "type_enum_to_dtype": type_enum_to_dtype_func,
            "get": get_func,
            "set": set_func,
            "is_graphical": is_graphical
        })
