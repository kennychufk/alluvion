from ._alluvion import NumericType
import numpy as np


class Variable:
    def __init__(self, var):
        self.var = var
        numeric_type = self.var.get_type()
        if (numeric_type == NumericType.f32):
            self.dtype = np.float32
        elif (numeric_type == NumericType.f64):
            self.dtype = np.float64
        elif (numeric_type == NumericType.i32):
            self.dtype = np.int32
        elif (numeric_type == NumericType.u32):
            self.dtype = np.uint32
        else:
            self.dtype = numeric_type

    def get(self):
        dst = np.empty(self.var.get_num_elements(), self.dtype)
        self.var.get_bytes(dst.view(np.ubyte))
        if self.var.get_vector_size() == 1:
            return dst.reshape(*self.var.shape)
        else:
            return dst.reshape(*self.var.shape, -1)

    def set(self, src):
        if src.dtype != self.dtype:
            src = src.astype(self.dtype)
        self.var.set_bytes(src.view(np.ubyte))
