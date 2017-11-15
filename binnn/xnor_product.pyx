"""
Computes the bitwise XNOR product.

Adapted from: https://gist.github.com/craffel/e470421958cad33df550
which was adapted from: https://gist.github.com/aldro61/f604a3fa79b3dec5436a by Alexandre Drouin
"""
import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport int32_t, uint32_t, uint64_t


cdef extern int __builtin_popcount(unsigned int) nogil
cdef extern int __builtin_popcountll(unsigned long long) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _xnor_product_32(uint32_t[:, :] a, uint32_t[:,:] b, int32_t[:,:] t) nogil:
    cdef int i
    cdef int j
    cdef int k
    cdef int sum
    cdef int min_row = -32 * a.shape[1]  # assume 100% -1
    for i in xrange(a.shape[0]):
        for j in xrange(b.shape[1]):
            sum = 0
            for k in xrange(a.shape[1]):
                sum += __builtin_popcount(~(a[i, k] ^ b[k, j]))
            t[i, j] = min_row + sum * 2  # each +1 also negates an assumed -1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _xnor_product_64(uint64_t[:, :] a, uint64_t[:,:] b, int32_t[:,:] t) nogil:
    cdef int i
    cdef int j
    cdef int k
    cdef int sum
    cdef int min_row = -64 * a.shape[1]  # assume 100% -1
    for i in xrange(a.shape[0]):
        for j in xrange(b.shape[1]):
            sum = 0
            for k in xrange(a.shape[1]):
                sum += __builtin_popcountll(~(a[i, k] ^ b[k, j]))
            t[i, j] = min_row + sum * 2  # each +1 also negates an assumed -1


def xnor_product(a, b, target=None):
    if target is None:
        a_rows, _ = a.shape
        _, b_cols = b.shape
        target_shape = (a_rows, b_cols)
        target = np.zeros(target_shape).astype(np.int32)
    if a.dtype == np.uint32:
        _xnor_product_32(a, b, target)
    elif a.dtype == np.uint64:
        _xnor_product_64(a, b, target)
    elif not np.issubdtype(a.dtype, np.integer):
        raise ValueError("dtype {} not supported.".format(a.dtype))
    return target
