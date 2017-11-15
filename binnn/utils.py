import numpy as np


def bit_spectrum(v):
    num_spectra = 1 + np.log2(v.dtype.itemsize * 8).astype(np.uint32)
    x = np.array([[2 ** i] for i in np.arange(num_spectra)], dtype=v.dtype)
    x = x + np.zeros((num_spectra, v.size)).astype(v.dtype)
    return np.bitwise_and(x, v)


if __name__ == '__main__':
    a = np.arange(3).reshape((1, 3)).astype(np.uint32)
    print(a)
    print(bit_spectrum(a))
