import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(
    name="numpy-binnn",
    version="0.0.1",
    description="Codes for binary neural networks with numpy.",
    author="Adam Wentz",
    author_email="adam@adamwentz.com",
    ext_modules=cythonize("binnn/*.pyx"),
    include_dirs=[numpy.get_include()],
    install_requires=[
        "Cython",
        "numpy",
    ],
)
