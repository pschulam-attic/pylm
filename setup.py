from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension(name="lm._lm",
              ["lm.pyx"],
              libraries=["oolm", ]
              )
    ]

setup(
    cmdclass = { "build_ext" : build_ext },
    ext_modules = ext_modules
    )
