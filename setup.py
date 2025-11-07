from setuptools import setup
from Cython.Build import cythonize
import glob

setup(
    ext_modules = cythonize(
        glob.glob("src/python_src/*.py"), 
        compiler_directives={'language_level': "3"}
    ),
)