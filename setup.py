from setuptools import setup,Extension
from Cython.Build import cythonize
import glob

extensions = [
    Extension("*", ["src/scripts/*.py"])  # This tells Cython where to find .py files
]
setup(
    ext_modules = cythonize(
        extensions, 
        compiler_directives={'language_level': "3"}
    ),
)
