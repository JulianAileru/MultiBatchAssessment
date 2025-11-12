from setuptools import setup,Extension
from Cython.Build import cythonize
import glob

files = glob.glob("./scripts/*.py")
setup(
    ext_modules = cythonize(files)
)
