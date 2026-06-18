
from setuptools import Extension, setup

ext_modules = [Extension("onescience.datapipes.materials.matris.cygraph", ["cygraph.pyx"])]

setup(ext_modules=ext_modules, setup_requires=["Cython"])

# python setup.py build_ext --inplace