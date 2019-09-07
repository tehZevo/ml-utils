from setuptools import setup, find_packages

setup(name='ml_utils',
  version='0.1.0',
  install_requires = [
    'matplotlib',
    'tensorflow',
    'numpy',
    'seaborn'
  ],
  packages=find_packages())
