"""
structure taken from https://github.com/pypa/sampleproject
"""
from setuptools import setup, find_packages

setup(name='naoth_common_tools',
      version="0.2",
      author='NaoTH Berlin United',
      author_email='nao-team@informatik.hu-berlin.de',
      description='Python helper functions for the NAOTH deep learning tools',
      packages=find_packages(where="src"),  # Required
      license='Apache License 2.0',
      zip_safe=False,
      package_dir={"": "src"},  # Optional
      )
