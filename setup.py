#!/usr/bin/env python

import os

from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as fp:
    install_requires = [line.strip() for line in fp if line.strip()]

setup(
    name='Nail Gun',
    version='0.1',
    author='Aliaksei Sukharevich',
    packages=['nailgun'],
    install_requires=install_requires
)
