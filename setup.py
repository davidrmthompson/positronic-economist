try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='The Postronic Economist',
    version="1",
    description='A computational system for analyzing economic mechanisms',
    author='David R. M. Thompson',
    author_email='daveth@cs.ubc.ca',
    url='https://github.com/davidrmthompson/positronic-economist',
    packages=['posec']
)
