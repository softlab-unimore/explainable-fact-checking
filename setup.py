import os
import re

from setuptools import setup, find_packages


def read(*names, **kwargs):
    with open(os.path.join(os.path.dirname(__file__), *names)) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('requirements.txt') as f:
    required = f.read().splitlines()

VERSION = find_version('explainable_fact_checking', '__init__.py')
long_description = read('README.md')

setup(
    name='explainable_fact_checking',
    version=VERSION,
    description='This repository is dedicated to '
                'explaining fact-checking with post-hoc SOTA explainers (like LIME, SHAP). ',
    long_description=long_description,
    author='Andrea Baraldi, Francesco Guerra',
    author_email='baraldian@gmail.com, francesco.guerra@unimore.it',
    url='https://github.com/softlab-unimore/explainable-fact-checking',
    packages=find_packages(where='explainable_fact_checking', exclude=('*test*')),
    license='CC BY-NC 4.0 License',
    package_dir={'': 'src'},
    install_requires=required,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: CC BY-NC 4.0 License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8',
)

