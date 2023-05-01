from __future__ import annotations

from setuptools import setup

setup(
    name='ssa-solvers',
    version='0.0.1',
    description='A minimalistic Python package for simulating stochastic Master equations',
    url='https://github.com/aivarsoo/ssa-solvers',
    author='Aivar Sootla',
    author_email='aivsoo10@gmail.com',
    license='MIT',
    packages=['ssa_solvers'],
    install_requires=['torch',  # tested on torch=1.8.2  and torch=1.12.0
                      'einops',
                      'jupyterlab',
                      'matplotlib',
                      'line-profiler',
                      'scipy',
                      'numpy'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
    ],
)
