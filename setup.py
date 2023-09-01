from setuptools import setup

setup(
    name='ssa-solvers',
    version='0.0.1',
    description='A small Python package for simulating stochastic Master equations using Pytorch',
    url='https://github.com/aivarsoo/ssa-solvers',
    author='Aivar Sootla',
    license='MIT',
    packages=['ssa_solvers'],
    install_requires=['torch==2.0.0',
                      'einops==0.6.1',
                      'jupyterlab==3.4.3',
                      'matplotlib==3.5.2',
                      'line-profiler==3.5.1',
                      'numpy==1.25.2',
                      'scipy==1.11.2',
                      'xitorch==0.5.1',
                      "pandas==1.4.3",
                      'pre-commit==3.3.3',
                      'fire==0.5.0'
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
    ],
)
