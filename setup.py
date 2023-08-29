from setuptools import setup

setup(
    name='ssa-solvers',
    version='0.0.1',
    description='A minimalistic Python package for simulating stochastic Master equations',
    url='https://github.com/aivarsoo/ssa-solvers',
    author='Aivar Sootla',
    author_email='aivar.sootla@gmail.com',
    license='MIT',
    packages=['ssa_solvers'],
    install_requires=['torch==2.0.0',
                      'einops==0.4.1',
                      'jupyterlab==3.4.3',
                      'matplotlib==3.5.2',
                      'line-profiler==3.5.1',
                      'scipy==1.8.1',
                      'numpy==1.23.1',
                      "pandas==1.4.3",
                      'pre-commit==3.3.3',
                      'fire==0.5.0'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
    ],
)
