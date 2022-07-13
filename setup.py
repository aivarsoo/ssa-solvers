from setuptools import setup

setup(
    name='ssa-solvers',
    version='0.0.1',    
    description='A simple Python package for simulating stochastic Master equations',
    url='https://github.com/aivarsoo/ssa-solvers',
    author='Aivar Sootla',
    author_email='aivsoo10@gmail.com',
    license='MIT',
    packages=['ssa_solvers'],
    install_requires=['torch>=1.8.2',
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