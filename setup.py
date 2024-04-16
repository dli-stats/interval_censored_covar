"""Setup script."""

from setuptools import setup, find_packages

setup(
    name='interval_censored_covar',
    version='0.1dev',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'chex==0.1.6',
        'jax==0.4.4',
        'jax-newton-raphson @ git+https://github.com/thisiscam/jax_newton_raphson@3ea78f5d158d4a1a3094996f613626884ecc6995',
        'jaxlib==0.4.4',
        'jaxopt==0.6',
        'matplotlib==3.7.0',
        'numba==0.56.4',
        'sacred==0.8.4',
        'scipy==1.10.0',
        'simple-slurm==0.2.5',
        'simpleeval==0.9.12',
        'tqdm==4.64.1',
    ],
    license='MIT license',
    long_description=open('README.md', encoding='utf-8').read(),
)
