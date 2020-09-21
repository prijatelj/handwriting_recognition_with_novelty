"""Install experiments package."""
from setuptools import setup

long_description = ''
with open('README.md', 'r') as f:
    long_description = f.read()

install_requires = ''
with open('requirements.txt', 'r') as f:
    install_requires = f.read()

setup(
    name='hwr_novelty',
    version='0.1.0',
    author='Derek S. Prijatelj',
    author_email='dprijate@nd.edu',
    packages=[
        'experiments',
        'experiments.research',
        'experiments.research.par_v1',
        'experiments.research.par_v1.grieggs',
    ],
    #scripts
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    install_requires=install_requires,
    python_requires='>=3.7',
)
