from setuptools import setup, find_packages

setup(
    name='driving_simulator',
    version='0.0.1',
    description='Machine-learning based driving simulator',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)