from setuptools import setup

setup(
    name="cal",
    packages=[
        'data',
        'inference',
        'train'
    ],
    package_dir={
        'data': './data',
        'inference': './inference',
        'train': './train'
    },
)