from setuptools import find_packages, setup

setup(
    name='atariari',
    packages=find_packages(exclude=['scripts']),
    version='0.0.1',
    install_requires=['gym', 'opencv-python']
)