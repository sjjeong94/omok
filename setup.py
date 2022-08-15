import re
from setuptools import setup, find_packages

exec(open('omok/version.py').read())

setup(
    name='omok',
    packages=find_packages(),
    version=VERSION,
    license='MIT',
    description='Omok',
    author='sjjeong94',
    author_email='sjjeong94@gmail.com',
    url='https://github.com/sjjeong94/omok',
    install_requires=['numpy', 'pygame', 'onnxruntime'],
)
