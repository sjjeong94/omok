from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

exec(open('omok/version.py').read())

setup(
    name='omok',
    packages=find_packages(),
    version=VERSION,
    license='MIT',
    description='Omok',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='sjjeong94',
    author_email='sjjeong94@gmail.com',
    url='https://github.com/sjjeong94/omok',
    install_requires=['numpy', 'pygame', 'onnxruntime'],
)
