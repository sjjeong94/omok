from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

exec(open('connect6/version.py').read())

setup(
    name='connect6',
    packages=find_packages(),
    version=VERSION,
    license='MIT',
    description='A Python library for developing Connect6 AI algorithms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='sjjeong94',
    author_email='sjjeong94@gmail.com',
    url='https://github.com/sjjeong94/connect6',
    install_requires=['numpy', 'pygame'],
)
