from setuptools import setup, find_packages
from LatticePy import __version__

def parse_requirements(file):
    with open('requirements.txt', 'rb') as f:
        return f.read().decode().splitlines()
    
with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = parse_requirements("requirements.txt")
print("REQUIREMENTS: ", requirements)

DESCRIPTION = 'A package for simulating molecules on a lattice.'

setup(
    name="LatticePy",
    version=__version__,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sohit Miglani",
    author_email="sohitmiglani@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    keywords='proteins,simulation,lattice',
    classifiers= [
        "Development Status :: 3 - Alpha",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
