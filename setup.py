from setuptools import setup, find_packages

def parse_requirements(file):
    with open('requirements.txt', 'rb') as f:
        return f.read().decode().splitlines()

requirements = parse_requirements("requirements.txt")
print("REQUIREMENTS: ", requirements)

VERSION = '0.1.0'
DESCRIPTION = 'A package for simulating molecules on a lattice.'
LONG_DESCRIPTION = 'A package that simulates multiple types of micromolecules such as proteins on a simple cubic or FCC lattice.'

setup(
    name="LatticePy",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Sohit Miglani",
    author_email="sohitmiglani@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    keywords='proteins,simulation,lattice',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Protein Biologists",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)
