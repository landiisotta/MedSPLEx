# setup.py
from setuptools import setup, find_packages

setup(
    name='MedSPLEx',               # Replace with your package name
    version='0.1.0',               # Replace with your version
    description='Stigmatizing and Privileging Language Extractor for Clinical Text',
    author='Isotta Landi, Eugenia Alleva',
    author_email='isotta.landi2@mssm.edu',
    url='https://github.com/landiisotta/MedSPLEx',
    packages=find_packages(),      # Automatically finds and includes all packages in the current directory
    install_requires=[             # Add your required dependencies here
        # 'numpy>=1.18.5',
        # 'pandas>=1.1.0',
    ],
    python_requires='>=3.6',       # Specify supported Python versions
)