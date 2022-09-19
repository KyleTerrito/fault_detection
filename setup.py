from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2"]

setup(
    name="autocluster",
    version="0.0.1",
    author="Luis Briceno-Mena",
    author_email="lbrice1@lsu.edu",
    description="An unsupervised learning optimizer",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/lbrice1/fault_detection",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
)