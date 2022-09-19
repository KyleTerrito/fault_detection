from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
                "cmd2==2.4.2",
                "colorama==0.4.4",
                "cycler==0.11.0",
                "fonttools==4.31.2",
                "hdbscan==0.8.28",
                "joblib==1.1.0",
                "kiwisolver==1.4.2",
                "llvmlite==0.38.0",
                "matplotlib==3.5.1",
                "numba==0.55.1",
                "numpy==1.21.5",
                "openpyxl==3.0.10",
                "packaging==21.3",
                "pandas==1.4.1",
                "Pillow==9.1.0",
                "pymoo==0.5.0",
                "pynndescent==0.5.6",
                "pyparsing==3.0.7",
                "python-dateutil==2.8.2",
                "pytz==2022.1",
                "scikit-learn==1.0.2",
                "scipy==1.8.0",
                "six==1.16.0",
                "tabulate==0.8.9",
                "threadpoolctl==3.1.0",
                "tqdm==4.63.1",
                "umap==0.1.1",
                "umap-learn==0.5.2",
]

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