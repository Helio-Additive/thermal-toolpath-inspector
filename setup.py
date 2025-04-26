from setuptools import setup, find_packages

setup(
    name="MyApp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "dash",
        "plotly"
    ],
)
