from setuptools import find_packages, setup

setup(
    name="digitalproject06",
    version="0.1.0",
    packages=find_packages(include=["data*", "models*", "training*", "utils*"]),
    python_requires=">=3.8",
)
