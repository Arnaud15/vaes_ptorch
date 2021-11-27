from setuptools import find_packages, setup

setup(
    name="vaes-ptorch",
    version="0.0.1",
    author="Arnaud Autef",
    license="MIT",
    description="Simple VAE implementations in pytorch.",
    python_requires=">=3.6.2",
    packages=find_packages(),
    install_requires=["torch"],
    url="https://github.com/Arnaud15/vaes_ptorch",
)
