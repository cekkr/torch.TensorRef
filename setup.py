# setup.py

from setuptools import find_packages, setup

setup(
    name="torchTensorRef",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here.
        # For example: 'requests >= 2.22.0',
        "torch"
    ],
    author="Riccardo Cecchini",
    author_email="rcecchini.ds@gmail.com",
    description="A torch.Tensor wrapper and manager",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cekkr/torch.TensorRef",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
