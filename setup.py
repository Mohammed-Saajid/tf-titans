from setuptools import setup, find_packages

setup(
    name="tf-titans",
    version="0.1.3",
    packages=find_packages(),
    install_requires=["tensorflow>=2.11.0"],
    author="Mohammed Saajid",
    author_email="mohammedsaajid23@gmail.com",
    description="Implementation of Titans Architecture in TensorFlow 2.x",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Mohammed-Saajid/tf-titans",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
