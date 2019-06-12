import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="onnx2graphml",
    version="0.0.1",
    author="Mike Holcomb",
    author_email="mike-holcomb@users.noreply.github.com",
    description="A conversion utility for generating GraphML files from ONNX files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mike-holcomb/onnx2graphml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 3 - Alpha",
        "Topic :: Text Processing"
    ],
)
