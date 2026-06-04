from setuptools import setup, find_packages

setup(
    name="custom_objects",
    version="1.0.0",
    description="Custom objects for TensorFlow/Keras models",
    packages=find_packages(),  # Automatically find submodules
    install_requires=[
        "tensorflow",  # Add any other dependencies here
    ],
)
