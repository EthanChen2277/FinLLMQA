from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='stream_autogen',
    version='0.0.1',
    packages=find_packages(),
    description='User modified autogen',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=['data'],
    classifiers=[
        "Intended Audience :: All Groups",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas'
    ],
    python_requires='>=3.10',
    include_package_data=True,
    package_data={'': ['*.csv', '*.yaml', '*.dll', '*.lib', '*.pyd', '*.crt', '*.ini', '*.log4cxx']},
)
