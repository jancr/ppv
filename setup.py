import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

print(required)
with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    install_requires=required,
    name='ppv',
    version='0.1',
    scripts=[],
    author="Jan Christian Refsgaard",
    author_email="jancrefsgaard@gmail.com",
    description="Petidomics: Predict Peptide Variant tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #  url="https://github.com/jancr/sequtils",
    #  packages=setuptools.find_packages(),
    packages=['ppv'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
