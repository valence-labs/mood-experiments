from setuptools import setup, find_packages

setup(
    name="mood",
    version="0.0.2",
    author="Valence Discovery",
    author_email="hello@valencediscovery.com",
    url="https://github.com/valence-platform/mood",
    description="Methods for Molecular Out Of Distribution Generalization",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "mood=mood.cli:main",
        ],
    },
) 
