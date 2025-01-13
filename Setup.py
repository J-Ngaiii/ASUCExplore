from setuptools import setup, find_packages

setup(
    name="asuc_explore",  # Package name
    version="1.0.0",  # Version of your package
    author="Jonathan Ngai (ASUC Senator 2024-2025)",  # Your name or organization
    author_email="jngai_@berkeley.edu",  # Optional: Your email address
    description="Tools for exploring and analyzing ASUC financial data.",  # Short description
    long_description=open("README.md").read(),  # Optionally include README.md as long description
    long_description_content_type="text/markdown",  # Specify README format
    url="https://github.com/yourusername/asuc-explore",  # Optional: Link to project repo
    packages=find_packages(),  # Automatically find all packages in the project
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.19.0",
        "pytest>=6.0.0",  # Add any other dependencies
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version
)