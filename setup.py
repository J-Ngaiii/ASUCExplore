from setuptools import setup, find_packages

setup(
    name="ASUCExplore",  # Package name
    version="1.0.0",  # Version of your package
    author="Jonathan Ngai (ASUC)",  # Your name or organization
    author_email="jngai_@berkeley.edu",  # Optional: Your email address
    description="Tools for exploring and analyzing ASUC financial data.",  # Short description
    long_description=open("README.md").read(),  # Optionally include README.md as long description
    long_description_content_type="text/markdown",  # Specify README format
    url="https://github.com/J-Ngaiii/asuc-explore",  # Optional: Link to project repo
    packages=find_packages("src"),  # Check src specifically
    package_dir={"": "src"}, 
    install_requires=[
        "numpy",      # numpy 1.26.4 is required for Python 3.12 compatibility
        "pandas",
        "matplotlib",
        "seaborn",
        "spacy",
        "scikit-learn",  # Ensure scikit-learn 1.5.1 is used (note: incompatible with Python 3.12)
        "rapidfuzz"
    ],
    python_requires='>=3.11,<3.12', # generally use python 3.11.8
)