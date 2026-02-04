"""
Setup configuration for Bidder Collusion Detection System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="collusion-detection",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine learning system for detecting bidder collusion in coal e-auctions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CollusionDetection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "collusion-detection=main_pipeline:main",
        ],
    },
    include_package_data=True,
    keywords="collusion detection machine-learning anomaly-detection network-analysis auction",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/CollusionDetection/issues",
        "Source": "https://github.com/yourusername/CollusionDetection",
        "Documentation": "https://github.com/yourusername/CollusionDetection#readme",
    },
)
