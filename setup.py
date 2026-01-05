#!/usr/bin/env python3
"""
ArionXiv Setup Script for PyPI Distribution
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Core requirements (essential for basic functionality)
install_requires = [
    "pymongo>=4.9,<5.0",
    "motor>=3.0.0",
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.20.0",
    "pydantic[email]>=2.0.0",
    "PyJWT>=2.8.0",
    "requests>=2.31.0",
    "aiohttp>=3.8.0",
    "python-dotenv>=1.0.0",
    "bcrypt>=4.0.0",
    "python-multipart>=0.0.6",
    "PyPDF2>=3.0.0",
    "pymupdf>=1.23.0",
    "groq>=0.4.0",
    "google-generativeai>=0.3.0",
    "pydantic-settings>=2.0.0",
    "structlog>=23.0.0",
    "arxiv>=2.0.0",
    "rich>=13.0.0",
    "click>=8.0.0",
    "colorama>=0.4.6",
    "numpy>=1.24.0,<2.0.0",
    "APScheduler>=3.10.0",
]

# Optional dependencies
extras_require = {
    'advanced-pdf': [
        'pdfplumber>=0.10.0',
        'pytesseract>=0.3.10',
        'tabula-py>=2.9.0',
        'Pillow>=9.0.0',
        'opencv-python>=4.8.0',
    ],
    'ml': [
        'sentence-transformers>=3.0.0',
        'torch>=2.0.0',
        'transformers>=4.20.0',
    ],
    'enhanced-ui': [
        'inquirer>=3.1.0',
        'tabulate>=0.9.0',
        'prompt-toolkit>=3.0.0',
        'alive-progress>=3.1.0',
    ],
    'dev': [
        'pytest>=7.0.0',
        'pytest-asyncio>=0.21.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
        'mypy>=1.0.0',
        'pre-commit>=3.0.0',
    ],
}

# All optional dependencies
extras_require['all'] = [
    dep for deps in extras_require.values() for dep in deps
]

setup(
    name="arionxiv",
    version="1.0.27",
    author="Arion Das",
    author_email="ariondasad@gmail.com",
    description="AI-Powered Research Paper Analysis and Management System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArionDas/ArionXiv",
    project_urls={
        "Bug Tracker": "https://github.com/ArionDas/ArionXiv/issues",
        "Documentation": "https://github.com/ArionDas/ArionXiv#readme",
        "Source Code": "https://github.com/ArionDas/ArionXiv",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Database :: Front-Ends",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: Web Environment",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "arionxiv=arionxiv.main:main",
            "arionxiv-server=arionxiv.server_main:main",
            "arionxiv-scheduler=arionxiv.scheduler_daemon:main",
        ],
    },
    include_package_data=True,
    package_data={
        "arionxiv": [
            "*.json",
            "*.txt",
            "*.md",
            "templates/*",
            "static/*",
            "config/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "arxiv", "research", "papers", "ai", "machine-learning", 
        "nlp", "analysis", "academic", "scientific", "publication",
        "pdf", "text-extraction", "mongodb", "cli", "api"
    ],
    license="MIT",
    platforms=["any"],
)