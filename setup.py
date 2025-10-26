from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Try to read requirements-macos.txt first (for macOS), fallback to requirements.txt
try:
    with open("requirements-macos.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="expose-ai-ml",
    version="0.1.0",
    author="Joey Lamping",
    author_email="your.email@example.com",
    description="A flexible ML library supporting PyTorch and ONNX with Google Cloud integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaylamping/expose-ai-ml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "onnxruntime-gpu>=1.15.0",
        ],
    },
)


