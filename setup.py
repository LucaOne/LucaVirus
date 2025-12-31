from setuptools import setup, find_packages

setup(
    name="lucavirus",
    version="1.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=1.13.1",
        "transformers>=4.26.0",
    ],
    author="YongHe",
    description="LucaVirus: LucaVirus Model compatible with HuggingFace Transformers",
)

