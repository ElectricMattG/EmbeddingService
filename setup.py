from setuptools import find_packages, setup

version = open("version").read().strip()

setup_args = {
    "name": 'embedding service',
    "version": version,
    "description": 'a service that offers embeddings',
    "author": 'Matt Gerhold',
    "author_email": '',
    "python_requires": '>=3.12',  # Specify the minimum Python version required
    "packages": find_packages(["src*"]),  # Automatically find all packages in the project
}

if __name__ == "__main__":
    setup(**setup_args)
