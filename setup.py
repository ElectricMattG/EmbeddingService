from setuptools import find_packages, setup

version = open("version").read().strip()

setup_args = {
    "packages": find_packages(include=["vision"]),
    "version": version,
}

if __name__ == "__main__":
    setup(**setup_args)
