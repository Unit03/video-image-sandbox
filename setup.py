import codecs
import os.path

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="video-image-sandbox",
    use_scm_version={
        "root": here,
        "write_to": os.path.join(here, "video/_version.py"),
    },
    description="Sandbox for various video/image-related stuff",
    long_description=long_description,
    packages=find_packages(where=here),
    entry_points={
        "console_scripts": [
            "video = video.cli:video",
        ],
    },
    install_requires=[
        "click",
        "jupyter",
        "matplotlib",
        "opencv-python",
        "tensorflow",
    ],
    # setup_requires=[
    #     "setuptools_scm>=1.10.1,<2",
    # ],
    extras_require={
        "tests": [
            "black",
            "flake8",
            "isort",
            "pytest",
            "unify",
        ],
    },
    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",

        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ],
)
