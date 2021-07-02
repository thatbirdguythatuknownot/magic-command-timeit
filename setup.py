from pathlib import Path
from setuptools import setup

HERE = Path(__file__).parent
README = (HERE/"README.md").read_text()
setup(
    name="magic-commands",
    version="0.0.12.1",
    description="IPython magic commands, now available in pure Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/thatbirdguythatuknownot/magic-commands",
    download_url="https://github.com/thatbirdguythatuknownot/magic-commands/archive/refs/tags/v0.0.10.zip",
    author="Crowthebird",
    author_email="nohackingofkrowten@gmail.com",
    license="AGPL-3.0",
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3"
    ],
    packages=["magiccmds"],
    include_package_data=True
)
