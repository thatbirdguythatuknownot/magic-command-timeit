from pathlib import Path
from setuptools import setup

HERE = Path(__file__).parent
README = (HERE/"README.md").read_text()
setup(
    name="magic-commands",
    version="0.0.1",
    description="IPython magic commands, now available in pure Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/thatbirdguythatuknownot/magic-commands",
    author="Crowthebird",
    author_email=,
    ,
