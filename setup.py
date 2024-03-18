import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hera_commissioning_tools",
    version="0.0.0",
    author="Dara Storer",
    author_email="darajstorer@gmail.com",
    description="Collection of HERA commissioning analysis tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HERA-Team/hera_commissioning_tools",
    license="MIT",
    packages=["hera_commissioning_tools"],
    install_requires=["numpy>=1.18", "matplotlib", "pyuvdata"],
)
