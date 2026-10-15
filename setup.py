"""
setup.py — Permite instalar la carpeta `src` como paquete de Python.

Con `pip install -e .` (modo editable) el notebook y los scripts pueden hacer
`from src.config import ...` desde cualquier carpeta, sin trucos con sys.path.
"""

from setuptools import find_packages, setup

setup(
    name="vuelos_retrasos",                     # nombre interno del paquete
    version="1.0.0",                            # versión de la entrega final
    description="Predicción de retrasos (>=15 min) en vuelos comerciales de EE.UU.",
    author="Carlos Bedoya",
    packages=find_packages(include=["src", "src.*"]),  # instala src y sus subcarpetas
    python_requires=">=3.10",
)
