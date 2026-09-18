"""Wheel-Bau: das Paket enthaelt eine vorgebaute CUDA-Bibliothek (nwc/lib/), darum ein plattformspezifisches
Wheel (kein 'none-any'). Metadaten stehen in pyproject.toml."""
from setuptools import setup
from setuptools.dist import Distribution


class Binaer(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=Binaer)
