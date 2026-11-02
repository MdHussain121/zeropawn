"""Setuptools shim for ZeroPawn's binary-bearing wheel (audit S6-01).

All project metadata lives in ``pyproject.toml``. This file exists solely to
declare a :class:`BinaryDistribution` whose ``has_ext_modules()`` returns
True: the wheel carries the staged native extension
(``python/_native/zeropawn_cpp*.pyd``, copied from ``build/cpp/`` by
``scripts/build_wheel.py``), and without this marker bdist_wheel would tag
the wheel ``py3-none-any`` even though it is platform-specific. With it, the
wheel is built with ``Root-Is-Purelib: false`` and a ``<platform>`` tag
(``win_amd64`` on Windows).

Build the wheel with ``python scripts/build_wheel.py`` (stages the native
extension and data files, redirects setuptools' intermediate build dir away
from the ``build/`` junction, and invokes PEP 517).
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """A distribution that always reports containing extension modules."""

    def has_ext_modules(self) -> bool:  # noqa: D102 - setuptools contract
        return True


setup(distclass=BinaryDistribution)
