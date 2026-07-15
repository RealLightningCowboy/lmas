"""Public GLM reader API for LMAS.

This module is intentionally small so research scripts can use the fast native
reader without importing future GUI overlay code.
"""

from .overlays.satellite.glm import *  # noqa: F401,F403
