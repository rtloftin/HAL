"""
This package defines the different imitation learning agents that we will be
evaluating, along with common components which they will rely on.
"""

from .dataset import generate as dataset
from .cloning import manager as cloning
from .gail_ppo import manager as gail_ppo

__all__ = []
