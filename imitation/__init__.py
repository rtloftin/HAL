"""
This package defines the different imitation learning agents that we will be
evaluating, along with common components which they will rely on.
"""

from .dataset import Dataset
from .cloning import manager as cloning
from .gail_ppo import manager as gail_ppo

__all__ = []
