"""
Configuration Module

This module handles project-wide configuration and paths.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
FIGURES_DIR = DATA_DIR / "figures"
INDICATORS_DIR = DATA_DIR / "with_indicators"

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, FIGURES_DIR, INDICATORS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 