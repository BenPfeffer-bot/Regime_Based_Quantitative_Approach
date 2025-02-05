"""
Technical Analysis Package

This package provides functionality for technical analysis of financial data,
including indicator calculations, visualization, and regime detection.
"""

from .models import TechnicalAnalysis
from .data import DataFetcher
from .visualization import TechnicalPlotter

__all__ = ["TechnicalAnalysis", "DataFetcher", "TechnicalPlotter"]
