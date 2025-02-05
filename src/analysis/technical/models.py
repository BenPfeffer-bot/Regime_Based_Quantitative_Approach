"""
Models Module

This module contains the main TechnicalAnalysis class that orchestrates
the technical analysis process.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd

from src.config import FIGURES_DIR
from .data import DataFetcher
from .visualization import TechnicalPlotter


class TechnicalAnalysis:
    """Main class for performing technical analysis on financial data."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        tickers: Optional[List[str]] = None,
    ) -> None:
        """Initialize TechnicalAnalysis with data fetcher."""
        self.data_fetcher = DataFetcher(cache_dir, output_dir, tickers)

    def plot_technical_analysis(
        self, df: pd.DataFrame, output_path: Optional[Path] = None
    ) -> None:
        """Create and save technical analysis plots."""
        if output_path is None:
            output_path = FIGURES_DIR
        output_path.mkdir(parents=True, exist_ok=True)

        plotter = TechnicalPlotter(df)
        plotter.create_and_save_plot(output_path)

    def process_and_plot_all(self) -> None:
        """Process all stocks and create plots."""
        # Process stocks
        self.data_fetcher.process_and_save_stocks()

        # Create plots
        run_dir = FIGURES_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        for ticker in self.data_fetcher.tickers:
            sample_file = (
                self.data_fetcher.output_dir / f"{ticker.replace('.', '_')}_indicators.parquet"
            )
            if sample_file.exists():
                df = pd.read_parquet(sample_file)
                df.attrs["ticker"] = ticker
                self.plot_technical_analysis(df, output_path=run_dir)
            else:
                logging.warning(
                    f"Sample file {sample_file} not found. Skipping plotting for {ticker}."
                )
