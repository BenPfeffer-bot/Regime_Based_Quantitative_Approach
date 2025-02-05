#!/usr/bin/env python3
"""
Technical Analysis Pipeline

This module serves as the main entry point for running technical analysis
on EUROSTOXX50 stocks. It orchestrates the process of fetching data,
computing indicators, and generating visualizations.
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.technical import TechnicalAnalysis
from src.config import CACHE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """Main function to run the technical analysis pipeline."""
    analyzer = TechnicalAnalysis()
    
    # Check if we need to fetch data first
    if not any(CACHE_DIR.glob("*.parquet")):
        logging.info("No cached data found. Fetching market data...")
        data = analyzer.data_fetcher.fetch_data()
        logging.info("Data fetching complete.")
    
    # Process and plot
    analyzer.process_and_plot_all()


if __name__ == "__main__":
    main()
