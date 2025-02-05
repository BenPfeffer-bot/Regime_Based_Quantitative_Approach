#!/usr/bin/env python3
"""Script to clean cache data and refresh market data."""

import sys
import shutil
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.technical.data import DataFetcher


def clean_cache():
    """Clean the cache directory containing market data."""
    cache_dir = project_root / "data"
    cache_subdir = project_root / "data" / "cache"
    
    # Create directories if they don't exist
    cache_dir.mkdir(exist_ok=True)
    cache_subdir.mkdir(exist_ok=True)
    
    # Clean only the cache subdirectory
    for file in cache_subdir.glob("*"):
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)
    print(f"Cleaned all files in cache directory: {cache_subdir}")


def main():
    """Main function to clean cache and refresh market data."""
    # First clean the cache
    clean_cache()

    # Then fetch fresh data
    fetcher = DataFetcher()
    data = fetcher.fetch_data()

    # Process and save the data
    processed_data = fetcher.process_all_stocks(data)
    fetcher.process_and_save_stocks()


if __name__ == "__main__":
    main()
