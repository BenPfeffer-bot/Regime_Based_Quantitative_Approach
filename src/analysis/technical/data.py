"""
Data Module

This module handles data fetching and processing for technical analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import CACHE_DIR, INDICATORS_DIR


class DataFetcher:
    """Class for fetching and processing financial data."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        tickers: Optional[List[str]] = None,
    ) -> None:
        """Initialize the DataFetcher with directories and tickers."""
        self.cache_dir: Path = cache_dir if cache_dir else CACHE_DIR
        self.output_dir: Path = output_dir if output_dir else INDICATORS_DIR
        self.tickers = tickers or [
            "RMS_PA", "VOW3_DE", "ASML_AS", "SAN_MC", "BNP_PA",
            "ITX_MC", "MC_PA", "RACE_MI", "ENI_MI", "MUV2_DE",
            "ENEL_MI", "BAS_DE", "DG_PA", "SU_PA", "BMW_DE",
            "NOKIA_HE", "BBVA_MC", "AI_PA", "SGO_PA", "ADS_DE",
            "BAYN_DE", "DHL_DE", "OR_PA", "CS_PA", "BN_PA",
            "INGA_AS", "ADYEN_AS", "TTE_PA", "IFX_DE", "RI_PA",
            "WKL_AS", "DB1_DE", "SIE_DE", "AIR_PA", "PRX_AS",
            "ABI_BR", "STLAM_MI", "NDA-FI_HE", "MBG_DE", "ISP_MI",
            "IBE_MC", "KER_PA", "SAP_DE", "AD_AS", "ALV_DE",
            "UCG_MI", "EL_PA", "SAN_PA", "BBVA_MC", "ENEL_MI"
        ]

        if len(self.tickers) != 50:
            logging.warning(
                f"Expected 50 tickers for EUROSTOXX50, but got {len(self.tickers)}."
            )

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Load pre-processed data from indicators directory."""
        data: Dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            logging.info(f"Loading data for {ticker}...")
            indicator_file = self.output_dir / f"{ticker}_indicators.parquet"
            
            if indicator_file.exists():
                df = pd.read_parquet(indicator_file)
                data[ticker] = df
                logging.info(f"Loaded {ticker} data from {indicator_file}")
            else:
                logging.warning(f"No data file found for {ticker}")
        
        return data

    def process_all_stocks(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Process all stocks with technical indicators."""
        return dfs  # Data is already processed

    def process_and_save_stocks(self) -> None:
        """Process stocks from cache and save with indicators."""
        logging.info("Data is already processed in the indicators directory.")
        return
