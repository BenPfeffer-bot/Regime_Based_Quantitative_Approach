# Regime Based Quantitative Approach

This package provides tools and functionality for regime-based quantitative trading strategies, with a focus on technical analysis and market regime detection.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Regime_Based_Quantitative_Approach.git
cd Regime_Based_Quantitative_Approach
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Refresh Market Data
To fetch and process the latest market data:
```bash
python src/scripts/refresh_data.py
```

### Run Technical Analysis Pipeline
To run the complete technical analysis pipeline:
```bash
python src/analysis/_pipeline.py
```

## Project Structure

```
.
├── data/                  # Data directory (created automatically)
│   ├── cache/            # Raw data cache
│   ├── figures/          # Generated plots
│   └── with_indicators/  # Processed data with indicators
├── src/
│   ├── analysis/
│   │   ├── technical/
│   │   │   ├── __init__.py
│   │   │   ├── data.py        # Data fetching and processing
│   │   │   ├── indicators.py  # Technical indicator calculations
│   │   │   ├── models.py      # Main analysis classes
│   │   │   └── visualization.py # Plotting functionality
│   │   └── _pipeline.py       # Main analysis pipeline
│   ├── scripts/
│   │   └── refresh_data.py    # Data refresh script
│   └── config.py          # Project configuration and paths
└── setup.py              # Package installation configuration
```

## Data Organization

The package automatically creates and manages the following data directories:

- `data/cache/`: Stores raw market data downloaded from Yahoo Finance
- `data/figures/`: Contains generated technical analysis plots
- `data/with_indicators/`: Stores processed data with technical indicators

## Dependencies

- Python >= 3.7
- numpy
- pandas
- matplotlib
- seaborn
- yfinance 