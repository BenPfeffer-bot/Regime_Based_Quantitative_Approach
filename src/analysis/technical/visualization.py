"""
Visualization Module

This module provides plotting functionality for technical analysis visualization.
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set a global Seaborn style for professional visuals
sns.set_theme(style="whitegrid", context="talk")


class TechnicalPlotter:
    """Class for creating technical analysis plots."""

    def __init__(self, df):
        """Initialize the plotter with a DataFrame containing technical indicators."""
        self.df = df

    def _create_keltner_channel_plot(
        self, ax: plt.Axes, color: str, level: int, linewidth: float = 2.0
    ) -> None:
        """Create Keltner Channel plot for a specific level."""
        ax.plot(
            self.df.index,
            self.df[f"KC_upper{level}"],
            linestyle="--",
            color=color,
            linewidth=linewidth,
            label=f"KC Upper {level}",
        )
        ax.plot(
            self.df.index,
            self.df[f"KC_lower{level}"],
            linestyle="--",
            color=color,
            linewidth=linewidth,
            label=f"KC Lower {level}",
        )
        ax.fill_between(
            self.df.index,
            self.df[f"KC_upper{level}"],
            self.df[f"KC_lower{level}"],
            color=color,
            alpha=0.1,
        )

    def plot_price_and_keltner(self, ax: plt.Axes) -> None:
        """Plot price and Keltner channels."""
        ax.plot(
            self.df.index,
            self.df["Close"],
            label="Close Price",
            color="black",
            linewidth=1.5,
        )
        colors = ["#d62728", "#1f77b4", "#2ca02c"]  # professional red, blue, green
        for level, color in enumerate(colors, 1):
            self._create_keltner_channel_plot(ax, color, level)
        ax.set_title("Price & Keltner Channels", fontsize=18)
        ax.set_ylabel("Price", fontsize=14)
        ax.legend(fontsize=12, loc="upper left")
        ax.tick_params(axis="both", labelsize=12)

    def plot_macd(self, ax: plt.Axes) -> None:
        """Plot MACD indicator."""
        ax.plot(
            self.df.index,
            self.df["MACD_line"],
            label="MACD",
            color="#1f77b4",
            linewidth=1.5,
        )
        ax.plot(
            self.df.index,
            self.df["MACD_signal"],
            label="Signal",
            color="#ff7f0e",
            linewidth=1.5,
        )
        ax.bar(
            self.df.index,
            self.df["MACD_histogram"],
            label="Histogram",
            color="#2ca02c",
            alpha=0.4,
        )
        ax.set_title("MACD", fontsize=18)
        ax.legend(fontsize=12, loc="upper left")
        ax.tick_params(axis="both", labelsize=12)

    def plot_rsi(self, ax: plt.Axes) -> None:
        """Plot RSI indicator."""
        ax.plot(
            self.df.index, self.df["RSI"], label="RSI", color="#9467bd", linewidth=1.5
        )
        ax.axhline(
            y=70, color="red", linestyle="--", linewidth=1.5, label="Overbought (70)"
        )
        ax.axhline(
            y=30, color="green", linestyle="--", linewidth=1.5, label="Oversold (30)"
        )
        ax.set_title("RSI", fontsize=18)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=12, loc="upper left")
        ax.set_ylabel("RSI", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

    def plot_moving_averages(self, ax: plt.Axes) -> None:
        """Plot moving averages and market regime."""
        ax.plot(
            self.df.index,
            self.df["SMA_50"],
            label="50-day SMA",
            color="#8c564b",
            linewidth=1.5,
        )
        ax.plot(
            self.df.index,
            self.df["SMA_200"],
            label="200-day SMA",
            color="#e377c2",
            linewidth=1.5,
        )
        ax.plot(
            self.df.index,
            self.df["EMA_50"],
            label="50-day EMA",
            color="#7f7f7f",
            linewidth=1.5,
        )
        ax.plot(
            self.df.index,
            self.df["EMA_200"],
            label="200-day EMA",
            color="#bcbd22",
            linewidth=1.5,
        )

        # Highlight regime blocks
        for regime in self.df["Regime"].unique():
            regime_dates = self.df[self.df["Regime"] == regime].index
            if not regime_dates.empty:
                ax.axvspan(
                    regime_dates[0],
                    regime_dates[-1],
                    color="limegreen" if regime == "Trending" else "orange",
                    alpha=0.1,
                    label=regime,
                )
        ax.set_title("Moving Averages & Market Regime", fontsize=18)
        ax.set_ylabel("Price", fontsize=14)
        ax.legend(fontsize=12, loc="upper left")
        ax.tick_params(axis="both", labelsize=12)

    def plot_atr_variance(self, ax: plt.Axes) -> None:
        """Plot ATR and variance."""
        ax.plot(
            self.df.index, self.df["ATR"], label="ATR", color="#17becf", linewidth=1.5
        )
        ax.set_ylabel("ATR", fontsize=14, color="#17becf")
        ax2 = ax.twinx()
        variance = self.df["Close"].rolling(window=20).var()
        ax2.plot(
            self.df.index,
            variance,
            label="20-day Variance",
            color="#d62728",
            linewidth=1.5,
        )
        ax2.set_ylabel("Variance", fontsize=14, color="#d62728")
        ax.set_title("ATR & Rolling Variance", fontsize=18)
        ax.legend(fontsize=12, loc="upper left")
        ax2.legend(fontsize=12, loc="upper right")
        ax.tick_params(axis="both", labelsize=12)
        ax2.tick_params(axis="both", labelsize=12)

    def create_and_save_plot(self, output_path: Path) -> None:
        """Create and save the complete technical analysis plot."""
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(5, 1, height_ratios=[2, 1, 1, 1, 1], hspace=0.4)
        axes = [fig.add_subplot(gs[i]) for i in range(5)]

        self.plot_price_and_keltner(axes[0])
        self.plot_macd(axes[1])
        self.plot_rsi(axes[2])
        self.plot_moving_averages(axes[3])
        self.plot_atr_variance(axes[4])

        fig.suptitle("Advanced Technical Analysis Dashboard", fontsize=22, y=0.96)
        ticker = self.df.attrs.get("ticker", "stock")
        output_file = output_path / f"{ticker}_advanced_technical_indicators.png"
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logging.info(f"Saved advanced technical analysis plot to {output_file}")
