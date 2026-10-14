# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""Technical indicators from first principles.

These are not wrappers around TA-Lib. Moving averages use a rolling mean.
RSI uses Wilder's smoothing. MACD is the difference of two EMAs. Bollinger
Bands use the population standard deviation of the window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SMA_SHORT = 50
SMA_LONG = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0
TRADING_DAYS = 252


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()


def rsi_wilder(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder RSI. A flat window (no gain and no loss) is reported as 50."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    values = 100 - (100 / (1 + rs))
    values = values.mask(avg_loss == 0, 100.0)
    values = values.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return values


def macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": line,
            "macd_signal": signal_line,
            "macd_histogram": line - signal_line,
        }
    )


def bollinger_bands(
    close: pd.Series,
    window: int = BOLLINGER_WINDOW,
    n_std: float = BOLLINGER_STD,
) -> pd.DataFrame:
    # ddof=0 is the population standard deviation used by the original bands.
    middle = close.rolling(window=window, min_periods=window).mean()
    spread = close.rolling(window=window, min_periods=window).std(ddof=0)
    return pd.DataFrame(
        {
            "bollinger_middle": middle,
            "bollinger_upper": middle + n_std * spread,
            "bollinger_lower": middle - n_std * spread,
        }
    )


def attach_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["Close"].astype(float)
    out["sma_50"] = sma(close, SMA_SHORT)
    out["sma_200"] = sma(close, SMA_LONG)
    out["rsi_14"] = rsi_wilder(close, RSI_PERIOD)
    out = pd.concat([out, macd(close), bollinger_bands(close)], axis=1)
    return out


def annualized_volatility(close: pd.Series, window: int) -> float | None:
    """Standard deviation of log returns over `window`, scaled by sqrt(252)."""
    if window < 2 or len(close) < window + 1:
        return None
    log_returns = np.log(close.astype(float) / close.astype(float).shift(1))
    sample = log_returns.dropna().tail(window)
    if len(sample) < window:
        return None
    return float(sample.std(ddof=1) * np.sqrt(TRADING_DAYS))
