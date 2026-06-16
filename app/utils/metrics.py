import numpy as np
from typing import Optional


def calculate_volatility(prices: list[float], periods: int = 21) -> float:
    if len(prices) < periods + 1:
        return 0.0
    returns = np.diff(prices[-periods-1:]) / np.array(prices[-periods-1:-1])
    return float(np.std(returns) * np.sqrt(252) * 100)


def calculate_momentum(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 0.0
    return ((prices[-1] - prices[-period-1]) / prices[-period-1]) * 100


def calculate_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.05) -> float:
    if len(returns) < 2:
        return 0.0
    excess = np.mean(returns) - risk_free_rate / 252
    std = np.std(returns)
    if std == 0:
        return 0.0
    return float(np.sqrt(252) * excess / std)
