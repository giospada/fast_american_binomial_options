"""
CPU-based naive binomial option pricing calculator.
"""

import numpy as np
from typing import override
from ..data_models import AmericanOptions
from ..calculators import OptionPriceCalculatorInterface


def vanilla_american_binomial_cpu_naive_py(S, K, T, r, sigma, q, n, option_type):
    """
    CPU-based American option pricing using naive binomial model.

    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized)
    q : float
        Dividend yield (annualized)
    n : int
        Number of steps in binomial tree
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Option price
    """
    deltaT = T / n
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1 / u
    discount_factor = np.exp(-r * deltaT)
    p = (np.exp((r - q) * deltaT) - d) / (u - d)

    # Risk-neutral probabilities * discount factor
    up_factor = p * discount_factor
    down_factor = (1 - p) * discount_factor

    # Determine sign for call (1) or put (-1)
    sign = 1 if option_type.lower() == "call" else -1

    # Initialize option values at maturity
    option_values = np.zeros(n + 1)
    for i in range(n + 1):
        ST = S * (u ** (2.0 * i - n))
        option_values[i] = max(sign * (ST - K), 0.0)

    # Iterate backwards through tree
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            ST = S * (u ** (2.0 * i - j))
            hold_value = (
                up_factor * option_values[i + 1] + down_factor * option_values[i]
            )
            exercise_value = max(sign * (ST - K), 0.0)
            option_values[i] = max(hold_value, exercise_value)

    return option_values[0]


class CPUNAIVEOptionPriceCalculator(OptionPriceCalculatorInterface):
    """CPU-based naive American option price calculator."""

    @override
    def resolve_options(self, options: AmericanOptions) -> np.ndarray:
        """
        Calculate American option prices using CPU naive binomial method.

        Parameters
        ----------
        options : AmericanOptions
            Batch of American options to price

        Returns
        -------
        np.ndarray
            Array of option prices
        """
        cpu_prices = []
        for i in range(len(options.S)):
            price = vanilla_american_binomial_cpu_naive_py(
                S=options.S[i],
                K=options.K[i],
                T=options.T[i],
                r=options.r[i],
                sigma=options.sigma[i],
                q=options.q[i],
                n=options.n[i],
                option_type=options.option_type[i],
            )
            cpu_prices.append(price)
        return np.array(cpu_prices)
