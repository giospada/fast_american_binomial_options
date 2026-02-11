"""
Abstract interface for option price calculators.
"""

from abc import ABC, abstractmethod
import numpy as np
from ..data_models import AmericanOptions


class OptionPriceCalculatorInterface(ABC):
    """Abstract base class for American option price calculators."""

    @abstractmethod
    def resolve_options(self, options: AmericanOptions) -> np.ndarray:
        """
        Calculate prices for a batch of American options.

        Parameters
        ----------
        options : AmericanOptions
            Batch of American options to price

        Returns
        -------
        np.ndarray
            Array of option prices
        """
        pass
