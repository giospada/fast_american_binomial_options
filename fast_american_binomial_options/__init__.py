"""
Fast American Binomial Options Library

A high-performance library for pricing American-style options using
the binomial method with GPU acceleration via PyCUDA.
"""

from .data_models import AmericanOptions, AmericanOption
from .calculators import OptionPriceCalculatorInterface
from .calculators.cpu_naive import CPUNAIVEOptionPriceCalculator
from .calculators.gpu_naive import GPUNAIVEOptionPriceCalculator
from .calculators.gpu_optimized import GPUOPTOptionPriceCalculator, OperationType

__all__ = [
    # Data Models
    "AmericanOption",
    "AmericanOptions",
    # Calculators
    "OptionPriceCalculatorInterface",
    "CPUNAIVEOptionPriceCalculator",
    "GPUNAIVEOptionPriceCalculator",
    "GPUOPTOptionPriceCalculator",
    "OperationType",
]
