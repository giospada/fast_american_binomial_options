"""
Fast American Binomial Options Library

A high-performance library for pricing American-style options using
the binomial method with GPU acceleration via PyCUDA.
"""

from .data_models import AmericanOptions, AmericanOption
from .calculators import OptionPriceCalculatorInterface
from .calculators.cpu_naive import VanillaAmericanBinomialCpuNaive
from .calculators.gpu_naive import VanillaAmericanBinomialGpuNaive
from .calculators.gpu_optimized import VanillaAmericanBinomialGpuOptimized

__all__ = [
    "AmericanOption",
    "AmericanOptions",
    "OptionPriceCalculatorInterface",
    "VanillaAmericanBinomialCpuNaive",
    "VanillaAmericanBinomialGpuNaive",
    "VanillaAmericanBinomialGpuOptimized",
]
