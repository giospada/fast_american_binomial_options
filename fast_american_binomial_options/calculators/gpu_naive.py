"""
GPU-based naive binomial option pricing calculator using PyCUDA.
"""

import numpy as np
from typing import override
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context
from pycuda.driver import mem_alloc, memcpy_htod, Stream
import pycuda.gpuarray as gpuarray

from ..data_models import AmericanOptions
from ..calculators import OptionPriceCalculatorInterface


naive_implementation = SourceModule("""
struct OptionParameters {
    double S;
    double K;
    int n;
    double u;
    double up;
    double down;
    int sign;
};

__global__ void naive_compute_first_layer_kernel_batch(
    double* d_option_values,
    const OptionParameters* d_options,
    const int max_n
) {
    int option_idx = blockIdx.y;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int n = d_options[option_idx].n;

    if (thread_id > n) return;

    double S = d_options[option_idx].S;
    double u = d_options[option_idx].u;
    double K = d_options[option_idx].K;
    int sign = d_options[option_idx].sign;

    double ST = S * pow(u, 2 * thread_id - n);
    int idx = option_idx * (max_n + 1) + thread_id;
    d_option_values[idx] = max(0.0, sign * (ST - K));
}

__global__ void naive_compute_next_layer_kernel_batch(
    double* d_option_values,
    double* d_option_values_next,
    const OptionParameters* d_options,
    const int max_n,
    const int level
) {
    int option_idx = blockIdx.y;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int n = d_options[option_idx].n;
    int idx = option_idx * (max_n + 1) + thread_id;

    if (level >= n) {
        if (thread_id <= n) {
            d_option_values_next[idx] = d_option_values[idx];
        }
        return;
    }

    if (thread_id > level) return;

    double S = d_options[option_idx].S;
    double K = d_options[option_idx].K;
    double up = d_options[option_idx].up;
    double down = d_options[option_idx].down;
    double u = d_options[option_idx].u;
    int sign = d_options[option_idx].sign;

    double ST = S * pow(u, 2.0 * thread_id - level);
    double hold = up * d_option_values[idx + 1] + down * d_option_values[idx];
    double exercise = max(sign * (ST - K), 0.0);
    d_option_values_next[idx] = max(hold, exercise);
}
""")


class GPUNAIVEOptionPriceCalculator(OptionPriceCalculatorInterface):
    """GPU-based naive American option price calculator."""

    def __init__(self):
        """Initialize GPU calculator."""
        if not drv.Context.get_current():
            make_default_context()

        self.naive_compute_first_layer_kernel_batch = naive_implementation.get_function(
            "naive_compute_first_layer_kernel_batch"
        )
        self.naive_compute_next_layer_kernel_batch = naive_implementation.get_function(
            "naive_compute_next_layer_kernel_batch"
        )

    @override
    def resolve_options(self, options: AmericanOptions) -> np.ndarray:
        """
        Calculate American option prices using GPU naive binomial method.

        Parameters
        ----------
        options : AmericanOptions
            Batch of American options to price

        Returns
        -------
        np.ndarray
            Array of option prices
        """
        gpu_params = options.to_gpu_parameters()

        max_n = np.max(options.n)
        m = len(options.S)

        # Allocate GPU memory
        d_option_values = gpuarray.zeros((m, max_n + 1), dtype=np.float64)
        d_option_values_next = gpuarray.zeros((m, max_n + 1), dtype=np.float64)

        d_options = mem_alloc(gpu_params.nbytes)
        memcpy_htod(d_options, gpu_params)

        stream = Stream()
        num_threads_per_block = 256

        blocks_x = (max_n + num_threads_per_block) // num_threads_per_block
        blocks_y = m

        grid_first_layer = (int(blocks_x), blocks_y, 1)
        block_dim = (num_threads_per_block, 1, 1)

        self.naive_compute_first_layer_kernel_batch(
            d_option_values.gpudata,
            d_options,
            np.int32(max_n),
            grid=grid_first_layer,
            block=block_dim,
            stream=stream,
        )

        for level in range(max_n - 1, -1, -1):
            grid_next_layer = (
                (level + num_threads_per_block) // num_threads_per_block,
                blocks_y,
                1,
            )

            self.naive_compute_next_layer_kernel_batch(
                d_option_values.gpudata,
                d_option_values_next.gpudata,
                d_options,
                np.int32(max_n),
                np.int32(level),
                grid=grid_next_layer,
                block=block_dim,
                stream=stream,
            )

            d_option_values, d_option_values_next = (
                d_option_values_next,
                d_option_values,
            )

        pycuda.driver.Context.synchronize()
        option_prices = d_option_values.get()[:, 0]

        return option_prices
