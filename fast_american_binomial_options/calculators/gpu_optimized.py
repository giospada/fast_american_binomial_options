"""
GPU-optimized option pricing calculator with caching.
"""

import numpy as np
import math
from enum import Enum
from typing import override, List, Tuple

from pycuda.compiler import SourceModule
from pycuda.driver import mem_alloc, memcpy_htod, Stream
import pycuda.gpuarray as gpuarray

from ..data_models import AmericanOptions
from ..calculators import OptionPriceCalculatorInterface


class OperationType(Enum):
    """Supported operation types for GPU computation."""

    FLOAT = "float"
    DOUBLE = "double"
    DS_FLOAT = "ds_float"

    def get_dtype_numpy(self):
        """Get NumPy dtype for operation type."""
        if self == OperationType.FLOAT:
            return np.float32
        elif self == OperationType.DOUBLE:
            return np.float64
        elif self == OperationType.DS_FLOAT:
            return np.dtype([("x", np.float32), ("y", np.float32)])
        else:
            raise ValueError(f"Unsupported operation type: {self}")

    def get_dtype_cuda(self):
        """Get CUDA type string."""
        if self == OperationType.FLOAT:
            return "float"
        elif self == OperationType.DOUBLE:
            return "double"
        elif self == OperationType.DS_FLOAT:
            return "ds_float"
        else:
            raise ValueError(f"Unsupported operation type: {self}")

    def get_to_type(self):
        """Get type conversion string."""
        if self == OperationType.FLOAT:
            return "(float)"
        elif self == OperationType.DOUBLE:
            return "(double)"
        elif self == OperationType.DS_FLOAT:
            return "double_to_ds"
        else:
            raise ValueError(f"Unsupported operation type: {self}")

    def get_max_function(self):
        """Get max function name."""
        if self == OperationType.FLOAT:
            return "float_max"
        elif self == OperationType.DOUBLE:
            return "double_max"
        elif self == OperationType.DS_FLOAT:
            return "ds_max"
        else:
            raise ValueError(f"Unsupported operation type: {self}")

    def get_add_multiply_function(self):
        """Get add-multiply function name."""
        if self == OperationType.FLOAT:
            return "float_add_mult"
        elif self == OperationType.DOUBLE:
            return "double_add_mult"
        elif self == OperationType.DS_FLOAT:
            return "ds_add_two_mults_streamlined"
        else:
            raise ValueError(f"Unsupported operation type: {self}")


LIB_CODE_TEMPLATE = """
struct OptionParameters {
    double S;
    double K;
    int n;
    double u;
    double up;
    double down;
    int sign;
};


// DS_FLOAT type and operations
typedef float2 ds_float;

__device__ inline bool ds_greater(ds_float a, ds_float b) {
    if (a.x > b.x) return true;
    if (a.x < b.x) return false;
    return a.y > b.y;
}

__device__ inline ds_float ds_max(ds_float a, ds_float b) {
    return ds_greater(a, b) ? a : b;
}

__device__ __host__ inline ds_float double_to_ds(double x) {
    float hi = (float)x;
    float lo = (float)(x - (double)hi);
    return make_float2(hi, lo);
}

__device__ __host__ inline ds_float quickTwoSum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return make_float2(s, e);
}

__device__ inline ds_float ds_add_two_mults_streamlined(
    ds_float my_up, ds_float up_val, 
    ds_float my_down, ds_float val) 
{
    float p_AB = my_up.x * up_val.x;
    float err_AB_h = __fmaf_rn(my_up.x, up_val.x, -p_AB);
    
    float p_CD = my_down.x * val.x;
    float err_CD_h = __fmaf_rn(my_down.x, val.x, -p_CD);

    ds_float s_high = quickTwoSum(p_AB, p_CD);

    float low_sum = s_high.y;
    low_sum += err_AB_h;
    low_sum += err_CD_h;
    
    low_sum = __fmaf_rn(my_up.x, up_val.y, low_sum); 
    low_sum = __fmaf_rn(my_up.y, up_val.x, low_sum);
    low_sum = __fmaf_rn(my_down.x, val.y, low_sum);
    low_sum = __fmaf_rn(my_down.y, val.x, low_sum);

    ds_float final_result = quickTwoSum(s_high.x, low_sum); 
    return final_result;
}


// DOUBLE Operations
__device__ __forceinline__ double double_max(double a, double b) {
    return fmax(a, b);
}

__device__ __forceinline__ double double_add_mult(double a, double b, double c, double d) {
    return fma(a, b, c * d);
}

// FLOAT Operations

__device__ __forceinline__ float float_max(float a, float b) {
    return fmaxf(a, b);
}

__device__ __forceinline__ float float_add_mult(float a, float b, float c, float d) {
    return fmaf(a, b, c * d);
}


// Power function for integer exponents for GPU
__device__ __forceinline__ double powi(double a, int e) {
    double r = 1.0;
    int neg = (e < 0);
    unsigned int k = neg ? (unsigned int)(-e) : (unsigned int)e;

    while (k) {
        if (k & 1u) r *= a;
        a *= a;
        k >>= 1;
    }
    return neg ? (1.0 / r) : r;
}

__global__ void fill_st_buffers_kernel_batch(
    DTYPE* __restrict__ cached_even_layer, DTYPE* __restrict__ cached_odd_layer,
    const OptionParameters* __restrict__ option_params, const int max_num_steps,
    DTYPE* __restrict__ layer_values_out, DTYPE* __restrict__ layer_values_out_next) {
    \"\"\"
    Compute option values at maturity and fill ST buffers for the first layer.
    \"\"\"
    const int option_idx = blockIdx.y;
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    const OptionParameters p = option_params[option_idx];
    const int num_steps = p.n;
    if (threadId > num_steps ) return;

    const double u_pow = powi(p.u, 2 * threadId - num_steps);
    const int buffer_index = option_idx * (max_num_steps+1) + threadId;

    double val0 = fmax(p.sign * fma(p.S, u_pow, -p.K), 0.0);
    double val1 = fmax(p.sign * fma(p.S, u_pow * p.u, -p.K), 0.0);
    
    layer_values_out_next[buffer_index] = layer_values_out[buffer_index] = TODTYPE(val0);

    if(num_steps % 2 == 0){
        double t = val0;
        val0 = val1;
        val1 = t;
    }

    cached_even_layer[buffer_index] = TODTYPE(val0);
    cached_odd_layer[buffer_index] = TODTYPE(val1);
}
"""

SOURCE_CODE_TEMPLATE = """
__global__ void compute_next_layers_kernel_batch_THREADS_PER_BLOCK_UNROLL_FACTOR(
    const DTYPE* __restrict__ layer_values_read, DTYPE* __restrict__ layer_values_write,
    const DTYPE* __restrict__ cached_even_layer, const DTYPE* __restrict__ cached_odd_layer,
    const OptionParameters* __restrict__ params, const int starting_layer,
    const int max_n) {
    const int option_idx = blockIdx.y;

    const OptionParameters p = params[option_idx];
    if(p.n < starting_layer - UNROLL_FACTOR) return;
    
    const int base_st_buffer = (max_n+1) * option_idx;

    __shared__ DTYPE layer_values_tile[2][THREADS_PER_BLOCK + 1];

    const int tile_stride = THREADS_PER_BLOCK - UNROLL_FACTOR;
    const int tile_base = tile_stride * blockIdx.x;
    const int node_id = tile_base + threadIdx.x;

    if (node_id > starting_layer) {return;}
    layer_values_tile[1][threadIdx.x] = layer_values_tile[0][threadIdx.x] = layer_values_read[base_st_buffer + node_id];

    __syncthreads();
    const DTYPE up = TODTYPE(p.up), down = TODTYPE(p.down);

#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        if(p.n < starting_layer - i){ continue;}
        int read_idx = i % 2;
        int write_idx = (i + 1) % 2;

        int current_level_read = starting_layer - i;

        const DTYPE* staging_bank = (current_level_read) % 2 ? cached_odd_layer : cached_even_layer;

        DTYPE hold = ADD_MULTIPLY(up, layer_values_tile[read_idx][threadIdx.x + 1], down, layer_values_tile[read_idx][threadIdx.x]);

        int st_index = node_id + (p.n - (current_level_read-1)) / 2;
        DTYPE exercise = staging_bank[base_st_buffer + st_index];
        layer_values_tile[write_idx][threadIdx.x] = MAX(hold, exercise);

        __syncthreads();
    }

    if (threadIdx.x < THREADS_PER_BLOCK - UNROLL_FACTOR) {
        layer_values_write[base_st_buffer + node_id] = layer_values_tile[UNROLL_FACTOR % 2][threadIdx.x];
    }
}
"""


def substitute_macro(
    input_code: str,
    unroll: int = 1,
    threads_per_block: int = 512,
    operation_type: OperationType = OperationType.DOUBLE,
) -> str:
    """Substitute macros in CUDA code template."""
    code = input_code.replace("THREADS_PER_BLOCK", str(threads_per_block))
    code = code.replace("UNROLL_FACTOR", str(unroll))
    code = code.replace("TODTYPE", operation_type.get_to_type())
    code = code.replace("DTYPE", operation_type.get_dtype_cuda())
    code = code.replace("MAX", operation_type.get_max_function())
    code = code.replace("ADD_MULTIPLY", operation_type.get_add_multiply_function())
    return code


class GPUOPTOptionPriceCalculator(OptionPriceCalculatorInterface):
    """GPU-optimized option price calculator with caching."""

    def __init__(
        self,
        operation_type: OperationType = OperationType.DOUBLE,
        uf_scheduler: List[Tuple[int, int]] = None,
        threads_per_block: int = 512,
    ):
        """
        Initialize GPU optimized calculator.

        Parameters
        ----------
        operation_type : OperationType
            Data type for calculations
        uf_scheduler : List[Tuple[int, int]]
            Unroll factor scheduler (layer, factor) pairs
        threads_per_block : int
            CUDA threads per block
        """
        if uf_scheduler is None:
            uf_scheduler = [(64, 64), (16, 16), (4, 4), (2, 2), (0, 1)]

        self.operation_type = operation_type
        self.threads_per_block = threads_per_block

        parsed_lib = substitute_macro(
            LIB_CODE_TEMPLATE,
            unroll=1,
            threads_per_block=threads_per_block,
            operation_type=operation_type,
        )

        parsed_code = "\n".join(
            substitute_macro(
                SOURCE_CODE_TEMPLATE,
                unroll=uf,
                threads_per_block=threads_per_block,
                operation_type=operation_type,
            )
            for uf in map(lambda x: x[1], uf_scheduler)
        )

        self.uf_scheduler = sorted(uf_scheduler, key=lambda x: x[0], reverse=True)

        self.bkdstprcmp_implementation = SourceModule(parsed_lib + "\n" + parsed_code)

        self.fill_st_buffers_kernel_batch = self.bkdstprcmp_implementation.get_function(
            "fill_st_buffers_kernel_batch"
        )
        self.compute_next_layers_kernel_batch = {
            uf: self.bkdstprcmp_implementation.get_function(
                f"compute_next_layers_kernel_batch_{threads_per_block}_{uf}"
            )
            for uf in map(lambda x: x[1], uf_scheduler)
        }

    @override
    def resolve_options(self, options: AmericanOptions) -> np.ndarray:
        """
        Calculate American option prices using GPU optimized method.

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

        d_option_values = gpuarray.zeros(
            (m, max_n + 1), dtype=self.operation_type.get_dtype_numpy()
        )
        d_option_values_next = gpuarray.zeros(
            (m, max_n + 1), dtype=self.operation_type.get_dtype_numpy()
        )
        d_cache_odd_layer = gpuarray.zeros(
            (m, max_n + 1), dtype=self.operation_type.get_dtype_numpy()
        )
        d_cache_even_layer = gpuarray.zeros(
            (m, max_n + 1), dtype=self.operation_type.get_dtype_numpy()
        )

        d_options = mem_alloc(gpu_params.nbytes)
        memcpy_htod(d_options, gpu_params)

        blocks_x = math.ceil(max_n / self.threads_per_block)
        blocks_y = m

        grid_first_layer = (blocks_x, blocks_y, 1)
        block_dim = (self.threads_per_block, 1, 1)

        stream = Stream()
        self.fill_st_buffers_kernel_batch(
            d_cache_even_layer.gpudata,
            d_cache_odd_layer.gpudata,
            d_options,
            np.int32(max_n),
            d_option_values.gpudata,
            d_option_values_next.gpudata,
            grid=grid_first_layer,
            block=block_dim,
            stream=stream,
        )

        def get_unroll_factor_for_layer(layer: int) -> int:
            for starting_layer, unroll_factor in self.uf_scheduler:
                if layer >= starting_layer and layer >= unroll_factor:
                    return unroll_factor
            return 1

        current_layer = int(max_n - 1)
        while current_layer >= 0:
            unroll_factor = get_unroll_factor_for_layer(current_layer)
            self.compute_next_layers_kernel_batch[unroll_factor](
                d_option_values.gpudata,
                d_option_values_next.gpudata,
                d_cache_even_layer.gpudata,
                d_cache_odd_layer.gpudata,
                d_options,
                np.int32(current_layer + 1),
                np.int32(max_n),
                grid=grid_first_layer,
                block=block_dim,
                stream=stream,
            )
            d_option_values, d_option_values_next = (
                d_option_values_next,
                d_option_values,
            )
            current_layer -= unroll_factor

        stream.synchronize()

        if self.operation_type == OperationType.DS_FLOAT:
            temp = d_option_values.get()[:, 0]
            temp = np.vectorize(
                lambda x: np.float64(x[0]) + np.float64(x[1]), otypes=[np.float64]
            )(temp)
            return temp

        return d_option_values.get()[:, 0]
