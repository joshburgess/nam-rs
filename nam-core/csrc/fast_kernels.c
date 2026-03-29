/**
 * Performance-critical DSP kernels compiled with -ffast-math.
 *
 * Coarse-grained functions that process entire operations in one call
 * to avoid FFI overhead. Each function does enough work to amortize
 * the cross-language call cost.
 *
 * Compiled via the `cc` crate with -O3 -ffast-math.
 */

#include <math.h>
#include <stddef.h>
#include <string.h>

/* ── Full Conv1d block processing (depthwise case) ──────────────────────
 * Equivalent to the entire Conv1d::process_block for depthwise weights.
 * Processes all kernel taps and adds bias in one call.
 *
 * weights: [kernel_size][ch] flattened as weights[k * ch + c]
 * tap_ptrs: array of pointers to ring buffer data for each tap
 * bias: [ch]
 * output: [num_frames * ch], written (not accumulated)
 */
void fast_conv1d_depthwise(
    float *restrict output,
    const float *const *restrict tap_ptrs,
    const float *restrict weights,
    const float *restrict bias,
    size_t ch,
    size_t kernel_size,
    size_t num_frames
) {
    size_t total = ch * num_frames;

    /* Initialize output with bias */
    for (size_t f = 0; f < num_frames; f++) {
        size_t off = f * ch;
        for (size_t c = 0; c < ch; c++) {
            output[off + c] = bias[c];
        }
    }

    /* Accumulate all taps */
    for (size_t k = 0; k < kernel_size; k++) {
        const float *tap = tap_ptrs[k];
        const float *w = weights + k * ch;
        for (size_t f = 0; f < num_frames; f++) {
            size_t off = f * ch;
            for (size_t c = 0; c < ch; c++) {
                output[off + c] += w[c] * tap[off + c];
            }
        }
    }
}

/* ── Full Conv1d block processing (general/small-matrix case) ───────────
 * For the small dot-product path (out_ch * in_ch < SGEMM threshold).
 * Processes all kernel taps with fused bias in one call.
 *
 * weights: [kernel_size][in_ch * out_ch] col-major per tap
 *          weights[k * (out_ch * in_ch) + i * out_ch + o]
 * tap_ptrs: array of pointers to input data for each tap
 */
void fast_conv1d_small_gemv(
    float *restrict output,
    const float *const *restrict tap_ptrs,
    const float *restrict weights,
    const float *restrict bias,
    size_t out_ch,
    size_t in_ch,
    size_t kernel_size,
    size_t num_frames
) {
    /* Initialize output with bias */
    for (size_t f = 0; f < num_frames; f++) {
        size_t off = f * out_ch;
        for (size_t o = 0; o < out_ch; o++) {
            output[off + o] = bias[o];
        }
    }

    /* Accumulate all taps */
    size_t w_stride = out_ch * in_ch;
    for (size_t k = 0; k < kernel_size; k++) {
        const float *tap = tap_ptrs[k];
        const float *w = weights + k * w_stride;
        for (size_t f = 0; f < num_frames; f++) {
            size_t in_off = f * in_ch;
            size_t out_off = f * out_ch;
            for (size_t o = 0; o < out_ch; o++) {
                float sum = 0.0f;
                for (size_t i = 0; i < in_ch; i++) {
                    sum += w[i * out_ch + o] * tap[in_off + i];
                }
                output[out_off + o] += sum;
            }
        }
    }
}

/* ── Fused z = conv + mixin, then activation ────────────────────────────
 * z_out[i] = activation(conv_out[i] + mixin_out[i])
 * Eliminates separate add pass and activation pass.
 */
void fast_add_activate(
    float *restrict z_out,
    const float *restrict conv_out,
    const float *restrict mixin_out,
    size_t len,
    int use_fast_tanh
) {
    if (use_fast_tanh) {
        for (size_t i = 0; i < len; i++) {
            float x = conv_out[i] + mixin_out[i];
            /* NAM fast_tanh polynomial */
            float ax = fabsf(x);
            float x2 = x * x;
            z_out[i] = (x * (2.45550750702956f + 2.45550750702956f * ax
                        + (0.893229853513558f + 0.821226666969744f * ax) * x2))
                     / (2.44506634652299f + (2.44506634652299f + x2)
                        * fabsf(x + 0.814642734961073f * x * ax));
        }
    } else {
        for (size_t i = 0; i < len; i++) {
            z_out[i] = tanhf(conv_out[i] + mixin_out[i]);
        }
    }
}

/* ── Tanh in-place ──────────────────────────────────────────────────────
 */
void fast_tanh_inplace(float *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        data[i] = tanhf(data[i]);
    }
}

/* ── Fast tanh polynomial in-place ──────────────────────────────────────
 */
/* ── Conv1x1 small GEMM with bias ───────────────────────────────────────
 * output[f*out_ch + o] = sum_i(w[i*out_ch+o] * input[f*in_stride+i]) + bias[o]
 * Handles the generic case and rank-1 (in_ch=1) efficiently.
 */
void fast_conv1x1_small(
    float *restrict output,
    const float *restrict weights,
    const float *restrict input,
    const float *restrict bias,  /* NULL if no bias */
    size_t out_ch,
    size_t in_ch,
    size_t input_stride,
    size_t num_frames
) {
    for (size_t f = 0; f < num_frames; f++) {
        size_t in_off = f * input_stride;
        size_t out_off = f * out_ch;
        for (size_t o = 0; o < out_ch; o++) {
            float sum = bias ? bias[o] : 0.0f;
            for (size_t i = 0; i < in_ch; i++) {
                sum += weights[i * out_ch + o] * input[in_off + i];
            }
            output[out_off + o] = sum;
        }
    }
}

/* ── FiLM scale+shift: output[i] = input[i] * scale[i] + shift[i] ──────
 * scale_shift layout: [scale_0..scale_{dim-1}, shift_0..shift_{dim-1}]
 * per frame, with ss_rows stride.
 */
void fast_film_scale_shift(
    float *restrict output,
    const float *restrict input,
    const float *restrict scale_shift,
    size_t dim,
    size_t input_stride,
    size_t output_stride,
    size_t ss_rows,
    size_t num_frames
) {
    for (size_t f = 0; f < num_frames; f++) {
        size_t in_off = f * input_stride;
        size_t out_off = f * output_stride;
        size_t ss_off = f * ss_rows;
        for (size_t i = 0; i < dim; i++) {
            output[out_off + i] = input[in_off + i] * scale_shift[ss_off + i]
                                + scale_shift[ss_off + dim + i];
        }
    }
}

/* ── FiLM scale only: output[i] = input[i] * scale[i] ──────────────────
 */
void fast_film_scale(
    float *restrict output,
    const float *restrict input,
    const float *restrict scale,
    size_t dim,
    size_t input_stride,
    size_t output_stride,
    size_t ss_rows,
    size_t num_frames
) {
    for (size_t f = 0; f < num_frames; f++) {
        size_t in_off = f * input_stride;
        size_t out_off = f * output_stride;
        size_t ss_off = f * ss_rows;
        for (size_t i = 0; i < dim; i++) {
            output[out_off + i] = input[in_off + i] * scale[ss_off + i];
        }
    }
}

/* ── FiLM in-place scale+shift: data[i] = data[i] * scale[i] + shift[i]
 */
void fast_film_inplace_scale_shift(
    float *restrict data,
    const float *restrict scale_shift,
    size_t dim,
    size_t data_stride,
    size_t ss_rows,
    size_t num_frames
) {
    for (size_t f = 0; f < num_frames; f++) {
        size_t d_off = f * data_stride;
        size_t ss_off = f * ss_rows;
        for (size_t i = 0; i < dim; i++) {
            data[d_off + i] = data[d_off + i] * scale_shift[ss_off + i]
                            + scale_shift[ss_off + dim + i];
        }
    }
}

/* ── FiLM in-place scale only: data[i] = data[i] * scale[i] ────────────
 */
void fast_film_inplace_scale(
    float *restrict data,
    const float *restrict scale,
    size_t dim,
    size_t data_stride,
    size_t ss_rows,
    size_t num_frames
) {
    for (size_t f = 0; f < num_frames; f++) {
        size_t d_off = f * data_stride;
        size_t ss_off = f * ss_rows;
        for (size_t i = 0; i < dim; i++) {
            data[d_off + i] *= scale[ss_off + i];
        }
    }
}

void fast_tanh_poly_inplace(float *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        float x = data[i];
        float ax = fabsf(x);
        float x2 = x * x;
        data[i] = (x * (2.45550750702956f + 2.45550750702956f * ax
                    + (0.893229853513558f + 0.821226666969744f * ax) * x2))
                 / (2.44506634652299f + (2.44506634652299f + x2)
                    * fabsf(x + 0.814642734961073f * x * ax));
    }
}
