/**
 * Performance-critical DSP kernels compiled with optimization enabled.
 *
 * Coarse-grained functions that process entire operations in one call
 * to avoid FFI overhead. Each function does enough work to amortize
 * the cross-language call cost.
 *
 * Compiled via the `cc` crate with -O3.
 */

#include <math.h>
#include <stddef.h>
#include <string.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif
#if defined(__linux__) && defined(__GLIBC__) && defined(__x86_64__)
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#define NAM_GLIBC_VECTOR_TANH 1
#endif

#if defined(NAM_GLIBC_VECTOR_TANH)
typedef float nam_v4sf __attribute__((vector_size(16)));
typedef nam_v4sf (*nam_tanhf4_fn)(nam_v4sf);

static pthread_once_t nam_vector_math_once = PTHREAD_ONCE_INIT;
static _Atomic(nam_tanhf4_fn) nam_tanhf4 = NULL;

static void nam_resolve_vector_math(void) {
    void *handle = dlopen("libmvec.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle == NULL) {
        return;
    }
    /* Callbacks retain the function pointer, so libmvec stays loaded. */

    void *symbol4 = dlsym(handle, "_ZGVbN4v_tanhf");
    if (symbol4 != NULL) {
        nam_tanhf4_fn function4 = NULL;
        _Static_assert(sizeof(function4) == sizeof(symbol4), "function pointer size mismatch");
        memcpy(&function4, &symbol4, sizeof(function4));
        atomic_store_explicit(&nam_tanhf4, function4, memory_order_release);
    }
}

int fast_init_vector_math(void) {
    pthread_once(&nam_vector_math_once, nam_resolve_vector_math);
    return atomic_load_explicit(&nam_tanhf4, memory_order_acquire) != NULL;
}

__attribute__((target("sse2")))
static size_t nam_add_tanh_sse2(
    float *restrict output,
    const float *restrict left,
    const float *restrict right,
    size_t len,
    nam_tanhf4_fn tanh4
) {
    size_t offset = 0;
    for (; offset + 4 <= len; offset += 4) {
        nam_v4sf left_values;
        nam_v4sf right_values;
        memcpy(&left_values, left + offset, sizeof(left_values));
        memcpy(&right_values, right + offset, sizeof(right_values));
        nam_v4sf result = tanh4(left_values + right_values);
        memcpy(output + offset, &result, sizeof(result));
    }
    return offset;
}

#else
int fast_init_vector_math(void) {
    return 0;
}
#endif

/* ── Full Conv1d block processing (depthwise case) ──────────────────────
 * Equivalent to the entire Conv1d::process_block for depthwise weights.
 * Processes all kernel taps and adds bias in one call.
 *
 * weights: [kernel_size][ch] flattened as weights[k * ch + c]
 * input: ring-buffer storage
 * tap_offsets: element offsets into input for each tap
 * bias: [ch]
 * output: [num_frames * ch], written (not accumulated)
 */
void fast_conv1d_depthwise(
    float *restrict output,
    const float *restrict input,
    const size_t *restrict tap_offsets,
    const float *restrict weights,
    const float *restrict bias,
    size_t ch,
    size_t kernel_size,
    size_t num_frames
) {
    /* Accumulate taps before adding bias to match Eigen. */
    for (size_t f = 0; f < num_frames; f++) {
        size_t off = f * ch;
        for (size_t c = 0; c < ch; c++) {
            output[off + c] = 0.0f;
        }
    }

    /* Accumulate all taps */
    for (size_t k = 0; k < kernel_size; k++) {
        const float *tap = input + tap_offsets[k];
        const float *w = weights + k * ch;
        for (size_t f = 0; f < num_frames; f++) {
            size_t off = f * ch;
            for (size_t c = 0; c < ch; c++) {
                output[off + c] += w[c] * tap[off + c];
            }
        }
    }
    for (size_t f = 0; f < num_frames; f++) {
        size_t off = f * ch;
        for (size_t c = 0; c < ch; c++) {
            output[off + c] += bias[c];
        }
    }
}

/* ── Full Conv1d block processing (general/small-matrix case) ───────────
 * For the small dot-product path (out_ch * in_ch < SGEMM threshold).
 * Processes all kernel taps and then adds bias in one call.
 *
 * weights: [kernel_size][in_ch * out_ch] col-major per tap
 *          weights[k * (out_ch * in_ch) + i * out_ch + o]
 * input: ring-buffer storage
 * tap_offsets: element offsets into input for each tap
 */
void fast_conv1d_small_gemv(
    float *restrict output,
    const float *restrict input,
    const size_t *restrict tap_offsets,
    const float *restrict weights,
    const float *restrict bias,
    size_t out_ch,
    size_t in_ch,
    size_t kernel_size,
    size_t num_frames
) {
    /* Accumulate taps before adding bias to match Eigen. */
    for (size_t f = 0; f < num_frames; f++) {
        size_t off = f * out_ch;
        for (size_t o = 0; o < out_ch; o++) {
            output[off + o] = 0.0f;
        }
    }

    /* Accumulate all taps */
    size_t w_stride = out_ch * in_ch;
    for (size_t k = 0; k < kernel_size; k++) {
        const float *tap = input + tap_offsets[k];
        const float *w = weights + k * w_stride;
        for (size_t f = 0; f < num_frames; f++) {
            size_t in_off = f * in_ch;
            size_t out_off = f * out_ch;
            if (out_ch == 1 && in_ch == 4) {
                float product =
                    (w[0] * tap[in_off] + w[1] * tap[in_off + 1])
                    + (w[2] * tap[in_off + 2] + w[3] * tap[in_off + 3]);
                output[out_off] += product;
                continue;
            }
            for (size_t o = 0; o < out_ch; o++) {
                float sum = 0.0f;
                for (size_t i = 0; i < in_ch; i++) {
                    sum += w[i * out_ch + o] * tap[in_off + i];
                }
                output[out_off + o] += sum;
            }
        }
    }
    for (size_t f = 0; f < num_frames; f++) {
        size_t off = f * out_ch;
        for (size_t o = 0; o < out_ch; o++) {
            output[off + o] += bias[o];
        }
    }
}

/* ── Vector add: c[i] = a[i] + b[i] ────────────────────────────────────
 */
void fast_vec_add(
    float *restrict c,
    const float *restrict a,
    const float *restrict b,
    size_t len
) {
    for (size_t i = 0; i < len; i++) {
        c[i] = a[i] + b[i];
    }
}

/* ── Vector add in-place: a[i] += b[i] ──────────────────────────────────
 */
void fast_vec_add_inplace(
    float *restrict a,
    const float *restrict b,
    size_t len
) {
    for (size_t i = 0; i < len; i++) {
        a[i] += b[i];
    }
}

/* ── Add bias to each column: output[f*ch + o] += bias[o] ──────────────
 */
void fast_add_bias(
    float *restrict output,
    const float *restrict bias,
    size_t ch,
    size_t num_frames
) {
    for (size_t f = 0; f < num_frames; f++) {
        size_t off = f * ch;
        for (size_t c = 0; c < ch; c++) {
            output[off + c] += bias[c];
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
#if defined(__APPLE__)
        float sums[256];
        const size_t chunk_capacity = sizeof(sums) / sizeof(sums[0]);
        for (size_t offset = 0; offset < len; offset += chunk_capacity) {
            size_t remaining = len - offset;
            int count = (int)(remaining < chunk_capacity ? remaining : chunk_capacity);
            for (int i = 0; i < count; i++) {
                sums[i] = conv_out[offset + (size_t)i] + mixin_out[offset + (size_t)i];
            }
            vvtanhf(z_out + offset, sums, &count);
        }
#elif defined(NAM_GLIBC_VECTOR_TANH)
        size_t offset = 0;
        nam_tanhf4_fn tanh4 =
            atomic_load_explicit(&nam_tanhf4, memory_order_acquire);
        if (tanh4 != NULL) {
            offset += nam_add_tanh_sse2(
                z_out + offset,
                conv_out + offset,
                mixin_out + offset,
                len - offset,
                tanh4);
        }
        for (; offset < len; offset++) {
            z_out[offset] = tanhf(conv_out[offset] + mixin_out[offset]);
        }
#else
        for (size_t i = 0; i < len; i++) {
            z_out[i] = tanhf(conv_out[i] + mixin_out[i]);
        }
#endif
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
            float sum = weights[o] * input[in_off];
            for (size_t i = 1; i < in_ch; i++) {
                sum += weights[i * out_ch + o] * input[in_off + i];
            }
            output[out_off + o] = sum + (bias ? bias[o] : 0.0f);
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

/* ── Gated activation: z[c] = primary(z[c]) * secondary(z[bottleneck+c])
 * activation_type: 0=Tanh, 1=SiLU, 2=Hardswish, 3=Softsign,
 *                  4=HardTanh, 5=ReLU, 6=Sigmoid, 7=Softsigmoid
 * Applies to topRows(bottleneck) of z, which has z_rows stride.
 */
static inline float apply_activation(float x, int type, int use_fast_tanh) {
    switch (type) {
        case 0: /* Tanh */
            if (use_fast_tanh) {
                float ax = fabsf(x);
                float x2 = x * x;
                return (x * (2.45550750702956f + 2.45550750702956f * ax
                        + (0.893229853513558f + 0.821226666969744f * ax) * x2))
                     / (2.44506634652299f + (2.44506634652299f + x2)
                        * fabsf(x + 0.814642734961073f * x * ax));
            }
            return tanhf(x);
        case 1: { /* SiLU = x * sigmoid(x) */
            return x / (1.0f + expf(-x));
        }
        case 2: { /* Hardswish = x * clamp(x+3, 0, 6) / 6 */
            float t = x + 3.0f;
            float clamped = t < 0.0f ? 0.0f : (t > 6.0f ? 6.0f : t);
            return (x * (1.0f / 6.0f)) * clamped;
        }
        case 3: /* Softsign = x / (1 + |x|) */
            return x / (1.0f + fabsf(x));
        case 4: /* HardTanh = clamp(x, -1, 1) */
            return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
        case 5: /* ReLU */
            return x > 0.0f ? x : 0.0f;
        case 6: /* Sigmoid */
            return 1.0f / (1.0f + expf(-x));
        case 7: /* Softsigmoid */
            return 0.5f * (1.0f + x / (1.0f + fabsf(x)));
        default:
            return x;
    }
}

void fast_gated_activation(
    float *restrict z,
    size_t z_rows,
    size_t bottleneck,
    size_t num_frames,
    int primary_type,
    int secondary_type,
    int use_fast_tanh
) {
    for (size_t f = 0; f < num_frames; f++) {
        size_t z_off = f * z_rows;
        for (size_t c = 0; c < bottleneck; c++) {
            float primary = apply_activation(z[z_off + c], primary_type, use_fast_tanh);
            float gate = apply_activation(z[z_off + bottleneck + c], secondary_type, use_fast_tanh);
            z[z_off + c] = primary * gate;
        }
    }
}

void fast_blended_activation(
    float *restrict z,
    size_t z_rows,
    size_t bottleneck,
    size_t num_frames,
    int primary_type,
    int secondary_type,
    int use_fast_tanh
) {
    for (size_t f = 0; f < num_frames; f++) {
        size_t z_off = f * z_rows;
        for (size_t c = 0; c < bottleneck; c++) {
            float pre_act = z[z_off + c];
            float activated = apply_activation(pre_act, primary_type, use_fast_tanh);
            float alpha = apply_activation(z[z_off + bottleneck + c], secondary_type, use_fast_tanh);
            z[z_off + c] = fmaf(alpha, activated - pre_act, pre_act);
        }
    }
}

/* ── Activation in-place (any type) ─────────────────────────────────────
 */
void fast_activation_inplace(float *data, size_t len, int type, int use_fast_tanh) {
    for (size_t i = 0; i < len; i++) {
        data[i] = apply_activation(data[i], type, use_fast_tanh);
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
