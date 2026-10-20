// Portable kernels: correct on every supported NVIDIA architecture, simple rather than fast.
// They are the fallback for any op a hardware branch has not tuned yet, so a feature added on main
// works on every branch immediately. Shapes are validated on the Rust side before launch.
#include <math.h>

#include "velox_common.cuh"

namespace {

// y[m, n] = x[m, k] . w[n, k]^T. One warp per output element.
__global__ void matmul_warp_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                   int m, int n, int k) {
    const int col = blockIdx.x * (blockDim.x / VELOX_WARP) + threadIdx.x / VELOX_WARP;
    const int lane = threadIdx.x % VELOX_WARP;
    if (col >= n) return;
    const float* wr = w + static_cast<size_t>(col) * k;
    for (int row = blockIdx.y; row < m; row += gridDim.y) {
        const float* xr = x + static_cast<size_t>(row) * k;
        float acc = 0.0f;
        for (int i = lane; i < k; i += VELOX_WARP) acc += xr[i] * wr[i];
        acc = velox_warp_sum(acc);
        if (lane == 0) y[static_cast<size_t>(row) * n + col] = acc;
    }
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ y,
                                int dim, float eps) {
    __shared__ float scratch[VELOX_WARP];
    const float* xr = x + static_cast<size_t>(blockIdx.x) * dim;
    float* yr = y + static_cast<size_t>(blockIdx.x) * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    ss = velox_block_sum(ss, scratch);
    const float inv = rsqrtf(ss / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = xr[i] * inv * weight[i];
}

// The angle is formed in double: positions reach 1e5+, where a float angle loses ~1e-2 rad.
__global__ void rope_kernel(float* __restrict__ x, long long pairs, int heads, int head_dim, int pos0, float theta) {
    const int half = head_dim / 2;
    for (long long p = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; p < pairs;
         p += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(p % half);
        const long long row = p / half;  // token * heads + head
        const int t = static_cast<int>(row / heads);
        const double freq = pow(static_cast<double>(theta), -2.0 * i / head_dim);
        double s, c;
        sincos(static_cast<double>(pos0 + t) * freq, &s, &c);
        float* base = x + row * head_dim;
        const float a = base[i], b = base[i + half];
        base[i] = static_cast<float>(a * c - b * s);
        base[i + half] = static_cast<float>(a * s + b * c);
    }
}

__global__ void silu_mul_kernel(float* __restrict__ gate, const float* __restrict__ up, long long len) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

__global__ void add_kernel(float* __restrict__ acc, const float* __restrict__ x, long long len) {
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < len;
         i += static_cast<long long>(gridDim.x) * blockDim.x)
        acc[i] += x[i];
}

// One warp per (token, head); keys are visited sequentially with an online softmax. Each lane owns
// PER elements of the head dimension.
template <int PER>
__global__ void attention_warp_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, float* __restrict__ out, int heads, int kv_heads,
                                      int head_dim, int pos0, float scale) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = threadIdx.x;
    const int kvh = h / (heads / kv_heads);
    const size_t q_off = (static_cast<size_t>(t) * heads + h) * head_dim;

    float qr[PER], acc[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        qr[p] = d < head_dim ? q[q_off + d] * scale : 0.0f;
        acc[p] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    const int visible = pos0 + t + 1;
    for (int j = 0; j < visible; ++j) {
        const size_t kv_off = (static_cast<size_t>(j) * kv_heads + kvh) * head_dim;
        float s = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) s += qr[p] * k[kv_off + d];
        }
        s = velox_warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new);
        const float pj = expf(s - m_new);
        l = l * corr + pj;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * VELOX_WARP;
            if (d < head_dim) acc[p] = acc[p] * corr + pj * v[kv_off + d];
        }
        m = m_new;
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int d = lane + p * VELOX_WARP;
        if (d < head_dim) out[q_off + d] = acc[p] / l;
    }
}

}  // namespace

VELOX_API int velox_portable_matmul(const float* x, const float* w, float* y, int m, int n, int k) {
    constexpr int kWarps = 8;
    dim3 grid((n + kWarps - 1) / kWarps, m < 65535 ? m : 65535);
    matmul_warp_kernel<<<grid, kWarps * VELOX_WARP>>>(x, w, y, m, n, k);
    return velox_launch_status();
}

VELOX_API int velox_portable_rms_norm(const float* x, const float* weight, float* y, int rows, int dim, float eps) {
    rms_norm_kernel<<<rows, 256>>>(x, weight, y, dim, eps);
    return velox_launch_status();
}

VELOX_API int velox_portable_rope(float* x, int tokens, int heads, int head_dim, int pos0, float theta) {
    const long long pairs = static_cast<long long>(tokens) * heads * (head_dim / 2);
    rope_kernel<<<velox_grid_1d(pairs, 256), 256>>>(x, pairs, heads, head_dim, pos0, theta);
    return velox_launch_status();
}

VELOX_API int velox_portable_silu_mul(float* gate, const float* up, long long len) {
    silu_mul_kernel<<<velox_grid_1d(len, 256), 256>>>(gate, up, len);
    return velox_launch_status();
}

VELOX_API int velox_portable_add(float* acc, const float* x, long long len) {
    add_kernel<<<velox_grid_1d(len, 256), 256>>>(acc, x, len);
    return velox_launch_status();
}

VELOX_API int velox_portable_attention(const float* q, const float* k, const float* v, float* out, int tokens,
                                       int heads, int kv_heads, int head_dim, int pos0, float scale) {
    if (tokens > 65535) return VELOX_ERR_UNSUPPORTED;
    dim3 grid(heads, tokens);
    if (head_dim <= 32) {
        attention_warp_kernel<1><<<grid, VELOX_WARP>>>(q, k, v, out, heads, kv_heads, head_dim, pos0, scale);
    } else if (head_dim <= 64) {
        attention_warp_kernel<2><<<grid, VELOX_WARP>>>(q, k, v, out, heads, kv_heads, head_dim, pos0, scale);
    } else if (head_dim <= 128) {
        attention_warp_kernel<4><<<grid, VELOX_WARP>>>(q, k, v, out, heads, kv_heads, head_dim, pos0, scale);
    } else if (head_dim <= 256) {
        attention_warp_kernel<8><<<grid, VELOX_WARP>>>(q, k, v, out, heads, kv_heads, head_dim, pos0, scale);
    } else {
        return VELOX_ERR_UNSUPPORTED;
    }
    return velox_launch_status();
}
