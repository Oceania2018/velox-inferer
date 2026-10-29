// RTX 5060 Ti Gated DeltaNet at decode widths (up to kMaxT tokens), in ninfer's
// recurrent_batch_update layout: a warp owns four state rows and each lane the row's k_dim / 32
// consecutive elements, so a row's 512-byte load and store are one coalesced vector access per
// lane (float4 at k_dim 128). Main's per-row kernel gives a warp 8 rows of 4 lanes reading strided
// scalars; at 48 heads x 128 x 128 (3 MB of state per layer) that streams at ~216 GB/s, 13.9 us
// per layer, where this layout runs at ninfer's ~9 us. The math is main's (gdn_rows_shared in
// portable_model.cu: decay, delta, update, readout in f32); only the dot products' summation order
// differs (4 consecutive elements per lane, then a warp tree).
//
// The head's normalized q / k, alpha and beta for the window go to shared memory first (one warp
// per token), then every row runs the recurrence over the window in registers. `pp`, when set,
// carries [state_in, state_out] staged by a graph step, as in the portable kernel.
#include <stdint.h>

#include "velox_common.cuh"

namespace {

constexpr int kMaxT = 16;
constexpr int kWarps = 4;
constexpr int kRowsPerWarp = 4;
constexpr int kRowsPerBlock = kWarps * kRowsPerWarp;
constexpr int kThreads = kWarps * VELOX_WARP;

template <int PER>
struct LaneVec;
template <>
struct LaneVec<1> {
    using T = float;
};
template <>
struct LaneVec<2> {
    using T = float2;
};
template <>
struct LaneVec<4> {
    using T = float4;
};

template <int PER>
__device__ __forceinline__ void load_lane(float (&r)[PER], const float* p) {
    const typename LaneVec<PER>::T v = *reinterpret_cast<const typename LaneVec<PER>::T*>(p);
    const float* f = reinterpret_cast<const float*>(&v);
#pragma unroll
    for (int c = 0; c < PER; ++c) r[c] = f[c];
}

template <int PER>
__device__ __forceinline__ void store_lane(const float (&r)[PER], float* p) {
    typename LaneVec<PER>::T v;
    float* f = reinterpret_cast<float*>(&v);
#pragma unroll
    for (int c = 0; c < PER; ++c) f[c] = r[c];
    *reinterpret_cast<typename LaneVec<PER>::T*>(p) = v;
}

// Grid: heads x row blocks of kRowsPerBlock; block: kWarps warps.
template <int PER>
__global__ void __launch_bounds__(kThreads, 2)
    gdn_step_kernel(const float* __restrict__ qkv, const float* __restrict__ ab, const float* __restrict__ a_log,
                    const float* __restrict__ dt_bias, const float* state_in, float* state, float* __restrict__ out,
                    int tokens, int k_heads, int v_heads, int v_dim, const long long* __restrict__ pp) {
    constexpr int DK = PER * VELOX_WARP;
    if (pp != nullptr) {  // staged by a graph step: [state_in, state_out]
        state_in = reinterpret_cast<const float*>(pp[0]);
        state = reinterpret_cast<float*>(pp[1]);
    }
    __shared__ __align__(16) float sq[kMaxT][DK];
    __shared__ __align__(16) float sk[kMaxT][DK];
    __shared__ float salpha[kMaxT], sbeta[kMaxT];
    const int h = blockIdx.x, warp = threadIdx.x / VELOX_WARP, lane = threadIdx.x % VELOX_WARP;
    const int kh = h / (v_heads / k_heads);
    const int width = 2 * k_heads * DK + v_heads * v_dim;
    {
        const float neg_a = -expf(a_log[h]), bias = dt_bias[h];
        for (int t = warp; t < tokens; t += kWarps) {
            const float* row = qkv + static_cast<size_t>(t) * width;
            float q[PER], k[PER];
            load_lane<PER>(q, row + kh * DK + lane * PER);
            load_lane<PER>(k, row + (k_heads + kh) * DK + lane * PER);
            float qq = 0.0f, kk = 0.0f;
#pragma unroll
            for (int c = 0; c < PER; ++c) {
                qq += q[c] * q[c];
                kk += k[c] * k[c];
            }
            const float qn = rsqrtf(velox_warp_sum(qq) + 1e-6f), kn = rsqrtf(velox_warp_sum(kk) + 1e-6f);
#pragma unroll
            for (int c = 0; c < PER; ++c) {
                sq[t][lane * PER + c] = q[c] * qn;
                sk[t][lane * PER + c] = k[c] * kn;
            }
            if (lane == 0) {
                const float a = ab[static_cast<size_t>(t) * 2 * v_heads + h] + bias;
                const float b = ab[static_cast<size_t>(t) * 2 * v_heads + v_heads + h];
                salpha[t] = expf(neg_a * (a > 20.0f ? a : log1pf(expf(a))));
                sbeta[t] = 1.0f / (1.0f + expf(-b));
            }
        }
    }
    __syncthreads();
    const int i0 = blockIdx.y * kRowsPerBlock + warp * kRowsPerWarp;  // v_dim % kRowsPerBlock == 0
    const size_t row0 = (static_cast<size_t>(h) * v_dim + i0) * DK + lane * PER;
    float s[kRowsPerWarp][PER];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) load_lane<PER>(s[r], state_in + row0 + r * DK);
    const float inv_sqrt_dk = rsqrtf(static_cast<float>(DK));
    for (int t = 0; t < tokens; ++t) {
        const float alpha = salpha[t], beta = sbeta[t];
        float q[PER], k[PER];  // the lane's consecutive elements: one conflict-free vector read
        load_lane<PER>(q, &sq[t][lane * PER]);
        load_lane<PER>(k, &sk[t][lane * PER]);
        const float* vrow = qkv + static_cast<size_t>(t) * width + 2 * k_heads * DK + h * v_dim + i0;
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            float sk_ = 0.0f;
#pragma unroll
            for (int c = 0; c < PER; ++c) {
                s[r][c] *= alpha;
                sk_ += s[r][c] * k[c];
            }
            const float delta = beta * (vrow[r] - velox_warp_sum(sk_));
            float o = 0.0f;
#pragma unroll
            for (int c = 0; c < PER; ++c) {
                s[r][c] += delta * k[c];
                o += s[r][c] * q[c];
            }
            o = velox_warp_sum(o);
            if (lane == r) out[(static_cast<size_t>(t) * v_heads + h) * v_dim + i0 + r] = o * inv_sqrt_dk;
        }
    }
    if (state == nullptr) return;
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) store_lane<PER>(s[r], state + row0 + r * DK);
}

template <int PER>
int run(const float* qkv, const float* ab, const float* a_log, const float* dt_bias, const float* state_in,
        float* state, float* out, int tokens, int k_heads, int v_heads, int v_dim, const long long* pp) {
    const dim3 grid(v_heads, v_dim / kRowsPerBlock), block(kThreads);
    gdn_step_kernel<PER><<<grid, block>>>(qkv, ab, a_log, dt_bias, state_in, state, out, tokens, k_heads, v_heads,
                                          v_dim, pp);
    return velox_launch_status();
}

}  // namespace

// Main's gated_delta_net contract for 1..16 tokens, k_dim in {32, 64, 128}, v_dim a multiple of
// 16 and qkv 16-byte aligned (row 0 of the window); `state` null writes outputs only; `pp` as in
// the portable kernel. Returns VELOX_ERR_UNSUPPORTED for other shapes.
VELOX_API int velox_4090_gdn_step(const float* qkv, const float* ab, const float* a_log, const float* dt_bias,
                                  const float* state_in, float* state, float* out, int tokens, int k_heads,
                                  int v_heads, int k_dim, int v_dim, const long long* pp) {
    if (tokens < 1 || tokens > kMaxT || v_dim % kRowsPerBlock != 0 || k_heads <= 0 || v_heads % k_heads != 0 ||
        reinterpret_cast<uintptr_t>(qkv) % 16 != 0 || (2 * k_heads * k_dim + v_heads * v_dim) % 4 != 0)
        return VELOX_ERR_UNSUPPORTED;
    switch (k_dim) {
        case 32:
            return run<1>(qkv, ab, a_log, dt_bias, state_in, state, out, tokens, k_heads, v_heads, v_dim, pp);
        case 64:
            return run<2>(qkv, ab, a_log, dt_bias, state_in, state, out, tokens, k_heads, v_heads, v_dim, pp);
        case 128:
            return run<4>(qkv, ab, a_log, dt_bias, state_in, state, out, tokens, k_heads, v_heads, v_dim, pp);
        default:
            return VELOX_ERR_UNSUPPORTED;
    }
}
