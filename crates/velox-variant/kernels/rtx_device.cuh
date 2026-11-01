// Device facts the tuned kernels size their grids by. This kernel set runs on more than one GB202
// SKU (RTX 5090: 170 SMs, RTX PRO 6000: 188 SMs), so the SM count is read at runtime rather than
// compiled in; a grid sized for 170 leaves the extra SMs idle or unevenly loaded.
#pragma once

#include <cuda_runtime.h>

inline int rtx_sm_count() {
    static const int sms = [] {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n <= 0)
            n = 188;  // RTX PRO 6000, this branch's card
        return n;
    }();
    return sms;
}
