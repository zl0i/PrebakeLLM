// overlap: does a host->device copy on one stream run at the same time as a
// kernel on another stream, on this OS/driver (Windows WDDM)? And does a copy
// done by a kernel (reading pinned host memory) overlap instead?

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("%s -> %s\n", #x, cudaGetErrorString(e)); std::exit(1);} } while (0)

__global__ void k_busy(float * p, int iters) {
    float v = p[blockIdx.x * blockDim.x + threadIdx.x];
    for (int i = 0; i < iters; ++i) v = v * 1.000001f + 0.5f;
    p[blockIdx.x * blockDim.x + threadIdx.x] = v;
}

__global__ void k_copy(int4 * dst, const int4 * src, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}

int main() {
    const size_t bytes = 64ull << 20;
    void * h; CK(cudaHostAlloc(&h, bytes, cudaHostAllocMapped));
    void * hd; CK(cudaHostGetDevicePointer(&hd, h, 0));
    void * d; CK(cudaMalloc(&d, bytes));
    float * buf; CK(cudaMalloc(&buf, 56 * 4 * 256 * 4));
    cudaStream_t s1, s2;
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    cudaEvent_t a0, a1;
    CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
    const int iters = 400000;

    auto timeit = [&](const char * name, auto fn) {
        fn(); CK(cudaDeviceSynchronize());
        auto t0 = std::chrono::steady_clock::now();
        fn(); CK(cudaDeviceSynchronize());
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%-46s %8.2f ms\n", name, ms);
    };
    timeit("kernel alone", [&] { k_busy<<<56 * 4, 256, 0, s1>>>(buf, iters); });
    timeit("memcpy H2D 64 MiB alone", [&] { CK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s2)); });
    timeit("kernel (s1) + memcpy (s2) together", [&] {
        k_busy<<<56 * 4, 256, 0, s1>>>(buf, iters);
        CK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s2));
    });
    timeit("memcpy (s2) first, then kernel (s1)", [&] {
        CK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s2));
        k_busy<<<56 * 4, 256, 0, s1>>>(buf, iters);
    });
    timeit("copy kernel 16 blocks alone", [&] { k_copy<<<16, 256, 0, s2>>>((int4 *) d, (const int4 *) hd, bytes / 16); });
    timeit("kernel (s1, 200 blocks) + copy kernel (s2)", [&] {
        k_busy<<<200, 256, 0, s1>>>(buf, iters);
        k_copy<<<16, 256, 0, s2>>>((int4 *) d, (const int4 *) hd, bytes / 16);
    });
    timeit("kernel (s1, 200 blocks) alone", [&] { k_busy<<<200, 256, 0, s1>>>(buf, iters); });
    return 0;
}
