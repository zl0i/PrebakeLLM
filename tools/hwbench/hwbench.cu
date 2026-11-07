// hwbench: measures the hardware "ceilings" that bound LLM token generation.
//
// Why: generating one token reads every active weight once. So tokens/s is
// limited by how fast bytes can be streamed from where the weights live:
//   - VRAM -> GPU cores           (vram_read)
//   - RAM  -> CPU cores           (ram_read, per thread count)
//   - RAM  -> VRAM over PCIe      (h2d_pinned / h2d_pageable)
// Knowing these numbers tells us the best possible speed for any placement
// of weights, before we optimize anything.
//
// Build: tools/hwbench/build.bat   (MSVC + nvcc, sm_89 = RTX 40xx "Ada")

#include <cuda_runtime.h>
#include <immintrin.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(1);} } while (0)

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// ---------------------------------------------------------------- GPU side
// A "kernel" is a function that runs on the GPU in thousands of copies
// (threads) at once. Each thread here sums a strided slice of the buffer,
// so all threads together read the whole buffer exactly once.
__global__ void read_kernel(const int4* __restrict__ p, size_t n, int* out) {
    int4 acc = make_int4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) *out = 1;  // keeps the compiler from skipping the reads
}

static void gpu_benches() {
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    size_t freeB = 0, totalB = 0;
    CK(cudaMemGetInfo(&freeB, &totalB));
    std::printf("gpu: %s  sm_%d%d  SMs=%d  L2=%d KB  VRAM free/total = %.2f/%.2f GiB\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize / 1024,
                freeB / 1073741824.0, totalB / 1073741824.0);

    const size_t bytes = 1ull << 30;  // 1 GiB, far larger than L2 so we measure real VRAM
    int4* d = nullptr; int* dout = nullptr;
    CK(cudaMalloc(&d, bytes)); CK(cudaMalloc(&dout, 4));
    CK(cudaMemset(d, 1, bytes));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // VRAM read bandwidth
    {
        const int iters = 20;
        read_kernel<<<prop.multiProcessorCount * 8, 256>>>(d, bytes / 16, dout);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) read_kernel<<<prop.multiProcessorCount * 8, 256>>>(d, bytes / 16, dout);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("vram_read:      %7.1f GB/s\n", iters * bytes / (ms * 1e6));
    }
    // Kernel launch overhead: an empty-ish launch, timed back to back. Matters
    // because a decode step launches hundreds of small kernels.
    {
        const int iters = 2000;
        read_kernel<<<1, 32>>>(d, 0, dout);
        CK(cudaDeviceSynchronize());
        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i) read_kernel<<<1, 32>>>(d, 0, dout);
        CK(cudaDeviceSynchronize());
        std::printf("launch_overhead: %6.2f us/kernel (stream, no graph)\n", secs(t0, clk::now()) * 1e6 / iters);
        // round trip: launch + wait, what a CPU<->GPU ping-pong costs
        auto t1 = clk::now();
        for (int i = 0; i < 500; ++i) { read_kernel<<<1, 32>>>(d, 0, dout); CK(cudaStreamSynchronize(0)); }
        std::printf("launch+sync:     %6.2f us\n", secs(t1, clk::now()) * 1e6 / 500);
    }
    // PCIe host->device, pinned (page-locked) and pageable host memory
    for (int pinned = 1; pinned >= 0; --pinned) {
        for (size_t sz : {64ull << 10, 1ull << 20, 8ull << 20, 256ull << 20}) {
            void* h = nullptr;
            if (pinned) CK(cudaMallocHost(&h, sz)); else h = std::malloc(sz);
            std::memset(h, 2, sz);
            int iters = (int)std::max<size_t>(4, (1ull << 30) / sz);
            CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
            CK(cudaEventRecord(e0));
            for (int i = 0; i < iters; ++i) CK(cudaMemcpyAsync(d, h, sz, cudaMemcpyHostToDevice));
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            std::printf("h2d_%-8s %6zu KB: %6.2f GB/s\n", pinned ? "pinned" : "pageable", sz >> 10, (double)iters * sz / (ms * 1e6));
            if (pinned) CK(cudaFreeHost(h)); else std::free(h);
        }
    }
    CK(cudaFree(d)); CK(cudaFree(dout));
}

// ---------------------------------------------------------------- CPU side
// Each thread streams its own slice of a large buffer with AVX2 loads.
// Run with different thread counts: memory bandwidth saturates long before
// all cores are busy, which tells us how many threads the CPU experts need.
static double ram_read(const char* buf, size_t bytes, int nthreads, const std::vector<DWORD_PTR>& masks) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> th;
    std::vector<double> sink(nthreads * 8);
    const size_t chunk = bytes / nthreads / 32 * 32;
    for (int t = 0; t < nthreads; ++t) {
        th.emplace_back([&, t] {
            if (!masks.empty()) SetThreadAffinityMask(GetCurrentThread(), masks[t % masks.size()]);
            const __m256i* p = (const __m256i*)(buf + t * chunk);
            const size_t n = chunk / 32;
            ready++;
            while (!go) {}
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
            for (int rep = 0; rep < 3; ++rep)
                for (size_t i = 0; i + 4 <= n; i += 4) {
                    a0 = _mm256_xor_si256(a0, _mm256_load_si256(p + i));
                    a1 = _mm256_xor_si256(a1, _mm256_load_si256(p + i + 1));
                    a2 = _mm256_xor_si256(a2, _mm256_load_si256(p + i + 2));
                    a3 = _mm256_xor_si256(a3, _mm256_load_si256(p + i + 3));
                }
            a0 = _mm256_xor_si256(_mm256_xor_si256(a0, a1), _mm256_xor_si256(a2, a3));
            sink[t * 8] = (double)_mm256_extract_epi64(a0, 0);
        });
    }
    while (ready < nthreads) {}
    auto t0 = clk::now();
    go = true;
    for (auto& x : th) x.join();
    double s = secs(t0, clk::now());
    return 3.0 * chunk * nthreads / s / 1e9;
}

static void cpu_benches() {
    // Map logical processors to P-cores / E-cores using EfficiencyClass
    // (Intel hybrid CPUs: P-cores have the higher class). Hyperthread siblings
    // share a core, so we take one logical CPU per physical core.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> info(len);
    GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)info.data(), &len);
    std::vector<DWORD_PTR> pcores, ecores;
    BYTE maxEff = 0;
    for (char* p = info.data(); p < info.data() + len;) {
        auto* r = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
        maxEff = std::max(maxEff, r->Processor.EfficiencyClass);
        p += r->Size;
    }
    for (char* p = info.data(); p < info.data() + len;) {
        auto* r = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
        KAFFINITY m = r->Processor.GroupMask[0].Mask;
        DWORD_PTR first = m & (~m + 1);  // lowest set bit = first logical CPU of the core
        (r->Processor.EfficiencyClass == maxEff ? pcores : ecores).push_back(first);
        p += r->Size;
    }
    std::printf("cpu: %zu P-cores, %zu E-cores\n", pcores.size(), ecores.size());

    const size_t bytes = 4ull << 30;  // 4 GiB
    char* buf = (char*)_aligned_malloc(bytes, 4096);
    std::memset(buf, 3, bytes);
    std::vector<DWORD_PTR> all = pcores; all.insert(all.end(), ecores.begin(), ecores.end());
    for (int n : {1, 2, 4, 6, 8}) std::printf("ram_read P-cores x%-2d: %6.1f GB/s\n", n, ram_read(buf, bytes, n, pcores));
    std::printf("ram_read E-cores x8 : %6.1f GB/s\n", ram_read(buf, bytes, 8, ecores));
    std::printf("ram_read all   x16 : %6.1f GB/s\n", ram_read(buf, bytes, 16, all));
    std::printf("ram_read OS    x24 : %6.1f GB/s  (no pinning, hyperthreads)\n", ram_read(buf, bytes, 24, {}));
    _aligned_free(buf);
}

int main() {
    cpu_benches();
    gpu_benches();
    return 0;
}
