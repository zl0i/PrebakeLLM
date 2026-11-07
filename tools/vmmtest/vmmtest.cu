// vmmtest: can one contiguous GPU virtual address range mix pages that live in
// VRAM and pages that live in host RAM, on this OS/driver?
//
// If yes, an expert tensor of a layer (256 experts, one 2 MiB page each) can
// stay a normal contiguous ggml tensor for the stock mul_mat_id kernel, while
// each expert's page physically sits either in VRAM (hot) or in RAM (cold,
// read over PCIe). Moving an expert = copy + remap one page, no kernel changes.
//
// Checks: (1) cuMemCreate with host location, (2) kernel read speed of a mixed
// range, (3) cost of remapping a page.

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CU(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s; cuGetErrorString(r, &s); \
    std::printf("%s -> %s (%d) at line %d\n", #x, s, (int)r, __LINE__); return r; } } while (0)
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("%s -> %s\n", #x, cudaGetErrorString(e)); std::exit(1);} } while (0)

__global__ void read_kernel(const int4* __restrict__ p, size_t n, int* out) {
    int4 acc = make_int4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) *out = 1;
}

__global__ void sum_kernel(const unsigned char* p, size_t n, unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) s += p[i];
    atomicAdd(out, s);
}

static int run() {
    CK(cudaFree(0));  // init runtime + primary context
    CUdevice dev; CU(cuDeviceGet(&dev, 0));

    CUmemAllocationProp pd = {};
    pd.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    pd.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    pd.location.id = 0;
    size_t gran_d = 0;
    CU(cuMemGetAllocationGranularity(&gran_d, &pd, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    CUmemAllocationProp ph = pd;
    ph.location.type = CU_MEM_LOCATION_TYPE_HOST;
    ph.location.id = 0;
    size_t gran_h = 0;
    CUresult rg = cuMemGetAllocationGranularity(&gran_h, &ph, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    std::printf("granularity: device %zu KiB, host %s %zu KiB\n", gran_d >> 10, rg == CUDA_SUCCESS ? "ok" : "ERR", gran_h >> 10);

    const size_t page = gran_d > gran_h ? gran_d : gran_h;
    const int npages = 64;  // 64 "experts"
    std::vector<CUmemGenericAllocationHandle> hd(npages), hh(npages);
    for (int i = 0; i < npages; ++i) {
        CU(cuMemCreate(&hd[i], page, &pd, 0));
        CUresult r = cuMemCreate(&hh[i], page, &ph, 0);
        if (r != CUDA_SUCCESS) {
            const char* s; cuGetErrorString(r, &s);
            std::printf("cuMemCreate(HOST location) FAILED: %s (%d) -> host pages in VMM not supported here\n", s, (int)r);
            return 1;
        }
    }
    std::printf("cuMemCreate with HOST location: OK\n");

    CUdeviceptr va;
    CU(cuMemAddressReserve(&va, page * npages, page, 0, 0));
    // even pages -> VRAM, odd pages -> host RAM
    for (int i = 0; i < npages; ++i) CU(cuMemMap(va + i * page, page, 0, (i % 2) ? hh[i] : hd[i], 0));
    CUmemAccessDesc acc = {};
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE; acc.location.id = 0; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU(cuMemSetAccess(va, page * npages, &acc, 1));
    std::printf("mixed VA range mapped: %d pages of %zu KiB\n", npages, page >> 10);

    // correctness: fill every page with its index (from the GPU side), sum back
    for (int i = 0; i < npages; ++i) CK(cudaMemset((void*)(va + i * page), i & 0xff, page));
    unsigned long long* dsum; CK(cudaMalloc(&dsum, 8)); CK(cudaMemset(dsum, 0, 8));
    sum_kernel<<<256, 256>>>((const unsigned char*)va, page * npages, dsum);
    unsigned long long s = 0; CK(cudaMemcpy(&s, dsum, 8, cudaMemcpyDeviceToHost));
    unsigned long long expect = 0; for (int i = 0; i < npages; ++i) expect += (unsigned long long)(i & 0xff) * page;
    std::printf("checksum %s (%llu vs %llu)\n", s == expect ? "OK" : "MISMATCH", s, expect);

    // speed: read all-VRAM pages vs all-host pages (separate ranges via strided read of one page)
    int* dout; CK(cudaMalloc(&dout, 4));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int which = 0; which < 2; ++which) {
        for (int blocks : {448, 56, 16}) {
            const int iters = 20;
            CK(cudaEventRecord(e0));
            for (int it = 0; it < iters; ++it)
                for (int i = which; i < npages; i += 2)
                    read_kernel<<<blocks, 256>>>((const int4*)(va + i * page), page / 16, dout);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            double bytes = (double)iters * (npages / 2) * page;
            std::printf("read %s pages, blocks=%3d: %7.2f GB/s  (%.1f us per 2MiB page)\n", which ? "HOST" : "VRAM", blocks,
                        bytes / (ms * 1e6), ms * 1000 / (iters * npages / 2));
        }
    }

    // remap cost: move page 1 (host) to a VRAM handle and back
    {
        CUmemGenericAllocationHandle spare; CU(cuMemCreate(&spare, page, &pd, 0));
        const int n = 200;
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < n; ++k) {
            CU(cuMemUnmap(va + 1 * page, page));
            CU(cuMemMap(va + 1 * page, page, 0, (k % 2) ? hh[1] : spare, 0));
            CU(cuMemSetAccess(va + 1 * page, page, &acc, 1));
        }
        CK(cudaDeviceSynchronize());
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
        std::printf("remap one page (unmap+map+setaccess): %.1f us\n", us);
    }
    std::printf("done\n");
    return 0;
}

int main() { return run(); }
