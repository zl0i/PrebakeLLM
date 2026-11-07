// pingpong: how fast can a running CUDA graph hand work to a CPU thread and
// get the answer back, without any CUDA API call on the CPU side?
//
// Mechanism (what a "GPU asks CPU to compute the cold experts" step needs):
//   GPU: kernel `signal` writes seq into pinned host memory (mapped)  ->
//   CPU: a thread spinning on that memory sees it, does its work, writes `done = seq` ->
//   GPU: kernel `wait` spins on `done` (reads pinned host memory over PCIe) and returns.
// The whole sequence is captured into a CUDA graph, 40 times (one per layer),
// and replayed. We measure the cost per round trip, and how much GPU work
// placed between signal and wait hides it.

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("%s -> %s\n", #x, cudaGetErrorString(e)); std::exit(1);} } while (0)

struct mailbox {
    volatile unsigned int to_cpu;   // written by GPU
    unsigned int pad0[31];
    volatile unsigned int to_gpu;   // written by CPU
    unsigned int pad1[31];
};

__global__ void k_signal(mailbox * mb, unsigned int * counter) {
    // counter lives in device memory, incremented per signal so graph replays get new sequence numbers
    unsigned int seq = ++(*counter);
    __threadfence_system();
    mb->to_cpu = seq;
    __threadfence_system();
}

__global__ void k_wait(mailbox * mb, const unsigned int * counter) {
    const unsigned int seq = *counter;
    while (mb->to_gpu < seq) { }
}

__global__ void k_busy(float * p, int iters) {  // stand-in for GPU work done meanwhile
    float v = p[threadIdx.x];
    for (int i = 0; i < iters; ++i) v = v * 1.000001f + 0.5f;
    p[threadIdx.x] = v;
}

int main() {
    mailbox * mb = nullptr;
    CK(cudaHostAlloc((void **) &mb, sizeof(mailbox), cudaHostAllocMapped));
    mb->to_cpu = 0; mb->to_gpu = 0;
    mailbox * d_mb = nullptr;
    CK(cudaHostGetDevicePointer((void **) &d_mb, mb, 0));
    unsigned int * d_counter; CK(cudaMalloc(&d_counter, 4)); CK(cudaMemset(d_counter, 0, 4));
    float * d_buf; CK(cudaMalloc(&d_buf, 4096)); CK(cudaMemset(d_buf, 0, 4096));

    std::atomic<bool> stop{false};
    std::atomic<int> cpu_work_us{0};
    std::thread cpu([&] {
        unsigned int seen = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            unsigned int s = mb->to_cpu;
            if (s != seen) {
                std::atomic_thread_fence(std::memory_order_acquire);
                const int us = cpu_work_us.load();
                if (us > 0) {
                    auto t0 = std::chrono::steady_clock::now();
                    while (std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(us)) {}
                }
                std::atomic_thread_fence(std::memory_order_release);
                mb->to_gpu = s;
                seen = s;
            }
        }
    });

    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    const int layers = 40;

    auto run = [&](const char * name, int busy_iters, int cpu_us, bool with_pingpong) {
        cpu_work_us = cpu_us;
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
        for (int l = 0; l < layers; ++l) {
            if (with_pingpong) k_signal<<<1, 1, 0, st>>>(d_mb, d_counter);
            k_busy<<<1, 256, 0, st>>>(d_buf, busy_iters);
            if (with_pingpong) k_wait<<<1, 1, 0, st>>>(d_mb, d_counter);
        }
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int w = 0; w < 5; ++w) CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
        const int reps = 200;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("%-44s %8.1f us per graph (40 layers) = %6.2f us per layer\n", name, us, us / layers);
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    };

    run("busy only (no handoff), busy=0", 0, 0, false);
    run("handoff, no GPU/CPU work", 0, 0, true);
    run("busy only, busy=2000", 2000, 0, false);
    run("handoff + GPU busy=2000, CPU 0us", 2000, 0, true);
    run("handoff + GPU busy=2000, CPU 20us", 2000, 20, true);
    run("handoff, GPU 0, CPU 20us", 0, 20, true);

    stop = true;
    cpu.join();
    return 0;
}
