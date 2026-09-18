/* Geraetecheck + reale VRAM-Bandbreite: der Nenner fuer die NWC1-Bilanz. */
#include <cstdio>
#include <cuda_runtime.h>

__global__ void lesen(const float4 *__restrict__ in, float4 *__restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t schritt = gridDim.x * (size_t)blockDim.x;
    float4 acc = make_float4(0, 0, 0, 0);
    for (; i < n; i += schritt) {
        float4 v = in[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1e30f) out[0] = acc;
}

__global__ void kopieren(const float4 *__restrict__ in, float4 *__restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t schritt = gridDim.x * (size_t)blockDim.x;
    for (; i < n; i += schritt) out[i] = in[i];
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    int takt = 0, bus = 0;
    cudaDeviceGetAttribute(&takt, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus,  cudaDevAttrGlobalMemoryBusWidth, 0);

    printf("GPU            : %s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("SMs            : %d,  max Threads/SM: %d\n", p.multiProcessorCount, p.maxThreadsPerMultiProcessor);
    printf("Shared Mem/SM  : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("VRAM           : %.1f GB\n", p.totalGlobalMem / 1e9);
    printf("Speicher       : %.0f MHz, %d bit -> theoretisch %.0f GB/s\n",
           takt / 1000.0, bus, 2.0 * takt * 1000.0 * (bus / 8) / 1e9);

    size_t bytes = 512ull << 20;
    size_t n = bytes / sizeof(float4);
    float4 *a, *b;
    if (cudaMalloc(&a, bytes) != cudaSuccess || cudaMalloc(&b, bytes) != cudaSuccess) {
        printf("cudaMalloc fehlgeschlagen\n"); return 1;
    }
    cudaMemset(a, 1, bytes);

    int bl = p.multiProcessorCount * 32, th = 256;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float ms;

    for (int w = 0; w < 3; w++) lesen<<<bl, th>>>(a, b, n);
    cudaDeviceSynchronize();
    cudaEventRecord(e0); for (int w = 0; w < 10; w++) lesen<<<bl, th>>>(a, b, n);
    cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
    printf("\nnur lesen      : %.1f GB/s\n", 10.0 * bytes / (ms / 1000.0) / 1e9);

    for (int w = 0; w < 3; w++) kopieren<<<bl, th>>>(a, b, n);
    cudaDeviceSynchronize();
    cudaEventRecord(e0); for (int w = 0; w < 10; w++) kopieren<<<bl, th>>>(a, b, n);
    cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
    printf("lesen+schreiben: %.1f GB/s (beruehrt)\n", 10.0 * 2 * bytes / (ms / 1000.0) / 1e9);

    cudaError_t err = cudaGetLastError();
    printf("\nStatus         : %s\n", err == cudaSuccess ? "OK" : cudaGetErrorString(err));
    return 0;
}
