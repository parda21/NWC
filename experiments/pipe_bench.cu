/* pipe_bench -- Issue-Durchsatz einzelner SASS-Befehle je SM-Takt (unabhaengige Ketten, 1024 Threads je SM).
   nvcc -O3 -arch=sm_86 -o build/pipe_bench experimente/pipe_bench.cu && ./build/pipe_bench                     */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define NCH 8
#define N   4096

template<int MODE>
__global__ void __launch_bounds__(1024, 1) kern(uint32_t *out, uint32_t seed) {
    uint32_t a[NCH]; uint64_t t[NCH]; float f[NCH];
    uint32_t b = seed | 1u, c = seed * 3u;
    #pragma unroll
    for (int i = 0; i < NCH; i++) { a[i] = seed + i * 977u + threadIdx.x; t[i] = a[i]; f[i] = (float)a[i]; }
    for (int it = 0; it < N; it++) {
        #pragma unroll
        for (int i = 0; i < NCH; i++) {
            const uint32_t a2 = a[(i + 1) % NCH]; const float f2 = f[(i + 1) % NCH];   /* zweiter Operand datenabhaengig */
            if (MODE == 0) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(a[i]) : "r"(a2), "r"(c));
            if (MODE == 1) asm volatile("mul.hi.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2));
            if (MODE == 2) asm volatile("{.reg .u32 lo, hi; mov.b64 {lo, hi}, %0; mad.wide.u32 %0, lo, %1, %0;}" : "+l"(t[i]) : "r"(a2));
            if (MODE == 3) asm volatile("add.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2));
            if (MODE == 4) asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a[i]) : "r"(c), "r"(a2));
            if (MODE == 5) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[i]) : "f"(f2), "f"(__uint_as_float(c)));
            if (MODE == 6) asm volatile("shf.l.wrap.b32 %0, %0, %0, %1;" : "+r"(a[i]) : "r"(a2));
            if (MODE == 7) asm volatile("{.reg .pred p; setp.ne.u32 p, %0, %1; selp.u32 %0, %0, %2, p;}" : "+r"(a[i]) : "r"(a2), "r"(c));
            if (MODE == 8) asm volatile("mul.lo.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2));
            if (MODE == 9) asm volatile("{.reg .u32 lo, hi; mov.b64 {lo, hi}, %0; mad.wide.u32 %0, lo, %1, 0;}" : "+l"(t[i]) : "r"(a2));
            if (MODE == 10) asm volatile("xor.b32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2));
            if (MODE == 11) asm volatile("mov.b32 %0, %1;" : "=r"(a[i]) : "r"(a2));
            if (MODE == 12) asm volatile("clz.b32 %0, %0;" : "+r"(a[i]));
            if (MODE == 13) asm volatile("min.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2));
            if (MODE == 14) asm volatile("mul.hi.u32 %0, %0, 4096;" : "+r"(a[i]));
            if (MODE == 15) asm volatile("shr.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2));
            /* Mischungen: gerade Ketten Befehl A, ungerade Befehl B -> zeigt, ob beide dieselbe Pipe nutzen */
            if (MODE == 20) { if (i & 1) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(a[i]) : "r"(a2), "r"(c)); else asm volatile("add.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2)); }
            if (MODE == 21) { if (i & 1) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(a[i]) : "r"(a2), "r"(c)); else asm volatile("shr.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2)); }
            if (MODE == 22) { if (i & 1) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(a[i]) : "r"(a2), "r"(c)); else asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a[i]) : "r"(c), "r"(a2)); }
            if (MODE == 23) { if (i & 1) asm volatile("shr.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2)); else asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a[i]) : "r"(c), "r"(a2)); }
            if (MODE == 24) { if (i & 1) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(a[i]) : "r"(a2), "r"(c)); else asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[i]) : "f"(f2), "f"(__uint_as_float(c))); }
            if (MODE == 25) { if (i & 1) asm volatile("shr.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2)); else asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[i]) : "f"(f2), "f"(__uint_as_float(c))); }
            if (MODE == 26) { if (i & 1) asm volatile("mul.hi.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2)); else asm volatile("shr.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2)); }
            if (MODE == 27) { if (i & 1) asm volatile("shr.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2)); else asm volatile("add.u32 %0, %0, %1;" : "+r"(a[i]) : "r"(a2)); }
            if (MODE == 28) { if (i & 1) asm volatile("{.reg .pred p; setp.ne.u32 p, %0, %1; selp.u32 %0, %0, %2, p;}" : "+r"(a[i]) : "r"(a2), "r"(c)); else asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(a[i]) : "r"(a2), "r"(c)); }
            if (MODE == 29) { if (i & 1) asm volatile("shf.l.wrap.b32 %0, %0, %0, %1;" : "+r"(a[i]) : "r"(a2)); else asm volatile("shf.l.wrap.b32 %0, %0, %0, %1;" : "+r"(a[i]) : "r"(a2)); }
        }
    }
    uint32_t s = 0;
    #pragma unroll
    for (int i = 0; i < NCH; i++) s += a[i] + (uint32_t)t[i] + __float_as_uint(f[i]);
    if (s == 0x12345678u) out[0] = s;
}

template<int MODE> static void lauf(const char *name, int sms, int clk, uint32_t *d) {
    cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
    kern<MODE><<<sms, 1024>>>(d, 1); cudaDeviceSynchronize();
    cudaEventRecord(a); kern<MODE><<<sms, 1024>>>(d, 1); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, a, e);
    double winstr = (double)sms * 32 * NCH * N;                   /* Warp-Befehle gesamt (32 Warps je SM) */
    double takte = ms * 1e-3 * sms * (clk * 1e3);
    printf("%-14s %.3f SM-Takte je Warp-Befehl  (%.2f je SMSP)\n", name, takte / winstr, 4 * takte / winstr);
}

int main() {
    int sms, clk; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0); cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    uint32_t *d; cudaMalloc(&d, 4);
    printf("%d SMs, %.3f GHz\n", sms, clk * 1e-6);
    lauf<0>("IMAD", sms, clk, d);
    lauf<8>("IMUL(lo)", sms, clk, d);
    lauf<1>("IMAD.HI", sms, clk, d);
    lauf<2>("IMAD.WIDE+acc", sms, clk, d);
    lauf<9>("IMAD.WIDE", sms, clk, d);
    lauf<14>("IMAD.HI imm", sms, clk, d);
    lauf<3>("IADD3", sms, clk, d);
    lauf<10>("LOP3", sms, clk, d);
    lauf<6>("SHF rot", sms, clk, d);
    lauf<15>("SHR", sms, clk, d);
    lauf<4>("PRMT", sms, clk, d);
    lauf<5>("FFMA", sms, clk, d);
    lauf<7>("ISETP+SEL", sms, clk, d);
    lauf<11>("MOV", sms, clk, d);
    lauf<12>("FLO", sms, clk, d);
    lauf<13>("IMNMX", sms, clk, d);
    printf("-- Mischungen (je Warp-Befehl gemittelt; = Mittel der Einzelwerte -> gleiche Pipe, deutlich kleiner -> parallel)\n");
    lauf<20>("IMAD+IADD3", sms, clk, d);
    lauf<21>("IMAD+SHF", sms, clk, d);
    lauf<22>("IMAD+PRMT", sms, clk, d);
    lauf<23>("SHF+PRMT", sms, clk, d);
    lauf<24>("IMAD+FFMA", sms, clk, d);
    lauf<25>("SHF+FFMA", sms, clk, d);
    lauf<26>("IMAD.HI+SHF", sms, clk, d);
    lauf<27>("SHF+IADD3", sms, clk, d);
    lauf<28>("ISETPSEL+IMAD", sms, clk, d);
    return 0;
}
