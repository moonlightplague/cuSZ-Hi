#include "bottleneck_test.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "compressor.hh"
#include "mem.hh"
#include "port.hh"
#include "tehm.hh"
#include "utils/config.hh"
#include "utils/err.hh"

namespace cusz {
namespace {

struct TimingStats {
  float min_ms{0};
  float median_ms{0};
  float avg_ms{0};
};

size_t filesize_or_throw(const char* fname)
{
  std::ifstream in(fname, std::ios::binary | std::ios::ate);
  if (not in.is_open())
    throw std::runtime_error(std::string("failed to open ") + fname);
  return static_cast<size_t>(in.tellg());
}

TimingStats summarize(std::vector<float> values)
{
  if (values.empty()) throw std::runtime_error("no timing samples collected");

  std::sort(values.begin(), values.end());
  double sum = 0.0;
  for (auto v : values) sum += v;

  TimingStats stats;
  stats.min_ms = values.front();
  stats.median_ms = values[values.size() / 2];
  stats.avg_ms = static_cast<float>(sum / values.size());
  return stats;
}

double gib_per_s(double bytes, double ms)
{
  return bytes / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
}

double gflop_per_s(double flops, double ms)
{
  return flops / (ms / 1000.0) / 1.0e9;
}

const char* predictor_name(psz_predtype pred)
{
  return pred == Spline ? "spline3" : "lorenzo";
}

const char* mode_name(psz_mode mode)
{
  return mode == Rel ? "r2r" : "abs";
}

__global__ void rw_probe_kernel(
    const float* __restrict__ in, uint8_t* __restrict__ q,
    float* __restrict__ scratch, size_t n, float scale)
{
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) return;

  auto x = in[id] * scale;
  scratch[id] = x;
  q[id] = static_cast<uint8_t>(static_cast<int>(fabsf(x)) & 0xff);
}

__global__ void fma_probe_kernel(
    const float* __restrict__ in, uint8_t* __restrict__ q,
    float* __restrict__ scratch, size_t n, int iters)
{
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) return;

  auto x = in[id];
  for (int i = 0; i < iters; ++i) {
    x = fmaf(x, 1.000000119f, 0.000000119f);
    x = fmaf(x, -0.999999881f, 0.25f);
  }

  scratch[id] = x;
  q[id] = static_cast<uint8_t>(__float2int_rn(fabsf(x)) & 0xff);
}

float time_device_copy(
    const float* in, float* out, size_t n, int repeats, cudaStream_t stream)
{
  CHECK_GPU(cudaMemcpyAsync(
      out, in, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  CHECK_GPU(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_GPU(cudaEventCreate(&start));
  CHECK_GPU(cudaEventCreate(&stop));

  CHECK_GPU(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i)
    CHECK_GPU(cudaMemcpyAsync(
        out, in, n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  CHECK_GPU(cudaEventRecord(stop, stream));
  CHECK_GPU(cudaEventSynchronize(stop));

  float ms = 0;
  CHECK_GPU(cudaEventElapsedTime(&ms, start, stop));
  CHECK_GPU(cudaEventDestroy(start));
  CHECK_GPU(cudaEventDestroy(stop));
  return ms / repeats;
}

float time_rw_probe(
    const float* in, uint8_t* q, float* scratch, size_t n, int repeats,
    cudaStream_t stream)
{
  constexpr auto block = 256;
  auto grid = static_cast<unsigned int>((n + block - 1) / block);

  rw_probe_kernel<<<grid, block, 0, stream>>>(in, q, scratch, n, 1.0f);
  CHECK_GPU(cudaStreamSynchronize(stream));
  CHECK_GPU(cudaGetLastError());

  cudaEvent_t start, stop;
  CHECK_GPU(cudaEventCreate(&start));
  CHECK_GPU(cudaEventCreate(&stop));

  CHECK_GPU(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i)
    rw_probe_kernel<<<grid, block, 0, stream>>>(in, q, scratch, n, 1.0f);
  CHECK_GPU(cudaEventRecord(stop, stream));
  CHECK_GPU(cudaEventSynchronize(stop));
  CHECK_GPU(cudaGetLastError());

  float ms = 0;
  CHECK_GPU(cudaEventElapsedTime(&ms, start, stop));
  CHECK_GPU(cudaEventDestroy(start));
  CHECK_GPU(cudaEventDestroy(stop));
  return ms / repeats;
}

float time_fma_probe(
    const float* in, uint8_t* q, float* scratch, size_t n, int iters,
    int repeats, cudaStream_t stream)
{
  constexpr auto block = 256;
  auto grid = static_cast<unsigned int>((n + block - 1) / block);

  fma_probe_kernel<<<grid, block, 0, stream>>>(in, q, scratch, n, iters);
  CHECK_GPU(cudaStreamSynchronize(stream));
  CHECK_GPU(cudaGetLastError());

  cudaEvent_t start, stop;
  CHECK_GPU(cudaEventCreate(&start));
  CHECK_GPU(cudaEventCreate(&stop));

  CHECK_GPU(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i)
    fma_probe_kernel<<<grid, block, 0, stream>>>(in, q, scratch, n, iters);
  CHECK_GPU(cudaEventRecord(stop, stream));
  CHECK_GPU(cudaEventSynchronize(stop));
  CHECK_GPU(cudaGetLastError());

  float ms = 0;
  CHECK_GPU(cudaEventElapsedTime(&ms, start, stop));
  CHECK_GPU(cudaEventDestroy(start));
  CHECK_GPU(cudaEventDestroy(stop));
  return ms / repeats;
}

std::vector<float> time_production_predictor(
    CompressorF4& cor, pszctx* ctx, float* data, int warmups, int repeats,
    cudaStream_t stream)
{
  std::vector<float> samples;
  samples.reserve(repeats);

  for (int i = 0; i < warmups + repeats; ++i) {
    CHECK_GPU(cudaMemsetAsync(
        cor.mem->compact->num(), 0, sizeof(uint32_t), stream));
    cor.compress_predict(ctx, data, stream);
    if (i >= warmups) samples.push_back(cor.predictor_time_ms());
  }

  return samples;
}

}  // namespace

int run_bottleneck_test(pszctx* ctx)
{
  if (ctx->dtype != F4)
    throw std::runtime_error("--bottleneck-test currently supports f32 only");
  if (ctx->eb <= 0)
    throw std::runtime_error("--bottleneck-test requires a positive -e/--eb");

  constexpr int warmups = 2;
  constexpr int fma_iters = 256;

  auto n = static_cast<size_t>(ctx->x) * ctx->y * ctx->z;
  auto expected_bytes = n * sizeof(float);
  auto input_bytes = filesize_or_throw(ctx->infile);
  if (input_bytes < expected_bytes) {
    throw std::runtime_error(
        "input file is smaller than the requested f32 dimensions");
  }
  if (input_bytes != expected_bytes) {
    std::printf(
        "[bottleneck-test] warning: input has %zu bytes; reading first %zu "
        "bytes for requested dimensions\n",
        input_bytes, expected_bytes);
  }

  int device = 0;
  CHECK_GPU(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CHECK_GPU(cudaGetDeviceProperties(&prop, device));

  cudaStream_t stream{};
  CHECK_GPU(cudaStreamCreate(&stream));

  pszmem_cxx<float> input(ctx->x, ctx->y, ctx->z, "bneck-in");
  input.control({MallocHost, Malloc})
      ->file(ctx->infile, FromFile)
      ->control({H2D});

  psz_context bench_ctx = *ctx;
  bench_ctx.rel_eb = ctx->mode == Rel ? ctx->eb : 0.0;
  if (ctx->mode == Rel) {
    double maxv = 0, minv = 0, range = 0;
    input.extrema_scan(maxv, minv, range);
    bench_ctx.eb = ctx->eb * range;
    std::printf(
        "[bottleneck-test] data range %.9g, relative eb %.9g -> absolute eb "
        "%.9g\n",
        range, ctx->eb, bench_ctx.eb);
  }

  CompressorHelper::autotune_coarse_parhf(&bench_ctx);
  CompressorF4 cor;
  cor.init(&bench_ctx);

  auto predictor_samples = time_production_predictor(
      cor, &bench_ctx, input.dptr(), warmups, bench_ctx.bottleneck_repeats,
      stream);
  auto predictor = summarize(predictor_samples);
  auto outliers = bench_ctx.splen;

  uint8_t* d_q = nullptr;
  float* d_scratch = nullptr;
  float* d_copy = nullptr;
  CHECK_GPU(cudaMalloc(&d_q, n * sizeof(uint8_t)));
  CHECK_GPU(cudaMalloc(&d_scratch, n * sizeof(float)));
  CHECK_GPU(cudaMalloc(&d_copy, n * sizeof(float)));

  auto copy_ms =
      time_device_copy(input.dptr(), d_copy, n, bench_ctx.bottleneck_repeats,
                       stream);
  auto rw_ms = time_rw_probe(
      input.dptr(), d_q, d_scratch, n, bench_ctx.bottleneck_repeats, stream);
  auto fma_ms = time_fma_probe(
      input.dptr(), d_q, d_scratch, n, fma_iters,
      bench_ctx.bottleneck_repeats, stream);

  CHECK_GPU(cudaFree(d_copy));
  CHECK_GPU(cudaFree(d_scratch));
  CHECK_GPU(cudaFree(d_q));
  CHECK_GPU(cudaStreamDestroy(stream));

  auto anchor_bytes =
      bench_ctx.pred_type == Spline ? cor.mem->ac->bytes() : size_t{0};
  auto outlier_bytes = outliers * (sizeof(float) + sizeof(uint32_t));
  auto pred_min_bytes =
      n * (sizeof(float) + sizeof(uint8_t)) + anchor_bytes + outlier_bytes;
  auto rw_bytes = n * (sizeof(float) + sizeof(uint8_t) + sizeof(float));
  auto copy_bytes = 2.0 * expected_bytes;
  auto fma_flops = static_cast<double>(n) * fma_iters * 4.0;

  auto pred_min_bw = gib_per_s(static_cast<double>(pred_min_bytes),
                               predictor.median_ms);
  auto copy_bw = gib_per_s(copy_bytes, copy_ms);
  auto rw_bw = gib_per_s(static_cast<double>(rw_bytes), rw_ms);
  auto fma_gflops = gflop_per_s(fma_flops, fma_ms);
  auto pred_to_rw = predictor.median_ms / rw_ms;
  auto pred_bw_fraction = pred_min_bw / rw_bw;

  int mem_clock_khz = 0;
  int mem_bus_width_bits = 0;
  CHECK_GPU(cudaDeviceGetAttribute(
      &mem_clock_khz, cudaDevAttrMemoryClockRate, device));
  CHECK_GPU(cudaDeviceGetAttribute(
      &mem_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device));

  double theoretical_gb_s = 0.0;
  if (mem_clock_khz > 0 and mem_bus_width_bits > 0) {
    theoretical_gb_s =
        2.0 * mem_clock_khz * 1000.0 * (mem_bus_width_bits / 8.0) /
        1.0e9;
  }

  std::printf("\n(bottleneck-test) PREDICTOR BOTTLENECK REPORT\n");
  std::printf("device              : %s\n", prop.name);
  if (theoretical_gb_s > 0.0)
    std::printf("theoretical mem bw  : %.1f GB/s\n", theoretical_gb_s);
  std::printf("input               : %s\n", ctx->infile);
  std::printf("dims                : %u x %u x %u (%zu f32 values)\n", ctx->x,
              ctx->y, ctx->z, n);
  std::printf("predictor           : %s\n", predictor_name(bench_ctx.pred_type));
  std::printf("mode / abs eb       : %s / %.9g\n", mode_name(ctx->mode),
              bench_ctx.eb);
  std::printf("repeats / warmups   : %d / %d\n", bench_ctx.bottleneck_repeats,
              warmups);
  std::printf("outliers last run   : %zu\n", static_cast<size_t>(outliers));

  std::printf("\nproduction predictor\n");
  std::printf("  min/median/avg ms : %.3f / %.3f / %.3f\n", predictor.min_ms,
              predictor.median_ms, predictor.avg_ms);
  std::printf("  min effective bw  : %.1f GiB/s (%zu minimum bytes)\n",
              pred_min_bw, pred_min_bytes);

  std::printf("\nreference probes\n");
  std::printf("  device copy       : %.3f ms, %.1f GiB/s\n", copy_ms, copy_bw);
  std::printf("  read/write traffic: %.3f ms, %.1f GiB/s\n", rw_ms, rw_bw);
  std::printf("  fma-amplified     : %.3f ms, %.1f GFLOP/s (%d iters)\n",
              fma_ms, fma_gflops, fma_iters);

  std::printf("\nratios\n");
  std::printf("  predictor / rw probe time       : %.2fx\n", pred_to_rw);
  std::printf("  predictor min bw / rw probe bw  : %.2f\n", pred_bw_fraction);

  if (pred_bw_fraction >= 0.55 or pred_to_rw <= 1.5) {
    std::printf(
        "\nverdict: predictor is likely limited primarily by global memory "
        "traffic/bandwidth.\n");
  }
  else if (pred_to_rw >= 3.0) {
    std::printf(
        "\nverdict: raw global memory bandwidth does not explain the predictor "
        "time; CUDA core work, shared-memory traffic, atomics, or instruction "
        "latency are more likely limiting it.\n");
  }
  else {
    std::printf(
        "\nverdict: mixed signal; predictor is slower than the traffic probe "
        "but not enough to isolate a pure CUDA-core bottleneck.\n");
  }

  return 0;
}

}  // namespace cusz
