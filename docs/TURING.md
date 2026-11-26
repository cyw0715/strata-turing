# strata-turing: dual RTX 2080 Ti

This fork is based on Strata 0.2.20 / engine 0.1.66, upstream commit `2e87c08db18aa4577c22fd623d6ba53252719d63`. The engine changes are in commit `a8c592f911114c302e3001209e7dbd84df6e4c68`. They reproduce the accepted source patch exactly: SHA256 `eb0c4dc1a5fb392fcffe472a75c380962a15041c04d88f69dd96cb05adba3870`.

## What was measured

Qwen3.8-Flash-Next, IQ3_XXS native weights and the same MTP files, on two RTX 2080 Ti cards with **22 GiB per card**, NVLink, Linux and CUDA 13.0. The host has 32 GiB RAM. The measured preset places 22 GiB of routed experts in host RAM. A standard 11 GiB 2080 Ti pair needs a different memory budget; this preset has not been validated on that configuration.

Fixed workload: 32,768 prompt tokens, 4,096 output tokens, zero cached prompt tokens, seed 42 and temperature 0.4. Context limit: 262,144. The effective target sampling configuration and weight precision were preserved. MTP uses K=6, row cost 2.0 and proposal probability exponent 0.5. Acceptance/rejection and residual sampling use the actual transformed proposal probability.

| Paired run, both at 250W | Optimized decode | Original layer-split decode | Speedup | Prefill / original |
|---|---:|---:|---:|---:|
| Initial | 110.38 tok/s | 71.40 tok/s | 54.60% | 98.12% |
| Repeat 1 | 110.40 tok/s | 73.49 tok/s | 50.22% | 97.51% |
| Repeat 2 | 110.58 tok/s | 71.81 tok/s | 53.99% | 97.97% |

Ratios are calculated from raw milliseconds. The 280W online follow-up measured prefill 1908.26 tok/s, decode 109.68 tok/s and 54.636 seconds total. Its comparison with the earlier 250W optimized run is historical: prefill +3.19%, decode -0.64%. The highest sampled temperatures were 58°C / 53°C, with no thermal slowdown events. The power limits remained at 280W; sampled prefill power peaks were 294.45W / 301.58W. Raw readings are retained in the evidence.

## Build on Linux

Requirements: a C++ compiler supported by CUDA 13.0, CMake >=3.24, Ninja and two CUDA SM75 GPUs. NCCL's shared library is needed at runtime for this prefill preset. We used Ninja `-j4` and NVCC split compilation `2` on the four-core build host.

```sh
git clone --branch codex/qwen-expert-tp https://github.com/cyw0715/strata-turing.git
cd strata-turing
CUDA_ROOT=/usr/local/cuda-13.0
export PATH="$CUDA_ROOT/bin:$PATH"
export NVCC_APPEND_FLAGS=--split-compile=2
cmake -S engine -B build-turing -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_SPLIT=ON \
  -DSTRATA_SPLIT_CUDA_PEER=ON \
  -DSTRATA_SPLIT_ARCH=75 \
  -DSTRATA_SPLIT_CARDS=2 \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT"
cmake --build build-turing --target strata-split-2 -j4
```

The executable is `build-turing/src/split/strata-split-2`. Use the explicit CMake commands above: upstream's `build_split_cuda.sh` performs a no-peer check for its original host-exchange policy, whereas this preset explicitly enables peer transport. Upstream files describing no-peer behavior refer to that original default path.

## Enable the preset

Keep a working Strata 0.1.66 serve configuration for your own model, tokenizer, API authentication and optional vision encoder. Point its `exe` at the absolute path of the new `strata-split-2`, and `cwd` at the executable's directory. Merge the `env` object from [the preset](../presets/turing-dual-2080ti.json) into that configuration. Paths below are local paths that each installation supplies:

```text
FNX_RCCL_LIB=/absolute/path/to/libnccl.so.2
FNX_EXPERT_PROFILE=/absolute/path/to/strata-turing/engine/src/split/data/expert-routes.bin
FNX_CONVERT_SPILL_DIR=/absolute/path/to/a/writable/spill-directory
FNX_ROUTE_FILE=/absolute/path/to/a/writable/observed-routes.bin
```

`FNX_EXPERT_PROFILE` is the initial expert ranking; `FNX_ROUTE_FILE` is an output file recording observed routes. Create the spill directory. Set the serve configuration's `lib_dirs` to your CUDA runtime library directory and any additional runtime directories your installation needs. The tested visible device order was `1,0`, included in the preset.

Set `drafting` to `mtp`, `parallel` to `1`, and use these engine arguments, substituting your existing native and MTP paths:

```text
--native /path/to/native-model \
--mtp /path/to/mtp-directory \
--draft-vocab /path/to/mtp-directory/draft_vocab.bin \
--max-context 262144 --spec 6 --lookup 15 \
--conversation-cache-mib 0 --batch 1
```

Start your usual Strata serve entry point with the resulting configuration, for example:

```sh
python terminal/strata.py serve --config /path/to/your-config.json --host 127.0.0.1 --port 8080
```

The served API retains the upstream controller and image encoder. The live deployment passed image OCR, shape/color recognition, two-image ordering, and a text request after image requests. Its encoder remained on CPU with a limit of 300 visual tokens per image.

## Validation and limits

Passed on the stated hardware and selected configuration:

- Three fresh optimized/original 32K/4K pairs, with all measurements retained.
- Selected-policy numerical shadow checks: per card, 34,752 exact GU/down comparisons, 2,043 TopK comparisons, 55,210 HC checks and 734,664 attention heads, with zero failures at the original thresholds.
- 6,000 CPU probability-law checks; deliberately using stale proposal probabilities was detected in 4,925 negative cases.
- Actual 32K input / 10,240 output context growth and reclamation: growth at 41,280 positions, ten expert-payload byte comparisons, up to 5,921 experts checked per card, K/V pages returned, and a fresh 8K request after growth.
- Production health, actual loaded binary/runtime identity, image requests, and the final 280W 32K/4K run.

The three selected 4K outputs and the 280W output have the same hash. This does not require equality with the original engine's generated text. The user permitted small floating-point differences and different generation trajectories while preserving weight precision, target sampling and numerical validation.

Earlier candidates and failures are summarized in the benchmark evidence. The raw GU fast path that failed the downstream error gate was rejected. MTP graph I/O, verifier candidate caching and a TopK call graph were explored and retired; their prototypes are outside the accepted engine patch. No numerical threshold was relaxed to reach the throughput target.

Other CUDA architectures, HIP, SYCL, Metal and Windows have not been built or validated for this change set. Three/four-GPU execution and the standard 11 GiB 2080 Ti are outside the measured scope. The changes remain opt-in. This fork is an experimental source publication, with no upstream PR opened.

The original prompt source is not bundled. [benchmark.py](../benchmarks/turing-sm75/benchmark.py) accepts a local UTF-8 `--source` fixture and constructs an exact token-count request; reproducing the historical output hash also requires the historical prompt content, weights, tokenizer and runtime settings.

## Provenance

Upstream Strata and sFNX notices and their MIT licensing are retained. [strata-ampere](https://github.com/zongxinqian/strata-ampere) and [vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) were consulted for ideas; the dense-model project's implementation was not transplanted into this MoE engine. Investigation and implementation were AI-assisted. Performance and validation claims here come from actual executions on the stated hardware.
