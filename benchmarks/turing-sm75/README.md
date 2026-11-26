# Turing SM75 measurements

These measurements were executed on two RTX 2080 Ti cards with 22 GiB each and NVLink, under Linux with CUDA 13.0. They cover Qwen3.8-Flash-Next IQ3_XXS and the selected MTP policy. See [build and configuration](../../docs/TURING.md) and [source provenance](provenance.json).

## Accepted 250W paired runs

Each pair uses 32,768 input / 4,096 output tokens and zero cached tokens. Raw HTTP usage, millisecond timings and output hashes are retained in the per-trial JSONL files. The output text and private server configuration are omitted.

| Trial | Optimized decode | Original decode | Ratio | Prefill ratio |
|---|---:|---:|---:|---:|
| [104937](trial-20261126-104937/tp-32k4k.jsonl) / [original](trial-20261126-104937/pp-32k4k.jsonl) | 110.3832 | 71.3992 | 1.546001 | 0.981156 |
| [105534](trial-20261126-105534/tp-32k4k.jsonl) / [original](trial-20261126-105534/pp-32k4k.jsonl) | 110.4016 | 73.4946 | 1.502174 | 0.975138 |
| [110128](trial-20261126-110128/tp-32k4k.jsonl) / [original](trial-20261126-110128/pp-32k4k.jsonl) | 110.5846 | 71.8130 | 1.539897 | 0.979691 |

The minimum decode ratio is 1.5021738 and minimum prefill ratio 0.9751384, recomputed from raw milliseconds in [the completion audit](goal-completion-audit.json). All three selected 4K output hashes match.

## Follow-up checks

- [Selected-policy numerical shadow](sampling-policy-selected-shadow-results.json): the accepted flags with diagnostics enabled and the original numerical thresholds.
- [Actual long-context reclamation](sampling-policy-reclaim-results.json): 32K/10K, actual growth, expert payload comparisons and post-growth reset.
- [280W online run](online-280w-final-results.json): prefill 1908.26 tok/s, decode 109.68 tok/s, total 54.636 seconds, matching 4K output. Its 250W comparison is historical, rather than a new paired run. [GPU telemetry](online-280w-32k4k.gpu.csv) has host identifiers removed and uses relative timestamps.
- [Image API checks](multimodal-production-check.json): two distinct OCR/color/shape fixtures, a two-image ordering request and a later text request. The [fixture images](vision-smoke) contain only synthetic test content.

## Earlier candidates and failures

[Sampling policy exploration](sampling-policy-results.json) retains all three candidate attempts. The first two failed the 1.5x paired decode target; the third selected row cost 2.0 and probability exponent 0.5. Earlier MTP graph-I/O and TopK graph experiments did not provide the required net speedup and were retired. A raw GU prototype failed the downstream numerical gate and was rejected; the accepted exact-pair GU keeps the original gate. GDN alternative partitions were slower and were not integrated. These outcomes limit the claims to the accepted configuration above.

## Run a workload

Install the upstream terminal's Python dependencies and provide a working local serve configuration. `benchmark.py` reads its credentials locally and emits only usage, timing and hashes unless `--save-text` is explicitly supplied. Run from the repository root:

```sh
PYTHONPATH="$PWD/terminal:$PWD/terminal/tools" python benchmarks/turing-sm75/benchmark.py \
  reproduction --config /path/to/your-config.json --port 8080 \
  --source /path/to/a/long-utf8-source.txt --prompt 32768 --output 4096 \
  --prompt-id trial-182625-32k4k
```

The supplied source must be long enough to construct the requested token count. The original historical prompt source is not bundled; a different source can reproduce the token-count workload but changes prompt content and the expected output hash. Use a fresh prompt context; the script asserts zero cached tokens.

The CPU probability-law test used during acceptance is included:

```sh
mkdir -p build-turing
c++ -std=c++17 -O2 -Iengine/src/split/src \
  benchmarks/turing-sm75/test_proposal_power.cpp -o build-turing/test_proposal_power
build-turing/test_proposal_power
```

The accepted test checked 6,000 cases and detected stale-q misuse in 4,925 negative cases. To generate new image fixtures, install Pillow and run `make_vision_smoke_images.py`; set `STRATA_TEST_FONT` to a suitable TrueType font path if DejaVu Sans Bold is unavailable.

Evidence was sanitized before publication: API keys, model paths, full launch configurations, recovery snapshots, host paths, process IDs and GPU UUIDs are excluded. This is an independently measured fork report; upstream benchmark-protocol approval has not been obtained.
