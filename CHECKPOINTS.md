# Anonymous checkpoint inventory

The paths below are relative to the external checkpoint bundle. The Git
repository intentionally contains no large model files. The anonymous base
weights are available at [https://anonymous-hf.com/a/yd05u6f8p76h/](https://anonymous-hf.com/a/yd05u6f8p76h/); that mirror contains the two base
model directories and does not claim to contain the trained checkpoint
components listed below.

| Architecture | Base model | Revision | Default Stop threshold |
|---|---|---|---:|
| Qwen3-VL-8B | Qwen/Qwen3-VL-8B-Instruct | 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b | 2.0 |
| InternVL3-8B | OpenGVLab/InternVL3-8B-hf | 259a3b64a14623c0ec91a045cb43f7c5af5fa6af | 0.0 |

Expected relative components:

- qwen3-vl-8b/foundation
- qwen3-vl-8b/move_no_kd
- qwen3-vl-8b/move_opd
- qwen3-vl-8b/move_composed
- qwen3-vl-8b/stop
- qwen3-vl-8b/stop_merged
- internvl3-8b/foundation
- internvl3-8b/move_composed
- internvl3-8b/stop
- internvl3-8b/stop_merged

The external artifact manifest records SHA-256 values for every adapter and
merged shard. The release checkpoint selection is historical and does not
guarantee bit-identical metrics on a different simulator or asset install.
Preserve the exact environment and evaluation logs when reporting numbers.

The Qwen3 merged Stop directory is a complete Transformers model. The Stop
adapter is retained separately as a training artifact. The InternVL3 Move
adapter is directly loadable for evaluation; its pre-composition training
adapters are outside this release scope.
