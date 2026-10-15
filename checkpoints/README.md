# Spatial-Nav-100K checkpoints

These weights are loaded by `codes/scripts/evaluate.py`. The complete
environment and command are in [`REPRODUCTION.md`](../REPRODUCTION.md), and
the file/hash inventory is in [`CHECKPOINTS.md`](../CHECKPOINTS.md).

| Model | Move component | Stop component | Stop threshold |
|---|---|---|---:|
| Qwen3-VL-8B | `qwen3-vl-8b/move_composed` | `qwen3-vl-8b/stop_merged` | 2.0 |
| InternVL3-8B | `internvl3-8b/move_composed` | `internvl3-8b/stop_merged` | 0.0 |

The evaluator needs only the two components in the table. Qwen3 training is
included in this release; InternVL3 is provided as an evaluation checkpoint.
Use the matching pinned base model and `--variant checkpoint`.
