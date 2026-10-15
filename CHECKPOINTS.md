# Checkpoint inventory

The paths below are relative to this repository. All adapter SHA-256 values refer to `adapter_model.safetensors`; they let users verify the files without depending on the original training machine.

| Architecture | Expected base model | Snapshot / note | Default Stop threshold |
|---|---|---|---:|
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | `2.0` |
| InternVL3-8B | `OpenGVLab/InternVL3-8B-hf` | `259a3b64a14623c0ec91a045cb43f7c5af5fa6af` (Transformers-native `InternVLForConditionalGeneration`) | `0.0` |

| Architecture | Component | Path | SHA-256 |
|---|---|---|---|
| Qwen3-VL-8B | Foundation | `checkpoints/qwen3-vl-8b/foundation` | `b310a127e1095c63f72d461c92b01b0735534005aa4c2125cb52e13b578ffbb0` |
| Qwen3-VL-8B | Move no-KD | `checkpoints/qwen3-vl-8b/move_no_kd` | `cd5a8093289247d23790a7f1a4f262f84755fa5fe7bb4855c1f9fe54c5a0ccee` |
| Qwen3-VL-8B | Move OPD delta | `checkpoints/qwen3-vl-8b/move_opd` | `7718342d1a96b70f77584db285070094d6c455a4c2debb677ea3dc024cd1fd6c` |
| Qwen3-VL-8B | Move composed (evaluation) | `checkpoints/qwen3-vl-8b/move_composed` | `330f86c3f04e2c92ecca3200528b56d85ab633c7fb0fc1276fce9136b74193e1` |
| Qwen3-VL-8B | Stop adapter | `checkpoints/qwen3-vl-8b/stop` | `6ba37c3d2dc0fc9edd0eaba6a5ecb4b9fd4cf95630b7efc8d8798188f6ee3eb2` |
| Qwen3-VL-8B | Stop merged (evaluation) | `checkpoints/qwen3-vl-8b/stop_merged` | Five shard hashes listed below |
| InternVL3-8B | Foundation | `checkpoints/internvl3-8b/foundation` | `9d9b320932212ba72af522085dcfc1bd97771b328e56d911e8e587b3c41e0b3e` |
| InternVL3-8B | Move composed (evaluation) | `checkpoints/internvl3-8b/move_composed` | `e55900b38b7e1279fa26eb493f572ffbb69902850e08d29688fc06f1d965fcea` |
| InternVL3-8B | Stop adapter | `checkpoints/internvl3-8b/stop` | `84b6cac6cdb5a922299624789a450e86cdfca23fb4ef8e4fc27fbf47c1104258` |
| InternVL3-8B | Stop merged (evaluation) | `checkpoints/internvl3-8b/stop_merged` | Five shard hashes listed below |

SHA-256 for each Qwen merged Stop weight shard:

| File | SHA-256 |
|---|---|
| `model-00001-of-00005.safetensors` | `55fe94eaff8c2767decdb14a96190b3611b5d5ff4e54d71866b96ae072c236cc` |
| `model-00002-of-00005.safetensors` | `26bb5a89a59a92a1fa433d659fcfc6feab5135964f8cc49b8f759ebbf2134468` |
| `model-00003-of-00005.safetensors` | `5f57c50c07e3c515d87d6bf8ca1059f2bd4b342534c7a754ec8bff572ce2048c` |
| `model-00004-of-00005.safetensors` | `054bd81ada9c7483b2d8ffc4a021e70ca0c1d8b05d4610ada716c0f8e1ad90e1` |
| `model-00005-of-00005.safetensors` | `5e23f8f757e9b537e00ade40843e66a91ff573c4104184083e18bbd0a44e7a6c` |

SHA-256 for each InternVL merged Stop weight shard (`sha256sum checkpoints/internvl3-8b/stop_merged/model-*.safetensors`):

| File | SHA-256 |
|---|---|
| `model-00001-of-00005.safetensors` | `7690a43fc4c284df55925fd1f70bf6122cbfe07e275bfcab525cdb45a8cb86e8` |
| `model-00002-of-00005.safetensors` | `d5128ee74231fd03c608e9fb0a4bfdcda8e8c3dea5592eadcc62c9251cd86a29` |
| `model-00003-of-00005.safetensors` | `7db527eae49f49610ed9f51bfe478680be6ca051164891b265af37ffd40743e1` |
| `model-00004-of-00005.safetensors` | `626d7199ffafb0a98492471f29a8309ab8b7ebf667ee9c7da932610af8dabcdb` |
| `model-00005-of-00005.safetensors` | `6814f162cc1ca061db92cb8808c7202876d8c1a56a0e7e856db12b21cba23a28` |

Both architectures default to `move_composed` + `stop_merged`, matching the selected historical evaluation. `stop_merged` is a complete Hugging Face model, not a LoRA adapter. The adapter `stop` is retained as the training artifact. Each PEFT config points to its public base-model identifier; pass a local base-model directory to `evaluate.py` for offline runs.

These artifacts are the selected release checkpoints, not a guarantee that rerunning evaluation on a different simulator/data installation yields identical metrics. Preserve evaluation logs and exact environment versions when publishing numbers.

The InternVL3-8B `move_composed` adapter is directly loadable for evaluation. Its two pre-composition Move source adapters are not bundled because InternVL3 training is outside this release's scope. Qwen3-VL-8B includes both source adapters and its composition manifest can be reproduced immediately.

The pinned InternVL3 base has the Transformers-native `InternVLForConditionalGeneration` configuration and processor layout. The similarly named `OpenGVLab/InternVL3-8B` has the different `InternVLChatModel` custom-code architecture and must not be substituted.
