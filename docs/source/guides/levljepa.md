# LeVLJEPA integration and validation

This implementation adds **LeVLJEPA: End-to-End Vision-Language Pretraining
Without Negatives** as a reusable method and composable forward function.
It follows the [official implementation](https://github.com/MLO-lab/LeVLJEPA)
at revision `1fd6545c38407f0ab98af3e59070df483914e92b`.

## Scope and implementation plan

| Requirement | Implementation | Verification |
|---|---|---|
| Image and text encoders | User-supplied modules; no compulsory model downloads | Offline tiny encoders, real ViT/GPT-2 example |
| Separate marginal projections | BN/GELU hidden layers and linear outputs | Feature shape and encoder gradient tests |
| Bidirectional prediction | Vision predicts detached text; text predicts detached vision | Independent objective/gradient comparison |
| Anti-collapse regularization | Sliced Epps–Pulley SIGReg on each marginal | Quadrature formula, collapsed inputs, bf16 tests |
| Framework integration | `LeVLJEPA(Module)`, `forward.levljepa`, `LeVLJEPALoss` | Dict API, composable-forward equivalence, exports |
| Training and resume | Standard Module optimizer configuration and Manager | Real optimizer steps and full checkpoint resume |
| Distributed behavior | Shared projection directions; differentiable global moments | Two-process gloo gradient comparison test |
| Data and evaluation | Existing Flickr8k image-level train/validation splits | Exact image overlap rejection; held-out retrieval |
| Documentation | Method catalog, API exports and this guide | Ruff, import tests and diff checks |

The reduced example does not reproduce DataComp-scale pretraining, released
checkpoint quality, segmentation, or VLM downstream experiments. These are
outside the scope of a small library integration validation.

## Objective and gradients

For image/text encoder outputs, separate pre-projection heads produce `v` and
`t`. The vision predictor `p_v(v)` has the width of `t`, and the text predictor
`p_t(t)` has the width of `v`:

```text
prediction = (MSE(p_v(v), stopgrad(t)) + MSE(p_t(t), stopgrad(v))) / 2
loss = (1 - lambda_vision - lambda_text) * prediction
       + lambda_vision * SIGReg(v) + lambda_text * SIGReg(t)
```

The default weights are 0.02 per modality. An optional cosine alignment uses
the reference's `2 - 2*cos(prediction, target)` formula. Targets are detached
only for prediction; SIGReg continues to train both encoders. No negative
pairs, temperature, teacher, or EMA update is introduced.

The text encoder must return `[B, L, D]` hidden states (tensor, dictionary or
HuggingFace output). Pooling selects the last unmasked position, including
left-padded input. The example explicitly appends EOS before padding. Empty
text masks fail clearly instead of pooling padding.

SIGReg uses random unit directions, a 17-point grid on `[0, 3]`, Gaussian
characteristic-function weighting, and the two-sided trapezoidal statistic
from the reference. It runs in fp32 under mixed precision. The random sequence
counter is checkpointed, advances during training only, and does not depend on
unrelated calls to the process RNG.

Under DDP, all ranks share directions and differentiably sum characteristic
function moments before forming the statistic. Global moments are normalized
by the true global count. The statistic is scaled by `global_count/world_size`,
matching the reference's local-batch scale for equal batches. Use equal batch
sizes on all ranks (`drop_last=True`) for the full objective's local means and
DDP gradient averaging. All ranks must enter paired loss evaluation together;
use `stage="predict"` for inference without loss collectives.

The existing `methods.lejepa.SlicedEppsPulley` is not changed. Its construction-
time distributed detection and collective-return handling cannot provide the
runtime and autograd behavior needed here; a private regularizer implements
the same statistic with explicit distributed semantics.

## API

```python
import stable_pretraining as spt

model = spt.LeVLJEPA(
    vision_encoder=vision_encoder,  # returns [B, vision_dim]
    text_encoder=text_encoder,      # returns last_hidden_state [B, L, text_dim]
    vision_dim=192,
    text_dim=192,
    embedding_dim=192,
    optim={"optimizer": {"type": "AdamW", "lr": 2e-4}},
)
output = model(batch, stage="fit")
output["loss"].backward()
```

The model returns a dict preserving batch metadata. `embedding` is the raw
vision feature for probes; `image_embedding` and `text_embedding` are the
regularized spaces. `image_prediction` means prediction **from image to text**,
and `text_prediction` means prediction **from text to image**. Loss components
are separately available. Supported stages are `fit`/`train`, `validate`/`val`,
`test`, and `predict`. Call `model.eval()` for inference, as usual.

To compose `spt.Module` directly, use `forward=spt.forward.levljepa` and attach
`vision_encoder`, `text_encoder`, `vision_pre_proj`, `text_pre_proj`,
`projector_vision`, `projector_text`, and `levljepa_loss` as attributes.

## Existing data and reduced training recipe

The example uses [jxie/flickr8k](https://huggingface.co/datasets/jxie/flickr8k),
revision `56f58c967835f7c508d684f36bd7897cca9d7634`. It contains 6,000 train,
1,000 validation and 1,000 test images, with five caption columns. The small
experiment uses only `caption_0` and seeded subsets of the existing train and
validation splits. It does not use the test split or add a new dataset wrapper
to `stable-datasets`. Exact pixel-duplicate images are removed within subsets,
and any exact train/validation image overlap raises an error. Near-duplicates
are not exhaustively detected. The dataset mirror's card is incomplete; use
the original Flickr8k terms for any redistribution. No dataset is bundled here.

The model uses randomly initialized reduced ViT and GPT-2 encoders, smaller
projection/predictor heads, 64-pixel inputs, and shorter captions. This retains
the method's two encoder families and objective while keeping local validation
practical. It is not the published tiny/base architecture recipe.

```bash
pip install -e . pytest ruff
python examples/levljepa_flickr8k.py \
  --train-samples 1024 --val-samples 256 --epochs 30 --seed 42 \
  --accelerator cpu --output outputs/levljepa-flickr8k-seed42
```

The first run downloads the existing dataset and GPT-2 **tokenizer**, not model
weights. `--train-parquet` and `--val-parquet` accept already downloaded Flickr8k
shards with the same schema; `--tokenizer` also accepts a local directory.

The script records before/after image-to-text and text-to-image retrieval R@1
and R@5 over the full validation gallery, random R@1, feature standard
deviations, epoch loss components, configuration, selected image hashes,
software versions, and the final step in `results.json`. Retrieval compares
each predictor to targets in the opposite modality. Identical caption strings
are treated as interchangeable positives. No retrieval negatives enter the
training objective. A full local checkpoint is saved as `final.ckpt`.

```bash
python examples/levljepa_flickr8k.py \
  --train-samples 1024 --val-samples 256 --epochs 30 --seed 42 \
  --resume /absolute/path/to/final.ckpt \
  --output outputs/levljepa-flickr8k-resumed
```

Keep the architecture/data arguments unchanged when resuming. Manager restores
optimizer state and the step counter; the example's before metric then measures
the loaded model. Resume is not claimed to reproduce stochastic augmentations
bit-for-bit across restarts.

## Verification commands

### Local reduced-model results

CPU runs used 1,024 training images, 256 held-out validation images, a
5,525,376-parameter model (width 96, two encoder layers), batch size 32,
128 SIGReg projections, and 30 epochs / 960 optimizer steps. All encoders
were randomly initialized. Retrieval percentages below use 256 candidates;
random R@1 is 0.39%. Each seed selects its own seeded subset, so variation
includes both initialization and subset selection.

| Seed | Image→text R@1, before→after | Text→image R@1, before→after | Image→text R@5, before→after | Text→image R@5, before→after | Mean epoch loss, first→last |
|---|---|---|---|---|---|
| 42 | 0.39% → 1.56% | 0.39% → 0.78% | 1.56% → 2.73% | 1.56% → 4.69% | 0.5890 → 0.3645 |
| 43 | 0.39% → 1.56% | 0.78% → 0.39% | 1.95% → 4.30% | 2.34% → 1.95% | 0.5967 → 0.3445 |

Mean per-feature validation standard deviations after training were seed 42: vision 0.906, text 0.881; seed 43: vision 0.933, text 0.897.
These are reduced-data integration experiments; retrieval quality should be
judged from the numbers above, not inferred from a finite or decreasing loss.
They do not reproduce paper-scale quality. Final optimizer, SIGReg and
BatchNorm batch counters were all 960. Evaluation restores the prior model
mode, so initial retrieval evaluation cannot leave training in eval mode.

### Checks

```bash
python -m pytest stable_pretraining/tests/unit/test_levljepa.py -q
python -m pytest stable_pretraining/tests/integration/test_levljepa_training.py -q
ruff check stable_pretraining/losses/levljepa.py \
  stable_pretraining/methods/levljepa.py
ruff check --select E4,E7,E9,F examples/levljepa_flickr8k.py
git diff --check
```

The two-process integration test uses CPU gloo with an actual DDP-wrapped model,
deliberately different rank RNG histories, and a concatenated-batch baseline.
It compares both the loss and every parameter gradient. It needs permission
to create local network sockets; a restricted runtime can block gloo before
the method is reached. Such a failure is **not** a distributed pass.

Local verification passed 18 new unit tests, five existing CLIP loss tests,
one Manager training/full-resume integration test, and the real two-process
CPU gloo test (25 checks total). The DDP test verifies loss and every parameter
gradient using tiny encoders and linear heads. It does not test distributed
BatchNorm heads or multi-node/GPU execution. An earlier restricted-runtime
attempt failed during gloo initialization; the successful run allowed local
sockets and selected the macOS loopback interface with `GLOO_SOCKET_IFNAME=lo0`.
Use an interface appropriate to your host when running this test.

Applicable pre-commit checks passed. Sphinx generated HTML with gallery execution
disabled, reporting 28 warnings including existing docstring formatting and
cross-reference issues and unavailable external inventories. This was not a
warning-free documentation build. The full repository test suite and GPU
execution were not tested. The Flickr8k results above are from the supplied
experiment records, not a fresh training run during local PR preparation.

The actual ViT/GPT-2 Flickr8k model was also resumed from a step-448 checkpoint
through step 960 using the original 30-epoch configuration. Final optimizer,
SIGReg and BatchNorm counters all reached 960. This checks full training-state
restoration; it does not assert an identical stochastic trajectory.

## Sources

- [Official LeVLJEPA repository](https://github.com/MLO-lab/LeVLJEPA)
- [Reference forward at the inspected revision](https://github.com/MLO-lab/LeVLJEPA/blob/1fd6545c38407f0ab98af3e59070df483914e92b/forwards.py)
- [Reference architecture](https://github.com/MLO-lab/LeVLJEPA/blob/1fd6545c38407f0ab98af3e59070df483914e92b/main.py)
- [Reference SIGReg](https://github.com/MLO-lab/LeVLJEPA/blob/1fd6545c38407f0ab98af3e59070df483914e92b/utils/sigreg.py)
- [Pinned Flickr8k dataset card](https://huggingface.co/datasets/jxie/flickr8k/blob/56f58c967835f7c508d684f36bd7897cca9d7634/README.md)
