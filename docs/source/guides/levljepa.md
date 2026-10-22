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

## Framework reuse

`LeVLJEPA` inherits `spt.Module` and supplies the method forward function;
training, optimization and checkpoint resume use the existing Module/Manager
path. Projection and predictor heads use the framework's `backbone.MLP`
with BatchNorm/GELU. Zero-dropout modules are omitted to retain the existing
LeVLJEPA checkpoint parameter names. Three state-layout regression cases
cover linear, one-hidden-layer and two-hidden-layer heads.

The example's `LossHistory` callback records epoch component means for its
JSON report and updates the caption dataset epoch. It does not stop training,
replace EarlyStopping, implement optimization, or implement checkpoint saving.
The small dataset adapter supplies Flickr8k-specific EOS, caption selection
and image-identity checks. Its CLIP control reuses the paired encoder forward
and existing `CLIPLoss`; no separate training framework is introduced.

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
1,000 validation and 1,000 test images, with five caption columns. The earlier reduced
experiments use only `caption_0` and seeded subsets of the existing train and
validation splits, without using the test split. The full-data follow-up below
also evaluates an independent test split. No new dataset wrapper is added
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
shards with the same schema; `--train-parquet` can take multiple shard paths.
`--tokenizer` also accepts a local directory.

By default, training uses `caption_0`. With `--caption-mode random`, each
training image uses one of its five captions per epoch, seeded by the run seed,
epoch and image index. Caption choices are independent of model RNG consumption
and repeat across objectives for identical image ordering. Validation keeps
`caption_0`; image identity, rather than caption rows, defines the data split.
The example uses zero-worker data loaders so epoch updates reach the dataset.

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

### Earlier recorded reduced-model results

The supplied earlier records describe CPU runs with 1,024 training images, 256 held-out validation images, a
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

### Fresh paired-control validation

Four new CPU runs compare LeVLJEPA with a CLIP control on the same seeded
subsets of the first pinned Flickr8k training shard and the existing validation
split. Requesting 2,048 training images leaves 2,047 after exact deduplication;
each run uses 512 validation images, batch size 64, and 50 epochs / 1,550
optimizer steps. Seeds 42 and 43 each use identical data across methods.
No pretrained model weights are used.

The control uses the same seeded ViT/GPT-2 encoders and marginal projection
heads, AdamW settings, and augmentation recipe. It omits LeVLJEPA-specific
predictors: parameter counts are 5,525,376 for LeVLJEPA and 5,375,424 for CLIP.
CLIP uses the existing symmetric `CLIPLoss` at temperature 0.07. Retrieval is
predictor-to-target for LeVLJEPA and embedding-to-embedding for CLIP. The
objectives have different loss scales, so training losses are not comparable
between methods. No hyperparameter search was performed.

| Method | Seed | Image→text R@1, before→after | Text→image R@1, before→after | Image→text R@5, after | Text→image R@5, after | First→last loss |
|---|---|---|---|---|---|---|
| LeVLJEPA | 42 | 0.20% → 0.59% | 0.20% → 0.59% | 2.73% | 2.34% | 0.8050 → 0.4475 |
| CLIP | 42 | 0.20% → 0.98% | 0.00% → 0.59% | 2.34% | 2.34% | 4.4576 → 0.2256 |
| LeVLJEPA | 43 | 0.39% → 0.20% | 0.20% → 0.39% | 2.34% | 1.56% | 0.8050 → 0.4290 |
| CLIP | 43 | 0.59% → 1.37% | 0.20% → 0.59% | 3.52% | 2.54% | 4.4605 → 0.2390 |

Random R@1 is approximately 0.196% on this gallery (identical caption strings
are interchangeable positives). LeVLJEPA remains weak on held-out retrieval;
seed 43 image-to-text R@1 declines to the random level. The control also
generalizes poorly, so this small experiment cannot establish a general
ranking of the methods. Lower training loss is not evidence of strong retrieval.

Reloading every final checkpoint reproduced its stored validation metrics.
Evaluation on the training images, with deterministic resizing, shows the gap:

| Method | Seed | Training image→text R@1 | Training text→image R@1 |
|---|---|---|---|
| LeVLJEPA | 42 | 3.91% | 23.74% |
| CLIP | 42 | 62.48% | 69.52% |
| LeVLJEPA | 43 | 5.23% | 25.31% |
| CLIP | 43 | 60.58% | 66.98% |

Training-gallery results are not held-out performance. Their much higher values
show substantial overfitting under this reduced-data, from-scratch recipe.

The fresh LeVLJEPA seed-42 checkpoint was resumed from step 1,550 to 1,581
for one additional epoch. Optimizer, both SIGReg, and all BatchNorm counters
reached 1,581; the loaded model reproduced the pre-resume validation metrics.
This verifies state continuity, not a bit-identical stochastic training path.

Reproduce each method/seed pair using the same local pinned shards:

```bash
python examples/levljepa_flickr8k.py \
  --method levljepa --seed 42 \
  --train-parquet /path/to/train-00000-of-00002-2f8f6bfa852eac4b.parquet \
  --val-parquet /path/to/validation-00000-of-00001-7025a2b596f14b7b.parquet \
  --tokenizer /path/to/gpt2-tokenizer \
  --train-samples 2048 --val-samples 512 --batch-size 64 \
  --epochs 50 --threads 2 --accelerator cpu \
  --output /path/outside/test-working-directory/levljepa-seed42
```

Repeat with `--method clip` and seeds 42/43, using separate output directories.
Repository tests delete `outputs/` relative to their working directory. Keep
retained experiment artifacts outside that path.

### Full Flickr8k follow-up

Four further CPU runs use both training shards: 5,999 unique training
images after exact deduplication, 1,000 validation images, and
1,000 independent test images after the fixed exact-image overlap and
deduplication checks. Training samples one of the five captions for each
visited image per epoch with `--caption-mode random`. Each epoch drops the
incomplete final training batch. Seeds 42/43 use the same reduced architecture,
optimizer and batch size as the paired control above, with 50 fixed epochs /
4,650 optimizer steps. No pretrained weights, hyperparameter search or
test-driven checkpoint selection are used.

Validation keeps `caption_0`. After all four runs finish, the original final
checkpoints are evaluated on all five test captions per image. Image-to-text
retrieval has 5,000 text candidates and five owner-image positives;
text-to-image has 1,000 image candidates and one owner-image positive.
Random R@1 is approximately 0.100% in either direction.

| Method | Seed | Test image→text R@1 | Test text→image R@1 | Test image→text R@5 | Test text→image R@5 |
|---|---|---|---|---|---|
| levljepa | 42 | 1.30% | 0.70% | 5.60% | 3.92% |
| clip | 42 | 2.20% | 2.16% | 9.50% | 8.42% |
| levljepa | 43 | 1.20% | 1.00% | 4.30% | 4.26% |
| clip | 43 | 1.60% | 1.90% | 6.70% | 7.58% |

Every final checkpoint reproduces its stored validation metrics after reload.
Full-data LeVLJEPA seed 42 also resumes from
4,650 to 4,743 steps,
with finite state and matching optimizer, SIGReg and BatchNorm counters.
These runs change both image count and caption diversity, so they do not
isolate either factor. The previous 512-image and current 1,000-image
validation galleries also differ. Two seeds and reduced, randomly initialized
encoders do not establish paper-scale quality or general superiority.

Reproduce each fixed method/seed training run with the complete pinned data:

```bash
python examples/levljepa_flickr8k.py \
  --method levljepa --seed 42 --caption-mode random \
  --train-samples 6000 --val-samples 1000 --batch-size 64 \
  --epochs 50 --threads 2 --accelerator cpu \
  --output /path/outside/test-working-directory/full-levljepa-seed42
```

Repeat with `--method clip` and seeds 42/43, using separate output directories.
The example saves validation metrics; the five-caption test numbers above
come from a separate final-checkpoint audit using the stated retrieval protocol.

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

Targeted verification includes 18 formula/API unit cases, three head checkpoint
layout cases, five existing CLIP
loss cases, two Manager training/full-resume cases (tiny and real ViT/GPT-2
encoders), two real two-process CPU gloo comparisons, and an Apple MPS
real-encoder optimizer test: 31 tests on the Mac. The MPS case skips on hosts
without MPS. The real-encoder cases use reduced BatchNorm/GELU heads. In the
DDP reference, each rank's batch passes through the heads separately before
concatenating features, matching ordinary DDP's rank-local BatchNorm semantics.
The original linear-head case still compares a full concatenated-batch forward.

The CPU gloo gradient tests passed on macOS with local sockets enabled and
`GLOO_SOCKET_IFNAME=lo0`. Apple MPS forward/backward and optimizer updates also
passed with real encoders and BatchNorm heads. The preceding 28-case selection
on Linux aarch64 reported 27 passed and one expected MPS skip; both real
two-process CPU gloo comparisons passed with `GLOO_SOCKET_IFNAME=lo`.
CUDA/NCCL, multi-node training, and synchronized BatchNorm remain unverified.

The complete unit selection reported 3,765 passed, 20 skipped, and three failed.
An untouched checkout of upstream commit `ab836bf69` in the same environment
reported 3,747 passed, 20 skipped, and the same three failures.
Two DINO visualization failures (MPS/CPU device mismatch) reproduce against
untouched upstream source. The third is a fork/autograd threading failure in
`test_latent_viz`; it passes alone on both branches, but the same failure can
be reproduced on upstream after priming multithreaded autograd. These results
do not constitute an all-green repository suite. Non-unit tests outside the
focused LeVLJEPA integration cases were not run.

All-files pre-commit passed in an isolated copy. Sphinx HTML is generated with
gallery execution disabled; existing docstring/cross-reference warnings remain.
The earlier 1,024/256-image results and step-448-to-960 resume above originate
from supplied records. The new 2,047/512-image control and step-1,550-to-1,581
resume were executed locally and audited from their newly generated checkpoints.

## Sources

- [Official LeVLJEPA repository](https://github.com/MLO-lab/LeVLJEPA)
- [Reference forward at the inspected revision](https://github.com/MLO-lab/LeVLJEPA/blob/1fd6545c38407f0ab98af3e59070df483914e92b/forwards.py)
- [Reference architecture](https://github.com/MLO-lab/LeVLJEPA/blob/1fd6545c38407f0ab98af3e59070df483914e92b/main.py)
- [Reference SIGReg](https://github.com/MLO-lab/LeVLJEPA/blob/1fd6545c38407f0ab98af3e59070df483914e92b/utils/sigreg.py)
- [Pinned Flickr8k dataset card](https://huggingface.co/datasets/jxie/flickr8k/blob/56f58c967835f7c508d684f36bd7897cca9d7634/README.md)
