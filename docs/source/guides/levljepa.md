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
from the reference. It runs in fp32 under mixed precision; explicit half/bfloat16
module conversions reconstruct quadrature constants in fp32 rather than
upcasting already-rounded constants. The random sequence
counter is checkpointed, advances during training only, and does not depend on
unrelated calls to the process RNG.

Under DDP, all ranks share directions and differentiably sum characteristic
function moments before forming the statistic. Global moments are normalized
by the true global count. The statistic is scaled by `global_count/world_size`,
matching the reference's local-batch scale for equal batches. Use equal batch
sizes on all ranks (`drop_last=True`) for the full objective's local means and
DDP gradient averaging. Gradient accumulation averages microbatch objectives;
it does not enlarge the batch used to estimate each SIGReg statistic.
All ranks must enter paired loss evaluation together;
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
and existing `CLIPLoss`; no separate training framework is introduced. This is
a locally trained CLIP-style control, not teacher-provided scores or a pretrained
OpenAI CLIP checkpoint. The upstream CLIP loss source is unchanged. Both methods
use the same randomly initialized reduced encoders and update budget, although
LeVLJEPA adds cross-modal predictors and therefore has more parameters. The later
exploratory search used more LeVLJEPA candidates, so it is not an equal
hyperparameter-search-budget comparison.

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
each predictor to targets in the opposite modality. Positives are the paired
image identity; two different images sharing a caption are not merged. Earlier
first-caption experiment records used caption-string equality; their recorded
values are retained as historical results under that older rule. The full
1,000-image test has no duplicate first captions, and the five-caption test
already uses image-owner identities. No retrieval negatives enter the
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

### Validation-selected recipe diagnosis

A fixed 64-image, fixed-caption, no-augmentation diagnostic reached 100%
training-gallery R@1 in both directions after 600 optimizer steps for
LeVLJEPA with each per-modality SIGReg weight 0.02, 0.1 and 0.2, and for CLIP.
The randomly initialized reduced encoders used AdamW at lr=0.001, weight
decay=0.01, without scheduling. This direct-optimizer fitting diagnostic is
not a held-out result or a substitute for Manager/DDP integration tests.

A bounded validation search then used seed 42, 1,024 training candidates,
512 validation images and 30 epochs / 480 steps. Training captions were
sampled as above; validation retained caption_0. Stage one compared
lr={0.0002, 0.001} and LeVLJEPA per-modality lambda={0.02, 0.1}; CLIP used
the same learning-rate choices. All candidates added 5% linear warmup, cosine
decay to 10% of peak lr, and gradient-norm clipping at 1.0. Stage two retained
each method's best configuration and compared 64px resize and 96px crop
against the original 64px crop. Each stage's candidate budget and ranking rule
were recorded before running that stage: final-epoch mean bidirectional
validation R@1, then mean R@5; complete stage-two ties prefer the unchanged
recipe, then resize, then 96px. No best-epoch or test-based selection was used.

| Stage | Method | lr | SIGReg weight/modality | Input | Image→text R@1 | Text→image R@1 |
|---|---|---|---|---|---|---|
| 1 | levljepa | 0.0002 | 0.02 | 64px crop | 0.59% | 0.59% |
| 1 | levljepa | 0.0002 | 0.1 | 64px crop | 0.20% | 0.00% |
| 1 | levljepa | 0.001 | 0.02 | 64px crop | 0.20% | 0.78% |
| 1 | levljepa | 0.001 | 0.1 | 64px crop | 0.39% | 0.20% |
| 1 | clip | 0.0002 | — | 64px crop | 0.98% | 0.39% |
| 1 | clip | 0.001 | — | 64px crop | 0.98% | 0.59% |
| 2 | levljepa | 0.0002 | 0.02 | 64px resize | 0.59% | 0.39% |
| 2 | levljepa | 0.0002 | 0.02 | 96px crop | 0.59% | 0.00% |
| 2 | clip | 0.001 | — | 64px resize | 0.78% | 0.20% |
| 2 | clip | 0.001 | — | 96px crop | 0.98% | 0.39% |

The selected recipes were fixed before a further four full-data runs, each
with 5,999 training images, 1,000 validation images, seeds 42/43, 50 epochs
and 4,650 steps. Encoders remain randomly initialized width-96/depth-2 models.
The dataset, candidate gallery and caption protocol remain unchanged; input
resolution/augmentation and optimizer settings follow the selected recipe:

- levljepa: lr=0.0002, per-modality lambda=0.02 (used), 64px crop.
- clip: lr=0.001, per-modality lambda=0.02 (not used by CLIP), 64px crop.

| Method | Seed | Previous validation image→text R@1 | Current validation image→text R@1 | Previous validation text→image R@1 | Current validation text→image R@1 |
|---|---|---|---|---|---|
| levljepa | 42 | 1.00% | 2.20% | 1.30% | 1.40% |
| clip | 42 | 1.50% | 1.70% | 1.70% | 1.50% |
| levljepa | 43 | 1.10% | 1.50% | 0.80% | 1.10% |
| clip | 43 | 2.30% | 1.70% | 1.80% | 1.50% |

These are **validation-selected exploratory results**, not a new independent
test. The Flickr8k test set had already been evaluated in the preceding fixed
recipe experiments; those historical test results above remain unchanged.
No test examples or metrics were accessed again for this search. Validation
selection and two seeds do not establish significance, paper-scale quality or
method superiority. Small-set fitting establishes the ability to learn those
training pairs, not generalization. Each method selects its own configuration,
so this follow-up compares tuned recipes rather than isolating the loss alone.

All four final checkpoints reload strictly and reproduce their stored validation
metrics. Optimizer steps, BatchNorm/SIGReg counters, and scheduler steps agree
at 4,650; warmup lasts 232 steps and the final lr is 10% of peak.
Warmup and clipping use the existing framework scheduler and Trainer options;
no extra training loop or scheduling callback is added. Resume rejects a mismatch
between warmup and cosine-only checkpoint scheduling before loading data.
Saved scheduling horizons are restored; increasing epochs does not retime a
checkpointed schedule. Defaults of the original
example and method are retained. To reproduce, append the selected lr/input
settings plus `--warmup-fraction 0.05 --gradient-clip-val 1.0` to the full-data
commands above; set weights with `--lambda-vision` and `--lambda-text`, and
augmentation with `--image-augmentation crop|resize`.

### Additional alignment and readout diagnostics

After the initial search, a final bounded alignment check compared the official
optional cosine objective with the same four learning-rate/regularization pairs,
using the same seed-42, 1,024/512-image, 30-epoch budget. Selection retained
mean bidirectional R@1 then R@5; a complete tie favored the frozen MSE result.
Both the additional search and its conditional two-seed full-data follow-up
were declared before running this stage. Default MSE behavior is unchanged.

| Alignment | lr | SIGReg weight/modality | Validation image→text R@1 | Validation text→image R@1 |
|---|---|---|---|---|
| cosine | 0.0002 | 0.02 | 0.59% | 0.78% |
| cosine | 0.0002 | 0.1 | 0.20% | 0.20% |
| cosine | 0.001 | 0.02 | 0.20% | 0.20% |
| cosine | 0.001 | 0.1 | 0.39% | 0.20% |

The cosine candidate lr=0.0002, lambda=0.02 won the declared small-validation ranking. Two extra full-data runs use this fixed candidate, 50 epochs / 4,650 steps, seeds 42/43, and the same 1,000-image validation gallery:

| Alignment | Seed | Image→text validation R@1 | Text→image validation R@1 | Image→text validation R@5 | Text→image validation R@5 |
|---|---|---|---|---|---|
| cosine | 42 | 0.90% | 0.80% | 4.80% | 4.30% |
| cosine | 43 | 1.60% | 0.90% | 4.10% | 4.20% |

Both extra final checkpoints were strictly reloaded; saved validation metrics,
optimizer/SIGReg/BatchNorm counters and warmup scheduler state were verified.

A separate readout audit evaluated the reference's average of the two directional
cosine-score matrices on the historical full-data checkpoints. Scores below use
the same 1,000-image validation gallery; they do not replace historical test results.

| Seed | Directional image→text R@1 | Averaged image→text R@1 | Directional text→image R@1 | Averaged text→image R@1 |
|---|---|---|---|---|
| 42 | 1.00% | 0.80% | 1.30% | 1.60% |
| 43 | 1.10% | 1.20% | 0.80% | 0.60% |

The averaged readout does not improve consistently across seeds/directions, so
the example retains its directional readout. Historical LeVLJEPA marginal
effective ranks (exp entropy of normalized singular values after centering)
are about 40–43 at width 96, while predictor effective ranks are about 29–34.
Feature standard deviations near one therefore do not establish isotropy.
These measurements show nonconstant features, not ideal Gaussianity or strong
semantics. No new default or performance claim is based on this diagnostic.

Larger-dataset training remains unverified; all training here uses Flickr8k.
The frozen external Flickr30k transfer test below does not train on Flickr30k.

All recipe/readout follow-ups use validation data. The previously observed
Flickr8k test results remain historical; validation-selected gains must not be
reported as fresh test gains. Limited grid search, small validation samples and
two confirmation seeds do not establish an optimal recipe or significance.


The new warmup/clipping MSE checkpoint also resumed through the example CLI
from 4,650 to 4,743 steps. Before resuming, its original validation metrics
were reproduced; optimizer, BatchNorm and SIGReg counters agreed afterward.
The saved scheduler horizon (4,650) and warmup length (232) were preserved.
This extra epoch is a resume audit and does not replace the fixed-50-epoch
comparison. Exact continuation of data/augmentation RNG trajectories is not claimed.

### Checks

```bash
python -m pytest stable_pretraining/tests/unit/test_levljepa.py \
  stable_pretraining/tests/unit/test_clip_loss.py \
  stable_pretraining/tests/integration/test_levljepa_training.py -q
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
layout cases, two CLI resume-scheduler rejection cases, five existing CLIP
loss cases, four Manager training/full-resume cases (tiny and real ViT/GPT-2
encoders, each with default scheduling and explicit warmup/gradient clipping), two real two-process CPU gloo comparisons, and an Apple MPS
real-encoder optimizer test: 35 tests on the Mac. The MPS case skips on hosts
without MPS. The real-encoder cases use reduced BatchNorm/GELU heads. In the
DDP reference, each rank's batch passes through the heads separately before
concatenating features, matching ordinary DDP's rank-local BatchNorm semantics.
The original linear-head case still compares a full concatenated-batch forward.

The CPU gloo gradient tests passed on macOS with local sockets enabled and
`GLOO_SOCKET_IFNAME=lo0`. Apple MPS forward/backward and optimizer updates also
passed with real encoders and BatchNorm heads. The preceding 28-case selection
on Linux aarch64 reported 27 passed and one expected MPS skip; both real
two-process CPU gloo comparisons passed with `GLOO_SOCKET_IFNAME=lo`.
A later Linux CUDA selection passed both two-GPU NCCL numerical cases on
GeForce RTX 2080 Ti GPUs (PyTorch 2.14.0+cu126, NCCL 2.29.3): the injectable
linear encoders and real ViT/GPT-2 encoders with BatchNorm heads. Both use the
same reference construction as the CPU cases, compare loss with
`rtol=2e-5, atol=2e-6`, and compare every parameter gradient with
`rtol=2e-4, atol=2e-6`. There is no CPU/gloo fallback in these CUDA cases.
The preceding Mac targeted selection reports **35 passed, 2 expected CUDA
skips**. A subsequent Linux NCCL selection passed **4 tests, 7 deselected**:
the two numerical cases above, real-encoder `SyncBatchNorm` compared with a
global-batch reference, and native Manager two-GPU training/full-state resume.
The resume case advances from step 3 to step 5 and checks AdamW counters,
BatchNorm/SIGReg counters, the preplanned warmup/cosine scheduler, finite
encoder gradients and exact parameter equality between ranks. Private test
callbacks audit those states; production training uses the existing Manager
and Trainer without an additional training wrapper.

The same linear and real-encoder loss/every-parameter-gradient comparisons
also passed across **two different GPU nodes**, one RTX 2080 Ti per node,
using the actual NCCL backend over TCP (`NCCL_IB_DISABLE=1`). The initial
RDMA transport attempt failed with `ibv_modify_qp: Invalid argument`; it is
retained as a failed transport attempt, not counted as a pass. These
cross-node checks validate forward/backward numerical equivalence, not a
multi-node dataset training run or RDMA support.
Run the NCCL cases inside an allocation exposing at least two CUDA GPUs:

```bash
python -m pytest stable_pretraining/tests/integration/test_levljepa_training.py \
  -k nccl -vv -ra
```

A follow-up complete unit selection passed **3,773 tests**, with 20 skips
and 172 non-unit cases deselected. Before portability fixes, this environment
reported 3,765 passes and three failures; untouched upstream `ab836bf69`
reproduced the same three failures. The DINO test stub now creates noise on
the feature device, allowing the existing assertions to run on MPS. The
LatentViz CPU DDP test now launches its original fork-based training in a
fresh interpreter, avoiding autograd threads inherited from preceding tests;
its optimizer, exported-coordinate and cross-rank parameter assertions remain.
These changes repair test portability rather than alter method or callback
behavior. They can be reviewed separately from the LeVLJEPA changes.

A later full repository attempt collected 3,968 cases and reported **3,926
passed, 39 skipped, 3 failed**. The three failures were in unchanged upstream
test setup: an unattached Trainer in the FSDP2 custom-hook fixture, a stale
strict-xfail marker after the collectives had been fixed, and a local dataset
transform that macOS spawn workers could not pickle. The fixtures now attach
a real Trainer, retain the collective assertions without the stale xfail,
and use a module-level transform with two actual DataLoader workers. Both
complete affected files then passed **32 tests, 2 CUDA skips**. The distributed
test harness now also rejects timed-out or abnormally exited workers, with
both failure paths checked explicitly. Installing optional TensorBoard and
imageio/FFmpeg dependencies allowed two additional previously skipped tests
to pass. These repairs change test infrastructure, not production FSDP or
dataset behavior; they can be reviewed separately from the method.

A further GPU audit corrected two errors in an upstream FSDP2 reference test:
FSDP already averages gradients, so the local mean loss must not be divided
by world size again; `DTensor.full_tensor()` must run on every participating
rank. Both FSDP2 GPU optimizer-update and full-batch-reference cases then
passed with the existing parameter tolerances. Production FSDP behavior was
not changed.

After those repairs and optional dependency installation, an **unfiltered
macOS repository run passed 3,931 tests, with 38 skips and no failures** (3,969
collected). It includes integration, regression and download cases as well as
unit tests. CUDA requirements, optional/hardware-specific paths and intentional
upstream skips remain skips in that Mac result; separate Linux GPU results
must not be added to it as one all-platform total. The new native NCCL Manager
resume case was added after the first full collection and is included in the
final collection.

A separate real-data NCCL run uses the same 5,999-image Flickr8k training set
and 1,000-image validation set, with two GPUs, per-rank batch 64 and global
batch 128. It completes two epochs / 92 steps, saves a full checkpoint, then
resumes to the preplanned third epoch / 138 steps. Both ranks pass the same
optimizer, BatchNorm/SIGReg, scheduler and exact parameter-parity audits as
the synthetic integration case. With sampler/drop-last behavior, 46 updates
consume 5,888 training pairs per epoch; 5,999 is the dataset cardinality.
This three-epoch run verifies actual data training and state restoration, not
50-epoch multi-GPU performance or improved retrieval.

Two later upstream GPU fixtures also needed correction: environment metadata
is stored in `cuda.nvidia_smi_output`, not a nonexistent `pytorch` field; Slurm
requeue requires `SLURM_RESTART_COUNT >= 1` and uses zero-based checkpoint
epochs. The repaired environment test preserves GPU count/name assertions and
checks an actual two-rank CUDA Trainer. The requeue test additionally checks
the restored initial step, exactly the remaining epoch/batches, the original
run directory and exact final step. Both cases passed on both CUDA ranks;
two affected single-GPU cases also passed. Their affected Mac files separately
reported **26 passed, 4 hardware skips**. This simulates restart metadata,
not an actual scheduler preemption/requeue event.

A separate remaining-GPU selection reported **21 passed, 3 skipped**. The
skips are Hopper-only behavior on non-Hopper hardware and two optional
`mamba-ssm` CUDA-kernel comparisons without the extension. Intentional upstream
skips also remain unexecuted. The full Mac run precedes the last two fixture
repairs, whose affected files were verified separately as above.

The whole-repository passing record uses separately reviewable upstream test
repairs in the local validation checkout. A method-only contribution does not
include those repairs and must not claim to fix unrelated upstream failures.
The method-focused Mac coverage is **35 passed, 4 CUDA skips**, and its four
NCCL cases pass independently of the unrelated test-fixture changes.

All-files pre-commit passed in an isolated copy. Sphinx HTML is generated with
gallery execution disabled; existing docstring/cross-reference warnings remain.
The final local HTML build disables external intersphinx inventories; external
cross-references are not verified by that offline build.
The earlier 1,024/256-image results and step-448-to-960 resume above originate
from supplied records. The new 2,047/512-image control and step-1,550-to-1,581
resume were executed locally and audited from their newly generated checkpoints.

## Frozen external transfer test and CUDA training

A separate single-TITAN-RTX run completed the fixed MSE recipe for seed 42:
50 epochs / 4,650 steps, with the same 5,999-image Flickr8k training set.
Its final checkpoint obtained 1.30% image-to-text and 1.10% text-to-image
R@1 on the existing 1,000-image, first-caption validation gallery. This is
a real CUDA training run, not a new independent test or a CUDA-versus-CPU
numerical equivalence claim.

Before reading external scores, this final checkpoint and evaluation protocol
were fixed. The external data are the pinned Flickr30k mirror at revision
`765d117f3eec816f2bfdc2d73ebb50a6f77b86a4`, with image-level membership and
all five captions taken from `dataset_flickr30k.json` in the original Karpathy
split archive. The mirror's single split named `test` contains the entire
31,783-image collection; it is not itself the canonical test partition.

Of the 1,000 canonical test images, **239 were excluded** because their numeric
Flickr IDs occur in any previously used Flickr8k train, validation or test
split (all 8,000 IDs were checked). Decoded RGB pixel hashes were also checked
against all previous Flickr8k splits and for duplicates within the new gallery;
these checks removed no additional images. The resulting external gallery has
**761 images and 3,805 captions**. No new training, hyperparameter selection
or checkpoint selection used this gallery. Near-duplicate detection was not
exhaustive.

Strict checkpoint loading, finite state and exact reproduction of the original
validation retrieval metrics were checked before the external evaluation.
Inference used CPU with the same resize/normalization, tokenizer, explicit EOS
and 32-token limit. Directional prediction-to-embedding cosine scores and
image-identity positives were fixed in advance; each image has five positive
captions and each caption targets its source image.

| External transfer metric | Image-to-text | Text-to-image |
|---|---|---|
| R@1 | 1.05% (8/761) | 0.66% (25/3,805) |
| R@5 | 3.15% | 2.58% |
| R@10 | 4.99% | 4.60% |

Random R@1 is about **0.13%** in either direction. Retrieval remains weak.
This is a filtered external transfer test, not the unmodified standard
Flickr30k benchmark, and its gallery differs from the Flickr8k evaluations.
It covers one frozen seed without a fresh matched CLIP control and establishes
neither statistical superiority nor paper-scale reproduction. One exact
caption string is shared by two caption entries; primary positives remain
image-identity based.

For audit, the final checkpoint SHA-256 is
`8496f4b677492efa4de206e11130c5060690d6b5e2ae08b88fa790976a9c338d`;
the Karpathy split archive SHA-256 is
`4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817`.
Datasets, weights, machine inventories and local audit artifacts are excluded
from the repository.

## Sources

- [Official LeVLJEPA repository](https://github.com/MLO-lab/LeVLJEPA)
- [Reference forward at the inspected revision](https://github.com/MLO-lab/LeVLJEPA/blob/1fd6545c38407f0ab98af3e59070df483914e92b/forwards.py)
- [Reference architecture](https://github.com/MLO-lab/LeVLJEPA/blob/1fd6545c38407f0ab98af3e59070df483914e92b/main.py)
- [Reference SIGReg](https://github.com/MLO-lab/LeVLJEPA/blob/1fd6545c38407f0ab98af3e59070df483914e92b/utils/sigreg.py)
- [Pinned Flickr8k dataset card](https://huggingface.co/datasets/jxie/flickr8k/blob/56f58c967835f7c508d684f36bd7897cca9d7634/README.md)

- [Pinned Flickr30k mirror](https://huggingface.co/datasets/lmms-lab-encoder/flickr30k/tree/765d117f3eec816f2bfdc2d73ebb50a6f77b86a4)
- [Original Karpathy caption split archive](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)


## Mathematical and evaluation audit

The objective uses elementwise-mean MSE in each direction, averages the two
directions, and detaches only the opposite prediction targets. SIGReg is applied
to the unnormalized pre-projected marginals. Its two-sided trapezoidal weights
 approximate the Gaussian-windowed characteristic-function integral on `[0, 3]`.
An independent unit test expands this statistic as a sum of pairwise cosine
kernels in one dimension and checks its analytic derivative, without computing
empirical characteristic-function means. The quadrature truncation and the
local-batch DDP scale are reference conventions, not claims of exact integration
on the real line or invariance to changing the number of devices.

The pinned [reference overview](https://github.com/MLO-lab/LeVLJEPA/tree/1fd6545c38407f0ab98af3e59070df483914e92b)
identifies dense visual features and frozen VLM transfer as the main benefits;
it explicitly allows lower zero-shot retrieval performance than contrastive
training. A small Flickr retrieval experiment therefore checks a different and
much smaller setting, and does not establish or refute those transfer claims.
Increasing data alone does not guarantee a retrieval win. Frozen-feature or
dense-prediction comparisons need their own matched downstream protocol.


The reduction convention follows the pinned executable `forwards.py`, not a
literal unnormalized reading of [paper Eq. (5)](https://arxiv.org/html/2607.00784v1#S3.E5).
For equal embedding width `D`, the code's prediction term is the paper's written
sum-of-squared-distances term divided by `2*D`: MSE averages over features,
and the two directional losses are then averaged. This affects the relative
SIGReg weight, not merely the display scale of the full loss. The paper's
notation does not resolve this executable-code discrepancy; we keep the explicit
reference-code convention rather than silently changing the objective.
The v1 paper's Appendix A also lists learning rate `0.04`, per-modality weight
`0.01`, and predictor dropout `0.1`, whereas the pinned configuration lists
`0.0002`, `0.02`, and `0.0`. Neither that reference default configuration nor
our smaller Flickr recipe is claimed to reproduce the published training run.


## CC3M subset and frozen downstream follow-up

A larger, fixed-budget follow-up used the first 20 train shards and all 16
validation shards of `pixparse/cc3m-wds`, revision
`46f3d69f840e59d77d52e8decfe5baec97e94c7f`. After exact RGB deduplication,
there were 100,741 training images and 13,429 held-out images. The latter were
sorted by a fixed key hash and split into 6,714 validation and 6,715 locked-test
images before any model scores were observed. All downloaded images decoded,
and archive sizes and SHA256 matched the Hub metadata. Exact pixel checks
against the previously used Flickr8k splits and evaluated Flickr30k gallery
found no matches; this does not rule out resized or near duplicates.

Both objectives used width-192, four-layer ViT/GPT-2 encoders, 96-pixel images,
64-token text including EOS, batch size 128, and 20 epochs / 15,740 updates.
AdamW used learning rate 0.0002, weight decay 0.01, 5% linear warmup, cosine
decay to 0.00002, and gradient clipping at 1. LeVLJEPA used 1,024 SIGReg
projections and per-modality weights 0.02. The CLIP control reused the existing
`CLIPLoss` with fixed temperature 0.07; it is not a full tuned OpenCLIP recipe.
Shared encoder/pre-projection initialization hashes matched. Extra LeVLJEPA
predictors mean equal sample/update budgets do not imply equal compute.
Only seed 42 and each fixed final checkpoint were evaluated.

Locked-test retrieval over 6,715 image/caption pairs:

| Objective | I→T R@1 | T→I R@1 | I→T R@5 | T→I R@5 | I→T R@10 | T→I R@10 |
|---|---:|---:|---:|---:|---:|---:|
| LeVLJEPA | 0.3872% | 0.4021% | 1.6232% | 1.5339% | 2.7997% | 2.7699% |
| CLIP control | 0.3574% | 0.4914% | 1.6530% | 2.4572% | 3.0082% | 3.8719% |

Random R@1 is 0.01489%. The I→T R@1 advantage is only 26 correct queries
versus 24; the remaining listed metrics favor CLIP. This is not evidence of
stable or overall superiority. Gallery/data differences prevent direct
comparison to the earlier Flickr percentages.

### Frozen visual-feature evaluations

The same final vision encoders were frozen and compared with a seed-42 random
encoder. Only the downstream heads were optimized; encoder state hashes were
unchanged, and head initialization hashes matched across all three encoders.
These evaluations consume the raw vision CLS/patch features, not the
cross-modal predictors or marginal pre-projection heads.

- **STL-10:** all 5,000 labeled training images and 8,000 test images; no
  unlabeled images. A bias-free 192-to-10 linear head was trained for 50 epochs
  with AdamW, learning rate 0.01, weight decay 0.0001, batch size 256, and
  step-wise cosine decay to zero. CLS features were L2-normalized and then
  standardized using training-set statistics only. This is a supervised linear
  probe, not zero-shot classification or the official 10-fold SSL protocol.
- **ADE20K:** all 20,210 training images and 2,000 official validation images.
  A 192-to-150 linear 1×1 convolution was trained for 20 epochs with AdamW,
  learning rate 0.001, weight decay 0.0001, batch size 32, and step-wise cosine
  decay to zero. Inputs were resized to 96×96, producing a 12×12 patch grid;
  training masks used nearest-neighbor resizing. Evaluation bilinearly resized
  logits to each original mask size. Original label 0 (and void 255) was ignored;
  mIoU averages classes with nonzero union. Patch features were cached in fp16,
  while encoder inference and head training used fp32.

| Frozen encoder | STL-10 top-1 | ADE20K mIoU | ADE20K pixel accuracy |
|---|---:|---:|---:|
| LeVLJEPA | 54.6375% | 3.8257% | 45.7587% |
| CLIP control | 62.6500% | 4.1407% | 47.5885% |
| Random encoder | 32.0500% | 1.3311% | 32.3050% |

The random row is a trained head on frozen random features, not uniform random
guessing. Both pretrained encoders improve over this control, but LeVLJEPA is
below CLIP on both downstream metrics in this run. ADE20K absolute quality is
low, and the paper's dense-feature advantage was not reproduced under this
reduced protocol. One pretraining/head seed, small encoders and resolution,
fixed untuned probe recipes, and no exhaustive web-pretraining/downstream
near-duplicate audit limit the conclusions. No holdout-based hyperparameter
or checkpoint selection was performed.

Data sources: [STL-10](https://cs.stanford.edu/~acoates/stl10/) (original archive
MD5 `91f7769df0f17e558f3565bffb0c7dfb`, including torchvision binary-file checks)
and [ADE20K](https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/download_ADE20K.sh).
The follow-up used local-only tar/cached-feature adapters and native Module/
Manager training. Those experiment helpers, datasets, caches, weights, and
machine records are outside this PR; the public runnable example remains the
reduced Flickr8k example. Protocols, source hashes, epoch histories and final
metrics were retained separately.
