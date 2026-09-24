# atml-pa1: Learning beyond IID: inductive biases, domain adaptation, domain generalization and open-set recognition

Programming Assignment 1

This repository contains the code, split indices, machine-readable results and figures behind the report. Each task is a
self-contained Colab notebook; every number in the report traces to a file under `task*/results/`.

## Repository layout

```
task1/   Inductive biases and feature representations (STL-10; ResNet-50, ViT-B/16, CLIP ViT-B/32)
task2/   Unsupervised domain adaptation on PACS (Source-only, DAN, DANN, CDAN)
task3/   Domain generalization on PACS (ERM, DAN-DG, SAM)
task4/   Open-set recognition on CIFAR-10 with CIFAR-100 unknowns (Vanilla, GCSC, PROSER)
shared/  splits/pacs_sketch_seed6304.json — the PACS source train/val split shared by Tasks 2 and 3
```

Inside each task folder:

```
taskN/taskN.ipynb        the complete pipeline for that task (data, training, evaluation, figures)
taskN/results/           tables (.csv and .tex), logs, JSON result files, figures/ (.pdf and .png)
taskN/results/run_info.json   the exact configuration and library versions of the run
```

Datasets, cached tensors and checkpoints are not committed. They are downloaded or built by the notebooks.

## Environment

All experiments were run on Google Colab (Tesla T4, 16 GB) with the library versions pinned in `requirements.txt`.
The notebooks install the two packages that Colab does not ship with (`open_clip_torch` for Task 1, `datasets` for
Tasks 2 and 3).

## How to reproduce

The notebooks expect Google Drive mounted at `/content/drive` and use two folders:

- `MyDrive/atml-pa1/` — a clone of this repository (the notebooks write results here);
- `MyDrive/pa1-data/` — datasets, cached tensors and checkpoints (large, never committed).

For each task: open the notebook in Colab, select a T4 GPU runtime, and run all cells. All random seeds are fixed to
6304. Finished training runs are cached on Drive.

**Task 1** (`task1/task1.ipynb`). 

**Task 2** (`task2/task2.ipynb`). Downloads PACS from the Hugging Face hub (`flwrlabs/pacs`), builds a
256×256 tensor of all images, trains Source-only, DAN, DANN, CDAN and the two extra DAN runs of the controlled study,
then evaluates. Target labels are read only in the final evaluation section, after
`results/locked_before_target_eval.json` records the fixed checkpoints.

**Task 3** (`task3/task3.ipynb`). Requires the Task 2 outputs on Drive (the PACS tensor, the Source-only
checkpoint, the split file). Sketch images are dropped from memory at load time and read again only in the final
section, after `results/locked_before_sketch_eval.json` is written; the source-side diagnostics (source-domain
separability, sharpness proxy) run before that point.

**Task 4** (`task4/task4.ipynb`). Downloads CIFAR-10 and CIFAR-100 with torchvision, trains Vanilla and
GCSC from scratch and PROSER from the Vanilla checkpoint, extracts features and logits for all splits, then evaluates
the four post-hoc scores and the trained models. CIFAR-100 images are scored only after
`results/locked_before_unknown_eval.json` records the checkpoints and the threshold rule.

Task order: Task 2 must run before Task 3. Tasks 1 and 4 are independent.

## Where each reported number comes from

| Task | Main tables | Other result files |
|---|---|---|
| 1 | `table_step1_clean_baseline`, `table_compact_clean_color_patch`, `table_step3_shape_bias`, `table_step4_translation`, `table_step6_cosine_stability` | `splits.json` (subset ids), `predictions.json` (every prediction), `head_training.json`, `patch_permutations.json`, `projection_settings.json`, `cue_conflict/` |
| 2 | `table_main_comparison`, `table_per_class_target`, `table_alignment_strength_study` | `logs/*.json` (per-epoch losses and validation), `final_evaluation.json` (confusions, per-class accuracy, separability), `table_dominant_confusions.csv` |
| 3 | `table_main_comparison`, `table_per_class_sketch`, `table_lambda_study`, `table_dan_vs_dan_dg` | `logs/*.json`, `source_diagnostics.json` (separability, sharpness, the fixed sharpness batch), `final_evaluation.json` |
| 4 | `table_vanilla_scores`, `table_model_comparison`, `table_unknown_classes` | `osr_metrics.json` (all AUROCs, thresholds, rejection rates), `accepted_unknown_examples.json`, `table_score_rank_correlation.csv`, `table_score_disagreement.csv`, `logs/*.json` |

Tables exist as `.csv` (machine-readable) and `.tex` (as pasted into the report). Figures are in
`taskN/results/figures/` as `.pdf` (used in the report) and `.png`.

## Implementation notes and recorded choices

Settings that the assignment leaves open, or that were needed to make a specified method train, are listed here and
stated in the report.

- **Task 1.** Linear heads use batch size 64 (not specified). The extra colour intervention is a 180° hue rotation in
  YIQ space, which preserves luminance. Cue conflicts use AdaIN at style strength 1.0 on five class pairs in both
  directions. Visualisation is t-SNE (perplexity 30, cosine metric, seed 6304), one fit per backbone.
  OpenCLIP is loaded with `force_quick_gelu=True`, which is the activation the `openai` weights were trained with.
- **Tasks 2 and 3.** One "source epoch" is one pass over the largest source training split (235 updates of 8 images
  per source domain). BatchNorm running statistics are frozen at their ImageNet values for every method; BatchNorm
  scale and shift are trained. The domain discriminator in DANN and CDAN uses a learning rate of 1e-3 (10× the
  network's), following the CDAN reference implementation: with an equal learning rate the adversarial game diverged in
  the first epoch (domain loss above 2000; the run is documented in the report). The controlled studies vary
  λ_MMD (Task 2) and λ_DG (Task 3) over {0.1, 1, 10}.
- **Task 4.** Mixed precision (`torch.autocast`) is used for training; it does not change the recipe. PROSER's
  augmented output is the ten known logits plus the maximum over the five dummy classifiers, as in the paper; its
  detection score follows the reference implementation (softmax over the eleven logits, dummy probability minus the
  largest known probability). Manifold mixup pairs are drawn only between examples of different classes.

## Attribution of external code and data

- **AdaIN** (Task 1): network definition, the `adain` function and the pretrained `vgg_normalised.pth` /
  `decoder.pth` weights are from Naoto Inoue's PyTorch implementation, <https://github.com/naoto0804/pytorch-AdaIN>
  (MIT licence), of Huang & Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*,
  ICCV 2017. Only the inference part is reproduced, in the notebook.
- **PROSER** (Task 4): the losses follow Zhou, Ye & Zhan, *Learning Placeholders for Open-Set Recognition*, CVPR 2021,
  and the placeholder detection score follows the authors' code, <https://github.com/zhoudw-zdw/CVPR21-Proser>.
- **Pretrained models:** torchvision (`ResNet50_Weights.IMAGENET1K_V2`, `ViT_B_16_Weights.IMAGENET1K_V1`,
  `ResNet18_Weights.IMAGENET1K_V1`) and OpenCLIP (`ViT-B-32`, `pretrained="openai"`).
- **Datasets:** STL-10 (Coates, Ng & Lee, 2011) and CIFAR-10/100 (Krizhevsky, 2009) via torchvision; PACS
  (Li et al., 2017) via the Hugging Face dataset `flwrlabs/pacs`.
- Gradient reversal, MMD, SAM and the four novelty scores are implemented from the respective papers.
