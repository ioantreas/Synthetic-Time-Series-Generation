# VIGIL: Variance-Informed Latent Guidance for Private Time-Series Generation

This repository contains the implementation of **VIGIL**, a framework for privacy-preserving conditional time-series generation using latent-space diffusion.

The main implementation is:

```text
diffusion/src/diffusion/sampler/sample_VIGIL.py
```

This file contains the sampling procedure described in the paper, including latent anchor optimisation, variance-informed confidence estimation, and latent-guided reverse diffusion.

## Repository Structure

```text
.
├── autoencoders/
│   ├── prepare_latent_transformer.py        # Prepare and normalise latent representations
│   └── transformer_autoencoder.py           # Transformer autoencoder
│
├── baselines/
│   ├── CSDI/                                # CSDI baseline
│   ├── Diffusion-TS/                        # Diffusion-TS baseline
│   └── SSSD/                                # SSSD-S4 baseline
│
├── diffusion/
│   ├── bin/
│   │   ├── train_tsdiff.py                  # Train TSDiff model
│   │   └── train_unconditional_backbone.py  # Train backbone of VIGIL (TSDiff-based)
│   ├── configs/                             # Data-space and latent-space configs
│   ├── results/                             # Training and inference outputs
│   └── src/
│       └── diffusion/
│           ├── arch/                       # Model architecture components
│           ├── evaluation/                 # Evaluation utilities
│           ├── model/                      # Diffusion model definitions
│           ├── sampler/
│           │   ├── observation_guidance.py
│           │   ├── sample_data_space.py
│           │   ├── sample_spinning_decoder.py
│           │   ├── sample_tsdiff.py
│           │   └── sample_VIGIL.py         # Main VIGIL implementation
│           ├── configs.py
│           ├── predictor.py
│           └── utils.py
│
├── scripts/
│   ├── air_quality/
│   ├── appliances/
│   ├── har/
│   ├── metro/
│   └── tep/                            # Dataset preprocessing scripts
│
└── README.md
```

## Running VIGIL

The general workflow is:

1. Preprocess each dataset using the corresponding scripts under `scripts/`.
2. Construct the train/test time-series sequences and remove time features where required.
3. Train the Transformer autoencoder with `autoencoders/transformer_autoencoder.py`.
4. Run `autoencoders/prepare_latent_transformer.py` to encode the data and compute the latent normalisation statistics.
5. Train the unconditional latent diffusion model using the appropriate latent config under `diffusion/configs/`.
6. Pass the trained diffusion model, autoencoder, prepared latent data, and test sequences to `sample_VIGIL.py`.

A typical VIGIL inference command is:

```bash
python sample_VIGIL.py \
  --version <MODEL_VERSION> \
  --latent_steps 16 \
  --orig_seq_len <SEQUENCE_LENGTH> \
  --num_samples 500 \
  --train_data <TEST_DATA> \
  --missing_ratio 0.5 \
  --out_dir <OUTPUT_DIR> \
  --scenario <SCENARIO> \
  --ae_root <AUTOENCODER_DIR> \
  --latents_root <LATENTS_DIR> \
  --base_scale 1 \
  --base_repeats 100 \
  --latent_keep_percent 0.2 \
  --num_imputation_samples 10 \
  --mask_type variance \
  --seed 42 \
  --inference_seed 1 \
  --device cuda
```

The evaluated scenarios are:

```text
single_block
blackout
forecast
random
```

The paper reports results over inference seeds `1` to `5`.

## Baselines

Baseline implementations are provided under `baselines/` for **CSDI**, **Diffusion-TS**, and **SSSD-S4**.

Additional inference-time comparisons are implemented in:

```text
sample_tsdiff.py
sample_data_space.py
sample_spinning_decoder.py
```

These use the same evaluation data and missingness masks as VIGIL.