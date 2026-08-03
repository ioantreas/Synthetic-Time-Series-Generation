# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from uncond_ts_diff.utils import linear_beta_schedule

custom_backbone = {
    "input_dim": 1,
    "hidden_dim": 256,
    "output_dim": 1,
    "step_emb": 128,
    "num_residual_blocks": 8,
    "dropout": 0.1,
    "residual_block": "s4",
}

custom_big_backbone = {
    "input_dim": 1,
    "hidden_dim": 256,
    "output_dim": 1,
    "step_emb": 128,
    "num_residual_blocks": 12,
    "dropout": 0.1,
    "residual_block": "s4",
}

residual_block_s4_backbone = {
    "input_dim": 1,
    "hidden_dim": 128,
    "output_dim": 1,
    "step_emb": 128,
    "num_residual_blocks": 6,
    "residual_block": "s4",
}

residual_block_s4_backbone_smallv2 = {
    "input_dim": 1,
    "hidden_dim": 512,
    "output_dim": 1,
    "step_emb": 128,
    "num_residual_blocks": 3,
    "residual_block": "s4",
}

residual_block_s4_backbone_small = {
    "input_dim": 1,
    "hidden_dim": 64,
    "output_dim": 1,
    "step_emb": 128,
    "num_residual_blocks": 3,
    "residual_block": "s4",
}

residual_block_s4_backbone_small_dropout01 = {
    "input_dim": 1,
    "hidden_dim": 64,
    "output_dim": 1,
    "step_emb": 128,
    "num_residual_blocks": 3,
    "dropout": 0.1,
    "residual_block": "s4",
}

residual_block_s4_backbone_small_dropout02 = {
    "input_dim": 1,
    "hidden_dim": 64,
    "output_dim": 1,
    "step_emb": 128,
    "num_residual_blocks": 3,
    "dropout": 0.2,
    "residual_block": "s4",
}

residual_block_s4_backbone_small_dropout03 = {
    "input_dim": 1,
    "hidden_dim": 64,
    "output_dim": 1,
    "step_emb": 128,
    "num_residual_blocks": 3,
    "dropout": 0.3,
    "residual_block": "s4",
}

residual_block_s4_backbone_large = {
    "input_dim": 1,
    "hidden_dim": 128,
    "output_dim": 1,
    "step_emb": 128,
    "num_residual_blocks": 18,
    "residual_block": "s4",
}

transformer_latent_backbone = {
    "input_dim": 1,
    "hidden_dim": 768,
    "output_dim": 1,
    "step_emb": 256,
    "num_residual_blocks": 16,
    "dropout": 0.03,
    "residual_block": "s4",
}

transformer_latent_config = {
    "backbone_parameters": transformer_latent_backbone,
    "timesteps": 200,
    "diffusion_scheduler": linear_beta_schedule,
}

custom_big_config = {
    "backbone_parameters": custom_big_backbone,
    "timesteps": 300,
    "diffusion_scheduler": linear_beta_schedule,
}

custom_config = {
    "backbone_parameters": custom_backbone,
    "timesteps": 200,
    "diffusion_scheduler": linear_beta_schedule,
}

diffusion_config = {
    "backbone_parameters": residual_block_s4_backbone,
    "timesteps": 100,
    "diffusion_scheduler": linear_beta_schedule,
}
diffusion_small_config = {
    "backbone_parameters": residual_block_s4_backbone_small,
    "timesteps": 100,
    "diffusion_scheduler": linear_beta_schedule,
}

diffusion_small_configv2 = {
    "backbone_parameters": residual_block_s4_backbone_smallv2,
    "timesteps": 100,
    "diffusion_scheduler": linear_beta_schedule,
}


diffusion_small_config_dropout = {
    "backbone_parameters": residual_block_s4_backbone_small_dropout01,
    "timesteps": 100,
    "diffusion_scheduler": linear_beta_schedule,
}

diffusion_small_config_dropout02 = {
    "backbone_parameters": residual_block_s4_backbone_small_dropout02,
    "timesteps": 100,
    "diffusion_scheduler": linear_beta_schedule,
}

diffusion_small_config_dropout03 = {
    "backbone_parameters": residual_block_s4_backbone_small_dropout03,
    "timesteps": 100,
    "diffusion_scheduler": linear_beta_schedule,
}

diffusion_large_config = {
    "backbone_parameters": residual_block_s4_backbone_large,
    "timesteps": 100,
    "diffusion_scheduler": linear_beta_schedule,
}
