# -*- coding: utf-8 -*-
#
# @File:   configuration_dynamicvla.py
# @Author: Haozhe Xie
# @Date:   2025-08-21 15:22:31
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2026-02-26 10:04:17
# @Email:  root@haozhexie.com

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("dynamicvla")
@dataclass
class DynamicVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32
    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    # Add empty images. Used by dynamicvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0
    # Converts the joint and gripper values from the standard Aloha space to the space
    # used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False
    # Converts joint dimensions to deltas with respect to the current state before
    # passing to the model. Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False
    # Use delta action prediction (relative to the current robot state)
    use_delta_action: bool = True
    # Streaming inference
    enable_streaming: bool = False
    # Multi-timestep fusion ("conv" or "attn")
    temporal_fusion: str = "conv"
    # Tokenizer
    tokenizer_max_length: int = 48
    # Decoding
    num_steps: int = 10
    # Attention utils
    use_cache: bool = True
    # Finetuning settings
    freeze_vision_model: bool = True
    freeze_connector: bool = True
    freeze_text_model: bool = True
    train_state_proj: bool = True
    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6
    # Select the VLM backbone.
    attention_mode: str = "cross_attn"
    prefix_length: int = -1
    pad_language_to: str = "longest"  # "max_length"
    # Less or equal to 0 is the default where the action expert has the same number of
    # layers of VLM. Otherwise the expert have less layers.
    num_expert_layers: int = -1
    num_expert_skip_layers: int = 0
    # VLM settings
    vlm_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    num_vlm_layers: int = 16
    # SmolVLM Settings
    smolvlm_patch_size: int = 16
    smolvlm_attention_heads: int = 12
    smolvlm_hidden_size: int = 768
    smolvlm_intermediate_size: int = 3072
    # FastVLM Settings
    fastvlm_inference_mode: bool = True
    # Interleave SA layers each self_attn_every_n_layers
    self_attn_every_n_layers: int = 2
    # The action expert hidden size (wrt to the VLM)
    expert_width_multiplier: float = 0.75
    # sensitivity range for the timestep used in sine-cosine positional encoding
    min_period: float = 4e-3
    max_period: float = 4.0
    # Residual proactive flow settings. When enabled, the action expert predicts
    # the residual flow after subtracting analytical object-motion drift.
    enable_residual_flow: bool = False
    # DOM+Velocity: current robot-base XYZ velocity occupies unused state slots
    # 6:9. The existing state projection learns its contribution; no drift.
    condition_on_object_velocity: bool = False
    velocity_input_mean: list[float] = field(default_factory=lambda: [0., 0., 0.])
    velocity_input_std: list[float] = field(default_factory=lambda: [1., 1., 1.])
    # field: legacy Case 1; coordinate: train/sample in A - centered drift.
    residual_flow_formulation: str = "field"
    residual_lambda_type: str = "constant"
    residual_drift_scale: float = 1.0
    residual_drift_mode: str = "trajectory"
    residual_action_dt: float = 1.0 / 25.0
    residual_axis_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    residual_position_key: str = "dynamic_object.position_xyz"
    residual_velocity_key: str = "dynamic_object.velocity_xyz"
    residual_covariance_key: str = "dynamic_object.covariance_xyz"
    residual_lookahead_position_key: str = "dynamic_object.lookahead_position_xyz"
    residual_lookahead_horizon_key: str = "dynamic_object.lookahead_horizon_s"
    residual_lookahead_horizon_s: float = 1.0
    residual_normalize_drift: bool = True
    residual_center_drift: bool = True
    residual_drift_mean: list | None = None
    residual_estimate_drift_mean: bool = True
    residual_drift_mean_max_batches: int = 0
    residual_drift_mean_cache: bool = True
    residual_drift_mean_cache_dir: str | None = None
    use_checkpoint_normalization: bool = False

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                "The chunk size is the upper bound for the number of action steps per"
                f" model invocation. Got {self.n_action_steps} for `n_action_steps` and"
                f" {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by dynamicvla for aloha real"
                " models. It is not ported yet in LeRobot."
            )
        self.validate_residual_formulation()
        if self.residual_lambda_type not in {"constant", "linear", "cosine"}:
            raise ValueError(
                "residual_lambda_type must be one of {'constant', 'linear', 'cosine'}, "
                f"got {self.residual_lambda_type!r}."
            )
        if self.residual_drift_mode not in {"trajectory", "velocity", "lookahead"}:
            raise ValueError(
                "residual_drift_mode must be one of {'trajectory', 'velocity', 'lookahead'}, "
                f"got {self.residual_drift_mode!r}."
            )

    def validate_residual_formulation(self) -> None:
        if self.residual_flow_formulation not in {"field", "coordinate"}:
            raise ValueError("residual_flow_formulation must be field or coordinate")
        if self.enable_residual_flow and self.residual_flow_formulation == "coordinate" and self.residual_lambda_type != "constant":
            raise ValueError("Residual-coordinate flow currently requires constant lambda=1")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
