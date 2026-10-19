"""Configuration for the five published HiLS/TTR sparse attention variants."""

import math

from models.FlashHiLS.configuration_hils import HiLSConfig


class HyDenConfig(HiLSConfig):
    model_type = "qwen_hyden"

    def __init__(
        self,
        compressor_type: str = "ttr",
        compressor_x_apply_rope: bool = True,
        compressor_ttr_value_init_std: float | None = None,
        compressor_ttr_ridge_learnable_floor: float | None = None,
        compressor_ttr_ridge_learnable_init: float | None = None,
        **kwargs,
    ):
        if compressor_type not in {"hils", "ttr", "ttr_conv"}:
            raise ValueError("compressor_type must be hils, ttr or ttr_conv")
        if not isinstance(compressor_x_apply_rope, bool):
            raise ValueError("compressor_x_apply_rope must be a bool")
        unknown = sorted(key for key in kwargs if key.startswith("compressor_"))
        if unknown:
            raise ValueError(f"Unsupported compressor settings: {', '.join(unknown)}")
        if kwargs.get("insert_landmarks", False) or kwargs.get("adjust_lmk_pos", False):
            raise ValueError("Published variants do not insert or reposition landmarks")
        kwargs.pop("insert_landmarks", None)
        kwargs.pop("adjust_lmk_pos", None)
        self.compressor_type = compressor_type
        self.compressor_x_apply_rope = compressor_x_apply_rope
        ttr_values = (
            compressor_ttr_value_init_std,
            compressor_ttr_ridge_learnable_floor,
            compressor_ttr_ridge_learnable_init,
        )
        if compressor_type == "hils":
            if any(value is not None for value in ttr_values):
                raise ValueError("HiLS does not use TTR initialization or ridge settings")
            if kwargs.get("chunk_size", 64) < 2 or not kwargs.get("enable_lmk_q_proj", False):
                raise ValueError("HiLS requires chunk_size >= 2 and enable_lmk_q_proj=true")
        else:
            std, floor, init = (
                default if value is None else value
                for value, default in zip(ttr_values, (0.02, 0.01, 0.1))
            )
            if not all(math.isfinite(value) for value in (std, floor, init)):
                raise ValueError("TTR initialization settings must be finite")
            if std < 0 or floor <= 0 or init <= floor:
                raise ValueError("TTR requires std >= 0 and ridge init > floor > 0")
            self.compressor_ttr_value_init_std = std
            self.compressor_ttr_ridge_learnable_floor = floor
            self.compressor_ttr_ridge_learnable_init = init
        super().__init__(**kwargs)


__all__ = ["HyDenConfig"]
