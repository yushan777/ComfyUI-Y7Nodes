"""
Y7Nodes Krea 2 Raw Sampling Shift

Patches a model's sampling shift for Krea 2 Raw. ComfyUI loads every Krea 2 model with a fixed
shift of 1.15, which suits Turbo but not Raw, so raw_dynamic applies Krea 2's resolution-dependent
Raw schedule (0.5 at 256x256 up to 1.15 at 1280x1280). Turbo needs no patch.

Cloned from krea2-twostage-sampler by Auryg (MIT licence), ported to the V3 schema, minus its
turbo_fixed mode (identical to ComfyUI's default for Krea 2).
"""

import math

import comfy.model_sampling
import nodes
from comfy_api.latest import io

from ..utils.logger import logger


MODES = ["raw_dynamic", "manual"]
DEFAULT_MANUAL_SHIFT = 1.15  # ComfyUI's own Krea 2 shift


def _krea2_raw_shift(width, height):
    """Linear in token count: 0.5 at 256 tokens (256x256) to 1.15 at 6400 tokens (1280x1280)."""
    min_tokens = 256
    max_tokens = 6400
    min_shift = 0.5
    max_shift = 1.15
    sampling_width = math.ceil(width / 16) * 16
    sampling_height = math.ceil(height / 16) * 16
    image_tokens = sampling_width * sampling_height / (8 * 8 * 2 * 2)
    slope = (max_shift - min_shift) / (max_tokens - min_tokens)
    return slope * image_tokens + (min_shift - slope * min_tokens)


class ModelSamplingKrea2(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
    pass


class Y7Nodes_Krea2_RawSamplingShift(io.ComfyNode):
    """Sets the Krea 2 Raw sampling shift: resolution-dependent or manual."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_Krea2_RawSamplingShift",
            display_name="Y7 Krea 2 Raw Sampling Shift",
            category="Y7Nodes/Sampling",
            description="Sets the Krea 2 Raw sampling shift: resolution-dependent or manual.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "sampling_mode", options=MODES, default="raw_dynamic",
                    tooltip="raw_dynamic uses Krea 2's official resolution-dependent Raw shift. "
                            "manual uses manual_shift.",
                ),
                io.Int.Input(
                    "width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=8,
                    tooltip="Sampling width; used only by raw_dynamic.",
                ),
                io.Int.Input(
                    "height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=8,
                    tooltip="Sampling height; used only by raw_dynamic.",
                ),
                io.Float.Input(
                    "manual_shift", default=DEFAULT_MANUAL_SHIFT, min=0.0, max=100.0, step=0.01, round=False, advanced=True,
                    tooltip="Constant shift used only by manual mode.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, sampling_mode, width, height, manual_shift=DEFAULT_MANUAL_SHIFT) -> io.NodeOutput:
        if sampling_mode == "raw_dynamic":
            shift = _krea2_raw_shift(width, height)
        elif sampling_mode == "manual":
            shift = manual_shift
        else:
            raise ValueError(f"Unknown Krea 2 sampling mode: {sampling_mode}")

        model_sampling = ModelSamplingKrea2(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        patched_model = model.clone()
        patched_model.add_object_patch("model_sampling", model_sampling)

        logger.info(f"[Y7 Krea 2 Raw Sampling Shift] mode={sampling_mode}, shift={shift:.8f}, width={width}, height={height}")
        return io.NodeOutput(patched_model)
