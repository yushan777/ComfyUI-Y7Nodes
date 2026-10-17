"""
Y7Nodes Dual Resolution Selector

One aspect ratio, two megapixel targets: gives a base resolution for a first sampling stage and a
final resolution for the stage that finishes at a higher size (see the Krea 2 Two/Three-Stage Samplers).

Cloned from krea2-twostage-sampler by Auryg (MIT licence), minus its seed and random aspect modes.
"""

import math

from comfy_api.latest import io


ASPECT_RATIOS = {
    "1:1": (1.0, 1.0),
    "4:5": (4.0, 5.0),
    "5:4": (5.0, 4.0),
    "2:3": (2.0, 3.0),
    "3:2": (3.0, 2.0),
    "3:4": (3.0, 4.0),
    "4:3": (4.0, 3.0),
    "9:16": (9.0, 16.0),
    "16:9": (16.0, 9.0),
    "2.35:1": (2.35, 1.0),
    "21:9": (21.0, 9.0),
}


def _round_to_multiple(value, multiple):
    if multiple <= 1:
        return int(round(value))
    return int(round(value / multiple) * multiple)


def _dimensions_for(aspect_ratio, megapixels, multiple):
    ratio_w, ratio_h = ASPECT_RATIOS[aspect_ratio]
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (ratio_w * ratio_h))
    width = max(multiple, _round_to_multiple(ratio_w * scale, multiple))
    height = max(multiple, _round_to_multiple(ratio_h * scale, multiple))
    return width, height


class Y7Nodes_DualResolutionSelector(io.ComfyNode):
    """Base and final width/height for one aspect ratio at two megapixel targets."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_DualResolutionSelector",
            display_name="Y7 Dual Resolution Selector",
            category="Y7Nodes/Utils",
            description="Base and final width/height for one aspect ratio at two megapixel targets.",
            inputs=[
                io.Combo.Input("aspect_ratio", options=list(ASPECT_RATIOS), default="1:1"),
                io.Float.Input("base_megapixels", default=1.0, min=0.1, max=16.0, step=0.1),
                io.Float.Input("final_megapixels", default=2.0, min=0.1, max=16.0, step=0.1),
                io.Int.Input("multiple", default=16, min=8, max=128, step=8),
            ],
            outputs=[
                io.Int.Output(display_name="base_width"),
                io.Int.Output(display_name="base_height"),
                io.Int.Output(display_name="final_width"),
                io.Int.Output(display_name="final_height"),
            ],
        )

    @classmethod
    def execute(cls, aspect_ratio, base_megapixels, final_megapixels, multiple) -> io.NodeOutput:
        base_width, base_height = _dimensions_for(aspect_ratio, base_megapixels, multiple)
        final_width, final_height = _dimensions_for(aspect_ratio, final_megapixels, multiple)
        return io.NodeOutput(base_width, base_height, final_width, final_height)
